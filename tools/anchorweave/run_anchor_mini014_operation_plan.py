#!/usr/bin/env python3
"""Run ANCHOR-MINI-014 operation-plan parser bisect."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any


CONTRACT_FILE = Path("docs/research/ANCHOR_MINI_014_OPERATION_PLAN_CONTRACT.md")

MODELS = ["CHAR_CNN", "CHAR_GRU"]
ARMS = [
    "ANSWER_ONLY_DIRECT",
    "GLOBAL_PLAN_FIRST",
    "BLOCK_ONLY_PLAN_FIRST",
    "QUERY_FULL_TEXT_PLAN_FIRST",
    "SHUFFLED_QUERY_PLAN_FIRST",
    "SHORTCUT_TEACHER",
    "ORACLE_PARSED_PLAN_VISIBLE",
]

LABELS = list("ABCDEFGH")
MAX_CANDIDATES = 8
MAX_STEPS = 3
VALUE_MIN = 0
VALUE_MAX = 40
VALUE_CLASSES = VALUE_MAX + 1
OPERATIONS = ["ADD", "SUB", "MUL"]
OP_TO_ID = {name: index for index, name in enumerate(OPERATIONS)}
MAX_OPERAND = 9

PAD = 0
UNK = 1
ASCII_ALPHABET = "\n\t" + "".join(chr(code) for code in range(32, 127))
CHAR_TO_ID = {char: index + 2 for index, char in enumerate(ASCII_ALPHABET)}
VOCAB_SIZE = len(CHAR_TO_ID) + 2

JOB_COMPLETE = "ANCHOR_MINI_014_JOB_COMPLETE"
JOB_INVALID = "ANCHOR_MINI_014_JOB_INVALID_STRESS"
STATUS_QUERY = "ANCHOR_MINI_014_QUERY_FULL_STRONG_POSITIVE"
STATUS_BLOCK = "ANCHOR_MINI_014_BLOCK_ONLY_POSITIVE"
STATUS_DEPTH = "ANCHOR_MINI_014_DEPTH_LIMIT_FOUND"
STATUS_SCALE = "ANCHOR_MINI_014_CANDIDATE_SCALE_LIMIT_FOUND"
STATUS_NEGATIVE = "ANCHOR_MINI_014_NEGATIVE"
STATUS_INVALID = "ANCHOR_MINI_014_INVALID_STRESS"
STATUS_BLOCKED = "ANCHOR_MINI_014_RESOURCE_BLOCKED"
STATUS_PARTIAL = "ANCHOR_MINI_014_PARTIAL_BUDGET"


@dataclass(frozen=True)
class Operation:
    op_id: int
    arg: int


OP_OPTIONS = tuple(
    [Operation(OP_TO_ID["ADD"], arg) for arg in range(1, MAX_OPERAND + 1)]
    + [Operation(OP_TO_ID["SUB"], arg) for arg in range(1, MAX_OPERAND + 1)]
    + [Operation(OP_TO_ID["MUL"], arg) for arg in (2, 3)]
)


@dataclass(frozen=True)
class Candidate:
    ops: list[Operation]
    intermediates: list[int]
    final_value: int
    surface_bucket: int


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str
    start_value: int
    goal_value: int
    steps: int
    candidate_count: int
    candidates: list[Candidate]
    answer_label: int
    surface_shortcut_label: int
    surface_shortcut_is_gold: bool
    task_text: str
    header_text: str
    block_texts: list[str]
    query_texts: list[str]
    policy_bits: list[int]
    shuffled_policy_bits: list[int]
    shuffled_answer_label: int
    shortcut_policy_bits: list[int]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_seed_spec(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("--seeds must not be empty")
    return sorted(dict.fromkeys(seeds))


def parse_csv(raw: str, cast: Any = str) -> list[Any]:
    return [cast(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-014 operation-plan parser bisect.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default="2026-2035")
    parser.add_argument("--models", default="CHAR_CNN")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--candidate-counts", default="4,8")
    parser.add_argument("--steps", default="1,2,3")
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--eval-examples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-chars", type=int, default=1024)
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument("--train-surface-gold-prob", type=float, default=0.90)
    parser.add_argument("--eval-surface-wrong-prob", type=float, default=0.90)
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_completed(metrics_path: Path) -> set[str]:
    completed: set[str] = set()
    if not metrics_path.exists():
        return completed
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(json.loads(line)["job_id"])
    return completed


def clamp_value(value: int) -> int:
    return max(VALUE_MIN, min(VALUE_MAX, value))


def apply_op(value: int, op: Operation) -> int:
    if op.op_id == OP_TO_ID["ADD"]:
        return value + op.arg
    if op.op_id == OP_TO_ID["SUB"]:
        return value - op.arg
    if op.op_id == OP_TO_ID["MUL"]:
        return value * op.arg
    raise ValueError(f"unknown op id: {op.op_id}")


def run_ops(start: int, ops: list[Operation]) -> tuple[list[int], int] | None:
    value = start
    intermediates: list[int] = []
    for op in ops:
        value = apply_op(value, op)
        if value < VALUE_MIN or value > VALUE_MAX:
            return None
        intermediates.append(value)
    return intermediates, value


def random_op(rng: random.Random) -> Operation:
    op_name = rng.choice(OPERATIONS)
    if op_name == "MUL":
        arg = rng.choice([2, 3])
    else:
        arg = rng.randint(1, MAX_OPERAND)
    return Operation(OP_TO_ID[op_name], arg)


def random_ops(rng: random.Random, steps: int) -> list[Operation]:
    return [random_op(rng) for _ in range(steps)]


@lru_cache(maxsize=None)
def valid_sequences(start: int, steps: int) -> tuple[tuple[tuple[Operation, ...], tuple[int, ...], int], ...]:
    sequences: list[tuple[tuple[Operation, ...], tuple[int, ...], int]] = []
    for ops_tuple in product(OP_OPTIONS, repeat=steps):
        result = run_ops(start, list(ops_tuple))
        if result is None:
            continue
        intermediates, final_value = result
        sequences.append((ops_tuple, tuple(intermediates), final_value))
    if not sequences:
        raise RuntimeError("no valid operation sequences")
    return tuple(sequences)


def make_valid_ops(rng: random.Random, start: int, steps: int) -> tuple[list[Operation], list[int], int]:
    ops_tuple, intermediates, final_value = rng.choice(valid_sequences(start, steps))
    return list(ops_tuple), list(intermediates), final_value


def make_distractor_ops(
    rng: random.Random,
    *,
    start: int,
    steps: int,
    goal: int,
    prefer_near_miss: bool,
) -> tuple[list[Operation], list[int], int]:
    sequences = [item for item in valid_sequences(start, steps) if item[2] != goal]
    if prefer_near_miss:
        near = [item for item in sequences if abs(item[2] - goal) <= 2]
        if near:
            ops_tuple, intermediates, final_value = rng.choice(near)
            return list(ops_tuple), list(intermediates), final_value
    if sequences:
        ops_tuple, intermediates, final_value = rng.choice(sequences)
        return list(ops_tuple), list(intermediates), final_value
    raise RuntimeError("failed to generate distractor operation sequence")


def choose_surface_shortcut(
    *,
    split: str,
    answer_slot: int,
    candidate_count: int,
    rng: random.Random,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
) -> int:
    wrong_slots = [slot for slot in range(candidate_count) if slot != answer_slot]
    if split == "train":
        if rng.random() < train_surface_gold_prob:
            return answer_slot
        return rng.choice(wrong_slots)
    if split == "eval":
        if rng.random() < eval_surface_wrong_prob:
            return rng.choice(wrong_slots)
        return answer_slot
    raise RuntimeError(f"unknown split: {split}")


def make_surface_buckets(shortcut_slot: int, candidate_count: int, rng: random.Random) -> list[int]:
    buckets: list[int] = []
    for slot in range(candidate_count):
        if slot == shortcut_slot:
            buckets.append(9)
        else:
            buckets.append(rng.randint(0, 2))
    return buckets


def render_op(op: Operation, step_index: int) -> str:
    return f"  OP{step_index + 1}: {OPERATIONS[op.op_id]} {op.arg}"


def render_header(start: int, goal: int, steps: int) -> str:
    return f"START: {start}\nGOAL: {goal}\nSTEPS: {steps}"


def render_candidate_block(label: str, candidate: Candidate, steps: int) -> str:
    lines = [f"{label}:"]
    for index in range(steps):
        lines.append(render_op(candidate.ops[index], index))
    lines.append(f"  SURFACE: {candidate.surface_bucket}")
    return "\n".join(lines)


def serialize_example(start: int, goal: int, steps: int, candidates: list[Candidate]) -> tuple[str, str, list[str], list[str]]:
    header = render_header(start, goal, steps)
    blocks = [render_candidate_block(LABELS[index], candidate, steps) for index, candidate in enumerate(candidates)]
    task = header + "\n\n" + "\n\n".join(blocks)
    block_texts = [header + "\n\n" + block for block in blocks]
    query_texts = [f"QUERY: {LABELS[index]}\n" + task for index in range(len(candidates))]
    return task, header, block_texts, query_texts


def make_examples(
    *,
    split: str,
    count: int,
    seed: int,
    steps: int,
    candidate_count: int,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
) -> list[Example]:
    rng = random.Random(seed + steps * 1009 + candidate_count * 7919 + (0 if split == "train" else 1000003))
    rows: list[Example] = []
    for index in range(count):
        answer_slot = index % candidate_count
        start = rng.randint(0, 12)
        gold_ops, gold_intermediates, goal = make_valid_ops(rng, start, steps)
        final_values = {goal}
        candidates: list[Candidate] = []
        for slot in range(candidate_count):
            if slot == answer_slot:
                candidates.append(Candidate(gold_ops, gold_intermediates, goal, 0))
                continue
            prefer_near = slot == ((answer_slot + 1) % candidate_count)
            for _ in range(100):
                ops, intermediates, final_value = make_distractor_ops(
                    rng, start=start, steps=steps, goal=goal, prefer_near_miss=prefer_near
                )
                if final_value not in final_values:
                    final_values.add(final_value)
                    candidates.append(Candidate(ops, intermediates, final_value, 0))
                    break
            else:
                ops, intermediates, final_value = make_distractor_ops(
                    rng, start=start, steps=steps, goal=goal, prefer_near_miss=False
                )
                candidates.append(Candidate(ops, intermediates, final_value, 0))

        shortcut = choose_surface_shortcut(
            split=split,
            answer_slot=answer_slot,
            candidate_count=candidate_count,
            rng=rng,
            train_surface_gold_prob=train_surface_gold_prob,
            eval_surface_wrong_prob=eval_surface_wrong_prob,
        )
        buckets = make_surface_buckets(shortcut, candidate_count, rng)
        candidates = [
            Candidate(candidate.ops, candidate.intermediates, candidate.final_value, buckets[slot])
            for slot, candidate in enumerate(candidates)
        ]
        task_text, header_text, block_texts, query_texts = serialize_example(start, goal, steps, candidates)
        policy_bits = [int(slot == answer_slot) for slot in range(candidate_count)]
        shuffled_answer = (answer_slot + 1) % candidate_count
        shuffled_policy = [int(slot == shuffled_answer) for slot in range(candidate_count)]
        shortcut_policy = [int(slot == shortcut) for slot in range(candidate_count)]
        rows.append(
            Example(
                example_id=f"{split}_{steps}_{candidate_count}_{index:05d}",
                split=split,
                start_value=start,
                goal_value=goal,
                steps=steps,
                candidate_count=candidate_count,
                candidates=candidates,
                answer_label=answer_slot,
                surface_shortcut_label=shortcut,
                surface_shortcut_is_gold=shortcut == answer_slot,
                task_text=task_text,
                header_text=header_text,
                block_texts=block_texts,
                query_texts=query_texts,
                policy_bits=policy_bits,
                shuffled_policy_bits=shuffled_policy,
                shuffled_answer_label=shuffled_answer,
                shortcut_policy_bits=shortcut_policy,
            )
        )
    return rows


def make_dataset(job: dict[str, Any]) -> tuple[list[Example], list[Example]]:
    train_rows = make_examples(
        split="train",
        count=int(job["train_examples"]),
        seed=int(job["seed"]),
        steps=int(job["steps"]),
        candidate_count=int(job["candidate_count"]),
        train_surface_gold_prob=float(job["train_surface_gold_prob"]),
        eval_surface_wrong_prob=float(job["eval_surface_wrong_prob"]),
    )
    eval_rows = make_examples(
        split="eval",
        count=int(job["eval_examples"]),
        seed=int(job["seed"]),
        steps=int(job["steps"]),
        candidate_count=int(job["candidate_count"]),
        train_surface_gold_prob=float(job["train_surface_gold_prob"]),
        eval_surface_wrong_prob=float(job["eval_surface_wrong_prob"]),
    )
    return train_rows, eval_rows


def encode_text(text: str, max_chars: int) -> list[int]:
    clipped = text[:max_chars]
    ids = [CHAR_TO_ID.get(char, UNK) for char in clipped]
    if len(ids) < max_chars:
        ids.extend([PAD] * (max_chars - len(ids)))
    return ids


def encode_full_rows(rows: list[Example], max_chars: int) -> list[list[int]]:
    return [encode_text(row.task_text, max_chars) for row in rows]


def encode_header_rows(rows: list[Example], max_chars: int) -> list[list[int]]:
    return [encode_text(row.header_text, max_chars) for row in rows]


def encode_block_rows(rows: list[Example], max_chars: int) -> list[list[list[int]]]:
    return [[encode_text(text, max_chars) for text in row.block_texts] for row in rows]


def encode_query_rows(rows: list[Example], max_chars: int) -> list[list[list[int]]]:
    return [[encode_text(text, max_chars) for text in row.query_texts] for row in rows]


def surface_alignment(rows: list[Example]) -> float:
    return sum(1 for row in rows if row.surface_shortcut_is_gold) / len(rows)


def surface_flip_rate(rows: list[Example]) -> float:
    return sum(1 for row in rows if not row.surface_shortcut_is_gold) / len(rows)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class CnnEncoder:
    def __init__(self, embed_dim: int, hidden: int):
        import torch

        self.embedding = torch.nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD)
        self.conv = torch.nn.Conv1d(embed_dim, hidden, kernel_size=5, padding=2)
        self.pool_width = 16
        self.output_dim = hidden * self.pool_width

    def __call__(self, x: Any) -> Any:
        import torch

        emb = self.embedding(x).transpose(1, 2)
        hidden = torch.relu(self.conv(emb))
        pooled = torch.nn.functional.adaptive_max_pool1d(hidden, self.pool_width)
        return pooled.flatten(start_dim=1)

    def parameters(self) -> Any:
        return list(self.embedding.parameters()) + list(self.conv.parameters())

    def train(self) -> None:
        self.embedding.train()
        self.conv.train()

    def eval(self) -> None:
        self.embedding.eval()
        self.conv.eval()


class GruEncoder:
    def __init__(self, embed_dim: int, hidden: int):
        import torch

        self.embedding = torch.nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD)
        self.gru = torch.nn.GRU(embed_dim, hidden, batch_first=True)
        self.output_dim = hidden

    def __call__(self, x: Any) -> Any:
        import torch

        emb = self.embedding(x)
        output, _ = self.gru(emb)
        lengths = (x != PAD).sum(dim=1).clamp(min=1) - 1
        batch = torch.arange(x.shape[0], device=output.device)
        return output[batch, lengths]

    def parameters(self) -> Any:
        return list(self.embedding.parameters()) + list(self.gru.parameters())

    def train(self) -> None:
        self.embedding.train()
        self.gru.train()

    def eval(self) -> None:
        self.embedding.eval()
        self.gru.eval()


def make_encoder(model_name: str, embed_dim: int, hidden: int) -> Any:
    if model_name == "CHAR_CNN":
        return CnnEncoder(embed_dim, hidden)
    if model_name == "CHAR_GRU":
        return GruEncoder(embed_dim, hidden)
    raise ValueError(f"unknown model: {model_name}")


class GlobalModel:
    def __init__(self, model_name: str, hidden: int, embed_dim: int, candidate_count: int):
        import torch

        self.candidate_count = candidate_count
        self.encoder = make_encoder(model_name, embed_dim, hidden)
        dim = self.encoder.output_dim
        self.answer_head = torch.nn.Linear(dim, candidate_count)
        self.start_head = torch.nn.Linear(dim, VALUE_CLASSES)
        self.goal_head = torch.nn.Linear(dim, VALUE_CLASSES)
        self.steps_head = torch.nn.Linear(dim, MAX_STEPS)
        self.op_head = torch.nn.Linear(dim, candidate_count * MAX_STEPS * len(OPERATIONS))
        self.arg_head = torch.nn.Linear(dim, candidate_count * MAX_STEPS * (MAX_OPERAND + 1))
        self.final_head = torch.nn.Linear(dim, candidate_count * VALUE_CLASSES)
        self.policy_head = torch.nn.Linear(dim, candidate_count)
        self.shortcut_head = torch.nn.Linear(dim, candidate_count)

    def __call__(self, x_task: Any, _x_units: Any = None, _x_header: Any = None) -> dict[str, Any]:
        h = self.encoder(x_task)
        return {
            "answer": self.answer_head(h),
            "start": self.start_head(h),
            "goal": self.goal_head(h),
            "steps": self.steps_head(h),
            "op": self.op_head(h).reshape(-1, self.candidate_count, MAX_STEPS, len(OPERATIONS)),
            "arg": self.arg_head(h).reshape(-1, self.candidate_count, MAX_STEPS, MAX_OPERAND + 1),
            "final": self.final_head(h).reshape(-1, self.candidate_count, VALUE_CLASSES),
            "policy": self.policy_head(h),
            "shortcut": self.shortcut_head(h),
        }

    def parameters(self) -> Any:
        params = list(self.encoder.parameters())
        for head in [
            self.answer_head,
            self.start_head,
            self.goal_head,
            self.steps_head,
            self.op_head,
            self.arg_head,
            self.final_head,
            self.policy_head,
            self.shortcut_head,
        ]:
            params.extend(list(head.parameters()))
        return params

    def train(self) -> None:
        self.encoder.train()
        for head in [
            self.answer_head,
            self.start_head,
            self.goal_head,
            self.steps_head,
            self.op_head,
            self.arg_head,
            self.final_head,
            self.policy_head,
            self.shortcut_head,
        ]:
            head.train()

    def eval(self) -> None:
        self.encoder.eval()
        for head in [
            self.answer_head,
            self.start_head,
            self.goal_head,
            self.steps_head,
            self.op_head,
            self.arg_head,
            self.final_head,
            self.policy_head,
            self.shortcut_head,
        ]:
            head.eval()


class UnitModel:
    def __init__(self, model_name: str, hidden: int, embed_dim: int, candidate_count: int):
        import torch

        self.candidate_count = candidate_count
        self.header_encoder = make_encoder(model_name, embed_dim, hidden)
        self.unit_encoder = make_encoder(model_name, embed_dim, hidden)
        hdim = self.header_encoder.output_dim
        udim = self.unit_encoder.output_dim
        self.answer_head = torch.nn.Linear(hdim, candidate_count)
        self.start_head = torch.nn.Linear(hdim, VALUE_CLASSES)
        self.goal_head = torch.nn.Linear(hdim, VALUE_CLASSES)
        self.steps_head = torch.nn.Linear(hdim, MAX_STEPS)
        self.op_head = torch.nn.Linear(udim, MAX_STEPS * len(OPERATIONS))
        self.arg_head = torch.nn.Linear(udim, MAX_STEPS * (MAX_OPERAND + 1))
        self.final_head = torch.nn.Linear(udim, VALUE_CLASSES)
        self.policy_head = torch.nn.Linear(udim, 1)
        self.shortcut_head = torch.nn.Linear(udim, 1)

    def __call__(self, _x_task: Any, x_units: Any, x_header: Any) -> dict[str, Any]:
        h = self.header_encoder(x_header)
        batch, slots, length = x_units.shape
        unit_h = self.unit_encoder(x_units.reshape(batch * slots, length))
        return {
            "answer": self.answer_head(h),
            "start": self.start_head(h),
            "goal": self.goal_head(h),
            "steps": self.steps_head(h),
            "op": self.op_head(unit_h).reshape(batch, slots, MAX_STEPS, len(OPERATIONS)),
            "arg": self.arg_head(unit_h).reshape(batch, slots, MAX_STEPS, MAX_OPERAND + 1),
            "final": self.final_head(unit_h).reshape(batch, slots, VALUE_CLASSES),
            "policy": self.policy_head(unit_h).reshape(batch, slots),
            "shortcut": self.shortcut_head(unit_h).reshape(batch, slots),
        }

    def parameters(self) -> Any:
        params = list(self.header_encoder.parameters()) + list(self.unit_encoder.parameters())
        for head in [
            self.answer_head,
            self.start_head,
            self.goal_head,
            self.steps_head,
            self.op_head,
            self.arg_head,
            self.final_head,
            self.policy_head,
            self.shortcut_head,
        ]:
            params.extend(list(head.parameters()))
        return params

    def train(self) -> None:
        self.header_encoder.train()
        self.unit_encoder.train()
        for head in [
            self.answer_head,
            self.start_head,
            self.goal_head,
            self.steps_head,
            self.op_head,
            self.arg_head,
            self.final_head,
            self.policy_head,
            self.shortcut_head,
        ]:
            head.train()

    def eval(self) -> None:
        self.header_encoder.eval()
        self.unit_encoder.eval()
        for head in [
            self.answer_head,
            self.start_head,
            self.goal_head,
            self.steps_head,
            self.op_head,
            self.arg_head,
            self.final_head,
            self.policy_head,
            self.shortcut_head,
        ]:
            head.eval()


def is_unit_arm(arm: str) -> bool:
    return arm in {"BLOCK_ONLY_PLAN_FIRST", "QUERY_FULL_TEXT_PLAN_FIRST", "SHUFFLED_QUERY_PLAN_FIRST", "SHORTCUT_TEACHER"}


def make_model(model_name: str, arm: str, hidden: int, embed_dim: int, candidate_count: int) -> Any:
    if is_unit_arm(arm):
        return UnitModel(model_name, hidden, embed_dim, candidate_count)
    return GlobalModel(model_name, hidden, embed_dim, candidate_count)


def arm_targets(rows: list[Example], arm: str) -> dict[str, Any]:
    import torch

    candidate_count = rows[0].candidate_count
    if arm == "SHUFFLED_QUERY_PLAN_FIRST":
        policy = torch.tensor([row.shuffled_policy_bits for row in rows], dtype=torch.float32)
        answer = torch.tensor([row.shuffled_answer_label for row in rows], dtype=torch.long)
    elif arm == "SHORTCUT_TEACHER":
        policy = torch.tensor([row.shortcut_policy_bits for row in rows], dtype=torch.float32)
        answer = torch.tensor([row.surface_shortcut_label for row in rows], dtype=torch.long)
    else:
        policy = torch.tensor([row.policy_bits for row in rows], dtype=torch.float32)
        answer = torch.tensor([row.answer_label for row in rows], dtype=torch.long)

    op_targets: list[list[list[int]]] = []
    arg_targets: list[list[list[int]]] = []
    final_targets: list[list[int]] = []
    for row in rows:
        row_ops: list[list[int]] = []
        row_args: list[list[int]] = []
        row_finals: list[int] = []
        for candidate in row.candidates:
            ops = [operation.op_id for operation in candidate.ops] + [0] * (MAX_STEPS - row.steps)
            args = [operation.arg for operation in candidate.ops] + [0] * (MAX_STEPS - row.steps)
            row_ops.append(ops)
            row_args.append(args)
            row_finals.append(candidate.final_value)
        op_targets.append(row_ops)
        arg_targets.append(row_args)
        final_targets.append(row_finals)

    return {
        "answer": answer,
        "start": torch.tensor([row.start_value for row in rows], dtype=torch.long),
        "goal": torch.tensor([row.goal_value for row in rows], dtype=torch.long),
        "steps": torch.tensor([row.steps - 1 for row in rows], dtype=torch.long),
        "op": torch.tensor(op_targets, dtype=torch.long),
        "arg": torch.tensor(arg_targets, dtype=torch.long),
        "final": torch.tensor(final_targets, dtype=torch.long),
        "policy": policy,
        "shortcut": torch.tensor(
            [[int(slot == row.surface_shortcut_label) for slot in range(candidate_count)] for row in rows],
            dtype=torch.float32,
        ),
    }


def final_logits(outputs: dict[str, Any], arm: str) -> Any:
    if arm in {
        "GLOBAL_PLAN_FIRST",
        "BLOCK_ONLY_PLAN_FIRST",
        "QUERY_FULL_TEXT_PLAN_FIRST",
        "SHUFFLED_QUERY_PLAN_FIRST",
        "SHORTCUT_TEACHER",
    }:
        return outputs["policy"]
    return outputs["answer"]


def compute_loss(outputs: dict[str, Any], targets: dict[str, Any], arm: str, aux_weight: float, steps: int) -> Any:
    import torch
    import torch.nn.functional as F

    loss = F.cross_entropy(final_logits(outputs, arm), targets["answer"])
    if arm != "ANSWER_ONLY_DIRECT":
        loss = loss + aux_weight * F.cross_entropy(outputs["start"], targets["start"])
        loss = loss + aux_weight * F.cross_entropy(outputs["goal"], targets["goal"])
        loss = loss + aux_weight * F.cross_entropy(outputs["steps"], targets["steps"])
        loss = loss + aux_weight * F.cross_entropy(
            outputs["op"][:, :, :steps, :].reshape(-1, len(OPERATIONS)), targets["op"][:, :, :steps].reshape(-1)
        )
        loss = loss + aux_weight * F.cross_entropy(
            outputs["arg"][:, :, :steps, :].reshape(-1, MAX_OPERAND + 1), targets["arg"][:, :, :steps].reshape(-1)
        )
        loss = loss + aux_weight * F.cross_entropy(outputs["final"].reshape(-1, VALUE_CLASSES), targets["final"].reshape(-1))
        pos_weight = torch.tensor(float(outputs["policy"].shape[1] - 1), device=outputs["policy"].device)
        loss = loss + aux_weight * F.binary_cross_entropy_with_logits(
            outputs["policy"], targets["policy"], pos_weight=pos_weight
        )
        loss = loss + 0.25 * aux_weight * F.binary_cross_entropy_with_logits(outputs["shortcut"], targets["shortcut"])
    return loss


def evaluate(outputs: dict[str, Any], rows: list[Example], arm: str) -> dict[str, float]:
    pred_answer = final_logits(outputs, arm).argmax(dim=1).detach().cpu().tolist()
    pred_start = outputs["start"].argmax(dim=1).detach().cpu().tolist()
    pred_goal = outputs["goal"].argmax(dim=1).detach().cpu().tolist()
    pred_steps = (outputs["steps"].argmax(dim=1) + 1).detach().cpu().tolist()
    pred_ops = outputs["op"].argmax(dim=3).detach().cpu().tolist()
    pred_args = outputs["arg"].argmax(dim=3).detach().cpu().tolist()
    pred_final = outputs["final"].argmax(dim=2).detach().cpu().tolist()
    pred_policy_logits = outputs["policy"].detach().cpu()
    pred_policy_bits = (pred_policy_logits > 0.0).int().tolist()
    pred_policy_answer = outputs["policy"].argmax(dim=1).detach().cpu().tolist()

    correct = 0
    trap = 0
    start_correct = 0
    goal_correct = 0
    steps_correct = 0
    ops_correct = 0
    ops_total = 0
    final_correct = 0
    policy_correct = 0
    plan_exact = 0
    consistency = 0
    for index, row in enumerate(rows):
        answer_ok = int(pred_answer[index] == row.answer_label)
        correct += answer_ok
        if (not row.surface_shortcut_is_gold) and pred_answer[index] == row.surface_shortcut_label:
            trap += 1
        start_ok = int(pred_start[index] == row.start_value)
        goal_ok = int(pred_goal[index] == row.goal_value)
        steps_ok = int(pred_steps[index] == row.steps)
        start_correct += start_ok
        goal_correct += goal_ok
        steps_correct += steps_ok
        all_ops_ok = True
        all_final_ok = True
        all_policy_ok = True
        for slot, candidate in enumerate(row.candidates):
            for step in range(row.steps):
                op_ok = int(
                    pred_ops[index][slot][step] == candidate.ops[step].op_id
                    and pred_args[index][slot][step] == candidate.ops[step].arg
                )
                ops_correct += op_ok
                ops_total += 1
                all_ops_ok = all_ops_ok and bool(op_ok)
            final_ok = int(pred_final[index][slot] == candidate.final_value)
            final_correct += final_ok
            all_final_ok = all_final_ok and bool(final_ok)
            policy_ok = int(pred_policy_bits[index][slot] == row.policy_bits[slot])
            policy_correct += policy_ok
            all_policy_ok = all_policy_ok and bool(policy_ok)
        plan_exact += int(start_ok and goal_ok and steps_ok and all_ops_ok and all_final_ok and all_policy_ok)
        consistency += int(pred_answer[index] == pred_policy_answer[index])

    denom_candidates = len(rows) * rows[0].candidate_count
    return {
        "answer_eval_accuracy": correct / len(rows),
        "shortcut_trap_rate": trap / len(rows),
        "start_accuracy": start_correct / len(rows),
        "goal_accuracy": goal_correct / len(rows),
        "steps_accuracy": steps_correct / len(rows),
        "candidate_ops_accuracy": ops_correct / max(1, ops_total),
        "candidate_final_value_accuracy": final_correct / denom_candidates,
        "candidate_policy_accuracy": policy_correct / denom_candidates,
        "plan_exact_row_accuracy": plan_exact / len(rows),
        "answer_from_policy_consistency": consistency / len(rows),
    }


def oracle_metrics(rows: list[Example]) -> dict[str, float]:
    return {
        "answer_eval_accuracy": 1.0,
        "shortcut_trap_rate": 0.0,
        "start_accuracy": 1.0,
        "goal_accuracy": 1.0,
        "steps_accuracy": 1.0,
        "candidate_ops_accuracy": 1.0,
        "candidate_final_value_accuracy": 1.0,
        "candidate_policy_accuracy": 1.0,
        "plan_exact_row_accuracy": 1.0,
        "answer_from_policy_consistency": 1.0,
    }


def train_and_eval(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    try:
        import torch

        set_determinism(
            int(job["seed"])
            + int(job["candidate_count"]) * 101
            + int(job["steps"]) * 1009
            + MODELS.index(job["model"]) * 37
            + ARMS.index(job["arm"]) * 17
        )
        train_rows, eval_rows = make_dataset(job)
        train_alignment = surface_alignment(train_rows)
        eval_flip = surface_flip_rate(eval_rows)
        valid_stress = train_alignment >= 0.85 and eval_flip >= 0.85
        leak_tokens = ["ANSWER", "POLICY", "MATCH", "FINAL", "CHOOSE"]
        feature_leak_audit = all(
            not any(token in row.task_text.upper() for token in leak_tokens)
            for row in train_rows[:50] + eval_rows[:50]
        )

        if job["arm"] == "ORACLE_PARSED_PLAN_VISIBLE":
            train_metrics = oracle_metrics(train_rows)
            eval_metrics = oracle_metrics(eval_rows)
            status = JOB_COMPLETE if valid_stress and feature_leak_audit else JOB_INVALID
            return {
                "event": "job_complete",
                "job_id": job["job_id"],
                "job": job,
                "status": status,
                "elapsed_s": round(time.time() - started, 3),
                "stress": {
                    "train_surface_alignment": train_alignment,
                    "eval_surface_flip_rate": eval_flip,
                    "valid_stress": valid_stress,
                    "feature_leak_audit": feature_leak_audit,
                },
                "metrics": {"train_accuracy": train_metrics["answer_eval_accuracy"], **eval_metrics},
                "time": time.time(),
            }

        x_train = torch.tensor(encode_full_rows(train_rows, int(job["max_chars"])), dtype=torch.long)
        x_eval = torch.tensor(encode_full_rows(eval_rows, int(job["max_chars"])), dtype=torch.long)
        header_train = torch.tensor(encode_header_rows(train_rows, int(job["max_chars"])), dtype=torch.long)
        header_eval = torch.tensor(encode_header_rows(eval_rows, int(job["max_chars"])), dtype=torch.long)
        if job["arm"] == "BLOCK_ONLY_PLAN_FIRST":
            unit_train = torch.tensor(encode_block_rows(train_rows, int(job["max_chars"])), dtype=torch.long)
            unit_eval = torch.tensor(encode_block_rows(eval_rows, int(job["max_chars"])), dtype=torch.long)
        else:
            unit_train = torch.tensor(encode_query_rows(train_rows, int(job["max_chars"])), dtype=torch.long)
            unit_eval = torch.tensor(encode_query_rows(eval_rows, int(job["max_chars"])), dtype=torch.long)

        train_targets = arm_targets(train_rows, job["arm"])
        model = make_model(job["model"], job["arm"], int(job["hidden"]), int(job["embed_dim"]), int(job["candidate_count"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(job["lr"]))
        batch_size = max(1, int(job["batch_size"]))

        for _epoch in range(int(job["epochs"])):
            model.train()
            for start in range(0, len(train_rows), batch_size):
                end = min(len(train_rows), start + batch_size)
                batch_targets = {key: value[start:end] for key, value in train_targets.items()}
                optimizer.zero_grad()
                outputs = model(x_train[start:end], unit_train[start:end], header_train[start:end])
                loss = compute_loss(outputs, batch_targets, job["arm"], float(job["aux_weight"]), int(job["steps"]))
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            train_outputs = model(x_train, unit_train, header_train)
            eval_outputs = model(x_eval, unit_eval, header_eval)
        train_metrics = evaluate(train_outputs, train_rows, job["arm"])
        eval_metrics = evaluate(eval_outputs, eval_rows, job["arm"])
        status = JOB_COMPLETE if valid_stress and feature_leak_audit else JOB_INVALID
        return {
            "event": "job_complete",
            "job_id": job["job_id"],
            "job": job,
            "status": status,
            "elapsed_s": round(time.time() - started, 3),
            "stress": {
                "train_surface_alignment": train_alignment,
                "eval_surface_flip_rate": eval_flip,
                "valid_stress": valid_stress,
                "feature_leak_audit": feature_leak_audit,
            },
            "metrics": {"train_accuracy": train_metrics["answer_eval_accuracy"], **eval_metrics},
            "time": time.time(),
        }
    except Exception as exc:  # pragma: no cover
        return {
            "event": "job_complete",
            "job_id": job.get("job_id", "unknown"),
            "job": job,
            "status": STATUS_BLOCKED,
            "elapsed_s": round(time.time() - started, 3),
            "error": repr(exc),
            "time": time.time(),
        }


def job_id(job: dict[str, Any]) -> str:
    return (
        f"{job['model']}_{job['arm']}_c{job['candidate_count']}_s{job['steps']}"
        f"_seed{job['seed']}_n{job['train_examples']}_ep{job['epochs']}"
    )


def build_queue(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_seed_spec(args.seeds)
    models = parse_csv(args.models)
    arms = parse_csv(args.arms)
    candidate_counts = parse_csv(args.candidate_counts, int)
    steps_list = parse_csv(args.steps, int)
    primary_seeds = seeds[:5]
    remaining_seeds = seeds[5:]
    seed_groups = [primary_seeds]
    if remaining_seeds:
        seed_groups.append(remaining_seeds)
    queue: list[dict[str, Any]] = []
    for seed_group in seed_groups:
        for candidate_count in candidate_counts:
            if candidate_count not in (4, 8):
                raise ValueError("--candidate-counts supports 4 and 8")
            for steps in steps_list:
                if steps < 1 or steps > MAX_STEPS:
                    raise ValueError("--steps supports 1,2,3")
                for seed in seed_group:
                    for model in models:
                        if model not in MODELS:
                            raise ValueError(f"unknown model: {model}")
                        for arm in arms:
                            if arm not in ARMS:
                                raise ValueError(f"unknown arm: {arm}")
                            job = {
                                "seed": seed,
                                "model": model,
                                "arm": arm,
                                "candidate_count": candidate_count,
                                "steps": steps,
                                "train_examples": args.train_examples,
                                "eval_examples": args.eval_examples,
                                "epochs": args.epochs,
                                "lr": args.lr,
                                "hidden": args.hidden,
                                "embed_dim": args.embed_dim,
                                "batch_size": args.batch_size,
                                "max_chars": args.max_chars,
                                "aux_weight": args.aux_weight,
                                "train_surface_gold_prob": args.train_surface_gold_prob,
                                "eval_surface_wrong_prob": args.eval_surface_wrong_prob,
                            }
                            job["job_id"] = job_id(job)
                            queue.append(job)
    if args.max_jobs is not None:
        queue = queue[: args.max_jobs]
    return queue


def mean_metric(group: list[dict[str, Any]], key: str) -> float:
    return sum(row["metrics"][key] for row in group) / len(group)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == JOB_COMPLETE]
    blocked = [row for row in rows if row.get("status") == STATUS_BLOCKED]
    invalid = [row for row in rows if row.get("status") == JOB_INVALID]
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in valid:
        job = row["job"]
        key = (job["model"], job["arm"], int(job["candidate_count"]), int(job["steps"]))
        grouped.setdefault(key, []).append(row)

    by_group: dict[str, dict[str, Any]] = {}
    metrics_keys = [
        "train_accuracy",
        "answer_eval_accuracy",
        "shortcut_trap_rate",
        "start_accuracy",
        "goal_accuracy",
        "steps_accuracy",
        "candidate_ops_accuracy",
        "candidate_final_value_accuracy",
        "candidate_policy_accuracy",
        "plan_exact_row_accuracy",
        "answer_from_policy_consistency",
    ]
    for (model, arm, candidate_count, steps), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "model": model,
            "arm": arm,
            "candidate_count": candidate_count,
            "steps": steps,
            "jobs": len(group),
            "valid_seed_count": len({row["job"]["seed"] for row in group}),
        }
        for key in metrics_keys:
            item[key] = mean_metric(group, key)
        item["train_surface_alignment"] = sum(row["stress"]["train_surface_alignment"] for row in group) / len(group)
        item["eval_surface_flip_rate"] = sum(row["stress"]["eval_surface_flip_rate"] for row in group) / len(group)
        by_group[f"{model}|{arm}|c{candidate_count}|s{steps}"] = item

    def lookup(arm: str, candidate_count: int, steps: int, model: str = "CHAR_CNN") -> dict[str, Any] | None:
        return by_group.get(f"{model}|{arm}|c{candidate_count}|s{steps}")

    def arm_pass(item: dict[str, Any] | None, answer: dict[str, Any] | None, shuffled: dict[str, Any] | None, shortcut: dict[str, Any] | None) -> bool:
        if not item or not answer or not shuffled or not shortcut:
            return False
        return (
            item["answer_eval_accuracy"] >= 0.80
            and item["shortcut_trap_rate"] <= 0.25
            and item["candidate_final_value_accuracy"] >= 0.80
            and item["candidate_policy_accuracy"] >= 0.80
            and item["answer_from_policy_consistency"] >= 0.95
            and item["answer_eval_accuracy"] >= answer["answer_eval_accuracy"] + 0.25
            and item["answer_eval_accuracy"] >= shuffled["answer_eval_accuracy"] + 0.25
            and item["answer_eval_accuracy"] >= shortcut["answer_eval_accuracy"] + 0.25
        )

    query_passes: list[tuple[int, int]] = []
    block_passes: list[tuple[int, int]] = []
    for candidate_count in [4, 8]:
        for steps in [1, 2, 3]:
            answer = lookup("ANSWER_ONLY_DIRECT", candidate_count, steps)
            shuffled = lookup("SHUFFLED_QUERY_PLAN_FIRST", candidate_count, steps)
            shortcut = lookup("SHORTCUT_TEACHER", candidate_count, steps)
            query = lookup("QUERY_FULL_TEXT_PLAN_FIRST", candidate_count, steps)
            block = lookup("BLOCK_ONLY_PLAN_FIRST", candidate_count, steps)
            if arm_pass(query, answer, shuffled, shortcut):
                query_passes.append((candidate_count, steps))
            if arm_pass(block, answer, shuffled, shortcut):
                block_passes.append((candidate_count, steps))

    status = STATUS_NEGATIVE
    if blocked and not valid:
        status = STATUS_BLOCKED
    elif invalid and not valid:
        status = STATUS_INVALID
    elif any(candidate_count == 4 and steps == 3 for candidate_count, steps in query_passes):
        if not any(candidate_count == 8 and steps == 3 for candidate_count, steps in query_passes):
            status = STATUS_SCALE
        else:
            status = STATUS_QUERY
    elif any(candidate_count == 4 and steps == 1 for candidate_count, steps in query_passes + block_passes) and not any(
        candidate_count == 4 and steps >= 2 for candidate_count, steps in query_passes + block_passes
    ):
        status = STATUS_DEPTH
    elif block_passes:
        status = STATUS_BLOCK
    elif not valid:
        status = STATUS_PARTIAL

    return {
        "status": status,
        "completed_jobs": len(rows),
        "valid_jobs": len(valid),
        "invalid_jobs": len(invalid),
        "blocked_jobs": len(blocked),
        "by_group": by_group,
        "query_passes": query_passes,
        "block_passes": block_passes,
    }


def write_report(out: Path, summary: dict[str, Any], queue_len: int) -> None:
    lines = [
        "# ANCHOR-MINI-014 Operation-Plan Parser Result",
        "",
        f"status: `{summary['status']}`",
        f"completed_jobs: {summary['completed_jobs']} / {queue_len}",
        f"valid_jobs: {summary['valid_jobs']}",
        f"blocked_jobs: {summary['blocked_jobs']}",
        f"query_passes: {summary['query_passes']}",
        f"block_passes: {summary['block_passes']}",
        "",
        "## Grouped Metrics",
        "",
        "| model | arm | cand | steps | jobs | eval_acc | trap | start | goal | ops | final | policy | plan_exact | consistency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["by_group"].values():
        lines.append(
            f"| `{item['model']}` | `{item['arm']}` | {item['candidate_count']} | {item['steps']} | {item['jobs']} | "
            f"{item['answer_eval_accuracy']:.3f} | {item['shortcut_trap_rate']:.3f} | "
            f"{item['start_accuracy']:.3f} | {item['goal_accuracy']:.3f} | "
            f"{item['candidate_ops_accuracy']:.3f} | {item['candidate_final_value_accuracy']:.3f} | "
            f"{item['candidate_policy_accuracy']:.3f} | {item['plan_exact_row_accuracy']:.3f} | "
            f"{item['answer_from_policy_consistency']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This is a toy operation-plan parser/binding test, not symbol grounding at scale.",
            "Raw target outputs are intentionally kept outside tracked docs.",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_examples_sample(path: Path) -> None:
    rows = make_examples(
        split="eval",
        count=6,
        seed=2026,
        steps=3,
        candidate_count=4,
        train_surface_gold_prob=0.90,
        eval_surface_wrong_prob=0.90,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "example_id": row.example_id,
                        "task_text": row.task_text,
                        "start_value": row.start_value,
                        "goal_value": row.goal_value,
                        "steps": row.steps,
                        "answer_label": LABELS[row.answer_label],
                        "surface_shortcut_label": LABELS[row.surface_shortcut_label],
                        "final_values": [candidate.final_value for candidate in row.candidates],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    out = args.out
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    progress_path = out / "progress.jsonl"

    queue = build_queue(args)
    write_json(out / "queue.json", queue)
    write_examples_sample(out / "examples_sample.jsonl")
    if CONTRACT_FILE.exists():
        shutil.copyfile(CONTRACT_FILE, out / "contract_snapshot.md")
    else:
        (out / "contract_snapshot.md").write_text("contract file missing\n", encoding="utf-8")

    completed = load_completed(metrics_path)
    pending = [job for job in queue if job["job_id"] not in completed]

    budget_s = None
    if args.budget_hours is not None:
        budget_s = args.budget_hours * 3600.0
    if args.budget_minutes is not None:
        budget_s = args.budget_minutes * 60.0
    deadline = time.time() + budget_s if budget_s is not None else None

    active: dict[Any, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        while pending or active:
            while pending and len(active) < max(1, args.jobs):
                if deadline is not None and time.time() >= deadline:
                    break
                job = pending.pop(0)
                active[executor.submit(train_and_eval, job)] = job
                append_jsonl(progress_path, {"event": "job_started", "job_id": job["job_id"], "time": time.time()})
            if not active:
                break
            done, _ = wait(active.keys(), timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover
                    result = {
                        "event": "job_complete",
                        "job_id": job["job_id"],
                        "job": job,
                        "status": STATUS_BLOCKED,
                        "error": repr(exc),
                        "time": time.time(),
                    }
                append_jsonl(metrics_path, result)
            if deadline is not None and time.time() >= deadline and not active:
                break

    rows: list[dict[str, Any]] = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    summary = aggregate(rows)
    summary["queue_jobs"] = len(queue)
    summary["pending_jobs"] = len(queue) - len(rows)
    if summary["pending_jobs"] and summary["status"] != STATUS_BLOCKED:
        summary["status"] = STATUS_PARTIAL
    write_json(out / "summary.json", summary)
    write_json(out / "operation_curve.json", summary["by_group"])
    write_report(out, summary, len(queue))
    print(f"status: {summary['status']}")
    print(f"completed_jobs: {summary['completed_jobs']} / {len(queue)}")
    print(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
