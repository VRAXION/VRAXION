#!/usr/bin/env python3
"""Run ANCHOR-MINI-013 candidate-scoped parser A/B."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any


CONTRACT_FILE = Path("docs/research/ANCHOR_MINI_013_CANDIDATE_SCOPED_PARSER_CONTRACT.md")

MODELS = ["CHAR_CNN_QUERY_SCOPED", "CHAR_GRU_QUERY_SCOPED"]
ARMS = [
    "ANSWER_ONLY_DIRECT",
    "GLOBAL_PLAN_FIRST",
    "SCALE_ONLY_GLOBAL",
    "QUERY_SCOPED_PLAN_FIRST",
    "SHUFFLED_QUERY_SCOPED",
]

CANDIDATE_COUNT = 4
CATEGORY_COUNT = 4
GOAL_COUNT = 16
EFFECT_COUNT = 24
LABELS = ["A", "B", "C", "D"]

PAD = 0
UNK = 1
ASCII_ALPHABET = "\n\t" + "".join(chr(code) for code in range(32, 127))
CHAR_TO_ID = {char: index + 2 for index, char in enumerate(ASCII_ALPHABET)}
VOCAB_SIZE = len(CHAR_TO_ID) + 2

JOB_COMPLETE = "ANCHOR_MINI_013_JOB_COMPLETE"
JOB_INVALID = "ANCHOR_MINI_013_JOB_INVALID_STRESS"
STATUS_CANDIDATE_SCOPED = "ANCHOR_MINI_013_CANDIDATE_SCOPED_FIXES_BINDING"
STATUS_SCALE_ONLY = "ANCHOR_MINI_013_SCALE_ONLY_FIXES_BINDING"
STATUS_BOTH_PASS = "ANCHOR_MINI_013_BOTH_PASS"
STATUS_BOTH_FAIL = "ANCHOR_MINI_013_BOTH_FAIL"
STATUS_INVALID = "ANCHOR_MINI_013_INVALID_STRESS"
STATUS_BLOCKED = "ANCHOR_MINI_013_RESOURCE_BLOCKED"
STATUS_PARTIAL = "ANCHOR_MINI_013_PARTIAL_BUDGET"


@dataclass(frozen=True)
class TemplateSpec:
    template_id: int
    goal_alias: str
    effect_alias: str
    surface_alias: str
    delimiter: str
    pair_delimiter: str
    wrapper: str
    field_order: str
    slot_order: tuple[int, ...]


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str
    template_id: int
    task_text: str
    goal_category: int
    effect_categories: list[int]
    surface_buckets: list[int]
    answer_label: int
    surface_shortcut_label: int
    surface_shortcut_is_gold: bool
    match_bits: list[int]
    shuffled_goal_category: int
    shuffled_match_bits: list[int]


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
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-013 candidate-scoped parser A/B.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default="2026-2030")
    parser.add_argument("--models", default="CHAR_CNN_QUERY_SCOPED")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--template-counts", default="64,128,256")
    parser.add_argument("--eval-template-count", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--eval-examples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-chars", type=int, default=224)
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument("--train-surface-gold-prob", type=float, default=0.90)
    parser.add_argument("--eval-surface-wrong-prob", type=float, default=0.90)
    parser.add_argument("--template-seed", type=int, default=4242)
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


def effect_category(effect_id: int) -> int:
    return effect_id % CATEGORY_COUNT


def choose_surface_shortcut(
    *,
    split: str,
    answer_slot: int,
    rng: random.Random,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
) -> int:
    wrong_slots = [slot for slot in range(CANDIDATE_COUNT) if slot != answer_slot]
    if split == "train":
        if rng.random() < train_surface_gold_prob:
            return answer_slot
        return rng.choice(wrong_slots)
    if split == "eval":
        if rng.random() < eval_surface_wrong_prob:
            return rng.choice(wrong_slots)
        return answer_slot
    raise RuntimeError(f"unknown split: {split}")


def make_surface_priors(shortcut_slot: int, rng: random.Random) -> list[float]:
    surface = [0.03 + 0.24 * rng.random() for _ in range(CANDIDATE_COUNT)]
    surface[shortcut_slot] = 0.82 + 0.17 * rng.random()
    return surface


def surface_buckets(surface_priors: list[float], shortcut_slot: int) -> list[int]:
    buckets: list[int] = []
    for slot, value in enumerate(surface_priors):
        if slot == shortcut_slot:
            buckets.append(9)
        else:
            buckets.append(max(0, min(2, int(value * 10))))
    return buckets


def make_template_pool(count: int, *, seed: int) -> list[TemplateSpec]:
    rng = random.Random(seed)
    goal_aliases = ["G", "GOAL", "TARGET", "NEED", "REQ", "WANT", "OBJ", "Q"]
    effect_aliases = ["E", "EFF", "FX", "OUT", "DOES", "CAT", "RESULT", "R"]
    surface_aliases = ["S", "SURF", "PRIOR", "CUE", "LOOK", "BUMP", "VIS", "P"]
    delimiters = [";", "|", "\n", " / ", " :: ", " ~ "]
    pair_delimiters = [":", ",", " ", "/", "="]
    wrappers = ["compact", "paren", "bracket", "words", "arrow", "csv"]
    field_orders = ["effect_surface", "surface_effect"]
    slot_orders = [
        (0, 1, 2, 3),
        (2, 0, 3, 1),
        (1, 3, 0, 2),
        (3, 2, 1, 0),
        (0, 2, 1, 3),
        (1, 0, 3, 2),
    ]
    specs: list[TemplateSpec] = []
    seen: set[tuple[str, str, str, str, str, str, str, tuple[int, ...]]] = set()
    while len(specs) < count:
        spec = TemplateSpec(
            template_id=len(specs),
            goal_alias=rng.choice(goal_aliases),
            effect_alias=rng.choice(effect_aliases),
            surface_alias=rng.choice(surface_aliases),
            delimiter=rng.choice(delimiters),
            pair_delimiter=rng.choice(pair_delimiters),
            wrapper=rng.choice(wrappers),
            field_order=rng.choice(field_orders),
            slot_order=rng.choice(slot_orders),
        )
        key = (
            spec.goal_alias,
            spec.effect_alias,
            spec.surface_alias,
            spec.delimiter,
            spec.pair_delimiter,
            spec.wrapper,
            spec.field_order,
            spec.slot_order,
        )
        if key in seen:
            continue
        seen.add(key)
        specs.append(spec)
    return specs


def render_candidate(spec: TemplateSpec, label: str, effect: int, surface: int) -> str:
    if spec.field_order == "effect_surface":
        first_name, first_value = spec.effect_alias, effect
        second_name, second_value = spec.surface_alias, surface
    else:
        first_name, first_value = spec.surface_alias, surface
        second_name, second_value = spec.effect_alias, effect
    pd = spec.pair_delimiter
    if spec.wrapper == "compact":
        return f"{label}={first_name}{first_value}{pd}{second_name}{second_value}"
    if spec.wrapper == "paren":
        return f"{label}({first_name}={first_value}{pd}{second_name}={second_value})"
    if spec.wrapper == "bracket":
        return f"[{label} {first_name}{pd}{first_value} {second_name}{pd}{second_value}]"
    if spec.wrapper == "words":
        return f"{label} {first_name} {first_value} {second_name} {second_value}"
    if spec.wrapper == "arrow":
        return f"{label}->{first_name}{pd}{first_value}->{second_name}{pd}{second_value}"
    if spec.wrapper == "csv":
        return f"{label},{first_name},{first_value},{second_name},{second_value}"
    raise ValueError(f"unknown wrapper: {spec.wrapper}")


def serialize_task(goal_category: int, effect_categories: list[int], buckets: list[int], spec: TemplateSpec) -> str:
    chunks = [f"{spec.goal_alias}={goal_category}"]
    for slot in spec.slot_order:
        chunks.append(render_candidate(spec, LABELS[slot], effect_categories[slot], buckets[slot]))
    return spec.delimiter.join(chunks)


def make_examples(
    *,
    split: str,
    count: int,
    rng: random.Random,
    templates: list[TemplateSpec],
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
) -> list[Example]:
    effects_by_category = {
        category: [effect for effect in range(EFFECT_COUNT) if effect_category(effect) == category]
        for category in range(CATEGORY_COUNT)
    }
    rows: list[Example] = []
    for index in range(count):
        spec = templates[index % len(templates)]
        goal_id = rng.randrange(GOAL_COUNT)
        goal_category = goal_id % CATEGORY_COUNT
        answer_slot = rng.randrange(CANDIDATE_COUNT)
        non_goal_categories = [category for category in range(CATEGORY_COUNT) if category != goal_category]
        rng.shuffle(non_goal_categories)
        effect_ids: list[int] = []
        distractor_index = 0
        for slot in range(CANDIDATE_COUNT):
            if slot == answer_slot:
                effect_ids.append(rng.choice(effects_by_category[goal_category]))
            else:
                effect_ids.append(rng.choice(effects_by_category[non_goal_categories[distractor_index]]))
                distractor_index += 1
        effect_categories = [effect_category(effect) for effect in effect_ids]
        match_bits = [int(category == goal_category) for category in effect_categories]
        shortcut = choose_surface_shortcut(
            split=split,
            answer_slot=answer_slot,
            rng=rng,
            train_surface_gold_prob=train_surface_gold_prob,
            eval_surface_wrong_prob=eval_surface_wrong_prob,
        )
        surface_priors = make_surface_priors(shortcut, rng)
        buckets = surface_buckets(surface_priors, shortcut)
        shifted_goal = (goal_category + 1) % CATEGORY_COUNT
        shifted_bits = [int(category == shifted_goal) for category in effect_categories]
        rows.append(
            Example(
                example_id=f"{split}_{index:05d}",
                split=split,
                template_id=spec.template_id,
                task_text=serialize_task(goal_category, effect_categories, buckets, spec),
                goal_category=goal_category,
                effect_categories=effect_categories,
                surface_buckets=buckets,
                answer_label=answer_slot,
                surface_shortcut_label=shortcut,
                surface_shortcut_is_gold=shortcut == answer_slot,
                match_bits=match_bits,
                shuffled_goal_category=shifted_goal,
                shuffled_match_bits=shifted_bits,
            )
        )
    return rows


def make_dataset(job: dict[str, Any]) -> tuple[list[Example], list[Example], list[TemplateSpec], list[TemplateSpec]]:
    pool_size = int(job["train_template_count"]) + int(job["eval_template_count"])
    template_pool = make_template_pool(pool_size, seed=int(job["template_seed"]))
    train_templates = template_pool[: int(job["train_template_count"])]
    eval_templates = template_pool[int(job["train_template_count"]) :]
    rng = random.Random(int(job["seed"]))
    train_rows = make_examples(
        split="train",
        count=int(job["train_examples"]),
        rng=rng,
        templates=train_templates,
        train_surface_gold_prob=float(job["train_surface_gold_prob"]),
        eval_surface_wrong_prob=float(job["eval_surface_wrong_prob"]),
    )
    eval_rows = make_examples(
        split="eval",
        count=int(job["eval_examples"]),
        rng=rng,
        templates=eval_templates,
        train_surface_gold_prob=float(job["train_surface_gold_prob"]),
        eval_surface_wrong_prob=float(job["eval_surface_wrong_prob"]),
    )
    return train_rows, eval_rows, train_templates, eval_templates


def encode_text(text: str, max_chars: int) -> list[int]:
    clipped = text[:max_chars]
    ids = [CHAR_TO_ID.get(char, UNK) for char in clipped]
    if len(ids) < max_chars:
        ids.extend([PAD] * (max_chars - len(ids)))
    return ids


def encode_rows(rows: list[Example], max_chars: int) -> list[list[int]]:
    return [encode_text(row.task_text, max_chars) for row in rows]


def encode_query_rows(rows: list[Example], max_chars: int) -> list[list[list[int]]]:
    encoded: list[list[list[int]]] = []
    for row in rows:
        encoded.append([encode_text(f"QUERY={label}\n{row.task_text}", max_chars) for label in LABELS])
    return encoded


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
    if model_name == "CHAR_CNN_QUERY_SCOPED":
        return CnnEncoder(embed_dim, hidden)
    if model_name == "CHAR_GRU_QUERY_SCOPED":
        return GruEncoder(embed_dim, hidden)
    raise ValueError(f"unknown model: {model_name}")


class GlobalModel:
    def __init__(self, model_name: str, hidden: int, embed_dim: int):
        import torch

        self.encoder = make_encoder(model_name, embed_dim, hidden)
        self.answer_head = torch.nn.Linear(self.encoder.output_dim, CANDIDATE_COUNT)
        self.goal_head = torch.nn.Linear(self.encoder.output_dim, CATEGORY_COUNT)
        self.effect_head = torch.nn.Linear(self.encoder.output_dim, CANDIDATE_COUNT * CATEGORY_COUNT)
        self.policy_head = torch.nn.Linear(self.encoder.output_dim, CANDIDATE_COUNT)

    def __call__(self, x_task: Any, _x_query: Any = None) -> dict[str, Any]:
        h = self.encoder(x_task)
        return {
            "answer": self.answer_head(h),
            "goal": self.goal_head(h),
            "effect": self.effect_head(h).reshape(-1, CANDIDATE_COUNT, CATEGORY_COUNT),
            "policy": self.policy_head(h),
        }

    def parameters(self) -> Any:
        return (
            list(self.encoder.parameters())
            + list(self.answer_head.parameters())
            + list(self.goal_head.parameters())
            + list(self.effect_head.parameters())
            + list(self.policy_head.parameters())
        )

    def train(self) -> None:
        self.encoder.train()
        self.answer_head.train()
        self.goal_head.train()
        self.effect_head.train()
        self.policy_head.train()

    def eval(self) -> None:
        self.encoder.eval()
        self.answer_head.eval()
        self.goal_head.eval()
        self.effect_head.eval()
        self.policy_head.eval()


class QueryScopedModel:
    def __init__(self, model_name: str, hidden: int, embed_dim: int):
        import torch

        self.goal_encoder = make_encoder(model_name, embed_dim, hidden)
        self.query_encoder = make_encoder(model_name, embed_dim, hidden)
        self.goal_head = torch.nn.Linear(self.goal_encoder.output_dim, CATEGORY_COUNT)
        self.effect_head = torch.nn.Linear(self.query_encoder.output_dim, CATEGORY_COUNT)
        self.policy_head = torch.nn.Linear(self.query_encoder.output_dim, 1)
        self.answer_head = torch.nn.Linear(self.goal_encoder.output_dim, CANDIDATE_COUNT)

    def __call__(self, x_task: Any, x_query: Any) -> dict[str, Any]:
        import torch

        goal_h = self.goal_encoder(x_task)
        batch, slots, length = x_query.shape
        query_h = self.query_encoder(x_query.reshape(batch * slots, length))
        effect = self.effect_head(query_h).reshape(batch, slots, CATEGORY_COUNT)
        policy = self.policy_head(query_h).reshape(batch, slots)
        return {
            "answer": self.answer_head(goal_h),
            "goal": self.goal_head(goal_h),
            "effect": effect,
            "policy": policy,
        }

    def parameters(self) -> Any:
        return (
            list(self.goal_encoder.parameters())
            + list(self.query_encoder.parameters())
            + list(self.goal_head.parameters())
            + list(self.effect_head.parameters())
            + list(self.policy_head.parameters())
            + list(self.answer_head.parameters())
        )

    def train(self) -> None:
        self.goal_encoder.train()
        self.query_encoder.train()
        self.goal_head.train()
        self.effect_head.train()
        self.policy_head.train()
        self.answer_head.train()

    def eval(self) -> None:
        self.goal_encoder.eval()
        self.query_encoder.eval()
        self.goal_head.eval()
        self.effect_head.eval()
        self.policy_head.eval()
        self.answer_head.eval()


def is_query_arm(arm: str) -> bool:
    return arm in {"QUERY_SCOPED_PLAN_FIRST", "SHUFFLED_QUERY_SCOPED"}


def make_model(model_name: str, arm: str, hidden: int, embed_dim: int) -> Any:
    if is_query_arm(arm):
        return QueryScopedModel(model_name, hidden, embed_dim)
    return GlobalModel(model_name, hidden, embed_dim)


def arm_targets(rows: list[Example], arm: str) -> dict[str, Any]:
    import torch

    if arm == "SHUFFLED_QUERY_SCOPED":
        goal = torch.tensor([row.shuffled_goal_category for row in rows], dtype=torch.long)
        policy = torch.tensor([row.shuffled_match_bits for row in rows], dtype=torch.float32)
        answer = torch.tensor([row.shuffled_match_bits.index(1) for row in rows], dtype=torch.long)
    else:
        goal = torch.tensor([row.goal_category for row in rows], dtype=torch.long)
        policy = torch.tensor([row.match_bits for row in rows], dtype=torch.float32)
        answer = torch.tensor([row.answer_label for row in rows], dtype=torch.long)
    return {
        "answer": answer,
        "goal": goal,
        "effect": torch.tensor([row.effect_categories for row in rows], dtype=torch.long),
        "policy": policy,
    }


def final_logits(outputs: dict[str, Any], arm: str) -> Any:
    if arm in {"GLOBAL_PLAN_FIRST", "SCALE_ONLY_GLOBAL", "QUERY_SCOPED_PLAN_FIRST", "SHUFFLED_QUERY_SCOPED"}:
        return outputs["policy"]
    return outputs["answer"]


def compute_loss(outputs: dict[str, Any], targets: dict[str, Any], arm: str, aux_weight: float) -> Any:
    import torch
    import torch.nn.functional as F

    loss = F.cross_entropy(final_logits(outputs, arm), targets["answer"])
    if arm != "ANSWER_ONLY_DIRECT":
        loss = loss + aux_weight * F.cross_entropy(outputs["goal"], targets["goal"])
        loss = loss + aux_weight * F.cross_entropy(outputs["effect"].reshape(-1, CATEGORY_COUNT), targets["effect"].reshape(-1))
        pos_weight = torch.tensor(3.0, device=outputs["policy"].device)
        loss = loss + aux_weight * F.binary_cross_entropy_with_logits(
            outputs["policy"], targets["policy"], pos_weight=pos_weight
        )
    return loss


def evaluate(outputs: dict[str, Any], rows: list[Example], arm: str) -> dict[str, float]:
    pred_answer = final_logits(outputs, arm).argmax(dim=1).detach().cpu().tolist()
    pred_goal = outputs["goal"].argmax(dim=1).detach().cpu().tolist()
    pred_effect = outputs["effect"].argmax(dim=2).detach().cpu().tolist()
    pred_policy_logits = outputs["policy"].detach().cpu()
    pred_policy_bits = (pred_policy_logits > 0.0).int().tolist()
    pred_policy_answer = outputs["policy"].argmax(dim=1).detach().cpu().tolist()

    correct = 0
    trap = 0
    goal_correct = 0
    effect_correct = 0
    policy_correct = 0
    plan_exact = 0
    consistency = 0
    by_template: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        answer_ok = int(pred_answer[index] == row.answer_label)
        correct += answer_ok
        if (not row.surface_shortcut_is_gold) and pred_answer[index] == row.surface_shortcut_label:
            trap += 1
        goal_ok = int(pred_goal[index] == row.goal_category)
        goal_correct += goal_ok
        effect_ok_count = 0
        for slot in range(CANDIDATE_COUNT):
            effect_ok_count += int(pred_effect[index][slot] == row.effect_categories[slot])
            policy_correct += int(pred_policy_bits[index][slot] == row.match_bits[slot])
        effect_correct += effect_ok_count
        effect_ok = effect_ok_count == CANDIDATE_COUNT
        policy_ok = all(pred_policy_bits[index][slot] == row.match_bits[slot] for slot in range(CANDIDATE_COUNT))
        plan_exact += int(goal_ok and effect_ok and policy_ok)
        consistency += int(pred_answer[index] == pred_policy_answer[index])
        by_template.setdefault(row.template_id, [0, 0])
        by_template[row.template_id][0] += answer_ok
        by_template[row.template_id][1] += 1

    heldout_rates = [hits / total for hits, total in by_template.values() if total]
    return {
        "answer_eval_accuracy": correct / len(rows),
        "shortcut_trap_rate": trap / len(rows),
        "goal_category_accuracy": goal_correct / len(rows),
        "candidate_effect_accuracy": effect_correct / (len(rows) * CANDIDATE_COUNT),
        "candidate_policy_accuracy": policy_correct / (len(rows) * CANDIDATE_COUNT),
        "plan_exact_row_accuracy": plan_exact / len(rows),
        "answer_from_policy_consistency": consistency / len(rows),
        "heldout_template_accuracy": sum(heldout_rates) / len(heldout_rates) if heldout_rates else 0.0,
    }


def train_and_eval(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    try:
        import torch

        set_determinism(
            int(job["seed"])
            + int(job["train_template_count"]) * 1009
            + MODELS.index(job["model"]) * 101
            + ARMS.index(job["arm"]) * 17
        )
        train_rows, eval_rows, train_templates, eval_templates = make_dataset(job)
        train_alignment = surface_alignment(train_rows)
        eval_flip = surface_flip_rate(eval_rows)
        valid_stress = train_alignment >= 0.85 and eval_flip >= 0.85
        feature_leak_audit = all(
            "ANSWER" not in row.task_text.upper()
            and "MATCH" not in row.task_text.upper()
            and "POLICY" not in row.task_text.upper()
            for row in train_rows[:50] + eval_rows[:50]
        )

        x_train = torch.tensor(encode_rows(train_rows, int(job["max_chars"])), dtype=torch.long)
        x_eval = torch.tensor(encode_rows(eval_rows, int(job["max_chars"])), dtype=torch.long)
        xq_train = torch.tensor(encode_query_rows(train_rows, int(job["max_chars"])), dtype=torch.long)
        xq_eval = torch.tensor(encode_query_rows(eval_rows, int(job["max_chars"])), dtype=torch.long)
        train_targets = arm_targets(train_rows, job["arm"])

        model = make_model(job["model"], job["arm"], int(job["hidden"]), int(job["embed_dim"]))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(job["lr"]))
        batch_size = max(1, int(job["batch_size"]))

        for _epoch in range(int(job["epochs"])):
            model.train()
            for start in range(0, len(train_rows), batch_size):
                end = min(len(train_rows), start + batch_size)
                batch_targets = {key: value[start:end] for key, value in train_targets.items()}
                optimizer.zero_grad()
                outputs = model(x_train[start:end], xq_train[start:end])
                loss = compute_loss(outputs, batch_targets, job["arm"], float(job["aux_weight"]))
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            train_outputs = model(x_train, xq_train)
            eval_outputs = model(x_eval, xq_eval)
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
                "train_template_ids": [spec.template_id for spec in train_templates],
                "eval_template_ids": [spec.template_id for spec in eval_templates],
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
        f"{job['model']}_{job['arm']}_tc{job['train_template_count']}"
        f"_seed{job['seed']}_n{job['train_examples']}_ep{job['epochs']}"
    )


def build_queue(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_seed_spec(args.seeds)
    models = parse_csv(args.models)
    arms = parse_csv(args.arms)
    template_counts = parse_csv(args.template_counts, int)
    queue: list[dict[str, Any]] = []
    for seed in seeds:
        for model in models:
            if model not in MODELS:
                raise ValueError(f"unknown model: {model}")
            for template_count in template_counts:
                for arm in arms:
                    if arm not in ARMS:
                        raise ValueError(f"unknown arm: {arm}")
                    job = {
                        "seed": seed,
                        "model": model,
                        "arm": arm,
                        "train_template_count": template_count,
                        "eval_template_count": args.eval_template_count,
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
                        "template_seed": args.template_seed,
                    }
                    job["job_id"] = job_id(job)
                    queue.append(job)
    if args.max_jobs is not None:
        queue = queue[: args.max_jobs]
    return queue


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == JOB_COMPLETE]
    blocked = [row for row in rows if row.get("status") == STATUS_BLOCKED]
    invalid = [row for row in rows if row.get("status") == JOB_INVALID]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in valid:
        job = row["job"]
        key = (job["model"], job["arm"], int(job["train_template_count"]))
        grouped.setdefault(key, []).append(row)

    by_group: dict[str, dict[str, Any]] = {}
    for (model, arm, template_count), group in sorted(grouped.items()):
        metrics_keys = [
            "train_accuracy",
            "answer_eval_accuracy",
            "shortcut_trap_rate",
            "goal_category_accuracy",
            "candidate_effect_accuracy",
            "candidate_policy_accuracy",
            "plan_exact_row_accuracy",
            "answer_from_policy_consistency",
            "heldout_template_accuracy",
        ]
        item: dict[str, Any] = {
            "model": model,
            "arm": arm,
            "template_count": template_count,
            "jobs": len(group),
            "valid_seed_count": len({row["job"]["seed"] for row in group}),
        }
        for key in metrics_keys:
            item[key] = sum(row["metrics"][key] for row in group) / len(group)
        item["train_surface_alignment"] = sum(row["stress"]["train_surface_alignment"] for row in group) / len(group)
        item["eval_surface_flip_rate"] = sum(row["stress"]["eval_surface_flip_rate"] for row in group) / len(group)
        by_group[f"{model}|{arm}|{template_count}"] = item

    def lookup(model: str, arm: str, tc: int) -> dict[str, Any] | None:
        return by_group.get(f"{model}|{arm}|{tc}")

    max_tc = max((item["template_count"] for item in by_group.values()), default=0)
    query_pass = False
    global_pass = False
    for model in MODELS:
        query = lookup(model, "QUERY_SCOPED_PLAN_FIRST", max_tc)
        shuffled = lookup(model, "SHUFFLED_QUERY_SCOPED", max_tc)
        global_plan = lookup(model, "GLOBAL_PLAN_FIRST", max_tc) or lookup(model, "SCALE_ONLY_GLOBAL", max_tc)
        scale = lookup(model, "SCALE_ONLY_GLOBAL", max_tc)
        if query and shuffled and global_plan:
            query_pass = query_pass or (
                query["answer_eval_accuracy"] >= 0.80
                and query["shortcut_trap_rate"] <= 0.25
                and query["candidate_effect_accuracy"] >= 0.80
                and query["candidate_policy_accuracy"] >= 0.80
                and query["answer_from_policy_consistency"] >= 0.95
                and query["answer_eval_accuracy"] >= global_plan["answer_eval_accuracy"] + 0.25
                and query["answer_eval_accuracy"] >= shuffled["answer_eval_accuracy"] + 0.25
            )
        if scale:
            global_pass = global_pass or (
                scale["answer_eval_accuracy"] >= 0.80
                and scale["shortcut_trap_rate"] <= 0.25
                and scale["candidate_effect_accuracy"] >= 0.80
                and scale["candidate_policy_accuracy"] >= 0.80
            )

    if blocked and not valid:
        status = STATUS_BLOCKED
    elif invalid and not valid:
        status = STATUS_INVALID
    elif query_pass and global_pass:
        status = STATUS_BOTH_PASS
    elif query_pass:
        status = STATUS_CANDIDATE_SCOPED
    elif global_pass:
        status = STATUS_SCALE_ONLY
    elif valid:
        status = STATUS_BOTH_FAIL
    else:
        status = STATUS_PARTIAL

    return {
        "status": status,
        "completed_jobs": len(rows),
        "valid_jobs": len(valid),
        "invalid_jobs": len(invalid),
        "blocked_jobs": len(blocked),
        "by_group": by_group,
        "query_pass": query_pass,
        "global_pass": global_pass,
        "max_template_count": max_tc,
    }


def write_report(out: Path, summary: dict[str, Any], queue_len: int) -> None:
    lines = [
        "# ANCHOR-MINI-013 Candidate-Scoped Parser Result",
        "",
        f"status: `{summary['status']}`",
        f"completed_jobs: {summary['completed_jobs']} / {queue_len}",
        f"valid_jobs: {summary['valid_jobs']}",
        f"blocked_jobs: {summary['blocked_jobs']}",
        f"query_pass: {summary['query_pass']}",
        f"global_pass: {summary['global_pass']}",
        "",
        "## Grouped Metrics",
        "",
        "| model | arm | templates | jobs | eval_acc | trap | goal | cand_effect | cand_policy | plan_exact | consistency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["by_group"].values():
        lines.append(
            f"| `{item['model']}` | `{item['arm']}` | {item['template_count']} | {item['jobs']} | "
            f"{item['answer_eval_accuracy']:.3f} | {item['shortcut_trap_rate']:.3f} | "
            f"{item['goal_category_accuracy']:.3f} | {item['candidate_effect_accuracy']:.3f} | "
            f"{item['candidate_policy_accuracy']:.3f} | {item['plan_exact_row_accuracy']:.3f} | "
            f"{item['answer_from_policy_consistency']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This is a toy candidate-local binding test. It does not prove natural-language grounding.",
            "Query labels scope the slot being inspected but do not reveal effect, match, policy, or answer.",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_examples_sample(path: Path, *, seed: int, template_seed: int) -> None:
    pool = make_template_pool(288, seed=template_seed)
    rng = random.Random(seed)
    rows = make_examples(
        split="train",
        count=4,
        rng=rng,
        templates=pool[:4],
        train_surface_gold_prob=0.90,
        eval_surface_wrong_prob=0.90,
    ) + make_examples(
        split="eval",
        count=4,
        rng=rng,
        templates=pool[256:260],
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
                        "split": row.split,
                        "template_id": row.template_id,
                        "task_text": row.task_text,
                        "query_A": f"QUERY=A\n{row.task_text}",
                        "goal_category": row.goal_category,
                        "effect_categories": row.effect_categories,
                        "surface_buckets": row.surface_buckets,
                        "answer_label": row.answer_label,
                        "surface_shortcut_label": row.surface_shortcut_label,
                        "surface_shortcut_is_gold": row.surface_shortcut_is_gold,
                        "match_bits": row.match_bits,
                        "shuffled_goal_category": row.shuffled_goal_category,
                        "shuffled_match_bits": row.shuffled_match_bits,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    progress_path = out / "progress.jsonl"
    queue = build_queue(args)
    completed = load_completed(metrics_path)
    pending = [job for job in queue if job["job_id"] not in completed]

    write_json(out / "queue.json", {"jobs": queue})
    if (root / CONTRACT_FILE).exists():
        shutil.copyfile(root / CONTRACT_FILE, out / "contract_snapshot.md")
    write_examples_sample(out / "examples_sample.jsonl", seed=2026, template_seed=args.template_seed)

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
    write_json(out / "template_curve.json", summary["by_group"])
    write_report(out, summary, len(queue))
    print(f"status: {summary['status']}")
    print(f"completed_jobs: {summary['completed_jobs']} / {len(queue)}")
    print(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
