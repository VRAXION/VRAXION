#!/usr/bin/env python3
"""Run ANCHOR-MINI-007 literal text-carrier format sweep."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import shutil
import string
import sys
import time
from typing import Any


CONTRACT_FILE = Path("docs/research/ANCHOR_MINI_007_TEXT_CARRIER_CONTRACT.md")

FORMAT_ARMS = [
    "ANSWER_ONLY",
    "PROSE_PROCESS",
    "INNER_MONOLOGUE_PROCESS",
    "STRICT_JSON_PROCESS",
    "FLAT_KEY_VALUE_PROCESS",
    "RELATIONAL_TRIPLES_PROCESS",
    "ACTION_OUTCOME_TABLE_PROCESS",
    "RELATION_PLUS_ACTION_PROCESS",
    "COMPACT_HYBRID_PROCESS",
    "SHUFFLED_ACTION_OUTCOME_TABLE",
    "SHUFFLED_COMPACT_HYBRID",
]

MODELS = ["CHAR_BOW_MLP", "CHAR_CNN", "CHAR_GRU"]

CANDIDATE_COUNT = 4
CATEGORY_COUNT = 4
GOAL_COUNT = 16
EFFECT_COUNT = 24
LABELS = ["A", "B", "C", "D"]

STATUS_STRONG = "ANCHOR_MINI_007_TEXT_FORMAT_STRONG_SIGNAL"
STATUS_WEAK = "ANCHOR_MINI_007_TEXT_FORMAT_WEAK_SIGNAL"
STATUS_NEGATIVE = "ANCHOR_MINI_007_TEXT_FORMAT_NEGATIVE"
STATUS_INVALID = "ANCHOR_MINI_007_TEXT_FORMAT_INVALID_STRESS"
STATUS_BLOCKED = "ANCHOR_MINI_007_TEXT_FORMAT_RESOURCE_BLOCKED"
JOB_COMPLETE = "ANCHOR_MINI_007_JOB_COMPLETE"
JOB_INVALID = "ANCHOR_MINI_007_JOB_INVALID_STRESS"

PAD = 0
UNK = 1
ASCII_ALPHABET = "\n\t" + "".join(chr(code) for code in range(32, 127))
CHAR_TO_ID = {char: index + 2 for index, char in enumerate(ASCII_ALPHABET)}
VOCAB_SIZE = len(CHAR_TO_ID) + 2


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str
    goal_id: int
    goal_category: int
    effect_ids: list[int]
    effect_categories: list[int]
    surface_priors: list[float]
    answer_label: int
    surface_shortcut_label: int
    surface_shortcut_is_gold: bool


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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-007 text carrier sweep.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default="2026-2125")
    parser.add_argument("--models", default="CHAR_GRU,CHAR_BOW_MLP")
    parser.add_argument("--format-arms", default=",".join(FORMAT_ARMS))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--budget-hours", type=float, default=None)
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--train-examples", type=int, default=1024)
    parser.add_argument("--eval-examples", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-chars", type=int, default=768)
    parser.add_argument("--teacher-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.5)
    parser.add_argument("--train-surface-gold-prob", type=float, default=0.90)
    parser.add_argument("--eval-surface-wrong-prob", type=float, default=0.90)
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_completed(metrics_path: Path) -> set[str]:
    completed: set[str] = set()
    if not metrics_path.exists():
        return completed
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add(row["job_id"])
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


def make_examples(
    *,
    split: str,
    count: int,
    rng: random.Random,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
) -> list[Example]:
    effects_by_category = {
        category: [effect for effect in range(EFFECT_COUNT) if effect_category(effect) == category]
        for category in range(CATEGORY_COUNT)
    }
    rows: list[Example] = []
    for index in range(count):
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
        shortcut = choose_surface_shortcut(
            split=split,
            answer_slot=answer_slot,
            rng=rng,
            train_surface_gold_prob=train_surface_gold_prob,
            eval_surface_wrong_prob=eval_surface_wrong_prob,
        )
        rows.append(
            Example(
                example_id=f"{split}_{index:05d}",
                split=split,
                goal_id=goal_id,
                goal_category=goal_category,
                effect_ids=effect_ids,
                effect_categories=effect_categories,
                surface_priors=make_surface_priors(shortcut, rng),
                answer_label=answer_slot,
                surface_shortcut_label=shortcut,
                surface_shortcut_is_gold=shortcut == answer_slot,
            )
        )
    return rows


def make_dataset(
    *,
    seed: int,
    train_examples: int,
    eval_examples: int,
    train_surface_gold_prob: float,
    eval_surface_wrong_prob: float,
) -> tuple[list[Example], list[Example]]:
    rng = random.Random(seed)
    train_rows = make_examples(
        split="train",
        count=train_examples,
        rng=rng,
        train_surface_gold_prob=train_surface_gold_prob,
        eval_surface_wrong_prob=eval_surface_wrong_prob,
    )
    eval_rows = make_examples(
        split="eval",
        count=eval_examples,
        rng=rng,
        train_surface_gold_prob=train_surface_gold_prob,
        eval_surface_wrong_prob=eval_surface_wrong_prob,
    )
    return train_rows, eval_rows


def task_text(example: Example) -> str:
    rows = [
        "TASK",
        f"goal: category_{example.goal_category}",
        "candidates:",
    ]
    for slot, label in enumerate(LABELS):
        rows.append(
            f"{label} surface={example.surface_priors[slot]:.2f} "
            f"effect=category_{example.effect_categories[slot]}"
        )
    return "\n".join(rows)


def process_matches(example: Example, *, shifted: bool = False) -> list[int]:
    goal = (example.goal_category + 1) % CATEGORY_COUNT if shifted else example.goal_category
    return [int(category == goal) for category in example.effect_categories]


def process_text(format_arm: str, example: Example) -> str:
    if format_arm == "ANSWER_ONLY":
        return ""
    shuffled = format_arm in {"SHUFFLED_ACTION_OUTCOME_TABLE", "SHUFFLED_COMPACT_HYBRID"}
    goal = (example.goal_category + 1) % CATEGORY_COUNT if shuffled else example.goal_category
    matches = process_matches(example, shifted=shuffled)
    effects = example.effect_categories
    surface = [round(value, 2) for value in example.surface_priors]
    if format_arm == "PROSE_PROCESS":
        return (
            f"The goal needs category {goal}. Look at what each candidate changes. "
            "The useful candidate is the one whose effect category fits the goal. "
            "The large surface number is only a tempting shortcut."
        )
    if format_arm == "INNER_MONOLOGUE_PROCESS":
        return (
            f"I notice a loud surface cue {surface}, and part of me wants to follow it. "
            f"But the goal is category {goal}, so I compare the effect categories {effects}. "
            "I should pick the candidate that actually fits the goal, not the shiny-looking one."
        )
    if format_arm == "STRICT_JSON_PROCESS":
        return json.dumps(
            {
                "implicit_job": "select_candidate_by_goal_effect_fit",
                "goal": {"category": f"category_{goal}"},
                "candidates": [
                    {
                        "slot": LABELS[idx],
                        "effect_category": f"category_{category}",
                        "matches_goal": bool(matches[idx]),
                    }
                    for idx, category in enumerate(effects)
                ],
                "decision_rule": "choose_matches_goal_true",
            },
            sort_keys=True,
        )
    if format_arm == "FLAT_KEY_VALUE_PROCESS":
        return (
            f"job=select_fit\n"
            f"goal_category=category_{goal}\n"
            f"effect_categories={','.join('category_' + str(category) for category in effects)}\n"
            f"surface_priors={','.join(str(value) for value in surface)}"
        )
    if format_arm == "RELATIONAL_TRIPLES_PROCESS":
        triples = [f"goal -> requires -> category_{goal}"]
        for idx, category in enumerate(effects):
            triples.append(f"candidate_{LABELS[idx]} -> has_effect -> category_{category}")
            triples.append(f"candidate_{LABELS[idx]} -> matches_goal -> {'yes' if matches[idx] else 'no'}")
        return "\n".join(triples)
    if format_arm in {"ACTION_OUTCOME_TABLE_PROCESS", "SHUFFLED_ACTION_OUTCOME_TABLE"}:
        rows = ["candidate | effect_category | surface_prior | policy"]
        for idx, category in enumerate(effects):
            policy = "choose" if matches[idx] else "reject"
            rows.append(f"{LABELS[idx]} | category_{category} | {surface[idx]:.2f} | {policy}")
        return "\n".join(rows)
    if format_arm == "RELATION_PLUS_ACTION_PROCESS":
        return process_text("RELATIONAL_TRIPLES_PROCESS", example) + "\n" + process_text(
            "ACTION_OUTCOME_TABLE_PROCESS", example
        )
    if format_arm in {"COMPACT_HYBRID_PROCESS", "SHUFFLED_COMPACT_HYBRID"}:
        return (
            "ImplicitJob: choose the candidate whose effect fits the goal.\n"
            f"Salience: goal_category=category_{goal}; surface score is a shortcut risk.\n"
            f"Relations: effects={','.join('category_' + str(category) for category in effects)}; "
            f"match={','.join(str(bit) for bit in matches)}.\n"
            "ActionOutcome: matching effect supports the goal; nonmatching effect is a trap.\n"
            "DecisionRule: choose the matching candidate."
        )
    raise ValueError(f"unknown format arm: {format_arm}")


def compose_input(example: Example, format_arm: str, *, include_process: bool) -> str:
    base = task_text(example)
    if include_process and format_arm != "ANSWER_ONLY":
        base += "\n\nPROCESS\n" + process_text(format_arm, example)
    return base


def token_count(text: str) -> int:
    if not text:
        return 0
    normalized = text
    for char in "{}[]():,;|\"'`=\n":
        normalized = normalized.replace(char, " ")
    return len([part for part in normalized.split(" ") if part])


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def format_stats(rows: list[Example], format_arm: str) -> dict[str, float]:
    texts = [process_text(format_arm, row) for row in rows[: min(200, len(rows))]]
    chars = [len(text) for text in texts]
    tokens = [token_count(text) for text in texts]
    return {
        "format_char_count_mean": sum(chars) / len(chars) if chars else 0.0,
        "format_char_count_p95": percentile(chars, 0.95),
        "format_token_count_mean": sum(tokens) / len(tokens) if tokens else 0.0,
        "format_token_count_p95": percentile(tokens, 0.95),
    }


def encode_text(text: str, max_chars: int) -> list[int]:
    clipped = text[:max_chars]
    ids = [CHAR_TO_ID.get(char, UNK) for char in clipped]
    if len(ids) < max_chars:
        ids.extend([PAD] * (max_chars - len(ids)))
    return ids


def encode_rows(rows: list[Example], format_arm: str, *, include_process: bool, max_chars: int) -> list[list[int]]:
    return [encode_text(compose_input(row, format_arm, include_process=include_process), max_chars) for row in rows]


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


class CharBowMlp:
    def __init__(self, vocab_size: int, hidden: int):
        import torch

        self.model = torch.nn.Sequential(
            torch.nn.Linear(vocab_size, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, CANDIDATE_COUNT),
        )

    def __call__(self, x: Any) -> Any:
        return self.model(x)

    def parameters(self) -> Any:
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()


class CharCnn:
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int):
        import torch

        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.conv = torch.nn.Conv1d(embed_dim, hidden, kernel_size=5, padding=2)
        self.pool_width = 16
        self.head = torch.nn.Linear(hidden * self.pool_width, CANDIDATE_COUNT)

    def __call__(self, x: Any) -> Any:
        import torch

        emb = self.embedding(x).transpose(1, 2)
        hidden = torch.relu(self.conv(emb))
        pooled = torch.nn.functional.adaptive_max_pool1d(hidden, self.pool_width)
        return self.head(pooled.flatten(start_dim=1))

    def parameters(self) -> Any:
        return list(self.embedding.parameters()) + list(self.conv.parameters()) + list(self.head.parameters())

    def train(self) -> None:
        self.embedding.train()
        self.conv.train()
        self.head.train()

    def eval(self) -> None:
        self.embedding.eval()
        self.conv.eval()
        self.head.eval()


class CharGru:
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int):
        import torch

        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.gru = torch.nn.GRU(embed_dim, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, CANDIDATE_COUNT)

    def __call__(self, x: Any) -> Any:
        import torch

        emb = self.embedding(x)
        lengths = (x != PAD).sum(dim=1).clamp(min=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, state = self.gru(packed)
        return self.head(state[-1])

    def parameters(self) -> Any:
        return list(self.embedding.parameters()) + list(self.gru.parameters()) + list(self.head.parameters())

    def train(self) -> None:
        self.embedding.train()
        self.gru.train()
        self.head.train()

    def eval(self) -> None:
        self.embedding.eval()
        self.gru.eval()
        self.head.eval()


def make_model(model_name: str, hidden: int, embed_dim: int) -> Any:
    if model_name == "CHAR_BOW_MLP":
        return CharBowMlp(VOCAB_SIZE, hidden)
    if model_name == "CHAR_CNN":
        return CharCnn(VOCAB_SIZE, embed_dim, hidden)
    if model_name == "CHAR_GRU":
        return CharGru(VOCAB_SIZE, embed_dim, hidden)
    raise ValueError(f"unknown model: {model_name}")


def bow_features(x: Any) -> Any:
    import torch

    counts = torch.zeros((x.shape[0], VOCAB_SIZE), dtype=torch.float32)
    ones = torch.ones(x.shape[1], dtype=torch.float32)
    for index in range(x.shape[0]):
        counts[index].scatter_add_(0, x[index], ones)
    counts[:, PAD] = 0.0
    return counts / x.shape[1]


def accuracy_and_trap(logits: Any, rows: list[Example]) -> dict[str, float]:
    import torch

    pred = logits.argmax(dim=1).detach().cpu().tolist()
    correct = 0
    trap = 0
    for value, row in zip(pred, rows):
        if value == row.answer_label:
            correct += 1
        if (not row.surface_shortcut_is_gold) and value == row.surface_shortcut_label:
            trap += 1
    return {
        "accuracy": correct / len(rows),
        "shortcut_trap_rate": trap / len(rows),
    }


def train_and_eval(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    try:
        import torch
        import torch.nn.functional as F

        set_determinism(int(job["seed"]) + int(job["model_offset"]) + int(job["format_offset"]))
        train_rows, eval_rows = make_dataset(
            seed=job["seed"],
            train_examples=job["train_examples"],
            eval_examples=job["eval_examples"],
            train_surface_gold_prob=job["train_surface_gold_prob"],
            eval_surface_wrong_prob=job["eval_surface_wrong_prob"],
        )
        train_alignment = surface_alignment(train_rows)
        eval_flip = surface_flip_rate(eval_rows)
        stats = format_stats(train_rows, job["format_arm"])
        valid_stress = train_alignment >= 0.85 and eval_flip >= 0.85

        x_task_train = torch.tensor(
            encode_rows(train_rows, job["format_arm"], include_process=False, max_chars=job["max_chars"]),
            dtype=torch.long,
        )
        x_proc_train = None
        if job["format_arm"] != "ANSWER_ONLY":
            x_proc_train = torch.tensor(
                encode_rows(train_rows, job["format_arm"], include_process=True, max_chars=job["max_chars"]),
                dtype=torch.long,
            )
        x_eval = torch.tensor(
            encode_rows(eval_rows, job["format_arm"], include_process=False, max_chars=job["max_chars"]),
            dtype=torch.long,
        )
        y_train = torch.tensor([row.answer_label for row in train_rows], dtype=torch.long)
        y_eval = torch.tensor([row.answer_label for row in eval_rows], dtype=torch.long)

        model = make_model(job["model"], job["hidden"], job["embed_dim"])
        if job["model"] == "CHAR_BOW_MLP":
            x_task_train = bow_features(x_task_train)
            if x_proc_train is not None:
                x_proc_train = bow_features(x_proc_train)
            x_eval = bow_features(x_eval)
        optimizer = torch.optim.Adam(model.parameters(), lr=job["lr"])

        batch_size = max(1, int(job["batch_size"]))
        for _epoch in range(job["epochs"]):
            model.train()
            for start in range(0, len(train_rows), batch_size):
                end = min(len(train_rows), start + batch_size)
                xb_task = x_task_train[start:end]
                yb = y_train[start:end]
                optimizer.zero_grad()
                logits_task = model(xb_task)
                loss = F.cross_entropy(logits_task, yb)
                if x_proc_train is not None:
                    xb_proc = x_proc_train[start:end]
                    logits_proc = model(xb_proc)
                    loss = loss + job["teacher_loss_weight"] * F.cross_entropy(logits_proc, yb)
                    teacher = F.softmax(logits_proc.detach(), dim=1)
                    student_log = F.log_softmax(logits_task, dim=1)
                    loss = loss + job["consistency_loss_weight"] * F.kl_div(
                        student_log, teacher, reduction="batchmean"
                    )
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(x_task_train)
            eval_logits = model(x_eval)
        train_metrics = accuracy_and_trap(train_logits, train_rows)
        eval_metrics = accuracy_and_trap(eval_logits, eval_rows)

        status = JOB_COMPLETE if valid_stress else JOB_INVALID
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
                "eval_process_visible": False,
            },
            "metrics": {
                "train_accuracy": train_metrics["accuracy"],
                "answer_eval_ood_accuracy": eval_metrics["accuracy"],
                "shortcut_trap_rate": eval_metrics["shortcut_trap_rate"],
            },
            "format": stats,
            "time": time.time(),
        }
    except Exception as exc:  # pragma: no cover - reported in run artifacts
        return {
            "event": "job_complete",
            "job_id": job["job_id"],
            "job": job,
            "status": STATUS_BLOCKED,
            "elapsed_s": round(time.time() - started, 3),
            "error": repr(exc),
            "time": time.time(),
        }


def job_id(job: dict[str, Any]) -> str:
    return (
        f"{job['model']}_{job['format_arm']}_seed{job['seed']}"
        f"_n{job['train_examples']}_ep{job['epochs']}_mc{job['max_chars']}"
    )


def build_queue(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds = parse_seed_spec(args.seeds)
    models = parse_csv(args.models)
    arms = parse_csv(args.format_arms)
    unknown_models = sorted(set(models) - set(MODELS))
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    unknown_arms = sorted(set(arms) - set(FORMAT_ARMS))
    if unknown_arms:
        raise ValueError(f"unknown format arms: {unknown_arms}")
    queue: list[dict[str, Any]] = []
    for seed in seeds:
        for model_index, model in enumerate(models):
            for arm_index, arm in enumerate(arms):
                job = {
                    "seed": seed,
                    "model": model,
                    "format_arm": arm,
                    "train_examples": args.train_examples,
                    "eval_examples": args.eval_examples,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "hidden": args.hidden,
                    "embed_dim": args.embed_dim,
                    "batch_size": args.batch_size,
                    "max_chars": args.max_chars,
                    "teacher_loss_weight": args.teacher_loss_weight,
                    "consistency_loss_weight": args.consistency_loss_weight,
                    "train_surface_gold_prob": args.train_surface_gold_prob,
                    "eval_surface_wrong_prob": args.eval_surface_wrong_prob,
                    "model_offset": (model_index + 1) * 1009,
                    "format_offset": (arm_index + 1) * 2039,
                }
                job["job_id"] = job_id(job)
                queue.append(job)
    if args.max_jobs is not None:
        queue = queue[: args.max_jobs]
    return queue


def matching_shuffled_arm(format_arm: str) -> str:
    if format_arm == "ACTION_OUTCOME_TABLE_PROCESS":
        return "SHUFFLED_ACTION_OUTCOME_TABLE"
    if format_arm == "COMPACT_HYBRID_PROCESS":
        return "SHUFFLED_COMPACT_HYBRID"
    return "SHUFFLED_COMPACT_HYBRID"


def aggregate(rows: list[dict[str, Any]], requested_seeds: list[int]) -> dict[str, Any]:
    blocked = [row for row in rows if row["status"] == STATUS_BLOCKED]
    completed = [row for row in rows if row["status"] in {JOB_COMPLETE, JOB_INVALID}]
    valid = [row for row in completed if row.get("stress", {}).get("valid_stress")]
    by_model_format: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in valid:
        job = row["job"]
        by_model_format.setdefault((job["model"], job["format_arm"]), []).append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for key, group in by_model_format.items():
        model, arm = key
        ood = [row["metrics"]["answer_eval_ood_accuracy"] for row in group]
        trap = [row["metrics"]["shortcut_trap_rate"] for row in group]
        train_acc = [row["metrics"]["train_accuracy"] for row in group]
        chars = [row["format"]["format_char_count_mean"] for row in group]
        tokens = [row["format"]["format_token_count_mean"] for row in group]
        summaries[f"{model}/{arm}"] = {
            "model": model,
            "format_arm": arm,
            "valid_jobs": len(group),
            "valid_seeds": len({row["job"]["seed"] for row in group}),
            "answer_eval_ood_accuracy": sum(ood) / len(ood),
            "shortcut_trap_rate": sum(trap) / len(trap),
            "train_accuracy": sum(train_acc) / len(train_acc),
            "format_char_count_mean": sum(chars) / len(chars),
            "format_token_count_mean": sum(tokens) / len(tokens),
        }
        summaries[f"{model}/{arm}"]["format_efficiency"] = summaries[f"{model}/{arm}"][
            "answer_eval_ood_accuracy"
        ] / max(1.0, summaries[f"{model}/{arm}"]["format_token_count_mean"])

    useful: dict[str, dict[str, Any]] = {}
    for model in sorted({row["job"]["model"] for row in valid}):
        answer = summaries.get(f"{model}/ANSWER_ONLY")
        if not answer:
            continue
        for arm in FORMAT_ARMS:
            if arm.startswith("SHUFFLED") or arm == "ANSWER_ONLY":
                continue
            current = summaries.get(f"{model}/{arm}")
            shuffled = summaries.get(f"{model}/{matching_shuffled_arm(arm)}")
            if not current or not shuffled:
                continue
            answer_gap = current["answer_eval_ood_accuracy"] - answer["answer_eval_ood_accuracy"]
            shuffled_gap = current["answer_eval_ood_accuracy"] - shuffled["answer_eval_ood_accuracy"]
            current["answer_only_gap"] = answer_gap
            current["same_format_shuffled_gap"] = shuffled_gap
            is_useful = (
                answer_gap >= 0.20
                and shuffled_gap >= 0.20
                and current["shortcut_trap_rate"] <= 0.30
            )
            if is_useful:
                useful[f"{model}/{arm}"] = current

    available_models = sorted({row["job"]["model"] for row in valid})
    if "CHAR_GRU" in available_models:
        primary_model = "CHAR_GRU"
    elif "CHAR_CNN" in available_models:
        primary_model = "CHAR_CNN"
    elif "CHAR_BOW_MLP" in available_models:
        primary_model = "CHAR_BOW_MLP"
    else:
        primary_model = "NONE"

    primary_action = summaries.get(f"{primary_model}/ACTION_OUTCOME_TABLE_PROCESS")
    primary_answer = summaries.get(f"{primary_model}/ANSWER_ONLY")
    primary_prose = summaries.get(f"{primary_model}/PROSE_PROCESS")
    primary_inner = summaries.get(f"{primary_model}/INNER_MONOLOGUE_PROCESS")
    primary_shuffled_action = summaries.get(f"{primary_model}/SHUFFLED_ACTION_OUTCOME_TABLE")
    primary_compact = summaries.get(f"{primary_model}/COMPACT_HYBRID_PROCESS")
    primary_relation_action = summaries.get(f"{primary_model}/RELATION_PLUS_ACTION_PROCESS")

    valid_seed_count = len({row["job"]["seed"] for row in valid})
    required_valid_seeds = min(80, len(requested_seeds))
    full_valid_stress = valid_seed_count >= required_valid_seeds

    answer_fails_ood = bool(
        primary_answer
        and primary_answer["answer_eval_ood_accuracy"] <= 0.60
        and primary_answer["shortcut_trap_rate"] >= 0.30
    )
    action_useful = f"{primary_model}/ACTION_OUTCOME_TABLE_PROCESS" in useful
    action_beats_prose_inner = bool(
        primary_action
        and primary_prose
        and primary_inner
        and primary_action["answer_eval_ood_accuracy"] > primary_prose["answer_eval_ood_accuracy"]
        and primary_action["answer_eval_ood_accuracy"] > primary_inner["answer_eval_ood_accuracy"]
    )
    action_beats_shuffled = bool(
        primary_action
        and primary_shuffled_action
        and primary_action["answer_eval_ood_accuracy"] >= primary_shuffled_action["answer_eval_ood_accuracy"] + 0.20
    )
    structured_positive = bool(
        primary_compact
        and primary_answer
        and primary_compact["answer_eval_ood_accuracy"] >= primary_answer["answer_eval_ood_accuracy"] + 0.10
    ) or bool(
        primary_relation_action
        and primary_answer
        and primary_relation_action["answer_eval_ood_accuracy"] >= primary_answer["answer_eval_ood_accuracy"] + 0.10
    )

    if blocked and not completed:
        status = STATUS_BLOCKED
    elif not full_valid_stress:
        status = STATUS_INVALID
    elif action_useful and action_beats_prose_inner and action_beats_shuffled and structured_positive and answer_fails_ood:
        status = STATUS_STRONG
    elif useful:
        status = STATUS_WEAK
    else:
        status = STATUS_NEGATIVE

    return {
        "status": status,
        "completed_jobs": len(completed),
        "blocked_jobs": len(blocked),
        "valid_jobs": len(valid),
        "valid_seed_count": valid_seed_count,
        "required_valid_seeds": required_valid_seeds,
        "primary_model": primary_model,
        "primary_model_is_gru_gate": primary_model == "CHAR_GRU",
        "conditions": {
            "full_valid_stress": full_valid_stress,
            "action_outcome_table_useful_on_primary_model": action_useful,
            "action_beats_prose_and_inner": action_beats_prose_inner,
            "action_beats_shuffled_action": action_beats_shuffled,
            "compact_or_relation_plus_action_directionally_positive": structured_positive,
            "answer_only_fails_ood": answer_fails_ood,
            "primary_model_is_gru_gate": primary_model == "CHAR_GRU",
            "no_blocked_jobs": not blocked,
        },
        "by_model_format": summaries,
        "useful_formats": sorted(useful),
    }


def write_report(out: Path, summary: dict[str, Any], elapsed: float, queue_len: int) -> None:
    lines = [
        "# ANCHOR-MINI-007 Text Carrier Format Test",
        "",
        "## Verdict",
        "",
        "```text",
        summary["status"],
        "```",
        "",
        "## Run Summary",
        "",
        "```text",
        f"completed_jobs: {summary['completed_jobs']} / {queue_len}",
        f"valid_jobs: {summary['valid_jobs']}",
        f"valid_seed_count: {summary['valid_seed_count']}",
        f"blocked_jobs: {summary['blocked_jobs']}",
        f"elapsed_seconds: {elapsed:.2f}",
        "```",
        "",
        "## Conditions",
        "",
        "```text",
    ]
    for key, value in summary["conditions"].items():
        lines.append(f"{key}: {value}")
    lines.extend(["```", "", "## Aggregate Metrics", ""])
    lines.append("| model | format | OOD accuracy | trap rate | train accuracy | token mean | efficiency |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for item in sorted(summary["by_model_format"].values(), key=lambda row: (row["model"], row["format_arm"])):
        lines.append(
            f"| `{item['model']}` | `{item['format_arm']}` | "
            f"{item['answer_eval_ood_accuracy']:.3f} | "
            f"{item['shortcut_trap_rate']:.3f} | "
            f"{item['train_accuracy']:.3f} | "
            f"{item['format_token_count_mean']:.1f} | "
            f"{item['format_efficiency']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Useful Formats",
            "",
            "```text",
            "\n".join(summary["useful_formats"]) if summary["useful_formats"] else "none",
            "```",
            "",
            "## Claim Boundary",
            "",
            "MINI-007 is a deterministic toy text-carrier test. It does not prove LLM",
            "fine-tuning performance, natural-language AnchorCells, or symbol grounding.",
            "It only tests whether serialized process text can act as training-time",
            "privileged supervision while eval remains task-only.",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_result_doc(root: Path, summary: dict[str, Any], command: str, elapsed: float) -> None:
    path = root / "docs" / "research" / "ANCHOR_MINI_007_RESULT.md"
    lines = [
        "# ANCHOR-MINI-007 Result",
        "",
        "## Verdict",
        "",
        "```text",
        summary["status"],
        "```",
        "",
        "ANCHOR-MINI-007 tested literal serialized process text as training-time",
        "privileged information on the MINI-006 shortcut-flip task. Eval inputs hid",
        "the process section and contained only the task text.",
        "",
        "## Run",
        "",
        "```bash",
        command,
        "```",
        "",
        "Runtime:",
        "",
        "```text",
        f"{elapsed:.2f} seconds",
        "```",
        "",
        "## Summary",
        "",
        "```text",
        f"completed_jobs: {summary['completed_jobs']}",
        f"valid_jobs: {summary['valid_jobs']}",
        f"valid_seed_count: {summary['valid_seed_count']}",
        f"blocked_jobs: {summary['blocked_jobs']}",
        "```",
        "",
        "| model | format | OOD accuracy | trap rate | train accuracy | token mean | efficiency |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(summary["by_model_format"].values(), key=lambda row: (row["model"], row["format_arm"])):
        lines.append(
            f"| `{item['model']}` | `{item['format_arm']}` | "
            f"{item['answer_eval_ood_accuracy']:.3f} | "
            f"{item['shortcut_trap_rate']:.3f} | "
            f"{item['train_accuracy']:.3f} | "
            f"{item['format_token_count_mean']:.1f} | "
            f"{item['format_efficiency']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This result should be read through the eval-masking boundary: process text",
            "was available only as training-time privileged information, while eval",
            "remained task-only. A positive result means the carrier internalized some",
            "benefit from the serialized process text instead of requiring the process",
            "text as an oracle eval input.",
            "",
            "## Claim Boundary",
            "",
            "This is still a toy text-carrier result. It does not prove full AnchorCell",
            "grounding, Qwen/VRAXION fine-tuning behavior, or natural-language cell",
            "quality. It gates whether scaling to a small human AnchorCell family is",
            "reasonable.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    out = args.out if args.out.is_absolute() else root / args.out
    out.mkdir(parents=True, exist_ok=True)
    requested_seeds = parse_seed_spec(args.seeds)
    queue = build_queue(args)
    write_json(out / "queue.json", queue)
    if CONTRACT_FILE.exists():
        shutil.copyfile(root / CONTRACT_FILE, out / "contract_snapshot.md")

    metrics_path = out / "metrics.jsonl"
    progress_path = out / "progress.jsonl"
    completed_ids = load_completed(metrics_path)
    pending = [job for job in queue if job["job_id"] not in completed_ids]

    budget_seconds = None
    if args.budget_hours is not None:
        budget_seconds = args.budget_hours * 3600
    if args.budget_minutes is not None:
        budget_seconds = args.budget_minutes * 60

    started = time.time()
    append_jsonl(
        progress_path,
        {
            "event": "run_start",
            "time": started,
            "pending_jobs": len(pending),
            "completed_existing": len(completed_ids),
            "args": vars(args) | {"out": str(args.out)},
        },
    )
    pending_iter = iter(pending)
    futures: dict[Any, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        while len(futures) < max(1, args.jobs):
            try:
                job = next(pending_iter)
            except StopIteration:
                break
            append_jsonl(progress_path, {"event": "job_start", "job_id": job["job_id"], "time": time.time()})
            futures[executor.submit(train_and_eval, job)] = job
        while futures:
            done, _not_done = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    row = future.result()
                except Exception as exc:  # pragma: no cover
                    row = {
                        "event": "job_complete",
                        "job_id": job["job_id"],
                        "job": job,
                        "status": STATUS_BLOCKED,
                        "error": repr(exc),
                        "time": time.time(),
                    }
                append_jsonl(metrics_path, row)
                append_jsonl(
                    progress_path,
                    {"event": "job_done", "job_id": job["job_id"], "status": row["status"], "time": time.time()},
                )
                if budget_seconds is not None and time.time() - started >= budget_seconds:
                    continue
                try:
                    next_job = next(pending_iter)
                except StopIteration:
                    continue
                append_jsonl(
                    progress_path, {"event": "job_start", "job_id": next_job["job_id"], "time": time.time()}
                )
                futures[executor.submit(train_and_eval, next_job)] = next_job

    rows = []
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elapsed = time.time() - started
    summary = aggregate(rows, requested_seeds)
    summary["elapsed_seconds"] = elapsed
    summary["queue_jobs"] = len(queue)
    write_json(out / "summary.json", summary)
    write_json(
        out / "format_curve.json",
        sorted(summary["by_model_format"].values(), key=lambda row: (row["model"], row["format_arm"])),
    )
    write_report(out, summary, elapsed, len(queue))
    append_jsonl(progress_path, {"event": "run_done", "status": summary["status"], "time": time.time()})
    if summary["status"] in {STATUS_STRONG, STATUS_WEAK}:
        command = "python " + " ".join([str(Path(__file__).as_posix()), *argv])
        write_result_doc(root, summary, command, elapsed)
    print(summary["status"])
    print(f"report: {out / 'report.md'}")
    return 0 if summary["status"] != STATUS_BLOCKED else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
