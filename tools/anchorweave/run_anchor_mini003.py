#!/usr/bin/env python3
"""Run ANCHOR-MINI-003 surface-biased training-view stress test."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
import statistics
import sys
from typing import Any


STATUS_STRONG = "ANCHOR_MINI_003_STRONG_POSITIVE"
STATUS_WEAK = "ANCHOR_MINI_003_WEAK_POSITIVE"
STATUS_NEGATIVE = "ANCHOR_MINI_003_NEGATIVE"
STATUS_INVALID = "ANCHOR_MINI_003_INVALID_STRESS"
STATUS_BLOCKED = "ANCHOR_MINI_003_RESOURCE_BLOCKED"
CONTRACT_FILE = Path("docs/research/ANCHOR_MINI_003_CONTRACT.md")
ARMS = ["ANSWER_ONLY", "ANCHOR_MULTI_TASK", "SHUFFLED_ANCHOR_MULTI_TASK"]
MODELS = ["tiny_mlp"]
ANSWER_HEAD_MODES = ["compatibility", "direct_match", "hybrid"]
ARM_OFFSETS = {
    "ANSWER_ONLY": 101,
    "ANCHOR_MULTI_TASK": 211,
    "SHUFFLED_ANCHOR_MULTI_TASK": 307,
}
MODEL_OFFSETS = {"tiny_mlp": 2003}
GOAL_COUNT = 16
EFFECT_COUNT = 24
CATEGORY_COUNT = 4
CANDIDATE_COUNT = 4
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030]


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
    match_bits: list[int]
    surface_shortcut_label: int
    surface_shortcut_is_gold: bool


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-003 toy stress A/B.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-examples", type=int, default=1024)
    parser.add_argument("--eval-examples", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--aux-loss-weight", type=float, default=2.0)
    parser.add_argument("--train-surface-gold-prob", type=float, default=0.90)
    parser.add_argument("--eval-surface-wrong-prob", type=float, default=0.90)
    parser.add_argument(
        "--answer-head-mode",
        choices=ANSWER_HEAD_MODES,
        default="compatibility",
        help="compatibility preserves the validated MINI-003 default; direct_match and hybrid are carrier diagnostics",
    )
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one integer")
    return seeds


def one_hot(index: int, size: int) -> list[float]:
    row = [0.0] * size
    row[index] = 1.0
    return row


def goal_category(goal_id: int) -> int:
    return goal_id % CATEGORY_COUNT


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
    raise RuntimeError(f"Unknown split: {split}")


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
    rows: list[Example] = []
    effects_by_category = {
        category: [effect for effect in range(EFFECT_COUNT) if effect_category(effect) == category]
        for category in range(CATEGORY_COUNT)
    }
    effects_not_category = {
        category: [effect for effect in range(EFFECT_COUNT) if effect_category(effect) != category]
        for category in range(CATEGORY_COUNT)
    }
    for index in range(count):
        goal_id = rng.randrange(GOAL_COUNT)
        goal_cat = goal_category(goal_id)
        answer_slot = rng.randrange(CANDIDATE_COUNT)
        correct_effect = rng.choice(effects_by_category[goal_cat])
        distractors = rng.sample(effects_not_category[goal_cat], CANDIDATE_COUNT - 1)
        effect_ids: list[int] = []
        distractor_index = 0
        for slot in range(CANDIDATE_COUNT):
            if slot == answer_slot:
                effect_ids.append(correct_effect)
            else:
                effect_ids.append(distractors[distractor_index])
                distractor_index += 1
        effect_cats = [effect_category(effect) for effect in effect_ids]
        match_bits = [int(category == goal_cat) for category in effect_cats]
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
                goal_category=goal_cat,
                effect_ids=effect_ids,
                effect_categories=effect_cats,
                surface_priors=make_surface_priors(shortcut, rng),
                answer_label=answer_slot,
                match_bits=match_bits,
                surface_shortcut_label=shortcut,
                surface_shortcut_is_gold=shortcut == answer_slot,
            )
        )
    return rows


def features(example: Example) -> list[float]:
    row = one_hot(example.goal_id, GOAL_COUNT)
    for effect_id in example.effect_ids:
        row.extend(one_hot(effect_id, EFFECT_COUNT))
    row.extend(example.surface_priors)
    row.append(max(example.surface_priors))
    row.append(min(example.surface_priors))
    return row


def shuffled_aux_targets(examples: list[Example], rng: random.Random) -> list[tuple[int, list[int], list[int]]]:
    targets = [
        (example.goal_category, list(example.effect_categories), list(example.match_bits))
        for example in examples
    ]
    rng.shuffle(targets)
    return targets


def tensorize(
    examples: list[Example],
    *,
    arm: str,
    rng: random.Random,
) -> tuple[Any, dict[str, Any]]:
    import torch

    x = torch.tensor([features(example) for example in examples], dtype=torch.float32)
    answer = torch.tensor([example.answer_label for example in examples], dtype=torch.long)
    if arm == "SHUFFLED_ANCHOR_MULTI_TASK":
        aux = shuffled_aux_targets(examples, rng)
        goal_cat = torch.tensor([item[0] for item in aux], dtype=torch.long)
        effect_cats = [
            torch.tensor([item[1][slot] for item in aux], dtype=torch.long)
            for slot in range(CANDIDATE_COUNT)
        ]
        match = torch.tensor([item[2] for item in aux], dtype=torch.float32)
    else:
        goal_cat = torch.tensor([example.goal_category for example in examples], dtype=torch.long)
        effect_cats = [
            torch.tensor([example.effect_categories[slot] for example in examples], dtype=torch.long)
            for slot in range(CANDIDATE_COUNT)
        ]
        match = torch.tensor([example.match_bits for example in examples], dtype=torch.float32)
    return x, {
        "answer": answer,
        "goal_category": goal_cat,
        "effect_categories": effect_cats,
        "match": match,
    }


class MultiTaskNet:
    def __init__(self, input_dim: int, hidden: int, answer_head_mode: str):
        import torch

        if answer_head_mode not in ANSWER_HEAD_MODES:
            raise RuntimeError(f"Unknown answer head mode: {answer_head_mode}")
        self.answer_head_mode = answer_head_mode
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.Tanh(),
        )
        self.goal_category_head = torch.nn.Linear(hidden, CATEGORY_COUNT)
        self.effect_category_heads = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, CATEGORY_COUNT) for _ in range(CANDIDATE_COUNT)]
        )
        self.direct_match_head = (
            torch.nn.Linear(hidden, CANDIDATE_COUNT) if answer_head_mode in {"direct_match", "hybrid"} else None
        )

    def parameters(self) -> Any:
        params = (
            list(self.trunk.parameters())
            + list(self.goal_category_head.parameters())
            + list(self.effect_category_heads.parameters())
        )
        if self.direct_match_head is not None:
            params += list(self.direct_match_head.parameters())
        return params

    def train(self) -> None:
        self.trunk.train()
        self.goal_category_head.train()
        self.effect_category_heads.train()
        if self.direct_match_head is not None:
            self.direct_match_head.train()

    def eval(self) -> None:
        self.trunk.eval()
        self.goal_category_head.eval()
        self.effect_category_heads.eval()
        if self.direct_match_head is not None:
            self.direct_match_head.eval()

    def __call__(self, x: Any) -> dict[str, Any]:
        h = self.trunk(x)
        goal_logits = self.goal_category_head(h)
        effect_logits = [head(h) for head in self.effect_category_heads]
        compatibility_logits = torch_stack(
            [(goal_logits * logits).sum(dim=1) / (CATEGORY_COUNT**0.5) for logits in effect_logits]
        )
        if self.answer_head_mode == "compatibility":
            match_logits = compatibility_logits
        elif self.answer_head_mode == "direct_match":
            if self.direct_match_head is None:
                raise RuntimeError("direct_match_head missing")
            match_logits = self.direct_match_head(h)
        elif self.answer_head_mode == "hybrid":
            if self.direct_match_head is None:
                raise RuntimeError("direct_match_head missing")
            match_logits = compatibility_logits + self.direct_match_head(h)
        else:
            raise RuntimeError(f"Unknown answer head mode: {self.answer_head_mode}")
        return {
            "answer": match_logits,
            "goal_category": goal_logits,
            "effect_categories": effect_logits,
            "match": match_logits,
        }


def torch_stack(rows: list[Any]) -> Any:
    import torch

    return torch.stack(rows, dim=1)


def loss_for_outputs(outputs: dict[str, Any], targets: dict[str, Any], arm: str, aux_weight: float) -> Any:
    import torch

    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    answer_loss = ce(outputs["answer"], targets["answer"])
    if arm == "ANSWER_ONLY":
        return answer_loss
    effect_loss = sum(
        ce(outputs["effect_categories"][slot], targets["effect_categories"][slot])
        for slot in range(CANDIDATE_COUNT)
    ) / CANDIDATE_COUNT
    aux_loss = (
        ce(outputs["goal_category"], targets["goal_category"])
        + effect_loss
        + bce(outputs["match"], targets["match"])
    ) / 3.0
    return answer_loss + aux_weight * aux_loss


def train_model(
    *,
    arm: str,
    train_x: Any,
    train_targets: dict[str, Any],
    eval_x: Any,
    eval_targets: dict[str, Any],
    seed: int,
    epochs: int,
    lr: float,
    hidden: int,
    aux_weight: float,
    answer_head_mode: str,
) -> tuple[MultiTaskNet, dict[str, float]]:
    import torch

    torch.manual_seed(seed)
    model = MultiTaskNet(int(train_x.shape[1]), hidden, answer_head_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for_outputs(model(train_x), train_targets, arm, aux_weight)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_loss = float(loss_for_outputs(model(train_x), train_targets, arm, aux_weight).item())
        eval_loss = float(loss_for_outputs(model(eval_x), eval_targets, arm, aux_weight).item())
    return model, {"train_loss": train_loss, "eval_loss": eval_loss}


def predict(model: MultiTaskNet, x: Any) -> dict[str, Any]:
    import torch

    model.eval()
    with torch.no_grad():
        outputs = model(x)
        return {
            "answer": torch.argmax(outputs["answer"], dim=1).cpu().tolist(),
            "goal_category": torch.argmax(outputs["goal_category"], dim=1).cpu().tolist(),
            "effect_categories": [
                torch.argmax(outputs["effect_categories"][slot], dim=1).cpu().tolist()
                for slot in range(CANDIDATE_COUNT)
            ],
            "match": (torch.sigmoid(outputs["match"]) >= 0.5).int().cpu().tolist(),
        }


def accuracy(preds: list[int], labels: list[int]) -> float:
    return sum(int(pred == label) for pred, label in zip(preds, labels)) / len(labels)


def bit_accuracy(preds: list[list[int]], labels: list[list[int]]) -> float:
    total = 0
    correct = 0
    for pred_row, label_row in zip(preds, labels):
        for pred, label in zip(pred_row, label_row):
            total += 1
            correct += int(int(pred) == int(label))
    return correct / total


def positive_bit_accuracy(preds: list[list[int]], labels: list[list[int]]) -> float:
    total = 0
    correct = 0
    for pred_row, label_row in zip(preds, labels):
        for pred, label in zip(pred_row, label_row):
            if int(label) == 1:
                total += 1
                correct += int(int(pred) == 1)
    return correct / total if total else 0.0


def exact_row_accuracy(preds: list[list[int]], labels: list[list[int]]) -> float:
    return sum(int([int(value) for value in pred] == [int(value) for value in label]) for pred, label in zip(preds, labels)) / len(labels)


def effect_category_accuracy(preds: list[list[int]], examples: list[Example]) -> float:
    total = 0
    correct = 0
    for slot, slot_preds in enumerate(preds):
        for pred, example in zip(slot_preds, examples):
            total += 1
            correct += int(pred == example.effect_categories[slot])
    return correct / total


def shortcut_trap_rate(preds: list[int], examples: list[Example]) -> float:
    traps = 0
    opportunities = 0
    for pred, example in zip(preds, examples):
        shortcut = example.surface_shortcut_label
        if shortcut != example.answer_label:
            opportunities += 1
            traps += int(pred == shortcut)
    return traps / opportunities if opportunities else 0.0


def surface_shortcut_alignment(examples: list[Example]) -> float:
    return sum(int(example.surface_shortcut_is_gold) for example in examples) / len(examples)


def surface_shortcut_flip_rate(examples: list[Example]) -> float:
    return 1.0 - surface_shortcut_alignment(examples)


def run_arm_model(
    *,
    arm: str,
    train_examples: list[Example],
    eval_examples: list[Example],
    seed: int,
    epochs: int,
    lr: float,
    hidden: int,
    aux_weight: float,
    answer_head_mode: str,
) -> dict[str, Any]:
    rng = random.Random(seed + ARM_OFFSETS[arm] + MODEL_OFFSETS["tiny_mlp"])
    train_x, train_targets = tensorize(train_examples, arm=arm, rng=rng)
    eval_x, eval_targets = tensorize(eval_examples, arm=arm, rng=rng)
    model, losses = train_model(
        arm=arm,
        train_x=train_x,
        train_targets=train_targets,
        eval_x=eval_x,
        eval_targets=eval_targets,
        seed=seed,
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        aux_weight=aux_weight,
        answer_head_mode=answer_head_mode,
    )
    train_preds = predict(model, train_x)
    eval_preds = predict(model, eval_x)
    train_answer_labels = [example.answer_label for example in train_examples]
    eval_answer_labels = [example.answer_label for example in eval_examples]
    eval_goal_labels = [example.goal_category for example in eval_examples]
    eval_match_labels = [example.match_bits for example in eval_examples]
    return {
        "seed": seed,
        "arm": arm,
        "model": "tiny_mlp",
        "answer_head_mode": answer_head_mode,
        "feature_dim": int(train_x.shape[1]),
        "answer_train_accuracy": accuracy(train_preds["answer"], train_answer_labels),
        "answer_eval_ood_accuracy": accuracy(eval_preds["answer"], eval_answer_labels),
        "goal_category_eval_accuracy": accuracy(eval_preds["goal_category"], eval_goal_labels),
        "effect_category_eval_accuracy": effect_category_accuracy(eval_preds["effect_categories"], eval_examples),
        "match_bit_accuracy": bit_accuracy(eval_preds["match"], eval_match_labels),
        "match_positive_accuracy": positive_bit_accuracy(eval_preds["match"], eval_match_labels),
        "match_exact_row_accuracy": exact_row_accuracy(eval_preds["match"], eval_match_labels),
        "shortcut_trap_rate": shortcut_trap_rate(eval_preds["answer"], eval_examples),
        **losses,
    }


def seed_conditions(rows: list[dict[str, Any]], train_examples: list[Example], eval_examples: list[Example]) -> dict[str, bool]:
    by_arm = {row["arm"]: row for row in rows}
    base = by_arm["ANSWER_ONLY"]
    anchor = by_arm["ANCHOR_MULTI_TASK"]
    shuffled = by_arm["SHUFFLED_ANCHOR_MULTI_TASK"]
    return {
        "stress_answer_only_shortcut_trap_gte_0_45": base["shortcut_trap_rate"] >= 0.45,
        "stress_eval_surface_flip_gte_0_85": surface_shortcut_flip_rate(eval_examples) >= 0.85,
        "anchor_answer_eval_gte_base_plus_0_25": anchor["answer_eval_ood_accuracy"] >= base["answer_eval_ood_accuracy"] + 0.25,
        "anchor_answer_eval_gte_shuffled_plus_0_25": anchor["answer_eval_ood_accuracy"] >= shuffled["answer_eval_ood_accuracy"] + 0.25,
        "anchor_answer_eval_gte_0_65": anchor["answer_eval_ood_accuracy"] >= 0.65,
        "anchor_shortcut_trap_lte_0_25": anchor["shortcut_trap_rate"] <= 0.25,
        "anchor_goal_category_eval_gte_0_90": anchor["goal_category_eval_accuracy"] >= 0.90,
        "anchor_effect_category_eval_gte_0_90": anchor["effect_category_eval_accuracy"] >= 0.90,
        "shuffled_goal_category_eval_lte_0_50": shuffled["goal_category_eval_accuracy"] <= 0.50,
        "shuffled_effect_category_eval_lte_0_50": shuffled["effect_category_eval_accuracy"] <= 0.50,
        "main_effect_not_reversed_vs_base": anchor["answer_eval_ood_accuracy"] > base["answer_eval_ood_accuracy"],
        "main_effect_not_reversed_vs_shuffled": anchor["answer_eval_ood_accuracy"] > shuffled["answer_eval_ood_accuracy"],
    }


def seed_is_invalid(conditions: dict[str, bool]) -> bool:
    return not (
        conditions["stress_answer_only_shortcut_trap_gte_0_45"]
        and conditions["stress_eval_surface_flip_gte_0_85"]
    )


def seed_is_strong_positive(conditions: dict[str, bool]) -> bool:
    required = [
        "anchor_answer_eval_gte_base_plus_0_25",
        "anchor_answer_eval_gte_shuffled_plus_0_25",
        "anchor_answer_eval_gte_0_65",
        "anchor_shortcut_trap_lte_0_25",
        "anchor_goal_category_eval_gte_0_90",
        "anchor_effect_category_eval_gte_0_90",
        "shuffled_goal_category_eval_lte_0_50",
        "shuffled_effect_category_eval_lte_0_50",
    ]
    return all(conditions[key] for key in required)


def seed_has_reversal(conditions: dict[str, bool]) -> bool:
    return not (
        conditions["main_effect_not_reversed_vs_base"]
        and conditions["main_effect_not_reversed_vs_shuffled"]
    )


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def aggregate_by_arm(metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = [
        "answer_train_accuracy",
        "answer_eval_ood_accuracy",
        "goal_category_eval_accuracy",
        "effect_category_eval_accuracy",
        "match_bit_accuracy",
        "match_positive_accuracy",
        "match_exact_row_accuracy",
        "shortcut_trap_rate",
    ]
    out: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        rows = [row for row in metrics if row["arm"] == arm]
        out[arm] = {field: mean([float(row[field]) for row in rows]) for field in fields}
    return out


def determine_status(seed_reports: list[dict[str, Any]]) -> str:
    conditions = [report["conditions"] for report in seed_reports]
    if any(seed_is_invalid(item) for item in conditions):
        return STATUS_INVALID
    strong_count = sum(int(seed_is_strong_positive(item)) for item in conditions)
    if strong_count == len(conditions):
        return STATUS_STRONG
    if strong_count >= max(1, len(conditions) - 1) and not any(seed_has_reversal(item) for item in conditions):
        return STATUS_WEAK
    return STATUS_NEGATIVE


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ANCHOR-MINI-003 Report",
        "",
        f"Status: `{report['status']}`",
        f"Seeds: `{','.join(str(seed) for seed in report['seeds'])}`",
        "",
        "## Aggregate Metrics",
        "",
        "| arm | ans_train | ans_eval_ood | goal_eval | effect_eval | match_bit | match_exact | shortcut_trap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = report["aggregate_metrics"][arm]
        lines.append(
            f"| `{arm}` | {row['answer_train_accuracy']:.3f} | {row['answer_eval_ood_accuracy']:.3f} | "
            f"{row['goal_category_eval_accuracy']:.3f} | {row['effect_category_eval_accuracy']:.3f} | "
            f"{row['match_bit_accuracy']:.3f} | {row['match_exact_row_accuracy']:.3f} | "
            f"{row['shortcut_trap_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Summary",
            "",
            "| seed | status | answer_only | anchor | shuffled | base_trap | anchor_trap | train_align | eval_flip |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed_report in report["seed_reports"]:
        by_arm = {row["arm"]: row for row in seed_report["metrics"]}
        lines.append(
            f"| {seed_report['seed']} | `{seed_report['seed_status']}` | "
            f"{by_arm['ANSWER_ONLY']['answer_eval_ood_accuracy']:.3f} | "
            f"{by_arm['ANCHOR_MULTI_TASK']['answer_eval_ood_accuracy']:.3f} | "
            f"{by_arm['SHUFFLED_ANCHOR_MULTI_TASK']['answer_eval_ood_accuracy']:.3f} | "
            f"{by_arm['ANSWER_ONLY']['shortcut_trap_rate']:.3f} | "
            f"{by_arm['ANCHOR_MULTI_TASK']['shortcut_trap_rate']:.3f} | "
            f"{seed_report['surface_shortcut_train_alignment']:.3f} | "
            f"{seed_report['surface_shortcut_eval_flip_rate']:.3f} |"
        )
    lines.extend(["", "## Claim Boundary", ""])
    lines.append(
        "This is deterministic toy-level evidence for decomposed anchor supervision "
        "under a flipped surface shortcut. It does not prove LLM, VRAXION, or "
        "natural-language AnchorCell behavior."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_examples = make_examples(
        split="train",
        count=args.train_examples,
        rng=rng,
        train_surface_gold_prob=args.train_surface_gold_prob,
        eval_surface_wrong_prob=args.eval_surface_wrong_prob,
    )
    eval_examples = make_examples(
        split="eval",
        count=args.eval_examples,
        rng=rng,
        train_surface_gold_prob=args.train_surface_gold_prob,
        eval_surface_wrong_prob=args.eval_surface_wrong_prob,
    )
    metrics: list[dict[str, Any]] = []
    for arm in ARMS:
        metrics.append(
            run_arm_model(
                arm=arm,
                train_examples=train_examples,
                eval_examples=eval_examples,
                seed=seed,
                epochs=args.epochs,
                lr=args.lr,
                hidden=args.hidden,
                aux_weight=args.aux_loss_weight,
                answer_head_mode=args.answer_head_mode,
            )
        )
    conditions = seed_conditions(metrics, train_examples, eval_examples)
    if seed_is_invalid(conditions):
        seed_status = STATUS_INVALID
    elif seed_is_strong_positive(conditions):
        seed_status = STATUS_STRONG
    else:
        seed_status = STATUS_NEGATIVE
    return {
        "seed": seed,
        "seed_status": seed_status,
        "surface_shortcut_train_alignment": surface_shortcut_alignment(train_examples),
        "surface_shortcut_eval_flip_rate": surface_shortcut_flip_rate(eval_examples),
        "metrics": metrics,
        "conditions": conditions,
        "sample_examples": [asdict(row) for row in train_examples[:5] + eval_examples[:5]],
    }


def run(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError as exc:
        return write_resource_blocked(args, f"torch import failed: {exc}")

    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    args.out.mkdir(parents=True, exist_ok=True)
    contract = repo_root() / CONTRACT_FILE
    if contract.exists():
        shutil.copyfile(contract, args.out / "contract_snapshot.md")

    seeds = parse_seeds(args.seeds)
    seed_reports = [run_seed(args, seed) for seed in seeds]
    all_metrics = [row for seed_report in seed_reports for row in seed_report["metrics"]]
    status = determine_status(seed_reports)
    metadata = {
        "seeds": seeds,
        "train_examples": args.train_examples,
        "eval_examples": args.eval_examples,
        "goal_count": GOAL_COUNT,
        "effect_count": EFFECT_COUNT,
        "category_count": CATEGORY_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "arms": ARMS,
        "models": MODELS,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden": args.hidden,
        "aux_loss_weight": args.aux_loss_weight,
        "answer_head_mode": args.answer_head_mode,
        "train_surface_gold_prob": args.train_surface_gold_prob,
        "eval_surface_wrong_prob": args.eval_surface_wrong_prob,
        "eval_input_identical": True,
        "gold_rule": "choose candidate whose effect category matches goal category",
        "surface_stress": "train shortcut points to gold; eval shortcut points to wrong candidate",
    }
    report = {
        "status": status,
        "seeds": seeds,
        "metadata": metadata,
        "aggregate_metrics": aggregate_by_arm(all_metrics),
        "seed_reports": seed_reports,
    }
    write_json(args.out / "dataset_metadata.json", metadata)
    write_jsonl(args.out / "metrics_by_seed.jsonl", all_metrics)
    write_jsonl(
        args.out / "examples_sample.jsonl",
        [example for seed_report in seed_reports for example in seed_report["sample_examples"]],
    )
    write_json(args.out / "report.json", report)
    write_report_md(args.out / "report.md", report)
    print(f"wrote {args.out}")
    print(f"status: {status}")
    return 0


def write_resource_blocked(args: argparse.Namespace, message: str) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"status": STATUS_BLOCKED, "error": message}
    write_json(args.out / "report.json", report)
    print(f"resource blocked: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        return run(args)
    except (RuntimeError, OSError, ValueError) as exc:
        return write_resource_blocked(args, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
