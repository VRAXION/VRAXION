#!/usr/bin/env python3
"""Run ANCHOR-MINI-002 training-view toy A/B.

This probe keeps inference input identical across arms. The only difference is
the training view: answer-only supervision versus answer + decomposed
relational auxiliary supervision.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any


STATUS_POSITIVE = "ANCHOR_MINI_002_POSITIVE"
STATUS_NEGATIVE = "ANCHOR_MINI_002_NEGATIVE"
STATUS_BLOCKED = "ANCHOR_MINI_002_RESOURCE_BLOCKED"
CONTRACT_FILE = Path("docs/research/ANCHOR_MINI_002_CONTRACT.md")
ARMS = ["ANSWER_ONLY", "ANCHOR_MULTI_TASK", "SHUFFLED_ANCHOR_MULTI_TASK"]
MODELS = ["tiny_mlp"]
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


class MultiTaskNet:
    def __init__(self, model_name: str, input_dim: int, hidden: int):
        import torch

        if model_name == "tiny_mlp":
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.Tanh(),
            )
            trunk_dim = hidden
        else:
            raise RuntimeError(f"Unknown model: {model_name}")
        self.goal_category_head = torch.nn.Linear(trunk_dim, CATEGORY_COUNT)
        self.effect_category_heads = torch.nn.ModuleList(
            [torch.nn.Linear(trunk_dim, CATEGORY_COUNT) for _ in range(CANDIDATE_COUNT)]
        )
        self.match_head = torch.nn.Linear(trunk_dim, CANDIDATE_COUNT)

    def parameters(self) -> Any:
        return (
            list(self.trunk.parameters())
            + list(self.goal_category_head.parameters())
            + list(self.effect_category_heads.parameters())
            + list(self.match_head.parameters())
        )

    def train(self) -> None:
        self.trunk.train()
        self.goal_category_head.train()
        self.effect_category_heads.train()
        self.match_head.train()

    def eval(self) -> None:
        self.trunk.eval()
        self.goal_category_head.eval()
        self.effect_category_heads.eval()
        self.match_head.eval()

    def __call__(self, x: Any) -> dict[str, Any]:
        h = self.trunk(x)
        match_logits = self.match_head(h)
        return {
            "answer": match_logits,
            "goal_category": self.goal_category_head(h),
            "effect_categories": [head(h) for head in self.effect_category_heads],
            "match": match_logits,
        }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-002 training-view A/B.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--eval-examples", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--aux-loss-weight", type=float, default=1.5)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def one_hot(index: int, size: int) -> list[float]:
    row = [0.0] * size
    row[index] = 1.0
    return row


def goal_category(goal_id: int) -> int:
    return goal_id % CATEGORY_COUNT


def effect_category(effect_id: int) -> int:
    return effect_id % CATEGORY_COUNT


def make_examples(
    *,
    split: str,
    count: int,
    rng: random.Random,
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

        surface = [rng.random() for _ in range(CANDIDATE_COUNT)]
        if rng.random() < 0.72:
            wrong_slots = [slot for slot in range(CANDIDATE_COUNT) if slot != answer_slot]
            lure = rng.choice(wrong_slots)
            surface[lure] = max(surface[lure], 0.78 + 0.22 * rng.random())
        rows.append(
            Example(
                example_id=f"{split}_{index:05d}",
                split=split,
                goal_id=goal_id,
                goal_category=goal_cat,
                effect_ids=effect_ids,
                effect_categories=effect_cats,
                surface_priors=surface,
                answer_label=answer_slot,
                match_bits=match_bits,
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
    model_name: str,
    train_x: Any,
    train_targets: dict[str, Any],
    eval_x: Any,
    eval_targets: dict[str, Any],
    seed: int,
    epochs: int,
    lr: float,
    hidden: int,
    aux_weight: float,
) -> tuple[MultiTaskNet, dict[str, float]]:
    import torch

    torch.manual_seed(seed)
    model = MultiTaskNet(model_name, int(train_x.shape[1]), hidden)
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
        shortcut = max(range(CANDIDATE_COUNT), key=lambda slot: example.surface_priors[slot])
        if shortcut != example.answer_label:
            opportunities += 1
            traps += int(pred == shortcut)
    return traps / opportunities if opportunities else 0.0


def run_arm_model(
    *,
    arm: str,
    model_name: str,
    train_examples: list[Example],
    eval_examples: list[Example],
    seed: int,
    epochs: int,
    lr: float,
    hidden: int,
    aux_weight: float,
) -> dict[str, Any]:
    rng = random.Random(seed + ARM_OFFSETS[arm] + MODEL_OFFSETS[model_name])
    train_x, train_targets = tensorize(train_examples, arm=arm, rng=rng)
    eval_x, eval_targets = tensorize(eval_examples, arm=arm, rng=rng)
    model, losses = train_model(
        arm=arm,
        model_name=model_name,
        train_x=train_x,
        train_targets=train_targets,
        eval_x=eval_x,
        eval_targets=eval_targets,
        seed=seed,
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        aux_weight=aux_weight,
    )
    train_preds = predict(model, train_x)
    eval_preds = predict(model, eval_x)
    train_answer_labels = [example.answer_label for example in train_examples]
    eval_answer_labels = [example.answer_label for example in eval_examples]
    eval_goal_labels = [example.goal_category for example in eval_examples]
    eval_match_labels = [example.match_bits for example in eval_examples]
    return {
        "arm": arm,
        "model": model_name,
        "feature_dim": int(train_x.shape[1]),
        "answer_train_accuracy": accuracy(train_preds["answer"], train_answer_labels),
        "answer_eval_accuracy": accuracy(eval_preds["answer"], eval_answer_labels),
        "goal_category_eval_accuracy": accuracy(eval_preds["goal_category"], eval_goal_labels),
        "effect_category_eval_accuracy": effect_category_accuracy(eval_preds["effect_categories"], eval_examples),
        "match_eval_accuracy": bit_accuracy(eval_preds["match"], eval_match_labels),
        "match_positive_eval_accuracy": positive_bit_accuracy(eval_preds["match"], eval_match_labels),
        "match_exact_row_eval_accuracy": exact_row_accuracy(eval_preds["match"], eval_match_labels),
        "shortcut_trap_rate": shortcut_trap_rate(eval_preds["answer"], eval_examples),
        **losses,
    }


def model_family_positive(rows: list[dict[str, Any]], model_name: str) -> tuple[bool, dict[str, bool]]:
    by_arm = {row["arm"]: row for row in rows if row["model"] == model_name}
    base = by_arm["ANSWER_ONLY"]
    anchor = by_arm["ANCHOR_MULTI_TASK"]
    shuffled = by_arm["SHUFFLED_ANCHOR_MULTI_TASK"]
    conditions = {
        "anchor_answer_eval_gte_base_plus_0_15": anchor["answer_eval_accuracy"] >= base["answer_eval_accuracy"] + 0.15,
        "anchor_answer_eval_gte_shuffled_plus_0_20": anchor["answer_eval_accuracy"] >= shuffled["answer_eval_accuracy"] + 0.20,
        "anchor_goal_category_eval_gte_0_90": anchor["goal_category_eval_accuracy"] >= 0.90,
        "anchor_effect_category_eval_gte_0_90": anchor["effect_category_eval_accuracy"] >= 0.90,
        "anchor_shortcut_trap_lte_base": anchor["shortcut_trap_rate"] <= base["shortcut_trap_rate"],
        "shuffled_goal_category_eval_lte_0_50": shuffled["goal_category_eval_accuracy"] <= 0.50,
        "shuffled_effect_category_eval_lte_0_50": shuffled["effect_category_eval_accuracy"] <= 0.50,
    }
    return all(conditions.values()), conditions


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ANCHOR-MINI-002 Report",
        "",
        f"Status: `{report['status']}`",
        f"Seed: `{report['seed']}`",
        "",
        "## Metrics",
        "",
        "| model | arm | ans_train | ans_eval | match_bit | match_pos | match_exact | goal_eval | effect_eval | shortcut_trap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["metrics"]:
        lines.append(
            f"| `{row['model']}` | `{row['arm']}` | {row['answer_train_accuracy']:.3f} | "
            f"{row['answer_eval_accuracy']:.3f} | {row['match_eval_accuracy']:.3f} | "
            f"{row['match_positive_eval_accuracy']:.3f} | {row['match_exact_row_eval_accuracy']:.3f} | "
            f"{row['goal_category_eval_accuracy']:.3f} | {row['effect_category_eval_accuracy']:.3f} | "
            f"{row['shortcut_trap_rate']:.3f} |"
        )
    lines.extend(["", "## Verdict Conditions", ""])
    for model_name, conditions in report["verdict_conditions"].items():
        lines.append(f"### `{model_name}`")
        for key, value in conditions.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError as exc:
        return write_resource_blocked(args, f"torch import failed: {exc}")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train_examples = make_examples(split="train", count=args.train_examples, rng=rng)
    eval_examples = make_examples(split="eval", count=args.eval_examples, rng=rng)

    args.out.mkdir(parents=True, exist_ok=True)
    contract = repo_root() / CONTRACT_FILE
    if contract.exists():
        shutil.copyfile(contract, args.out / "contract_snapshot.md")

    metrics: list[dict[str, Any]] = []
    for model_name in MODELS:
        for arm in ARMS:
            metrics.append(
                run_arm_model(
                    arm=arm,
                    model_name=model_name,
                    train_examples=train_examples,
                    eval_examples=eval_examples,
                    seed=args.seed,
                    epochs=args.epochs,
                    lr=args.lr,
                    hidden=args.hidden,
                    aux_weight=args.aux_loss_weight,
                )
            )

    verdict_conditions: dict[str, dict[str, bool]] = {}
    positives = []
    for model_name in MODELS:
        positive, conditions = model_family_positive(metrics, model_name)
        verdict_conditions[model_name] = conditions
        positives.append(positive)
    status = STATUS_POSITIVE if all(positives) else STATUS_NEGATIVE

    metadata = {
        "seed": args.seed,
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
        "eval_input_identical": True,
        "gold_rule": "choose candidate whose effect category matches goal category",
    }
    report = {
        "status": status,
        "seed": args.seed,
        "metadata": metadata,
        "metrics": metrics,
        "verdict_conditions": verdict_conditions,
    }
    write_json(args.out / "dataset_metadata.json", metadata)
    write_jsonl(args.out / "examples_sample.jsonl", [asdict(row) for row in train_examples[:20] + eval_examples[:20]])
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
    except (RuntimeError, OSError) as exc:
        return write_resource_blocked(args, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
