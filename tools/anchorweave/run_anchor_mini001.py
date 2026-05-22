#!/usr/bin/env python3
"""Run ANCHOR-MINI-001 deterministic relational-anchor toy probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any


STATUS_POSITIVE = "ANCHOR_MINI_001_POSITIVE"
STATUS_NEGATIVE = "ANCHOR_MINI_001_NEGATIVE"
STATUS_BLOCKED = "ANCHOR_MINI_001_RESOURCE_BLOCKED"
CONTRACT_FILE = Path("docs/research/ANCHOR_MINI_001_CONTRACT.md")
ARMS = ["BASE", "ANCHOR", "SHUFFLED_ANCHOR", "NOISE_ANCHOR"]
MODELS = ["logistic_regression", "tiny_mlp"]
ARM_OFFSETS = {
    "BASE": 101,
    "ANCHOR": 211,
    "SHUFFLED_ANCHOR": 307,
    "NOISE_ANCHOR": 401,
}
MODEL_OFFSETS = {
    "logistic_regression": 1009,
    "tiny_mlp": 2003,
}


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str
    object_id: int
    site_a_id: int
    site_b_id: int
    intent_next_step: int
    candidate_a_is_use: int
    candidate_b_is_use: int
    surface_prior_a: float
    surface_prior_b: float
    label: int


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANCHOR-MINI-001 toy core probe.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-examples", type=int, default=800)
    parser.add_argument("--eval-examples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=32)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def make_examples(
    *,
    split: str,
    count: int,
    rng: random.Random,
    object_pool: range,
    site_pool: range,
) -> list[Example]:
    rows: list[Example] = []
    for index in range(count):
        object_id = rng.choice(list(object_pool))
        site_a_id, site_b_id = rng.sample(list(site_pool), 2)
        intent_next_step = rng.randrange(2)
        a_is_use = rng.randrange(2)
        b_is_use = 1 - a_is_use
        # Noisy surface prior: it is intentionally not the rule. It gives the
        # base learner a tempting but unreliable shortcut.
        surface_a = rng.random()
        surface_b = rng.random()
        if rng.random() < 0.58:
            if a_is_use:
                surface_b = max(surface_b, 0.65 + 0.35 * rng.random())
            else:
                surface_a = max(surface_a, 0.65 + 0.35 * rng.random())
        if intent_next_step:
            label = 0 if a_is_use else 1
        else:
            label = 0 if not a_is_use else 1
        rows.append(
            Example(
                example_id=f"{split}_{index:05d}",
                split=split,
                object_id=object_id,
                site_a_id=site_a_id,
                site_b_id=site_b_id,
                intent_next_step=intent_next_step,
                candidate_a_is_use=a_is_use,
                candidate_b_is_use=b_is_use,
                surface_prior_a=surface_a,
                surface_prior_b=surface_b,
                label=label,
            )
        )
    return rows


def anchor_match_bits(example: Example) -> tuple[float, float]:
    if example.intent_next_step:
        return float(example.candidate_a_is_use), float(example.candidate_b_is_use)
    return float(1 - example.candidate_a_is_use), float(1 - example.candidate_b_is_use)


def base_features(example: Example, max_object_id: int, max_site_id: int) -> list[float]:
    object_scale = max(1, max_object_id)
    site_scale = max(1, max_site_id)
    return [
        example.object_id / object_scale,
        example.site_a_id / site_scale,
        example.site_b_id / site_scale,
        float(example.intent_next_step),
        example.surface_prior_a,
        example.surface_prior_b,
        example.surface_prior_a - example.surface_prior_b,
    ]


def build_feature_rows(
    examples: list[Example],
    *,
    arm: str,
    max_object_id: int,
    max_site_id: int,
    anchor_source: list[Example] | None,
    rng: random.Random,
) -> list[list[float]]:
    shuffled_bits: list[tuple[float, float]] = []
    if arm == "SHUFFLED_ANCHOR":
        if anchor_source is None:
            raise RuntimeError("SHUFFLED_ANCHOR requires anchor_source")
        shuffled_bits = [anchor_match_bits(example) for example in anchor_source]
        rng.shuffle(shuffled_bits)

    rows: list[list[float]] = []
    for index, example in enumerate(examples):
        features = base_features(example, max_object_id, max_site_id)
        if arm == "BASE":
            pass
        elif arm == "ANCHOR":
            features.extend(anchor_match_bits(example))
        elif arm == "SHUFFLED_ANCHOR":
            features.extend(shuffled_bits[index % len(shuffled_bits)])
        elif arm == "NOISE_ANCHOR":
            features.extend([float(rng.randrange(2)), float(rng.randrange(2))])
        else:
            raise RuntimeError(f"Unknown arm: {arm}")
        return_width = 9 if arm != "BASE" else 7
        if len(features) != return_width:
            raise RuntimeError(f"{arm} feature width {len(features)} != {return_width}")
        rows.append(features)
    return rows


def tensorize(rows: list[list[float]], labels: list[int]) -> tuple[Any, Any]:
    import torch

    x = torch.tensor(rows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def make_model(model_name: str, input_dim: int, hidden: int) -> Any:
    import torch

    if model_name == "logistic_regression":
        return torch.nn.Linear(input_dim, 2)
    if model_name == "tiny_mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 2),
        )
    raise RuntimeError(f"Unknown model: {model_name}")


def train_model(
    *,
    model_name: str,
    train_x: Any,
    train_y: Any,
    eval_x: Any,
    eval_y: Any,
    seed: int,
    epochs: int,
    lr: float,
    hidden: int,
) -> Any:
    import torch

    torch.manual_seed(seed)
    model = make_model(model_name, int(train_x.shape[1]), hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_loss = float(loss_fn(model(train_x), train_y).item())
        eval_loss = float(loss_fn(model(eval_x), eval_y).item())
    return model, {"train_loss": train_loss, "eval_loss": eval_loss}


def predictions(model: Any, x: Any) -> list[int]:
    import torch

    with torch.no_grad():
        return torch.argmax(model(x), dim=1).cpu().tolist()


def accuracy(preds: list[int], labels: list[int]) -> float:
    return sum(int(pred == label) for pred, label in zip(preds, labels)) / len(labels)


def shortcut_trap_rate(preds: list[int], examples: list[Example]) -> float:
    traps = 0
    opportunities = 0
    for pred, example in zip(preds, examples):
        surface_choice = 0 if example.surface_prior_a >= example.surface_prior_b else 1
        if surface_choice != example.label:
            opportunities += 1
            traps += int(pred == surface_choice)
    return traps / opportunities if opportunities else 0.0


def split_accuracy(preds: list[int], examples: list[Example], intent_next_step: int) -> float:
    pairs = [(pred, example.label) for pred, example in zip(preds, examples) if example.intent_next_step == intent_next_step]
    if not pairs:
        return 0.0
    return sum(int(pred == label) for pred, label in pairs) / len(pairs)


def run_arm_model(
    *,
    arm: str,
    model_name: str,
    train_examples: list[Example],
    eval_examples: list[Example],
    max_object_id: int,
    max_site_id: int,
    seed: int,
    epochs: int,
    lr: float,
    hidden: int,
) -> dict[str, Any]:
    feature_rng = random.Random(seed + ARM_OFFSETS[arm] + MODEL_OFFSETS[model_name])
    train_rows = build_feature_rows(
        train_examples,
        arm=arm,
        max_object_id=max_object_id,
        max_site_id=max_site_id,
        anchor_source=train_examples,
        rng=feature_rng,
    )
    eval_rows = build_feature_rows(
        eval_examples,
        arm=arm,
        max_object_id=max_object_id,
        max_site_id=max_site_id,
        anchor_source=eval_examples,
        rng=feature_rng,
    )
    train_labels = [example.label for example in train_examples]
    eval_labels = [example.label for example in eval_examples]
    train_x, train_y = tensorize(train_rows, train_labels)
    eval_x, eval_y = tensorize(eval_rows, eval_labels)
    model, losses = train_model(
        model_name=model_name,
        train_x=train_x,
        train_y=train_y,
        eval_x=eval_x,
        eval_y=eval_y,
        seed=seed,
        epochs=epochs,
        lr=lr,
        hidden=hidden,
    )
    train_preds = predictions(model, train_x)
    eval_preds = predictions(model, eval_x)

    anchor_ablation_accuracy = None
    anchor_reliance_drop = None
    if arm == "ANCHOR":
        ablated_eval_rows = [row[:-2] + [0.0, 0.0] for row in eval_rows]
        ablated_eval_x, _ = tensorize(ablated_eval_rows, eval_labels)
        ablated_preds = predictions(model, ablated_eval_x)
        anchor_ablation_accuracy = accuracy(ablated_preds, eval_labels)
        anchor_reliance_drop = accuracy(eval_preds, eval_labels) - anchor_ablation_accuracy

    return {
        "arm": arm,
        "model": model_name,
        "feature_dim": len(train_rows[0]),
        "train_accuracy": accuracy(train_preds, train_labels),
        "eval_accuracy": accuracy(eval_preds, eval_labels),
        "next_step_accuracy": split_accuracy(eval_preds, eval_examples, 1),
        "put_away_accuracy": split_accuracy(eval_preds, eval_examples, 0),
        "shortcut_trap_rate": shortcut_trap_rate(eval_preds, eval_examples),
        "anchor_ablation_accuracy": anchor_ablation_accuracy,
        "anchor_reliance_drop": anchor_reliance_drop,
        **losses,
    }


def model_family_positive(rows: list[dict[str, Any]], model_name: str) -> tuple[bool, dict[str, bool]]:
    by_arm = {row["arm"]: row for row in rows if row["model"] == model_name}
    anchor = by_arm["ANCHOR"]
    base = by_arm["BASE"]
    shuffled = by_arm["SHUFFLED_ANCHOR"]
    noise = by_arm["NOISE_ANCHOR"]
    best_control = max(base["eval_accuracy"], shuffled["eval_accuracy"], noise["eval_accuracy"])
    conditions = {
        "anchor_eval_accuracy_gte_0_90": anchor["eval_accuracy"] >= 0.90,
        "anchor_beats_controls_plus_0_25": anchor["eval_accuracy"] >= best_control + 0.25,
        "anchor_reliance_drop_gte_0_20": (anchor["anchor_reliance_drop"] or 0.0) >= 0.20,
        "shuffled_not_above_base_plus_0_10": shuffled["eval_accuracy"] <= base["eval_accuracy"] + 0.10,
        "noise_not_above_base_plus_0_10": noise["eval_accuracy"] <= base["eval_accuracy"] + 0.10,
    }
    return all(conditions.values()), conditions


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# ANCHOR-MINI-001 Report",
        "",
        f"Status: `{report['status']}`",
        f"Seed: `{report['seed']}`",
        "",
        "## Metrics",
        "",
        "| model | arm | train_acc | eval_acc | next_step | put_away | shortcut_trap | anchor_drop |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["metrics"]:
        drop = row["anchor_reliance_drop"]
        drop_text = "" if drop is None else f"{drop:.3f}"
        lines.append(
            f"| `{row['model']}` | `{row['arm']}` | {row['train_accuracy']:.3f} | "
            f"{row['eval_accuracy']:.3f} | {row['next_step_accuracy']:.3f} | "
            f"{row['put_away_accuracy']:.3f} | {row['shortcut_trap_rate']:.3f} | {drop_text} |"
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
    train_examples = make_examples(
        split="train",
        count=args.train_examples,
        rng=rng,
        object_pool=range(0, 40),
        site_pool=range(0, 40),
    )
    eval_examples = make_examples(
        split="eval",
        count=args.eval_examples,
        rng=rng,
        object_pool=range(40, 60),
        site_pool=range(40, 60),
    )
    max_object_id = 59
    max_site_id = 59

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
                    max_object_id=max_object_id,
                    max_site_id=max_site_id,
                    seed=args.seed,
                    epochs=args.epochs,
                    lr=args.lr,
                    hidden=args.hidden,
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
        "train_object_ids": [0, 39],
        "eval_object_ids": [40, 59],
        "train_site_ids": [0, 39],
        "eval_site_ids": [40, 59],
        "arms": ARMS,
        "models": MODELS,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden": args.hidden,
        "gold_rule": "next_step -> use candidate; put_away -> storage candidate",
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
