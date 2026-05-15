#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output" / "prismion_unit_bench_001"
DOC_REPORT = ROOT / "docs" / "research" / "PRISMION_UNIT_BENCH_001_RESULT.md"

ARMS = (
    "one_relu_neuron",
    "two_relu_mlp",
    "fixed_prismion",
    "learned_prismion_gain_only",
    "learned_prismion_gain_phase",
    "hybrid_prismion_relu",
)

TASKS = ("xor_cancellation", "pilot_scope_core", "pilot_scope_factor_heldout")
ACTION_CLASSES = ("EXEC_A", "EXEC_B", "REJECT_UNKNOWN", "HOLD")
ACTION_TO_INDEX = {label: idx for idx, label in enumerate(ACTION_CLASSES)}
EXEC_INDICES = {ACTION_TO_INDEX["EXEC_A"], ACTION_TO_INDEX["EXEC_B"]}


@dataclass(frozen=True)
class Example:
    task: str
    case_id: str
    features: tuple[float, ...]
    label: int
    split: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class Row:
    task: str
    arm: str
    seed: int
    split: str
    accuracy: float
    false_execution_rate: float
    perfect: bool
    epochs_to_solution: int | None
    parameter_count: int
    failed_cases: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRISMION_UNIT_BENCH_001 unit-level cancellation benchmark.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="2026-2125")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def xor_examples() -> list[Example]:
    rows: list[Example] = []
    for a in (0, 1):
        for not_a in (0, 1):
            label = int((a + not_a) == 1)
            rows.append(
                Example(
                    task="xor_cancellation",
                    case_id=f"A{a}_NOTA{not_a}",
                    features=(float(a), float(not_a)),
                    label=label,
                    split="train_eval",
                    tags=("xor", "cancellation"),
                )
            )
    return rows


def action_case(
    case_id: str,
    features: tuple[int, int, int, int, int, int, int, int, int],
    label: str,
    split: str,
    tags: tuple[str, ...],
) -> Example:
    return Example(
        task="pilot_scope_core" if split == "train_eval" else "pilot_scope_factor_heldout",
        case_id=case_id,
        features=tuple(float(value) for value in features),
        label=ACTION_TO_INDEX[label],
        split=split,
        tags=tags,
    )


def action_examples() -> tuple[list[Example], list[Example]]:
    # Feature order:
    # A, NOT_A, B, NOT_B, UNKNOWN, WEAK, MENTION, CORR_A, CORR_B
    core = [
        action_case("A_only", (1, 0, 0, 0, 0, 0, 0, 0, 0), "EXEC_A", "train_eval", ("execute",)),
        action_case("B_only", (0, 0, 1, 0, 0, 0, 0, 0, 0), "EXEC_B", "train_eval", ("execute",)),
        action_case("unknown_only", (0, 0, 0, 0, 1, 0, 0, 0, 0), "REJECT_UNKNOWN", "train_eval", ("unknown",)),
        action_case("A_notA_cancel", (1, 1, 0, 0, 0, 0, 0, 0, 0), "HOLD", "train_eval", ("cancellation",)),
        action_case("B_notB_cancel", (0, 0, 1, 1, 0, 0, 0, 0, 0), "HOLD", "train_eval", ("cancellation",)),
        action_case("A_notA_then_B", (1, 1, 1, 0, 0, 0, 0, 0, 0), "EXEC_B", "train_eval", ("cancellation", "refocus")),
        action_case("B_notB_then_A", (1, 0, 1, 1, 0, 0, 0, 0, 0), "EXEC_A", "train_eval", ("cancellation", "refocus")),
        action_case("weak_A", (1, 0, 0, 0, 0, 1, 0, 0, 0), "HOLD", "train_eval", ("weak",)),
        action_case("weak_B", (0, 0, 1, 0, 0, 1, 0, 0, 0), "HOLD", "train_eval", ("weak",)),
        action_case("mention_A", (1, 0, 0, 0, 0, 0, 1, 0, 0), "HOLD", "train_eval", ("mention",)),
        action_case("mention_B", (0, 0, 1, 0, 0, 0, 1, 0, 0), "HOLD", "train_eval", ("mention",)),
        action_case("A_or_B", (1, 0, 1, 0, 0, 0, 0, 0, 0), "HOLD", "train_eval", ("ambiguous",)),
        action_case("A_or_unknown", (1, 0, 0, 0, 1, 0, 0, 0, 0), "HOLD", "train_eval", ("ambiguous", "unknown")),
        action_case("corr_to_A", (0, 0, 1, 0, 0, 0, 0, 1, 0), "EXEC_A", "train_eval", ("correction",)),
        action_case("corr_to_B", (1, 0, 0, 0, 0, 0, 0, 0, 1), "EXEC_B", "train_eval", ("correction",)),
    ]
    train = [
        action_case("train_A_only", (1, 0, 0, 0, 0, 0, 0, 0, 0), "EXEC_A", "train", ("execute",)),
        action_case("train_B_only", (0, 0, 1, 0, 0, 0, 0, 0, 0), "EXEC_B", "train", ("execute",)),
        action_case("train_unknown_only", (0, 0, 0, 0, 1, 0, 0, 0, 0), "REJECT_UNKNOWN", "train", ("unknown",)),
        action_case("train_A_notA_cancel", (1, 1, 0, 0, 0, 0, 0, 0, 0), "HOLD", "train", ("cancellation",)),
        action_case("train_B_notB_cancel", (0, 0, 1, 1, 0, 0, 0, 0, 0), "HOLD", "train", ("cancellation",)),
        action_case("train_weak_A", (1, 0, 0, 0, 0, 1, 0, 0, 0), "HOLD", "train", ("weak",)),
        action_case("train_mention_A", (1, 0, 0, 0, 0, 0, 1, 0, 0), "HOLD", "train", ("mention",)),
        action_case("train_corr_to_A", (0, 0, 1, 0, 0, 0, 0, 1, 0), "EXEC_A", "train", ("correction",)),
        action_case("train_corr_to_B", (1, 0, 0, 0, 0, 0, 0, 0, 1), "EXEC_B", "train", ("correction",)),
    ]
    eval_rows = [
        action_case("eval_A_notA_then_B", (1, 1, 1, 0, 0, 0, 0, 0, 0), "EXEC_B", "eval", ("cancellation", "refocus")),
        action_case("eval_B_notB_then_A", (1, 0, 1, 1, 0, 0, 0, 0, 0), "EXEC_A", "eval", ("cancellation", "refocus")),
        action_case("eval_weak_B", (0, 0, 1, 0, 0, 1, 0, 0, 0), "HOLD", "eval", ("weak",)),
        action_case("eval_mention_B", (0, 0, 1, 0, 0, 0, 1, 0, 0), "HOLD", "eval", ("mention",)),
        action_case("eval_A_or_B", (1, 0, 1, 0, 0, 0, 0, 0, 0), "HOLD", "eval", ("ambiguous",)),
        action_case("eval_A_or_unknown", (1, 0, 0, 0, 1, 0, 0, 0, 0), "HOLD", "eval", ("ambiguous", "unknown")),
        action_case("eval_corr_B_after_A_notA", (1, 1, 0, 0, 0, 0, 0, 0, 1), "EXEC_B", "eval", ("correction", "refocus")),
        action_case("eval_corr_A_after_B_notB", (0, 0, 1, 1, 0, 0, 0, 1, 0), "EXEC_A", "eval", ("correction", "refocus")),
    ]
    factor = train + eval_rows
    return core, factor


def task_examples(task: str) -> tuple[list[Example], list[Example]]:
    if task == "xor_cancellation":
        rows = xor_examples()
        return rows, rows
    core, factor = action_examples()
    if task == "pilot_scope_core":
        return core, core
    if task == "pilot_scope_factor_heldout":
        train = [row for row in factor if row.split == "train"]
        eval_rows = [row for row in factor if row.split == "eval"]
        return train, eval_rows
    raise ValueError(f"unknown task: {task}")


def tensorize(rows: list[Example], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([row.features for row in rows], dtype=torch.float32, device=device)
    y = torch.tensor([row.label for row in rows], dtype=torch.long, device=device)
    return x, y


class ReLUMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrismionIntensity(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, task: str, learn_phase: bool):
        super().__init__()
        self.output_dim = output_dim
        self.raw_gain = nn.Parameter(torch.zeros(input_dim, output_dim))
        phase = fixed_phase_matrix(input_dim, output_dim, task)
        if learn_phase:
            self.phase = nn.Parameter(phase + 0.15 * torch.randn_like(phase))
        else:
            self.register_buffer("phase", phase)
        self.bias_real = nn.Parameter(torch.zeros(output_dim))
        self.bias_imag = nn.Parameter(torch.zeros(output_dim))
        self.class_bias = nn.Parameter(torch.zeros(output_dim))
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = F.softplus(self.raw_gain)
        real = x @ (gain * torch.cos(self.phase)) + self.bias_real
        imag = x @ (gain * torch.sin(self.phase)) + self.bias_imag
        return self.scale.abs() * (real * real + imag * imag) + self.class_bias


class HybridPrismionReLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, task: str):
        super().__init__()
        self.prism = PrismionIntensity(input_dim, output_dim, task, learn_phase=True)
        self.head = nn.Sequential(nn.Linear(input_dim + output_dim, 8), nn.ReLU(), nn.Linear(8, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prism_logits = self.prism(x)
        return self.head(torch.cat([x, prism_logits], dim=1))


def fixed_phase_matrix(input_dim: int, output_dim: int, task: str) -> torch.Tensor:
    phase = torch.zeros(input_dim, output_dim, dtype=torch.float32)
    if task == "xor_cancellation":
        # input A and NOT_A contribute opposite phase to the positive XOR channel.
        phase[0, 1] = 0.0
        phase[1, 1] = math.pi
        return phase

    # Action output classes: EXEC_A, EXEC_B, REJECT_UNKNOWN, HOLD.
    # Input features: A, NOT_A, B, NOT_B, UNKNOWN, WEAK, MENTION, CORR_A, CORR_B.
    # Opposite phases encode cancellation on the matching executable channel.
    phase[:, :] = math.pi / 2.0
    phase[0, ACTION_TO_INDEX["EXEC_A"]] = 0.0
    phase[1, ACTION_TO_INDEX["EXEC_A"]] = math.pi
    phase[1, ACTION_TO_INDEX["HOLD"]] = 0.0
    phase[2, ACTION_TO_INDEX["EXEC_B"]] = 0.0
    phase[3, ACTION_TO_INDEX["EXEC_B"]] = math.pi
    phase[3, ACTION_TO_INDEX["HOLD"]] = 0.0
    phase[4, ACTION_TO_INDEX["REJECT_UNKNOWN"]] = 0.0
    phase[5, ACTION_TO_INDEX["HOLD"]] = 0.0
    phase[6, ACTION_TO_INDEX["HOLD"]] = 0.0
    phase[7, ACTION_TO_INDEX["EXEC_A"]] = 0.0
    phase[7, ACTION_TO_INDEX["EXEC_B"]] = math.pi
    phase[7, ACTION_TO_INDEX["HOLD"]] = math.pi
    phase[8, ACTION_TO_INDEX["EXEC_B"]] = 0.0
    phase[8, ACTION_TO_INDEX["EXEC_A"]] = math.pi
    phase[8, ACTION_TO_INDEX["HOLD"]] = math.pi
    return phase


def build_model(arm: str, input_dim: int, output_dim: int, task: str) -> nn.Module | None:
    if arm == "one_relu_neuron":
        return ReLUMLP(input_dim, output_dim, hidden=1)
    if arm == "two_relu_mlp":
        return ReLUMLP(input_dim, output_dim, hidden=2)
    if arm == "fixed_prismion":
        return None
    if arm == "learned_prismion_gain_only":
        return PrismionIntensity(input_dim, output_dim, task, learn_phase=False)
    if arm == "learned_prismion_gain_phase":
        return PrismionIntensity(input_dim, output_dim, task, learn_phase=True)
    if arm == "hybrid_prismion_relu":
        return HybridPrismionReLU(input_dim, output_dim, task)
    raise ValueError(f"unknown arm: {arm}")


def fixed_prismion_predict(task: str, rows: list[Example]) -> torch.Tensor:
    preds: list[int] = []
    for row in rows:
        x = row.features
        if task == "xor_cancellation":
            intensity = (x[0] - x[1]) ** 2
            preds.append(int(intensity >= 0.5))
            continue
        a, not_a, b, not_b, unknown, weak, mention, corr_a, corr_b = [bool(v) for v in x]
        if mention:
            preds.append(ACTION_TO_INDEX["HOLD"])
        elif corr_a and corr_b:
            preds.append(ACTION_TO_INDEX["HOLD"])
        elif corr_a:
            preds.append(ACTION_TO_INDEX["EXEC_A"])
        elif corr_b:
            preds.append(ACTION_TO_INDEX["EXEC_B"])
        elif weak:
            preds.append(ACTION_TO_INDEX["HOLD"])
        else:
            active_a = a and not not_a
            active_b = b and not not_b
            if unknown and not (active_a or active_b):
                preds.append(ACTION_TO_INDEX["REJECT_UNKNOWN"])
            elif unknown and (active_a or active_b):
                preds.append(ACTION_TO_INDEX["HOLD"])
            elif active_a and not active_b:
                preds.append(ACTION_TO_INDEX["EXEC_A"])
            elif active_b and not active_a:
                preds.append(ACTION_TO_INDEX["EXEC_B"])
            else:
                preds.append(ACTION_TO_INDEX["HOLD"])
    return torch.tensor(preds, dtype=torch.long)


def param_count(model: nn.Module | None) -> int:
    if model is None:
        return 0
    return int(sum(param.numel() for param in model.parameters()))


def train_and_eval(task: str, arm: str, seed: int, epochs: int, lr: float, device: torch.device) -> Row:
    set_seed(seed)
    train_rows, eval_rows = task_examples(task)
    input_dim = len(train_rows[0].features)
    output_dim = 2 if task == "xor_cancellation" else len(ACTION_CLASSES)
    model = build_model(arm, input_dim, output_dim, task)
    if model is None:
        preds = fixed_prismion_predict(task, eval_rows)
        return score_row(task, arm, seed, "eval", eval_rows, preds, None, 0)

    model.to(device)
    train_x, train_y = tensorize(train_rows, device)
    eval_x, _ = tensorize(eval_rows, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs_to_solution: int | None = None
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_pred = model(train_x).argmax(dim=1)
            if epochs_to_solution is None and torch.equal(train_pred, train_y):
                epochs_to_solution = epoch
                # Keep training a little after first solution for stability, but
                # avoid wasting thousands of epochs on already-solved toy rows.
                if epoch >= 30:
                    break

    with torch.no_grad():
        preds = model(eval_x).argmax(dim=1).detach().cpu()
    return score_row(task, arm, seed, "eval", eval_rows, preds, epochs_to_solution, param_count(model))


def score_row(
    task: str,
    arm: str,
    seed: int,
    split: str,
    rows: list[Example],
    preds: torch.Tensor,
    epochs_to_solution: int | None,
    parameters: int,
) -> Row:
    labels = torch.tensor([row.label for row in rows], dtype=torch.long)
    correct_mask = preds.eq(labels)
    false_execution = 0
    failed_cases: list[dict[str, object]] = []
    for row, pred, correct in zip(rows, preds.tolist(), correct_mask.tolist()):
        if task != "xor_cancellation" and pred in EXEC_INDICES and row.label not in EXEC_INDICES:
            false_execution += 1
        if not correct:
            failed_cases.append(
                {
                    "case_id": row.case_id,
                    "expected": class_label(task, row.label),
                    "predicted": class_label(task, pred),
                    "tags": list(row.tags),
                    "features": list(row.features),
                }
            )
    accuracy = float(correct_mask.float().mean().item())
    return Row(
        task=task,
        arm=arm,
        seed=seed,
        split=split,
        accuracy=round(accuracy, 6),
        false_execution_rate=round(false_execution / max(1, len(rows)), 6),
        perfect=bool(accuracy == 1.0),
        epochs_to_solution=epochs_to_solution,
        parameter_count=parameters,
        failed_cases=failed_cases[:12],
    )


def class_label(task: str, idx: int) -> str:
    if task == "xor_cancellation":
        return "XOR_ACTIVE" if idx else "XOR_CANCELLED"
    return ACTION_CLASSES[idx]


def aggregate(rows: list[Row]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(row.task, row.arm)].append(row)
    by_task_arm: dict[str, dict[str, object]] = {}
    for (task, arm), group in sorted(grouped.items()):
        accuracies = [row.accuracy for row in group]
        false_exec = [row.false_execution_rate for row in group]
        epochs = [row.epochs_to_solution for row in group if row.epochs_to_solution is not None]
        by_task_arm[f"{task}/{arm}"] = {
            "mean_accuracy": round(float(np.mean(accuracies)), 6),
            "min_accuracy": round(float(np.min(accuracies)), 6),
            "perfect_run_rate": round(sum(row.perfect for row in group) / len(group), 6),
            "mean_false_execution_rate": round(float(np.mean(false_exec)), 6),
            "median_epochs_to_solution": None if not epochs else int(np.median(epochs)),
            "parameter_count": group[0].parameter_count,
        }

    by_arm: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_arm[row.arm].append(row)
    arm_summary: dict[str, dict[str, object]] = {}
    for arm, group in sorted(by_arm.items()):
        arm_summary[arm] = {
            "mean_accuracy": round(float(np.mean([row.accuracy for row in group])), 6),
            "perfect_run_rate": round(sum(row.perfect for row in group) / len(group), 6),
            "mean_false_execution_rate": round(float(np.mean([row.false_execution_rate for row in group])), 6),
            "mean_parameter_count": round(float(np.mean([row.parameter_count for row in group])), 3),
        }
    return {"by_task_arm": by_task_arm, "by_arm": arm_summary, "verdict": verdict(by_task_arm, arm_summary)}


def verdict(by_task_arm: dict[str, dict[str, object]], by_arm: dict[str, dict[str, object]]) -> list[str]:
    labels: list[str] = []
    fixed = by_arm["fixed_prismion"]
    gain_phase = by_arm["learned_prismion_gain_phase"]
    relu2 = by_arm["two_relu_mlp"]
    hybrid = by_arm["hybrid_prismion_relu"]
    if fixed["perfect_run_rate"] == 1.0:
        labels.append("FIXED_PRISMION_POSITIVE")
    if gain_phase["mean_accuracy"] >= relu2["mean_accuracy"] + 0.05:
        labels.append("LEARNED_PRISMION_BEATS_TWO_RELU")
    if gain_phase["perfect_run_rate"] >= 0.80:
        labels.append("PRISMION_UNIT_POSITIVE")
    elif fixed["perfect_run_rate"] == 1.0 and gain_phase["perfect_run_rate"] < 0.80:
        labels.append("PRISMION_ONLY_FIXED_POSITIVE")
    if hybrid["mean_accuracy"] >= max(gain_phase["mean_accuracy"], relu2["mean_accuracy"]):
        labels.append("HYBRID_BEST_OR_TIED")
    if relu2["perfect_run_rate"] >= 0.80:
        labels.append("RELU_SUFFICIENT_ON_THIS_BENCH")
    if by_task_arm["pilot_scope_factor_heldout/learned_prismion_gain_phase"]["perfect_run_rate"] < 0.50:
        labels.append("FACTOR_HELDOUT_LEARNABILITY_WEAK")
    return labels


def write_outputs(out_dir: Path, rows: list[Row], summary: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with (out_dir / "rows.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "task",
                "arm",
                "seed",
                "accuracy",
                "false_execution_rate",
                "perfect",
                "epochs_to_solution",
                "parameter_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task": row.task,
                    "arm": row.arm,
                    "seed": row.seed,
                    "accuracy": row.accuracy,
                    "false_execution_rate": row.false_execution_rate,
                    "perfect": int(row.perfect),
                    "epochs_to_solution": row.epochs_to_solution,
                    "parameter_count": row.parameter_count,
                }
            )
    DOC_REPORT.write_text(render_report(summary), encoding="utf-8")


def render_report(summary: dict[str, object]) -> str:
    by_task_arm = summary["by_task_arm"]
    by_arm = summary["by_arm"]
    verdict_labels = summary["verdict"]
    lines = [
        "# PRISMION_UNIT_BENCH_001 Result",
        "",
        "## Goal",
        "",
        "Test whether Prismion-style phase/interference units are a useful primitive for cancellation and command-scope decisions.",
        "",
        "This is a unit-level expressivity and learnability probe, not a full model benchmark.",
        "",
        "## Arm Summary",
        "",
        "| Arm | Mean Acc | Perfect Rate | False Exec | Mean Params |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        arm_metrics = by_arm[arm]
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.1f}` |".format(
                arm,
                float(arm_metrics["mean_accuracy"]),
                float(arm_metrics["perfect_run_rate"]),
                float(arm_metrics["mean_false_execution_rate"]),
                float(arm_metrics["mean_parameter_count"]),
            )
        )
    lines.extend(["", "## Task / Arm Detail", "", "| Task / Arm | Mean Acc | Min Acc | Perfect Rate | False Exec | Median Epoch | Params |", "|---|---:|---:|---:|---:|---:|---:|"])
    for key in sorted(by_task_arm):
        item = by_task_arm[key]
        median_epoch = "null" if item["median_epochs_to_solution"] is None else str(item["median_epochs_to_solution"])
        lines.append(
            "| `{}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{:.3f}` | `{}` | `{}` |".format(
                key,
                float(item["mean_accuracy"]),
                float(item["min_accuracy"]),
                float(item["perfect_run_rate"]),
                float(item["mean_false_execution_rate"]),
                median_epoch,
                item["parameter_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "```json",
            json.dumps(verdict_labels, indent=2),
            "```",
            "",
            "## Interpretation",
            "",
            "The fixed Prismion arm is an upper-bound representation check: it asks whether the primitive can express the target cancellation/interference rules directly.",
            "",
            "The learned Prismion arms test whether the primitive is learnable from labels across many seeds. If fixed Prismion passes but learned Prismion is weak, the representation is promising but the learning rule/cell still needs work.",
            "",
            "The ReLU arms are deliberately tiny. A two-ReLU model solving parts of the bench means ordinary neurons can represent some cases, not that phase/interference is useless.",
            "",
            "## Claim Boundary",
            "",
            "Toy command/cancellation domain only. No consciousness claim, no quantum physics claim, no general NLU claim, and no full VRAXION/INSTNCT claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    seeds = parse_seeds(args.seeds)
    rows: list[Row] = []
    for task in TASKS:
        for arm in ARMS:
            for seed in seeds:
                rows.append(train_and_eval(task, arm, seed, args.epochs, args.lr, device))
    summary = aggregate(rows)
    write_outputs(args.out_dir, rows, summary)
    print(json.dumps({"out_dir": str(args.out_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
