#!/usr/bin/env python3
"""ITERATIVE_REFINEMENT_DYNAMICS_001.

Deck-local deterministic probe for output-refeed iterative state refinement.

The model is trained on short integer transition trajectories and evaluated in
free-run mode on longer trajectories where its own previous output becomes the
next input. This is not a language-model or assistant-readiness gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "ITERATIVE_REFINEMENT_DYNAMICS_001"
DEFAULT_OUT = Path("target/pilot_wave/iterative_refinement_dynamics_001/smoke")
OPS = [-5, -3, -2, -1, 1, 2, 3, 5]
DELTA_CLASSES = list(range(-5, 6))
DELTA_TO_CLASS = {delta: idx for idx, delta in enumerate(DELTA_CLASSES)}
VALUE_SCALE = 160.0
BOUNDARY_TEXT = (
    "ITERATIVE_REFINEMENT_DYNAMICS_001 tests deterministic output-refeed state dynamics on a toy integer "
    "transition task. It is not GPT-like assistant readiness, not open-domain language understanding, "
    "not production readiness, not safety alignment, and not a claim of general reasoning."
)


@dataclass(frozen=True)
class TransitionExample:
    split: str
    family: str
    current: int
    target: int
    op: int
    expected_next: int
    expected_delta: int


@dataclass(frozen=True)
class TrajectoryCase:
    split: str
    family: str
    start: int
    target: int
    op: int
    expected_horizon: int


class DeltaClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(DELTA_CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_hash(model: nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return sha256_bytes(buf.getvalue())


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise SystemExit("--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise SystemExit("--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def oracle_next(current: int, target: int, op: int) -> int:
    if current == target:
        return current
    distance = target - current
    if distance * op <= 0:
        # The command points away from target. Safe deterministic behavior is hold.
        return current
    if abs(distance) <= abs(op):
        return target
    return current + op


def features(current: int, target: int, op: int) -> list[float]:
    distance = target - current
    abs_distance = abs(distance)
    abs_op = abs(op)
    aligned = 1.0 if distance * op > 0 else 0.0
    clipped_steps_remaining = min(abs_distance / max(1, abs_op), 20.0) / 20.0
    near_target = 1.0 if abs_distance <= abs_op else 0.0
    op_one_hot = [1.0 if op == candidate else 0.0 for candidate in OPS]
    sign_one_hot = [
        1.0 if distance < 0 else 0.0,
        1.0 if distance == 0 else 0.0,
        1.0 if distance > 0 else 0.0,
    ]
    residual = (abs_distance % max(1, abs_op)) / max(1, abs_op)
    return [
        current / VALUE_SCALE,
        target / VALUE_SCALE,
        op / 5.0,
        distance / VALUE_SCALE,
        abs_distance / VALUE_SCALE,
        math.copysign(1.0, distance) if distance else 0.0,
        math.copysign(1.0, op),
        abs_op / 5.0,
        aligned,
        near_target,
        clipped_steps_remaining,
        residual,
        1.0,
        *op_one_hot,
        *sign_one_hot,
    ]


def build_transition_examples(max_train_steps: int) -> tuple[list[TransitionExample], list[TransitionExample], list[TrajectoryCase]]:
    train: list[TransitionExample] = []
    teacher_forced_eval: list[TransitionExample] = []
    trajectory_eval: list[TrajectoryCase] = []

    # Short trajectories for training. They cover the rule locally but not the long free-run horizons.
    for start in range(-120, 121, 5):
        for op in OPS:
            for steps in range(1, max_train_steps + 1):
                target = start + op * steps
                if not -150 <= target <= 150:
                    continue
                current = start
                while current != target:
                    nxt = oracle_next(current, target, op)
                    train.append(
                        TransitionExample(
                            split="train",
                            family="short_teacher_forced",
                            current=current,
                            target=target,
                            op=op,
                            expected_next=nxt,
                            expected_delta=nxt - current,
                        )
                    )
                    current = nxt

    # Clamp/overshoot examples: target is closer than abs(op).
    for target in range(-140, 141, 7):
        for op in OPS:
            direction = 1 if op > 0 else -1
            for residual in range(1, abs(op)):
                current = target - direction * residual
                if not -150 <= current <= 150:
                    continue
                nxt = oracle_next(current, target, op)
                train.append(
                    TransitionExample(
                        split="train",
                        family="short_clamp",
                        current=current,
                        target=target,
                        op=op,
                        expected_next=nxt,
                        expected_delta=nxt - current,
                    )
                )

    # Wrong-direction examples teach safe HOLD instead of drifting away.
    for current in range(-150, 151, 10):
        for target in range(-150, 151, 15):
            if current == target:
                continue
            distance = target - current
            for op in OPS:
                if distance * op < 0:
                    train.append(
                        TransitionExample(
                            split="train",
                            family="wrong_direction_hold",
                            current=current,
                            target=target,
                            op=op,
                            expected_next=current,
                            expected_delta=0,
                        )
                    )

    # Heldout teacher-forced transitions include long remaining distances and off-grid final clamps.
    for start in range(-135, 136, 9):
        for op in OPS:
            for steps in [25, 35, 50, 75, 100, 125, 150]:
                target = start + op * steps
                if not -150 <= target <= 150:
                    continue
                for offset_steps in [0, max(0, steps // 3), max(0, (2 * steps) // 3), max(0, steps - 1)]:
                    current = start + op * offset_steps
                    if current == target:
                        continue
                    nxt = oracle_next(current, target, op)
                    teacher_forced_eval.append(
                        TransitionExample(
                            split="heldout_long_teacher_forced",
                            family=f"horizon_{bucket_horizon(steps)}",
                            current=current,
                            target=target,
                            op=op,
                            expected_next=nxt,
                            expected_delta=nxt - current,
                        )
                    )
                trajectory_eval.append(
                    TrajectoryCase(
                        split="heldout_long_free_run",
                        family=f"horizon_{bucket_horizon(steps)}",
                        start=start,
                        target=target,
                        op=op,
                        expected_horizon=steps,
                    )
                )

    # Wrong-direction safety cases should hold, not drift away.
    for current in range(-90, 91, 30):
        for target in range(-120, 121, 40):
            if current == target:
                continue
            for op in OPS:
                if (target - current) * op < 0:
                    nxt = oracle_next(current, target, op)
                    teacher_forced_eval.append(
                        TransitionExample(
                            split="heldout_wrong_direction",
                            family="wrong_direction_hold",
                            current=current,
                            target=target,
                            op=op,
                            expected_next=nxt,
                            expected_delta=0,
                        )
                    )
                    trajectory_eval.append(
                        TrajectoryCase(
                            split="heldout_wrong_direction",
                            family="wrong_direction_hold",
                            start=current,
                            target=target,
                            op=op,
                            expected_horizon=0,
                        )
                    )
                    break

    return train, teacher_forced_eval, trajectory_eval


def bucket_horizon(steps: int) -> str:
    if steps < 25:
        return "short"
    if steps < 50:
        return "25_49"
    if steps < 75:
        return "50_74"
    if steps < 100:
        return "75_99"
    return "100_plus"


def rows_to_tensors(rows: list[TransitionExample]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([features(row.current, row.target, row.op) for row in rows], dtype=torch.float32)
    y = torch.tensor([DELTA_TO_CLASS[row.expected_delta] for row in rows], dtype=torch.long)
    return x, y


@torch.no_grad()
def eval_teacher_forced(
    model: DeltaClassifier,
    rows: list[TransitionExample],
    batch_size: int = 2048,
) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for idx in range(0, len(rows), batch_size):
        batch = rows[idx : idx + batch_size]
        x, y = rows_to_tensors(batch)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
        pred = logits.argmax(dim=-1)
        for row, pred_idx, gold_idx in zip(batch, pred.tolist(), y.tolist()):
            ok = int(pred_idx == gold_idx)
            correct += ok
            total += 1
            by_family[row.family][0] += ok
            by_family[row.family][1] += 1
    return {
        "teacher_forced_loss": loss_sum / max(1, total),
        "teacher_forced_transition_accuracy": correct / max(1, total),
        "teacher_forced_count": total,
        "teacher_forced_accuracy_by_family": {family: ok / max(1, count) for family, (ok, count) in sorted(by_family.items())},
    }


def predict_delta(model: DeltaClassifier, current: int, target: int, op: int) -> int:
    with torch.no_grad():
        x = torch.tensor([features(current, target, op)], dtype=torch.float32)
        pred_idx = int(model(x).argmax(dim=-1).item())
    return DELTA_CLASSES[pred_idx]


def checker_repair_delta(current: int, target: int, op: int, delta: int) -> tuple[int, bool]:
    if current == target:
        return 0, delta != 0
    oracle_delta = oracle_next(current, target, op) - current
    proposed_next = current + delta
    distance = target - current
    if delta == oracle_delta:
        return delta, False
    if delta == 0 and oracle_delta == 0:
        return delta, False
    if delta == 0 and oracle_delta != 0:
        return oracle_delta, True
    if distance * delta <= 0:
        return oracle_delta, True
    if abs(proposed_next - current) > abs(distance):
        return oracle_delta, True
    if (target - proposed_next) * distance < 0:
        return oracle_delta, True
    return delta, False


def free_run_model(
    name: str,
    cases: list[TrajectoryCase],
    predictor: Callable[[int, int, int], int],
    out: Path,
    use_checker: bool = False,
    max_extra_steps: int = 8,
) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_cases = 0
    converged = 0
    final_target_correct = 0
    transition_ok = 0
    transition_total = 0
    overshoot = 0
    wrong_direction = 0
    drift = 0
    cycles = 0
    checker_repairs = 0
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    steps_to_target: list[int] = []
    max_stable_horizon = 0

    for case in cases:
        current = case.start
        seen = {current}
        case_transition_ok = 0
        case_transition_total = 0
        case_wrong_direction = False
        case_overshoot = False
        case_drift = False
        case_cycle = False
        reached = False
        previous_distance = abs(case.target - current)
        max_steps = max(1, case.expected_horizon + max_extra_steps)
        for step_idx in range(max_steps):
            oracle_nxt = oracle_next(current, case.target, case.op)
            oracle_delta = oracle_nxt - current
            raw_delta = predictor(current, case.target, case.op)
            repaired = False
            delta = raw_delta
            if use_checker:
                delta, repaired = checker_repair_delta(current, case.target, case.op, raw_delta)
                checker_repairs += int(repaired)
            nxt = current + delta
            ok = int(delta == oracle_delta)
            case_transition_ok += ok
            case_transition_total += 1
            transition_ok += ok
            transition_total += 1
            new_distance = abs(case.target - nxt)
            if case.target != current and delta * (case.target - current) < 0:
                wrong_direction += 1
                case_wrong_direction = True
            if case.target != current and new_distance > previous_distance:
                drift += 1
                case_drift = True
            if (case.target - current) and (case.target - nxt) and (case.target - current) * (case.target - nxt) < 0:
                overshoot += 1
                case_overshoot = True
            trace_row = {
                "model_name": name,
                "split": case.split,
                "family": case.family,
                "start": case.start,
                "target": case.target,
                "op": case.op,
                "expected_horizon": case.expected_horizon,
                "step_index": step_idx,
                "current": current,
                "oracle_delta": oracle_delta,
                "raw_predicted_delta": raw_delta,
                "predicted_delta": delta,
                "next": nxt,
                "transition_correct": bool(ok),
                "checker_repaired": repaired,
            }
            if len(traces) < 5000:
                traces.append(trace_row)
            current = nxt
            previous_distance = new_distance
            if case.expected_horizon == 0:
                reached = delta == 0 and current == case.start
                if reached:
                    steps_to_target.append(0)
                break
            if current == case.target:
                reached = True
                steps_to_target.append(step_idx + 1)
                break
            if current in seen:
                cycles += 1
                case_cycle = True
                break
            seen.add(current)
        total_cases += 1
        case_ok = int(reached)
        converged += case_ok
        final_target_correct += case_ok
        by_family[case.family][0] += case_ok
        by_family[case.family][1] += 1
        if reached and case.expected_horizon > max_stable_horizon:
            max_stable_horizon = case.expected_horizon
        if not reached or case_wrong_direction or case_overshoot or case_drift or case_cycle:
            if len(failures) < 200:
                failures.append(
                    {
                        "model_name": name,
                        "split": case.split,
                        "family": case.family,
                        "start": case.start,
                        "target": case.target,
                        "op": case.op,
                        "expected_horizon": case.expected_horizon,
                        "final_state": current,
                        "reached_target": reached,
                        "wrong_direction": case_wrong_direction,
                        "overshoot": case_overshoot,
                        "drift": case_drift,
                        "cycle": case_cycle,
                        "case_transition_accuracy": case_transition_ok / max(1, case_transition_total),
                    }
                )
    write_jsonl(out / f"trajectory_traces_{name}.jsonl", traces)
    write_jsonl(out / f"failure_examples_{name}.jsonl", failures)
    return {
        "model_name": name,
        "free_run_case_count": total_cases,
        "free_run_convergence_rate": converged / max(1, total_cases),
        "final_target_accuracy": final_target_correct / max(1, total_cases),
        "free_run_transition_accuracy": transition_ok / max(1, transition_total),
        "wrong_direction_rate": wrong_direction / max(1, transition_total),
        "overshoot_rate": overshoot / max(1, transition_total),
        "drift_rate": drift / max(1, transition_total),
        "cycle_rate": cycles / max(1, total_cases),
        "mean_steps_to_target": sum(steps_to_target) / max(1, len(steps_to_target)),
        "max_stable_horizon": max_stable_horizon,
        "checker_repair_rate": checker_repairs / max(1, transition_total),
        "convergence_by_family": {family: ok / max(1, count) for family, (ok, count) in sorted(by_family.items())},
        "failure_count": len(failures),
    }


def train_delta_classifier(
    args: argparse.Namespace,
    out: Path,
    train_rows: list[TransitionExample],
    eval_rows: list[TransitionExample],
) -> tuple[DeltaClassifier, dict[str, Any]]:
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    random.seed(args.seed)
    model = DeltaClassifier(input_dim=len(features(0, 1, 1)), hidden=args.hidden)
    before_hash = model_state_hash(model)
    x_train, y_train = rows_to_tensors(train_rows)
    x_eval, y_eval = rows_to_tensors(eval_rows)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    with torch.no_grad():
        train_loss_initial = float(F.cross_entropy(model(x_train[: min(4096, x_train.shape[0])]), y_train[: min(4096, y_train.shape[0])]).item())
        eval_loss_initial = float(F.cross_entropy(model(x_eval), y_eval).item())
        eval_acc_initial = float((model(x_eval).argmax(dim=-1) == y_eval).float().mean().item())
    last = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x_train.shape[0], (args.batch_size,), generator=generator)
        logits = model(x_train[idx])
        loss = F.cross_entropy(logits, y_train[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(1, args.steps // 20) == 0:
            with torch.no_grad():
                eval_logits = model(x_eval)
                eval_loss = float(F.cross_entropy(eval_logits, y_eval).item())
                eval_acc = float((eval_logits.argmax(dim=-1) == y_eval).float().mean().item())
            append_jsonl(
                out / "training_metrics.jsonl",
                {
                    "ts": utc_now(),
                    "step": step,
                    "train_batch_loss": float(loss.item()),
                    "eval_loss": eval_loss,
                    "teacher_forced_transition_accuracy": eval_acc,
                },
            )
            last = time.time()
        elif time.time() - last >= args.heartbeat_sec:
            append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "training heartbeat", "step": step})
            last = time.time()
    with torch.no_grad():
        train_loss_final = float(F.cross_entropy(model(x_train[: min(4096, x_train.shape[0])]), y_train[: min(4096, y_train.shape[0])]).item())
        eval_loss_final = float(F.cross_entropy(model(x_eval), y_eval).item())
        eval_acc_final = float((model(x_eval).argmax(dim=-1) == y_eval).float().mean().item())
    after_hash = model_state_hash(model)
    metrics = {
        "checkpoint_before_hash": before_hash,
        "checkpoint_after_hash": after_hash,
        "checkpoint_changed": before_hash != after_hash,
        "train_loss_initial": train_loss_initial,
        "train_loss_final": train_loss_final,
        "train_loss_delta": train_loss_initial - train_loss_final,
        "eval_loss_initial": eval_loss_initial,
        "eval_loss_final": eval_loss_final,
        "eval_loss_delta": eval_loss_initial - eval_loss_final,
        "teacher_forced_transition_accuracy_initial": eval_acc_initial,
        "teacher_forced_transition_accuracy_final": eval_acc_final,
    }
    checkpoint_dir = out / "checkpoints/iterative_refinement_delta_classifier"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "delta_classes": DELTA_CLASSES,
            "feature_count": len(features(0, 1, 1)),
            "config": vars(args),
        },
        checkpoint_path,
    )
    write_json(
        out / "checkpoint_manifest.json",
        {
            "schema_version": "iterative_refinement_checkpoint_manifest_v1",
            "checkpoint_path": rel(checkpoint_path),
            "checkpoint_file_sha256": sha256_file(checkpoint_path),
            **metrics,
        },
    )
    return model, metrics


def oracle_predictor(current: int, target: int, op: int) -> int:
    return oracle_next(current, target, op) - current


def direct_to_target_predictor(current: int, target: int, op: int) -> int:
    del op
    distance = target - current
    if distance > 5:
        return 5
    if distance < -5:
        return -5
    return distance


def summarize_verdict(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    if not metrics.get("checkpoint_changed"):
        failures.append("CHECKPOINT_DID_NOT_CHANGE")
    if metrics.get("teacher_forced_transition_accuracy_final", 0.0) < 0.995:
        failures.append("TEACHER_FORCED_TRANSITION_WEAK")
    if metrics.get("learned_free_run_convergence_rate", 0.0) < 0.95:
        failures.append("FREE_RUN_CONVERGENCE_WEAK")
    if metrics.get("learned_final_target_accuracy", 0.0) < 0.95:
        failures.append("FINAL_TARGET_WEAK")
    if metrics.get("learned_wrong_direction_rate", 1.0) > 0.01:
        failures.append("WRONG_DIRECTION_HIGH")
    if metrics.get("learned_cycle_rate", 1.0) > 0.01:
        failures.append("CYCLE_HIGH")
    if metrics.get("teacher_forced_vs_free_run_gap", 1.0) > 0.05:
        failures.append("TEACHER_FORCED_FREE_RUN_GAP_HIGH")
    if metrics.get("direct_to_target_per_step_transition_accuracy", 1.0) >= 0.50:
        failures.append("DIRECT_TO_TARGET_CONTROL_TOO_GOOD")
    verdicts = [
        "CHECKPOINT_CHANGED" if metrics.get("checkpoint_changed") else "CHECKPOINT_UNCHANGED",
        "ORACLE_CONTROL_PASSES",
        "DIRECT_TO_TARGET_SHORTCUT_REJECTED",
        "GPT_LIKE_READINESS_NOT_CLAIMED",
        "PRODUCTION_READINESS_NOT_CLAIMED",
    ]
    if failures:
        return "failed", ["ITERATIVE_REFINEMENT_DYNAMICS_FAILS", *failures, *verdicts]
    return "positive", [
        "ITERATIVE_REFINEMENT_DYNAMICS_POSITIVE",
        "LEARNED_TRANSITION_RULE_STABLE_IN_FREE_RUN",
        "LONG_HORIZON_REFEED_CONVERGES",
        "TEACHER_FORCED_FREE_RUN_GAP_ACCEPTABLE",
        *verdicts,
    ]


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_and_report(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], run_rows: list[dict[str, Any]]) -> None:
    summary = {
        "schema_version": "iterative_refinement_dynamics_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "boundary": BOUNDARY_TEXT,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_reasoning_claimed": False,
        "production_readiness_claimed": False,
        "safety_alignment_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
        "arms": run_rows,
    }
    write_json(out / "summary.json", summary)
    lines = [
        "# ITERATIVE_REFINEMENT_DYNAMICS_001 Report",
        "",
        BOUNDARY_TEXT,
        "",
        f"Status: `{status}`",
        "",
        "## Verdicts",
        "",
        "```text",
        *verdicts,
        "```",
        "",
        "## Key Metrics",
        "",
    ]
    for key in [
        "train_example_count",
        "teacher_forced_eval_count",
        "trajectory_eval_count",
        "train_loss_initial",
        "train_loss_final",
        "eval_loss_initial",
        "eval_loss_final",
        "teacher_forced_transition_accuracy_final",
        "learned_free_run_convergence_rate",
        "learned_final_target_accuracy",
        "learned_free_run_transition_accuracy",
        "teacher_forced_vs_free_run_gap",
        "learned_max_stable_horizon",
        "learned_wrong_direction_rate",
        "learned_overshoot_rate",
        "learned_drift_rate",
        "learned_cycle_rate",
        "direct_to_target_final_target_accuracy",
        "direct_to_target_per_step_transition_accuracy",
        "checker_guarded_convergence_rate",
        "checker_repair_rate",
        "wall_clock_sec",
    ]:
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")
    lines.extend(
        [
            "",
            "## Arm Summary",
            "",
            "| arm | convergence | final target | transition acc | wrong direction | cycle | max horizon |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in run_rows:
        lines.append(
            "| {model_name} | {free_run_convergence_rate:.6f} | {final_target_accuracy:.6f} | "
            "{free_run_transition_accuracy:.6f} | {wrong_direction_rate:.6f} | {cycle_rate:.6f} | {max_stable_horizon} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive result means the learned transition model remains stable when its own output is re-fed as the next input on longer heldout trajectories. "
            "This is a toy iterative dynamics result, not language understanding.",
            "",
            "The direct-to-target control is intentionally rejected as an iterative mechanism when it reaches the target without matching the expected per-step transition path.",
            "",
            "## Boundary",
            "",
            "Toy integer transition dynamics only.",
            "No GPT-like assistant readiness.",
            "No production readiness.",
            "No general reasoning claim.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-steps", type=int, default=60)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()
    args.out = resolve_target_out(args.out)
    return args


def main() -> int:
    started = time.time()
    args = parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "start", "status": "running"})
    write_json(
        out / "eval_config.json",
        {
            "schema_version": "iterative_refinement_dynamics_config_v1",
            "seed": args.seed,
            "ops": OPS,
            "delta_classes": DELTA_CLASSES,
            "max_train_steps": args.max_train_steps,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "hidden": args.hidden,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "boundary": BOUNDARY_TEXT,
        },
    )
    train_rows, tf_eval_rows, trajectory_cases = build_transition_examples(args.max_train_steps)
    append_jsonl(
        out / "progress.jsonl",
        {
            "ts": utc_now(),
            "event": "dataset built",
            "train_example_count": len(train_rows),
            "teacher_forced_eval_count": len(tf_eval_rows),
            "trajectory_eval_count": len(trajectory_cases),
        },
    )
    write_jsonl(out / "train_examples_sample.jsonl", [row.__dict__ for row in train_rows[:200]])
    write_jsonl(out / "teacher_forced_eval_sample.jsonl", [row.__dict__ for row in tf_eval_rows[:200]])
    write_jsonl(out / "trajectory_eval_sample.jsonl", [row.__dict__ for row in trajectory_cases[:200]])
    model, train_metrics = train_delta_classifier(args, out, train_rows, tf_eval_rows)
    tf_metrics = eval_teacher_forced(model, tf_eval_rows)

    learned = free_run_model("learned_delta_classifier", trajectory_cases, lambda c, t, o: predict_delta(model, c, t, o), out)
    checker = free_run_model("checker_guarded_learned_delta_classifier", trajectory_cases, lambda c, t, o: predict_delta(model, c, t, o), out, use_checker=True)
    oracle = free_run_model("oracle_transition", trajectory_cases, oracle_predictor, out)
    direct = free_run_model("direct_to_target_baseline", trajectory_cases, direct_to_target_predictor, out)
    run_rows = [learned, checker, oracle, direct]
    write_metrics_csv(out / "metrics.csv", run_rows)

    metrics: dict[str, Any] = {
        "train_example_count": len(train_rows),
        "teacher_forced_eval_count": len(tf_eval_rows),
        "trajectory_eval_count": len(trajectory_cases),
        **train_metrics,
        **tf_metrics,
        "learned_free_run_convergence_rate": learned["free_run_convergence_rate"],
        "learned_final_target_accuracy": learned["final_target_accuracy"],
        "learned_free_run_transition_accuracy": learned["free_run_transition_accuracy"],
        "learned_wrong_direction_rate": learned["wrong_direction_rate"],
        "learned_overshoot_rate": learned["overshoot_rate"],
        "learned_drift_rate": learned["drift_rate"],
        "learned_cycle_rate": learned["cycle_rate"],
        "learned_max_stable_horizon": learned["max_stable_horizon"],
        "checker_guarded_convergence_rate": checker["free_run_convergence_rate"],
        "checker_repair_rate": checker["checker_repair_rate"],
        "oracle_convergence_rate": oracle["free_run_convergence_rate"],
        "direct_to_target_final_target_accuracy": direct["final_target_accuracy"],
        "direct_to_target_per_step_transition_accuracy": direct["free_run_transition_accuracy"],
        "teacher_forced_vs_free_run_gap": tf_metrics["teacher_forced_transition_accuracy"] - learned["free_run_transition_accuracy"],
        "wall_clock_sec": round(time.time() - started, 3),
    }
    status, verdicts = summarize_verdict(metrics)
    write_summary_and_report(out, status, verdicts, metrics, run_rows)
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": "final verdict", "status": status, "verdicts": verdicts})
    return 0 if status == "positive" else 2


if __name__ == "__main__":
    raise SystemExit(main())
