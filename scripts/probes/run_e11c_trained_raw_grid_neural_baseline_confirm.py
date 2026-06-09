#!/usr/bin/env python3
"""E11C trained raw-grid neural baseline confirm over the E10 task family."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


MILESTONE = "E11C_TRAINED_RAW_GRID_NEURAL_BASELINE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/e11c_trained_raw_grid_neural_baseline_confirm")
DEFAULT_TRAIN_SEEDS = (130001, 130002, 130003, 130004)
DEFAULT_EVAL_SEEDS = (130101, 130102, 130103, 130104)
DEFAULT_TRAIN_ROWS_PER_SPLIT = 14
DEFAULT_EVAL_ROWS_PER_SPLIT = 18
DEFAULT_MLP_EPOCHS = 6
DEFAULT_SOFTMAX_EPOCHS = 8
E10_PATH = Path(__file__).with_name("run_e10_operator_library_transfer_noisy_route_confirm.py")

FLOW = "FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED"
ROUTE_ONLY = "OBSERVED_ROUTE_NO_TRAIN_BASELINE"
SOFTMAX = "TRAINED_RAW_GRID_ROUTE_SOFTMAX"
MLP = "TRAINED_RAW_GRID_ROUTE_MLP"
SYSTEMS = (FLOW, ROUTE_ONLY, SOFTMAX, MLP)
TRAINED_NEURAL_SYSTEMS = (SOFTMAX, MLP)
VALID_DECISIONS = (
    "e11c_flow_advantage_vs_trained_raw_grid_neural_confirmed",
    "e11c_trained_raw_grid_neural_baseline_not_quality_matched",
    "e11c_trained_raw_grid_neural_beats_or_matches_flow",
    "e11c_flow_quality_failure",
    "e11c_invalid_or_incomplete_run",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e11c_training_report.json",
    "e11c_quality_report.json",
    "e11c_cost_report.json",
    "e11c_split_report.json",
    "e11c_deterministic_replay_report.json",
)


def load_e10() -> Any:
    spec = importlib.util.spec_from_file_location("e10_operator_library_transfer_noisy_route_confirm", E10_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {E10_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E10 = load_e10()
SKILLS = tuple(E10.SKILLS)
SKILL_INDEX = {skill: idx for idx, skill in enumerate(SKILLS)}
MAX_ROUTE_LEN = 26


def rounded(value: float) -> float:
    return round(float(value), 6)


def rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return rounded(float(num) / float(den))


def stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_payload(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [stable_payload(item) for item in value]
    if isinstance(value, float):
        return rounded(value)
    return value


def stable_json(value: Any) -> str:
    return json.dumps(stable_payload(value), indent=2, sort_keys=True)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_git(args: list[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    except OSError as exc:
        return 127, str(exc)
    return done.returncode, done.stdout


def one_hot_skill(skill: str | None) -> list[float]:
    return [1.0 if skill == item else 0.0 for item in SKILLS] + [1.0 if skill is None else 0.0]


def route_token(row: Any, idx: int) -> str | None:
    if 0 <= idx < len(row.observed_route):
        return str(row.observed_route[idx])
    return None


def compact_features(grid: list[list[int]], row: Any, step_idx: int) -> list[float]:
    features: list[float] = []
    for grid_row in grid:
        features.extend(float(value) for value in grid_row)
    for offset in (-2, -1, 0, 1, 2):
        token = route_token(row, step_idx + offset)
        features.extend(one_hot_skill(token))
    features.append(step_idx / max(1, len(row.true_route) - 1))
    features.append(len(row.observed_route) / float(MAX_ROUTE_LEN))
    features.append(E10.grid_sum(grid) / float(E10.GRID * E10.GRID))
    return features


def full_route_features(grid: list[list[int]], row: Any, step_idx: int) -> list[float]:
    features: list[float] = []
    for grid_row in grid:
        features.extend(float(value) for value in grid_row)
    for pos in range(MAX_ROUTE_LEN):
        token = route_token(row, pos)
        current_bonus = 1.0 if pos == step_idx else 0.0
        for skill in SKILLS:
            features.append((1.0 if token == skill else 0.0) * (1.0 + current_bonus))
        features.append(1.0 if token is None else 0.0)
    for pos in range(MAX_ROUTE_LEN):
        features.append(1.0 if pos == step_idx else 0.0)
    features.append(step_idx / max(1, len(row.true_route) - 1))
    features.append(len(row.observed_route) / float(MAX_ROUTE_LEN))
    features.append(E10.grid_sum(grid) / float(E10.GRID * E10.GRID))
    return features


def build_examples(rows: list[Any], feature_fn: Any) -> list[tuple[list[float], int]]:
    examples: list[tuple[list[float], int]] = []
    for row in rows:
        current = E10.copy_grid(row.initial)
        for step_idx, true_skill in enumerate(row.true_route):
            examples.append((feature_fn(current, row, step_idx), SKILL_INDEX[str(true_skill)]))
            current = E10.apply_rule(current, E10.TRUE_RULES[str(true_skill)])
    return examples


def argmax(values: list[float]) -> int:
    best_idx = 0
    best_value = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_idx = idx
            best_value = value
    return best_idx


def softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


@dataclass
class SoftmaxModel:
    weights: list[list[float]]
    bias: list[float]
    feature_dim: int

    def predict_index(self, features: list[float]) -> int:
        logits = [self.bias[out] + sum(self.weights[out][idx] * value for idx, value in enumerate(features) if value) for out in range(len(SKILLS))]
        return argmax(logits)


@dataclass
class MlpModel:
    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]
    feature_dim: int
    hidden_dim: int

    def predict_index(self, features: list[float]) -> int:
        hidden = []
        for unit in range(self.hidden_dim):
            value = self.b1[unit] + sum(self.w1[unit][idx] * item for idx, item in enumerate(features))
            hidden.append(value if value > 0.0 else 0.0)
        logits = [self.b2[out] + sum(self.w2[out][unit] * hidden[unit] for unit in range(self.hidden_dim)) for out in range(len(SKILLS))]
        return argmax(logits)


def train_softmax(examples: list[tuple[list[float], int]], epochs: int, seed: int) -> tuple[SoftmaxModel, list[dict[str, float]]]:
    rng = random.Random(seed)
    feature_dim = len(examples[0][0])
    weights = [[rng.uniform(-0.01, 0.01) for _ in range(feature_dim)] for _ in SKILLS]
    bias = [0.0 for _ in SKILLS]
    history = []
    lr = 0.05
    l2 = 0.00001
    for epoch in range(epochs):
        rng.shuffle(examples)
        correct = 0
        loss = 0.0
        for features, label in examples:
            logits = [bias[out] + sum(weights[out][idx] * value for idx, value in enumerate(features) if value) for out in range(len(SKILLS))]
            probs = softmax(logits)
            correct += int(argmax(probs) == label)
            loss -= math.log(max(probs[label], 1e-12))
            for out in range(len(SKILLS)):
                delta = probs[out] - (1.0 if out == label else 0.0)
                bias[out] -= lr * delta
                row = weights[out]
                for idx, value in enumerate(features):
                    if value:
                        row[idx] -= lr * (delta * value + l2 * row[idx])
        history.append({"epoch": float(epoch), "train_accuracy": rate(correct, len(examples)), "train_loss": rate(loss, len(examples))})
    return SoftmaxModel(weights=weights, bias=bias, feature_dim=feature_dim), history


def train_mlp(examples: list[tuple[list[float], int]], epochs: int, seed: int, hidden_dim: int = 32) -> tuple[MlpModel, list[dict[str, float]]]:
    rng = random.Random(seed)
    feature_dim = len(examples[0][0])
    w1 = [[rng.uniform(-0.05, 0.05) for _ in range(feature_dim)] for _ in range(hidden_dim)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [[rng.uniform(-0.05, 0.05) for _ in range(hidden_dim)] for _ in SKILLS]
    b2 = [0.0 for _ in SKILLS]
    history = []
    lr = 0.012
    for epoch in range(epochs):
        rng.shuffle(examples)
        correct = 0
        loss = 0.0
        for features, label in examples:
            hidden = []
            for unit in range(hidden_dim):
                value = b1[unit] + sum(w1[unit][idx] * item for idx, item in enumerate(features))
                hidden.append(value if value > 0.0 else 0.0)
            logits = [b2[out] + sum(w2[out][unit] * hidden[unit] for unit in range(hidden_dim)) for out in range(len(SKILLS))]
            probs = softmax(logits)
            correct += int(argmax(probs) == label)
            loss -= math.log(max(probs[label], 1e-12))
            dz = probs[:]
            dz[label] -= 1.0
            dh = [0.0 for _ in range(hidden_dim)]
            for out in range(len(SKILLS)):
                delta = dz[out]
                b2[out] -= lr * delta
                row = w2[out]
                for unit in range(hidden_dim):
                    dh[unit] += delta * row[unit]
                    row[unit] -= lr * delta * hidden[unit]
            for unit in range(hidden_dim):
                if hidden[unit] <= 0.0:
                    continue
                delta = dh[unit]
                b1[unit] -= lr * delta
                row = w1[unit]
                for idx, item in enumerate(features):
                    if item:
                        row[idx] -= lr * delta * item
        history.append({"epoch": float(epoch), "train_accuracy": rate(correct, len(examples)), "train_loss": rate(loss, len(examples))})
    return MlpModel(w1=w1, b1=b1, w2=w2, b2=b2, feature_dim=feature_dim, hidden_dim=hidden_dim), history


@dataclass
class EvalStats:
    commits: int = 0
    accepted_good: int = 0
    accepted_bad: int = 0
    destructive: int = 0
    branch_contam: int = 0
    noisy_steps: int = 0
    route_repairs: int = 0
    route_false_repairs: int = 0
    transfer_steps: int = 0
    transfer_good: int = 0
    clean_steps: int = 0
    clean_good: int = 0
    oscillations: int = 0
    collapse: int = 0
    used_skills: set[str] | None = None

    def __post_init__(self) -> None:
        if self.used_skills is None:
            self.used_skills = set()


def model_predict(system: str, row: Any, step_idx: int, current: list[list[int]], models: dict[str, Any]) -> str:
    if system == ROUTE_ONLY:
        return str(E10.observed_skill_at(row, step_idx))
    if system == SOFTMAX:
        return SKILLS[models[SOFTMAX].predict_index(full_route_features(current, row, step_idx))]
    if system == MLP:
        return SKILLS[models[MLP].predict_index(compact_features(current, row, step_idx))]
    raise ValueError(system)


def evaluate_step_model(system: str, rows: list[Any], models: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    stats = EvalStats()
    split_stats = {split: EvalStats() for split in E10.SPLITS}
    row_metrics: list[dict[str, float]] = []
    split_rows: dict[str, list[dict[str, float]]] = {split: [] for split in E10.SPLITS}
    for row in rows:
        current = E10.copy_grid(row.initial)
        frames = [E10.copy_grid(current)]
        targets = (stats, split_stats[row.split])

        def add_stat(name: str, value: int) -> None:
            for target in targets:
                setattr(target, name, getattr(target, name) + value)

        for step_idx, true_skill in enumerate(row.true_route):
            observed = E10.observed_skill_at(row, step_idx)
            if observed != true_skill:
                add_stat("noisy_steps", 1)
            if row.split in E10.TRANSFER_SPLITS:
                add_stat("transfer_steps", 1)
            if observed == true_skill and row.split in {"validation", "heldout_transfer"}:
                add_stat("clean_steps", 1)
            predicted = model_predict(system, row, step_idx, current, models)
            good = predicted == true_skill
            before = E10.copy_grid(current)
            current = E10.apply_rule(current, E10.TRUE_RULES[predicted])
            oracle_after = row.oracle[min(step_idx + 1, len(row.oracle) - 1)]
            for target in targets:
                target.commits += 1
                target.accepted_good += int(good)
                target.accepted_bad += int(not good)
                if good:
                    target.used_skills.add(predicted)
                    if row.split in E10.TRANSFER_SPLITS:
                        target.transfer_good += 1
                    if observed == true_skill and row.split in {"validation", "heldout_transfer"}:
                        target.clean_good += 1
                if observed != true_skill and predicted != observed:
                    if good:
                        target.route_repairs += 1
                    else:
                        target.route_false_repairs += 1
                if E10.grid_similarity(before, oracle_after) > E10.grid_similarity(current, oracle_after) and not good:
                    target.destructive += 1
                if E10.grid_similarity(before, current) > 0.99 and E10.grid_similarity(current, oracle_after) < 0.95:
                    target.oscillations += 1
            frames.append(E10.copy_grid(current))
        if E10.grid_sum(current) in (0, E10.GRID * E10.GRID):
            add_stat("collapse", 1)
        metrics = E10.row_score(row, frames, system)
        row_metrics.append(metrics)
        split_rows[row.split].append(metrics)
    aggregate = aggregate_eval(row_metrics, stats, len(rows))
    splits = {split: aggregate_eval(split_rows[split], split_stats[split], max(1, len(split_rows[split]))) for split in E10.SPLITS}
    diagnostics = {
        "commits": stats.commits,
        "accepted_good": stats.accepted_good,
        "accepted_bad": stats.accepted_bad,
        "destructive_overwrites": stats.destructive,
        "branch_contamination": stats.branch_contam,
        "noisy_steps": stats.noisy_steps,
        "route_repairs": stats.route_repairs,
        "route_false_repairs": stats.route_false_repairs,
        "operator_skills_used": sorted(stats.used_skills),
    }
    return aggregate, diagnostics, splits


def aggregate_eval(rows: list[dict[str, float]], stats: EvalStats, row_count: int) -> dict[str, Any]:
    def mean(key: str) -> float:
        return rounded(sum(row[key] for row in rows) / max(1, len(rows)))

    tick_count = sum(row["route_length"] for row in rows)
    metrics = {
        "usefulness": mean("usefulness"),
        "answer_accuracy": mean("answer_accuracy"),
        "final_state_accuracy": mean("final_state_accuracy"),
        "trace_validity": mean("trace_validity"),
        "delta_validity": mean("delta_validity"),
        "observed_route_error_rate": mean("observed_route_error_rate"),
        "useful_writeback_recall": rate(stats.accepted_good, tick_count),
        "wrong_writeback_rate": rate(stats.accepted_bad, stats.commits),
        "destructive_overwrite_rate": rate(stats.destructive, stats.commits),
        "branch_contamination_rate": rate(stats.branch_contam, stats.commits),
        "route_repair_rate": rate(stats.route_repairs, stats.noisy_steps),
        "noisy_route_false_accept_rate": rate(stats.route_false_repairs, stats.noisy_steps),
        "transfer_coverage": rate(stats.transfer_good, stats.transfer_steps),
        "clean_route_preservation_rate": rate(stats.clean_good, stats.clean_steps),
        "operator_reuse_rate": rate(len(stats.used_skills or set()), len(SKILLS)),
        "temporal_drift_rate": mean("temporal_drift_rate"),
        "oscillation_rate": rate(stats.oscillations, row_count),
        "attractor_collapse_rate": rate(stats.collapse, row_count),
        "deterministic_replay_passed": True,
    }
    metrics["quality_matched"] = quality_matched(metrics)
    return metrics


def quality_matched(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["trace_validity"] >= 0.90
        and metrics["usefulness"] >= 0.85
        and metrics["useful_writeback_recall"] >= 0.85
        and metrics["wrong_writeback_rate"] <= 0.05
    )


def attach_cost(system: str, metrics: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    row = dict(metrics)
    row["quality_matched"] = quality_matched(row)
    if system == FLOW:
        ops = 200.0
        params = 0
        kind = "scheduled Flow block"
    elif system == ROUTE_ONLY:
        ops = 32.0
        params = 0
        kind = "observed-route control"
    elif system == SOFTMAX:
        feature_dim = models[SOFTMAX].feature_dim
        ops = float(feature_dim * len(SKILLS) + 64)
        params = feature_dim * len(SKILLS) + len(SKILLS)
        kind = "trained raw-grid/full-route softmax"
    elif system == MLP:
        feature_dim = models[MLP].feature_dim
        hidden_dim = models[MLP].hidden_dim
        ops = float(feature_dim * hidden_dim + hidden_dim * len(SKILLS) + 64)
        params = feature_dim * hidden_dim + hidden_dim + hidden_dim * len(SKILLS) + len(SKILLS)
        kind = "trained raw-grid/window-route MLP"
    else:
        raise ValueError(system)
    row["proxy_ops_per_tick"] = rounded(ops)
    row["parameter_count"] = int(params)
    row["cost_per_correct_trace"] = rate(ops, max(0.000001, float(row["trace_validity"])))
    row["cost_per_valid_writeback"] = rate(ops, max(0.000001, float(row["useful_writeback_recall"])))
    row["system_kind"] = kind
    return row


def positive_gate(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flow = metrics[FLOW]
    quality_neural = [system for system in TRAINED_NEURAL_SYSTEMS if metrics[system]["quality_matched"]]
    cheapest = min(quality_neural, key=lambda system: metrics[system]["proxy_ops_per_tick"]) if quality_neural else None
    checks = {
        "flow_quality_matched": flow["quality_matched"] is True,
        "flow_wrong_writeback_zero": flow["wrong_writeback_rate"] == 0.0,
        "flow_destructive_zero": flow["destructive_overwrite_rate"] == 0.0,
        "at_least_one_trained_raw_grid_neural_quality_matched": bool(quality_neural),
        "flow_cost_lower_than_cheapest_quality_neural": bool(cheapest and flow["proxy_ops_per_tick"] < metrics[cheapest]["proxy_ops_per_tick"]),
        "flow_cost_per_valid_writeback_lower_than_cheapest_quality_neural": bool(cheapest and flow["cost_per_valid_writeback"] < metrics[cheapest]["cost_per_valid_writeback"]),
        "no_detector_evidence_to_neural": True,
        "deterministic_replay_passed": True,
    }
    return {
        "schema_version": "e11c_positive_gate_v1",
        "checks": checks,
        "quality_matched_neural_systems": quality_neural,
        "cheapest_quality_matched_neural": cheapest,
        "passed": all(checks.values()),
    }


def decide(gate: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> str:
    if not metrics[FLOW]["quality_matched"]:
        return "e11c_flow_quality_failure"
    if gate["passed"]:
        return "e11c_flow_advantage_vs_trained_raw_grid_neural_confirmed"
    if not gate["checks"]["at_least_one_trained_raw_grid_neural_quality_matched"]:
        return "e11c_trained_raw_grid_neural_baseline_not_quality_matched"
    if any(metrics[system]["quality_matched"] and metrics[system]["cost_per_valid_writeback"] <= metrics[FLOW]["cost_per_valid_writeback"] for system in TRAINED_NEURAL_SYSTEMS):
        return "e11c_trained_raw_grid_neural_beats_or_matches_flow"
    return "e11c_invalid_or_incomplete_run"


def next_for(decision: str) -> str:
    return {
        "e11c_flow_advantage_vs_trained_raw_grid_neural_confirmed": "E12_RUST_BITPACKED_FLOW_LATENCY_BENCHMARK",
        "e11c_trained_raw_grid_neural_baseline_not_quality_matched": "E11C_STRONGER_RAW_GRID_NEURAL_BASELINE_OR_TASK_IDENTIFIABILITY_AUDIT",
        "e11c_trained_raw_grid_neural_beats_or_matches_flow": "E11C_FLOW_COST_MODEL_REPAIR_OR_NEURAL_BASELINE_AUDIT",
        "e11c_flow_quality_failure": "E10_FLOW_REGRESSION_REPAIR",
        "e11c_invalid_or_incomplete_run": "E11C_RETRY_WITH_FULL_AUDIT",
    }[decision]


def build_reports(train_seeds: tuple[int, ...], eval_seeds: tuple[int, ...], train_rows_per_split: int, eval_rows_per_split: int, softmax_epochs: int, mlp_epochs: int) -> dict[str, Any]:
    started = time.perf_counter()
    train_rows = E10.build_rows(train_seeds, train_rows_per_split)
    eval_rows = E10.build_rows(eval_seeds, eval_rows_per_split)
    softmax_examples = build_examples(train_rows, full_route_features)
    mlp_examples = build_examples(train_rows, compact_features)
    softmax_model, softmax_history = train_softmax(softmax_examples, softmax_epochs, seed=130401)
    mlp_model, mlp_history = train_mlp(mlp_examples, mlp_epochs, seed=130402)
    models = {SOFTMAX: softmax_model, MLP: mlp_model}
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    splits: dict[str, dict[str, dict[str, Any]]] = {}
    flow_metrics, flow_diag, _flow_samples = E10.run_system(E10.PRIMARY, eval_rows)
    metrics[FLOW] = attach_cost(FLOW, flow_metrics, models)
    diagnostics[FLOW] = {key: value for key, value in flow_diag.items() if key != "split_report"}
    splits[FLOW] = flow_diag["split_report"]
    for system in (ROUTE_ONLY, SOFTMAX, MLP):
        system_metrics, system_diag, split_report = evaluate_step_model(system, eval_rows, models)
        metrics[system] = attach_cost(system, system_metrics, models)
        diagnostics[system] = system_diag
        splits[system] = split_report
    gate = positive_gate(metrics)
    decision_label = decide(gate, metrics)
    decision = {
        "schema_version": "e11c_decision_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "next": next_for(decision_label),
        "primary_system": FLOW,
        "positive_gate_passed": gate["passed"],
        "deterministic_replay_passed": True,
    }
    aggregate = {
        "schema_version": "e11c_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "train_seeds": list(train_seeds),
        "eval_seeds": list(eval_seeds),
        "train_rows_per_split": train_rows_per_split,
        "eval_rows_per_split": eval_rows_per_split,
        "systems": metrics,
        "diagnostics": diagnostics,
        "split_metrics": splits,
        "positive_gate": gate,
    }
    training_report = {
        "schema_version": "e11c_training_report_v1",
        "dependency_mode": "stdlib-only backprop",
        "raw_grid_neural_training": True,
        "detector_evidence_used_by_neural": False,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "softmax_examples": len(softmax_examples),
        "mlp_examples": len(mlp_examples),
        "softmax_epochs": softmax_epochs,
        "mlp_epochs": mlp_epochs,
        "softmax_history": softmax_history,
        "mlp_history": mlp_history,
        "deterministic_elapsed_seconds": rounded(time.perf_counter() - started),
    }
    quality_report = {
        "schema_version": "e11c_quality_report_v1",
        "quality_matched_definition": "trace>=0.90 usefulness>=0.85 useful_writeback_recall>=0.85 wrong_writeback<=0.05",
        "systems": {
            system: {
                key: metrics[system][key]
                for key in (
                    "usefulness",
                    "trace_validity",
                    "answer_accuracy",
                    "useful_writeback_recall",
                    "wrong_writeback_rate",
                    "route_repair_rate",
                    "transfer_coverage",
                    "quality_matched",
                )
            }
            for system in SYSTEMS
        },
    }
    cost_report = {
        "schema_version": "e11c_cost_report_v1",
        "cost_units": "scalar proxy ops per tick; trained neural ops count dense scalar multiplies/adds plus fixed decoder cell ops",
        "systems": {
            system: {
                key: metrics[system][key]
                for key in (
                    "proxy_ops_per_tick",
                    "parameter_count",
                    "cost_per_correct_trace",
                    "cost_per_valid_writeback",
                    "system_kind",
                )
            }
            for system in SYSTEMS
        },
    }
    split_report = {"schema_version": "e11c_split_report_v1", "split_metrics": splits}
    summary = {
        "schema_version": "e11c_summary_v1",
        "milestone": MILESTONE,
        "decision": decision_label,
        "primary_system": FLOW,
        "positive_gate_passed": gate["passed"],
        "flow_trace_validity": metrics[FLOW]["trace_validity"],
        "flow_usefulness": metrics[FLOW]["usefulness"],
        "best_trained_raw_grid_neural_by_usefulness": max(TRAINED_NEURAL_SYSTEMS, key=lambda system: metrics[system]["usefulness"]),
        "quality_matched_neural_systems": gate["quality_matched_neural_systems"],
    }
    return {
        "decision.json": decision,
        "summary.json": summary,
        "aggregate_metrics.json": aggregate,
        "report.md": render_report(decision, aggregate),
        "e11c_training_report.json": training_report,
        "e11c_quality_report.json": quality_report,
        "e11c_cost_report.json": cost_report,
        "e11c_split_report.json": split_report,
    }


def render_report(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    metrics = aggregate["systems"]
    lines = [
        "# E11C Trained Raw-Grid Neural Baseline Confirm Report",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"next = {decision['next']}",
        f"primary_system = {decision['primary_system']}",
        f"positive_gate_passed = {decision['positive_gate_passed']}",
        f"deterministic_replay_passed = {decision['deterministic_replay_passed']}",
        "```",
        "",
        "## Quality And Cost",
        "",
        "| system | usefulness | trace | recall | wrong | repair | quality matched | ops/tick |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        row = metrics[system]
        lines.append(
            f"| {system} | {row['usefulness']:.3f} | {row['trace_validity']:.3f} | {row['useful_writeback_recall']:.3f} | "
            f"{row['wrong_writeback_rate']:.3f} | {row['route_repair_rate']:.3f} | {str(row['quality_matched']).lower()} | {row['proxy_ops_per_tick']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Positive Gate",
            "",
            "```json",
            json.dumps(stable_payload(aggregate["positive_gate"]["checks"]), indent=2, sort_keys=True),
            "```",
            "",
            "## Boundary",
            "",
            "This is a stdlib-only trained raw-grid neural baseline probe over the E10 synthetic task family.",
        ]
    )
    return "\n".join(lines)


def attach_replay(payloads: dict[str, Any], train_seeds: tuple[int, ...], eval_seeds: tuple[int, ...], train_rows_per_split: int, eval_rows_per_split: int, softmax_epochs: int, mlp_epochs: int) -> dict[str, Any]:
    replay_a = build_reports(train_seeds, eval_seeds, train_rows_per_split, eval_rows_per_split, softmax_epochs, mlp_epochs)
    replay_b = build_reports(train_seeds, eval_seeds, train_rows_per_split, eval_rows_per_split, softmax_epochs, mlp_epochs)
    # Timing is diagnostic only and intentionally excluded from replay hashes.
    for replay in (replay_a, replay_b):
        replay["e11c_training_report.json"]["deterministic_elapsed_seconds"] = 0.0
    payload_for_hash = dict(payloads)
    payload_for_hash["e11c_training_report.json"] = dict(payload_for_hash["e11c_training_report.json"])
    payload_for_hash["e11c_training_report.json"]["deterministic_elapsed_seconds"] = 0.0
    hash_a = stable_hash(replay_a)
    hash_b = stable_hash(replay_b)
    hash_primary = stable_hash(payload_for_hash)
    passed = hash_a == hash_b == hash_primary
    payloads["e11c_deterministic_replay_report.json"] = {
        "schema_version": "e11c_deterministic_replay_report_v1",
        "internal_replay_passed": passed,
        "hash_primary": hash_primary,
        "hash_a": hash_a,
        "hash_b": hash_b,
        "artifact_set": sorted(replay_a),
        "elapsed_seconds_excluded_from_replay_hash": True,
    }
    payloads["decision.json"]["deterministic_replay_passed"] = passed
    payloads["aggregate_metrics.json"]["positive_gate"]["checks"]["deterministic_replay_passed"] = passed
    payloads["aggregate_metrics.json"]["positive_gate"]["passed"] = all(payloads["aggregate_metrics.json"]["positive_gate"]["checks"].values())
    decision_label = decide(payloads["aggregate_metrics.json"]["positive_gate"], payloads["aggregate_metrics.json"]["systems"])
    payloads["decision.json"]["decision"] = decision_label
    payloads["decision.json"]["next"] = next_for(decision_label)
    payloads["decision.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["decision"] = decision_label
    payloads["summary.json"]["positive_gate_passed"] = payloads["aggregate_metrics.json"]["positive_gate"]["passed"]
    payloads["summary.json"]["deterministic_replay_passed"] = passed
    payloads["report.md"] = render_report(payloads["decision.json"], payloads["aggregate_metrics.json"])
    return payloads


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--train-seeds", type=parse_seeds, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--eval-seeds", type=parse_seeds, default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--train-rows-per-split", type=int, default=DEFAULT_TRAIN_ROWS_PER_SPLIT)
    parser.add_argument("--eval-rows-per-split", type=int, default=DEFAULT_EVAL_ROWS_PER_SPLIT)
    parser.add_argument("--softmax-epochs", type=int, default=DEFAULT_SOFTMAX_EPOCHS)
    parser.add_argument("--mlp-epochs", type=int, default=DEFAULT_MLP_EPOCHS)
    args = parser.parse_args(argv)
    out = Path(args.out)
    git_rc, git_head = run_git(["rev-parse", "--short", "HEAD"])
    payloads = build_reports(args.train_seeds, args.eval_seeds, args.train_rows_per_split, args.eval_rows_per_split, args.softmax_epochs, args.mlp_epochs)
    payloads = attach_replay(payloads, args.train_seeds, args.eval_seeds, args.train_rows_per_split, args.eval_rows_per_split, args.softmax_epochs, args.mlp_epochs)
    payloads["summary.json"]["git_head"] = git_head.strip() if git_rc == 0 else "unknown"
    for name in REQUIRED_ARTIFACTS:
        path = out / name
        if name.endswith(".md"):
            write_text(path, str(payloads[name]))
        else:
            write_json(path, payloads[name])
    print(stable_json({"out": str(out), "decision": payloads["decision.json"]["decision"], "positive_gate_passed": payloads["decision.json"]["positive_gate_passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
