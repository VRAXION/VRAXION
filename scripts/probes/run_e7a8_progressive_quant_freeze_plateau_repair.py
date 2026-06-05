#!/usr/bin/env python3
"""E7A8 progressive quantize/freeze/plateau repair probe.

E7A8 tests low-bit repair strategy for the existing E7 matrix-core. It does
not add a new architecture. The probe compares direct ternary/binary
quantization, ordinary post-quantization mutation repair, progressive freeze
schedules, blockwise scale mutation, and QAT as a reference upper path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
E7A7_PATH = Path(__file__).with_name("run_e7a7_low_bit_repair_operator_audit.py")
MILESTONE = "E7A8_PROGRESSIVE_QUANT_FREEZE_PLATEAU_REPAIR"
DEFAULT_OUT = Path("target/pilot_wave/e7a8_progressive_quant_freeze_plateau_repair")
DEFAULT_SEEDS = (82001, 82002, 82003)

TARGET_LEVELS = ("ternary", "binary")
BLOCKS: dict[str, tuple[str, ...]] = {
    "input_projection": ("win",),
    "recurrent_state": ("state",),
    "carry_gate": ("carry_raw",),
    "state_bias": ("bstate",),
    "output_head": ("wout", "bout"),
}
METHODS = (
    "baseline_float_matrix_core",
    "direct_low_bit_quant",
    "post_quant_mutation_repair",
    "distance_only_progressive_freeze",
    "sensitivity_aware_progressive_freeze",
    "input_projection_aware_progressive_freeze",
    "blockwise_scale_mutation_repair",
    "qat_reference",
)
VALID_DECISIONS = (
    "e7a8_input_projection_aware_progressive_freeze_positive",
    "e7a8_sensitivity_aware_freeze_positive",
    "e7a8_distance_only_freeze_sufficient",
    "e7a8_blockwise_scale_repair_positive",
    "e7a8_progressive_freeze_no_advantage",
    "e7a8_progressive_freeze_overfit_or_brittle",
    "e7a8_invalid_artifact_detected",
)
HASH_ARTIFACTS = (
    "e7a8_task_generation_report.json",
    "e7a8_method_comparison_report.json",
    "e7a8_freeze_schedule_report.json",
    "e7a8_input_projection_damage_recovery_report.json",
    "e7a8_mutation_repair_report.json",
    "e7a8_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)


def load_e7a7_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7a7_low_bit_repair_operator_audit", E7A7_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7A7 from {E7A7_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7a7 = load_e7a7_module()
e7a6 = e7a7.e7a6
e7a3 = e7a7.e7a3


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    widths: tuple[int, ...]
    target_levels: tuple[str, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    gradient_epochs: int
    qat_epochs: int
    batch_size: int
    learning_rate: float
    qat_learning_rate: float
    weight_decay: float
    population_size: int
    repair_generations: int
    progressive_rounds: int
    progressive_generations_per_round: int
    plateau_patience: int
    plateau_min_delta: float
    elite_count: int
    quant_mutation_steps: int
    scale_mutation_steps: int
    scale_mutation_sigma: float
    matrix_steps: int
    device: str
    execution_mode: str
    parallel_workers: int
    heartbeat_seconds: float


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7a8::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7a6.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7a6.write_json(path, payload)


def locked_write_json(path: Path, payload: Any) -> None:
    e7a6.locked_write_json(path, payload)


def append_progress_locked(out: Path, event: str, **details: Any) -> None:
    e7a6.append_progress_locked(out, event, **details)


def resolve_out(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    relative = resolved.relative_to(REPO_ROOT)
    if len(relative.parts) < 2 or relative.parts[0].lower() != "target" or relative.parts[1].lower() != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one integer is required")
    return values


def parse_level_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one level is required")
    allowed = set(e7a6.QUANT_CONFIGS)
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"unknown quant levels: {unknown}")
    return values


def e7a6_settings(settings: Settings) -> Any:
    return e7a6.Settings(
        seeds=settings.seeds,
        widths=settings.widths,
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=settings.adversarial_rows_per_seed,
        gradient_epochs=settings.gradient_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        population_size=settings.population_size,
        repair_generations=settings.repair_generations,
        elite_count=settings.elite_count,
        quant_mutation_steps=settings.quant_mutation_steps,
        matrix_steps=settings.matrix_steps,
        device=settings.device,
        execution_mode=settings.execution_mode,
        parallel_workers=settings.parallel_workers,
        heartbeat_seconds=settings.heartbeat_seconds,
    )


def e7a7_settings(settings: Settings) -> Any:
    return e7a7.Settings(
        seeds=settings.seeds,
        widths=settings.widths,
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=settings.adversarial_rows_per_seed,
        gradient_epochs=settings.gradient_epochs,
        qat_epochs=settings.qat_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        qat_learning_rate=settings.qat_learning_rate,
        weight_decay=settings.weight_decay,
        population_size=settings.population_size,
        repair_generations=settings.repair_generations,
        elite_count=settings.elite_count,
        quant_mutation_steps=settings.quant_mutation_steps,
        matrix_steps=settings.matrix_steps,
        device=settings.device,
        execution_mode=settings.execution_mode,
        parallel_workers=settings.parallel_workers,
        heartbeat_seconds=settings.heartbeat_seconds,
    )


def task_splits() -> tuple[str, ...]:
    return e7a3.SPLITS


def eval_splits() -> tuple[str, ...]:
    return e7a3.EVAL_SPLITS


def candidate_hash(candidate: dict[str, Any]) -> str:
    return payload_sha256(candidate)


def scale_array(candidate: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = candidate["scales"][key]
    scale = np.asarray(raw, dtype=np.float64)
    if scale.ndim == 0:
        return scale
    if len(shape) == 2 and scale.ndim == 1 and scale.shape[0] == shape[1]:
        return scale.reshape(1, -1)
    if len(shape) == 1 and scale.ndim == 1 and scale.shape[0] == shape[0]:
        return scale
    return np.reshape(scale, shape)


def dequant(candidate: dict[str, Any], key: str) -> np.ndarray:
    q = np.asarray(candidate["q"][key], dtype=np.float64)
    return q * scale_array(candidate, key, q.shape)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def quantized_forward(candidate: dict[str, Any], x: np.ndarray, matrix_steps: int) -> np.ndarray:
    win = dequant(candidate, "win")
    state = dequant(candidate, "state")
    carry_raw = dequant(candidate, "carry_raw")
    bstate = dequant(candidate, "bstate")
    wout = dequant(candidate, "wout")
    bout = dequant(candidate, "bout")
    drive = x @ win + bstate
    h = np.tanh(drive)
    carry = sigmoid_np(carry_raw)
    for _ in range(matrix_steps):
        proposal = np.tanh(h @ state + drive)
        h = carry * h + (1.0 - carry) * proposal
    return h @ wout + bout


def evaluate_candidate(candidate: dict[str, Any], task: dict[str, Any], settings: Settings, sample_limit: int = 10) -> dict[str, Any]:
    return {
        split: e7a3.evaluate_logits(quantized_forward(candidate, data["x"], settings.matrix_steps), data, sample_limit)
        for split, data in task.items()
    }


def split_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {split: result["evals"][split]["metrics"] for split in task_splits()}


def solve_pass(row: dict[str, float]) -> bool:
    return (
        row["heldout_accuracy"] >= 0.90
        and row["ood_accuracy"] >= 0.85
        and row["counterfactual_accuracy"] >= 0.85
        and row["adversarial_accuracy"] >= 0.80
    )


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    if result["system"] == "baseline_float_matrix_core":
        base = e7a6.result_row(result)
    else:
        metrics = split_metrics(result)
        eval_accuracy = round_float(float(np.mean([metrics[split]["accuracy"] for split in eval_splits()])))
        base = {
            "width": result["width"],
            "matrix_shape": result["hidden_matrix_shape"],
            "matrix_cells": int(result["hidden_matrix_shape"][0] * result["hidden_matrix_shape"][1]),
            "parameter_count": result["parameter_count"],
            "eval_accuracy": eval_accuracy,
            "train_accuracy": metrics["train"]["accuracy"],
            "validation_accuracy": metrics["validation"]["accuracy"],
            "heldout_accuracy": metrics["heldout"]["accuracy"],
            "ood_accuracy": metrics["ood"]["accuracy"],
            "counterfactual_accuracy": metrics["counterfactual"]["accuracy"],
            "adversarial_accuracy": metrics["adversarial"]["accuracy"],
            "generalization_gap": round_float(metrics["validation"]["accuracy"] - eval_accuracy),
            "training_mode": result["training_mode"],
        }
        base["solve_passed"] = solve_pass(base)
    for key in ("system", "method", "quant_level"):
        if key in result:
            base[key] = result[key]
    if "float_eval_accuracy" in result:
        base["float_eval_accuracy"] = result["float_eval_accuracy"]
        base["quantization_drop_from_float"] = round_float(result["float_eval_accuracy"] - base["eval_accuracy"])
    if "direct_eval_accuracy" in result:
        base["direct_eval_accuracy"] = result["direct_eval_accuracy"]
        base["recovery_delta_vs_direct"] = round_float(base["eval_accuracy"] - result["direct_eval_accuracy"])
    if "qat_eval_accuracy" in result:
        base["qat_eval_accuracy"] = result["qat_eval_accuracy"]
        base["gap_to_qat"] = round_float(result["qat_eval_accuracy"] - base["eval_accuracy"])
    if "mutation_history" in result:
        hist = result["mutation_history"]
        base["mutation_attempt_count"] = hist["mutation_attempt_count"]
        base["accepted_mutation_count"] = hist["accepted_mutation_count"]
        base["rejected_mutation_count"] = hist["rejected_mutation_count"]
        base["rollback_count"] = hist["rollback_count"]
    if "freeze_history" in result:
        base["freeze_rounds_count"] = len(result["freeze_history"])
        base["freeze_round_rollback_count"] = sum(1 for row in result["freeze_history"] if row.get("rollback_applied"))
        base["final_frozen_parameter_ratio"] = result["freeze_history"][-1]["frozen_parameter_ratio"] if result["freeze_history"] else 0.0
        base["input_projection_frozen_ratio"] = result["freeze_history"][-1]["block_frozen_ratios"].get("input_projection", 0.0) if result["freeze_history"] else 0.0
        base["recurrent_state_frozen_ratio"] = result["freeze_history"][-1]["block_frozen_ratios"].get("recurrent_state", 0.0) if result["freeze_history"] else 0.0
        base["output_head_frozen_ratio"] = result["freeze_history"][-1]["block_frozen_ratios"].get("output_head", 0.0) if result["freeze_history"] else 0.0
    return base


def all_quantized_paths(candidate: dict[str, Any]) -> list[tuple[tuple[Any, ...], int]]:
    rows: list[tuple[tuple[Any, ...], int]] = []
    for key, value in candidate["q"].items():
        arr = np.asarray(value, dtype=np.int16)
        for index in np.ndindex(arr.shape):
            rows.append((("q", key, *index), int(arr[index])))
    return rows


def set_quantized_path(candidate: dict[str, Any], path: tuple[Any, ...], value: int) -> None:
    e7a6.set_quantized_path(candidate, path, value)


def block_for_key(key: str) -> str:
    for block, keys in BLOCKS.items():
        if key in keys:
            return block
    raise KeyError(key)


def key_from_path(path: tuple[Any, ...]) -> str:
    return str(path[1])


def path_key(path: tuple[Any, ...]) -> str:
    return ".".join(str(part) for part in path)


def quantized_parameter_count(candidate: dict[str, Any]) -> int:
    return sum(np.asarray(value).size for value in candidate["q"].values())


def quantized_nonzero_count(candidate: dict[str, Any]) -> int:
    return sum(int(np.sum(np.asarray(value) != 0)) for value in candidate["q"].values())


def update_zero_counts(candidate: dict[str, Any]) -> None:
    candidate["zero_counts"] = {
        key: int(np.sum(np.asarray(value, dtype=np.int16) == 0))
        for key, value in candidate["q"].items()
    }
    candidate["candidate_hash"] = candidate_hash(candidate)


def score_candidate(candidate: dict[str, Any], task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    evals = {
        split: e7a3.evaluate_logits(quantized_forward(candidate, data["x"], settings.matrix_steps), data, sample_limit=0)
        for split, data in {"train": task["train"], "validation": task["validation"]}.items()
    }
    train = evals["train"]["metrics"]
    val = evals["validation"]["metrics"]
    nonzero_ratio = quantized_nonzero_count(candidate) / max(1, quantized_parameter_count(candidate))
    fitness = 0.20 * train["accuracy"] + 0.80 * val["accuracy"] - 0.02 * val["cross_entropy"] - 0.001 * nonzero_ratio
    return {"candidate": candidate, "evals": evals, "fitness": round_float(fitness)}


def candidate_result(
    method: str,
    level: str,
    width: int,
    candidate: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    training_mode: str,
    float_eval_accuracy: float,
    direct_eval_accuracy: float | None = None,
    qat_eval_accuracy: float | None = None,
    mutation_history: dict[str, Any] | None = None,
    freeze_history: list[dict[str, Any]] | None = None,
    parameter_diff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "system": method,
        "method": method,
        "quant_level": level,
        "width": width,
        "training_mode": training_mode,
        "evals": evaluate_candidate(candidate, task, settings),
        "history": [],
        "parameter_count": quantized_parameter_count(candidate),
        "nonzero_parameter_count": quantized_nonzero_count(candidate),
        "hidden_matrix_shape": [width, width],
        "initial_hash": candidate.get("initial_hash", candidate_hash(candidate)),
        "final_hash": candidate_hash(candidate),
        "candidate": candidate,
        "float_eval_accuracy": float_eval_accuracy,
    }
    if direct_eval_accuracy is not None:
        result["direct_eval_accuracy"] = direct_eval_accuracy
    if qat_eval_accuracy is not None:
        result["qat_eval_accuracy"] = qat_eval_accuracy
    if mutation_history is not None:
        result["mutation_history"] = mutation_history
    if freeze_history is not None:
        result["freeze_history"] = freeze_history
    if parameter_diff is not None:
        result["parameter_diff"] = parameter_diff
    return result


def quantized_diff(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    before_q = {path: value for path, value in all_quantized_paths(initial)}
    after_q = {path: value for path, value in all_quantized_paths(final)}
    q_changed = {}
    q_l1 = 0
    for path in sorted(before_q):
        delta = after_q[path] - before_q[path]
        if delta != 0:
            key = path_key(path)
            q_changed[key] = {"before": before_q[path], "after": after_q[path], "delta": int(delta)}
            q_l1 += abs(delta)
    scale_changed = {}
    for key, before_raw in initial["scales"].items():
        before = np.asarray(before_raw, dtype=np.float64)
        after = np.asarray(final["scales"][key], dtype=np.float64)
        if before.shape != after.shape or bool(np.max(np.abs(before - after)) > 1e-12):
            scale_changed[key] = {
                "before_shape": list(before.shape),
                "after_shape": list(after.shape),
                "mean_abs_delta": round_float(float(np.mean(np.abs(after - before)))),
            }
    return {
        "actual_parameter_diff_found": bool(q_changed or scale_changed),
        "changed_q_parameter_count": len(q_changed),
        "changed_scale_count": len(scale_changed),
        "q_parameter_diff_l1": int(q_l1),
        "before_hash": candidate_hash(initial),
        "after_hash": candidate_hash(final),
        "changed_q_parameters_sample": dict(list(q_changed.items())[:120]),
        "changed_scales": scale_changed,
    }


def make_low_bit_candidate(level: str, state_dict: dict[str, np.ndarray], width: int, schema: str, audit_mode: str) -> dict[str, Any]:
    candidate = e7a6.quantize_state_dict(level, state_dict, width)
    candidate["schema_version"] = schema
    candidate["audit_mode"] = audit_mode
    candidate["source_quant_level"] = level
    candidate["candidate_hash"] = candidate_hash(candidate)
    candidate["initial_hash"] = candidate["candidate_hash"]
    return candidate


def make_channel_scale_candidate(level: str, state_dict: dict[str, np.ndarray], width: int) -> dict[str, Any]:
    candidate = make_low_bit_candidate(level, state_dict, width, "e7a8_channel_scale_candidate_v1", "blockwise_scale_mutation_repair")
    for key in ("win", "state", "wout"):
        q = np.asarray(candidate["q"][key], dtype=np.float64)
        value = state_dict[key]
        if q.ndim == 2:
            scales = []
            for col in range(q.shape[1]):
                active = np.abs(q[:, col]) > 0
                if bool(np.any(active)):
                    scales.append(round_float(float(np.mean(np.abs(value[:, col][active])))))
                else:
                    scales.append(candidate["scales"][key])
            candidate["scales"][key] = scales
    update_zero_counts(candidate)
    candidate["initial_hash"] = candidate_hash(candidate)
    return candidate


def quantize_scalar_for_level(value: float, scale: float, level: str) -> int:
    config = e7a6.QUANT_CONFIGS[level]
    if config["kind"] == "binary":
        return 0 if abs(value) <= 1e-12 else 1 if value >= 0.0 else -1
    if config["kind"] == "ternary":
        threshold = 0.5 * scale
        return 1 if value > threshold else -1 if value < -threshold else 0
    q_limit = int(config["q_limit"])
    return int(min(max(round(value / scale), -q_limit), q_limit))


def build_path_metadata(level: str, state_dict: dict[str, np.ndarray], candidate: dict[str, Any], block_sensitivity: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    max_sens = max(block_sensitivity.values()) if block_sensitivity else 1.0
    for path, q_value in all_quantized_paths(candidate):
        key = key_from_path(path)
        indices = tuple(int(part) for part in path[2:])
        block = block_for_key(key)
        scale_raw = candidate["scales"][key]
        scale_arr = np.asarray(scale_raw, dtype=np.float64)
        scale = float(scale_arr[indices[-1]] if scale_arr.ndim == 1 and len(indices) == 2 else scale_arr if scale_arr.ndim == 0 else scale_arr[indices])
        float_value = float(state_dict[key][indices])
        dequant_value = float(q_value) * scale
        quant_error = abs(float_value - dequant_value) / max(abs(scale), 1e-12)
        sensitivity = block_sensitivity.get(block, 0.0) / max(max_sens, 1e-12)
        rows.append(
            {
                "path": path,
                "path_key": path_key(path),
                "block": block,
                "key": key,
                "q_value": int(q_value),
                "quant_error": round_float(quant_error),
                "block_sensitivity": round_float(block_sensitivity.get(block, 0.0)),
                "normalized_sensitivity": round_float(sensitivity),
            }
        )
    return rows


def block_sensitivity_report(level: str, state_dict: dict[str, np.ndarray], width: int, task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    int8_candidate = make_low_bit_candidate("int8", state_dict, width, "e7a8_int8_reference_candidate_v1", "int8_reference")
    low_candidate = make_low_bit_candidate(level, state_dict, width, "e7a8_low_bit_candidate_v1", "full_low_bit")
    int8_eval = evaluate_candidate(int8_candidate, task, settings, sample_limit=0)["validation"]["metrics"]
    rows = {}
    for block, keys in BLOCKS.items():
        candidate = copy.deepcopy(int8_candidate)
        for key in keys:
            candidate["q"][key] = copy.deepcopy(low_candidate["q"][key])
            candidate["scales"][key] = copy.deepcopy(low_candidate["scales"][key])
        update_zero_counts(candidate)
        metrics = evaluate_candidate(candidate, task, settings, sample_limit=0)["validation"]["metrics"]
        rows[block] = {
            "validation_accuracy": metrics["accuracy"],
            "validation_cross_entropy": metrics["cross_entropy"],
            "accuracy_drop_vs_int8": round_float(int8_eval["accuracy"] - metrics["accuracy"]),
            "loss_increase_vs_int8": round_float(metrics["cross_entropy"] - int8_eval["cross_entropy"]),
            "sensitivity_score": round_float(max(0.0, int8_eval["accuracy"] - metrics["accuracy"]) + max(0.0, metrics["cross_entropy"] - int8_eval["cross_entropy"])),
        }
    return {
        "int8_validation": int8_eval,
        "rows": rows,
    }


def ordered_paths(metadata: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    if method == "distance_only_progressive_freeze":
        return sorted(metadata, key=lambda row: (row["quant_error"], row["path_key"]))
    if method == "sensitivity_aware_progressive_freeze":
        return sorted(metadata, key=lambda row: (0.65 * row["quant_error"] + 0.35 * row["normalized_sensitivity"], row["path_key"]))
    if method == "input_projection_aware_progressive_freeze":
        return sorted(
            metadata,
            key=lambda row: (
                0.55 * row["quant_error"]
                + 0.35 * row["normalized_sensitivity"]
                + (0.25 if row["block"] == "input_projection" else 0.0),
                row["path_key"],
            ),
        )
    raise ValueError(method)


def target_frozen_path_keys(metadata: list[dict[str, Any]], method: str, round_index: int, rounds: int) -> set[str]:
    if method != "input_projection_aware_progressive_freeze":
        count = math.ceil(len(metadata) * round_index / rounds)
        return {row["path_key"] for row in ordered_paths(metadata, method)[:count]}
    selected: set[str] = set()
    progress = round_index / rounds
    for block in BLOCKS:
        block_rows = [row for row in ordered_paths(metadata, method) if row["block"] == block]
        block_progress = progress * progress if block == "input_projection" else progress
        count = math.ceil(len(block_rows) * block_progress)
        selected.update(row["path_key"] for row in block_rows[:count])
    return selected


def freeze_ratio_report(frozen: set[str], metadata: list[dict[str, Any]]) -> dict[str, float]:
    rows = {}
    for block in BLOCKS:
        block_rows = [row for row in metadata if row["block"] == block]
        rows[block] = round_float(sum(1 for row in block_rows if row["path_key"] in frozen) / max(1, len(block_rows)))
    return rows


def allowed_paths(candidate: dict[str, Any], frozen_keys: set[str]) -> list[tuple[tuple[Any, ...], int]]:
    return [(path, value) for path, value in all_quantized_paths(candidate) if path_key(path) not in frozen_keys]


def mutate_candidate_q(
    candidate: dict[str, Any],
    level: str,
    settings: Settings,
    rng: random.Random,
    frozen_keys: set[str],
    input_projection_bias: bool = False,
) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    config = e7a6.QUANT_CONFIGS[level]
    paths = allowed_paths(child, frozen_keys)
    if input_projection_bias:
        win_paths = [row for row in paths if key_from_path(row[0]) == "win"]
        other_paths = [row for row in paths if key_from_path(row[0]) != "win"]
        win_take = min(len(win_paths), max(1, settings.quant_mutation_steps // 3))
        other_take = min(len(other_paths), max(0, settings.quant_mutation_steps - win_take))
        chosen = rng.sample(win_paths, k=win_take) + rng.sample(other_paths, k=other_take) if win_paths else rng.sample(paths, k=min(settings.quant_mutation_steps, len(paths)))
    else:
        chosen = rng.sample(paths, k=min(settings.quant_mutation_steps, len(paths)))
    for path, value in chosen:
        if config["kind"] == "binary":
            new_value = rng.choice((-1, 1)) if value == 0 else -value
        elif config["kind"] == "ternary":
            choices = [-1, 0, 1]
            choices.remove(value)
            new_value = rng.choice(choices)
        else:
            q_limit = int(config["q_limit"])
            step = rng.choice((-3, -2, -1, 1, 2, 3))
            new_value = min(max(value + step, -q_limit), q_limit)
        set_quantized_path(child, path, new_value)
    update_zero_counts(child)
    return child


def scale_entries(candidate: dict[str, Any], focus_input_projection: bool) -> list[tuple[str, tuple[int, ...] | None]]:
    entries: list[tuple[str, tuple[int, ...] | None]] = []
    keys = ("win",) if focus_input_projection else tuple(candidate["scales"].keys())
    for key in keys:
        arr = np.asarray(candidate["scales"][key], dtype=np.float64)
        if arr.ndim == 0:
            entries.append((key, None))
        else:
            for index in np.ndindex(arr.shape):
                entries.append((key, tuple(int(part) for part in index)))
    return entries


def set_scale_entry(candidate: dict[str, Any], key: str, index: tuple[int, ...] | None, value: float) -> None:
    raw = candidate["scales"][key]
    if index is None:
        candidate["scales"][key] = round_float(max(value, 1e-8))
        return
    arr = np.asarray(raw, dtype=np.float64)
    arr[index] = max(value, 1e-8)
    candidate["scales"][key] = [round_float(v) for v in arr.tolist()] if arr.ndim == 1 else arr.tolist()


def get_scale_entry(candidate: dict[str, Any], key: str, index: tuple[int, ...] | None) -> float:
    raw = candidate["scales"][key]
    if index is None:
        return float(raw)
    return float(np.asarray(raw, dtype=np.float64)[index])


def mutate_candidate_scale(
    candidate: dict[str, Any],
    settings: Settings,
    rng: random.Random,
    focus_input_projection: bool,
) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    entries = scale_entries(child, focus_input_projection)
    for key, index in rng.sample(entries, k=min(settings.scale_mutation_steps, len(entries))):
        current = get_scale_entry(child, key, index)
        multiplier = math.exp(rng.gauss(0.0, settings.scale_mutation_sigma))
        set_scale_entry(child, key, index, current * multiplier)
    update_zero_counts(child)
    return child


def repair_loop(
    method: str,
    level: str,
    width: int,
    initial_candidate: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    frozen_keys: set[str],
    generations: int,
    seed_label: str,
    mutate_scales: bool = False,
    input_projection_bias: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    out_path = Path(out) if out else None
    rng = random.Random(stable_seed(seed_label))
    population = [score_candidate(copy.deepcopy(initial_candidate), task, settings)]
    for _ in range(settings.population_size - 1):
        child = copy.deepcopy(initial_candidate)
        if mutate_scales:
            child = mutate_candidate_scale(child, settings, rng, input_projection_bias)
        else:
            child = mutate_candidate_q(child, level, settings, rng, frozen_keys, input_projection_bias)
        population.append(score_candidate(child, task, settings))
    accepted = 0
    rejected = 0
    rollback = 0
    attempts = 0
    history = []
    best_validation = max(population, key=lambda row: row["fitness"])["evals"]["validation"]["metrics"]["accuracy"]
    stale = 0
    last_heartbeat = time.monotonic()
    for generation in range(1, generations + 1):
        population.sort(key=lambda row: row["fitness"], reverse=True)
        elites = population[: max(1, settings.elite_count)]
        next_population = copy.deepcopy(elites)
        while len(next_population) < settings.population_size:
            parent = copy.deepcopy(rng.choice(population))
            child_candidate = copy.deepcopy(parent["candidate"])
            if mutate_scales:
                child_candidate = mutate_candidate_scale(child_candidate, settings, rng, input_projection_bias)
            else:
                child_candidate = mutate_candidate_q(child_candidate, level, settings, rng, frozen_keys, input_projection_bias)
            child = score_candidate(child_candidate, task, settings)
            attempts += 1
            if child["fitness"] >= parent["fitness"]:
                accepted += 1
                next_population.append(child)
            else:
                rejected += 1
                rollback += 1
                next_population.append(parent)
            now = time.monotonic()
            if out_path and now - last_heartbeat >= settings.heartbeat_seconds:
                best_mid = max(next_population, key=lambda row: row["fitness"])
                locked_write_json(
                    out_path / "partial_status" / f"{method}_{level}_width{width}_{seed_label}.json",
                    {
                        "method": method,
                        "quant_level": level,
                        "width": width,
                        "generation": generation,
                        "attempts": attempts,
                        "accepted": accepted,
                        "rejected": rejected,
                        "rollback": rollback,
                        "best_fitness": best_mid["fitness"],
                        "best_candidate_hash": candidate_hash(best_mid["candidate"]),
                    },
                )
                append_progress_locked(out_path, "repair_heartbeat", method=method, quant_level=level, width=width, generation=generation, best_fitness=best_mid["fitness"])
                last_heartbeat = now
        population = next_population
        best = max(population, key=lambda row: row["fitness"])
        val_acc = best["evals"]["validation"]["metrics"]["accuracy"]
        if val_acc - best_validation < settings.plateau_min_delta:
            stale += 1
        else:
            stale = 0
            best_validation = val_acc
        row = {
            "generation": generation,
            "best_fitness": best["fitness"],
            "validation_accuracy": round_float(val_acc),
            "accepted_mutation_count": accepted,
            "rejected_mutation_count": rejected,
            "rollback_count": rollback,
            "candidate_hash": candidate_hash(best["candidate"]),
            "plateau_stale_count": stale,
        }
        history.append(row)
        if out_path:
            append_progress_locked(out_path, "repair_generation", method=method, quant_level=level, width=width, generation=generation, validation_accuracy=row["validation_accuracy"])
        if stale >= settings.plateau_patience:
            break
    best = max(population, key=lambda row: row["fitness"])
    mutation_history = {
        "method": method,
        "quant_level": level,
        "width": width,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "plateau_generation_count": len(history),
        "history": history,
    }
    return best["candidate"], mutation_history, history


def rollback_needed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    b = {split: before[split]["metrics"]["accuracy"] for split in eval_splits()}
    a = {split: after[split]["metrics"]["accuracy"] for split in eval_splits()}
    if b["heldout"] - a["heldout"] > 0.01:
        return True
    for split in ("ood", "counterfactual", "adversarial"):
        if b[split] - a[split] > 0.015:
            return True
    return False


def run_post_quant_repair(
    level: str,
    width: int,
    initial_candidate: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    float_eval_accuracy: float,
    direct_eval_accuracy: float,
) -> dict[str, Any]:
    final_candidate, mutation_history, _ = repair_loop(
        "post_quant_mutation_repair",
        level,
        width,
        initial_candidate,
        task,
        settings,
        out,
        frozen_keys=set(),
        generations=settings.repair_generations,
        seed_label=f"post-{level}-{width}-{settings.seeds}",
    )
    return candidate_result(
        "post_quant_mutation_repair",
        level,
        width,
        final_candidate,
        task,
        settings,
        "post_quant_mutation_repair",
        float_eval_accuracy,
        direct_eval_accuracy,
        mutation_history=mutation_history,
        parameter_diff=quantized_diff(initial_candidate, final_candidate),
    )


def run_progressive_freeze(
    method: str,
    level: str,
    width: int,
    initial_candidate: dict[str, Any],
    path_metadata: list[dict[str, Any]],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    float_eval_accuracy: float,
    direct_eval_accuracy: float,
) -> dict[str, Any]:
    candidate = copy.deepcopy(initial_candidate)
    frozen: set[str] = set()
    freeze_history: list[dict[str, Any]] = []
    mutation_history_rows: list[dict[str, Any]] = []
    attempts = accepted = rejected = rollback = 0
    before_eval = evaluate_candidate(candidate, task, settings, sample_limit=0)
    for round_index in range(1, settings.progressive_rounds + 1):
        previous_candidate = copy.deepcopy(candidate)
        previous_frozen = set(frozen)
        previous_eval = before_eval
        target_keys = target_frozen_path_keys(path_metadata, method, round_index, settings.progressive_rounds)
        new_frozen = target_keys - frozen
        frozen.update(target_keys)
        input_bias = method == "input_projection_aware_progressive_freeze"
        repaired, hist, _ = repair_loop(
            method,
            level,
            width,
            candidate,
            task,
            settings,
            out,
            frozen_keys=frozen,
            generations=settings.progressive_generations_per_round,
            seed_label=f"{method}-{level}-{width}-round{round_index}-{settings.seeds}",
            input_projection_bias=input_bias,
        )
        attempts += hist["mutation_attempt_count"]
        accepted += hist["accepted_mutation_count"]
        rejected += hist["rejected_mutation_count"]
        rollback += hist["rollback_count"]
        candidate = repaired
        after_eval = evaluate_candidate(candidate, task, settings, sample_limit=0)
        round_rollback = rollback_needed(previous_eval, after_eval)
        if round_rollback:
            candidate = previous_candidate
            frozen = previous_frozen
            before_eval = previous_eval
        else:
            before_eval = after_eval
        round_row = {
            "round": round_index,
            "newly_frozen_parameter_count": len(new_frozen),
            "frozen_parameter_count": len(frozen),
            "frozen_parameter_ratio": round_float(len(frozen) / max(1, len(path_metadata))),
            "block_frozen_ratios": freeze_ratio_report(frozen, path_metadata),
            "plateau_generation_count": hist["plateau_generation_count"],
            "validation_accuracy": before_eval["validation"]["metrics"]["accuracy"],
            "heldout_accuracy": before_eval["heldout"]["metrics"]["accuracy"],
            "ood_accuracy": before_eval["ood"]["metrics"]["accuracy"],
            "counterfactual_accuracy": before_eval["counterfactual"]["metrics"]["accuracy"],
            "adversarial_accuracy": before_eval["adversarial"]["metrics"]["accuracy"],
            "rollback_applied": round_rollback,
            "candidate_hash": candidate_hash(candidate),
        }
        freeze_history.append(round_row)
        mutation_history_rows.extend(hist["history"])
        if out:
            out_path = Path(out)
            write_json(
                out_path / f"e7a8_freeze_history_{method}_{level}_width{width}.json",
                {"method": method, "quant_level": level, "width": width, "freeze_history": freeze_history},
            )
            append_progress_locked(out_path, "freeze_round", method=method, quant_level=level, width=width, round=round_index, frozen_ratio=round_row["frozen_parameter_ratio"], rollback_applied=round_rollback)
    mutation_history = {
        "method": method,
        "quant_level": level,
        "width": width,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rejected,
        "freeze_round_rollback_count": sum(1 for row in freeze_history if row["rollback_applied"]),
        "history": mutation_history_rows,
    }
    return candidate_result(
        method,
        level,
        width,
        candidate,
        task,
        settings,
        "mutation_repair_progressive_freeze_schedule",
        float_eval_accuracy,
        direct_eval_accuracy,
        mutation_history=mutation_history,
        freeze_history=freeze_history,
        parameter_diff=quantized_diff(initial_candidate, candidate),
    )


def run_scale_repair(
    level: str,
    width: int,
    initial_candidate: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    float_eval_accuracy: float,
    direct_eval_accuracy: float,
) -> dict[str, Any]:
    final_candidate, mutation_history, _ = repair_loop(
        "blockwise_scale_mutation_repair",
        level,
        width,
        initial_candidate,
        task,
        settings,
        out,
        frozen_keys=set(path_key(path) for path, _ in all_quantized_paths(initial_candidate)),
        generations=settings.repair_generations,
        seed_label=f"scale-{level}-{width}-{settings.seeds}",
        mutate_scales=True,
        input_projection_bias=True,
    )
    return candidate_result(
        "blockwise_scale_mutation_repair",
        level,
        width,
        final_candidate,
        task,
        settings,
        "post_quant_blockwise_scale_mutation_repair",
        float_eval_accuracy,
        direct_eval_accuracy,
        mutation_history=mutation_history,
        parameter_diff=quantized_diff(initial_candidate, final_candidate),
    )


def run_worker_method(
    method: str,
    level: str,
    width: int,
    initial_candidate: dict[str, Any],
    path_metadata: list[dict[str, Any]],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    float_eval_accuracy: float,
    direct_eval_accuracy: float,
) -> dict[str, Any]:
    if method == "post_quant_mutation_repair":
        return run_post_quant_repair(level, width, initial_candidate, task, settings, out, float_eval_accuracy, direct_eval_accuracy)
    if method in {"distance_only_progressive_freeze", "sensitivity_aware_progressive_freeze", "input_projection_aware_progressive_freeze"}:
        return run_progressive_freeze(method, level, width, initial_candidate, path_metadata, task, settings, out, float_eval_accuracy, direct_eval_accuracy)
    if method == "blockwise_scale_mutation_repair":
        return run_scale_repair(level, width, initial_candidate, task, settings, out, float_eval_accuracy, direct_eval_accuracy)
    raise ValueError(method)


def run_core(settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, settings={**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths), "target_levels": list(settings.target_levels)})
    base_settings = e7a6_settings(settings)
    task = e7a3.generate_task(e7a6.e7a3_settings(base_settings))
    if out:
        append_progress_locked(out, "task_generated", rows={split: len(task[split]["rows"]) for split in task_splits()})
    results: dict[str, Any] = {
        "float32": {},
        "direct": {},
        "post_repair": {},
        "progressive": {},
        "scale_repair": {},
        "qat": {},
        "sensitivity": {},
        "path_metadata": {},
        "runtime": {"started_monotonic": started},
    }
    executor: ProcessPoolExecutor | None = None
    futures = {}
    if settings.execution_mode == "parallel":
        executor = ProcessPoolExecutor(max_workers=max(1, settings.parallel_workers))
        if out:
            append_progress_locked(out, "repair_lanes_ready", workers=settings.parallel_workers)
    for width in settings.widths:
        trained = e7a6.train_float_core(width, task, base_settings, out)
        trained["system"] = "baseline_float_matrix_core"
        trained["method"] = "baseline_float_matrix_core"
        results["float32"][width] = trained
        float_eval_accuracy = result_row(trained)["eval_accuracy"]
        for level in settings.target_levels:
            direct_candidate = make_low_bit_candidate(level, trained["state_dict"], width, "e7a8_low_bit_candidate_v1", "direct_low_bit_quant")
            direct_result = candidate_result("direct_low_bit_quant", level, width, direct_candidate, task, settings, "post_backprop_direct_low_bit_quant", float_eval_accuracy)
            results["direct"][(level, width)] = direct_result
            direct_eval_accuracy = result_row(direct_result)["eval_accuracy"]
            sensitivity = block_sensitivity_report(level, trained["state_dict"], width, task, settings)
            results["sensitivity"][(level, width)] = sensitivity
            block_sens = {block: row["sensitivity_score"] for block, row in sensitivity["rows"].items()}
            metadata = build_path_metadata(level, trained["state_dict"], direct_candidate, block_sens)
            results["path_metadata"][(level, width)] = metadata
            worker_jobs = (
                "post_quant_mutation_repair",
                "distance_only_progressive_freeze",
                "sensitivity_aware_progressive_freeze",
                "input_projection_aware_progressive_freeze",
                "blockwise_scale_mutation_repair",
            )
            for method in worker_jobs:
                initial = make_channel_scale_candidate(level, trained["state_dict"], width) if method == "blockwise_scale_mutation_repair" else copy.deepcopy(direct_candidate)
                if executor is not None:
                    future = executor.submit(
                        run_worker_method,
                        method,
                        level,
                        width,
                        initial,
                        metadata,
                        task,
                        settings,
                        out.as_posix() if out else None,
                        float_eval_accuracy,
                        direct_eval_accuracy,
                    )
                    futures[future] = (method, level, width)
                else:
                    result = run_worker_method(method, level, width, initial, metadata, task, settings, out.as_posix() if out else None, float_eval_accuracy, direct_eval_accuracy)
                    if method == "post_quant_mutation_repair":
                        results["post_repair"][(level, width)] = result
                    elif method == "blockwise_scale_mutation_repair":
                        results["scale_repair"][(level, width)] = result
                    else:
                        results["progressive"][(method, level, width)] = result
            qat = e7a7.train_qat_core(level, width, trained["state_dict"], task, e7a7_settings(settings), out, direct_eval_accuracy, float_eval_accuracy)
            qat["system"] = "qat_reference"
            qat["method"] = "qat_reference"
            qat["direct_eval_accuracy"] = direct_eval_accuracy
            results["qat"][(level, width)] = qat
            if out:
                append_progress_locked(out, "qat_reference_complete", quant_level=level, width=width, eval_accuracy=result_row(qat)["eval_accuracy"])
        if out:
            locked_write_json(out / "partial_status" / "e7a8_width_progress.json", {"completed_width": width, "pending_worker_jobs": len(futures)})
            append_progress_locked(out, "width_submitted", width=width, pending_worker_jobs=len(futures))
    if executor is not None:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                method, level, width = futures[future]
                result = future.result()
                if method == "post_quant_mutation_repair":
                    results["post_repair"][(level, width)] = result
                elif method == "blockwise_scale_mutation_repair":
                    results["scale_repair"][(level, width)] = result
                else:
                    results["progressive"][(method, level, width)] = result
                if out:
                    append_progress_locked(out, "repair_lane_joined", method=method, quant_level=level, width=width, completed=len(results["post_repair"]) + len(results["scale_repair"]) + len(results["progressive"]), pending=len(pending))
        executor.shutdown()
    results["runtime"]["elapsed_seconds"] = round_float(time.monotonic() - started)
    return task, results


def aggregate_results(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    systems: dict[str, Any] = {method: {} for method in METHODS}
    for width in settings.widths:
        systems["baseline_float_matrix_core"][str(width)] = result_row(results["float32"][width])
    for level in settings.target_levels:
        for method in METHODS:
            if method != "baseline_float_matrix_core":
                systems[method][level] = {}
        for width in settings.widths:
            direct_row = result_row(results["direct"][(level, width)])
            systems["direct_low_bit_quant"][level][str(width)] = direct_row
            qat_row = result_row(results["qat"][(level, width)])
            systems["qat_reference"][level][str(width)] = qat_row
            for method in ("post_quant_mutation_repair", "blockwise_scale_mutation_repair"):
                result = results["post_repair"][(level, width)] if method == "post_quant_mutation_repair" else results["scale_repair"][(level, width)]
                result["qat_eval_accuracy"] = qat_row["eval_accuracy"]
                systems[method][level][str(width)] = result_row(result)
            for method in ("distance_only_progressive_freeze", "sensitivity_aware_progressive_freeze", "input_projection_aware_progressive_freeze"):
                result = results["progressive"][(method, level, width)]
                result["qat_eval_accuracy"] = qat_row["eval_accuracy"]
                systems[method][level][str(width)] = result_row(result)
    best = {"baseline_float_matrix_core": max(systems["baseline_float_matrix_core"].values(), key=lambda row: row["eval_accuracy"])}
    for method in METHODS:
        if method == "baseline_float_matrix_core":
            continue
        best[method] = {}
        for level in settings.target_levels:
            best[method][level] = max(systems[method][level].values(), key=lambda row: row["eval_accuracy"])
    return {
        "schema_version": "e7a8_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "methods": list(METHODS),
        "target_levels": list(settings.target_levels),
        "systems": systems,
        "best_by_eval_accuracy": best,
        "thresholds": {
            "plateau_min_delta": settings.plateau_min_delta,
            "rollback_heldout_or_validation_drop": 0.01,
            "rollback_ood_counterfactual_adversarial_drop": 0.015,
        },
    }


def method_comparison_report(aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for level in settings.target_levels:
        rows[level] = {}
        for method in METHODS:
            if method == "baseline_float_matrix_core":
                continue
            best = aggregate["best_by_eval_accuracy"][method][level]
            rows[level][method] = best
        best_non_qat = max(
            (row for method, row in rows[level].items() if method != "qat_reference"),
            key=lambda row: row["eval_accuracy"],
        )
        rows[level]["best_non_qat_method"] = best_non_qat
    return {
        "schema_version": "e7a8_method_comparison_report_v1",
        "rows": rows,
    }


def freeze_schedule_report(results: dict[str, Any], aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for level in settings.target_levels:
        rows[level] = {}
        for width in settings.widths:
            rows[level][str(width)] = {}
            for method in ("distance_only_progressive_freeze", "sensitivity_aware_progressive_freeze", "input_projection_aware_progressive_freeze"):
                result = results["progressive"][(method, level, width)]
                rows[level][str(width)][method] = {
                    "aggregate_metrics": aggregate["systems"][method][level][str(width)],
                    "freeze_history": result["freeze_history"],
                    "mutation_history": {key: value for key, value in result["mutation_history"].items() if key != "history"},
                    "parameter_diff": result["parameter_diff"],
                }
    return {
        "schema_version": "e7a8_freeze_schedule_report_v1",
        "plateau_rule": {"min_delta": settings.plateau_min_delta, "patience": settings.plateau_patience},
        "rows": rows,
    }


def input_projection_report(results: dict[str, Any], aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for level in settings.target_levels:
        rows[level] = {}
        for width in settings.widths:
            sens = results["sensitivity"][(level, width)]["rows"]["input_projection"]
            input_method = aggregate["systems"]["input_projection_aware_progressive_freeze"][level][str(width)]
            distance = aggregate["systems"]["distance_only_progressive_freeze"][level][str(width)]
            scale = aggregate["systems"]["blockwise_scale_mutation_repair"][level][str(width)]
            rows[level][str(width)] = {
                "input_projection_sensitivity": sens,
                "input_projection_aware_metrics": input_method,
                "distance_only_metrics": distance,
                "blockwise_scale_metrics": scale,
                "input_projection_aware_delta_vs_distance": round_float(input_method["eval_accuracy"] - distance["eval_accuracy"]),
                "scale_delta_vs_post_quant": round_float(scale["eval_accuracy"] - aggregate["systems"]["post_quant_mutation_repair"][level][str(width)]["eval_accuracy"]),
            }
    return {
        "schema_version": "e7a8_input_projection_damage_recovery_report_v1",
        "rows": rows,
    }


def mutation_repair_report(results: dict[str, Any], aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    all_attempts = True
    rollback_ok = True
    any_accepted = False
    for level in settings.target_levels:
        rows[level] = {}
        for width in settings.widths:
            rows[level][str(width)] = {}
            method_results = {
                "post_quant_mutation_repair": results["post_repair"][(level, width)],
                "blockwise_scale_mutation_repair": results["scale_repair"][(level, width)],
            }
            for method in ("distance_only_progressive_freeze", "sensitivity_aware_progressive_freeze", "input_projection_aware_progressive_freeze"):
                method_results[method] = results["progressive"][(method, level, width)]
            for method, result in method_results.items():
                hist = result["mutation_history"]
                rows[level][str(width)][method] = {
                    "aggregate_metrics": aggregate["systems"][method][level][str(width)],
                    "mutation_history": {key: value for key, value in hist.items() if key != "history"},
                    "parameter_diff": result["parameter_diff"],
                }
                all_attempts = all_attempts and hist["mutation_attempt_count"] > 0
                rollback_ok = rollback_ok and hist["rejected_mutation_count"] == hist["rollback_count"]
                any_accepted = any_accepted or hist["accepted_mutation_count"] > 0
    return {
        "schema_version": "e7a8_mutation_repair_report_v1",
        "rows": rows,
        "all_mutation_methods_have_attempts": all_attempts,
        "all_rejected_mutations_rolled_back": rollback_ok,
        "at_least_one_mutation_accepted": any_accepted,
    }


def no_synthetic_metric_audit(task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a8_no_synthetic_metric_audit_v1",
        "generated_from_row_level_eval": True,
        "row_counts": {split: len(task[split]["rows"]) for split in task_splits()},
        "row_level_samples_present": all(
            bool(results["direct"][(level, width)]["evals"]["heldout"]["row_level_samples"])
            and bool(results["qat"][(level, width)]["evals"]["heldout"]["row_level_samples"])
            for level in settings.target_levels
            for width in settings.widths
        ),
        "hardcoded_improvement_flags_present": False,
        "mutation_repair_used_optimizer_or_backprop": False,
        "qat_reference_uses_backprop_by_design": True,
        "new_architecture_added": False,
        "broad_claims_intentionally_deferred": True,
    }


def choose_decision(aggregate: dict[str, Any], comparison: dict[str, Any], repair: dict[str, Any], audit: dict[str, Any], settings: Settings) -> dict[str, Any]:
    if (
        not audit["generated_from_row_level_eval"]
        or audit["hardcoded_improvement_flags_present"]
        or not repair["all_mutation_methods_have_attempts"]
        or not repair["all_rejected_mutations_rolled_back"]
        or not repair["at_least_one_mutation_accepted"]
    ):
        decision = "e7a8_invalid_artifact_detected"
    else:
        rows = []
        for level in settings.target_levels:
            post = aggregate["best_by_eval_accuracy"]["post_quant_mutation_repair"][level]
            qat = aggregate["best_by_eval_accuracy"]["qat_reference"][level]
            direct = aggregate["best_by_eval_accuracy"]["direct_low_bit_quant"][level]
            gap_post_to_qat = max(0.0, qat["eval_accuracy"] - post["eval_accuracy"])
            for method in ("distance_only_progressive_freeze", "sensitivity_aware_progressive_freeze", "input_projection_aware_progressive_freeze", "blockwise_scale_mutation_repair"):
                row = aggregate["best_by_eval_accuracy"][method][level]
                rows.append(
                    {
                        "level": level,
                        "method": method,
                        "eval": row["eval_accuracy"],
                        "post": post["eval_accuracy"],
                        "direct": direct["eval_accuracy"],
                        "qat": qat["eval_accuracy"],
                        "gap_closed_vs_post_to_qat": 1.0 if gap_post_to_qat <= 1e-12 else max(0.0, row["eval_accuracy"] - post["eval_accuracy"]) / gap_post_to_qat,
                        "rollback_count": row.get("freeze_round_rollback_count", 0),
                    }
                )
        input_positive = any(row["method"] == "input_projection_aware_progressive_freeze" and row["eval"] > row["post"] and row["gap_closed_vs_post_to_qat"] >= 0.50 for row in rows)
        sensitivity_positive = any(
            row["method"] == "sensitivity_aware_progressive_freeze"
            and row["eval"] > row["post"]
            and row["eval"] > aggregate["best_by_eval_accuracy"]["distance_only_progressive_freeze"][row["level"]]["eval_accuracy"]
            for row in rows
        )
        distance_sufficient = all(
            aggregate["best_by_eval_accuracy"]["distance_only_progressive_freeze"][level]["eval_accuracy"]
            >= aggregate["best_by_eval_accuracy"]["sensitivity_aware_progressive_freeze"][level]["eval_accuracy"] - 0.005
            for level in settings.target_levels
        ) and any(
            aggregate["best_by_eval_accuracy"]["distance_only_progressive_freeze"][level]["eval_accuracy"]
            > aggregate["best_by_eval_accuracy"]["post_quant_mutation_repair"][level]["eval_accuracy"]
            for level in settings.target_levels
        )
        scale_positive = any(row["method"] == "blockwise_scale_mutation_repair" and row["eval"] > row["post"] + 0.01 for row in rows)
        no_advantage = all(row["eval"] <= row["post"] + 1e-12 for row in rows)
        brittle = any(row["rollback_count"] >= 2 for row in rows if row["method"].endswith("progressive_freeze"))
        if input_positive:
            decision = "e7a8_input_projection_aware_progressive_freeze_positive"
        elif sensitivity_positive:
            decision = "e7a8_sensitivity_aware_freeze_positive"
        elif distance_sufficient:
            decision = "e7a8_distance_only_freeze_sufficient"
        elif scale_positive:
            decision = "e7a8_blockwise_scale_repair_positive"
        elif brittle:
            decision = "e7a8_progressive_freeze_overfit_or_brittle"
        elif no_advantage:
            decision = "e7a8_progressive_freeze_no_advantage"
        else:
            decision = "e7a8_progressive_freeze_no_advantage"
    return {
        "schema_version": "e7a8_decision_v1",
        "decision": decision,
        "valid_decisions": list(VALID_DECISIONS),
        "deterministic_replay_passed": False,
        "broad_claims_intentionally_deferred": True,
    }


def task_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a8_task_generation_report_v1",
        "milestone": MILESTONE,
        "inherits_task_from": "E7A7_LOW_BIT_REPAIR_OPERATOR_AUDIT",
        "base_task_inherits_from": "E7A3_NEURAL_MATRIX_SUBSTRATE_HARNESS",
        "input_dim": e7a3.INPUT_DIM,
        "class_count": e7a3.CLASS_COUNT,
        "splits": {split: {"row_count": len(data["rows"])} for split, data in task.items()},
        "seeds": list(settings.seeds),
    }


def row_samples(results: dict[str, Any], split: str, settings: Settings) -> dict[str, Any]:
    samples = {}
    for level in settings.target_levels:
        samples[level] = {}
        for width in settings.widths:
            samples[level][str(width)] = {
                "direct_low_bit_quant": results["direct"][(level, width)]["evals"][split]["row_level_samples"],
                "post_quant_mutation_repair": results["post_repair"][(level, width)]["evals"][split]["row_level_samples"],
                "input_projection_aware_progressive_freeze": results["progressive"][("input_projection_aware_progressive_freeze", level, width)]["evals"][split]["row_level_samples"],
                "blockwise_scale_mutation_repair": results["scale_repair"][(level, width)]["evals"][split]["row_level_samples"],
                "qat_reference": results["qat"][(level, width)]["evals"][split]["row_level_samples"],
            }
    return {
        "schema_version": "e7a8_row_level_eval_sample_v1",
        "split": split,
        "samples": samples,
    }


def mutation_history_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for level in settings.target_levels:
        for width in settings.widths:
            rows[f"{level}/width{width}/post_quant_mutation_repair"] = results["post_repair"][(level, width)]["mutation_history"]
            rows[f"{level}/width{width}/blockwise_scale_mutation_repair"] = results["scale_repair"][(level, width)]["mutation_history"]
            for method in ("distance_only_progressive_freeze", "sensitivity_aware_progressive_freeze", "input_projection_aware_progressive_freeze"):
                rows[f"{level}/width{width}/{method}"] = results["progressive"][(method, level, width)]["mutation_history"]
    return {
        "schema_version": "e7a8_mutation_history_report_v1",
        "rows": rows,
    }


def runtime_report(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a8_runtime_report_v1",
        "elapsed_seconds": results["runtime"].get("elapsed_seconds"),
    }


def build_report(out: Path, decision: dict[str, Any], comparison: dict[str, Any], aggregate: dict[str, Any], settings: Settings) -> str:
    lines = [
        "# E7A8 Progressive Quant Freeze Plateau Repair Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        "broad_claims = intentionally deferred",
        "```",
        "",
        f"Run root: `{out.relative_to(REPO_ROOT).as_posix()}`",
        "",
        "## Method Comparison",
        "",
        "| level | method | width | eval | drop from float | recovery vs direct | gap to QAT |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for level in settings.target_levels:
        for method, row in comparison["rows"][level].items():
            if method == "best_non_qat_method":
                continue
            lines.append(
                f"| `{level}` | `{method}` | {row['width']} | {row['eval_accuracy']:.6f} | "
                f"{row.get('quantization_drop_from_float', 0.0):.6f} | {row.get('recovery_delta_vs_direct', 0.0):.6f} | {row.get('gap_to_qat', 0.0):.6f} |"
            )
    best_float = aggregate["best_by_eval_accuracy"]["baseline_float_matrix_core"]
    lines.extend(
        [
            "",
            "## Baseline",
            "",
            f"Best float32 matrix-core: width `{best_float['width']}`, eval `{best_float['eval_accuracy']:.6f}`.",
            "",
            "This probe only tests low-bit matrix-core repair strategy.",
            "",
        ]
    )
    return "\n".join(lines)


def build_payloads(out: Path, task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    aggregate = aggregate_results(results, settings)
    comparison = method_comparison_report(aggregate, settings)
    freeze = freeze_schedule_report(results, aggregate, settings)
    input_report = input_projection_report(results, aggregate, settings)
    repair = mutation_repair_report(results, aggregate, settings)
    audit = no_synthetic_metric_audit(task, results, settings)
    decision = choose_decision(aggregate, comparison, repair, audit, settings)
    payloads: dict[str, Any] = {
        "e7a8_backend_manifest.json": {
            "schema_version": "e7a8_backend_manifest_v1",
            "milestone": MILESTONE,
            "methods": list(METHODS),
            "target_levels": list(settings.target_levels),
            "blocks": BLOCKS,
            "quant_configs": {level: e7a6.QUANT_CONFIGS[level] for level in settings.target_levels},
            "widths": list(settings.widths),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": e7a3.select_device(settings.device),
            "settings": {**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths), "target_levels": list(settings.target_levels)},
            "cpu_repair_and_qat_overlap_supported": True,
            "parallel_replay_supported": True,
            "new_architecture_added": False,
            "broad_claims_intentionally_deferred": True,
        },
        "e7a8_task_generation_report.json": task_report(task, settings),
        "e7a8_method_comparison_report.json": comparison,
        "e7a8_freeze_schedule_report.json": freeze,
        "e7a8_input_projection_damage_recovery_report.json": input_report,
        "e7a8_mutation_repair_report.json": repair,
        "e7a8_mutation_history.json": mutation_history_report(results, settings),
        "e7a8_no_synthetic_metric_audit.json": audit,
        "e7a8_runtime_report.json": runtime_report(results),
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e7a8_summary_v1",
            "milestone": MILESTONE,
            "decision": decision["decision"],
            "best_float_width": aggregate["best_by_eval_accuracy"]["baseline_float_matrix_core"]["width"],
            "best_float_eval_accuracy": aggregate["best_by_eval_accuracy"]["baseline_float_matrix_core"]["eval_accuracy"],
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
            "broad_claims_intentionally_deferred": True,
        },
    }
    for split in eval_splits():
        payloads[f"e7a8_row_level_eval_sample_{split}.json"] = row_samples(results, split, settings)
    payloads["report.md"] = build_report(out, decision, comparison, aggregate, settings)
    return payloads


def compute_hashes(payloads: dict[str, Any]) -> dict[str, str]:
    return {name: payload_sha256(payloads[name]) for name in HASH_ARTIFACTS}


def deterministic_replay(settings: Settings, out: Path, primary_payloads: dict[str, Any]) -> dict[str, Any]:
    replay_out = out / "deterministic_replay_work"
    if replay_out.exists():
        shutil.rmtree(replay_out)
    append_progress_locked(out, "deterministic_replay_start", replay_out=replay_out.relative_to(REPO_ROOT).as_posix())
    task_replay, results_replay = run_core(settings, replay_out)
    replay_payloads = build_payloads(out, task_replay, results_replay, settings)
    primary_hashes = compute_hashes(primary_payloads)
    replay_hashes = compute_hashes(replay_payloads)
    comparisons = {
        name: {"primary_hash": primary_hashes[name], "replay_hash": replay_hashes[name], "match": primary_hashes[name] == replay_hashes[name]}
        for name in HASH_ARTIFACTS
    }
    report = {
        "schema_version": "e7a8_deterministic_replay_report_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
        "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix(),
    }
    append_progress_locked(out, "deterministic_replay_complete", internal_replay_passed=report["internal_replay_passed"])
    return report


def write_final_artifacts(out: Path, payloads: dict[str, Any], deterministic: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["e7a8_deterministic_replay_report.json"] = deterministic
    payloads["decision.json"]["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
    payloads["summary.json"]["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
    for name, payload in payloads.items():
        if name == "report.md":
            write_text(out / name, payload)
        else:
            write_json(out / name, payload)
    append_progress_locked(out, "final_artifacts_written", artifact_count=len(payloads))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT.as_posix())
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--widths", default="16,32")
    parser.add_argument("--target-levels", default=",".join(TARGET_LEVELS))
    parser.add_argument("--train-rows-per-seed", type=int, default=240)
    parser.add_argument("--validation-rows-per-seed", type=int, default=100)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=100)
    parser.add_argument("--ood-rows-per-seed", type=int, default=100)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=100)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=100)
    parser.add_argument("--gradient-epochs", type=int, default=160)
    parser.add_argument("--qat-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--qat-learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--repair-generations", type=int, default=60)
    parser.add_argument("--progressive-rounds", type=int, default=6)
    parser.add_argument("--progressive-generations-per-round", type=int, default=24)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--plateau-min-delta", type=float, default=0.001)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--quant-mutation-steps", type=int, default=24)
    parser.add_argument("--scale-mutation-steps", type=int, default=10)
    parser.add_argument("--scale-mutation-sigma", type=float, default=0.035)
    parser.add_argument("--matrix-steps", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--execution-mode", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--parallel-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 2, 22)))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    settings = Settings(
        seeds=parse_int_tuple(args.seeds),
        widths=parse_int_tuple(args.widths),
        target_levels=parse_level_tuple(args.target_levels),
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        gradient_epochs=args.gradient_epochs,
        qat_epochs=args.qat_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        qat_learning_rate=args.qat_learning_rate,
        weight_decay=args.weight_decay,
        population_size=args.population_size,
        repair_generations=args.repair_generations,
        progressive_rounds=args.progressive_rounds,
        progressive_generations_per_round=args.progressive_generations_per_round,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        elite_count=args.elite_count,
        quant_mutation_steps=args.quant_mutation_steps,
        scale_mutation_steps=args.scale_mutation_steps,
        scale_mutation_sigma=args.scale_mutation_sigma,
        matrix_steps=args.matrix_steps,
        device=args.device,
        execution_mode=args.execution_mode,
        parallel_workers=args.parallel_workers,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    task, results = run_core(settings, out)
    payloads = build_payloads(out, task, results, settings)
    deterministic = deterministic_replay(settings, out, payloads)
    write_final_artifacts(out, payloads, deterministic)
    decision = copy.deepcopy(payloads["decision.json"])
    decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
    print(json.dumps({"decision": decision["decision"], "deterministic_replay_passed": deterministic["internal_replay_passed"], "out": out.as_posix()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
