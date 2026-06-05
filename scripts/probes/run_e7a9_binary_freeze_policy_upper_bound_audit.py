#!/usr/bin/env python3
"""E7A9 binary freeze policy upper-bound audit.

E7A9 decides whether binary matrix-core is worth pursuing as a quality path or
only as a compression side branch. It compares binary best-effort policies
against int4 and ternary references using the existing E7 matrix-core.
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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
E7A8_PATH = Path(__file__).with_name("run_e7a8_progressive_quant_freeze_plateau_repair.py")
MILESTONE = "E7A9_BINARY_FREEZE_POLICY_UPPER_BOUND_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e7a9_binary_freeze_policy_upper_bound_audit")
DEFAULT_SEEDS = (83001, 83002, 83003)

BLOCKS: dict[str, tuple[str, ...]] = {
    "input_projection": ("win",),
    "recurrent_state": ("state",),
    "carry_gate": ("carry_raw",),
    "state_bias": ("bstate",),
    "output_head": ("wout", "bout"),
}
METHODS = (
    "float32_matrix_core",
    "int8_direct",
    "int4_direct",
    "ternary_qat_reference",
    "binary_direct",
    "binary_qat_baseline",
    "binary_qat_best_effort",
    "binary_distance_paramwise_freeze",
    "binary_sensitivity_paramwise_freeze",
    "binary_qat_warmstart_paramwise_freeze",
    "binary_direct_mutation_repair",
    "mixed_input_int4_state_binary_output_int4",
    "mixed_input_ternary_state_binary_output_int4",
)
VALID_DECISIONS = (
    "e7a9_binary_quality_competitive",
    "e7a9_binary_not_quality_competitive",
    "e7a9_mixed_precision_matrix_core_preferred",
    "e7a9_binary_paramwise_freeze_positive",
    "e7a9_qat_upper_bound_remains_preferred",
    "e7a9_invalid_artifact_detected",
)
HASH_ARTIFACTS = (
    "e7a9_task_generation_report.json",
    "e7a9_method_comparison_report.json",
    "e7a9_binary_freeze_schedule_report.json",
    "e7a9_qat_upper_bound_report.json",
    "e7a9_precision_tradeoff_report.json",
    "e7a9_mixed_precision_report.json",
    "e7a9_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)


def load_e7a8_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7a8_progressive_quant_freeze_plateau_repair", E7A8_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7A8 from {E7A8_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7a8 = load_e7a8_module()
e7a7 = e7a8.e7a7
e7a6 = e7a8.e7a6
e7a3 = e7a8.e7a3


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    widths: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    gradient_epochs: int
    qat_epochs: int
    best_effort_qat_epochs: int
    batch_size: int
    learning_rate: float
    qat_learning_rate: float
    best_effort_learning_rate: float
    weight_decay: float
    population_size: int
    repair_generations: int
    plateau_patience: int
    plateau_min_delta: float
    paramwise_generation_per_step: int
    paramwise_step_limit: int
    elite_count: int
    quant_mutation_steps: int
    matrix_steps: int
    distillation_weight: float
    distillation_temperature: float
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
    return int(hashlib.sha256(f"e7a9::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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


def e7a7_settings(settings: Settings, qat_epochs: int | None = None) -> Any:
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
        qat_epochs=settings.qat_epochs if qat_epochs is None else qat_epochs,
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


def split_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {split: result["evals"][split]["metrics"] for split in task_splits()}


def solve_pass(row: dict[str, float]) -> bool:
    return (
        row["heldout_accuracy"] >= 0.90
        and row["ood_accuracy"] >= 0.85
        and row["counterfactual_accuracy"] >= 0.85
        and row["adversarial_accuracy"] >= 0.80
    )


def nominal_bits_for_key(candidate: dict[str, Any], key: str) -> float:
    if "key_quant_levels" in candidate:
        level = candidate["key_quant_levels"][key]
        return float(e7a6.QUANT_CONFIGS[level]["nominal_bits"])
    level = candidate.get("source_quant_level") or candidate.get("quant_level")
    return float(e7a6.QUANT_CONFIGS[level]["nominal_bits"])


def scale_count(candidate: dict[str, Any]) -> int:
    count = 0
    for value in candidate.get("scales", {}).values():
        arr = np.asarray(value)
        count += int(arr.size) if arr.ndim else 1
    return count


def bit_cost(candidate: dict[str, Any] | None, parameter_count: int, float_bits: int = 32) -> dict[str, Any]:
    if candidate is None:
        total = parameter_count * float_bits
        return {
            "nominal_weight_bits": float_bits,
            "weight_bit_cost": total,
            "scale_bit_cost": 0,
            "total_bit_cost": total,
            "compression_vs_float32": 1.0,
        }
    weight_bits = 0.0
    for key, value in candidate["q"].items():
        weight_bits += np.asarray(value).size * nominal_bits_for_key(candidate, key)
    scale_bits = scale_count(candidate) * 32
    total = weight_bits + scale_bits
    float_total = parameter_count * 32
    return {
        "nominal_weight_bits": round_float(weight_bits / max(1, parameter_count)),
        "weight_bit_cost": round_float(weight_bits),
        "scale_bit_cost": int(scale_bits),
        "total_bit_cost": round_float(total),
        "compression_vs_float32": round_float(float_total / max(total, 1e-12)),
    }


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    if result["method"] == "float32_matrix_core":
        metrics = e7a6.result_row(result)
    else:
        split = split_metrics(result)
        eval_accuracy = round_float(float(np.mean([split[name]["accuracy"] for name in eval_splits()])))
        metrics = {
            "width": result["width"],
            "matrix_shape": result["hidden_matrix_shape"],
            "matrix_cells": int(result["hidden_matrix_shape"][0] * result["hidden_matrix_shape"][1]),
            "parameter_count": result["parameter_count"],
            "eval_accuracy": eval_accuracy,
            "train_accuracy": split["train"]["accuracy"],
            "validation_accuracy": split["validation"]["accuracy"],
            "heldout_accuracy": split["heldout"]["accuracy"],
            "ood_accuracy": split["ood"]["accuracy"],
            "counterfactual_accuracy": split["counterfactual"]["accuracy"],
            "adversarial_accuracy": split["adversarial"]["accuracy"],
            "generalization_gap": round_float(split["validation"]["accuracy"] - eval_accuracy),
            "training_mode": result["training_mode"],
        }
        metrics["solve_passed"] = solve_pass(metrics)
    metrics["method"] = result["method"]
    if "quant_level" in result:
        metrics["quant_level"] = result["quant_level"]
    if "float_eval_accuracy" in result:
        metrics["float_eval_accuracy"] = result["float_eval_accuracy"]
        metrics["gap_to_float"] = round_float(result["float_eval_accuracy"] - metrics["eval_accuracy"])
    if "int4_eval_accuracy" in result:
        metrics["int4_eval_accuracy"] = result["int4_eval_accuracy"]
        metrics["gap_to_int4"] = round_float(result["int4_eval_accuracy"] - metrics["eval_accuracy"])
    if "ternary_eval_accuracy" in result:
        metrics["ternary_eval_accuracy"] = result["ternary_eval_accuracy"]
        metrics["gap_to_ternary"] = round_float(result["ternary_eval_accuracy"] - metrics["eval_accuracy"])
    if "bit_cost" in result:
        metrics["bit_cost"] = result["bit_cost"]
    if "mutation_history" in result:
        hist = result["mutation_history"]
        metrics["mutation_attempt_count"] = hist["mutation_attempt_count"]
        metrics["accepted_mutation_count"] = hist["accepted_mutation_count"]
        metrics["rejected_mutation_count"] = hist["rejected_mutation_count"]
        metrics["rollback_count"] = hist["rollback_count"]
    if "freeze_history" in result:
        metrics["freeze_rounds_count"] = len(result["freeze_history"])
        metrics["freeze_round_rollback_count"] = sum(1 for row in result["freeze_history"] if row.get("rollback_applied"))
        metrics["final_frozen_parameter_ratio"] = result["freeze_history"][-1]["frozen_parameter_ratio"] if result["freeze_history"] else 0.0
        metrics["input_projection_frozen_ratio"] = result["freeze_history"][-1]["block_frozen_ratios"].get("input_projection", 0.0) if result["freeze_history"] else 0.0
    return metrics


def make_candidate(level: str, state_dict: dict[str, np.ndarray], width: int, method: str) -> dict[str, Any]:
    candidate = e7a6.quantize_state_dict(level, state_dict, width)
    candidate["schema_version"] = "e7a9_quantized_matrix_core_candidate_v1"
    candidate["source_quant_level"] = level
    candidate["method_origin"] = method
    candidate["candidate_hash"] = candidate_hash(candidate)
    return candidate


def make_channel_scale_candidate(level: str, state_dict: dict[str, np.ndarray], width: int, method: str) -> dict[str, Any]:
    candidate = e7a8.make_channel_scale_candidate(level, state_dict, width)
    candidate["schema_version"] = "e7a9_channel_scale_candidate_v1"
    candidate["method_origin"] = method
    candidate["source_quant_level"] = level
    candidate["candidate_hash"] = candidate_hash(candidate)
    return candidate


def make_mixed_candidate(state_dict: dict[str, np.ndarray], width: int, input_level: str) -> dict[str, Any]:
    levels = {
        "win": input_level,
        "state": "binary",
        "carry_raw": "int4",
        "bstate": "int4",
        "wout": "int4",
        "bout": "int4",
    }
    q = {}
    scales = {}
    zero_counts = {}
    key_quant_levels = {}
    for key, level in levels.items():
        quantized, scale = e7a6.quantize_array(state_dict[key], level)
        q[key] = quantized.tolist()
        scales[key] = round_float(scale)
        zero_counts[key] = int(np.sum(quantized == 0))
        key_quant_levels[key] = level
    candidate = {
        "schema_version": "e7a9_mixed_precision_matrix_core_candidate_v1",
        "quant_level": "mixed",
        "source_quant_level": "mixed",
        "key_quant_levels": key_quant_levels,
        "width": width,
        "q": q,
        "scales": scales,
        "zero_counts": zero_counts,
        "candidate_hash": "",
    }
    candidate["candidate_hash"] = candidate_hash(candidate)
    return candidate


def evaluate_candidate(candidate: dict[str, Any], task: dict[str, Any], settings: Settings, sample_limit: int = 10) -> dict[str, Any]:
    return {
        split: e7a3.evaluate_logits(e7a8.quantized_forward(candidate, data["x"], settings.matrix_steps), data, sample_limit)
        for split, data in task.items()
    }


def candidate_result(
    method: str,
    width: int,
    candidate: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    training_mode: str,
    float_eval_accuracy: float,
    quant_level: str,
    int4_eval_accuracy: float | None = None,
    ternary_eval_accuracy: float | None = None,
    mutation_history: dict[str, Any] | None = None,
    freeze_history: list[dict[str, Any]] | None = None,
    parameter_diff: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": method,
        "system": method,
        "quant_level": quant_level,
        "width": width,
        "training_mode": training_mode,
        "evals": evaluate_candidate(candidate, task, settings),
        "history": [],
        "parameter_count": e7a8.quantized_parameter_count(candidate),
        "nonzero_parameter_count": e7a8.quantized_nonzero_count(candidate),
        "hidden_matrix_shape": [width, width],
        "initial_hash": candidate.get("initial_hash", candidate_hash(candidate)),
        "final_hash": candidate_hash(candidate),
        "candidate": candidate,
        "float_eval_accuracy": float_eval_accuracy,
        "bit_cost": bit_cost(candidate, e7a8.quantized_parameter_count(candidate)),
    }
    if int4_eval_accuracy is not None:
        result["int4_eval_accuracy"] = int4_eval_accuracy
    if ternary_eval_accuracy is not None:
        result["ternary_eval_accuracy"] = ternary_eval_accuracy
    if mutation_history is not None:
        result["mutation_history"] = mutation_history
    if freeze_history is not None:
        result["freeze_history"] = freeze_history
    if parameter_diff is not None:
        result["parameter_diff"] = parameter_diff
    if extra:
        result.update(extra)
    return result


def float_result(trained: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(trained)
    result["method"] = "float32_matrix_core"
    result["system"] = "float32_matrix_core"
    result["bit_cost"] = bit_cost(None, result["parameter_count"])
    return result


def fake_quant_binary_per_channel(param: torch.Tensor) -> torch.Tensor:
    eps = torch.tensor(1e-8, device=param.device, dtype=param.dtype)
    if param.ndim == 2:
        scale = torch.clamp(torch.mean(torch.abs(param), dim=0, keepdim=True).detach(), min=float(eps.detach().cpu().item()))
    else:
        scale = torch.clamp(torch.mean(torch.abs(param)).detach(), min=float(eps.detach().cpu().item()))
    quantized = torch.where(torch.abs(param) <= eps, torch.zeros_like(param), torch.where(param >= 0.0, torch.ones_like(param), -torch.ones_like(param)))
    dequantized = quantized * scale
    return param + (dequantized - param).detach()


def blended_qat_forward(model: Any, x: torch.Tensor, alpha: float, matrix_steps: int) -> torch.Tensor:
    def blend(param: torch.Tensor) -> torch.Tensor:
        fq = fake_quant_binary_per_channel(param)
        return (1.0 - alpha) * param + alpha * fq

    win = blend(model.win)
    state = blend(model.state)
    carry_raw = blend(model.carry_raw)
    bstate = blend(model.bstate)
    wout = blend(model.wout)
    bout = blend(model.bout)
    drive = x @ win + bstate
    h = torch.tanh(drive)
    carry = torch.sigmoid(carry_raw).unsqueeze(0)
    for _ in range(matrix_steps):
        proposal = torch.tanh(h @ state + drive)
        h = carry * h + (1.0 - carry) * proposal
    return h @ wout + bout


def train_binary_qat_best_effort(
    width: int,
    float_state_dict: dict[str, np.ndarray],
    task: dict[str, Any],
    settings: Settings,
    out: Path | None,
    float_eval_accuracy: float,
    int4_eval_accuracy: float,
    ternary_eval_accuracy: float,
) -> dict[str, Any]:
    device = e7a3.select_device(settings.device)
    e7a3.set_determinism(stable_seed(f"best-effort-qat-{width}-{settings.seeds}"), device)
    model = e7a6.FloatMatrixCore(width, settings.matrix_steps).to(device)
    model.load_state_dict({key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in float_state_dict.items()})
    teacher = e7a6.FloatMatrixCore(width, settings.matrix_steps).to(device)
    teacher.load_state_dict({key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in float_state_dict.items()})
    teacher.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.best_effort_learning_rate, weight_decay=settings.weight_decay)
    x_train = torch.as_tensor(task["train"]["x"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(task["train"]["y"], dtype=torch.long, device=device)
    x_val = torch.as_tensor(task["validation"]["x"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(task["validation"]["y"], dtype=torch.long, device=device)
    with torch.no_grad():
        teacher_train = teacher(x_train).detach()
    rng = np.random.default_rng(stable_seed(f"best-effort-batches-{width}-{settings.seeds}"))
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    history = []
    last_heartbeat = time.monotonic()
    temp = settings.distillation_temperature
    for epoch in range(1, settings.best_effort_qat_epochs + 1):
        alpha = epoch / max(1, settings.best_effort_qat_epochs)
        order = rng.permutation(x_train.shape[0])
        model.train()
        for start in range(0, len(order), settings.batch_size):
            idx = torch.as_tensor(order[start : start + settings.batch_size], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = blended_qat_forward(model, x_train[idx], alpha, settings.matrix_steps)
            ce = F.cross_entropy(logits, y_train[idx])
            kd = F.kl_div(
                F.log_softmax(logits / temp, dim=1),
                F.softmax(teacher_train[idx] / temp, dim=1),
                reduction="batchmean",
            ) * (temp * temp)
            loss = ce + settings.distillation_weight * kd
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = blended_qat_forward(model, x_val, 1.0, settings.matrix_steps)
            val_loss = F.cross_entropy(logits, y_val).detach().cpu().item()
            val_acc = (torch.argmax(logits, dim=1) == y_val).float().mean().detach().cpu().item()
        if val_acc >= best_val:
            best_val = float(val_acc)
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch,
            "quant_strength_alpha": round_float(alpha),
            "validation_accuracy": round_float(val_acc),
            "validation_loss": round_float(val_loss),
            "state_hash": e7a6.vector_hash(e7a6.torch_state_vector(model)),
        }
        history.append(row)
        now = time.monotonic()
        if out and (now - last_heartbeat >= settings.heartbeat_seconds or epoch == settings.best_effort_qat_epochs):
            write_json(out / f"e7a9_binary_best_effort_qat_history_width{width}.json", {"width": width, "history": history})
            append_progress_locked(out, "best_effort_qat_epoch", width=width, epoch=epoch, validation_accuracy=row["validation_accuracy"], alpha=row["quant_strength_alpha"])
            last_heartbeat = now
    state_dict = {key: value.detach().cpu().numpy().astype(np.float64) for key, value in best_state.items()}
    candidate = make_channel_scale_candidate("binary", state_dict, width, "binary_qat_best_effort")
    return candidate_result(
        "binary_qat_best_effort",
        width,
        candidate,
        task,
        settings,
        "annealed_binary_qat_with_teacher_distillation_per_channel_scale",
        float_eval_accuracy,
        "binary",
        int4_eval_accuracy=int4_eval_accuracy,
        ternary_eval_accuracy=ternary_eval_accuracy,
        extra={"history": history, "best_validation_accuracy": round_float(best_val), "device": device},
    )


def all_paths(candidate: dict[str, Any]) -> list[tuple[tuple[Any, ...], int]]:
    return e7a8.all_quantized_paths(candidate)


def path_key(path: tuple[Any, ...]) -> str:
    return e7a8.path_key(path)


def key_from_path(path: tuple[Any, ...]) -> str:
    return e7a8.key_from_path(path)


def block_for_key(key: str) -> str:
    return e7a8.block_for_key(key)


def freeze_ratio_report(frozen: set[str], metadata: list[dict[str, Any]]) -> dict[str, float]:
    return e7a8.freeze_ratio_report(frozen, metadata)


def score_candidate(candidate: dict[str, Any], task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    evals = {
        split: e7a3.evaluate_logits(e7a8.quantized_forward(candidate, data["x"], settings.matrix_steps), data, sample_limit=0)
        for split, data in {"train": task["train"], "validation": task["validation"]}.items()
    }
    train = evals["train"]["metrics"]
    val = evals["validation"]["metrics"]
    nonzero_ratio = e7a8.quantized_nonzero_count(candidate) / max(1, e7a8.quantized_parameter_count(candidate))
    fitness = 0.20 * train["accuracy"] + 0.80 * val["accuracy"] - 0.02 * val["cross_entropy"] - 0.001 * nonzero_ratio
    return {"candidate": candidate, "evals": evals, "fitness": round_float(fitness)}


def mutate_binary_candidate(candidate: dict[str, Any], rng: random.Random, frozen: set[str], steps: int) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    paths = [(path, value) for path, value in all_paths(child) if path_key(path) not in frozen]
    if not paths:
        return child
    for path, value in rng.sample(paths, k=min(steps, len(paths))):
        new_value = rng.choice((-1, 1)) if value == 0 else -int(value)
        e7a6.set_quantized_path(child, path, new_value)
    e7a8.update_zero_counts(child)
    return child


def tiny_repair_step(
    method: str,
    level: str,
    width: int,
    candidate: dict[str, Any],
    frozen: set[str],
    task: dict[str, Any],
    settings: Settings,
    rng: random.Random,
    out_path: Path | None,
    freeze_index: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    population = [score_candidate(copy.deepcopy(candidate), task, settings)]
    for _ in range(settings.population_size - 1):
        population.append(score_candidate(mutate_binary_candidate(candidate, rng, frozen, settings.quant_mutation_steps), task, settings))
    accepted = rejected = rollback = attempts = 0
    for generation in range(1, settings.paramwise_generation_per_step + 1):
        population.sort(key=lambda row: row["fitness"], reverse=True)
        elites = copy.deepcopy(population[: max(1, settings.elite_count)])
        next_population = elites
        while len(next_population) < settings.population_size:
            parent = copy.deepcopy(rng.choice(population))
            child_candidate = mutate_binary_candidate(parent["candidate"], rng, frozen, settings.quant_mutation_steps)
            child = score_candidate(child_candidate, task, settings)
            attempts += 1
            if child["fitness"] >= parent["fitness"]:
                accepted += 1
                next_population.append(child)
            else:
                rejected += 1
                rollback += 1
                next_population.append(parent)
        population = next_population
    best = max(population, key=lambda row: row["fitness"])
    if out_path and freeze_index % 50 == 0:
        append_progress_locked(out_path, "paramwise_freeze_repair_step", method=method, quant_level=level, width=width, freeze_index=freeze_index, best_fitness=best["fitness"])
    return best["candidate"], {"attempts": attempts, "accepted": accepted, "rejected": rejected, "rollback": rollback}


def order_metadata(metadata: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    if method == "binary_distance_paramwise_freeze":
        return sorted(metadata, key=lambda row: (row["quant_error"], row["path_key"]))
    if method == "binary_sensitivity_paramwise_freeze":
        return sorted(metadata, key=lambda row: (0.65 * row["quant_error"] + 0.35 * row["normalized_sensitivity"], row["path_key"]))
    if method == "binary_qat_warmstart_paramwise_freeze":
        return sorted(metadata, key=lambda row: (row["quant_error"], row["path_key"]))
    raise ValueError(method)


def eval_accuracy_map(evals: dict[str, Any]) -> dict[str, float]:
    return {split: evals[split]["metrics"]["accuracy"] for split in eval_splits()}


def rollback_needed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    b = eval_accuracy_map(before)
    a = eval_accuracy_map(after)
    if b["heldout"] - a["heldout"] > 0.01:
        return True
    for split in ("ood", "counterfactual", "adversarial"):
        if b[split] - a[split] > 0.015:
            return True
    return False


def run_paramwise_freeze(
    method: str,
    width: int,
    initial_candidate: dict[str, Any],
    metadata: list[dict[str, Any]],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    float_eval_accuracy: float,
    int4_eval_accuracy: float,
    ternary_eval_accuracy: float,
) -> dict[str, Any]:
    out_path = Path(out) if out else None
    rng = random.Random(stable_seed(f"{method}-{width}-{settings.seeds}"))
    candidate = copy.deepcopy(initial_candidate)
    frozen: set[str] = set()
    ordered = order_metadata(metadata, method)
    if settings.paramwise_step_limit > 0:
        ordered = ordered[: min(settings.paramwise_step_limit, len(ordered))]
    before_eval = evaluate_candidate(candidate, task, settings, sample_limit=0)
    freeze_history = []
    attempts = accepted = rejected = rollback = 0
    for index, row in enumerate(ordered, start=1):
        previous_candidate = copy.deepcopy(candidate)
        previous_frozen = set(frozen)
        previous_eval = before_eval
        frozen.add(row["path_key"])
        candidate, delta = tiny_repair_step(method, "binary", width, candidate, frozen, task, settings, rng, out_path, index)
        attempts += delta["attempts"]
        accepted += delta["accepted"]
        rejected += delta["rejected"]
        rollback += delta["rollback"]
        after_eval = evaluate_candidate(candidate, task, settings, sample_limit=0)
        round_rollback = rollback_needed(previous_eval, after_eval)
        if round_rollback:
            candidate = previous_candidate
            frozen = previous_frozen
            before_eval = previous_eval
        else:
            before_eval = after_eval
        if index % 10 == 0 or index == len(ordered):
            freeze_row = {
                "freeze_step": index,
                "frozen_path": row["path_key"],
                "frozen_block": row["block"],
                "frozen_parameter_count": len(frozen),
                "frozen_parameter_ratio": round_float(len(frozen) / max(1, len(metadata))),
                "block_frozen_ratios": freeze_ratio_report(frozen, metadata),
                "validation_accuracy": before_eval["validation"]["metrics"]["accuracy"],
                "heldout_accuracy": before_eval["heldout"]["metrics"]["accuracy"],
                "ood_accuracy": before_eval["ood"]["metrics"]["accuracy"],
                "counterfactual_accuracy": before_eval["counterfactual"]["metrics"]["accuracy"],
                "adversarial_accuracy": before_eval["adversarial"]["metrics"]["accuracy"],
                "rollback_applied": round_rollback,
                "candidate_hash": candidate_hash(candidate),
            }
            freeze_history.append(freeze_row)
            if out_path:
                append_progress_locked(out_path, "paramwise_freeze_step", method=method, width=width, freeze_step=index, frozen_ratio=freeze_row["frozen_parameter_ratio"], rollback_applied=round_rollback)
    mutation_history = {
        "method": method,
        "quant_level": "binary",
        "width": width,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "freeze_round_rollback_count": sum(1 for row in freeze_history if row["rollback_applied"]),
        "history": freeze_history,
    }
    return candidate_result(
        method,
        width,
        candidate,
        task,
        settings,
        "binary_paramwise_one_by_one_freeze_with_mutation_repair",
        float_eval_accuracy,
        "binary",
        int4_eval_accuracy=int4_eval_accuracy,
        ternary_eval_accuracy=ternary_eval_accuracy,
        mutation_history=mutation_history,
        freeze_history=freeze_history,
        parameter_diff=e7a8.quantized_diff(initial_candidate, candidate),
    )


def run_direct_binary_repair(
    width: int,
    initial_candidate: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    float_eval_accuracy: float,
    int4_eval_accuracy: float,
    ternary_eval_accuracy: float,
) -> dict[str, Any]:
    final_candidate, mutation_history, _ = e7a8.repair_loop(
        "binary_direct_mutation_repair",
        "binary",
        width,
        initial_candidate,
        task,
        settings,
        out,
        frozen_keys=set(),
        generations=settings.repair_generations,
        seed_label=f"binary-direct-repair-{width}-{settings.seeds}",
    )
    return candidate_result(
        "binary_direct_mutation_repair",
        width,
        final_candidate,
        task,
        settings,
        "binary_direct_post_quant_mutation_repair",
        float_eval_accuracy,
        "binary",
        int4_eval_accuracy=int4_eval_accuracy,
        ternary_eval_accuracy=ternary_eval_accuracy,
        mutation_history=mutation_history,
        parameter_diff=e7a8.quantized_diff(initial_candidate, final_candidate),
    )


def run_worker_method(
    method: str,
    width: int,
    initial_candidate: dict[str, Any],
    metadata: list[dict[str, Any]],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    float_eval_accuracy: float,
    int4_eval_accuracy: float,
    ternary_eval_accuracy: float,
) -> dict[str, Any]:
    if method in {"binary_distance_paramwise_freeze", "binary_sensitivity_paramwise_freeze", "binary_qat_warmstart_paramwise_freeze"}:
        return run_paramwise_freeze(method, width, initial_candidate, metadata, task, settings, out, float_eval_accuracy, int4_eval_accuracy, ternary_eval_accuracy)
    if method == "binary_direct_mutation_repair":
        return run_direct_binary_repair(width, initial_candidate, task, settings, out, float_eval_accuracy, int4_eval_accuracy, ternary_eval_accuracy)
    raise ValueError(method)


def run_core(settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, settings={**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths)})
    base_settings = e7a6_settings(settings)
    task = e7a3.generate_task(e7a6.e7a3_settings(base_settings))
    if out:
        append_progress_locked(out, "task_generated", rows={split: len(task[split]["rows"]) for split in task_splits()})
    results: dict[str, Any] = {
        "methods": {},
        "float32": {},
        "metadata": {},
        "runtime": {"started_monotonic": started},
    }
    executor: ProcessPoolExecutor | None = None
    futures = {}
    if settings.execution_mode == "parallel":
        executor = ProcessPoolExecutor(max_workers=max(1, settings.parallel_workers))
        if out:
            append_progress_locked(out, "binary_worker_lanes_ready", workers=settings.parallel_workers)
    for width in settings.widths:
        trained = e7a6.train_float_core(width, task, base_settings, out)
        f_result = float_result(trained)
        results["float32"][width] = f_result
        results["methods"][("float32_matrix_core", width)] = f_result
        float_eval = result_row(f_result)["eval_accuracy"]
        int8_candidate = make_candidate("int8", trained["state_dict"], width, "int8_direct")
        int8 = candidate_result("int8_direct", width, int8_candidate, task, settings, "direct_int8_quantization", float_eval, "int8")
        results["methods"][("int8_direct", width)] = int8
        int4_candidate = make_candidate("int4", trained["state_dict"], width, "int4_direct")
        int4 = candidate_result("int4_direct", width, int4_candidate, task, settings, "direct_int4_quantization", float_eval, "int4")
        int4_eval = result_row(int4)["eval_accuracy"]
        results["methods"][("int4_direct", width)] = int4
        ternary_qat = e7a7.train_qat_core("ternary", width, trained["state_dict"], task, e7a7_settings(settings), out, 0.0, float_eval)
        ternary_qat["method"] = "ternary_qat_reference"
        ternary_qat["system"] = "ternary_qat_reference"
        ternary_qat["float_eval_accuracy"] = float_eval
        ternary_qat["int4_eval_accuracy"] = int4_eval
        ternary_qat["bit_cost"] = bit_cost(ternary_qat["candidate"], ternary_qat["parameter_count"])
        ternary_eval = result_row(ternary_qat)["eval_accuracy"]
        results["methods"][("ternary_qat_reference", width)] = ternary_qat
        binary_candidate = make_candidate("binary", trained["state_dict"], width, "binary_direct")
        binary_direct = candidate_result("binary_direct", width, binary_candidate, task, settings, "direct_binary_quantization", float_eval, "binary", int4_eval, ternary_eval)
        results["methods"][("binary_direct", width)] = binary_direct
        binary_qat = e7a7.train_qat_core("binary", width, trained["state_dict"], task, e7a7_settings(settings), out, result_row(binary_direct)["eval_accuracy"], float_eval)
        binary_qat["method"] = "binary_qat_baseline"
        binary_qat["system"] = "binary_qat_baseline"
        binary_qat["float_eval_accuracy"] = float_eval
        binary_qat["int4_eval_accuracy"] = int4_eval
        binary_qat["ternary_eval_accuracy"] = ternary_eval
        binary_qat["bit_cost"] = bit_cost(binary_qat["candidate"], binary_qat["parameter_count"])
        results["methods"][("binary_qat_baseline", width)] = binary_qat
        best_effort = train_binary_qat_best_effort(width, trained["state_dict"], task, settings, out, float_eval, int4_eval, ternary_eval)
        results["methods"][("binary_qat_best_effort", width)] = best_effort
        sensitivity = e7a8.block_sensitivity_report("binary", trained["state_dict"], width, task, settings)
        block_sens = {block: row["sensitivity_score"] for block, row in sensitivity["rows"].items()}
        metadata = e7a8.build_path_metadata("binary", trained["state_dict"], binary_candidate, block_sens)
        results["metadata"][width] = {"sensitivity": sensitivity, "path_metadata": metadata}
        mixed_int4 = make_mixed_candidate(trained["state_dict"], width, "int4")
        results["methods"][("mixed_input_int4_state_binary_output_int4", width)] = candidate_result(
            "mixed_input_int4_state_binary_output_int4",
            width,
            mixed_int4,
            task,
            settings,
            "mixed_precision_input_int4_state_binary_output_int4",
            float_eval,
            "mixed",
            int4_eval,
            ternary_eval,
        )
        mixed_ternary = make_mixed_candidate(trained["state_dict"], width, "ternary")
        results["methods"][("mixed_input_ternary_state_binary_output_int4", width)] = candidate_result(
            "mixed_input_ternary_state_binary_output_int4",
            width,
            mixed_ternary,
            task,
            settings,
            "mixed_precision_input_ternary_state_binary_output_int4",
            float_eval,
            "mixed",
            int4_eval,
            ternary_eval,
        )
        worker_jobs = {
            "binary_distance_paramwise_freeze": binary_candidate,
            "binary_sensitivity_paramwise_freeze": binary_candidate,
            "binary_qat_warmstart_paramwise_freeze": best_effort["candidate"],
            "binary_direct_mutation_repair": binary_candidate,
        }
        for method, candidate in worker_jobs.items():
            if executor is not None:
                future = executor.submit(
                    run_worker_method,
                    method,
                    width,
                    candidate,
                    metadata,
                    task,
                    settings,
                    out.as_posix() if out else None,
                    float_eval,
                    int4_eval,
                    ternary_eval,
                )
                futures[future] = (method, width)
            else:
                results["methods"][(method, width)] = run_worker_method(method, width, candidate, metadata, task, settings, out.as_posix() if out else None, float_eval, int4_eval, ternary_eval)
        if out:
            locked_write_json(out / "partial_status" / "e7a9_width_progress.json", {"submitted_width": width, "pending_worker_jobs": len(futures)})
            append_progress_locked(out, "width_submitted", width=width, pending_worker_jobs=len(futures))
    if executor is not None:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                method, width = futures[future]
                results["methods"][(method, width)] = future.result()
                if out:
                    append_progress_locked(out, "binary_worker_joined", method=method, width=width, completed=len([key for key in results["methods"] if key[0] in {"binary_distance_paramwise_freeze", "binary_sensitivity_paramwise_freeze", "binary_qat_warmstart_paramwise_freeze", "binary_direct_mutation_repair"}]), pending=len(pending))
        executor.shutdown()
    results["runtime"]["elapsed_seconds"] = round_float(time.monotonic() - started)
    return task, results


def aggregate_results(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    systems = {method: {} for method in METHODS}
    for method in METHODS:
        for width in settings.widths:
            systems[method][str(width)] = result_row(results["methods"][(method, width)])
    best = {
        method: max(systems[method].values(), key=lambda row: row["eval_accuracy"])
        for method in METHODS
    }
    binary_methods = [method for method in METHODS if method.startswith("binary_")]
    best_binary = max((best[method] for method in binary_methods), key=lambda row: row["eval_accuracy"])
    best_mixed = max((best[method] for method in METHODS if method.startswith("mixed_")), key=lambda row: row["eval_accuracy"])
    return {
        "schema_version": "e7a9_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "best_by_eval_accuracy": best,
        "best_binary_method": best_binary,
        "best_mixed_method": best_mixed,
        "binary_competitiveness_thresholds": {
            "within_ternary": 0.015,
            "within_int4": 0.025,
        },
    }


def method_comparison_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a9_method_comparison_report_v1",
        "best_by_method": aggregate["best_by_eval_accuracy"],
        "best_binary_method": aggregate["best_binary_method"],
        "best_mixed_method": aggregate["best_mixed_method"],
    }


def freeze_schedule_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for width in settings.widths:
        rows[str(width)] = {}
        for method in ("binary_distance_paramwise_freeze", "binary_sensitivity_paramwise_freeze", "binary_qat_warmstart_paramwise_freeze"):
            result = results["methods"][(method, width)]
            rows[str(width)][method] = {
                "freeze_history": result["freeze_history"],
                "mutation_history": {key: value for key, value in result["mutation_history"].items() if key != "history"},
                "parameter_diff": result["parameter_diff"],
            }
    return {
        "schema_version": "e7a9_binary_freeze_schedule_report_v1",
        "freeze_semantics": "parameter-wise one-by-one binary freeze order with mutation repair after each frozen parameter",
        "rows": rows,
    }


def qat_upper_report(results: dict[str, Any], aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for width in settings.widths:
        rows[str(width)] = {
            "binary_qat_baseline": result_row(results["methods"][("binary_qat_baseline", width)]),
            "binary_qat_best_effort": result_row(results["methods"][("binary_qat_best_effort", width)]),
            "ternary_qat_reference": result_row(results["methods"][("ternary_qat_reference", width)]),
        }
    return {
        "schema_version": "e7a9_qat_upper_bound_report_v1",
        "best_effort_policy": "float warm-start + gradual binary ramp + STE fake quant + teacher distillation + per-channel scale",
        "rows": rows,
        "best_binary_method": aggregate["best_binary_method"],
    }


def precision_tradeoff_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    methods = ("float32_matrix_core", "int8_direct", "int4_direct", "ternary_qat_reference", "binary_qat_best_effort", "binary_distance_paramwise_freeze", "binary_qat_warmstart_paramwise_freeze")
    rows = {method: aggregate["best_by_eval_accuracy"][method] for method in methods}
    return {
        "schema_version": "e7a9_precision_tradeoff_report_v1",
        "rows": rows,
    }


def mixed_precision_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    rows = {
        "mixed_input_int4_state_binary_output_int4": aggregate["best_by_eval_accuracy"]["mixed_input_int4_state_binary_output_int4"],
        "mixed_input_ternary_state_binary_output_int4": aggregate["best_by_eval_accuracy"]["mixed_input_ternary_state_binary_output_int4"],
        "best_mixed_method": aggregate["best_mixed_method"],
    }
    return {
        "schema_version": "e7a9_mixed_precision_report_v1",
        "rows": rows,
    }


def mutation_history_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for width in settings.widths:
        for method in ("binary_distance_paramwise_freeze", "binary_sensitivity_paramwise_freeze", "binary_qat_warmstart_paramwise_freeze", "binary_direct_mutation_repair"):
            rows[f"width{width}/{method}"] = results["methods"][(method, width)]["mutation_history"]
    return {
        "schema_version": "e7a9_mutation_history_report_v1",
        "rows": rows,
    }


def no_synthetic_metric_audit(task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a9_no_synthetic_metric_audit_v1",
        "generated_from_row_level_eval": True,
        "row_counts": {split: len(task[split]["rows"]) for split in task_splits()},
        "row_level_samples_present": all(
            bool(results["methods"][("binary_qat_best_effort", width)]["evals"]["heldout"]["row_level_samples"])
            and bool(results["methods"][("binary_distance_paramwise_freeze", width)]["evals"]["heldout"]["row_level_samples"])
            for width in settings.widths
        ),
        "hardcoded_improvement_flags_present": False,
        "mutation_repair_used_optimizer_or_backprop": False,
        "qat_reference_uses_backprop_by_design": True,
        "new_architecture_added": False,
        "broad_claims_intentionally_deferred": True,
    }


def choose_decision(aggregate: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    if not audit["generated_from_row_level_eval"] or audit["hardcoded_improvement_flags_present"]:
        decision = "e7a9_invalid_artifact_detected"
    else:
        best_binary = aggregate["best_binary_method"]
        best_mixed = aggregate["best_mixed_method"]
        int4 = aggregate["best_by_eval_accuracy"]["int4_direct"]
        ternary = aggregate["best_by_eval_accuracy"]["ternary_qat_reference"]
        binary_close = (
            best_binary["eval_accuracy"] >= ternary["eval_accuracy"] - 0.015
            or best_binary["eval_accuracy"] >= int4["eval_accuracy"] - 0.025
        ) and best_binary["solve_passed"]
        mixed_beats_binary = best_mixed["eval_accuracy"] > best_binary["eval_accuracy"] + 0.005 and best_mixed["solve_passed"]
        freeze_best = max(
            (
                aggregate["best_by_eval_accuracy"][method]
                for method in ("binary_distance_paramwise_freeze", "binary_sensitivity_paramwise_freeze", "binary_qat_warmstart_paramwise_freeze")
            ),
            key=lambda row: row["eval_accuracy"],
        )
        qat_best = max(
            (aggregate["best_by_eval_accuracy"][method] for method in ("binary_qat_baseline", "binary_qat_best_effort")),
            key=lambda row: row["eval_accuracy"],
        )
        if mixed_beats_binary:
            decision = "e7a9_mixed_precision_matrix_core_preferred"
        elif freeze_best["eval_accuracy"] > qat_best["eval_accuracy"] + 0.005:
            decision = "e7a9_binary_paramwise_freeze_positive"
        elif binary_close:
            decision = "e7a9_binary_quality_competitive"
        elif qat_best["eval_accuracy"] >= freeze_best["eval_accuracy"] - 0.005:
            decision = "e7a9_qat_upper_bound_remains_preferred"
        else:
            decision = "e7a9_binary_not_quality_competitive"
    return {
        "schema_version": "e7a9_decision_v1",
        "decision": decision,
        "valid_decisions": list(VALID_DECISIONS),
        "deterministic_replay_passed": False,
        "broad_claims_intentionally_deferred": True,
    }


def task_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a9_task_generation_report_v1",
        "milestone": MILESTONE,
        "inherits_task_from": "E7A8_PROGRESSIVE_QUANT_FREEZE_PLATEAU_REPAIR",
        "base_task_inherits_from": "E7A3_NEURAL_MATRIX_SUBSTRATE_HARNESS",
        "input_dim": e7a3.INPUT_DIM,
        "class_count": e7a3.CLASS_COUNT,
        "splits": {split: {"row_count": len(data["rows"])} for split, data in task.items()},
        "seeds": list(settings.seeds),
    }


def row_samples(results: dict[str, Any], split: str, settings: Settings) -> dict[str, Any]:
    samples = {}
    for width in settings.widths:
        samples[str(width)] = {
            method: results["methods"][(method, width)]["evals"][split]["row_level_samples"]
            for method in (
                "int4_direct",
                "ternary_qat_reference",
                "binary_qat_best_effort",
                "binary_distance_paramwise_freeze",
                "binary_qat_warmstart_paramwise_freeze",
                "mixed_input_int4_state_binary_output_int4",
            )
        }
    return {
        "schema_version": "e7a9_row_level_eval_sample_v1",
        "split": split,
        "samples": samples,
    }


def runtime_report(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a9_runtime_report_v1",
        "elapsed_seconds": results["runtime"].get("elapsed_seconds"),
    }


def build_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# E7A9 Binary Freeze Policy Upper-Bound Audit Result",
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
        "## Best Methods",
        "",
        "| method | width | eval | heldout | OOD | counterfactual | adversarial | gap to int4 | gap to ternary | compression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate["best_by_eval_accuracy"][method]
        compression = row.get("bit_cost", {}).get("compression_vs_float32", 1.0)
        lines.append(
            f"| `{method}` | {row['width']} | {row['eval_accuracy']:.6f} | {row['heldout_accuracy']:.6f} | "
            f"{row['ood_accuracy']:.6f} | {row['counterfactual_accuracy']:.6f} | {row['adversarial_accuracy']:.6f} | "
            f"{row.get('gap_to_int4', 0.0):.6f} | {row.get('gap_to_ternary', 0.0):.6f} | {compression:.3f}x |"
        )
    lines.extend(
        [
            "",
            "This probe only audits the controlled low-bit matrix-core tradeoff.",
            "",
        ]
    )
    return "\n".join(lines)


def build_payloads(out: Path, task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    aggregate = aggregate_results(results, settings)
    comparison = method_comparison_report(aggregate)
    freeze = freeze_schedule_report(results, settings)
    qat = qat_upper_report(results, aggregate, settings)
    tradeoff = precision_tradeoff_report(aggregate)
    mixed = mixed_precision_report(aggregate)
    audit = no_synthetic_metric_audit(task, results, settings)
    decision = choose_decision(aggregate, audit)
    payloads: dict[str, Any] = {
        "e7a9_backend_manifest.json": {
            "schema_version": "e7a9_backend_manifest_v1",
            "milestone": MILESTONE,
            "methods": list(METHODS),
            "widths": list(settings.widths),
            "blocks": BLOCKS,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": e7a3.select_device(settings.device),
            "settings": {**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths)},
            "parallel_replay_supported": True,
            "new_architecture_added": False,
            "broad_claims_intentionally_deferred": True,
        },
        "e7a9_task_generation_report.json": task_report(task, settings),
        "e7a9_method_comparison_report.json": comparison,
        "e7a9_binary_freeze_schedule_report.json": freeze,
        "e7a9_qat_upper_bound_report.json": qat,
        "e7a9_precision_tradeoff_report.json": tradeoff,
        "e7a9_mixed_precision_report.json": mixed,
        "e7a9_mutation_history.json": mutation_history_report(results, settings),
        "e7a9_no_synthetic_metric_audit.json": audit,
        "e7a9_runtime_report.json": runtime_report(results),
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e7a9_summary_v1",
            "milestone": MILESTONE,
            "decision": decision["decision"],
            "best_binary_method": aggregate["best_binary_method"],
            "best_mixed_method": aggregate["best_mixed_method"],
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
            "broad_claims_intentionally_deferred": True,
        },
    }
    for split in eval_splits():
        payloads[f"e7a9_row_level_eval_sample_{split}.json"] = row_samples(results, split, settings)
    payloads["report.md"] = build_report(out, decision, aggregate)
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
        "schema_version": "e7a9_deterministic_replay_report_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
        "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix(),
    }
    append_progress_locked(out, "deterministic_replay_complete", internal_replay_passed=report["internal_replay_passed"])
    return report


def write_final_artifacts(out: Path, payloads: dict[str, Any], deterministic: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["e7a9_deterministic_replay_report.json"] = deterministic
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
    parser.add_argument("--widths", default="32")
    parser.add_argument("--train-rows-per-seed", type=int, default=240)
    parser.add_argument("--validation-rows-per-seed", type=int, default=100)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=100)
    parser.add_argument("--ood-rows-per-seed", type=int, default=100)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=100)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=100)
    parser.add_argument("--gradient-epochs", type=int, default=160)
    parser.add_argument("--qat-epochs", type=int, default=120)
    parser.add_argument("--best-effort-qat-epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--qat-learning-rate", type=float, default=7e-4)
    parser.add_argument("--best-effort-learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--repair-generations", type=int, default=60)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--plateau-min-delta", type=float, default=0.001)
    parser.add_argument("--paramwise-generation-per-step", type=int, default=1)
    parser.add_argument("--paramwise-step-limit", type=int, default=0)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--quant-mutation-steps", type=int, default=16)
    parser.add_argument("--matrix-steps", type=int, default=4)
    parser.add_argument("--distillation-weight", type=float, default=0.35)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
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
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        gradient_epochs=args.gradient_epochs,
        qat_epochs=args.qat_epochs,
        best_effort_qat_epochs=args.best_effort_qat_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        qat_learning_rate=args.qat_learning_rate,
        best_effort_learning_rate=args.best_effort_learning_rate,
        weight_decay=args.weight_decay,
        population_size=args.population_size,
        repair_generations=args.repair_generations,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        paramwise_generation_per_step=args.paramwise_generation_per_step,
        paramwise_step_limit=args.paramwise_step_limit,
        elite_count=args.elite_count,
        quant_mutation_steps=args.quant_mutation_steps,
        matrix_steps=args.matrix_steps,
        distillation_weight=args.distillation_weight,
        distillation_temperature=args.distillation_temperature,
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
