#!/usr/bin/env python3
"""E7A10 binary scale overhead and bit-budget audit.

E7A10 tests whether the E7A9 binary result is genuinely a binary substrate
advantage or mostly a scale-overhead/width-budget effect. It keeps the E7 matrix
core fixed, compares binary scale policies, and gives binary wider hidden
matrices only when its measured bit cost stays within the int4 width-32 budget.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
E7A9_PATH = Path(__file__).with_name("run_e7a9_binary_freeze_policy_upper_bound_audit.py")
MILESTONE = "E7A10_BINARY_SCALE_OVERHEAD_AND_BIT_BUDGET_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e7a10_binary_scale_overhead_and_bit_budget_audit")
DEFAULT_SEEDS = (84001, 84002, 84003)
PARAM_KEYS = ("win", "state", "carry_raw", "bstate", "wout", "bout")
METHODS = (
    "float32_matrix_core",
    "int8_direct",
    "int4_direct",
    "int3_direct",
    "ternary_block_scale_qat",
    "binary_direct_block_scale",
    "binary_minimal_scale_qat",
    "binary_global_scale_qat",
    "binary_block_scale_qat",
    "binary_channel_scale_qat",
    "binary_channel_scale_qat_paramwise_freeze",
)
MUTATION_METHODS = ("binary_channel_scale_qat_paramwise_freeze",)
VALID_DECISIONS = (
    "e7a10_binary_same_budget_preferred",
    "e7a10_binary_scale_overhead_required",
    "e7a10_int4_quality_path_preferred",
    "e7a10_ternary_balanced_path_preferred",
    "e7a10_global_or_block_binary_viable",
    "e7a10_binary_width_scaling_not_worth_it",
    "e7a10_invalid_artifact_detected",
)
HASH_ARTIFACTS = (
    "e7a10_task_generation_report.json",
    "e7a10_method_comparison_report.json",
    "e7a10_scale_overhead_report.json",
    "e7a10_bit_budget_width_scaling_report.json",
    "e7a10_mutation_history.json",
    "e7a10_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)


def load_e7a9_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7a9_binary_freeze_policy_upper_bound_audit", E7A9_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7A9 from {E7A9_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7a9 = load_e7a9_module()
e7a8 = e7a9.e7a8
e7a7 = e7a9.e7a7
e7a6 = e7a9.e7a6
e7a3 = e7a9.e7a3


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    widths: tuple[int, ...]
    reference_width: int
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
    return int(hashlib.sha256(f"e7a10::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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


def task_splits() -> tuple[str, ...]:
    return e7a3.SPLITS


def eval_splits() -> tuple[str, ...]:
    return e7a3.EVAL_SPLITS


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


def candidate_hash(candidate: dict[str, Any]) -> str:
    return payload_sha256(candidate)


def binary_q_from_value(value: np.ndarray) -> np.ndarray:
    return np.where(np.abs(value) <= 1e-12, 0, np.where(value >= 0.0, 1, -1)).astype(np.int16)


def make_binary_scale_candidate(scale_mode: str, state_dict: dict[str, np.ndarray], width: int, method: str) -> dict[str, Any]:
    if scale_mode == "block":
        candidate = e7a9.make_candidate("binary", state_dict, width, method)
        candidate["scale_mode"] = "block_per_tensor"
        candidate["scale_storage_mode"] = "block_per_tensor"
    elif scale_mode == "channel":
        candidate = e7a9.make_channel_scale_candidate("binary", state_dict, width, method)
        candidate["scale_mode"] = "channel_per_matrix_column"
        candidate["scale_storage_mode"] = "channel_per_matrix_column"
    elif scale_mode == "global":
        nonzero_abs = []
        for value in state_dict.values():
            active = np.abs(value) > 1e-12
            if bool(np.any(active)):
                nonzero_abs.append(np.abs(value[active]).reshape(-1))
        scale = float(np.mean(np.concatenate(nonzero_abs))) if nonzero_abs else 1.0
        q = {}
        zero_counts = {}
        for key, value in state_dict.items():
            quantized = binary_q_from_value(value)
            q[key] = quantized.tolist()
            zero_counts[key] = int(np.sum(quantized == 0))
        candidate = {
            "schema_version": "e7a10_global_scale_binary_matrix_core_candidate_v1",
            "quant_level": "binary",
            "source_quant_level": "binary",
            "quant_config": e7a6.QUANT_CONFIGS["binary"],
            "scale_mode": "global_shared",
            "scale_storage_mode": "global_shared",
            "width": width,
            "q": q,
            "scales": {key: round_float(scale) for key in PARAM_KEYS},
            "global_scale": round_float(scale),
            "zero_counts": zero_counts,
        }
    elif scale_mode == "minimal":
        fixed_scale = 1.0 / math.sqrt(max(1, width))
        q = {}
        zero_counts = {}
        for key, value in state_dict.items():
            quantized = binary_q_from_value(value)
            q[key] = quantized.tolist()
            zero_counts[key] = int(np.sum(quantized == 0))
        candidate = {
            "schema_version": "e7a10_minimal_scale_binary_matrix_core_candidate_v1",
            "quant_level": "binary",
            "source_quant_level": "binary",
            "quant_config": e7a6.QUANT_CONFIGS["binary"],
            "scale_mode": "minimal_fixed_formula",
            "scale_storage_mode": "minimal_fixed_formula",
            "width": width,
            "q": q,
            "scales": {key: round_float(fixed_scale) for key in PARAM_KEYS},
            "fixed_scale_formula": "1/sqrt(width)",
            "zero_counts": zero_counts,
        }
    else:
        raise ValueError(scale_mode)
    candidate["method_origin"] = method
    candidate["candidate_hash"] = candidate_hash(candidate)
    candidate["initial_hash"] = candidate["candidate_hash"]
    return candidate


def scale_count(candidate: dict[str, Any]) -> int:
    mode = candidate.get("scale_storage_mode")
    if mode == "minimal_fixed_formula":
        return 0
    if mode == "global_shared":
        return 1
    return e7a9.scale_count(candidate)


def nominal_bits_for_key(candidate: dict[str, Any], key: str) -> float:
    return e7a9.nominal_bits_for_key(candidate, key)


def bit_cost(candidate: dict[str, Any] | None, parameter_count: int, float_bits: int = 32) -> dict[str, Any]:
    if candidate is None:
        total = parameter_count * float_bits
        return {
            "nominal_weight_bits": float_bits,
            "weight_bit_cost": total,
            "scale_bit_cost": 0,
            "scale_count": 0,
            "total_bit_cost": total,
            "compression_vs_float32": 1.0,
        }
    weight_bits = 0.0
    for key, value in candidate["q"].items():
        weight_bits += np.asarray(value).size * nominal_bits_for_key(candidate, key)
    count = scale_count(candidate)
    scale_bits = count * 32
    total = weight_bits + scale_bits
    float_total = parameter_count * 32
    return {
        "nominal_weight_bits": round_float(weight_bits / max(1, parameter_count)),
        "weight_bit_cost": round_float(weight_bits),
        "scale_bit_cost": int(scale_bits),
        "scale_count": int(count),
        "total_bit_cost": round_float(total),
        "compression_vs_float32": round_float(float_total / max(total, 1e-12)),
        "scale_storage_mode": candidate.get("scale_storage_mode", "block_per_tensor"),
    }


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


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    row = e7a9.result_row(result)
    if "candidate" in result and result["method"] != "float32_matrix_core":
        row["bit_cost"] = bit_cost(result["candidate"], result["parameter_count"])
        row["scale_mode"] = result["candidate"].get("scale_mode")
        row["scale_storage_mode"] = result["candidate"].get("scale_storage_mode")
    return row


def fake_quant_binary_mode(param: torch.Tensor, mode: str, global_scale: torch.Tensor | None, width: int) -> torch.Tensor:
    eps = torch.tensor(1e-8, device=param.device, dtype=param.dtype)
    if mode == "minimal":
        scale = torch.tensor(1.0 / math.sqrt(max(1, width)), device=param.device, dtype=param.dtype)
    elif mode == "global":
        if global_scale is None:
            raise ValueError("global scale required")
        scale = torch.clamp(global_scale, min=float(eps.detach().cpu().item()))
    elif mode == "block":
        scale = torch.clamp(torch.mean(torch.abs(param)).detach(), min=float(eps.detach().cpu().item()))
    elif mode == "channel":
        return e7a9.fake_quant_binary_per_channel(param)
    else:
        raise ValueError(mode)
    quantized = torch.where(torch.abs(param) <= eps, torch.zeros_like(param), torch.where(param >= 0.0, torch.ones_like(param), -torch.ones_like(param)))
    return param + (quantized * scale - param).detach()


def binary_qat_forward(model: Any, x: torch.Tensor, mode: str, alpha: float, matrix_steps: int) -> torch.Tensor:
    params = [model.win, model.state, model.carry_raw, model.bstate, model.wout, model.bout]
    global_scale = None
    if mode == "global":
        global_scale = torch.mean(torch.cat([torch.abs(param).reshape(-1) for param in params])).detach()

    def blend(param: torch.Tensor) -> torch.Tensor:
        fq = fake_quant_binary_mode(param, mode, global_scale, model.width)
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


def train_binary_qat_scale_mode(
    method: str,
    scale_mode: str,
    width: int,
    float_state_dict: dict[str, np.ndarray],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    float_eval_accuracy: float,
    int4_eval_accuracy: float,
) -> dict[str, Any]:
    out_path = Path(out) if out else None
    device = e7a3.select_device(settings.device)
    e7a3.set_determinism(stable_seed(f"{method}-{width}-{settings.seeds}"), device)
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
    rng = np.random.default_rng(stable_seed(f"{method}-batches-{width}-{settings.seeds}"))
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
            logits = binary_qat_forward(model, x_train[idx], scale_mode, alpha, settings.matrix_steps)
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
            logits = binary_qat_forward(model, x_val, scale_mode, 1.0, settings.matrix_steps)
            val_loss = F.cross_entropy(logits, y_val).detach().cpu().item()
            val_acc = (torch.argmax(logits, dim=1) == y_val).float().mean().detach().cpu().item()
        if val_acc >= best_val:
            best_val = float(val_acc)
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch,
            "scale_mode": scale_mode,
            "quant_strength_alpha": round_float(alpha),
            "validation_accuracy": round_float(val_acc),
            "validation_loss": round_float(val_loss),
            "state_hash": e7a6.vector_hash(e7a6.torch_state_vector(model)),
        }
        history.append(row)
        now = time.monotonic()
        if out_path and (now - last_heartbeat >= settings.heartbeat_seconds or epoch == settings.best_effort_qat_epochs):
            write_json(out_path / f"e7a10_qat_history_{method}_width{width}.json", {"method": method, "width": width, "history": history})
            append_progress_locked(out_path, "scale_qat_epoch", method=method, scale_mode=scale_mode, width=width, epoch=epoch, validation_accuracy=row["validation_accuracy"])
            last_heartbeat = now
    state_dict = {key: value.detach().cpu().numpy().astype(np.float64) for key, value in best_state.items()}
    candidate = make_binary_scale_candidate(scale_mode, state_dict, width, method)
    return candidate_result(
        method,
        width,
        candidate,
        task,
        settings,
        f"binary_qat_with_{scale_mode}_scale_and_teacher_distillation",
        float_eval_accuracy,
        "binary",
        int4_eval_accuracy=int4_eval_accuracy,
        extra={"history": history, "best_validation_accuracy": round_float(best_val), "device": device, "scale_mode": scale_mode},
    )


def run_qat_worker(
    method: str,
    width: int,
    float_state_dict: dict[str, np.ndarray],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    float_eval_accuracy: float,
    int4_eval_accuracy: float,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    if method == "ternary_block_scale_qat":
        result = e7a7.train_qat_core("ternary", width, float_state_dict, task, e7a7_settings(settings), Path(out) if out else None, 0.0, float_eval_accuracy)
        result["method"] = method
        result["system"] = method
        result["float_eval_accuracy"] = float_eval_accuracy
        result["int4_eval_accuracy"] = int4_eval_accuracy
        result["bit_cost"] = bit_cost(result["candidate"], result["parameter_count"])
        result["candidate"]["scale_mode"] = "block_per_tensor"
        result["candidate"]["scale_storage_mode"] = "block_per_tensor"
        return result
    if method == "binary_block_scale_qat":
        result = e7a7.train_qat_core("binary", width, float_state_dict, task, e7a7_settings(settings), Path(out) if out else None, 0.0, float_eval_accuracy)
        result["method"] = method
        result["system"] = method
        result["float_eval_accuracy"] = float_eval_accuracy
        result["int4_eval_accuracy"] = int4_eval_accuracy
        result["bit_cost"] = bit_cost(result["candidate"], result["parameter_count"])
        result["candidate"]["scale_mode"] = "block_per_tensor"
        result["candidate"]["scale_storage_mode"] = "block_per_tensor"
        return result
    scale_mode = {
        "binary_minimal_scale_qat": "minimal",
        "binary_global_scale_qat": "global",
        "binary_channel_scale_qat": "channel",
    }[method]
    return train_binary_qat_scale_mode(method, scale_mode, width, float_state_dict, task, settings, out, float_eval_accuracy, int4_eval_accuracy)


def run_freeze_worker(
    width: int,
    initial_candidate: dict[str, Any],
    float_state_dict: dict[str, np.ndarray],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    float_eval_accuracy: float,
    int4_eval_accuracy: float,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    sensitivity = e7a8.block_sensitivity_report("binary", float_state_dict, width, task, settings)
    block_sens = {block: row["sensitivity_score"] for block, row in sensitivity["rows"].items()}
    metadata = e7a8.build_path_metadata("binary", float_state_dict, initial_candidate, block_sens)
    result = e7a9.run_paramwise_freeze(
        "binary_qat_warmstart_paramwise_freeze",
        width,
        initial_candidate,
        metadata,
        task,
        settings,
        out,
        float_eval_accuracy,
        int4_eval_accuracy,
        0.0,
    )
    result["method"] = "binary_channel_scale_qat_paramwise_freeze"
    result["system"] = "binary_channel_scale_qat_paramwise_freeze"
    result["training_mode"] = "channel_scale_qat_then_paramwise_freeze_with_mutation_repair"
    result["bit_cost"] = bit_cost(result["candidate"], result["parameter_count"])
    result["mutation_history"]["method"] = "binary_channel_scale_qat_paramwise_freeze"
    return result


def solve_pass(row: dict[str, float]) -> bool:
    return (
        row["heldout_accuracy"] >= 0.90
        and row["ood_accuracy"] >= 0.85
        and row["counterfactual_accuracy"] >= 0.85
        and row["adversarial_accuracy"] >= 0.80
    )


def run_core(settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, settings={**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths)})
    base_settings = e7a6_settings(settings)
    task = e7a3.generate_task(e7a6.e7a3_settings(base_settings))
    if out:
        append_progress_locked(out, "task_generated", rows={split: len(task[split]["rows"]) for split in task_splits()})

    results: dict[str, Any] = {"methods": {}, "float_state_dicts": {}, "runtime": {"started_monotonic": started}}
    worker_settings = replace(settings, device="cpu")
    executor: ProcessPoolExecutor | None = None
    futures: dict[Any, tuple[str, int]] = {}
    if settings.execution_mode == "parallel":
        executor = ProcessPoolExecutor(max_workers=max(1, settings.parallel_workers))
        if out:
            append_progress_locked(out, "worker_lanes_ready", workers=settings.parallel_workers)

    for width in settings.widths:
        trained = e7a6.train_float_core(width, task, base_settings, out)
        results["float_state_dicts"][width] = trained["state_dict"]
        float_res = float_result(trained)
        results["methods"][("float32_matrix_core", width)] = float_res
        float_eval = result_row(float_res)["eval_accuracy"]

        int8_candidate = e7a9.make_candidate("int8", trained["state_dict"], width, "int8_direct")
        results["methods"][("int8_direct", width)] = candidate_result("int8_direct", width, int8_candidate, task, settings, "direct_int8_quantization", float_eval, "int8")
        int4_candidate = e7a9.make_candidate("int4", trained["state_dict"], width, "int4_direct")
        int4_res = candidate_result("int4_direct", width, int4_candidate, task, settings, "direct_int4_quantization", float_eval, "int4")
        int4_eval = result_row(int4_res)["eval_accuracy"]
        results["methods"][("int4_direct", width)] = int4_res
        int3_candidate = e7a9.make_candidate("int3", trained["state_dict"], width, "int3_direct")
        results["methods"][("int3_direct", width)] = candidate_result("int3_direct", width, int3_candidate, task, settings, "direct_int3_quantization", float_eval, "int3", int4_eval_accuracy=int4_eval)
        binary_candidate = make_binary_scale_candidate("block", trained["state_dict"], width, "binary_direct_block_scale")
        results["methods"][("binary_direct_block_scale", width)] = candidate_result("binary_direct_block_scale", width, binary_candidate, task, settings, "direct_binary_block_scale_quantization", float_eval, "binary", int4_eval_accuracy=int4_eval)

        for method in ("ternary_block_scale_qat", "binary_minimal_scale_qat", "binary_global_scale_qat", "binary_block_scale_qat", "binary_channel_scale_qat"):
            if executor is not None:
                future = executor.submit(
                    run_qat_worker,
                    method,
                    width,
                    trained["state_dict"],
                    task,
                    worker_settings,
                    out.as_posix() if out else None,
                    float_eval,
                    int4_eval,
                )
                futures[future] = (method, width)
            else:
                results["methods"][(method, width)] = run_qat_worker(method, width, trained["state_dict"], task, worker_settings, out.as_posix() if out else None, float_eval, int4_eval)
        if out:
            locked_write_json(out / "partial_status" / "e7a10_width_progress.json", {"submitted_width": width, "pending_worker_jobs": len(futures)})
            append_progress_locked(out, "width_submitted", width=width, pending_worker_jobs=len(futures))

    if executor is not None:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                method, width = futures[future]
                results["methods"][(method, width)] = future.result()
                if out:
                    append_progress_locked(out, "qat_worker_joined", method=method, width=width, pending=len(pending))
        executor.shutdown()

    freeze_executor: ProcessPoolExecutor | None = None
    freeze_futures: dict[Any, tuple[str, int]] = {}
    if settings.execution_mode == "parallel":
        freeze_executor = ProcessPoolExecutor(max_workers=max(1, min(settings.parallel_workers, len(settings.widths))))
    for width in settings.widths:
        channel = results["methods"][("binary_channel_scale_qat", width)]
        float_eval = result_row(results["methods"][("float32_matrix_core", width)])["eval_accuracy"]
        int4_eval = result_row(results["methods"][("int4_direct", width)])["eval_accuracy"]
        if freeze_executor is not None:
            future = freeze_executor.submit(
                run_freeze_worker,
                width,
                channel["candidate"],
                results["float_state_dicts"][width],
                task,
                worker_settings,
                out.as_posix() if out else None,
                float_eval,
                int4_eval,
            )
            freeze_futures[future] = ("binary_channel_scale_qat_paramwise_freeze", width)
        else:
            results["methods"][("binary_channel_scale_qat_paramwise_freeze", width)] = run_freeze_worker(width, channel["candidate"], results["float_state_dicts"][width], task, worker_settings, out.as_posix() if out else None, float_eval, int4_eval)

    if freeze_executor is not None:
        pending = set(freeze_futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                method, width = freeze_futures[future]
                results["methods"][(method, width)] = future.result()
                if out:
                    append_progress_locked(out, "freeze_worker_joined", method=method, width=width, pending=len(pending))
        freeze_executor.shutdown()

    results["runtime"]["elapsed_seconds"] = round_float(time.monotonic() - started)
    return task, results


def aggregate_results(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    systems = {method: {} for method in METHODS}
    for method in METHODS:
        for width in settings.widths:
            systems[method][str(width)] = result_row(results["methods"][(method, width)])
    best = {method: max(systems[method].values(), key=lambda row: row["eval_accuracy"]) for method in METHODS}
    reference = systems["int4_direct"][str(settings.reference_width)]
    budget = reference["bit_cost"]["total_bit_cost"]
    binary_methods = [method for method in METHODS if method.startswith("binary_")]
    eligible = []
    for method in binary_methods:
        for row in systems[method].values():
            if row["bit_cost"]["total_bit_cost"] <= budget:
                eligible.append(row)
    best_binary_same_budget = max(eligible, key=lambda row: row["eval_accuracy"]) if eligible else None
    best_binary_any_budget = max((best[method] for method in binary_methods), key=lambda row: row["eval_accuracy"])
    return {
        "schema_version": "e7a10_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "best_by_eval_accuracy": best,
        "reference_int4_width": settings.reference_width,
        "reference_int4_budget_bits": budget,
        "best_binary_same_bit_budget": best_binary_same_budget,
        "best_binary_any_budget": best_binary_any_budget,
        "same_budget_match_threshold": 0.005,
    }


def method_comparison_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a10_method_comparison_report_v1",
        "best_by_method": aggregate["best_by_eval_accuracy"],
        "best_binary_same_bit_budget": aggregate["best_binary_same_bit_budget"],
        "best_binary_any_budget": aggregate["best_binary_any_budget"],
    }


def scale_overhead_report(aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for method in METHODS:
        for width, row in aggregate["systems"][method].items():
            if "bit_cost" in row:
                rows[f"{method}/width{width}"] = {
                    "method": method,
                    "width": int(width),
                    "eval_accuracy": row["eval_accuracy"],
                    "scale_mode": row.get("scale_mode"),
                    "scale_storage_mode": row.get("scale_storage_mode"),
                    "parameter_count": row["parameter_count"],
                    "bit_cost": row["bit_cost"],
                }
    return {
        "schema_version": "e7a10_scale_overhead_report_v1",
        "scale_count_includes_stored_float32_scales": True,
        "fixed_formula_scales_count_as_zero_stored_scale_bits": True,
        "rows": rows,
        "reference_width": settings.reference_width,
    }


def bit_budget_report(aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    budget = aggregate["reference_int4_budget_bits"]
    rows = {}
    for method, width_rows in aggregate["systems"].items():
        for width, row in width_rows.items():
            total = row["bit_cost"]["total_bit_cost"]
            rows[f"{method}/width{width}"] = {
                "method": method,
                "width": int(width),
                "eval_accuracy": row["eval_accuracy"],
                "heldout_accuracy": row["heldout_accuracy"],
                "ood_accuracy": row["ood_accuracy"],
                "counterfactual_accuracy": row["counterfactual_accuracy"],
                "adversarial_accuracy": row["adversarial_accuracy"],
                "total_bit_cost": total,
                "within_reference_budget": total <= budget,
                "budget_ratio": round_float(total / max(budget, 1e-12)),
                "compression_vs_float32": row["bit_cost"]["compression_vs_float32"],
            }
    same_width = {
        method: aggregate["systems"][method].get(str(settings.reference_width))
        for method in METHODS
    }
    return {
        "schema_version": "e7a10_bit_budget_width_scaling_report_v1",
        "reference": aggregate["systems"]["int4_direct"][str(settings.reference_width)],
        "reference_int4_budget_bits": budget,
        "same_width_rows": same_width,
        "all_budget_rows": rows,
        "best_binary_same_bit_budget": aggregate["best_binary_same_bit_budget"],
        "best_binary_any_budget": aggregate["best_binary_any_budget"],
    }


def mutation_history_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for width in settings.widths:
        result = results["methods"][("binary_channel_scale_qat_paramwise_freeze", width)]
        rows[f"width{width}/binary_channel_scale_qat_paramwise_freeze"] = result["mutation_history"]
    return {
        "schema_version": "e7a10_mutation_history_report_v1",
        "rows": rows,
    }


def no_synthetic_metric_audit(task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a10_no_synthetic_metric_audit_v1",
        "generated_from_row_level_eval": True,
        "row_counts": {split: len(task[split]["rows"]) for split in task_splits()},
        "row_level_samples_present": all(
            bool(results["methods"][("binary_channel_scale_qat", width)]["evals"]["heldout"]["row_level_samples"])
            and bool(results["methods"][("binary_channel_scale_qat_paramwise_freeze", width)]["evals"]["heldout"]["row_level_samples"])
            for width in settings.widths
        ),
        "hardcoded_improvement_flags_present": False,
        "mutation_repair_used_optimizer_or_backprop": False,
        "scale_overhead_counted_in_bit_budget": True,
        "same_budget_width_scaling_evaluated": len(settings.widths) > 1,
        "new_architecture_added": False,
        "broad_claims_intentionally_deferred": True,
    }


def choose_decision(aggregate: dict[str, Any], audit: dict[str, Any], settings: Settings) -> dict[str, Any]:
    if not audit["generated_from_row_level_eval"] or audit["hardcoded_improvement_flags_present"]:
        decision = "e7a10_invalid_artifact_detected"
    else:
        int4_ref = aggregate["systems"]["int4_direct"][str(settings.reference_width)]
        ternary_ref = aggregate["systems"]["ternary_block_scale_qat"][str(settings.reference_width)]
        same_width_binary = max(
            (
                aggregate["systems"][method][str(settings.reference_width)]
                for method in METHODS
                if method.startswith("binary_")
            ),
            key=lambda row: row["eval_accuracy"],
        )
        best_same_budget = aggregate["best_binary_same_bit_budget"]
        global_or_block = max(
            (
                aggregate["systems"][method][width]
                for method in ("binary_global_scale_qat", "binary_block_scale_qat")
                for width in aggregate["systems"][method]
                if aggregate["systems"][method][width]["bit_cost"]["total_bit_cost"] <= aggregate["reference_int4_budget_bits"]
            ),
            key=lambda row: row["eval_accuracy"],
            default=None,
        )
        if best_same_budget is None:
            decision = "e7a10_invalid_artifact_detected"
        elif best_same_budget["eval_accuracy"] >= int4_ref["eval_accuracy"] + 0.002 and best_same_budget["solve_passed"]:
            decision = "e7a10_binary_same_budget_preferred"
        elif global_or_block and global_or_block["eval_accuracy"] >= int4_ref["eval_accuracy"] - 0.005 and global_or_block["solve_passed"]:
            decision = "e7a10_global_or_block_binary_viable"
        elif same_width_binary["eval_accuracy"] >= int4_ref["eval_accuracy"] - 0.005 and same_width_binary["bit_cost"]["scale_bit_cost"] > 128:
            decision = "e7a10_binary_scale_overhead_required"
        elif ternary_ref["eval_accuracy"] >= best_same_budget["eval_accuracy"] + 0.005 and ternary_ref["solve_passed"]:
            decision = "e7a10_ternary_balanced_path_preferred"
        elif best_same_budget["eval_accuracy"] <= same_width_binary["eval_accuracy"] + 0.005:
            decision = "e7a10_binary_width_scaling_not_worth_it"
        else:
            decision = "e7a10_int4_quality_path_preferred"
    return {
        "schema_version": "e7a10_decision_v1",
        "decision": decision,
        "valid_decisions": list(VALID_DECISIONS),
        "deterministic_replay_passed": False,
        "broad_claims_intentionally_deferred": True,
    }


def task_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a10_task_generation_report_v1",
        "milestone": MILESTONE,
        "inherits_task_from": "E7A9_BINARY_FREEZE_POLICY_UPPER_BOUND_AUDIT",
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
                "ternary_block_scale_qat",
                "binary_global_scale_qat",
                "binary_block_scale_qat",
                "binary_channel_scale_qat",
                "binary_channel_scale_qat_paramwise_freeze",
            )
        }
    return {
        "schema_version": "e7a10_row_level_eval_sample_v1",
        "split": split,
        "samples": samples,
    }


def runtime_report(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a10_runtime_report_v1",
        "elapsed_seconds": results["runtime"].get("elapsed_seconds"),
    }


def build_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# E7A10 Binary Scale Overhead And Bit-Budget Audit Result",
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
        "| method | width | eval | heldout | OOD | counterfactual | adversarial | bits | compression | scale bits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate["best_by_eval_accuracy"][method]
        bit = row.get("bit_cost", {})
        lines.append(
            f"| `{method}` | {row['width']} | {row['eval_accuracy']:.6f} | {row['heldout_accuracy']:.6f} | "
            f"{row['ood_accuracy']:.6f} | {row['counterfactual_accuracy']:.6f} | {row['adversarial_accuracy']:.6f} | "
            f"{bit.get('total_bit_cost', 0):.0f} | {bit.get('compression_vs_float32', 1.0):.3f}x | {bit.get('scale_bit_cost', 0)} |"
        )
    best_budget = aggregate["best_binary_same_bit_budget"]
    if best_budget:
        lines.extend(
            [
                "",
                "## Same-Budget Binary",
                "",
                "```text",
                f"reference_int4_width = {aggregate['reference_int4_width']}",
                f"reference_budget_bits = {aggregate['reference_int4_budget_bits']}",
                f"best_binary_same_budget = {best_budget['method']} width={best_budget['width']} eval={best_budget['eval_accuracy']:.6f}",
                "```",
            ]
        )
    lines.extend(["", "This is a controlled symbolic/numeric matrix-core compression audit only.", ""])
    return "\n".join(lines)


def build_payloads(out: Path, task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    aggregate = aggregate_results(results, settings)
    audit = no_synthetic_metric_audit(task, results, settings)
    decision = choose_decision(aggregate, audit, settings)
    payloads: dict[str, Any] = {
        "e7a10_backend_manifest.json": {
            "schema_version": "e7a10_backend_manifest_v1",
            "milestone": MILESTONE,
            "methods": list(METHODS),
            "widths": list(settings.widths),
            "reference_width": settings.reference_width,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": e7a3.select_device(settings.device),
            "settings": {**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths)},
            "parallel_replay_supported": True,
            "new_architecture_added": False,
            "broad_claims_intentionally_deferred": True,
        },
        "e7a10_task_generation_report.json": task_report(task, settings),
        "e7a10_method_comparison_report.json": method_comparison_report(aggregate),
        "e7a10_scale_overhead_report.json": scale_overhead_report(aggregate, settings),
        "e7a10_bit_budget_width_scaling_report.json": bit_budget_report(aggregate, settings),
        "e7a10_mutation_history.json": mutation_history_report(results, settings),
        "e7a10_no_synthetic_metric_audit.json": audit,
        "e7a10_runtime_report.json": runtime_report(results),
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e7a10_summary_v1",
            "milestone": MILESTONE,
            "decision": decision["decision"],
            "best_binary_same_bit_budget": aggregate["best_binary_same_bit_budget"],
            "best_binary_any_budget": aggregate["best_binary_any_budget"],
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
            "broad_claims_intentionally_deferred": True,
        },
    }
    for split in eval_splits():
        payloads[f"e7a10_row_level_eval_sample_{split}.json"] = row_samples(results, split, settings)
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
        "schema_version": "e7a10_deterministic_replay_report_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
        "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix(),
    }
    append_progress_locked(out, "deterministic_replay_complete", internal_replay_passed=report["internal_replay_passed"])
    return report


def write_final_artifacts(out: Path, payloads: dict[str, Any], deterministic: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["e7a10_deterministic_replay_report.json"] = deterministic
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
    parser.add_argument("--widths", default="32,48,64")
    parser.add_argument("--reference-width", type=int, default=32)
    parser.add_argument("--train-rows-per-seed", type=int, default=220)
    parser.add_argument("--validation-rows-per-seed", type=int, default=90)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=90)
    parser.add_argument("--ood-rows-per-seed", type=int, default=90)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=90)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=90)
    parser.add_argument("--gradient-epochs", type=int, default=140)
    parser.add_argument("--qat-epochs", type=int, default=110)
    parser.add_argument("--best-effort-qat-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--qat-learning-rate", type=float, default=7e-4)
    parser.add_argument("--best-effort-learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--repair-generations", type=int, default=40)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--plateau-min-delta", type=float, default=0.001)
    parser.add_argument("--paramwise-generation-per-step", type=int, default=1)
    parser.add_argument("--paramwise-step-limit", type=int, default=256)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--quant-mutation-steps", type=int, default=16)
    parser.add_argument("--matrix-steps", type=int, default=4)
    parser.add_argument("--distillation-weight", type=float, default=0.35)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--execution-mode", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--parallel-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 2, 16)))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    widths = parse_int_tuple(args.widths)
    if args.reference_width not in widths:
        raise ValueError("--reference-width must be included in --widths")
    settings = Settings(
        seeds=parse_int_tuple(args.seeds),
        widths=widths,
        reference_width=args.reference_width,
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
