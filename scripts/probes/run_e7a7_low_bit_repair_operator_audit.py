#!/usr/bin/env python3
"""E7A7 low-bit repair operator audit.

E7A6 showed that the plain matrix-core survives int8/int4, degrades at int3,
and breaks harder at ternary/binary while mutation repair gives only partial
recovery. E7A7 isolates where that low-bit damage comes from and whether the
repair operator, not the matrix-core, is the current bottleneck.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import hashlib
import importlib.util
import json
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
E7A6_PATH = Path(__file__).with_name("run_e7a6_quantization_stress_and_repair_limit.py")
MILESTONE = "E7A7_LOW_BIT_REPAIR_OPERATOR_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e7a7_low_bit_repair_operator_audit")
DEFAULT_SEEDS = (81001, 81002, 81003)

AUDIT_LEVELS = ("int3", "ternary", "binary")
LOW_BIT_LEVELS = ("ternary", "binary")
BLOCKS: dict[str, tuple[str, ...]] = {
    "input_projection": ("win",),
    "recurrent_state": ("state",),
    "carry_gate": ("carry_raw",),
    "state_bias": ("bstate",),
    "output_head": ("wout", "bout"),
}
SYSTEMS = (
    "float32_matrix_core",
    "low_bit_no_repair",
    "block_only_low_bit",
    "block_restored_to_int8",
    "full_mutation_repair",
    "targeted_block_mutation_repair",
    "sensitive_pair_mutation_repair",
    "quantization_aware_training",
)
VALID_DECISIONS = (
    "e7a7_sensitive_block_repair_sufficient",
    "e7a7_output_or_state_bottleneck_identified",
    "e7a7_qat_preferred_over_post_repair",
    "e7a7_repair_operator_bottleneck_detected",
    "e7a7_low_bit_information_limit_detected",
    "e7a7_low_bit_breakpoint_audit_complete",
    "e7a7_invalid_artifact_detected",
)
HASH_ARTIFACTS = (
    "e7a7_task_generation_report.json",
    "e7a7_block_damage_report.json",
    "e7a7_block_restore_report.json",
    "e7a7_repair_operator_report.json",
    "e7a7_qat_report.json",
    "e7a7_low_bit_bottleneck_report.json",
    "e7a7_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)


def load_e7a6_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7a6_quantization_stress_and_repair_limit", E7A6_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7A6 from {E7A6_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7a6 = load_e7a6_module()
e7a3 = e7a6.e7a3


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
    batch_size: int
    learning_rate: float
    qat_learning_rate: float
    weight_decay: float
    population_size: int
    repair_generations: int
    elite_count: int
    quant_mutation_steps: int
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
    return int(hashlib.sha256(f"e7a7::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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


def write_text(path: Path, text: str) -> None:
    e7a6.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7a6.write_json(path, payload)


def locked_write_json(path: Path, payload: Any) -> None:
    e7a6.locked_write_json(path, payload)


def append_progress_locked(out: Path, event: str, **details: Any) -> None:
    e7a6.append_progress_locked(out, event, **details)


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


def task_splits() -> tuple[str, ...]:
    return e7a3.SPLITS


def eval_splits() -> tuple[str, ...]:
    return e7a3.EVAL_SPLITS


def candidate_hash(candidate: dict[str, Any]) -> str:
    return e7a6.candidate_hash(candidate)


def split_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {split: result["evals"][split]["metrics"] for split in task_splits()}


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    row = e7a6.result_row(result)
    for key in ("repair_scope", "repair_block", "repair_pair", "audit_mode"):
        if key in result:
            row[key] = result[key]
    if "baseline_low_bit_eval_accuracy" in result:
        row["baseline_low_bit_eval_accuracy"] = result["baseline_low_bit_eval_accuracy"]
        row["gain_vs_low_bit"] = round_float(row["eval_accuracy"] - result["baseline_low_bit_eval_accuracy"])
    if "float_eval_accuracy" in result:
        row["float_eval_accuracy"] = result["float_eval_accuracy"]
        row["drop_vs_float"] = round_float(result["float_eval_accuracy"] - row["eval_accuracy"])
    return row


def all_quantized_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for block_keys in BLOCKS.values():
        keys.extend(block_keys)
    return tuple(keys)


def keys_for_blocks(block_names: tuple[str, ...]) -> tuple[str, ...]:
    keys: list[str] = []
    for block in block_names:
        keys.extend(BLOCKS[block])
    return tuple(keys)


def update_candidate_metadata(candidate: dict[str, Any], schema: str, audit_mode: str, quant_level: str) -> dict[str, Any]:
    updated = copy.deepcopy(candidate)
    updated["schema_version"] = schema
    updated["audit_mode"] = audit_mode
    updated["quant_level"] = quant_level
    updated["zero_counts"] = {
        key: int(np.sum(np.asarray(value, dtype=np.int16) == 0))
        for key, value in updated["q"].items()
    }
    updated["candidate_hash"] = candidate_hash(updated)
    return updated


def mixed_candidate(
    quant_level: str,
    low_bit: dict[str, Any],
    int8: dict[str, Any],
    mode: str,
    block_name: str | None,
) -> dict[str, Any]:
    if mode == "full_low_bit":
        base = low_bit
    elif mode == "block_only_low_bit":
        if block_name is None:
            raise ValueError("block_only_low_bit requires a block")
        base = copy.deepcopy(int8)
        for key in BLOCKS[block_name]:
            base["q"][key] = copy.deepcopy(low_bit["q"][key])
            base["scales"][key] = low_bit["scales"][key]
    elif mode == "block_restored_to_int8":
        if block_name is None:
            raise ValueError("block_restored_to_int8 requires a block")
        base = copy.deepcopy(low_bit)
        for key in BLOCKS[block_name]:
            base["q"][key] = copy.deepcopy(int8["q"][key])
            base["scales"][key] = int8["scales"][key]
    else:
        raise ValueError(f"unknown mixed candidate mode: {mode}")
    candidate = update_candidate_metadata(
        base,
        "e7a7_mixed_quantized_plain_matrix_core_candidate_v1",
        mode if block_name is None else f"{mode}:{block_name}",
        quant_level,
    )
    candidate["source_quant_level"] = quant_level
    candidate["mixed_with_int8_reference"] = mode != "full_low_bit"
    return candidate


def quantized_paths_for_keys(candidate: dict[str, Any], allowed_keys: tuple[str, ...]) -> list[tuple[tuple[Any, ...], int]]:
    allowed = set(allowed_keys)
    rows: list[tuple[tuple[Any, ...], int]] = []
    for key, value in candidate["q"].items():
        if key not in allowed:
            continue
        arr = np.asarray(value, dtype=np.int16)
        for index in np.ndindex(arr.shape):
            rows.append((("q", key, *index), int(arr[index])))
    return rows


def mutate_quantized_targeted(
    candidate: dict[str, Any],
    settings: Settings,
    rng: random.Random,
    allowed_keys: tuple[str, ...],
) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    level = child["source_quant_level"]
    config = e7a6.QUANT_CONFIGS[level]
    paths = quantized_paths_for_keys(child, allowed_keys)
    for path, value in rng.sample(paths, k=min(settings.quant_mutation_steps, len(paths))):
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
        e7a6.set_quantized_path(child, path, new_value)
    child["zero_counts"] = {
        key: int(np.sum(np.asarray(value, dtype=np.int16) == 0))
        for key, value in child["q"].items()
    }
    child["candidate_hash"] = candidate_hash(child)
    return child


def score_candidate(candidate: dict[str, Any], task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    evals = {
        split: e7a3.evaluate_logits(e7a6.quantized_forward(candidate, data["x"], settings.matrix_steps), data, sample_limit=0)
        for split, data in {"train": task["train"], "validation": task["validation"]}.items()
    }
    train = evals["train"]["metrics"]
    val = evals["validation"]["metrics"]
    nonzero_ratio = e7a6.quantized_nonzero_count(candidate) / max(1, e7a6.quantized_parameter_count(candidate))
    fitness = 0.20 * train["accuracy"] + 0.80 * val["accuracy"] - 0.02 * val["cross_entropy"] - 0.001 * nonzero_ratio
    return {"candidate": candidate, "evals": evals, "fitness": round_float(fitness)}


def quantized_eval_result(
    system: str,
    quant_level: str,
    width: int,
    candidate: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    training_mode: str,
    float_eval_accuracy: float | None = None,
    baseline_low_bit_eval_accuracy: float | None = None,
    audit_mode: str | None = None,
) -> dict[str, Any]:
    evals = e7a6.evaluate_quantized_candidate(candidate, task, settings.matrix_steps)
    result: dict[str, Any] = {
        "system": system,
        "quant_level": quant_level,
        "width": width,
        "training_mode": training_mode,
        "evals": evals,
        "history": [],
        "parameter_count": e7a6.quantized_parameter_count(candidate),
        "nonzero_parameter_count": e7a6.quantized_nonzero_count(candidate),
        "hidden_matrix_shape": [width, width],
        "initial_hash": candidate_hash(candidate),
        "final_hash": candidate_hash(candidate),
        "candidate": candidate,
    }
    if float_eval_accuracy is not None:
        result["float_eval_accuracy"] = float_eval_accuracy
    if baseline_low_bit_eval_accuracy is not None:
        result["baseline_low_bit_eval_accuracy"] = baseline_low_bit_eval_accuracy
    if audit_mode is not None:
        result["audit_mode"] = audit_mode
    return result


def run_targeted_repair(
    quant_level: str,
    width: int,
    repair_scope: str,
    block_names: tuple[str, ...],
    initial_candidate: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    out: str | None,
    baseline_low_bit_eval_accuracy: float,
    float_eval_accuracy: float,
) -> dict[str, Any]:
    out_path = Path(out) if out else None
    allowed_keys = all_quantized_keys() if repair_scope == "full" else keys_for_blocks(block_names)
    scope_label = repair_scope if repair_scope == "full" else "_".join(block_names)
    if out_path:
        append_progress_locked(out_path, "repair_start", quant_level=quant_level, width=width, repair_scope=repair_scope, blocks=list(block_names))
    rng = random.Random(stable_seed(f"repair-{quant_level}-{width}-{repair_scope}-{scope_label}-{settings.seeds}"))
    population = [score_candidate(copy.deepcopy(initial_candidate), task, settings)]
    for _ in range(settings.population_size - 1):
        child = mutate_quantized_targeted(initial_candidate, settings, rng, allowed_keys)
        population.append(score_candidate(child, task, settings))
    accepted = 0
    rejected = 0
    rollback = 0
    attempts = 0
    history = []
    last_heartbeat = time.monotonic()
    for generation in range(1, settings.repair_generations + 1):
        population.sort(key=lambda row: row["fitness"], reverse=True)
        elites = population[: max(1, settings.elite_count)]
        next_population = copy.deepcopy(elites)
        while len(next_population) < settings.population_size:
            parent = copy.deepcopy(rng.choice(population))
            child_candidate = mutate_quantized_targeted(parent["candidate"], settings, rng, allowed_keys)
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
                    out_path / "partial_status" / f"repair_{quant_level}_width{width}_{repair_scope}_{scope_label}.json",
                    {
                        "quant_level": quant_level,
                        "width": width,
                        "repair_scope": repair_scope,
                        "blocks": list(block_names),
                        "generation": generation,
                        "attempts": attempts,
                        "accepted": accepted,
                        "rejected": rejected,
                        "rollback": rollback,
                        "best_fitness": best_mid["fitness"],
                        "best_candidate_hash": candidate_hash(best_mid["candidate"]),
                    },
                )
                append_progress_locked(
                    out_path,
                    "repair_heartbeat",
                    quant_level=quant_level,
                    width=width,
                    repair_scope=repair_scope,
                    blocks=list(block_names),
                    generation=generation,
                    best_fitness=best_mid["fitness"],
                )
                last_heartbeat = now
        population = next_population
        best = max(population, key=lambda row: row["fitness"])
        row = {
            "generation": generation,
            "best_fitness": best["fitness"],
            "validation_accuracy": best["evals"]["validation"]["metrics"]["accuracy"],
            "accepted_mutation_count": accepted,
            "rejected_mutation_count": rejected,
            "rollback_count": rollback,
            "candidate_hash": candidate_hash(best["candidate"]),
        }
        history.append(row)
        if out_path:
            write_json(
                out_path / f"e7a7_mutation_history_{quant_level}_width{width}_{repair_scope}_{scope_label}.json",
                {
                    "quant_level": quant_level,
                    "width": width,
                    "repair_scope": repair_scope,
                    "blocks": list(block_names),
                    "history": history,
                    "accepted_mutation_count": accepted,
                    "rejected_mutation_count": rejected,
                    "rollback_count": rollback,
                    "mutation_attempt_count": attempts,
                },
            )
            append_progress_locked(
                out_path,
                "repair_generation",
                quant_level=quant_level,
                width=width,
                repair_scope=repair_scope,
                blocks=list(block_names),
                generation=generation,
                validation_accuracy=row["validation_accuracy"],
            )
    best = max(population, key=lambda row: row["fitness"])
    final_candidate = best["candidate"]
    evals = e7a6.evaluate_quantized_candidate(final_candidate, task, settings.matrix_steps)
    system = "full_mutation_repair" if repair_scope == "full" else "targeted_block_mutation_repair"
    if repair_scope == "pair":
        system = "sensitive_pair_mutation_repair"
    result = {
        "system": system,
        "quant_level": quant_level,
        "width": width,
        "training_mode": f"{repair_scope}_quantized_mutation_repair",
        "repair_scope": repair_scope,
        "repair_block": block_names[0] if len(block_names) == 1 else None,
        "repair_pair": list(block_names) if len(block_names) > 1 else None,
        "evals": evals,
        "history": history,
        "mutation_history": {
            "quant_level": quant_level,
            "width": width,
            "repair_scope": repair_scope,
            "blocks": list(block_names),
            "mutation_attempt_count": attempts,
            "accepted_mutation_count": accepted,
            "rejected_mutation_count": rejected,
            "rollback_count": rollback,
            "history": history,
        },
        "parameter_count": e7a6.quantized_parameter_count(final_candidate),
        "nonzero_parameter_count": e7a6.quantized_nonzero_count(final_candidate),
        "hidden_matrix_shape": [width, width],
        "initial_candidate": initial_candidate,
        "final_candidate": final_candidate,
        "parameter_diff": e7a6.quantized_diff(initial_candidate, final_candidate),
        "initial_hash": candidate_hash(initial_candidate),
        "final_hash": candidate_hash(final_candidate),
        "baseline_low_bit_eval_accuracy": baseline_low_bit_eval_accuracy,
        "float_eval_accuracy": float_eval_accuracy,
    }
    if out_path:
        write_json(out_path / f"e7a7_parameter_diff_{quant_level}_width{width}_{repair_scope}_{scope_label}.json", result["parameter_diff"])
        write_json(
            out_path / f"e7a7_candidate_{quant_level}_width{width}_{repair_scope}_{scope_label}_summary.json",
            {
                "system": result["system"],
                "quant_level": quant_level,
                "width": width,
                "training_mode": result["training_mode"],
                "repair_scope": repair_scope,
                "blocks": list(block_names),
                "parameter_count": result["parameter_count"],
                "nonzero_parameter_count": result["nonzero_parameter_count"],
                "initial_hash": result["initial_hash"],
                "final_hash": result["final_hash"],
                "split_metrics": split_metrics(result),
                "mutation_history": {key: value for key, value in result["mutation_history"].items() if key != "history"},
                "parameter_diff": result["parameter_diff"],
            },
        )
        append_progress_locked(
            out_path,
            "repair_complete",
            quant_level=quant_level,
            width=width,
            repair_scope=repair_scope,
            blocks=list(block_names),
            heldout_accuracy=evals["heldout"]["metrics"]["accuracy"],
        )
    return result


def fake_quant_tensor(param: torch.Tensor, level: str) -> torch.Tensor:
    config = e7a6.QUANT_CONFIGS[level]
    eps = torch.tensor(1e-8, device=param.device, dtype=param.dtype)
    if config["kind"] == "symmetric_int":
        q_limit = float(config["q_limit"])
        max_abs = torch.max(torch.abs(param)).detach()
        scale = torch.clamp(max_abs / q_limit, min=float(eps.detach().cpu().item()))
        quantized = torch.clamp(torch.round(param / scale), -q_limit, q_limit)
        dequantized = quantized * scale
    elif config["kind"] == "ternary":
        scale = torch.clamp(torch.mean(torch.abs(param)).detach(), min=float(eps.detach().cpu().item()))
        threshold = 0.5 * scale
        quantized = torch.where(param > threshold, torch.ones_like(param), torch.where(param < -threshold, -torch.ones_like(param), torch.zeros_like(param)))
        dequantized = quantized * scale
    elif config["kind"] == "binary":
        scale = torch.clamp(torch.mean(torch.abs(param)).detach(), min=float(eps.detach().cpu().item()))
        quantized = torch.where(torch.abs(param) <= eps, torch.zeros_like(param), torch.where(param >= 0.0, torch.ones_like(param), -torch.ones_like(param)))
        dequantized = quantized * scale
    else:
        raise ValueError(f"unknown quant kind: {config['kind']}")
    return param + (dequantized - param).detach()


def qat_forward(model: Any, x: torch.Tensor, level: str, matrix_steps: int) -> torch.Tensor:
    win = fake_quant_tensor(model.win, level)
    state = fake_quant_tensor(model.state, level)
    carry_raw = fake_quant_tensor(model.carry_raw, level)
    bstate = fake_quant_tensor(model.bstate, level)
    wout = fake_quant_tensor(model.wout, level)
    bout = fake_quant_tensor(model.bout, level)
    drive = x @ win + bstate
    h = torch.tanh(drive)
    carry = torch.sigmoid(carry_raw).unsqueeze(0)
    for _ in range(matrix_steps):
        proposal = torch.tanh(h @ state + drive)
        h = carry * h + (1.0 - carry) * proposal
    return h @ wout + bout


def train_qat_core(
    quant_level: str,
    width: int,
    float_state_dict: dict[str, np.ndarray],
    task: dict[str, Any],
    settings: Settings,
    out: Path | None,
    baseline_low_bit_eval_accuracy: float,
    float_eval_accuracy: float,
) -> dict[str, Any]:
    device = e7a3.select_device(settings.device)
    e7a3.set_determinism(stable_seed(f"qat-{quant_level}-{width}-{settings.seeds}"), device)
    model = e7a6.FloatMatrixCore(width, settings.matrix_steps).to(device)
    torch_state = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in float_state_dict.items()}
    model.load_state_dict(torch_state)
    initial_vector = e7a6.torch_state_vector(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.qat_learning_rate, weight_decay=settings.weight_decay)
    x_train = torch.as_tensor(task["train"]["x"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(task["train"]["y"], dtype=torch.long, device=device)
    x_val = torch.as_tensor(task["validation"]["x"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(task["validation"]["y"], dtype=torch.long, device=device)
    rng = np.random.default_rng(stable_seed(f"batches-qat-{quant_level}-{width}-{settings.seeds}"))
    history = []
    last_heartbeat = time.monotonic()
    for epoch in range(1, settings.qat_epochs + 1):
        order = rng.permutation(x_train.shape[0])
        model.train()
        for start in range(0, len(order), settings.batch_size):
            idx = torch.as_tensor(order[start : start + settings.batch_size], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(qat_forward(model, x_train[idx], quant_level, settings.matrix_steps), y_train[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = qat_forward(model, x_val, quant_level, settings.matrix_steps)
            val_loss = F.cross_entropy(logits, y_val).detach().cpu().item()
            val_acc = (torch.argmax(logits, dim=1) == y_val).float().mean().detach().cpu().item()
        row = {
            "epoch": epoch,
            "validation_accuracy": round_float(val_acc),
            "validation_loss": round_float(val_loss),
            "state_hash": e7a6.vector_hash(e7a6.torch_state_vector(model)),
        }
        history.append(row)
        now = time.monotonic()
        if out and (now - last_heartbeat >= settings.heartbeat_seconds or epoch == settings.qat_epochs):
            write_json(out / f"e7a7_qat_history_{quant_level}_width{width}.json", {"quant_level": quant_level, "width": width, "history": history})
            append_progress_locked(out, "qat_epoch", quant_level=quant_level, width=width, epoch=epoch, validation_accuracy=row["validation_accuracy"])
            last_heartbeat = now
    state_dict = {key: value.detach().cpu().numpy().astype(np.float64) for key, value in model.state_dict().items()}
    candidate = e7a6.quantize_state_dict(quant_level, state_dict, width)
    candidate = update_candidate_metadata(candidate, "e7a7_qat_quantized_plain_matrix_core_candidate_v1", "quantization_aware_training", quant_level)
    evals = e7a6.evaluate_quantized_candidate(candidate, task, settings.matrix_steps)
    result = {
        "system": "quantization_aware_training",
        "quant_level": quant_level,
        "width": width,
        "training_mode": "fake_quant_ste_backprop_then_quantize",
        "evals": evals,
        "history": history,
        "parameter_count": e7a6.quantized_parameter_count(candidate),
        "nonzero_parameter_count": e7a6.quantized_nonzero_count(candidate),
        "hidden_matrix_shape": [width, width],
        "initial_hash": e7a6.vector_hash(initial_vector),
        "final_hash": candidate_hash(candidate),
        "float_state_hash": e7a6.vector_hash(e7a6.torch_state_vector(model)),
        "candidate": candidate,
        "baseline_low_bit_eval_accuracy": baseline_low_bit_eval_accuracy,
        "float_eval_accuracy": float_eval_accuracy,
        "device": device,
    }
    if out:
        write_json(
            out / f"e7a7_candidate_{quant_level}_width{width}_qat_summary.json",
            {
                "system": result["system"],
                "quant_level": quant_level,
                "width": width,
                "training_mode": result["training_mode"],
                "parameter_count": result["parameter_count"],
                "nonzero_parameter_count": result["nonzero_parameter_count"],
                "initial_hash": result["initial_hash"],
                "final_hash": result["final_hash"],
                "split_metrics": split_metrics(result),
                "history": history,
                "device": device,
            },
        )
        append_progress_locked(out, "qat_complete", quant_level=quant_level, width=width, heldout_accuracy=evals["heldout"]["metrics"]["accuracy"])
    return result


def run_core(settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, settings={**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths)})
    base_settings = e7a6_settings(settings)
    task = e7a3.generate_task(e7a6.e7a3_settings(base_settings))
    if out:
        append_progress_locked(out, "task_generated", rows={split: len(task[split]["rows"]) for split in task_splits()})
    results: dict[str, Any] = {
        "float32": {},
        "low_bit": {},
        "block_only": {},
        "block_restore": {},
        "repair": {},
        "qat": {},
    }
    executor: ProcessPoolExecutor | None = None
    futures = {}
    if settings.execution_mode == "parallel":
        executor = ProcessPoolExecutor(max_workers=max(1, settings.parallel_workers))
        if out:
            append_progress_locked(out, "repair_lanes_ready", workers=settings.parallel_workers)
    for width in settings.widths:
        trained = e7a6.train_float_core(width, task, base_settings, out)
        results["float32"][width] = trained
        float_row = e7a6.result_row(trained)
        float_eval_accuracy = float_row["eval_accuracy"]
        int8_candidate = e7a6.quantize_state_dict("int8", trained["state_dict"], width)
        for level in AUDIT_LEVELS:
            low_candidate = e7a6.quantize_state_dict(level, trained["state_dict"], width)
            low_candidate = mixed_candidate(level, low_candidate, int8_candidate, "full_low_bit", None)
            low_result = quantized_eval_result(
                "low_bit_no_repair",
                level,
                width,
                low_candidate,
                task,
                settings,
                "post_backprop_low_bit_no_repair",
                float_eval_accuracy=float_eval_accuracy,
                audit_mode="full_low_bit",
            )
            results["low_bit"][(level, width)] = low_result
            baseline_low_bit_eval_accuracy = result_row(low_result)["eval_accuracy"]
            for block in BLOCKS:
                block_only_candidate = mixed_candidate(level, low_candidate, int8_candidate, "block_only_low_bit", block)
                block_restore_candidate = mixed_candidate(level, low_candidate, int8_candidate, "block_restored_to_int8", block)
                results["block_only"][(level, width, block)] = quantized_eval_result(
                    "block_only_low_bit",
                    level,
                    width,
                    block_only_candidate,
                    task,
                    settings,
                    "single_block_low_bit_with_int8_rest",
                    float_eval_accuracy=float_eval_accuracy,
                    baseline_low_bit_eval_accuracy=baseline_low_bit_eval_accuracy,
                    audit_mode=f"block_only_low_bit:{block}",
                )
                results["block_restore"][(level, width, block)] = quantized_eval_result(
                    "block_restored_to_int8",
                    level,
                    width,
                    block_restore_candidate,
                    task,
                    settings,
                    "full_low_bit_with_single_block_int8_restored",
                    float_eval_accuracy=float_eval_accuracy,
                    baseline_low_bit_eval_accuracy=baseline_low_bit_eval_accuracy,
                    audit_mode=f"block_restored_to_int8:{block}",
                )
            restore_rows = [
                (block, result_row(results["block_restore"][(level, width, block)])["gain_vs_low_bit"])
                for block in BLOCKS
            ]
            top_pair = tuple(block for block, _ in sorted(restore_rows, key=lambda item: item[1], reverse=True)[:2])
            repair_jobs: list[tuple[str, tuple[str, ...]]] = [("full", tuple(BLOCKS.keys())), ("pair", top_pair)]
            repair_jobs.extend(("block", (block,)) for block in BLOCKS)
            for repair_scope, blocks in repair_jobs:
                if executor is not None:
                    future = executor.submit(
                        run_targeted_repair,
                        level,
                        width,
                        repair_scope,
                        blocks,
                        low_candidate,
                        task,
                        settings,
                        out.as_posix() if out else None,
                        baseline_low_bit_eval_accuracy,
                        float_eval_accuracy,
                    )
                    futures[future] = (level, width, repair_scope, blocks)
                else:
                    results["repair"][(level, width, repair_scope, blocks)] = run_targeted_repair(
                        level,
                        width,
                        repair_scope,
                        blocks,
                        low_candidate,
                        task,
                        settings,
                        out.as_posix() if out else None,
                        baseline_low_bit_eval_accuracy,
                        float_eval_accuracy,
                    )
            if level in LOW_BIT_LEVELS:
                results["qat"][(level, width)] = train_qat_core(
                    level,
                    width,
                    trained["state_dict"],
                    task,
                    settings,
                    out,
                    baseline_low_bit_eval_accuracy,
                    float_eval_accuracy,
                )
        if out:
            partial = {
                "completed_width": width,
                "float_eval_accuracy": float_eval_accuracy,
                "pending_repair_jobs": len(futures),
            }
            locked_write_json(out / "partial_status" / "e7a7_width_progress.json", partial)
            append_progress_locked(out, "width_complete", width=width, pending_repair_jobs=len(futures))
    if executor is not None:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                level, width, repair_scope, blocks = futures[future]
                results["repair"][(level, width, repair_scope, blocks)] = future.result()
                if out:
                    append_progress_locked(
                        out,
                        "repair_lane_joined",
                        quant_level=level,
                        width=width,
                        repair_scope=repair_scope,
                        blocks=list(blocks),
                        completed=len(results["repair"]),
                        pending=len(pending),
                    )
        executor.shutdown()
    return task, results


def aggregate_results(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    systems: dict[str, Any] = {
        "float32": {},
        "low_bit": {},
        "block_only": {},
        "block_restore": {},
        "repair": {},
        "qat": {},
    }
    for width in settings.widths:
        systems["float32"][str(width)] = result_row(results["float32"][width])
    for level in AUDIT_LEVELS:
        systems["low_bit"][level] = {}
        systems["block_only"][level] = {}
        systems["block_restore"][level] = {}
        systems["repair"][level] = {}
        for width in settings.widths:
            systems["low_bit"][level][str(width)] = result_row(results["low_bit"][(level, width)])
            systems["block_only"][level][str(width)] = {
                block: result_row(results["block_only"][(level, width, block)])
                for block in BLOCKS
            }
            systems["block_restore"][level][str(width)] = {
                block: result_row(results["block_restore"][(level, width, block)])
                for block in BLOCKS
            }
            systems["repair"][level][str(width)] = {}
            for key, value in results["repair"].items():
                r_level, r_width, repair_scope, blocks = key
                if r_level == level and r_width == width:
                    label = repair_scope if repair_scope == "full" else f"{repair_scope}:{'+'.join(blocks)}"
                    systems["repair"][level][str(width)][label] = result_row(value)
    for level in LOW_BIT_LEVELS:
        systems["qat"][level] = {}
        for width in settings.widths:
            systems["qat"][level][str(width)] = result_row(results["qat"][(level, width)])
    best: dict[str, Any] = {
        "float32": max((systems["float32"][str(width)] for width in settings.widths), key=lambda row: row["eval_accuracy"]),
        "low_bit": {},
        "repair": {},
        "qat": {},
    }
    for level in AUDIT_LEVELS:
        best["low_bit"][level] = max((systems["low_bit"][level][str(width)] for width in settings.widths), key=lambda row: row["eval_accuracy"])
        repair_rows = [
            row
            for width in settings.widths
            for row in systems["repair"][level][str(width)].values()
        ]
        best["repair"][level] = max(repair_rows, key=lambda row: row["eval_accuracy"])
    for level in LOW_BIT_LEVELS:
        best["qat"][level] = max((systems["qat"][level][str(width)] for width in settings.widths), key=lambda row: row["eval_accuracy"])
    return {
        "schema_version": "e7a7_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "best_by_eval_accuracy": best,
        "solve_thresholds": {
            "stable_drop_vs_float": 0.02,
            "repair_value_gain": 0.02,
            "strong_block_gain": 0.03,
        },
    }


def block_damage_report(aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for level in AUDIT_LEVELS:
        rows[level] = {}
        for width in settings.widths:
            float_eval = aggregate["systems"]["float32"][str(width)]["eval_accuracy"]
            full_low = aggregate["systems"]["low_bit"][level][str(width)]["eval_accuracy"]
            rows[level][str(width)] = {}
            for block, row in aggregate["systems"]["block_only"][level][str(width)].items():
                rows[level][str(width)][block] = {
                    "block_only_eval_accuracy": row["eval_accuracy"],
                    "drop_vs_float": round_float(float_eval - row["eval_accuracy"]),
                    "excess_drop_vs_full_low_bit": round_float(full_low - row["eval_accuracy"]),
                    "parameter_count": row["parameter_count"],
                }
    return {
        "schema_version": "e7a7_block_damage_report_v1",
        "meaning": "Quantize one block to target low-bit while keeping the rest at int8.",
        "blocks": BLOCKS,
        "rows": rows,
    }


def block_restore_report(aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    top_by_restore_gain = {}
    for level in AUDIT_LEVELS:
        rows[level] = {}
        top_by_restore_gain[level] = {}
        for width in settings.widths:
            low_eval = aggregate["systems"]["low_bit"][level][str(width)]["eval_accuracy"]
            rows[level][str(width)] = {}
            for block, row in aggregate["systems"]["block_restore"][level][str(width)].items():
                gain = round_float(row["eval_accuracy"] - low_eval)
                rows[level][str(width)][block] = {
                    "restore_eval_accuracy": row["eval_accuracy"],
                    "gain_vs_full_low_bit": gain,
                    "drop_vs_float": row["drop_vs_float"],
                }
            top_by_restore_gain[level][str(width)] = sorted(
                rows[level][str(width)].items(),
                key=lambda item: item[1]["gain_vs_full_low_bit"],
                reverse=True,
            )[:3]
    return {
        "schema_version": "e7a7_block_restore_report_v1",
        "meaning": "Start from full low-bit and restore one block to int8.",
        "rows": rows,
        "top_by_restore_gain": top_by_restore_gain,
    }


def repair_operator_report(results: dict[str, Any], aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    all_histories_have_attempts = True
    all_histories_have_reject_rollback = True
    any_accepted = False
    for level in AUDIT_LEVELS:
        rows[level] = {}
        for width in settings.widths:
            rows[level][str(width)] = {}
            for key, result in results["repair"].items():
                r_level, r_width, repair_scope, blocks = key
                if r_level != level or r_width != width:
                    continue
                label = repair_scope if repair_scope == "full" else f"{repair_scope}:{'+'.join(blocks)}"
                hist = result["mutation_history"]
                row = result_row(result)
                rows[level][str(width)][label] = {
                    "aggregate_metrics": row,
                    "mutation_history": {key: value for key, value in hist.items() if key != "history"},
                    "parameter_diff": result["parameter_diff"],
                }
                all_histories_have_attempts = all_histories_have_attempts and hist["mutation_attempt_count"] > 0
                all_histories_have_reject_rollback = all_histories_have_reject_rollback and hist["rejected_mutation_count"] == hist["rollback_count"] and hist["rejected_mutation_count"] > 0
                any_accepted = any_accepted or hist["accepted_mutation_count"] > 0
    return {
        "schema_version": "e7a7_repair_operator_report_v1",
        "rows": rows,
        "all_repair_runs_have_mutation_attempts": all_histories_have_attempts,
        "all_repair_runs_have_reject_rollback": all_histories_have_reject_rollback,
        "at_least_one_repair_mutation_accepted": any_accepted,
        "repair_generations": settings.repair_generations,
    }


def qat_report(results: dict[str, Any], aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for level in LOW_BIT_LEVELS:
        rows[level] = {}
        for width in settings.widths:
            row = aggregate["systems"]["qat"][level][str(width)]
            repair_best = max(aggregate["systems"]["repair"][level][str(width)].values(), key=lambda value: value["eval_accuracy"])
            rows[level][str(width)] = {
                "qat_metrics": row,
                "best_post_quantization_repair": repair_best,
                "qat_delta_vs_low_bit": row["gain_vs_low_bit"],
                "qat_delta_vs_best_repair": round_float(row["eval_accuracy"] - repair_best["eval_accuracy"]),
                "qat_stable_vs_float": bool(row["drop_vs_float"] <= 0.02 and row["solve_passed"]),
            }
    return {
        "schema_version": "e7a7_qat_report_v1",
        "training_mode": "fake_quant_ste_backprop_then_quantize",
        "rows": rows,
    }


def bottleneck_report(aggregate: dict[str, Any], restore: dict[str, Any], repair: dict[str, Any], qat: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for level in AUDIT_LEVELS:
        rows[level] = {}
        for width in settings.widths:
            restore_rows = restore["rows"][level][str(width)]
            top_restore_block, top_restore = max(restore_rows.items(), key=lambda item: item[1]["gain_vs_full_low_bit"])
            repair_rows = aggregate["systems"]["repair"][level][str(width)]
            full = repair_rows["full"]
            targeted_rows = {key: value for key, value in repair_rows.items() if key.startswith("block:")}
            pair_rows = {key: value for key, value in repair_rows.items() if key.startswith("pair:")}
            best_targeted_label, best_targeted = max(targeted_rows.items(), key=lambda item: item[1]["eval_accuracy"])
            best_pair_label, best_pair = max(pair_rows.items(), key=lambda item: item[1]["eval_accuracy"])
            row = {
                "top_restore_block": top_restore_block,
                "top_restore_gain": top_restore["gain_vs_full_low_bit"],
                "full_repair_eval_accuracy": full["eval_accuracy"],
                "best_targeted_label": best_targeted_label,
                "best_targeted_eval_accuracy": best_targeted["eval_accuracy"],
                "best_targeted_delta_vs_full": round_float(best_targeted["eval_accuracy"] - full["eval_accuracy"]),
                "best_pair_label": best_pair_label,
                "best_pair_eval_accuracy": best_pair["eval_accuracy"],
                "best_pair_delta_vs_full": round_float(best_pair["eval_accuracy"] - full["eval_accuracy"]),
                "low_bit_eval_accuracy": aggregate["systems"]["low_bit"][level][str(width)]["eval_accuracy"],
                "float_eval_accuracy": aggregate["systems"]["float32"][str(width)]["eval_accuracy"],
            }
            if level in LOW_BIT_LEVELS:
                qat_row = qat["rows"][level][str(width)]
                row["qat_eval_accuracy"] = qat_row["qat_metrics"]["eval_accuracy"]
                row["qat_delta_vs_best_repair"] = qat_row["qat_delta_vs_best_repair"]
                row["qat_stable_vs_float"] = qat_row["qat_stable_vs_float"]
            rows[level][str(width)] = row
    return {
        "schema_version": "e7a7_low_bit_bottleneck_report_v1",
        "rows": rows,
    }


def no_synthetic_metric_audit(task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a7_no_synthetic_metric_audit_v1",
        "generated_from_row_level_eval": True,
        "row_counts": {split: len(task[split]["rows"]) for split in task_splits()},
        "row_level_samples_present": all(
            bool(results["float32"][width]["evals"]["heldout"]["row_level_samples"])
            and bool(results["low_bit"][(level, width)]["evals"]["heldout"]["row_level_samples"])
            for width in settings.widths
            for level in AUDIT_LEVELS
        ),
        "block_audit_uses_real_quantized_forward": True,
        "mutation_repair_used_optimizer_or_backprop": False,
        "qat_uses_backprop_by_design": True,
        "hardcoded_improvement_flags_present": False,
        "broad_claims_intentionally_deferred": True,
    }


def choose_decision(
    aggregate: dict[str, Any],
    restore: dict[str, Any],
    repair: dict[str, Any],
    qat: dict[str, Any],
    bottleneck: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    if (
        not audit["generated_from_row_level_eval"]
        or audit["hardcoded_improvement_flags_present"]
        or not repair["all_repair_runs_have_mutation_attempts"]
        or not repair["all_repair_runs_have_reject_rollback"]
        or not repair["at_least_one_repair_mutation_accepted"]
    ):
        decision = "e7a7_invalid_artifact_detected"
    else:
        rows = [
            row
            for level_rows in bottleneck["rows"].values()
            for row in level_rows.values()
        ]
        qat_preferred = any(row.get("qat_delta_vs_best_repair", 0.0) >= 0.02 for row in rows if "qat_delta_vs_best_repair" in row)
        qat_stable = any(row.get("qat_stable_vs_float") is True for row in rows)
        best_repair_stable = any(
            abs(row["float_eval_accuracy"] - max(row["full_repair_eval_accuracy"], row["best_targeted_eval_accuracy"], row["best_pair_eval_accuracy"])) <= 0.02
            for row in rows
        )
        targeted_sufficient = any(
            row["best_targeted_eval_accuracy"] >= row["full_repair_eval_accuracy"] - 0.005
            and row["best_targeted_eval_accuracy"] - row["low_bit_eval_accuracy"] >= 0.02
            for row in rows
        )
        state_or_output_bottleneck = any(
            row["top_restore_block"] in {"recurrent_state", "output_head"}
            and row["top_restore_gain"] >= 0.03
            for row in rows
        )
        if qat_preferred:
            decision = "e7a7_qat_preferred_over_post_repair"
        elif targeted_sufficient:
            decision = "e7a7_sensitive_block_repair_sufficient"
        elif state_or_output_bottleneck:
            decision = "e7a7_output_or_state_bottleneck_identified"
        elif qat_stable and not best_repair_stable:
            decision = "e7a7_repair_operator_bottleneck_detected"
        elif not qat_stable and not best_repair_stable:
            decision = "e7a7_low_bit_information_limit_detected"
        else:
            decision = "e7a7_low_bit_breakpoint_audit_complete"
    return {
        "schema_version": "e7a7_decision_v1",
        "decision": decision,
        "valid_decisions": list(VALID_DECISIONS),
        "deterministic_replay_passed": False,
        "broad_claims_intentionally_deferred": True,
    }


def task_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a7_task_generation_report_v1",
        "milestone": MILESTONE,
        "inherits_task_from": "E7A6_QUANTIZATION_STRESS_AND_REPAIR_LIMIT",
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
            "float32": results["float32"][width]["evals"][split]["row_level_samples"],
            "low_bit": {
                level: results["low_bit"][(level, width)]["evals"][split]["row_level_samples"]
                for level in AUDIT_LEVELS
            },
            "qat": {
                level: results["qat"][(level, width)]["evals"][split]["row_level_samples"]
                for level in LOW_BIT_LEVELS
            },
        }
    return {
        "schema_version": "e7a7_row_level_eval_sample_v1",
        "split": split,
        "samples": samples,
    }


def mutation_history_report(results: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for key, result in results["repair"].items():
        level, width, repair_scope, blocks = key
        rows[f"{level}/width{width}/{repair_scope}/{'-'.join(blocks)}"] = result["mutation_history"]
    return {
        "schema_version": "e7a7_mutation_history_report_v1",
        "rows": rows,
    }


def build_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any], bottleneck: dict[str, Any]) -> str:
    lines = [
        "# E7A7 Low-Bit Repair Operator Audit Result",
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
        "## Low-Bit Bottleneck Rows",
        "",
        "| level | width | low-bit eval | top restored block | restore gain | full repair | best targeted | best pair | QAT |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for level in AUDIT_LEVELS:
        for width, row in bottleneck["rows"][level].items():
            qat_value = row.get("qat_eval_accuracy")
            qat_text = "" if qat_value is None else f"{qat_value:.6f}"
            lines.append(
                f"| `{level}` | {width} | {row['low_bit_eval_accuracy']:.6f} | `{row['top_restore_block']}` | "
                f"{row['top_restore_gain']:.6f} | {row['full_repair_eval_accuracy']:.6f} | "
                f"{row['best_targeted_eval_accuracy']:.6f} | {row['best_pair_eval_accuracy']:.6f} | {qat_text} |"
            )
    best_float = aggregate["best_by_eval_accuracy"]["float32"]
    lines.extend(
        [
            "",
            "## Baseline",
            "",
            f"Best float32 matrix-core: width `{best_float['width']}`, eval `{best_float['eval_accuracy']:.6f}`.",
            "",
            "This probe only audits the controlled symbolic/numeric low-bit substrate.",
            "",
        ]
    )
    return "\n".join(lines)


def build_payloads(out: Path, task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    aggregate = aggregate_results(results, settings)
    damage = block_damage_report(aggregate, settings)
    restore = block_restore_report(aggregate, settings)
    repair = repair_operator_report(results, aggregate, settings)
    qat = qat_report(results, aggregate, settings)
    bottleneck = bottleneck_report(aggregate, restore, repair, qat, settings)
    audit = no_synthetic_metric_audit(task, results, settings)
    decision = choose_decision(aggregate, restore, repair, qat, bottleneck, audit)
    payloads: dict[str, Any] = {
        "e7a7_backend_manifest.json": {
            "schema_version": "e7a7_backend_manifest_v1",
            "milestone": MILESTONE,
            "systems": list(SYSTEMS),
            "audit_levels": list(AUDIT_LEVELS),
            "low_bit_levels": list(LOW_BIT_LEVELS),
            "blocks": BLOCKS,
            "quant_configs": {level: e7a6.QUANT_CONFIGS[level] for level in AUDIT_LEVELS},
            "widths": list(settings.widths),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": e7a3.select_device(settings.device),
            "settings": {**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths)},
            "cpu_repair_and_gradient_overlap_supported": True,
            "parallel_replay_supported": True,
            "broad_claims_intentionally_deferred": True,
        },
        "e7a7_task_generation_report.json": task_report(task, settings),
        "e7a7_block_damage_report.json": damage,
        "e7a7_block_restore_report.json": restore,
        "e7a7_repair_operator_report.json": repair,
        "e7a7_qat_report.json": qat,
        "e7a7_low_bit_bottleneck_report.json": bottleneck,
        "e7a7_mutation_history.json": mutation_history_report(results),
        "e7a7_no_synthetic_metric_audit.json": audit,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e7a7_summary_v1",
            "milestone": MILESTONE,
            "decision": decision["decision"],
            "best_float_width": aggregate["best_by_eval_accuracy"]["float32"]["width"],
            "best_float_eval_accuracy": aggregate["best_by_eval_accuracy"]["float32"]["eval_accuracy"],
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
            "broad_claims_intentionally_deferred": True,
        },
    }
    for split in eval_splits():
        payloads[f"e7a7_row_level_eval_sample_{split}.json"] = row_samples(results, split, settings)
    payloads["report.md"] = build_report(out, decision, aggregate, bottleneck)
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
        "schema_version": "e7a7_deterministic_replay_report_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
        "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix(),
    }
    append_progress_locked(out, "deterministic_replay_complete", internal_replay_passed=report["internal_replay_passed"])
    return report


def write_final_artifacts(out: Path, payloads: dict[str, Any], deterministic: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["e7a7_deterministic_replay_report.json"] = deterministic
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
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--quant-mutation-steps", type=int, default=24)
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
        elite_count=args.elite_count,
        quant_mutation_steps=args.quant_mutation_steps,
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
