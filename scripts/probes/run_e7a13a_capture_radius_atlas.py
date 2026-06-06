#!/usr/bin/env python3
"""E7A13A capture radius atlas around a good binary seed.

This probe measures how far mutation/rollback repair can recover from a known
good binary matrix-core seed after controlled bit/scale corruptions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
from dataclasses import dataclass
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


REPO_ROOT = Path(__file__).resolve().parents[2]
E7A10_PATH = Path(__file__).with_name("run_e7a10_binary_scale_overhead_and_bit_budget_audit.py")
MILESTONE = "E7A13A_CAPTURE_RADIUS_ATLAS"
DEFAULT_OUT = Path("target/pilot_wave/e7a13a_capture_radius_atlas")
PARAM_KEYS = ("win", "state", "carry_raw", "bstate", "wout", "bout")
BLOCK_KEYS = {
    "input_projection": ("win",),
    "recurrent_state": ("state",),
    "carry_gate": ("carry_raw",),
    "state_bias": ("bstate",),
    "output_head": ("wout", "bout"),
}
CORRUPTION_MODES = (
    "random_bit_flip_shell",
    "least_sensitive_bit_flip_shell",
    "most_sensitive_bit_flip_shell",
    "block_corruption_shell",
    "scale_perturbation_shell",
    "bits_plus_scale_corruption_shell",
)
VALID_DECISIONS = (
    "e7a13a_capture_radius_measured",
    "e7a13a_invalid_artifact_detected",
)
VALID_CLASSIFICATIONS = (
    "sharp_capture_boundary",
    "smooth_falloff",
    "ragged_island_basin",
    "no_measurable_repair_basin",
)
HASH_ARTIFACTS = (
    "center_seed_report.json",
    "shell_metrics.json",
    "repair_metrics.json",
    "capture_radius_report.json",
    "falloff_model_report.json",
    "summary.json",
    "final_summary.md",
)


def load_e7a10_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7a10_binary_scale_overhead_and_bit_budget_audit", E7A10_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7A10 from {E7A10_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7a10 = load_e7a10_module()
e7a9 = e7a10.e7a9
e7a8 = e7a10.e7a8
e7a7 = e7a10.e7a7
e7a6 = e7a10.e7a6
e7a3 = e7a10.e7a3


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    width: int
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    gradient_epochs: int
    best_effort_qat_epochs: int
    batch_size: int
    learning_rate: float
    best_effort_learning_rate: float
    weight_decay: float
    matrix_steps: int
    distillation_weight: float
    distillation_temperature: float
    distances: tuple[float, ...]
    shell_replicates: int
    budget_multipliers: tuple[int, ...]
    skipped_budget_multipliers: tuple[int, ...]
    base_repair_generations: int
    mutation_population: int
    mutation_steps: int
    elite_count: int
    scale_sigma: float
    recovery_epsilon: float
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
    return int(hashlib.sha256(f"e7a13a::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def safe_file_id(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


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


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one float is required")
    return values


def e7a6_settings(settings: Settings) -> Any:
    return e7a6.Settings(
        seeds=settings.seeds,
        widths=(settings.width,),
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
        population_size=settings.mutation_population,
        repair_generations=settings.base_repair_generations,
        elite_count=settings.elite_count,
        quant_mutation_steps=settings.mutation_steps,
        matrix_steps=settings.matrix_steps,
        device=settings.device,
        execution_mode="serial",
        parallel_workers=1,
        heartbeat_seconds=settings.heartbeat_seconds,
    )


def e7a10_settings(settings: Settings) -> Any:
    return e7a10.Settings(
        seeds=settings.seeds,
        widths=(settings.width,),
        reference_width=settings.width,
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=settings.adversarial_rows_per_seed,
        gradient_epochs=settings.gradient_epochs,
        qat_epochs=max(1, settings.best_effort_qat_epochs),
        best_effort_qat_epochs=settings.best_effort_qat_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        qat_learning_rate=settings.best_effort_learning_rate,
        best_effort_learning_rate=settings.best_effort_learning_rate,
        weight_decay=settings.weight_decay,
        population_size=settings.mutation_population,
        repair_generations=settings.base_repair_generations,
        plateau_patience=1,
        plateau_min_delta=0.0,
        paramwise_generation_per_step=1,
        paramwise_step_limit=0,
        elite_count=settings.elite_count,
        quant_mutation_steps=settings.mutation_steps,
        matrix_steps=settings.matrix_steps,
        distillation_weight=settings.distillation_weight,
        distillation_temperature=settings.distillation_temperature,
        device=settings.device,
        execution_mode="serial",
        parallel_workers=1,
        heartbeat_seconds=settings.heartbeat_seconds,
    )


def task_splits() -> tuple[str, ...]:
    return e7a3.SPLITS


def eval_splits() -> tuple[str, ...]:
    return e7a3.EVAL_SPLITS


def candidate_hash(candidate: dict[str, Any]) -> str:
    return payload_sha256(candidate)


def update_candidate_hash(candidate: dict[str, Any]) -> None:
    candidate["zero_counts"] = {key: int(np.sum(np.asarray(value, dtype=np.int16) == 0)) for key, value in candidate["q"].items()}
    candidate["candidate_hash"] = candidate_hash(candidate)


def all_paths(candidate: dict[str, Any]) -> list[tuple[tuple[Any, ...], int]]:
    return e7a9.all_paths(candidate)


def path_key(path: tuple[Any, ...]) -> str:
    return e7a8.path_key(path)


def key_from_path(path: tuple[Any, ...]) -> str:
    return str(path[1])


def flip_path(candidate: dict[str, Any], path: tuple[Any, ...]) -> None:
    arr = np.asarray(candidate["q"][path[1]], dtype=np.int16)
    index = tuple(int(part) for part in path[2:])
    arr[index] = 1 if int(arr[index]) == 0 else -int(arr[index])
    candidate["q"][path[1]] = arr.tolist()


def scale_values(candidate: dict[str, Any]) -> np.ndarray:
    values = []
    for key in PARAM_KEYS:
        raw = np.asarray(candidate["scales"][key], dtype=np.float64)
        values.extend(raw.reshape(-1).tolist())
    return np.asarray(values, dtype=np.float64)


def eval_candidate(candidate: dict[str, Any], task: dict[str, Any], settings: Settings, sample_limit: int = 0) -> dict[str, Any]:
    return {
        split: e7a3.evaluate_logits(e7a8.quantized_forward(candidate, data["x"], settings.matrix_steps), data, sample_limit)
        for split, data in task.items()
    }


def eval_accuracy(evals: dict[str, Any]) -> float:
    return round_float(float(np.mean([evals[split]["metrics"]["accuracy"] for split in eval_splits()])))


def solve_pass_from_evals(evals: dict[str, Any]) -> bool:
    return (
        evals["heldout"]["metrics"]["accuracy"] >= 0.90
        and evals["ood"]["metrics"]["accuracy"] >= 0.85
        and evals["counterfactual"]["metrics"]["accuracy"] >= 0.85
        and evals["adversarial"]["metrics"]["accuracy"] >= 0.80
    )


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


def path_metadata(center: dict[str, Any], state_dict: dict[str, np.ndarray], task: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    sensitivity = e7a8.block_sensitivity_report("binary", state_dict, settings.width, task, e7a10_settings(settings))
    block_sens = {block: row["sensitivity_score"] for block, row in sensitivity["rows"].items()}
    metadata = e7a8.build_path_metadata("binary", state_dict, center, block_sens)
    return metadata


def corrupt_seed(
    center: dict[str, Any],
    metadata: list[dict[str, Any]],
    distance: float,
    mode: str,
    replicate: int,
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(stable_seed(f"corrupt-{mode}-{distance}-{replicate}-{settings.seeds}"))
    candidate = copy.deepcopy(center)
    paths = all_paths(candidate)
    flip_count = int(round(distance * len(paths)))
    sorted_low = sorted(metadata, key=lambda row: (row["normalized_sensitivity"], row["quant_error"], row["path_key"]))
    sorted_high = sorted(metadata, key=lambda row: (row["normalized_sensitivity"], row["quant_error"], row["path_key"]), reverse=True)
    selected_keys: set[str] = set()
    if mode == "random_bit_flip_shell":
        selected = rng.sample(paths, k=min(flip_count, len(paths))) if flip_count else []
        for path, _ in selected:
            flip_path(candidate, path)
            selected_keys.add(path_key(path))
    elif mode == "least_sensitive_bit_flip_shell":
        for row in sorted_low[:flip_count]:
            selected_keys.add(row["path_key"])
        for path, _ in paths:
            if path_key(path) in selected_keys:
                flip_path(candidate, path)
    elif mode == "most_sensitive_bit_flip_shell":
        for row in sorted_high[:flip_count]:
            selected_keys.add(row["path_key"])
        for path, _ in paths:
            if path_key(path) in selected_keys:
                flip_path(candidate, path)
    elif mode == "block_corruption_shell":
        blocks = tuple(BLOCK_KEYS)
        block = blocks[replicate % len(blocks)]
        block_paths = [(path, value) for path, value in paths if key_from_path(path) in BLOCK_KEYS[block]]
        for path, _ in rng.sample(block_paths, k=min(flip_count, len(block_paths))) if flip_count else []:
            flip_path(candidate, path)
            selected_keys.add(path_key(path))
    elif mode == "scale_perturbation_shell":
        sigma = max(distance, 0.0) * 1.5
        for key in PARAM_KEYS:
            raw = np.asarray(candidate["scales"][key], dtype=np.float64)
            factor = math.exp(rng.gauss(0.0, sigma))
            updated = np.maximum(raw * factor, 1e-8)
            candidate["scales"][key] = round_float(float(updated)) if updated.ndim == 0 else [round_float(v) for v in updated.reshape(-1).tolist()]
    elif mode == "bits_plus_scale_corruption_shell":
        for path, _ in rng.sample(paths, k=min(flip_count, len(paths))) if flip_count else []:
            flip_path(candidate, path)
            selected_keys.add(path_key(path))
        sigma = max(distance, 0.0)
        for key in PARAM_KEYS:
            raw = np.asarray(candidate["scales"][key], dtype=np.float64)
            factor = math.exp(rng.gauss(0.0, sigma))
            updated = np.maximum(raw * factor, 1e-8)
            candidate["scales"][key] = round_float(float(updated)) if updated.ndim == 0 else [round_float(v) for v in updated.reshape(-1).tolist()]
    else:
        raise ValueError(mode)
    update_candidate_hash(candidate)
    center_scales = scale_values(center)
    corrupted_scales = scale_values(candidate)
    return candidate, {
        "requested_distance": distance,
        "corruption_mode": mode,
        "replicate": replicate,
        "requested_flip_count": flip_count,
        "actual_flipped_key_count": len(selected_keys),
        "scale_l1_distance": round_float(float(np.mean(np.abs(corrupted_scales - center_scales)))) if center_scales.size else 0.0,
        "scale_l2_distance": round_float(float(np.sqrt(np.mean((corrupted_scales - center_scales) ** 2)))) if center_scales.size else 0.0,
        "corrupted_hash": candidate_hash(candidate),
    }


def distance_metrics(
    center: dict[str, Any],
    corrupted: dict[str, Any],
    metadata: list[dict[str, Any]],
    task: dict[str, Any],
    center_evals: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    center_values = {path_key(path): value for path, value in all_paths(center)}
    corrupted_values = {path_key(path): value for path, value in all_paths(corrupted)}
    total = len(center_values)
    changed = [key for key in center_values if center_values[key] != corrupted_values[key]]
    sensitivity_by_key = {row["path_key"]: float(row.get("normalized_sensitivity", 0.0)) for row in metadata}
    total_sensitivity = sum(sensitivity_by_key.get(key, 0.0) for key in center_values)
    changed_sensitivity = sum(sensitivity_by_key.get(key, 0.0) for key in changed)
    corrupted_evals = eval_candidate(corrupted, task, settings, sample_limit=0)
    center_logits = e7a8.quantized_forward(center, task["validation"]["x"], settings.matrix_steps)
    corrupted_logits = e7a8.quantized_forward(corrupted, task["validation"]["x"], settings.matrix_steps)
    output_distance = float(np.mean(np.abs(center_logits - corrupted_logits)))
    center_eval = eval_accuracy(center_evals)
    corrupted_eval = eval_accuracy(corrupted_evals)
    return {
        "raw_hamming_distance_to_center": len(changed),
        "normalized_hamming_distance_to_center": round_float(len(changed) / max(1, total)),
        "sensitivity_weighted_bit_distance_to_center": round_float(changed_sensitivity / max(total_sensitivity, 1e-12)),
        "output_distance_to_center_seed": round_float(output_distance),
        "output_distance_to_teacher": round_float(output_distance),
        "seed_eval_before_repair": corrupted_eval,
        "eval_gap_to_center_seed": round_float(center_eval - corrupted_eval),
        "eval_gap_to_qat_reference": round_float(center_eval - corrupted_eval),
        "seed_solve_passed_before_repair": solve_pass_from_evals(corrupted_evals),
    }


def mutate_for_repair(candidate: dict[str, Any], rng: random.Random, settings: Settings) -> tuple[dict[str, Any], str, int]:
    child = copy.deepcopy(candidate)
    operator = rng.choice(("single_bit_flip", "k_bit_flip", "block_flip", "row_channel_flip", "scale_mutation", "bits_plus_scale"))
    paths = all_paths(child)
    changed = 0
    if operator == "single_bit_flip":
        path, _ = rng.choice(paths)
        flip_path(child, path)
        changed += 1
    elif operator == "k_bit_flip":
        for path, _ in rng.sample(paths, k=min(settings.mutation_steps, len(paths))):
            flip_path(child, path)
            changed += 1
    elif operator == "block_flip":
        block = rng.choice(tuple(BLOCK_KEYS))
        block_paths = [(path, value) for path, value in paths if key_from_path(path) in BLOCK_KEYS[block]]
        for path, _ in rng.sample(block_paths, k=min(settings.mutation_steps * 3, len(block_paths))):
            flip_path(child, path)
            changed += 1
    elif operator == "row_channel_flip":
        key = rng.choice(("win", "state", "wout"))
        arr = np.asarray(child["q"][key], dtype=np.int16)
        if rng.random() < 0.5:
            row = rng.randrange(arr.shape[0])
            arr[row, :] = -arr[row, :]
            changed += int(arr.shape[1])
        else:
            col = rng.randrange(arr.shape[1])
            arr[:, col] = -arr[:, col]
            changed += int(arr.shape[0])
        child["q"][key] = arr.tolist()
    elif operator == "scale_mutation":
        key = rng.choice(PARAM_KEYS)
        raw = np.asarray(child["scales"][key], dtype=np.float64)
        factor = math.exp(rng.gauss(0.0, settings.scale_sigma))
        updated = np.maximum(raw * factor, 1e-8)
        child["scales"][key] = round_float(float(updated)) if updated.ndim == 0 else [round_float(v) for v in updated.reshape(-1).tolist()]
        changed += 1
    elif operator == "bits_plus_scale":
        for path, _ in rng.sample(paths, k=min(max(1, settings.mutation_steps // 2), len(paths))):
            flip_path(child, path)
            changed += 1
        key = rng.choice(PARAM_KEYS)
        raw = np.asarray(child["scales"][key], dtype=np.float64)
        factor = math.exp(rng.gauss(0.0, settings.scale_sigma))
        updated = np.maximum(raw * factor, 1e-8)
        child["scales"][key] = round_float(float(updated)) if updated.ndim == 0 else [round_float(v) for v in updated.reshape(-1).tolist()]
        changed += 1
    update_candidate_hash(child)
    return child, operator, changed


def repair_worker(job: dict[str, Any]) -> dict[str, Any]:
    settings = Settings(**job["settings"])
    task = job["task"]
    out = Path(job["out"]) if job.get("out") else None
    shell_id = job["shell_id"]
    budget_multiplier = int(job["budget_multiplier"])
    generations = settings.base_repair_generations * budget_multiplier
    rng = random.Random(stable_seed(f"repair-{shell_id}-{budget_multiplier}"))
    seed_candidate = job["candidate"]
    population = [score_candidate(copy.deepcopy(seed_candidate), task, settings)]
    for _ in range(settings.mutation_population - 1):
        child, _, _ = mutate_for_repair(seed_candidate, rng, settings)
        population.append(score_candidate(child, task, settings))
    accepted = rejected = rollback = attempts = changed_total = 0
    accepted_by_operator = {}
    rejected_by_operator = {}
    best_eval = -1.0
    budget_to_best = 0
    history = []
    last_heartbeat = time.monotonic()
    for generation in range(1, generations + 1):
        population.sort(key=lambda row: row["fitness"], reverse=True)
        next_population = copy.deepcopy(population[: max(1, settings.elite_count)])
        while len(next_population) < settings.mutation_population:
            parent = copy.deepcopy(rng.choice(population))
            child_candidate, operator, changed = mutate_for_repair(parent["candidate"], rng, settings)
            child = score_candidate(child_candidate, task, settings)
            attempts += 1
            changed_total += changed
            if child["fitness"] >= parent["fitness"]:
                accepted += 1
                accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
                next_population.append(child)
            else:
                rejected += 1
                rollback += 1
                rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
                next_population.append(parent)
        population = next_population
        best = max(population, key=lambda row: row["fitness"])
        current_eval = best["evals"]["validation"]["metrics"]["accuracy"]
        if current_eval > best_eval:
            best_eval = current_eval
            budget_to_best = attempts
        row = {"generation": generation, "best_fitness": best["fitness"], "validation_accuracy": current_eval, "attempts": attempts}
        history.append(row)
        if out and (time.monotonic() - last_heartbeat >= settings.heartbeat_seconds or generation == generations):
            safe_id = safe_file_id(f"{shell_id}_budget{budget_multiplier}")
            locked_write_json(out / "partial_status" / f"{safe_id}.json", row)
            locked_write_json(
                out / "mutation_history_snapshots" / f"{safe_id}.json",
                {
                    "schema_version": "e7a13a_mutation_history_snapshot_v1",
                    "shell_id": shell_id,
                    "budget_multiplier": budget_multiplier,
                    "history_tail": history[-25:],
                    "accepted_mutations": accepted,
                    "rejected_mutations": rejected,
                    "rollback_count": rollback,
                    "mutation_attempts": attempts,
                },
            )
            locked_write_json(
                out / "current_best_candidate_summary" / f"{safe_id}.json",
                {
                    "schema_version": "e7a13a_current_best_candidate_summary_v1",
                    "shell_id": shell_id,
                    "budget_multiplier": budget_multiplier,
                    "generation": generation,
                    "candidate_hash": candidate_hash(best["candidate"]),
                    "best_fitness": best["fitness"],
                    "validation_accuracy": current_eval,
                    "mutation_attempts": attempts,
                },
            )
            append_progress_locked(out, "repair_generation", shell_id=shell_id, budget_multiplier=budget_multiplier, generation=generation, validation_accuracy=current_eval)
            last_heartbeat = time.monotonic()
    best = max(population, key=lambda row: row["fitness"])
    final_candidate = best["candidate"]
    final_evals = eval_candidate(final_candidate, task, settings, sample_limit=10)
    final_eval = eval_accuracy(final_evals)
    center_eval = float(job["center_eval"])
    before_eval = float(job["seed_eval_before_repair"])
    return {
        "shell_id": shell_id,
        "budget_multiplier": budget_multiplier,
        "repair_generations": generations,
        "seed_eval_before_repair": before_eval,
        "seed_eval_after_repair": final_eval,
        "repair_gain": round_float(final_eval - before_eval),
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rollback,
        "mutation_attempts": attempts,
        "accepted_by_operator": accepted_by_operator,
        "rejected_by_operator": rejected_by_operator,
        "mean_changed_parameters_per_attempt": round_float(changed_total / max(1, attempts)),
        "budget_to_best_eval": budget_to_best,
        "solve_passed": solve_pass_from_evals(final_evals),
        "recovered_to_within_epsilon": final_eval >= center_eval - settings.recovery_epsilon,
        "final_eval": final_eval,
        "final_hash": candidate_hash(final_candidate),
        "row_level_samples": {split: final_evals[split]["row_level_samples"] for split in eval_splits()},
        "history_tail": history[-10:],
    }


def fit_models(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {"best_model": "flat_null", "models": {}}
    xs = np.asarray([row["distance"] for row in points], dtype=np.float64)
    ys = np.asarray([row["recovery_rate_to_center"] for row in points], dtype=np.float64)

    def sse(pred: np.ndarray) -> float:
        return float(np.sum((ys - np.clip(pred, 0.0, 1.0)) ** 2))

    flat_pred = np.full_like(ys, float(np.mean(ys)))
    models = {"flat_null": {"sse": round_float(sse(flat_pred)), "params": {"p": round_float(float(np.mean(ys)))}}}
    best_exp = (1e18, 0.0, 0.0)
    for a in np.linspace(0.1, 1.0, 10):
        for b in np.linspace(0.0, 40.0, 81):
            score = sse(a * np.exp(-b * xs))
            if score < best_exp[0]:
                best_exp = (score, a, b)
    models["exponential_falloff"] = {"sse": round_float(best_exp[0]), "params": {"a": round_float(best_exp[1]), "b": round_float(best_exp[2])}}
    best_power = (1e18, 0.0, 0.0)
    for a in np.linspace(0.1, 1.0, 10):
        for b in np.linspace(0.0, 8.0, 65):
            score = sse(a / np.power(1.0 + 20.0 * xs, b))
            if score < best_power[0]:
                best_power = (score, a, b)
    models["power_law_falloff"] = {"sse": round_float(best_power[0]), "params": {"a": round_float(best_power[1]), "b": round_float(best_power[2])}}
    best_log = (1e18, 0.0, 0.0)
    for center in np.linspace(0.0, 0.4, 41):
        for k in np.linspace(1.0, 80.0, 80):
            pred = 1.0 / (1.0 + np.exp(k * (xs - center)))
            score = sse(pred)
            if score < best_log[0]:
                best_log = (score, center, k)
    models["logistic_falloff"] = {"sse": round_float(best_log[0]), "params": {"center": round_float(best_log[1]), "k": round_float(best_log[2])}}
    best_model = min(models, key=lambda name: models[name]["sse"])
    return {"best_model": best_model, "models": models}


def build_center(settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    base_settings = e7a6_settings(settings)
    task = e7a3.generate_task(e7a6.e7a3_settings(base_settings))
    trained = e7a6.train_float_core(settings.width, task, base_settings, out)
    float_eval = e7a10.result_row(e7a10.float_result(trained))["eval_accuracy"]
    run_settings = e7a10_settings(settings)
    center_result = e7a10.train_binary_qat_scale_mode("e7a13a_center_binary_block_qat", "block", settings.width, trained["state_dict"], task, run_settings, out.as_posix() if out else None, float_eval, 0.0)
    center = center_result["candidate"]
    center_evals = eval_candidate(center, task, settings, sample_limit=10)
    metadata = path_metadata(center, trained["state_dict"], task, settings)
    center_report = {
        "schema_version": "e7a13a_center_seed_report_v1",
        "center_source": "deterministically_rebuilt_block_scale_binary_qat_seed",
        "inherits_from": "E7A12_BINARY_MUTATION_ONLY_VIABILITY_AUDIT",
        "width": settings.width,
        "seeds": list(settings.seeds),
        "center_hash": candidate_hash(center),
        "center_eval_accuracy": eval_accuracy(center_evals),
        "center_solve_passed": solve_pass_from_evals(center_evals),
        "center_split_metrics": {split: center_evals[split]["metrics"] for split in task_splits()},
        "scale_storage_mode": center.get("scale_storage_mode"),
        "parameter_count": e7a8.quantized_parameter_count(center),
        "bit_cost": e7a10.bit_cost(center, e7a8.quantized_parameter_count(center)),
    }
    return center, task, center_evals, center_report, metadata


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    payload["distances"] = list(settings.distances)
    payload["budget_multipliers"] = list(settings.budget_multipliers)
    payload["skipped_budget_multipliers"] = list(settings.skipped_budget_multipliers)
    return payload


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    started = time.monotonic()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, settings=settings_payload(settings))
    center, task, center_evals, center_report, metadata = build_center(settings, out)
    center_eval = center_report["center_eval_accuracy"]
    if out:
        append_progress_locked(out, "center_seed_built", center_hash=center_report["center_hash"], center_eval=center_eval)
    shells = []
    shell_candidates = {}
    for mode in CORRUPTION_MODES:
        for distance in settings.distances:
            for replicate in range(settings.shell_replicates):
                shell_id = f"{mode}/d{distance:.4f}/r{replicate}"
                corrupted, shell_meta = corrupt_seed(center, metadata, distance, mode, replicate, settings)
                dist = distance_metrics(center, corrupted, metadata, task, center_evals, settings)
                shell_row = {
                    "shell_id": shell_id,
                    **shell_meta,
                    **dist,
                }
                shells.append(shell_row)
                shell_candidates[shell_id] = corrupted
    repair_rows = []
    jobs = []
    for shell in shells:
        for budget in settings.budget_multipliers:
            jobs.append(
                {
                    "shell_id": shell["shell_id"],
                    "budget_multiplier": budget,
                    "candidate": shell_candidates[shell["shell_id"]],
                    "task": task,
                    "center_eval": center_eval,
                    "seed_eval_before_repair": shell["seed_eval_before_repair"],
                    "settings": settings.__dict__,
                    "out": out.as_posix() if out else None,
                }
            )
    if settings.execution_mode == "parallel":
        with ProcessPoolExecutor(max_workers=max(1, settings.parallel_workers)) as executor:
            future_map = {executor.submit(repair_worker, job): job["shell_id"] for job in jobs}
            pending = set(future_map)
            if out:
                append_progress_locked(out, "repair_jobs_submitted", job_count=len(jobs), workers=settings.parallel_workers)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    repair_rows.append(row)
                    if out:
                        locked_write_json(
                            out / "partial_aggregate_snapshot.json",
                            {
                                "schema_version": "e7a13a_partial_aggregate_snapshot_v1",
                                "completed_repair_jobs": len(repair_rows),
                                "total_repair_jobs": len(jobs),
                                "mean_final_eval_so_far": round_float(float(np.mean([item["final_eval"] for item in repair_rows]))),
                                "mean_repair_gain_so_far": round_float(float(np.mean([item["repair_gain"] for item in repair_rows]))),
                            },
                        )
                        append_progress_locked(out, "repair_job_complete", shell_id=row["shell_id"], budget_multiplier=row["budget_multiplier"], final_eval=row["final_eval"], pending=len(pending))
    else:
        for job in jobs:
            row = repair_worker(job)
            repair_rows.append(row)
            if out:
                locked_write_json(
                    out / "partial_aggregate_snapshot.json",
                    {
                        "schema_version": "e7a13a_partial_aggregate_snapshot_v1",
                        "completed_repair_jobs": len(repair_rows),
                        "total_repair_jobs": len(jobs),
                        "mean_final_eval_so_far": round_float(float(np.mean([item["final_eval"] for item in repair_rows]))),
                        "mean_repair_gain_so_far": round_float(float(np.mean([item["repair_gain"] for item in repair_rows]))),
                    },
                )
    repair_rows.sort(key=lambda row: (row["shell_id"], row["budget_multiplier"]))
    results = {
        "center": center_report,
        "shells": shells,
        "repair_rows": repair_rows,
        "skipped_budgets": [
            {
                "budget_multiplier": multiplier,
                "reason": "skipped_by_default_to_avoid_combinatorial_runtime; rerun with --budget-multipliers including this value for full long-range budget audit",
            }
            for multiplier in settings.skipped_budget_multipliers
        ],
        "runtime": {"elapsed_seconds": round_float(time.monotonic() - started)},
    }
    return results


def aggregate_radius(results: dict[str, Any], settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    shell_by_id = {row["shell_id"]: row for row in results["shells"]}
    grouped = {}
    for row in results["repair_rows"]:
        shell = shell_by_id[row["shell_id"]]
        key = (shell["requested_distance"], shell["corruption_mode"], row["budget_multiplier"])
        grouped.setdefault(key, []).append({**shell, **row})
    buckets = []
    for (distance, mode, budget), rows in sorted(grouped.items()):
        buckets.append(
            {
                "distance_bucket": distance,
                "corruption_mode": mode,
                "budget_multiplier": budget,
                "mean_seed_eval_before_repair": round_float(float(np.mean([r["seed_eval_before_repair"] for r in rows]))),
                "mean_repair_gain": round_float(float(np.mean([r["repair_gain"] for r in rows]))),
                "mean_final_eval": round_float(float(np.mean([r["final_eval"] for r in rows]))),
                "solve_rate": round_float(float(np.mean([1.0 if r["solve_passed"] else 0.0 for r in rows]))),
                "recovery_rate_to_center": round_float(float(np.mean([1.0 if r["recovered_to_within_epsilon"] else 0.0 for r in rows]))),
                "mean_budget_to_best": round_float(float(np.mean([r["budget_to_best_eval"] for r in rows]))),
                "n_cases": len(rows),
            }
        )
    by_distance = []
    for distance in settings.distances:
        rows = [row for row in buckets if row["distance_bucket"] == distance]
        by_distance.append(
            {
                "distance": distance,
                "mean_repair_gain": round_float(float(np.mean([row["mean_repair_gain"] for row in rows]))) if rows else 0.0,
                "mean_final_eval": round_float(float(np.mean([row["mean_final_eval"] for row in rows]))) if rows else 0.0,
                "recovery_rate_to_center": round_float(float(np.mean([row["recovery_rate_to_center"] for row in rows]))) if rows else 0.0,
                "solve_rate": round_float(float(np.mean([row["solve_rate"] for row in rows]))) if rows else 0.0,
                "n_buckets": len(rows),
            }
        )
    fit = fit_models(by_distance)
    best_model = fit["best_model"]
    positive_repair = any(row["mean_repair_gain"] > 0.001 for row in buckets)
    if not positive_repair:
        classification = "no_measurable_repair_basin"
    elif best_model == "logistic_falloff":
        classification = "sharp_capture_boundary"
    elif best_model in {"exponential_falloff", "power_law_falloff"}:
        classification = "smooth_falloff"
    else:
        classification = "ragged_island_basin"
    return (
        {
            "schema_version": "e7a13a_capture_radius_report_v1",
            "buckets": buckets,
            "by_distance": by_distance,
            "classification": classification,
            "recovery_epsilon": settings.recovery_epsilon,
            "skipped_budgets": results["skipped_budgets"],
        },
        {
            "schema_version": "e7a13a_falloff_model_report_v1",
            "fit_target": "recovery_rate_to_center_vs_normalized_hamming_distance_bucket",
            "classification": classification,
            **fit,
        },
    )


def build_payloads(out: Path, results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    capture, falloff = aggregate_radius(results, settings)
    decision = "e7a13a_capture_radius_measured"
    payloads: dict[str, Any] = {
        "backend_manifest.json": {
            "schema_version": "e7a13a_backend_manifest_v1",
            "milestone": MILESTONE,
            "settings": settings_payload(settings),
            "corruption_modes": list(CORRUPTION_MODES),
            "parallel_replay_supported": True,
            "broad_claims_intentionally_deferred": True,
        },
        "center_seed_report.json": results["center"],
        "shell_metrics.json": {
            "schema_version": "e7a13a_shell_metrics_v1",
            "shell_count": len(results["shells"]),
            "required_corruption_modes": list(CORRUPTION_MODES),
            "rows": results["shells"],
        },
        "repair_metrics.json": {
            "schema_version": "e7a13a_repair_metrics_v1",
            "repair_run_count": len(results["repair_rows"]),
            "budget_multipliers": list(settings.budget_multipliers),
            "skipped_budgets": results["skipped_budgets"],
            "rows": results["repair_rows"],
        },
        "capture_radius_report.json": capture,
        "falloff_model_report.json": falloff,
        "summary.json": {
            "schema_version": "e7a13a_summary_v1",
            "decision": decision,
            "classification": capture["classification"],
            "center_eval_accuracy": results["center"]["center_eval_accuracy"],
            "shell_count": len(results["shells"]),
            "repair_run_count": len(results["repair_rows"]),
            "deterministic_replay_passed": False,
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
            "broad_claims_intentionally_deferred": True,
        },
    }
    payloads["final_summary.md"] = build_markdown(out, payloads)
    return payloads


def build_markdown(out: Path, payloads: dict[str, Any]) -> str:
    summary = payloads["summary.json"]
    capture = payloads["capture_radius_report.json"]
    lines = [
        "# E7A13A Capture Radius Atlas Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"classification = {summary['classification']}",
        f"center_eval = {summary['center_eval_accuracy']}",
        "broad_claims = intentionally deferred",
        "```",
        "",
        f"Run root: `{out.relative_to(REPO_ROOT).as_posix()}`",
        "",
        "## Distance Summary",
        "",
        "| distance | recovery rate | solve rate | mean repair gain | mean final eval |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in capture["by_distance"]:
        lines.append(
            f"| {row['distance']:.4f} | {row['recovery_rate_to_center']:.6f} | {row['solve_rate']:.6f} | "
            f"{row['mean_repair_gain']:.6f} | {row['mean_final_eval']:.6f} |"
        )
    lines.extend(["", "This is a controlled binary matrix-core capture-radius audit only.", ""])
    return "\n".join(lines)


def compute_hashes(payloads: dict[str, Any]) -> dict[str, str]:
    return {name: payload_sha256(payloads[name]) for name in HASH_ARTIFACTS}


def deterministic_replay(settings: Settings, out: Path, primary_payloads: dict[str, Any]) -> dict[str, Any]:
    replay_out = out / "deterministic_replay_work"
    if replay_out.exists():
        shutil.rmtree(replay_out)
    append_progress_locked(out, "deterministic_replay_start", replay_out=replay_out.relative_to(REPO_ROOT).as_posix())
    replay_results = run_core(settings, replay_out)
    replay_payloads = build_payloads(out, replay_results, settings)
    primary_hashes = compute_hashes(primary_payloads)
    replay_hashes = compute_hashes(replay_payloads)
    comparisons = {
        name: {"primary_hash": primary_hashes[name], "replay_hash": replay_hashes[name], "match": primary_hashes[name] == replay_hashes[name]}
        for name in HASH_ARTIFACTS
    }
    report = {
        "schema_version": "e7a13a_deterministic_replay_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
        "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix(),
    }
    append_progress_locked(out, "deterministic_replay_complete", internal_replay_passed=report["internal_replay_passed"])
    return report


def write_final_artifacts(out: Path, payloads: dict[str, Any], replay: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["deterministic_replay.json"] = replay
    payloads["summary.json"]["deterministic_replay_passed"] = replay["internal_replay_passed"]
    for name, payload in payloads.items():
        if name == "final_summary.md":
            write_text(out / name, payload)
        else:
            write_json(out / name, payload)
    append_progress_locked(out, "final_artifacts_written", artifact_count=len(payloads))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT.as_posix())
    parser.add_argument("--seeds", default="87001,87002")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--train-rows-per-seed", type=int, default=180)
    parser.add_argument("--validation-rows-per-seed", type=int, default=80)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=80)
    parser.add_argument("--ood-rows-per-seed", type=int, default=80)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=80)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=80)
    parser.add_argument("--gradient-epochs", type=int, default=100)
    parser.add_argument("--best-effort-qat-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--best-effort-learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--matrix-steps", type=int, default=4)
    parser.add_argument("--distillation-weight", type=float, default=0.35)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--distances", default="0,0.005,0.01,0.02,0.05,0.10,0.20,0.40")
    parser.add_argument("--shell-replicates", type=int, default=1)
    parser.add_argument("--budget-multipliers", default="1,4")
    parser.add_argument("--skipped-budget-multipliers", default="16")
    parser.add_argument("--base-repair-generations", type=int, default=24)
    parser.add_argument("--mutation-population", type=int, default=14)
    parser.add_argument("--mutation-steps", type=int, default=16)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--scale-sigma", type=float, default=0.08)
    parser.add_argument("--recovery-epsilon", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--execution-mode", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--parallel-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 2, 20)))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    settings = Settings(
        seeds=parse_int_tuple(args.seeds),
        width=args.width,
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        gradient_epochs=args.gradient_epochs,
        best_effort_qat_epochs=args.best_effort_qat_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        best_effort_learning_rate=args.best_effort_learning_rate,
        weight_decay=args.weight_decay,
        matrix_steps=args.matrix_steps,
        distillation_weight=args.distillation_weight,
        distillation_temperature=args.distillation_temperature,
        distances=parse_float_tuple(args.distances),
        shell_replicates=args.shell_replicates,
        budget_multipliers=parse_int_tuple(args.budget_multipliers),
        skipped_budget_multipliers=parse_int_tuple(args.skipped_budget_multipliers) if args.skipped_budget_multipliers.strip() else tuple(),
        base_repair_generations=args.base_repair_generations,
        mutation_population=args.mutation_population,
        mutation_steps=args.mutation_steps,
        elite_count=args.elite_count,
        scale_sigma=args.scale_sigma,
        recovery_epsilon=args.recovery_epsilon,
        device=args.device,
        execution_mode=args.execution_mode,
        parallel_workers=args.parallel_workers,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    results = run_core(settings, out)
    payloads = build_payloads(out, results, settings)
    replay = deterministic_replay(settings, out, payloads)
    write_final_artifacts(out, payloads, replay)
    print(json.dumps({"decision": payloads["summary.json"]["decision"], "classification": payloads["summary.json"]["classification"], "deterministic_replay_passed": replay["internal_replay_passed"], "out": out.as_posix()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
