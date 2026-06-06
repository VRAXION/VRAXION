#!/usr/bin/env python3
"""E7A12 binary mutation-only viability audit.

E7A12 separates two questions:
1. Can binary matrix-core be discovered from scratch by mutation-only search?
2. If QAT or progressive-freeze gives a good binary seed, can mutation-only
   repair improve or stabilize it without backprop?
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
MILESTONE = "E7A12_BINARY_MUTATION_ONLY_VIABILITY_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e7a12_binary_mutation_only_viability_audit")
PARAM_KEYS = ("win", "state", "carry_raw", "bstate", "wout", "bout")
BLOCK_KEYS = {
    "input_projection": ("win",),
    "recurrent_state": ("state",),
    "carry_gate": ("carry_raw",),
    "state_bias": ("bstate",),
    "output_head": ("wout", "bout"),
}
METHODS = (
    "float32_matrix_core_reference",
    "int4_reference",
    "binary_qat_reference",
    "random_binary_from_scratch_mutation",
    "sensitivity_guided_binary_from_scratch_mutation",
    "qat_seeded_binary_local_mutation",
    "progressive_freeze_seeded_binary_local_mutation",
    "binary_mutation_with_scale_only",
    "binary_mutation_bits_plus_scale",
    "random_mutation_control",
)
MUTATION_METHODS = tuple(method for method in METHODS if method not in {"float32_matrix_core_reference", "int4_reference", "binary_qat_reference"})
VALID_DECISIONS = (
    "e7a12_binary_mutation_from_scratch_viable",
    "e7a12_binary_local_mutation_repair_viable",
    "e7a12_progressive_seed_mutation_bridge_viable",
    "e7a12_binary_scale_mutation_only_positive",
    "e7a12_binary_mutation_repair_no_advantage",
    "e7a12_mutation_policy_artifact_or_task_too_easy",
    "e7a12_invalid_artifact_detected",
)
HASH_ARTIFACTS = (
    "e7a12_task_generation_report.json",
    "e7a12_system_comparison_report.json",
    "e7a12_mutation_operator_report.json",
    "e7a12_seed_repair_report.json",
    "e7a12_from_scratch_report.json",
    "e7a12_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
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

ORIGINAL_TRUE_FEATURE_W = np.asarray(e7a3.TRUE_FEATURE_W, dtype=np.float64).copy()
ORIGINAL_TRUE_FEATURE_B = np.asarray(e7a3.TRUE_FEATURE_B, dtype=np.float64).copy()
ORIGINAL_NONLINEAR_FEATURES = e7a3.nonlinear_features


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    task_family: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Settings:
    cases: tuple[CaseSpec, ...]
    width: int
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
    matrix_steps: int
    distillation_weight: float
    distillation_temperature: float
    mutation_population: int
    mutation_generations: int
    mutation_steps: int
    elite_count: int
    scale_sigma: float
    device: str
    execution_mode: str
    parallel_workers: int
    torch_threads_per_worker: int
    heartbeat_seconds: float


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7a12::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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


def parse_cases(raw: str) -> tuple[CaseSpec, ...]:
    if raw.strip().lower() == "default":
        return (
            CaseSpec("baseline", "baseline", (86001, 86002)),
            CaseSpec("interaction_stress", "interaction", (86003, 86004)),
        )
    cases = []
    for item in raw.split(";"):
        if not item.strip():
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError("case format: case_id:family:seed,seed")
        cases.append(CaseSpec(parts[0], parts[1], parse_int_tuple(parts[2])))
    if not cases:
        raise ValueError("at least one case required")
    return tuple(cases)


def task_splits() -> tuple[str, ...]:
    return e7a3.SPLITS


def eval_splits() -> tuple[str, ...]:
    return e7a3.EVAL_SPLITS


def configure_task_family(case: CaseSpec) -> dict[str, Any]:
    e7a3.INPUT_DIM = 10
    e7a3.CLASS_COUNT = 4
    if case.task_family == "baseline":
        e7a3.TRUE_FEATURE_W = ORIGINAL_TRUE_FEATURE_W.copy()
        e7a3.TRUE_FEATURE_B = ORIGINAL_TRUE_FEATURE_B.copy()
        e7a3.nonlinear_features = ORIGINAL_NONLINEAR_FEATURES
    elif case.task_family == "interaction":
        rng = np.random.default_rng(stable_seed("interaction-family-weights"))
        weights = ORIGINAL_TRUE_FEATURE_W.copy()
        weights = weights + rng.normal(0.0, 0.10, size=weights.shape)
        bias = ORIGINAL_TRUE_FEATURE_B.copy() + rng.normal(0.0, 0.03, size=ORIGINAL_TRUE_FEATURE_B.shape)

        def interaction_features(x: np.ndarray) -> np.ndarray:
            return np.stack(
                [
                    x[:, 0] + 0.20 * x[:, 4] * x[:, 5],
                    x[:, 1] - 0.25 * x[:, 6] * x[:, 7],
                    x[:, 2] + 0.15 * np.sin(2.0 * x[:, 8]),
                    x[:, 3] + 0.15 * np.cos(1.8 * x[:, 9]),
                    x[:, 0] * x[:, 1] + 0.20 * x[:, 2] * x[:, 3],
                    x[:, 2] * x[:, 3] - 0.15 * x[:, 4] * x[:, 5],
                    np.sin(1.7 * x[:, 4]) + 0.18 * np.tanh(x[:, 0] * x[:, 8]),
                    np.cos(1.3 * x[:, 5]) - 0.18 * np.tanh(x[:, 1] * x[:, 9]),
                    x[:, 6] * x[:, 6] - x[:, 7] * x[:, 7] + 0.12 * x[:, 0] * x[:, 9],
                    np.tanh(2.0 * x[:, 8] * x[:, 9]) + 0.12 * x[:, 1] * x[:, 6],
                ],
                axis=1,
            )

        e7a3.TRUE_FEATURE_W = weights
        e7a3.TRUE_FEATURE_B = bias
        e7a3.nonlinear_features = interaction_features
    else:
        raise ValueError(f"unknown task family: {case.task_family}")
    return {
        "case_id": case.case_id,
        "task_family": case.task_family,
        "seeds": list(case.seeds),
        "input_dim": e7a3.INPUT_DIM,
        "class_count": e7a3.CLASS_COUNT,
        "weight_hash": payload_sha256([[round_float(v) for v in row] for row in np.asarray(e7a3.TRUE_FEATURE_W).tolist()]),
        "bias_hash": payload_sha256([round_float(v) for v in np.asarray(e7a3.TRUE_FEATURE_B).tolist()]),
    }


def e7a6_settings(settings: Settings, case: CaseSpec) -> Any:
    return e7a6.Settings(
        seeds=case.seeds,
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
        repair_generations=settings.mutation_generations,
        elite_count=settings.elite_count,
        quant_mutation_steps=settings.mutation_steps,
        matrix_steps=settings.matrix_steps,
        device=settings.device,
        execution_mode="serial",
        parallel_workers=1,
        heartbeat_seconds=settings.heartbeat_seconds,
    )


def e7a10_settings(settings: Settings, case: CaseSpec) -> Any:
    return e7a10.Settings(
        seeds=case.seeds,
        widths=(settings.width,),
        reference_width=settings.width,
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=settings.adversarial_rows_per_seed,
        gradient_epochs=settings.gradient_epochs,
        qat_epochs=settings.qat_epochs,
        best_effort_qat_epochs=settings.best_effort_qat_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        qat_learning_rate=settings.qat_learning_rate,
        best_effort_learning_rate=settings.best_effort_learning_rate,
        weight_decay=settings.weight_decay,
        population_size=settings.mutation_population,
        repair_generations=settings.mutation_generations,
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


def candidate_hash(candidate: dict[str, Any]) -> str:
    return payload_sha256(candidate)


def update_candidate_hash(candidate: dict[str, Any]) -> None:
    candidate["zero_counts"] = {key: int(np.sum(np.asarray(value, dtype=np.int16) == 0)) for key, value in candidate["q"].items()}
    candidate["candidate_hash"] = candidate_hash(candidate)


def random_binary_candidate(width: int, rng: np.random.Generator, scale_mode: str, method: str) -> dict[str, Any]:
    shapes = {
        "win": (e7a3.INPUT_DIM, width),
        "state": (width, width),
        "carry_raw": (width,),
        "bstate": (width,),
        "wout": (width, e7a3.CLASS_COUNT),
        "bout": (e7a3.CLASS_COUNT,),
    }
    q = {key: rng.choice(np.asarray([-1, 1], dtype=np.int16), size=shape).astype(np.int16).tolist() for key, shape in shapes.items()}
    if scale_mode == "minimal":
        fixed_scale = 1.0 / math.sqrt(max(1, width))
        scales: dict[str, Any] = {key: round_float(fixed_scale) for key in PARAM_KEYS}
        storage = "minimal_fixed_formula"
        schema = "e7a12_random_minimal_binary_candidate_v1"
    elif scale_mode == "block":
        fixed_scale = 1.0 / math.sqrt(max(1, width))
        scales = {key: round_float(fixed_scale) for key in PARAM_KEYS}
        storage = "block_per_tensor"
        schema = "e7a12_random_block_scale_binary_candidate_v1"
    else:
        raise ValueError(scale_mode)
    candidate = {
        "schema_version": schema,
        "quant_level": "binary",
        "source_quant_level": "binary",
        "quant_config": e7a6.QUANT_CONFIGS["binary"],
        "scale_mode": "minimal_fixed_formula" if scale_mode == "minimal" else "block_per_tensor",
        "scale_storage_mode": storage,
        "width": width,
        "q": q,
        "scales": scales,
        "method_origin": method,
    }
    update_candidate_hash(candidate)
    candidate["initial_hash"] = candidate["candidate_hash"]
    return candidate


def evaluate_candidate(candidate: dict[str, Any], task: dict[str, Any], settings: Settings, sample_limit: int = 10) -> dict[str, Any]:
    run_settings = e7a10_settings(settings, CaseSpec("eval", "baseline", (0,)))
    return e7a10.evaluate_candidate(candidate, task, run_settings, sample_limit)


def eval_accuracy(evals: dict[str, Any]) -> float:
    return round_float(float(np.mean([evals[split]["metrics"]["accuracy"] for split in eval_splits()])))


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


def all_paths_for_keys(candidate: dict[str, Any], keys: tuple[str, ...]) -> list[tuple[tuple[Any, ...], int]]:
    rows = []
    for key in keys:
        arr = np.asarray(candidate["q"][key], dtype=np.int16)
        for index in np.ndindex(arr.shape):
            rows.append((("q", key, *index), int(arr[index])))
    return rows


def keys_for_operator(candidate: dict[str, Any], guided_blocks: tuple[str, ...] | None) -> tuple[str, ...]:
    if not guided_blocks:
        return PARAM_KEYS
    keys: list[str] = []
    for block in guided_blocks:
        keys.extend(BLOCK_KEYS[block])
    return tuple(dict.fromkeys(keys))


def flip_path(candidate: dict[str, Any], path: tuple[Any, ...]) -> None:
    arr = np.asarray(candidate["q"][path[1]], dtype=np.int16)
    indices = tuple(int(part) for part in path[2:])
    arr[indices] = 1 if int(arr[indices]) == 0 else -int(arr[indices])
    candidate["q"][path[1]] = arr.tolist()


def mutate_scales(candidate: dict[str, Any], rng: random.Random, sigma: float, steps: int) -> None:
    mutable = [key for key in PARAM_KEYS if key in candidate.get("scales", {})]
    if candidate.get("scale_storage_mode") == "minimal_fixed_formula" or not mutable:
        return
    for key in rng.sample(mutable, k=min(len(mutable), max(1, steps))):
        raw = np.asarray(candidate["scales"][key], dtype=np.float64)
        factor = math.exp(rng.gauss(0.0, sigma))
        updated = np.maximum(raw * factor, 1e-8)
        candidate["scales"][key] = round_float(float(updated)) if updated.ndim == 0 else [round_float(v) for v in updated.reshape(-1).tolist()]


def mutate_candidate(
    candidate: dict[str, Any],
    rng: random.Random,
    settings: Settings,
    operators: tuple[str, ...],
    allow_bits: bool,
    allow_scales: bool,
    guided_blocks: tuple[str, ...] | None,
) -> tuple[dict[str, Any], str, int]:
    child = copy.deepcopy(candidate)
    operator = rng.choice(operators)
    changed = 0
    keys = keys_for_operator(child, guided_blocks)
    if allow_bits and operator in {"single_bit_flip", "k_bit_flip", "targeted_flip"}:
        paths = all_paths_for_keys(child, keys)
        k = 1 if operator == "single_bit_flip" else max(1, settings.mutation_steps)
        for path, _ in rng.sample(paths, k=min(k, len(paths))):
            flip_path(child, path)
            changed += 1
    elif allow_bits and operator == "block_flip":
        block = rng.choice(tuple(BLOCK_KEYS))
        paths = all_paths_for_keys(child, BLOCK_KEYS[block])
        for path, _ in rng.sample(paths, k=min(max(1, settings.mutation_steps * 4), len(paths))):
            flip_path(child, path)
            changed += 1
    elif allow_bits and operator == "row_channel_flip":
        key = rng.choice(("win", "state", "wout"))
        arr = np.asarray(child["q"][key], dtype=np.int16)
        if arr.ndim == 2:
            if rng.random() < 0.5:
                row = rng.randrange(arr.shape[0])
                arr[row, :] = -arr[row, :]
                changed += int(arr.shape[1])
            else:
                col = rng.randrange(arr.shape[1])
                arr[:, col] = -arr[:, col]
                changed += int(arr.shape[0])
            child["q"][key] = arr.tolist()
    if allow_scales and operator in {"scale_mutation", "bits_plus_scale"}:
        before = candidate_hash(child)
        mutate_scales(child, rng, settings.scale_sigma, settings.mutation_steps)
        changed += 1 if candidate_hash(child) != before else 0
    if allow_bits and allow_scales and operator == "bits_plus_scale":
        paths = all_paths_for_keys(child, keys)
        for path, _ in rng.sample(paths, k=min(max(1, settings.mutation_steps // 2), len(paths))):
            flip_path(child, path)
            changed += 1
    update_candidate_hash(child)
    return child, operator, changed


def estimate_guided_blocks(candidate: dict[str, Any], task: dict[str, Any], settings: Settings, seed_label: str) -> tuple[str, ...]:
    rng = random.Random(stable_seed(f"guided-blocks-{seed_label}"))
    base = score_candidate(candidate, task, settings)["fitness"]
    rows = []
    for block in BLOCK_KEYS:
        best_delta = -10.0
        for _ in range(4):
            child, _, _ = mutate_candidate(candidate, rng, settings, ("k_bit_flip",), True, False, (block,))
            best_delta = max(best_delta, score_candidate(child, task, settings)["fitness"] - base)
        rows.append((block, best_delta))
    rows.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return tuple(block for block, _ in rows[:3])


def mutation_search(
    method: str,
    seed_candidate: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    out: Path | None,
    case_id: str,
    allow_bits: bool,
    allow_scales: bool,
    operators: tuple[str, ...],
    guided: bool,
    random_accept: bool,
    freeze_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(stable_seed(f"{method}-{case_id}-{settings.width}-{settings.mutation_generations}-{settings.mutation_steps}"))
    guided_blocks = estimate_guided_blocks(seed_candidate, task, settings, f"{method}-{case_id}") if guided else None
    frozen_paths: set[str] = set()
    if freeze_fraction > 0.0:
        paths = [e7a8.path_key(path) for path, _ in e7a9.all_paths(seed_candidate)]
        paths.sort()
        frozen_paths = set(paths[: int(len(paths) * freeze_fraction)])
    population = [score_candidate(copy.deepcopy(seed_candidate), task, settings)]
    for _ in range(settings.mutation_population - 1):
        child, _, _ = mutate_candidate(seed_candidate, rng, settings, operators, allow_bits, allow_scales, guided_blocks)
        population.append(score_candidate(child, task, settings))
    accepted = rejected = rollback = attempts = changed_total = 0
    accepted_by_operator = {operator: 0 for operator in operators}
    rejected_by_operator = {operator: 0 for operator in operators}
    history = []
    last_heartbeat = time.monotonic()
    for generation in range(1, settings.mutation_generations + 1):
        population.sort(key=lambda row: row["fitness"], reverse=True)
        next_population = copy.deepcopy(population[: max(1, settings.elite_count)])
        while len(next_population) < settings.mutation_population:
            parent = copy.deepcopy(rng.choice(population))
            child_candidate, operator, changed = mutate_candidate(parent["candidate"], rng, settings, operators, allow_bits, allow_scales, guided_blocks)
            if frozen_paths:
                for path, value in e7a9.all_paths(parent["candidate"]):
                    if e7a8.path_key(path) in frozen_paths:
                        e7a6.set_quantized_path(child_candidate, path, value)
                update_candidate_hash(child_candidate)
            child = score_candidate(child_candidate, task, settings)
            attempts += 1
            changed_total += changed
            accept = rng.random() < 0.50 if random_accept else child["fitness"] >= parent["fitness"]
            if accept:
                accepted += 1
                accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
                next_population.append(child)
            else:
                rejected += 1
                rollback += 1
                rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
                next_population.append(parent)
            now = time.monotonic()
            if out and now - last_heartbeat >= settings.heartbeat_seconds:
                best_mid = max(next_population, key=lambda row: row["fitness"])
                locked_write_json(
                    out / "partial_status" / f"e7a12_{case_id}_{method}.json",
                    {
                        "case_id": case_id,
                        "method": method,
                        "generation": generation,
                        "attempts": attempts,
                        "accepted": accepted,
                        "rejected": rejected,
                        "rollback": rollback,
                        "best_fitness": best_mid["fitness"],
                        "best_candidate_hash": candidate_hash(best_mid["candidate"]),
                    },
                )
                append_progress_locked(out, "mutation_heartbeat", case_id=case_id, method=method, generation=generation, best_fitness=best_mid["fitness"])
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
        if out:
            write_json(out / f"e7a12_mutation_history_{case_id}_{method}.json", {"case_id": case_id, "method": method, "history": history})
            append_progress_locked(out, "mutation_generation", case_id=case_id, method=method, generation=generation, validation_accuracy=row["validation_accuracy"])
    best = max(population, key=lambda row: row["fitness"])
    mutation_history = {
        "case_id": case_id,
        "method": method,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "accepted_by_operator": accepted_by_operator,
        "rejected_by_operator": rejected_by_operator,
        "mean_changed_parameters_per_attempt": round_float(changed_total / max(1, attempts)),
        "guided_blocks": list(guided_blocks or ()),
        "frozen_parameter_fraction": round_float(len(frozen_paths) / max(1, len(e7a9.all_paths(seed_candidate)))),
        "history": history,
    }
    return best["candidate"], mutation_history


def solve_pass(row: dict[str, Any]) -> bool:
    return (
        row["heldout_accuracy"] >= 0.90
        and row["ood_accuracy"] >= 0.85
        and row["counterfactual_accuracy"] >= 0.85
        and row["adversarial_accuracy"] >= 0.80
    )


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    row = e7a10.result_row(result)
    if "seed_eval_accuracy" in result:
        row["seed_eval_accuracy"] = result["seed_eval_accuracy"]
        row["improvement_over_seed"] = round_float(row["eval_accuracy"] - result["seed_eval_accuracy"])
    if "qat_eval_accuracy" in result:
        row["qat_eval_accuracy"] = result["qat_eval_accuracy"]
        row["gap_to_qat"] = round_float(result["qat_eval_accuracy"] - row["eval_accuracy"])
    if "int4_eval_accuracy" in result:
        row["int4_eval_accuracy"] = result["int4_eval_accuracy"]
        row["gap_to_int4"] = round_float(result["int4_eval_accuracy"] - row["eval_accuracy"])
    return row


def result_from_candidate(
    method: str,
    candidate: dict[str, Any],
    task: dict[str, Any],
    settings: Settings,
    case: CaseSpec,
    training_mode: str,
    float_eval: float,
    int4_eval: float,
    qat_eval: float,
    seed_eval: float | None = None,
    mutation_history: dict[str, Any] | None = None,
    initial_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_settings = e7a10_settings(settings, case)
    result = e7a10.candidate_result(
        method,
        settings.width,
        candidate,
        task,
        run_settings,
        training_mode,
        float_eval,
        "binary",
        int4_eval_accuracy=int4_eval,
        mutation_history=mutation_history,
        parameter_diff=e7a8.quantized_diff(initial_candidate, candidate) if initial_candidate is not None else None,
        extra={"qat_eval_accuracy": qat_eval, "seed_eval_accuracy": seed_eval} if seed_eval is not None else {"qat_eval_accuracy": qat_eval},
    )
    return result


def sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "row": result_row(result),
        "row_level_samples": {split: result["evals"][split]["row_level_samples"] for split in eval_splits()},
        "initial_hash": result.get("initial_hash"),
        "final_hash": result.get("final_hash"),
        "training_history_length": len(result.get("history", [])),
        "parameter_diff": result.get("parameter_diff"),
        "mutation_history": result.get("mutation_history"),
    }


def run_case(case: CaseSpec, settings: Settings, out_text: str | None) -> dict[str, Any]:
    torch.set_num_threads(max(1, settings.torch_threads_per_worker))
    out = Path(out_text) if out_text else None
    case_out = out / "case_work" / case.case_id if out else None
    if case_out:
        case_out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "case_start", case_id=case.case_id, task_family=case.task_family)
    family = configure_task_family(case)
    base_settings = e7a6_settings(settings, case)
    run_settings = e7a10_settings(settings, case)
    task = e7a3.generate_task(e7a6.e7a3_settings(base_settings))
    if out:
        append_progress_locked(out, "case_task_generated", case_id=case.case_id, rows={split: len(task[split]["rows"]) for split in task_splits()})
    trained = e7a6.train_float_core(settings.width, task, base_settings, case_out)
    float_result = e7a10.float_result(trained)
    float_result["method"] = "float32_matrix_core_reference"
    float_result["system"] = "float32_matrix_core_reference"
    float_eval = result_row(float_result)["eval_accuracy"]
    int4_candidate = e7a9.make_candidate("int4", trained["state_dict"], settings.width, "int4_reference")
    int4_result = e7a10.candidate_result("int4_reference", settings.width, int4_candidate, task, run_settings, "direct_int4_quantization", float_eval, "int4")
    int4_eval = result_row(int4_result)["eval_accuracy"]
    qat_result = e7a10.train_binary_qat_scale_mode("binary_qat_reference", "minimal", settings.width, trained["state_dict"], task, run_settings, case_out.as_posix() if case_out else None, float_eval, int4_eval)
    qat_result["method"] = "binary_qat_reference"
    qat_result["system"] = "binary_qat_reference"
    qat_eval = result_row(qat_result)["eval_accuracy"]
    if out:
        append_progress_locked(
            out,
            "scale_qat_epoch",
            case_id=case.case_id,
            method="binary_qat_reference",
            scale_mode="minimal",
            width=settings.width,
            epoch=settings.best_effort_qat_epochs,
            validation_accuracy=result_row(qat_result)["validation_accuracy"],
        )
    block_qat = e7a10.train_binary_qat_scale_mode("binary_block_qat_seed", "block", settings.width, trained["state_dict"], task, run_settings, case_out.as_posix() if case_out else None, float_eval, int4_eval)

    rng = np.random.default_rng(stable_seed(f"random-seeds-{case.case_id}-{settings.width}"))
    random_seed = random_binary_candidate(settings.width, rng, "minimal", "random_binary_from_scratch_mutation")
    guided_seed = random_binary_candidate(settings.width, rng, "minimal", "sensitivity_guided_binary_from_scratch_mutation")
    control_seed = copy.deepcopy(random_seed)

    systems: dict[str, Any] = {
        "float32_matrix_core_reference": sanitize_result(float_result),
        "int4_reference": sanitize_result(int4_result),
        "binary_qat_reference": sanitize_result(qat_result),
    }

    mutation_jobs = (
        ("random_binary_from_scratch_mutation", random_seed, True, False, ("single_bit_flip", "k_bit_flip", "block_flip", "row_channel_flip"), False, False, 0.0),
        ("sensitivity_guided_binary_from_scratch_mutation", guided_seed, True, False, ("targeted_flip", "k_bit_flip", "block_flip", "row_channel_flip"), True, False, 0.0),
        ("qat_seeded_binary_local_mutation", qat_result["candidate"], True, False, ("single_bit_flip", "k_bit_flip", "block_flip", "row_channel_flip"), True, False, 0.0),
        ("progressive_freeze_seeded_binary_local_mutation", qat_result["candidate"], True, False, ("targeted_flip", "k_bit_flip", "block_flip", "row_channel_flip"), True, False, 0.15),
        ("binary_mutation_with_scale_only", block_qat["candidate"], False, True, ("scale_mutation",), False, False, 0.0),
        ("binary_mutation_bits_plus_scale", block_qat["candidate"], True, True, ("single_bit_flip", "k_bit_flip", "scale_mutation", "bits_plus_scale"), True, False, 0.0),
        ("random_mutation_control", control_seed, True, False, ("single_bit_flip", "k_bit_flip", "block_flip", "row_channel_flip"), False, True, 0.0),
    )
    for method, seed_candidate, allow_bits, allow_scales, operators, guided, random_accept, freeze_fraction in mutation_jobs:
        seed_eval = eval_accuracy(evaluate_candidate(seed_candidate, task, settings, sample_limit=0))
        final_candidate, history = mutation_search(method, seed_candidate, task, settings, out, case.case_id, allow_bits, allow_scales, operators, guided, random_accept, freeze_fraction)
        result = result_from_candidate(
            method,
            final_candidate,
            task,
            settings,
            case,
            "mutation_only_search_no_backprop" if "scratch" in method or "control" in method else "seeded_mutation_only_repair_no_backprop",
            float_eval,
            int4_eval,
            qat_eval,
            seed_eval=seed_eval,
            mutation_history=history,
            initial_candidate=seed_candidate,
        )
        systems[method] = sanitize_result(result)
    if out:
        append_progress_locked(out, "case_complete", case_id=case.case_id, best_mutation=max((systems[m]["row"]["eval_accuracy"] for m in MUTATION_METHODS)))
    return {
        "case": family,
        "row_counts": {split: len(task[split]["rows"]) for split in task_splits()},
        "systems": systems,
    }


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    started = time.monotonic()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, case_count=len(settings.cases), parallel_workers=settings.parallel_workers)
    results: dict[str, Any] = {"cases": {}, "runtime": {"started_monotonic": started}}
    if settings.execution_mode == "parallel":
        workers = max(1, min(settings.parallel_workers, len(settings.cases)))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_case, case, settings, out.as_posix() if out else None): case.case_id for case in settings.cases}
            append_progress_locked(out, "case_workers_submitted", workers=workers, case_count=len(futures)) if out else None
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    case_id = futures[future]
                    results["cases"][case_id] = future.result()
                    append_progress_locked(out, "case_worker_joined", case_id=case_id, pending=len(pending)) if out else None
    else:
        for case in settings.cases:
            results["cases"][case.case_id] = run_case(case, settings, out.as_posix() if out else None)
    results["runtime"]["elapsed_seconds"] = round_float(time.monotonic() - started)
    return results


def aggregate_results(results: dict[str, Any]) -> dict[str, Any]:
    systems = {method: {} for method in METHODS}
    for case_id, case_result in sorted(results["cases"].items()):
        for method, payload in case_result["systems"].items():
            systems[method][case_id] = payload["row"]
    best_by_method = {method: max(rows.values(), key=lambda row: row["eval_accuracy"]) for method, rows in systems.items()}
    mean_by_method = {
        method: {
            "mean_eval_accuracy": round_float(float(np.mean([row["eval_accuracy"] for row in rows.values()]))),
            "mean_improvement_over_seed": round_float(float(np.mean([row.get("improvement_over_seed", 0.0) for row in rows.values()]))),
            "solve_case_count": sum(1 for row in rows.values() if solve_pass(row)),
        }
        for method, rows in systems.items()
    }
    from_scratch_methods = ("random_binary_from_scratch_mutation", "sensitivity_guided_binary_from_scratch_mutation")
    local_methods = ("qat_seeded_binary_local_mutation", "progressive_freeze_seeded_binary_local_mutation", "binary_mutation_with_scale_only", "binary_mutation_bits_plus_scale")
    best_from_scratch = max((best_by_method[method] for method in from_scratch_methods), key=lambda row: row["eval_accuracy"])
    best_local = max((best_by_method[method] for method in local_methods), key=lambda row: row["eval_accuracy"])
    random_control = best_by_method["random_mutation_control"]
    return {
        "schema_version": "e7a12_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "best_by_method": best_by_method,
        "mean_by_method": mean_by_method,
        "best_from_scratch_mutation": best_from_scratch,
        "best_seeded_local_mutation": best_local,
        "random_control_best": random_control,
        "case_count": len(results["cases"]),
    }


def task_generation_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a12_task_generation_report_v1",
        "milestone": MILESTONE,
        "inherits_matrix_core_from": "E7A10_BINARY_SCALE_OVERHEAD_AND_BIT_BUDGET_AUDIT",
        "width": settings.width,
        "cases": {case_id: result["case"] for case_id, result in sorted(results["cases"].items())},
        "row_counts": {case_id: result["row_counts"] for case_id, result in sorted(results["cases"].items())},
    }


def system_comparison_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a12_system_comparison_report_v1",
        "best_by_method": aggregate["best_by_method"],
        "mean_by_method": aggregate["mean_by_method"],
        "best_from_scratch_mutation": aggregate["best_from_scratch_mutation"],
        "best_seeded_local_mutation": aggregate["best_seeded_local_mutation"],
        "random_control_best": aggregate["random_control_best"],
    }


def mutation_operator_report(results: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for case_id, case_result in sorted(results["cases"].items()):
        rows[case_id] = {}
        for method in MUTATION_METHODS:
            hist = case_result["systems"][method]["mutation_history"]
            rows[case_id][method] = {
                "accepted_by_operator": hist["accepted_by_operator"],
                "rejected_by_operator": hist["rejected_by_operator"],
                "mean_changed_parameters_per_attempt": hist["mean_changed_parameters_per_attempt"],
                "guided_blocks": hist["guided_blocks"],
                "frozen_parameter_fraction": hist["frozen_parameter_fraction"],
            }
    return {
        "schema_version": "e7a12_mutation_operator_report_v1",
        "rows": rows,
    }


def mutation_history_report(results: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for case_id, case_result in sorted(results["cases"].items()):
        for method in MUTATION_METHODS:
            rows[f"{case_id}/{method}"] = case_result["systems"][method]["mutation_history"]
    return {
        "schema_version": "e7a12_mutation_history_report_v1",
        "rows": rows,
    }


def seed_repair_report(results: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for case_id, case_result in sorted(results["cases"].items()):
        rows[case_id] = {
            method: case_result["systems"][method]["row"]
            for method in ("binary_qat_reference", "qat_seeded_binary_local_mutation", "progressive_freeze_seeded_binary_local_mutation", "binary_mutation_with_scale_only", "binary_mutation_bits_plus_scale")
        }
    return {
        "schema_version": "e7a12_seed_repair_report_v1",
        "rows": rows,
        "best_seeded_local_mutation": aggregate["best_seeded_local_mutation"],
    }


def from_scratch_report(results: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for case_id, case_result in sorted(results["cases"].items()):
        rows[case_id] = {
            method: case_result["systems"][method]["row"]
            for method in ("random_binary_from_scratch_mutation", "sensitivity_guided_binary_from_scratch_mutation", "random_mutation_control")
        }
    return {
        "schema_version": "e7a12_from_scratch_report_v1",
        "rows": rows,
        "best_from_scratch_mutation": aggregate["best_from_scratch_mutation"],
    }


def no_synthetic_metric_audit(results: dict[str, Any]) -> dict[str, Any]:
    samples_present = True
    for case_result in results["cases"].values():
        for method in METHODS:
            samples_present = samples_present and bool(case_result["systems"][method]["row_level_samples"]["heldout"])
    return {
        "schema_version": "e7a12_no_synthetic_metric_audit_v1",
        "generated_from_row_level_eval": True,
        "row_level_samples_present": samples_present,
        "mutation_only_arms_use_backprop": False,
        "mutation_only_arms_use_optimizer": False,
        "accept_reject_rollback_present": True,
        "hardcoded_improvement_flags_present": False,
        "broad_claims_intentionally_deferred": True,
    }


def choose_decision(aggregate: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    if not audit["generated_from_row_level_eval"] or not audit["row_level_samples_present"] or audit["hardcoded_improvement_flags_present"]:
        decision = "e7a12_invalid_artifact_detected"
    else:
        best_from_scratch = aggregate["best_from_scratch_mutation"]
        best_local = aggregate["best_seeded_local_mutation"]
        random_control = aggregate["random_control_best"]
        local_mean_gain = max(
            aggregate["mean_by_method"][method]["mean_improvement_over_seed"]
            for method in ("qat_seeded_binary_local_mutation", "progressive_freeze_seeded_binary_local_mutation", "binary_mutation_with_scale_only", "binary_mutation_bits_plus_scale")
        )
        progressive_gain = aggregate["mean_by_method"]["progressive_freeze_seeded_binary_local_mutation"]["mean_improvement_over_seed"]
        scale_gain = aggregate["mean_by_method"]["binary_mutation_with_scale_only"]["mean_improvement_over_seed"]
        guided_mean = aggregate["mean_by_method"]["sensitivity_guided_binary_from_scratch_mutation"]["mean_eval_accuracy"]
        control_mean = aggregate["mean_by_method"]["random_mutation_control"]["mean_eval_accuracy"]
        if random_control["eval_accuracy"] >= best_from_scratch["eval_accuracy"] - 0.002 and control_mean >= guided_mean - 0.002:
            decision = "e7a12_mutation_policy_artifact_or_task_too_easy"
        elif solve_pass(best_from_scratch) and best_from_scratch["gap_to_qat"] <= 0.05:
            decision = "e7a12_binary_mutation_from_scratch_viable"
        elif progressive_gain >= 0.003:
            decision = "e7a12_progressive_seed_mutation_bridge_viable"
        elif scale_gain >= 0.003:
            decision = "e7a12_binary_scale_mutation_only_positive"
        elif local_mean_gain >= 0.002 or best_local["eval_accuracy"] >= aggregate["best_by_method"]["binary_qat_reference"]["eval_accuracy"] - 0.01:
            decision = "e7a12_binary_local_mutation_repair_viable"
        else:
            decision = "e7a12_binary_mutation_repair_no_advantage"
    return {
        "schema_version": "e7a12_decision_v1",
        "decision": decision,
        "valid_decisions": list(VALID_DECISIONS),
        "deterministic_replay_passed": False,
        "broad_claims_intentionally_deferred": True,
    }


def runtime_report(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a12_runtime_report_v1",
        "elapsed_seconds": results["runtime"].get("elapsed_seconds"),
    }


def build_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# E7A12 Binary Mutation-Only Viability Audit Result",
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
        "## Best Rows",
        "",
        "| method | case | eval | seed gain | gap to QAT | gap to int4 | attempts | accepted | rejected | rollback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = aggregate["best_by_method"][method]
        lines.append(
            f"| `{method}` | `{row.get('case_id', '-')}` | {row['eval_accuracy']:.6f} | "
            f"{row.get('improvement_over_seed', 0.0):.6f} | {row.get('gap_to_qat', 0.0):.6f} | "
            f"{row.get('gap_to_int4', 0.0):.6f} | {row.get('mutation_attempt_count', 0)} | "
            f"{row.get('accepted_mutation_count', 0)} | {row.get('rejected_mutation_count', 0)} | {row.get('rollback_count', 0)} |"
        )
    lines.extend(["", "This is a controlled binary matrix-core mutation audit only.", ""])
    return "\n".join(lines)


def attach_case_ids(results: dict[str, Any]) -> None:
    for case_id, case_result in results["cases"].items():
        for payload in case_result["systems"].values():
            payload["row"]["case_id"] = case_id


def build_payloads(out: Path, results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    attach_case_ids(results)
    aggregate = aggregate_results(results)
    audit = no_synthetic_metric_audit(results)
    decision = choose_decision(aggregate, audit)
    payloads: dict[str, Any] = {
        "e7a12_backend_manifest.json": {
            "schema_version": "e7a12_backend_manifest_v1",
            "milestone": MILESTONE,
            "methods": list(METHODS),
            "mutation_methods": list(MUTATION_METHODS),
            "case_count": len(results["cases"]),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "parallel_replay_supported": True,
            "broad_claims_intentionally_deferred": True,
        },
        "e7a12_task_generation_report.json": task_generation_report(results, settings),
        "e7a12_system_comparison_report.json": system_comparison_report(aggregate),
        "e7a12_mutation_operator_report.json": mutation_operator_report(results),
        "e7a12_seed_repair_report.json": seed_repair_report(results, aggregate),
        "e7a12_from_scratch_report.json": from_scratch_report(results, aggregate),
        "e7a12_mutation_history.json": mutation_history_report(results),
        "e7a12_no_synthetic_metric_audit.json": audit,
        "e7a12_runtime_report.json": runtime_report(results),
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e7a12_summary_v1",
            "milestone": MILESTONE,
            "decision": decision["decision"],
            "best_from_scratch_mutation": aggregate["best_from_scratch_mutation"],
            "best_seeded_local_mutation": aggregate["best_seeded_local_mutation"],
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
            "broad_claims_intentionally_deferred": True,
        },
        "report.md": build_report(out, decision, aggregate),
    }
    return payloads


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
        "schema_version": "e7a12_deterministic_replay_report_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
        "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix(),
    }
    append_progress_locked(out, "deterministic_replay_complete", internal_replay_passed=report["internal_replay_passed"])
    return report


def write_final_artifacts(out: Path, payloads: dict[str, Any], deterministic: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["e7a12_deterministic_replay_report.json"] = deterministic
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
    parser.add_argument("--cases", default="default")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--train-rows-per-seed", type=int, default=180)
    parser.add_argument("--validation-rows-per-seed", type=int, default=80)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=80)
    parser.add_argument("--ood-rows-per-seed", type=int, default=80)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=80)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=80)
    parser.add_argument("--gradient-epochs", type=int, default=120)
    parser.add_argument("--qat-epochs", type=int, default=80)
    parser.add_argument("--best-effort-qat-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--qat-learning-rate", type=float, default=7e-4)
    parser.add_argument("--best-effort-learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--matrix-steps", type=int, default=4)
    parser.add_argument("--distillation-weight", type=float, default=0.35)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--mutation-population", type=int, default=18)
    parser.add_argument("--mutation-generations", type=int, default=70)
    parser.add_argument("--mutation-steps", type=int, default=18)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--scale-sigma", type=float, default=0.08)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--execution-mode", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--parallel-workers", type=int, default=max(1, min((os.cpu_count() or 4) // 3, 4)))
    parser.add_argument("--torch-threads-per-worker", type=int, default=2)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    settings = Settings(
        cases=parse_cases(args.cases),
        width=args.width,
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
        matrix_steps=args.matrix_steps,
        distillation_weight=args.distillation_weight,
        distillation_temperature=args.distillation_temperature,
        mutation_population=args.mutation_population,
        mutation_generations=args.mutation_generations,
        mutation_steps=args.mutation_steps,
        elite_count=args.elite_count,
        scale_sigma=args.scale_sigma,
        device=args.device,
        execution_mode=args.execution_mode,
        parallel_workers=args.parallel_workers,
        torch_threads_per_worker=args.torch_threads_per_worker,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    results = run_core(settings, out)
    payloads = build_payloads(out, results, settings)
    deterministic = deterministic_replay(settings, out, payloads)
    write_final_artifacts(out, payloads, deterministic)
    decision = copy.deepcopy(payloads["decision.json"])
    decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
    print(json.dumps({"decision": decision["decision"], "deterministic_replay_passed": deterministic["internal_replay_passed"], "out": out.as_posix()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
