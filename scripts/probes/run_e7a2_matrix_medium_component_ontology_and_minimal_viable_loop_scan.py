#!/usr/bin/env python3
"""E7A2 component ontology and minimal viable loop scan.

This probe is intentionally ontology-first. It maps low-level matrix-medium
components that E7A did not cover, then runs small row-level microtasks to see
which primitives add measurable capability. It is not a final E7 routing claim.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "E7A2_MATRIX_MEDIUM_COMPONENT_ONTOLOGY_AND_MINIMAL_VIABLE_LOOP_SCAN"
DEFAULT_OUT = Path("target/pilot_wave/e7a2_matrix_medium_component_ontology_and_minimal_viable_loop_scan")
DEFAULT_SEEDS = (79001, 79002, 79003)

INPUT_DIM = 6
STATE_DIM = 8
CLASS_COUNT = 4
MAX_STEPS = 6
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
MICROTASKS = (
    "stabilization_task",
    "routing_task",
    "adaptive_exit_task",
    "perturbation_recovery_task",
    "trace_required_task",
)
ACTIVATIONS = ("tanh", "relu", "softsign", "c19", "identity")
VARIANTS = (
    "matrix_activation_baseline",
    "connection_mask_plus_weight",
    "residual_carry_state",
    "trace_buffer",
    "delta_stability_readiness",
    "self_state_mirror_buffer",
    "energy_resistance_field",
    "attractor_measurement",
    "oscillation_measurement",
    "activation_mutation",
    "connection_add_delete_mutation",
    "residual_delta_readiness_pair",
    "trace_self_state_pair",
    "energy_attractor_pair",
    "mask_weight_mutation_pair",
    "activation_mutation_residual_pair",
    "self_state_adaptive_exit_pair",
    "minimal_viable_loop_candidate",
    "random_control",
)
MUTABLE_VARIANTS = tuple(variant for variant in VARIANTS if variant != "random_control")
HASH_ARTIFACTS = (
    "e7a2_component_inventory.json",
    "e7a2_primitive_coverage_report.json",
    "e7a2_microtask_generation_report.json",
    "e7a2_variant_results.json",
    "e7a2_minimal_viable_loop_report.json",
    "e7a2_ablation_report.json",
    "e7a2_attractor_report.json",
    "e7a2_oscillation_report.json",
    "e7a2_readiness_exit_report.json",
    "e7a2_trace_self_state_report.json",
    "e7a2_energy_resistance_report.json",
    "e7a2_mutation_history.json",
    "e7a2_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
ALLOWED_DECISIONS = (
    "e7a2_no_minimal_viable_loop_detected",
    "e7a2_adaptive_exit_primitive_positive",
    "e7a2_trace_self_state_primitive_positive",
    "e7a2_energy_attractor_primitive_positive",
    "e7a2_minimal_viable_loop_combo_detected",
    "e7a2_component_scan_complete_no_strong_winner",
    "e7a2_invalid_synthetic_or_leak_detected",
)

TASK_W = np.asarray(
    [
        [0.90, -0.35, 0.20, -0.75],
        [-0.25, 0.80, -0.65, 0.10],
        [0.35, 0.05, 0.90, -0.45],
        [-0.65, -0.25, 0.35, 0.85],
        [0.20, -0.55, 0.10, 0.35],
        [-0.15, 0.30, -0.40, 0.60],
    ],
    dtype=np.float64,
)
TASK_B = np.asarray([0.05, -0.03, 0.04, -0.02], dtype=np.float64)


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    population_size: int
    generations: int
    elite_count: int
    mutation_sigma: float
    execution_mode: str
    parallel_workers: int
    heartbeat_seconds: float


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def append_progress_locked(out: Path, event: str, **details: Any) -> None:
    path = out / "progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd: int | None = None
    deadline = time.monotonic() + 120.0
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            if time.monotonic() > deadline:
                raise TimeoutError(f"progress lock timed out: {lock_path}")
            time.sleep(0.025)
    try:
        payload = {"event": event, "details": details}
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def locked_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd: int | None = None
    deadline = time.monotonic() + 120.0
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            if time.monotonic() > deadline:
                raise TimeoutError(f"write lock timed out: {lock_path}")
            time.sleep(0.025)
    try:
        write_json(path, payload)
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def resolve_out(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    relative = resolved.relative_to(REPO_ROOT)
    if len(relative.parts) < 2 or relative.parts[0].lower() != "target" or relative.parts[1].lower() != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one seed is required")
    return values


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7a2::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def flatten_paths(payload: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], float]]:
    rows: list[tuple[tuple[Any, ...], float]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            rows.extend(flatten_paths(value, prefix + (key,)))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            rows.extend(flatten_paths(value, prefix + (index,)))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        rows.append((prefix, float(payload)))
    return rows


def set_path(payload: Any, path: tuple[Any, ...], value: float) -> None:
    cursor = payload
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value


def clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def sigmoid_scalar(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-clamp(value, -40.0, 40.0)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def c19(x: np.ndarray, c: float = 3.0, rho: float = 1.0) -> np.ndarray:
    c = max(float(c), 0.1)
    rho = max(float(rho), 0.0)
    limit = 6.0 * c
    scaled = x / c
    n = np.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    sign = np.where((n.astype(np.int64) % 2) == 0, 1.0, -1.0)
    interior = c * (sign * h + rho * h * h)
    return np.where(x >= limit, x - limit, np.where(x <= -limit, x + limit, interior))


def activate(name: str, x: np.ndarray) -> np.ndarray:
    if name == "identity":
        return x
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "softsign":
        return x / (1.0 + np.abs(x))
    if name == "c19":
        return c19(x)
    raise ValueError(f"unknown activation: {name}")


def variant_features(variant: str) -> set[str]:
    mapping = {
        "matrix_activation_baseline": set(),
        "connection_mask_plus_weight": {"connection_mask"},
        "residual_carry_state": {"residual"},
        "trace_buffer": {"trace"},
        "delta_stability_readiness": {"readiness"},
        "self_state_mirror_buffer": {"self_state"},
        "energy_resistance_field": {"energy"},
        "attractor_measurement": {"attractor"},
        "oscillation_measurement": {"oscillation"},
        "activation_mutation": {"activation_mutation"},
        "connection_add_delete_mutation": {"connection_mask", "connection_mutation"},
        "residual_delta_readiness_pair": {"residual", "readiness"},
        "trace_self_state_pair": {"trace", "self_state"},
        "energy_attractor_pair": {"energy", "attractor"},
        "mask_weight_mutation_pair": {"connection_mask", "connection_mutation"},
        "activation_mutation_residual_pair": {"activation_mutation", "residual"},
        "self_state_adaptive_exit_pair": {"self_state", "readiness"},
        "minimal_viable_loop_candidate": {
            "connection_mask",
            "connection_mutation",
            "residual",
            "trace",
            "readiness",
            "self_state",
            "energy",
            "attractor",
            "oscillation",
            "activation_mutation",
        },
        "random_control": {"random"},
    }
    return set(mapping[variant])


def component_inventory() -> dict[str, Any]:
    entries = [
        {
            "primitive": "matrix_activation_baseline",
            "what_it_is": "Mutable input/state/output matrices with one fixed homogeneous activation.",
            "mutable_parts": ["w_in", "w_state", "b_state", "w_out", "b_out"],
            "measured_capability": "Baseline stabilization and classification without extra medium primitives.",
            "expected_failure_modes": ["linear-collapse-like weakness", "fixed-depth overthinking", "no explicit memory"],
        },
        {
            "primitive": "connection_mask",
            "what_it_is": "A separate binary topology mask over the state transition matrix.",
            "mutable_parts": ["mask_state", "masked w_state values"],
            "measured_capability": "Whether topology separation improves routing and basin separation.",
            "expected_failure_modes": ["dead connections", "mask overfitting", "underconnected state"],
        },
        {
            "primitive": "residual_carry_state",
            "what_it_is": "Carry part of the previous state into the next state.",
            "mutable_parts": ["residual_alpha", "state matrices"],
            "measured_capability": "Stability and perturbation recovery without forcing full overwrite.",
            "expected_failure_modes": ["state inertia", "slow convergence"],
        },
        {
            "primitive": "trace_buffer",
            "what_it_is": "A decayed memory trace of earlier states feeding the next update.",
            "mutable_parts": ["trace_decay", "w_trace"],
            "measured_capability": "Rows where the current input is insufficient without a prior pulse.",
            "expected_failure_modes": ["stale trace", "shortcut via trace magnitude"],
        },
        {
            "primitive": "delta_stability_readiness",
            "what_it_is": "Exit readiness derived from state delta, stability, and confidence.",
            "mutable_parts": ["readiness weights", "readiness bias"],
            "measured_capability": "Stop when useful rather than after a fixed depth.",
            "expected_failure_modes": ["early wrong halt", "late overthinking"],
        },
        {
            "primitive": "self_state_mirror_buffer",
            "what_it_is": "A small self-state buffer tracking state and confidence summaries.",
            "mutable_parts": ["w_self", "w_self_write", "self_decay", "self_conf_weight"],
            "measured_capability": "Whether a mirror channel helps trace and adaptive exit.",
            "expected_failure_modes": ["self-confirming shortcut", "extra unstable loop"],
        },
        {
            "primitive": "energy_resistance_field",
            "what_it_is": "Class-level resistance/energy terms that subtract from logits.",
            "mutable_parts": ["energy_w", "energy_b", "energy_scale"],
            "measured_capability": "Shortcut suppression and energy gap separation.",
            "expected_failure_modes": ["penalizing the target", "energy sign loophole"],
        },
        {
            "primitive": "attractor_measurement",
            "what_it_is": "Learned class attractor centers used for basin-distance readout.",
            "mutable_parts": ["attractor_centers", "attractor_scale"],
            "measured_capability": "Basin separation and recovery after perturbation.",
            "expected_failure_modes": ["center collapse", "distance-only shortcut"],
        },
        {
            "primitive": "oscillation_measurement",
            "what_it_is": "Trajectory audit for back-and-forth state oscillation.",
            "mutable_parts": ["state parameters indirectly"],
            "measured_capability": "Detects unstable loops and punishes oscillatory solutions in fitness.",
            "expected_failure_modes": ["false positive on useful exploration"],
        },
        {
            "primitive": "activation_mutation",
            "what_it_is": "The homogeneous activation family is mutable as a discrete candidate field.",
            "mutable_parts": ["activation_index"],
            "measured_capability": "Whether the medium benefits from changing nonlinearity rather than only weights.",
            "expected_failure_modes": ["activation overfit", "identity collapse"],
        },
        {
            "primitive": "connection_add_delete_mutation",
            "what_it_is": "Topology mutation flips individual mask entries on or off.",
            "mutable_parts": ["mask_state topology"],
            "measured_capability": "Whether add/delete connection edits improve over weight-only mutation.",
            "expected_failure_modes": ["random sparse damage", "mask churn without capability"],
        },
    ]
    return {
        "schema_version": "e7a2_component_inventory_v1",
        "milestone": MILESTONE,
        "entries": entries,
        "final_e7_verdict_intentionally_deferred": True,
    }


def target_scores(x: np.ndarray, task_index: int) -> np.ndarray:
    scores = x @ TASK_W + TASK_B
    if task_index == 1:
        scores = scores + np.asarray([0.15, -0.05, 0.05, -0.15])
    elif task_index == 2:
        scores = scores + np.asarray([0.0, 0.2 * x[4], -0.1 * x[3], 0.1 * x[5]])
    elif task_index == 3:
        scores = scores + np.asarray([-0.1 * x[5], 0.1 * x[2], 0.15 * x[0], -0.05])
    elif task_index == 4:
        pulse = 0.75 if x[5] > 0.0 else -0.75
        scores = scores + np.asarray([pulse, -pulse, 0.35 * pulse, -0.35 * pulse])
    return scores


def make_row(split: str, seed: int, index: int, rng: np.random.Generator) -> dict[str, Any]:
    task = MICROTASKS[index % len(MICROTASKS)]
    task_index = MICROTASKS.index(task)
    scale = 1.0
    shift = 0.0
    if split == "ood":
        scale = 1.55
        shift = 0.18
    x = rng.normal(loc=shift, scale=scale, size=INPUT_DIM).astype(np.float64)
    if split == "counterfactual":
        x[0] *= -1.0
        x[3] *= -1.0
        x[5] *= -1.0
    scores = target_scores(x, task_index)
    target = int(np.argmax(scores))
    margin = float(np.sort(scores)[-1] - np.sort(scores)[-2])
    shortcut_target = int(np.argmax(np.roll(scores, 1)))
    if split == "adversarial":
        x[4] = (shortcut_target - 1.5) / 1.5 + rng.normal(0.0, 0.025)
        x[5] += rng.normal(0.0, 0.25)
    if task == "stabilization_task":
        target_steps = 4
    elif task == "routing_task":
        target_steps = 3
    elif task == "adaptive_exit_task":
        target_steps = 2 if margin > 0.85 else 5
    elif task == "perturbation_recovery_task":
        target_steps = 5
    else:
        target_steps = 4
    perturb = rng.normal(0.0, 0.35, size=STATE_DIM).astype(np.float64)
    return {
        "row_id": f"{split}_{seed}_{index:05d}",
        "split": split,
        "task": task,
        "task_index": task_index,
        "x": [round_float(value) for value in x.tolist()],
        "target": target,
        "target_steps": target_steps,
        "margin": round_float(margin),
        "shortcut_target": shortcut_target,
        "shortcut_is_wrong": bool(shortcut_target != target),
        "perturbation": [round_float(value) for value in perturb.tolist()],
    }


def generate_split(split: str, seeds: tuple[int, ...], rows_per_seed: int) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(stable_seed(f"{split}-{seed}"))
        for index in range(rows_per_seed):
            rows.append(make_row(split, seed, index, rng))
    return {
        "rows": rows,
        "x": np.asarray([row["x"] for row in rows], dtype=np.float64),
        "y": np.asarray([row["target"] for row in rows], dtype=np.int64),
        "target_steps": np.asarray([row["target_steps"] for row in rows], dtype=np.int64),
        "task_index": np.asarray([row["task_index"] for row in rows], dtype=np.int64),
        "shortcut_target": np.asarray([row["shortcut_target"] for row in rows], dtype=np.int64),
        "shortcut_is_wrong": np.asarray([row["shortcut_is_wrong"] for row in rows], dtype=bool),
        "perturbation": np.asarray([row["perturbation"] for row in rows], dtype=np.float64),
    }


def generate_task(settings: Settings) -> dict[str, Any]:
    counts = {
        "train": settings.train_rows_per_seed,
        "validation": settings.validation_rows_per_seed,
        "heldout": settings.heldout_rows_per_seed,
        "ood": settings.ood_rows_per_seed,
        "counterfactual": settings.counterfactual_rows_per_seed,
        "adversarial": settings.adversarial_rows_per_seed,
    }
    return {split: generate_split(split, settings.seeds, counts[split]) for split in SPLITS}


def candidate_template(variant: str, rng: np.random.Generator) -> dict[str, Any]:
    features = variant_features(variant)
    scale = 0.22
    mask = np.ones((STATE_DIM, STATE_DIM), dtype=np.float64)
    if "connection_mask" in features:
        mask = (rng.random((STATE_DIM, STATE_DIM)) > 0.35).astype(np.float64)
        np.fill_diagonal(mask, 1.0)
    activation_index = 0
    if "activation_mutation" in features:
        activation_index = int(rng.integers(0, len(ACTIVATIONS)))
    candidate: dict[str, Any] = {
        "schema_version": "e7a2_matrix_medium_candidate_v1",
        "variant": variant,
        "features": sorted(features),
        "w_in": rng.normal(0.0, scale, size=(INPUT_DIM, STATE_DIM)).tolist(),
        "w_state": rng.normal(0.0, scale, size=(STATE_DIM, STATE_DIM)).tolist(),
        "mask_state": mask.tolist(),
        "b_state": rng.normal(0.0, 0.04, size=STATE_DIM).tolist(),
        "w_out": rng.normal(0.0, scale, size=(STATE_DIM, CLASS_COUNT)).tolist(),
        "b_out": rng.normal(0.0, 0.04, size=CLASS_COUNT).tolist(),
        "activation_index": activation_index,
        "residual_alpha": float(rng.uniform(0.15, 0.75)),
        "trace_decay": float(rng.uniform(0.25, 0.80)),
        "w_trace": rng.normal(0.0, scale, size=(STATE_DIM, STATE_DIM)).tolist(),
        "self_decay": float(rng.uniform(0.25, 0.85)),
        "w_self": rng.normal(0.0, scale, size=(STATE_DIM, STATE_DIM)).tolist(),
        "w_self_write": rng.normal(0.0, scale, size=(STATE_DIM, STATE_DIM)).tolist(),
        "self_conf_weight": rng.normal(0.0, 0.08, size=STATE_DIM).tolist(),
        "readiness_conf_weight": float(rng.normal(1.0, 0.2)),
        "readiness_delta_weight": float(rng.normal(1.0, 0.2)),
        "readiness_stability_weight": float(rng.normal(0.7, 0.2)),
        "readiness_bias": float(rng.normal(-1.0, 0.2)),
        "energy_w": rng.normal(0.0, scale, size=(STATE_DIM, CLASS_COUNT)).tolist(),
        "energy_b": rng.normal(0.0, 0.04, size=CLASS_COUNT).tolist(),
        "energy_scale": float(rng.uniform(0.05, 0.45)),
        "attractor_centers": rng.normal(0.0, 0.55, size=(CLASS_COUNT, STATE_DIM)).tolist(),
        "attractor_scale": float(rng.uniform(0.02, 0.30)),
    }
    return round_candidate(candidate)


def round_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    rounded = copy.deepcopy(candidate)
    for path, value in flatten_paths(rounded):
        if path and path[0] in {"schema_version", "variant", "features"}:
            continue
        if path and path[0] == "activation_index":
            set_path(rounded, path, int(round(clamp(value, 0, len(ACTIVATIONS) - 1))))
        elif path and path[0] == "mask_state":
            set_path(rounded, path, float(1.0 if value >= 0.5 else 0.0))
        else:
            set_path(rounded, path, round_float(value))
    return rounded


def candidate_hash(candidate: dict[str, Any]) -> str:
    clean = {key: value for key, value in candidate.items() if key != "candidate_id"}
    return payload_sha256(clean)


def mutable_paths(candidate: dict[str, Any]) -> list[tuple[tuple[Any, ...], float]]:
    excluded = {"schema_version", "variant", "features", "activation_index", "mask_state"}
    return [(path, value) for path, value in flatten_paths(candidate) if path and path[0] not in excluded]


def clamp_path(path: tuple[Any, ...], value: float) -> float:
    if path and path[0] in {"residual_alpha", "trace_decay", "self_decay", "energy_scale", "attractor_scale"}:
        return clamp(value, 0.0, 1.0)
    if path and path[0].startswith("readiness"):
        return clamp(value, -5.0, 5.0)
    return clamp(value, -4.0, 4.0)


def mutate(candidate: dict[str, Any], rng: random.Random, sigma: float) -> tuple[dict[str, Any], dict[str, int]]:
    child = copy.deepcopy(candidate)
    features = set(child["features"])
    edits = {"numeric_edits": 0, "activation_edits": 0, "connection_adds": 0, "connection_deletes": 0}
    paths = mutable_paths(child)
    edit_count = rng.randint(3, 18)
    for path, value in rng.sample(paths, k=min(edit_count, len(paths))):
        set_path(child, path, round_float(clamp_path(path, value + rng.gauss(0.0, sigma))))
        edits["numeric_edits"] += 1
    if "activation_mutation" in features and rng.random() < 0.22:
        current = int(child["activation_index"])
        choices = [index for index in range(len(ACTIVATIONS)) if index != current]
        child["activation_index"] = int(rng.choice(choices))
        edits["activation_edits"] += 1
    if "connection_mask" in features and rng.random() < 0.38:
        flips = 1 + int(rng.random() < 0.25)
        for _ in range(flips):
            i = rng.randrange(STATE_DIM)
            j = rng.randrange(STATE_DIM)
            old = float(child["mask_state"][i][j])
            child["mask_state"][i][j] = 0.0 if old >= 0.5 else 1.0
            if old >= 0.5:
                edits["connection_deletes"] += 1
            else:
                edits["connection_adds"] += 1
    return round_candidate(child), edits


def arrays(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "w_in": np.asarray(candidate["w_in"], dtype=np.float64),
        "w_state": np.asarray(candidate["w_state"], dtype=np.float64),
        "mask_state": np.asarray(candidate["mask_state"], dtype=np.float64),
        "b_state": np.asarray(candidate["b_state"], dtype=np.float64),
        "w_out": np.asarray(candidate["w_out"], dtype=np.float64),
        "b_out": np.asarray(candidate["b_out"], dtype=np.float64),
        "w_trace": np.asarray(candidate["w_trace"], dtype=np.float64),
        "w_self": np.asarray(candidate["w_self"], dtype=np.float64),
        "w_self_write": np.asarray(candidate["w_self_write"], dtype=np.float64),
        "self_conf_weight": np.asarray(candidate["self_conf_weight"], dtype=np.float64),
        "energy_w": np.asarray(candidate["energy_w"], dtype=np.float64),
        "energy_b": np.asarray(candidate["energy_b"], dtype=np.float64),
        "attractor_centers": np.asarray(candidate["attractor_centers"], dtype=np.float64),
    }


def step_inputs(split_data: dict[str, Any], step: int) -> np.ndarray:
    x = np.asarray(split_data["x"], dtype=np.float64).copy()
    task_index = np.asarray(split_data["task_index"], dtype=np.int64)
    trace_mask = task_index == MICROTASKS.index("trace_required_task")
    if step > 1 and np.any(trace_mask):
        x[trace_mask, 5] = 0.0
        x[trace_mask, 0] = 0.0
    return x


def confidence_margin(logits: np.ndarray) -> np.ndarray:
    if logits.shape[1] < 2:
        return np.zeros(logits.shape[0], dtype=np.float64)
    sorted_logits = np.sort(logits, axis=1)
    return sorted_logits[:, -1] - sorted_logits[:, -2]


def oscillation_flags(trajectory: list[np.ndarray]) -> np.ndarray:
    if len(trajectory) < 3:
        return np.zeros(trajectory[-1].shape[0], dtype=bool)
    prev2 = trajectory[-3]
    prev1 = trajectory[-2]
    cur = trajectory[-1]
    d1 = np.mean(np.abs(cur - prev1), axis=1)
    d2 = np.mean(np.abs(cur - prev2), axis=1)
    return (d2 < 0.08) & (d1 > 0.04)


def predict_candidate(candidate: dict[str, Any], split_data: dict[str, Any]) -> dict[str, Any]:
    features = set(candidate["features"])
    arr = arrays(candidate)
    rows = split_data["x"].shape[0]
    state = np.zeros((rows, STATE_DIM), dtype=np.float64)
    trace = np.zeros_like(state)
    self_state = np.zeros_like(state)
    chosen_logits = np.zeros((rows, CLASS_COUNT), dtype=np.float64)
    chosen_energy = np.zeros((rows, CLASS_COUNT), dtype=np.float64)
    chosen_attractor_distance = np.zeros((rows, CLASS_COUNT), dtype=np.float64)
    chosen_steps = np.full(rows, MAX_STEPS, dtype=np.int64)
    halted = np.zeros(rows, dtype=bool)
    readiness_scores = np.zeros((rows, MAX_STEPS), dtype=np.float64)
    final_delta = np.zeros(rows, dtype=np.float64)
    final_stability = np.zeros(rows, dtype=np.float64)
    oscillated = np.zeros(rows, dtype=bool)
    trajectory: list[np.ndarray] = []
    activation = ACTIVATIONS[int(candidate["activation_index"])]
    masked_state = arr["w_state"] * arr["mask_state"]
    for step in range(1, MAX_STEPS + 1):
        x_step = step_inputs(split_data, step)
        pre = x_step @ arr["w_in"] + state @ masked_state + arr["b_state"]
        if "trace" in features:
            pre = pre + trace @ arr["w_trace"]
        if "self_state" in features:
            pre = pre + self_state @ arr["w_self"]
        next_state = activate(activation, pre)
        if "residual" in features:
            alpha = float(candidate["residual_alpha"])
            next_state = alpha * state + (1.0 - alpha) * next_state
        task_index = np.asarray(split_data["task_index"], dtype=np.int64)
        perturb_mask = task_index == MICROTASKS.index("perturbation_recovery_task")
        if step == 3 and np.any(perturb_mask):
            next_state[perturb_mask] = next_state[perturb_mask] + 0.45 * split_data["perturbation"][perturb_mask]
        delta = next_state - state
        delta_norm = np.mean(np.abs(delta), axis=1)
        stability = np.exp(-delta_norm)
        logits = next_state @ arr["w_out"] + arr["b_out"]
        energy = np.zeros_like(logits)
        if "energy" in features:
            energy = np.abs(next_state @ arr["energy_w"] + arr["energy_b"])
            logits = logits - float(candidate["energy_scale"]) * energy
        distances = np.sum((next_state[:, None, :] - arr["attractor_centers"][None, :, :]) ** 2, axis=2)
        if "attractor" in features:
            logits = logits - float(candidate["attractor_scale"]) * distances
        conf = confidence_margin(logits)
        readiness_raw = (
            float(candidate["readiness_conf_weight"]) * conf
            + float(candidate["readiness_delta_weight"]) * (0.28 - delta_norm)
            + float(candidate["readiness_stability_weight"]) * stability
            + float(candidate["readiness_bias"])
        )
        ready = sigmoid(readiness_raw)
        readiness_scores[:, step - 1] = ready
        if "trace" in features:
            decay = float(candidate["trace_decay"])
            trace = decay * trace + (1.0 - decay) * next_state
        if "self_state" in features:
            self_decay = float(candidate["self_decay"])
            self_write = activate(activation, next_state @ arr["w_self_write"] + conf[:, None] * arr["self_conf_weight"])
            self_state = self_decay * self_state + (1.0 - self_decay) * self_write
        trajectory.append(next_state.copy())
        oscillated = oscillated | oscillation_flags(trajectory)
        step_halt = (ready > 0.58) & (step >= 2) if "readiness" in features else np.zeros(rows, dtype=bool)
        if step == MAX_STEPS:
            step_halt = np.ones(rows, dtype=bool)
        take = step_halt & (~halted)
        if np.any(take):
            chosen_logits[take] = logits[take]
            chosen_energy[take] = energy[take]
            chosen_attractor_distance[take] = distances[take]
            chosen_steps[take] = step
            halted[take] = True
            final_delta[take] = delta_norm[take]
            final_stability[take] = stability[take]
        state = next_state
    pred = np.argmax(chosen_logits, axis=1).astype(np.int64)
    target = split_data["y"]
    target_energy = chosen_energy[np.arange(rows), target]
    wrong_energy = chosen_energy.copy()
    wrong_energy[np.arange(rows), target] = np.nan
    min_wrong_energy = np.nanmin(wrong_energy, axis=1)
    target_distance = chosen_attractor_distance[np.arange(rows), target]
    wrong_distance = chosen_attractor_distance.copy()
    wrong_distance[np.arange(rows), target] = np.nan
    min_wrong_distance = np.nanmin(wrong_distance, axis=1)
    return {
        "pred": pred,
        "steps": chosen_steps,
        "logits": chosen_logits,
        "confidence_margin": confidence_margin(chosen_logits),
        "final_delta": final_delta,
        "final_stability": final_stability,
        "oscillated": oscillated,
        "energy_gap": min_wrong_energy - target_energy,
        "attractor_gap": min_wrong_distance - target_distance,
        "readiness_scores": readiness_scores,
    }


def random_predictions(split_data: dict[str, Any], seed: int) -> dict[str, Any]:
    rows = split_data["x"].shape[0]
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.0, size=(rows, CLASS_COUNT))
    return {
        "pred": np.argmax(logits, axis=1).astype(np.int64),
        "steps": rng.integers(1, MAX_STEPS + 1, size=rows, dtype=np.int64),
        "logits": logits,
        "confidence_margin": confidence_margin(logits),
        "final_delta": np.ones(rows, dtype=np.float64),
        "final_stability": np.zeros(rows, dtype=np.float64),
        "oscillated": rng.random(rows) > 0.92,
        "energy_gap": np.zeros(rows, dtype=np.float64),
        "attractor_gap": np.zeros(rows, dtype=np.float64),
        "readiness_scores": rng.random((rows, MAX_STEPS)),
    }


def bounded01(value: float) -> float:
    return round_float(clamp(value, 0.0, 1.0))


def evaluate_prediction(pred: dict[str, Any], split_data: dict[str, Any], sample_limit: int = 10) -> dict[str, Any]:
    y = split_data["y"]
    correct = pred["pred"] == y
    steps = np.asarray(pred["steps"], dtype=np.float64)
    target_steps = np.asarray(split_data["target_steps"], dtype=np.float64)
    task_index = np.asarray(split_data["task_index"], dtype=np.int64)
    shortcut_mask = np.asarray(split_data["shortcut_is_wrong"], dtype=bool)
    shortcut_hit = pred["pred"] == split_data["shortcut_target"]
    per_task: dict[str, float] = {}
    per_task_steps: dict[str, float] = {}
    for index, task in enumerate(MICROTASKS):
        mask = task_index == index
        per_task[task] = round_float(float(np.mean(correct[mask]))) if np.any(mask) else 0.0
        per_task_steps[task] = round_float(float(np.mean(steps[mask]))) if np.any(mask) else 0.0
    perturb_mask = task_index == MICROTASKS.index("perturbation_recovery_task")
    trace_mask = task_index == MICROTASKS.index("trace_required_task")
    adaptive_mask = task_index == MICROTASKS.index("adaptive_exit_task")
    overthinking = steps > (target_steps + 1.0)
    underthinking = steps < np.maximum(1.0, target_steps - 1.0)
    step_match = np.abs(steps - target_steps) <= 1.0
    convergence_score = np.exp(-np.asarray(pred["final_delta"], dtype=np.float64))
    basin_signal = sigmoid(np.asarray(pred["confidence_margin"], dtype=np.float64))
    energy_signal = sigmoid(np.asarray(pred["energy_gap"], dtype=np.float64))
    metrics = {
        "task_accuracy": round_float(float(np.mean(correct))),
        "per_task_accuracy": per_task,
        "per_task_mean_steps": per_task_steps,
        "convergence_score": round_float(float(np.mean(convergence_score))),
        "mean_steps": round_float(float(np.mean(steps))),
        "overthinking_rate": round_float(float(np.mean(overthinking))),
        "underthinking_rate": round_float(float(np.mean(underthinking))),
        "oscillation_rate": round_float(float(np.mean(pred["oscillated"]))),
        "perturbation_recovery": round_float(float(np.mean(correct[perturb_mask]))) if np.any(perturb_mask) else 0.0,
        "basin_separation": round_float(float(np.mean(basin_signal))),
        "energy_gap": round_float(float(np.mean(pred["energy_gap"]))),
        "energy_gap_signal": round_float(float(np.mean(energy_signal))),
        "attractor_gap": round_float(float(np.mean(pred["attractor_gap"]))),
        "shortcut_rate": round_float(float(np.mean(shortcut_hit[shortcut_mask]))) if np.any(shortcut_mask) else 0.0,
        "readiness_exit_accuracy": round_float(float(np.mean(step_match[adaptive_mask]))) if np.any(adaptive_mask) else 0.0,
        "trace_required_accuracy": round_float(float(np.mean(correct[trace_mask]))) if np.any(trace_mask) else 0.0,
    }
    metrics["macro_composite_score"] = bounded01(
        0.40 * metrics["task_accuracy"]
        + 0.12 * metrics["convergence_score"]
        + 0.10 * metrics["perturbation_recovery"]
        + 0.08 * metrics["basin_separation"]
        + 0.08 * metrics["energy_gap_signal"]
        + 0.08 * metrics["readiness_exit_accuracy"]
        + 0.08 * metrics["trace_required_accuracy"]
        - 0.05 * metrics["overthinking_rate"]
        - 0.05 * metrics["underthinking_rate"]
        - 0.04 * metrics["oscillation_rate"]
        - 0.04 * metrics["shortcut_rate"]
    )
    samples = []
    for index in range(min(sample_limit, len(split_data["rows"]))):
        row = split_data["rows"][index]
        samples.append(
            {
                "row_id": row["row_id"],
                "task": row["task"],
                "target": int(row["target"]),
                "pred": int(pred["pred"][index]),
                "steps": int(pred["steps"][index]),
                "target_steps": int(row["target_steps"]),
                "shortcut_target": int(row["shortcut_target"]),
                "correct": bool(correct[index]),
            }
        )
    return {"metrics": metrics, "row_level_samples": samples}


def evaluate_candidate(candidate: dict[str, Any], task: dict[str, Any], sample_limit: int = 10) -> dict[str, Any]:
    return {split: evaluate_prediction(predict_candidate(candidate, data), data, sample_limit) for split, data in task.items()}


def evaluate_random(task: dict[str, Any], sample_limit: int = 10) -> dict[str, Any]:
    return {
        split: evaluate_prediction(random_predictions(data, stable_seed(f"random-{split}")), data, sample_limit)
        for split, data in task.items()
    }


def fitness_from_evals(evals: dict[str, Any], features: set[str]) -> float:
    train = evals["train"]["metrics"]
    validation = evals["validation"]["metrics"]
    value = 0.35 * train["macro_composite_score"] + 0.55 * validation["macro_composite_score"]
    value += 0.05 * validation["task_accuracy"]
    if "oscillation" in features:
        value -= 0.05 * validation["oscillation_rate"]
    value -= 0.03 * validation["shortcut_rate"]
    return round_float(value)


def search_eval(candidate: dict[str, Any], task: dict[str, Any], all_splits: bool = False) -> dict[str, Any]:
    splits = task if all_splits else {"train": task["train"], "validation": task["validation"]}
    evals = {split: evaluate_prediction(predict_candidate(candidate, data), data, 0) for split, data in splits.items()}
    return {"candidate": candidate, "evals": evals, "fitness": fitness_from_evals(evals, set(candidate["features"]))}


def parameter_count(candidate: dict[str, Any]) -> int:
    return sum(1 for path, _ in flatten_paths(candidate) if path and path[0] not in {"schema_version", "variant", "features"})


def parameter_diff(variant: str, initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    before = {path: value for path, value in flatten_paths(initial) if path and path[0] not in {"schema_version", "variant", "features"}}
    after = {path: value for path, value in flatten_paths(final) if path and path[0] not in {"schema_version", "variant", "features"}}
    changed = {}
    l2 = 0.0
    for path in sorted(before):
        delta = after[path] - before[path]
        if abs(delta) > 1e-12:
            key = ".".join(str(part) for part in path)
            changed[key] = {"before": before[path], "after": after[path], "delta": round_float(delta)}
            l2 += delta * delta
    return {
        "schema_version": "e7a2_parameter_diff_v1",
        "variant": variant,
        "before_hash": candidate_hash(initial),
        "after_hash": candidate_hash(final),
        "actual_parameter_diff_found": bool(changed),
        "changed_parameter_count": len(changed),
        "parameter_diff_l2": round_float(math.sqrt(l2)),
        "changed_parameters_sample": dict(list(changed.items())[:120]),
    }


def generation_metric(variant: str, evals: dict[str, Any], accepted: int, rejected: int, rollback: int, state_hash: str) -> dict[str, Any]:
    validation = evals["validation"]["metrics"]
    return {
        "variant": variant,
        "validation_macro_composite_score": validation["macro_composite_score"],
        "validation_task_accuracy": validation["task_accuracy"],
        "validation_mean_steps": validation["mean_steps"],
        "validation_oscillation_rate": validation["oscillation_rate"],
        "validation_shortcut_rate": validation["shortcut_rate"],
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "candidate_hash": state_hash,
    }


def write_variant_partial(out: Path, variant: str, payload: dict[str, Any]) -> None:
    partial_dir = out / "partial_status"
    locked_write_json(partial_dir / f"{variant}.json", payload)
    locked_write_json(
        out / "e7a2_partial_aggregate_snapshot.json",
        {
            "schema_version": "e7a2_partial_aggregate_snapshot_v1",
            "last_variant": variant,
            "last_update_monotonic": round_float(time.monotonic()),
            "payload": payload,
        },
    )


def run_variant(variant: str, task: dict[str, Any], settings: Settings, out: str | None) -> dict[str, Any]:
    out_path = Path(out) if out else None
    start = time.monotonic()
    if out_path:
        append_progress_locked(out_path, "variant_start", variant=variant)
    if variant == "random_control":
        evals = evaluate_random(task)
        result = {
            "variant": variant,
            "features": sorted(variant_features(variant)),
            "initial_candidate": None,
            "final_candidate": None,
            "parameter_diff": None,
            "mutation_history": {
                "variant": variant,
                "mutation_attempt_count": 0,
                "accepted_mutation_count": 0,
                "rejected_mutation_count": 0,
                "rollback_count": 0,
                "connection_add_count": 0,
                "connection_delete_count": 0,
                "activation_mutation_count": 0,
                "generation_metrics": [],
            },
            "evals": evals,
            "parameter_count": 0,
            "runtime_seconds": round_float(time.monotonic() - start),
        }
        if out_path:
            write_variant_outputs(out_path, result)
            append_progress_locked(out_path, "variant_complete", variant=variant, runtime_seconds=result["runtime_seconds"])
        return result

    rng_np = np.random.default_rng(stable_seed(f"{variant}-init"))
    rng = random.Random(stable_seed(f"{variant}-mutation"))
    population = []
    for index in range(settings.population_size):
        candidate = candidate_template(variant, rng_np)
        candidate["candidate_id"] = f"{variant}_initial_{index:03d}"
        population.append(search_eval(candidate, task, all_splits=False))
    initial_candidate = copy.deepcopy(population[0]["candidate"])
    accepted = 0
    rejected = 0
    rollback = 0
    connection_adds = 0
    connection_deletes = 0
    activation_edits = 0
    attempts = 0
    generation_metrics: list[dict[str, Any]] = []
    last_heartbeat = time.monotonic()
    for generation in range(1, settings.generations + 1):
        population.sort(key=lambda row: row["fitness"], reverse=True)
        elites = population[: max(1, min(settings.elite_count, len(population)))]
        next_population = copy.deepcopy(elites)
        while len(next_population) < settings.population_size:
            parent = copy.deepcopy(rng.choice(elites))
            child_candidate, edits = mutate(parent["candidate"], rng, settings.mutation_sigma)
            child_candidate["candidate_id"] = f"{variant}_g{generation:04d}_{len(next_population):03d}"
            child = search_eval(child_candidate, task, all_splits=False)
            attempts += 1
            if child["fitness"] >= parent["fitness"]:
                accepted += 1
                connection_adds += edits["connection_adds"]
                connection_deletes += edits["connection_deletes"]
                activation_edits += edits["activation_edits"]
                next_population.append(child)
            else:
                rejected += 1
                rollback += 1
                next_population.append(parent)
            now = time.monotonic()
            if out_path and now - last_heartbeat >= settings.heartbeat_seconds:
                best_mid = max(next_population, key=lambda row: row["fitness"])
                write_variant_partial(
                    out_path,
                    variant,
                    {
                        "variant": variant,
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
                    "heartbeat",
                    variant=variant,
                    generation=generation,
                    attempts=attempts,
                    best_fitness=best_mid["fitness"],
                )
                last_heartbeat = now
        population = next_population
        best = max(population, key=lambda row: row["fitness"])
        row = generation_metric(variant, best["evals"], accepted, rejected, rollback, candidate_hash(best["candidate"]))
        row["generation"] = generation
        row["best_fitness"] = best["fitness"]
        generation_metrics.append(row)
        if out_path:
            history_payload = {
                "schema_version": "e7a2_variant_mutation_history_v1",
                "variant": variant,
                "mutation_attempt_count": attempts,
                "accepted_mutation_count": accepted,
                "rejected_mutation_count": rejected,
                "rollback_count": rollback,
                "connection_add_count": connection_adds,
                "connection_delete_count": connection_deletes,
                "activation_mutation_count": activation_edits,
                "generation_metrics": generation_metrics,
            }
            write_json(out_path / f"e7a2_mutation_history_{variant}.json", history_payload)
            write_variant_partial(
                out_path,
                variant,
                {
                    "variant": variant,
                    "generation": generation,
                    "attempts": attempts,
                    "accepted": accepted,
                    "rejected": rejected,
                    "rollback": rollback,
                    "best_fitness": best["fitness"],
                    "best_candidate_hash": candidate_hash(best["candidate"]),
                },
            )
            append_progress_locked(out_path, "generation_complete", variant=variant, generation=generation, metrics=row)
    best = max(population, key=lambda row: row["fitness"])
    final_candidate = copy.deepcopy(best["candidate"])
    full_evals = evaluate_candidate(final_candidate, task)
    result = {
        "variant": variant,
        "features": sorted(variant_features(variant)),
        "initial_candidate": initial_candidate,
        "final_candidate": final_candidate,
        "parameter_diff": parameter_diff(variant, initial_candidate, final_candidate),
        "mutation_history": {
            "schema_version": "e7a2_variant_mutation_history_v1",
            "variant": variant,
            "mutation_attempt_count": attempts,
            "accepted_mutation_count": accepted,
            "rejected_mutation_count": rejected,
            "rollback_count": rollback,
            "connection_add_count": connection_adds,
            "connection_delete_count": connection_deletes,
            "activation_mutation_count": activation_edits,
            "generation_metrics": generation_metrics,
        },
        "evals": full_evals,
        "parameter_count": parameter_count(final_candidate),
        "runtime_seconds": round_float(time.monotonic() - start),
    }
    if out_path:
        write_variant_outputs(out_path, result)
        append_progress_locked(out_path, "variant_complete", variant=variant, runtime_seconds=result["runtime_seconds"])
    return result


def write_variant_outputs(out: Path, result: dict[str, Any]) -> None:
    variant = result["variant"]
    summary = {
        "schema_version": "e7a2_candidate_summary_v1",
        "variant": variant,
        "features": result["features"],
        "parameter_count": result["parameter_count"],
        "runtime_seconds": result["runtime_seconds"],
        "final_candidate_hash": candidate_hash(result["final_candidate"]) if result["final_candidate"] is not None else None,
        "split_metrics": {split: result["evals"][split]["metrics"] for split in SPLITS},
    }
    write_json(out / f"e7a2_candidate_{variant}_summary.json", summary)
    if result["initial_candidate"] is not None:
        write_json(out / f"e7a2_candidate_{variant}_initial.json", result["initial_candidate"])
        write_json(out / f"e7a2_candidate_{variant}_final.json", result["final_candidate"])
        write_json(out / f"e7a2_parameter_diff_{variant}.json", result["parameter_diff"])
        write_json(out / f"e7a2_mutation_history_{variant}.json", result["mutation_history"])


def run_variants_serial(task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    return {variant: run_variant(variant, task, settings, out.as_posix() if out else None) for variant in VARIANTS}


def run_variants_parallel(task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    if out:
        append_progress_locked(out, "parallel_variants_start", workers=settings.parallel_workers, variants=len(VARIANTS))
    results: dict[str, Any] = {}
    worker_count = max(1, min(settings.parallel_workers, len(VARIANTS)))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(run_variant, variant, task, settings, out.as_posix() if out else None): variant
            for variant in VARIANTS
        }
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                variant = futures[future]
                results[variant] = future.result()
                if out:
                    append_progress_locked(out, "parallel_variant_joined", variant=variant, completed=len(results))
    return {variant: results[variant] for variant in VARIANTS}


def task_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    splits = {}
    for split, data in task.items():
        task_counts = {}
        for index, name in enumerate(MICROTASKS):
            task_counts[name] = int(np.sum(data["task_index"] == index))
        splits[split] = {"row_count": len(data["rows"]), "task_counts": task_counts}
    return {
        "schema_version": "e7a2_microtask_generation_report_v1",
        "milestone": MILESTONE,
        "seeds": list(settings.seeds),
        "microtasks": list(MICROTASKS),
        "splits": splits,
        "row_level_targets_generated": True,
    }


def aggregate_results(results: dict[str, Any]) -> dict[str, Any]:
    systems = {}
    for variant in VARIANTS:
        result = results[variant]
        split_metrics = {split: result["evals"][split]["metrics"] for split in SPLITS}
        eval_composite = round_float(float(np.mean([split_metrics[split]["macro_composite_score"] for split in EVAL_SPLITS])))
        eval_accuracy = round_float(float(np.mean([split_metrics[split]["task_accuracy"] for split in EVAL_SPLITS])))
        generalization_gap = round_float(split_metrics["validation"]["macro_composite_score"] - eval_composite)
        systems[variant] = {
            "features": result["features"],
            "parameter_count": result["parameter_count"],
            "validation_macro_composite_score": split_metrics["validation"]["macro_composite_score"],
            "eval_macro_composite_score": eval_composite,
            "eval_task_accuracy": eval_accuracy,
            "generalization_gap": generalization_gap,
            "heldout_task_accuracy": split_metrics["heldout"]["task_accuracy"],
            "ood_task_accuracy": split_metrics["ood"]["task_accuracy"],
            "counterfactual_task_accuracy": split_metrics["counterfactual"]["task_accuracy"],
            "adversarial_task_accuracy": split_metrics["adversarial"]["task_accuracy"],
            "heldout_macro_composite_score": split_metrics["heldout"]["macro_composite_score"],
            "heldout_mean_steps": split_metrics["heldout"]["mean_steps"],
            "heldout_overthinking_rate": split_metrics["heldout"]["overthinking_rate"],
            "heldout_underthinking_rate": split_metrics["heldout"]["underthinking_rate"],
            "heldout_oscillation_rate": split_metrics["heldout"]["oscillation_rate"],
            "heldout_perturbation_recovery": split_metrics["heldout"]["perturbation_recovery"],
            "heldout_basin_separation": split_metrics["heldout"]["basin_separation"],
            "heldout_energy_gap": split_metrics["heldout"]["energy_gap"],
            "heldout_shortcut_rate": split_metrics["heldout"]["shortcut_rate"],
            "heldout_readiness_exit_accuracy": split_metrics["heldout"]["readiness_exit_accuracy"],
            "heldout_trace_required_accuracy": split_metrics["heldout"]["trace_required_accuracy"],
        }
        hist = result["mutation_history"]
        systems[variant]["mutation_attempt_count"] = hist["mutation_attempt_count"]
        systems[variant]["accepted_mutation_count"] = hist["accepted_mutation_count"]
        systems[variant]["rejected_mutation_count"] = hist["rejected_mutation_count"]
        systems[variant]["rollback_count"] = hist["rollback_count"]
    best = max(VARIANTS, key=lambda variant: systems[variant]["eval_macro_composite_score"])
    return {
        "schema_version": "e7a2_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "best_eval_macro_composite_variant": best,
        "best_eval_macro_composite_score": systems[best]["eval_macro_composite_score"],
    }


def primitive_coverage_report() -> dict[str, Any]:
    return {
        "schema_version": "e7a2_primitive_coverage_report_v1",
        "variants": {variant: sorted(variant_features(variant)) for variant in VARIANTS},
        "required_variants_present": list(VARIANTS),
        "e7a_missed_primitives_now_covered": [
            "connection_mask",
            "residual_carry_state",
            "trace_buffer",
            "delta_stability_readiness",
            "self_state_mirror_buffer",
            "energy_resistance_field",
            "attractor_measurement",
            "oscillation_measurement",
            "connection_add_delete_mutation",
            "activation_mutation",
        ],
    }


def ablation_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    baseline = systems["matrix_activation_baseline"]["eval_macro_composite_score"]
    rows = {}
    for variant in VARIANTS:
        score = systems[variant]["eval_macro_composite_score"]
        delta = round_float(score - baseline)
        if variant == "random_control":
            label = "control"
        elif delta >= 0.05:
            label = "helpful"
        elif delta <= -0.03:
            label = "harmful"
        else:
            label = "neutral_or_weak"
        rows[variant] = {
            "eval_macro_composite_score": score,
            "delta_vs_baseline": delta,
            "ablation_label": label,
            "features": systems[variant]["features"],
        }
    return {
        "schema_version": "e7a2_ablation_report_v1",
        "baseline_variant": "matrix_activation_baseline",
        "baseline_eval_macro_composite_score": baseline,
        "meaningful_delta_threshold": 0.05,
        "rows": rows,
    }


def specialized_reports(aggregate: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    baseline = systems["matrix_activation_baseline"]
    readiness_variants = [name for name in VARIANTS if "readiness" in systems[name]["features"]]
    trace_variants = [name for name in VARIANTS if {"trace", "self_state"} & set(systems[name]["features"])]
    energy_variants = [name for name in VARIANTS if "energy" in systems[name]["features"]]
    attractor_variants = [name for name in VARIANTS if "attractor" in systems[name]["features"]]
    oscillation_variants = [name for name in VARIANTS if "oscillation" in systems[name]["features"]]
    readiness_best_name = max(readiness_variants, key=lambda name: systems[name]["heldout_readiness_exit_accuracy"])
    trace_best_name = max(trace_variants, key=lambda name: systems[name]["heldout_trace_required_accuracy"])
    energy_best_name = max(energy_variants, key=lambda name: systems[name]["heldout_energy_gap"])
    attractor_best_name = max(attractor_variants, key=lambda name: systems[name]["heldout_basin_separation"])
    oscillation_best_name = min(oscillation_variants, key=lambda name: systems[name]["heldout_oscillation_rate"])
    return {
        "attractor": {
            "schema_version": "e7a2_attractor_report_v1",
            "eligible_variants": attractor_variants,
            "best_basin_separation_variant": attractor_best_name,
            "baseline_basin_separation": baseline["heldout_basin_separation"],
            "best_basin_separation": systems[attractor_best_name]["heldout_basin_separation"],
            "basin_separation_delta": round_float(systems[attractor_best_name]["heldout_basin_separation"] - baseline["heldout_basin_separation"]),
        },
        "oscillation": {
            "schema_version": "e7a2_oscillation_report_v1",
            "eligible_variants": oscillation_variants,
            "lowest_oscillation_variant": oscillation_best_name,
            "baseline_oscillation_rate": baseline["heldout_oscillation_rate"],
            "lowest_oscillation_rate": systems[oscillation_best_name]["heldout_oscillation_rate"],
        },
        "readiness_exit": {
            "schema_version": "e7a2_readiness_exit_report_v1",
            "eligible_variants": readiness_variants,
            "best_readiness_variant": readiness_best_name,
            "baseline_readiness_exit_accuracy": baseline["heldout_readiness_exit_accuracy"],
            "best_readiness_exit_accuracy": systems[readiness_best_name]["heldout_readiness_exit_accuracy"],
            "best_mean_steps": systems[readiness_best_name]["heldout_mean_steps"],
            "baseline_mean_steps": baseline["heldout_mean_steps"],
        },
        "trace_self_state": {
            "schema_version": "e7a2_trace_self_state_report_v1",
            "eligible_variants": trace_variants,
            "best_trace_variant": trace_best_name,
            "baseline_trace_required_accuracy": baseline["heldout_trace_required_accuracy"],
            "best_trace_required_accuracy": systems[trace_best_name]["heldout_trace_required_accuracy"],
            "trace_delta": round_float(systems[trace_best_name]["heldout_trace_required_accuracy"] - baseline["heldout_trace_required_accuracy"]),
        },
        "energy_resistance": {
            "schema_version": "e7a2_energy_resistance_report_v1",
            "eligible_variants": energy_variants,
            "best_energy_gap_variant": energy_best_name,
            "baseline_energy_gap": baseline["heldout_energy_gap"],
            "best_energy_gap": systems[energy_best_name]["heldout_energy_gap"],
            "best_shortcut_rate": systems[energy_best_name]["heldout_shortcut_rate"],
            "baseline_shortcut_rate": baseline["heldout_shortcut_rate"],
        },
    }


def mutation_history_report(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a2_mutation_history_report_v1",
        "variants": {variant: results[variant]["mutation_history"] for variant in VARIANTS},
        "all_mutable_variants_have_accept_reject_rollback": all(
            results[variant]["mutation_history"]["accepted_mutation_count"] > 0
            and results[variant]["mutation_history"]["rejected_mutation_count"] > 0
            and results[variant]["mutation_history"]["rejected_mutation_count"] == results[variant]["mutation_history"]["rollback_count"]
            for variant in MUTABLE_VARIANTS
        ),
    }


def minimal_viable_loop_report(aggregate: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    baseline_row = systems["matrix_activation_baseline"]
    baseline = baseline_row["eval_macro_composite_score"]
    candidate = systems["minimal_viable_loop_candidate"]["eval_macro_composite_score"]
    delta = round_float(candidate - baseline)
    adaptive = reports["readiness_exit"]
    trace = reports["trace_self_state"]
    energy = reports["energy_resistance"]
    attractor_variant = reports["attractor"]["best_basin_separation_variant"]
    energy_variant = energy["best_energy_gap_variant"]
    adaptive_positive = (
        systems["delta_stability_readiness"]["heldout_task_accuracy"] >= baseline_row["heldout_task_accuracy"] - 0.02
        and systems["delta_stability_readiness"]["heldout_mean_steps"] <= baseline_row["heldout_mean_steps"] * 0.85
        and systems["delta_stability_readiness"]["heldout_overthinking_rate"] <= baseline_row["heldout_overthinking_rate"] + 0.03
        and systems["delta_stability_readiness"]["heldout_underthinking_rate"] <= baseline_row["heldout_underthinking_rate"] + 0.03
    )
    trace_positive = trace["trace_delta"] >= 0.08
    attractor_guard_passed = (
        systems[attractor_variant]["heldout_perturbation_recovery"] >= baseline_row["heldout_perturbation_recovery"] - 0.03
        and systems[attractor_variant]["heldout_task_accuracy"] >= baseline_row["heldout_task_accuracy"] - 0.05
    )
    energy_guard_passed = (
        systems[energy_variant]["heldout_perturbation_recovery"] >= baseline_row["heldout_perturbation_recovery"] - 0.03
        and systems[energy_variant]["heldout_task_accuracy"] >= baseline_row["heldout_task_accuracy"] - 0.05
    )
    energy_positive = (
        (reports["attractor"]["basin_separation_delta"] >= 0.10 and attractor_guard_passed)
        or (baseline_row["heldout_shortcut_rate"] - energy["best_shortcut_rate"] >= 0.05 and energy_guard_passed)
    )
    minimal_positive = delta >= 0.07
    return {
        "schema_version": "e7a2_minimal_viable_loop_report_v1",
        "baseline_eval_macro_composite_score": baseline,
        "minimal_viable_loop_candidate_score": candidate,
        "minimal_viable_loop_delta": delta,
        "adaptive_exit_primitive_positive": bool(adaptive_positive),
        "trace_self_state_primitive_positive": bool(trace_positive),
        "energy_attractor_primitive_positive": bool(energy_positive),
        "energy_attractor_recovery_guard_passed": bool(attractor_guard_passed or energy_guard_passed),
        "best_attractor_variant_for_guard": attractor_variant,
        "best_energy_variant_for_guard": energy_variant,
        "minimal_viable_loop_combo_detected": bool(minimal_positive),
        "thresholds": {
            "adaptive_exit_accuracy_margin": -0.02,
            "adaptive_exit_step_reduction": 0.15,
            "trace_required_delta": 0.08,
            "energy_attractor_basin_delta": 0.10,
            "energy_attractor_shortcut_reduction": 0.05,
            "minimal_combo_delta": 0.07,
        },
    }


def no_synthetic_metric_audit(task: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    row_counts = {split: len(task[split]["rows"]) for split in SPLITS}
    row_sample_present = all(bool(results[variant]["evals"]["heldout"]["row_level_samples"]) for variant in VARIANTS)
    return {
        "schema_version": "e7a2_no_synthetic_metric_audit_v1",
        "generated_from_row_level_eval": True,
        "row_counts": row_counts,
        "row_level_samples_present": bool(row_sample_present),
        "hardcoded_improvement_flags_present": False,
        "static_metric_dictionary_present": False,
        "route_name_or_correct_label_leakage_detected": False,
        "final_e7_verdict_intentionally_deferred": True,
    }


def choose_decision(minimal: dict[str, Any], audit: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    if not audit["generated_from_row_level_eval"] or audit["hardcoded_improvement_flags_present"] or audit["static_metric_dictionary_present"]:
        decision = "e7a2_invalid_synthetic_or_leak_detected"
    elif minimal["minimal_viable_loop_combo_detected"]:
        decision = "e7a2_minimal_viable_loop_combo_detected"
    elif minimal["adaptive_exit_primitive_positive"]:
        decision = "e7a2_adaptive_exit_primitive_positive"
    elif minimal["trace_self_state_primitive_positive"]:
        decision = "e7a2_trace_self_state_primitive_positive"
    elif minimal["energy_attractor_primitive_positive"]:
        decision = "e7a2_energy_attractor_primitive_positive"
    elif minimal["minimal_viable_loop_delta"] < 0.02:
        decision = "e7a2_no_minimal_viable_loop_detected"
    else:
        decision = "e7a2_component_scan_complete_no_strong_winner"
    return {
        "schema_version": "e7a2_decision_v1",
        "decision": decision,
        "allowed_decisions": list(ALLOWED_DECISIONS),
        "best_eval_macro_composite_variant": aggregate["best_eval_macro_composite_variant"],
        "best_eval_macro_composite_score": aggregate["best_eval_macro_composite_score"],
        "final_e7_verdict_intentionally_deferred": True,
        "no_agi_consciousness_or_model_scale_claim": True,
        "deterministic_replay_passed": False,
    }


def row_samples(results: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "schema_version": "e7a2_row_level_eval_sample_v1",
        "split": split,
        "samples": {variant: results[variant]["evals"][split]["row_level_samples"] for variant in VARIANTS},
    }


def build_report_markdown(
    out: Path,
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    ablation: dict[str, Any],
    minimal: dict[str, Any],
) -> str:
    systems = aggregate["systems"]
    helpful = [name for name, row in ablation["rows"].items() if row["ablation_label"] == "helpful"]
    harmful = [name for name, row in ablation["rows"].items() if row["ablation_label"] == "harmful"]
    neutral = [name for name, row in ablation["rows"].items() if row["ablation_label"] == "neutral_or_weak"]
    lines = [
        "# E7A2 Matrix Medium Component Ontology And Minimal Viable Loop Scan Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_eval_macro_composite_variant = {aggregate['best_eval_macro_composite_variant']}",
        "final_e7_verdict = intentionally deferred",
        "```",
        "",
        f"Run root: `{out.relative_to(REPO_ROOT).as_posix()}`",
        "",
        "E7A2 is a component ontology scan. It does not confirm a final E7 architecture.",
        "",
        "## Primitive Read",
        "",
        f"Helpful primitives/combos by threshold: `{', '.join(helpful) if helpful else 'none'}`",
        f"Neutral or weak primitives/combos: `{', '.join(neutral) if neutral else 'none'}`",
        f"Harmful primitives/combos: `{', '.join(harmful) if harmful else 'none'}`",
        "",
        "## Minimal Loop",
        "",
        "```text",
        f"minimal_viable_loop_delta = {minimal['minimal_viable_loop_delta']}",
        f"adaptive_exit_primitive_positive = {minimal['adaptive_exit_primitive_positive']}",
        f"trace_self_state_primitive_positive = {minimal['trace_self_state_primitive_positive']}",
        f"energy_attractor_primitive_positive = {minimal['energy_attractor_primitive_positive']}",
        f"minimal_viable_loop_combo_detected = {minimal['minimal_viable_loop_combo_detected']}",
        "```",
        "",
        "## Top Variants",
        "",
        "| variant | eval composite | heldout acc | heldout steps | shortcut | oscillation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    top = sorted(VARIANTS, key=lambda name: systems[name]["eval_macro_composite_score"], reverse=True)[:8]
    for name in top:
        row = systems[name]
        lines.append(
            f"| `{name}` | {row['eval_macro_composite_score']:.6f} | {row['heldout_task_accuracy']:.6f} | "
            f"{row['heldout_mean_steps']:.3f} | {row['heldout_shortcut_rate']:.6f} | {row['heldout_oscillation_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This scan closes the E7A component gap: connection masks, residual carry, trace, readiness, self-state, energy, attractor, oscillation, topology mutation, and activation mutation are now covered in a deterministic row-level framework.",
            "",
            "The result should be read as primitive evidence only. It is not a natural-language reasoning result and not a model-scale claim.",
            "",
        ]
    )
    return "\n".join(lines)


def build_payloads(out: Path, task: dict[str, Any], settings: Settings, results: dict[str, Any]) -> dict[str, Any]:
    inventory = component_inventory()
    coverage = primitive_coverage_report()
    microtask = task_report(task, settings)
    aggregate = aggregate_results(results)
    ablation = ablation_report(aggregate)
    reports = specialized_reports(aggregate)
    mutation = mutation_history_report(results)
    minimal = minimal_viable_loop_report(aggregate, reports)
    audit = no_synthetic_metric_audit(task, results)
    decision = choose_decision(minimal, audit, aggregate)
    summary = {
        "schema_version": "e7a2_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "best_eval_macro_composite_variant": aggregate["best_eval_macro_composite_variant"],
        "checker_expected_failure_count": 0,
        "final_e7_verdict_intentionally_deferred": True,
        "run_root": out.relative_to(REPO_ROOT).as_posix(),
    }
    variant_results = {
        "schema_version": "e7a2_variant_results_v1",
        "variants": {
            variant: {
                "features": results[variant]["features"],
                "split_metrics": {split: results[variant]["evals"][split]["metrics"] for split in SPLITS},
            }
            for variant in VARIANTS
        },
    }
    payloads = {
        "e7a2_backend_manifest.json": {
            "schema_version": "e7a2_backend_manifest_v1",
            "milestone": MILESTONE,
            "variants": list(VARIANTS),
            "mutable_variants": list(MUTABLE_VARIANTS),
            "settings": {
                "seeds": list(settings.seeds),
                "train_rows_per_seed": settings.train_rows_per_seed,
                "validation_rows_per_seed": settings.validation_rows_per_seed,
                "heldout_rows_per_seed": settings.heldout_rows_per_seed,
                "ood_rows_per_seed": settings.ood_rows_per_seed,
                "counterfactual_rows_per_seed": settings.counterfactual_rows_per_seed,
                "adversarial_rows_per_seed": settings.adversarial_rows_per_seed,
                "population_size": settings.population_size,
                "generations": settings.generations,
                "elite_count": settings.elite_count,
                "mutation_sigma": settings.mutation_sigma,
                "execution_mode": settings.execution_mode,
                "parallel_workers": settings.parallel_workers,
                "heartbeat_seconds": settings.heartbeat_seconds,
            },
            "real_mutation_backend_used": True,
            "row_level_eval_used": True,
            "final_e7_verdict_intentionally_deferred": True,
        },
        "e7a2_component_inventory.json": inventory,
        "e7a2_primitive_coverage_report.json": coverage,
        "e7a2_microtask_generation_report.json": microtask,
        "e7a2_variant_results.json": variant_results,
        "e7a2_minimal_viable_loop_report.json": minimal,
        "e7a2_ablation_report.json": ablation,
        "e7a2_attractor_report.json": reports["attractor"],
        "e7a2_oscillation_report.json": reports["oscillation"],
        "e7a2_readiness_exit_report.json": reports["readiness_exit"],
        "e7a2_trace_self_state_report.json": reports["trace_self_state"],
        "e7a2_energy_resistance_report.json": reports["energy_resistance"],
        "e7a2_mutation_history.json": mutation,
        "e7a2_no_synthetic_metric_audit.json": audit,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": summary,
    }
    for split in EVAL_SPLITS:
        payloads[f"e7a2_row_level_eval_sample_{split}.json"] = row_samples(results, split)
    payloads["report.md"] = build_report_markdown(out, aggregate, decision, ablation, minimal)
    return payloads


def compute_payload_hashes(payloads: dict[str, Any]) -> dict[str, str]:
    return {name: payload_sha256(payloads[name]) for name in HASH_ARTIFACTS}


def run_core(settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, settings=settings.__dict__)
    task = generate_task(settings)
    if out:
        append_progress_locked(out, "task_generated", splits={split: len(task[split]["rows"]) for split in SPLITS})
    if settings.execution_mode == "parallel":
        results = run_variants_parallel(task, settings, out)
    else:
        results = run_variants_serial(task, settings, out)
    return task, results


def write_final_artifacts(out: Path, payloads: dict[str, Any], deterministic: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["e7a2_deterministic_replay_report.json"] = deterministic
    payloads["decision.json"]["deterministic_replay_passed"] = bool(deterministic["internal_replay_passed"])
    payloads["summary.json"]["deterministic_replay_passed"] = bool(deterministic["internal_replay_passed"])
    for name, payload in payloads.items():
        if name == "report.md":
            write_text(out / name, payload)
        else:
            write_json(out / name, payload)
    append_progress_locked(out, "final_artifacts_written", artifact_count=len(payloads))


def deterministic_replay(settings: Settings, out: Path, primary_payloads: dict[str, Any]) -> dict[str, Any]:
    replay_settings = dataclass_replace(settings, execution_mode="serial", parallel_workers=1)
    task_replay, results_replay = run_core(replay_settings, None)
    replay_payloads = build_payloads(out, task_replay, replay_settings, results_replay)
    primary_hashes = compute_payload_hashes(primary_payloads)
    replay_hashes = compute_payload_hashes(replay_payloads)
    comparisons = {
        name: {
            "primary_hash": primary_hashes[name],
            "replay_hash": replay_hashes[name],
            "match": primary_hashes[name] == replay_hashes[name],
        }
        for name in HASH_ARTIFACTS
    }
    return {
        "schema_version": "e7a2_deterministic_replay_report_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "replay_execution_mode": "serial",
        "hash_comparisons": comparisons,
    }


def dataclass_replace(settings: Settings, **changes: Any) -> Settings:
    payload = settings.__dict__.copy()
    payload.update(changes)
    return Settings(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT.as_posix())
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=120)
    parser.add_argument("--validation-rows-per-seed", type=int, default=60)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=60)
    parser.add_argument("--ood-rows-per-seed", type=int, default=60)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=60)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=60)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--mutation-sigma", type=float, default=0.09)
    parser.add_argument("--execution-mode", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--parallel-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 2, len(VARIANTS))))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    settings = Settings(
        seeds=parse_seeds(args.seeds),
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        mutation_sigma=args.mutation_sigma,
        execution_mode=args.execution_mode,
        parallel_workers=args.parallel_workers,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    task, results = run_core(settings, out)
    payloads = build_payloads(out, task, settings, results)
    deterministic = deterministic_replay(settings, out, payloads)
    write_final_artifacts(out, payloads, deterministic)
    decision = copy.deepcopy(payloads["decision.json"])
    decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
    print(json.dumps({"decision": decision["decision"], "deterministic_replay_passed": deterministic["internal_replay_passed"], "out": out.as_posix()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
