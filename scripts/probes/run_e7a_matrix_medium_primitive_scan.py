#!/usr/bin/env python3
"""E7A primitive scan for mutable matrix-medium components.

This is intentionally not a final E7 routing verdict. It probes the lowest
abstraction layer only: mutable matrices, shared activations, recurrence, and
halting behavior on controlled primitive tasks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
E4_PATH = Path(__file__).with_name("run_e4_decision_relevant_abstraction_routing_probe.py")
MILESTONE = "E7A_MATRIX_MEDIUM_PRIMITIVE_SCAN"
DEFAULT_OUT = Path("target/pilot_wave/e7a_matrix_medium_primitive_scan")
DEFAULT_SEEDS = (78001, 78002, 78003)

INPUT_DIM = 8
HIDDEN_DIM = 10
CLASS_COUNT = 4
MAX_STEPS = 6
FAMILIES = ("linear", "xor", "ring", "wave")
SYSTEMS = (
    "random_classifier",
    "linear_matrix_depth1",
    "linear_matrix_depth3",
    "linear_matrix_depth6",
    "tanh_matrix_depth3",
    "relu_matrix_depth3",
    "c19_fixed_matrix_depth3",
    "c19_rho0_matrix_depth3",
    "c19_c_mut_matrix_depth3",
    "c19_rho_mut_matrix_depth3",
    "c19_c_rho_mut_matrix_depth3",
    "c19_fixed_recurrent_fixed6",
    "c19_c_rho_mut_recurrent_fixed6",
    "c19_fixed_recurrent_halting6",
    "c19_c_rho_mut_recurrent_halting6",
    "c19_c_rho_mut_recurrent_halting_restart6",
)
MUTATION_SYSTEMS = tuple(system for system in SYSTEMS if system != "random_classifier")
HASH_ARTIFACTS = (
    "e7a_primitive_scan_report.json",
    "e7a_collapse_audit.json",
    "e7a_c19_parameter_mode_audit.json",
    "e7a_halting_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)

TRUE_LINEAR_W = np.asarray(
    [
        [0.80, -0.30, 0.15, -0.60],
        [-0.20, 0.75, -0.50, 0.10],
        [0.35, 0.10, 0.65, -0.45],
        [-0.50, -0.20, 0.25, 0.70],
        [0.20, -0.55, 0.05, 0.30],
        [-0.15, 0.25, -0.35, 0.55],
    ],
    dtype=np.float64,
)
TRUE_LINEAR_B = np.asarray([0.05, -0.02, 0.03, -0.04], dtype=np.float64)


def load_e4_module() -> Any:
    spec = importlib.util.spec_from_file_location("e4_abstraction_routing", E4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E4 backend from {E4_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e4 = load_e4_module()
e2 = e4.e2


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


e2.append_progress = append_progress_locked


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


def stable_seed(label: str) -> int:
    return e2.stable_seed(f"e7a-{label}")


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one seed is required")
    return values


def resolve_out(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    relative = resolved.relative_to(REPO_ROOT)
    if len(relative.parts) < 2 or relative.parts[0].lower() != "target" or relative.parts[1].lower() != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def c19(x: np.ndarray, c: float | np.ndarray = 3.0, rho: float | np.ndarray = 1.0) -> np.ndarray:
    c = np.maximum(np.asarray(c, dtype=np.float64), 0.1)
    rho = np.maximum(np.asarray(rho, dtype=np.float64), 0.0)
    limit = 6.0 * c
    scaled = x / c
    n = np.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    sign = np.where((n.astype(np.int64) % 2) == 0, 1.0, -1.0)
    interior = c * (sign * h + rho * h * h)
    return np.where(x >= limit, x - limit, np.where(x <= -limit, x + limit, interior))


def activate(name: str, x: np.ndarray, c: float | np.ndarray = 3.0, rho: float | np.ndarray = 1.0) -> np.ndarray:
    if name == "identity":
        return x
    if name == "tanh":
        return np.tanh(x)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "c19":
        return c19(x, c, rho)
    raise ValueError(f"unknown activation: {name}")


def target_for_family(family: str, base: np.ndarray) -> int:
    if family == "linear":
        scores = base[:6] @ TRUE_LINEAR_W + TRUE_LINEAR_B
        return int(np.argmax(scores))
    if family == "xor":
        return int((base[0] > 0.0) != (base[1] > 0.0))
    if family == "ring":
        radius = base[0] * base[0] + base[1] * base[1] + 0.35 * base[2] * base[2]
        return 2 if radius > 1.35 else 3
    if family == "wave":
        value = math.sin(2.4 * base[0]) + 0.55 * math.cos(2.1 * base[1]) - 0.25 * base[2]
        return 1 if value > 0.15 else 0
    raise ValueError(f"unknown family: {family}")


def make_row(split: str, seed: int, index: int, rng: np.random.Generator) -> dict[str, Any]:
    family = FAMILIES[index % len(FAMILIES)]
    scale = 1.0
    shift = 0.0
    if split == "ood":
        scale = 1.65
        shift = 0.25
    base = rng.normal(loc=shift, scale=scale, size=6).astype(np.float64)
    if split == "counterfactual":
        base[0] *= -1.0
        base[3] *= -1.0
    target = target_for_family(family, base)
    x = np.zeros(INPUT_DIM, dtype=np.float64)
    x[:6] = base
    x[6] = rng.normal(0.0, 1.0)
    x[7] = rng.normal(0.0, 1.0)
    if split == "adversarial":
        wrong = (target + 1) % CLASS_COUNT
        x[6] = (wrong - 1.5) / 1.5 + rng.normal(0.0, 0.03)
        x[7] = -x[6]
    return {
        "row_id": f"{split}_{seed}_{index:05d}",
        "split": split,
        "family": family,
        "x": [round_float(value) for value in x.tolist()],
        "target": target,
    }


def generate_split(split: str, seeds: tuple[int, ...], rows_per_seed: int) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(stable_seed(f"task-{split}-{seed}"))
        for index in range(rows_per_seed):
            rows.append(make_row(split, seed, index, rng))
    x = np.asarray([row["x"] for row in rows], dtype=np.float64)
    y = np.asarray([row["target"] for row in rows], dtype=np.int64)
    families = np.asarray([FAMILIES.index(row["family"]) for row in rows], dtype=np.int64)
    return {"rows": rows, "x": x, "y": y, "families": families}


def generate_task(settings: Settings) -> dict[str, Any]:
    return {
        "train": generate_split("train", settings.seeds, settings.train_rows_per_seed),
        "validation": generate_split("validation", settings.seeds, settings.validation_rows_per_seed),
        "heldout": generate_split("heldout", settings.seeds, settings.heldout_rows_per_seed),
        "ood": generate_split("ood", settings.seeds, settings.ood_rows_per_seed),
        "counterfactual": generate_split("counterfactual", settings.seeds, settings.counterfactual_rows_per_seed),
        "adversarial": generate_split("adversarial", settings.seeds, settings.adversarial_rows_per_seed),
    }


def system_spec(system: str) -> dict[str, Any]:
    if system == "linear_matrix_depth1":
        return {"mode": "feedforward", "activation": "identity", "c19_mode": "off", "depth": 1, "recurrent": False, "halting": False, "restart": False}
    if system == "linear_matrix_depth3":
        return {"mode": "feedforward", "activation": "identity", "c19_mode": "off", "depth": 3, "recurrent": False, "halting": False, "restart": False}
    if system == "linear_matrix_depth6":
        return {"mode": "feedforward", "activation": "identity", "c19_mode": "off", "depth": 6, "recurrent": False, "halting": False, "restart": False}
    if system == "tanh_matrix_depth3":
        return {"mode": "feedforward", "activation": "tanh", "c19_mode": "off", "depth": 3, "recurrent": False, "halting": False, "restart": False}
    if system == "relu_matrix_depth3":
        return {"mode": "feedforward", "activation": "relu", "c19_mode": "off", "depth": 3, "recurrent": False, "halting": False, "restart": False}
    if system == "c19_fixed_matrix_depth3":
        return {"mode": "feedforward", "activation": "c19", "c19_mode": "fixed", "depth": 3, "recurrent": False, "halting": False, "restart": False}
    if system == "c19_rho0_matrix_depth3":
        return {"mode": "feedforward", "activation": "c19", "c19_mode": "rho0", "depth": 3, "recurrent": False, "halting": False, "restart": False}
    if system == "c19_c_mut_matrix_depth3":
        return {"mode": "feedforward", "activation": "c19", "c19_mode": "c_mut", "depth": 3, "recurrent": False, "halting": False, "restart": False}
    if system == "c19_rho_mut_matrix_depth3":
        return {"mode": "feedforward", "activation": "c19", "c19_mode": "rho_mut", "depth": 3, "recurrent": False, "halting": False, "restart": False}
    if system == "c19_c_rho_mut_matrix_depth3":
        return {"mode": "feedforward", "activation": "c19", "c19_mode": "c_rho_mut", "depth": 3, "recurrent": False, "halting": False, "restart": False}
    if system == "c19_fixed_recurrent_fixed6":
        return {"mode": "recurrent", "activation": "c19", "c19_mode": "fixed", "depth": MAX_STEPS, "recurrent": True, "halting": False, "restart": False}
    if system == "c19_c_rho_mut_recurrent_fixed6":
        return {"mode": "recurrent", "activation": "c19", "c19_mode": "c_rho_mut", "depth": MAX_STEPS, "recurrent": True, "halting": False, "restart": False}
    if system == "c19_fixed_recurrent_halting6":
        return {"mode": "recurrent", "activation": "c19", "c19_mode": "fixed", "depth": MAX_STEPS, "recurrent": True, "halting": True, "restart": False}
    if system == "c19_c_rho_mut_recurrent_halting6":
        return {"mode": "recurrent", "activation": "c19", "c19_mode": "c_rho_mut", "depth": MAX_STEPS, "recurrent": True, "halting": True, "restart": False}
    if system == "c19_c_rho_mut_recurrent_halting_restart6":
        return {"mode": "recurrent", "activation": "c19", "c19_mode": "c_rho_mut", "depth": MAX_STEPS, "recurrent": True, "halting": True, "restart": True}
    raise ValueError(f"unknown system: {system}")


def candidate_template(system: str, rng: np.random.Generator) -> dict[str, Any]:
    spec = system_spec(system)
    scale = 0.28
    params: dict[str, Any] = {
        "schema_version": "e7a_matrix_medium_candidate_v1",
        "system": system,
        "spec": spec,
        "w_in": rng.normal(0.0, scale, size=(INPUT_DIM, HIDDEN_DIM)).tolist(),
        "b_in": rng.normal(0.0, 0.05, size=HIDDEN_DIM).tolist(),
        "w_out": rng.normal(0.0, scale, size=(HIDDEN_DIM, CLASS_COUNT)).tolist(),
        "b_out": rng.normal(0.0, 0.05, size=CLASS_COUNT).tolist(),
    }
    if spec["activation"] == "c19":
        if spec["c19_mode"] in {"c_mut", "c_rho_mut"}:
            params["c19_c"] = np.maximum(0.1, rng.normal(3.0, 0.18, size=HIDDEN_DIM)).tolist()
        if spec["c19_mode"] in {"rho_mut", "c_rho_mut"}:
            params["c19_rho"] = np.maximum(0.0, rng.normal(1.0, 0.12, size=HIDDEN_DIM)).tolist()
    if spec["mode"] == "feedforward":
        params["layers"] = [
            {
                "w": rng.normal(0.0, scale, size=(HIDDEN_DIM, HIDDEN_DIM)).tolist(),
                "b": rng.normal(0.0, 0.05, size=HIDDEN_DIM).tolist(),
            }
            for _ in range(spec["depth"])
        ]
    else:
        params["w_state"] = rng.normal(0.0, scale, size=(HIDDEN_DIM, HIDDEN_DIM)).tolist()
        params["w_skip"] = rng.normal(0.0, scale, size=(INPUT_DIM, HIDDEN_DIM)).tolist()
        params["b_state"] = rng.normal(0.0, 0.05, size=HIDDEN_DIM).tolist()
        params["halt_w"] = rng.normal(0.0, scale, size=HIDDEN_DIM).tolist()
        params["halt_b"] = float(rng.normal(0.0, 0.05))
        params["action_w"] = rng.normal(0.0, scale, size=(HIDDEN_DIM, 4)).tolist()
        params["action_b"] = rng.normal(0.0, 0.05, size=4).tolist()
    return round_candidate(params)


def round_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    rounded = copy.deepcopy(candidate)
    for path, value in e2.flatten_paths(rounded):
        if path and path[0] == "spec":
            continue
        e2.set_path(rounded, path, round_float(value))
    return rounded


def finite_candidate(candidate: dict[str, Any]) -> bool:
    return all(math.isfinite(value) for path, value in e2.flatten_paths(candidate) if not (path and path[0] == "spec"))


def vector_hash(candidate: dict[str, Any]) -> str:
    clean = {key: value for key, value in candidate.items() if key != "candidate_id"}
    return e2.payload_sha256(clean)


def mutable_paths(candidate: dict[str, Any]) -> list[tuple[tuple[Any, ...], float]]:
    return [(path, value) for path, value in e2.flatten_paths(candidate) if not (path and path[0] == "spec")]


def clamp_path(path: tuple[Any, ...], value: float) -> float:
    if "c19_c" in path:
        return e2.clamp(value, 0.1, 8.0)
    if "c19_rho" in path:
        return e2.clamp(value, 0.0, 5.0)
    if path and path[-1] in {"halt_b"}:
        return e2.clamp(value, -4.0, 4.0)
    return e2.clamp(value, -5.0, 5.0)


def mutate(candidate: dict[str, Any], rng: random.Random, sigma: float) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    paths = mutable_paths(child)
    edit_count = rng.randint(2, 14)
    for path, value in rng.sample(paths, k=min(edit_count, len(paths))):
        e2.set_path(child, path, round_float(clamp_path(path, value + rng.gauss(0.0, sigma))))
    return child


def arrays(candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "w_in": np.asarray(candidate["w_in"], dtype=np.float64),
        "b_in": np.asarray(candidate["b_in"], dtype=np.float64),
        "w_out": np.asarray(candidate["w_out"], dtype=np.float64),
        "b_out": np.asarray(candidate["b_out"], dtype=np.float64),
    }
    if candidate["spec"]["mode"] == "feedforward":
        result["layers"] = [
            {"w": np.asarray(layer["w"], dtype=np.float64), "b": np.asarray(layer["b"], dtype=np.float64)}
            for layer in candidate["layers"]
        ]
    else:
        for key in ("w_state", "w_skip", "b_state", "halt_w", "action_w", "action_b"):
            result[key] = np.asarray(candidate[key], dtype=np.float64)
        result["halt_b"] = float(candidate["halt_b"])
    if "c19_c" in candidate:
        result["c19_c"] = np.asarray(candidate["c19_c"], dtype=np.float64)
    if "c19_rho" in candidate:
        result["c19_rho"] = np.asarray(candidate["c19_rho"], dtype=np.float64)
    return result


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def c19_params(spec: dict[str, Any], arr: dict[str, Any]) -> tuple[float | np.ndarray, float | np.ndarray]:
    if spec.get("activation") != "c19":
        return 3.0, 1.0
    mode = spec.get("c19_mode", "fixed")
    c: float | np.ndarray = arr.get("c19_c", 3.0)
    rho: float | np.ndarray = arr.get("c19_rho", 1.0)
    if mode == "rho0":
        rho = 0.0
    if mode == "fixed":
        c, rho = 3.0, 1.0
    if mode == "c_mut":
        rho = 1.0
    if mode == "rho_mut":
        c = 3.0
    return c, rho


def predict_candidate(candidate: dict[str, Any], split_data: dict[str, Any]) -> dict[str, Any]:
    spec = candidate["spec"]
    arr = arrays(candidate)
    x = split_data["x"]
    activation = spec["activation"]
    act_c, act_rho = c19_params(spec, arr)
    if spec["mode"] == "feedforward":
        h = activate(activation, x @ arr["w_in"] + arr["b_in"], act_c, act_rho)
        states = [h]
        for layer in arr["layers"]:
            h = activate(activation, h @ layer["w"] + layer["b"], act_c, act_rho)
            states.append(h)
        logits = h @ arr["w_out"] + arr["b_out"]
        return {
            "pred": np.argmax(logits, axis=1).astype(np.int64),
            "steps": np.full(x.shape[0], spec["depth"], dtype=np.int64),
            "state": h,
            "halted_by": np.asarray(["fixed"] * x.shape[0], dtype=object),
        }

    initial = activate(activation, x @ arr["w_in"] + arr["b_in"], act_c, act_rho)
    state = initial.copy()
    chosen_logits = None
    chosen_state = None
    chosen_steps = np.full(x.shape[0], spec["depth"], dtype=np.int64)
    halted = np.zeros(x.shape[0], dtype=bool)
    halted_by = np.asarray(["max_depth"] * x.shape[0], dtype=object)
    for step in range(1, spec["depth"] + 1):
        state = activate(activation, state @ arr["w_state"] + x @ arr["w_skip"] + arr["b_state"], act_c, act_rho)
        logits = state @ arr["w_out"] + arr["b_out"]
        if chosen_logits is None:
            chosen_logits = logits.copy()
            chosen_state = state.copy()
        if spec["halting"]:
            halt_score = sigmoid(state @ arr["halt_w"] + arr["halt_b"])
            step_halt = halt_score > 0.58
            if spec["restart"]:
                actions = state @ arr["action_w"] + arr["action_b"]
                action = np.argmax(actions, axis=1)
                restart_mask = (action == 2) & (~halted) & (step < spec["depth"])
                sleep_mask = (action == 3) & (~halted) & (step < spec["depth"])
                if np.any(restart_mask):
                    state[restart_mask] = initial[restart_mask]
                    halted_by[restart_mask] = "restart_then_continue"
                if np.any(sleep_mask):
                    halted_by[sleep_mask] = "sleep_then_continue"
                step_halt = step_halt | (action == 1)
            take = step_halt & (~halted)
            if np.any(take):
                chosen_logits[take] = logits[take]
                chosen_state[take] = state[take]
                chosen_steps[take] = step
                halted[take] = True
                halted_by[take] = "halt"
        else:
            chosen_logits = logits.copy()
            chosen_state = state.copy()
    assert chosen_logits is not None and chosen_state is not None
    not_halted = ~halted
    if np.any(not_halted):
        final_logits = state @ arr["w_out"] + arr["b_out"]
        chosen_logits[not_halted] = final_logits[not_halted]
        chosen_state[not_halted] = state[not_halted]
    return {
        "pred": np.argmax(chosen_logits, axis=1).astype(np.int64),
        "steps": chosen_steps,
        "state": chosen_state,
        "halted_by": halted_by,
    }


def random_predictions(split_data: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows = len(split_data["rows"])
    return {
        "pred": rng.integers(0, CLASS_COUNT, size=rows, dtype=np.int64),
        "steps": rng.integers(1, MAX_STEPS + 1, size=rows, dtype=np.int64),
        "state": np.zeros((rows, HIDDEN_DIM), dtype=np.float64),
        "halted_by": np.asarray(["random"] * rows, dtype=object),
    }


def evaluate_prediction(pred: dict[str, Any], split_data: dict[str, Any], sample_limit: int = 8) -> dict[str, Any]:
    y = split_data["y"]
    correct = pred["pred"] == y
    families = split_data["families"]
    per_family = {}
    for family_index, family in enumerate(FAMILIES):
        mask = families == family_index
        per_family[family] = round_float(float(np.mean(correct[mask]))) if np.any(mask) else None
    steps = np.asarray(pred["steps"], dtype=np.float64)
    state = np.asarray(pred["state"], dtype=np.float64)
    metrics = {
        "accuracy": round_float(float(np.mean(correct))),
        "per_family_accuracy": per_family,
        "macro_family_accuracy": round_float(float(np.mean([value for value in per_family.values() if value is not None]))),
        "mean_steps": round_float(float(np.mean(steps))),
        "max_steps_used": int(np.max(steps)) if steps.size else 0,
        "halt_efficiency": round_float(float(np.mean(correct.astype(np.float64) * (1.0 - (steps - 1.0) / max(1.0, MAX_STEPS - 1.0))))),
        "state_mean_abs": round_float(float(np.mean(np.abs(state)))) if state.size else 0.0,
        "state_std": round_float(float(np.std(state))) if state.size else 0.0,
    }
    halted_by = pred["halted_by"]
    metrics["halted_by"] = {str(key): int(np.sum(halted_by == key)) for key in sorted(set(halted_by.tolist()))}
    samples = []
    for index in range(min(sample_limit, len(split_data["rows"]))):
        row = split_data["rows"][index]
        samples.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "target": int(row["target"]),
                "pred": int(pred["pred"][index]),
                "steps": int(pred["steps"][index]),
                "correct": bool(correct[index]),
            }
        )
    return {"metrics": metrics, "row_level_samples": samples}


def evaluate_candidate(candidate: dict[str, Any], task: dict[str, Any], sample_limit: int = 8) -> dict[str, Any]:
    evals = {}
    for split, data in task.items():
        evals[split] = evaluate_prediction(predict_candidate(candidate, data), data, sample_limit)
    return evals


def fitness_from_evals(evals: dict[str, Any]) -> float:
    train = evals["train"]["metrics"]
    validation = evals["validation"]["metrics"]
    mean_steps = validation["mean_steps"]
    return (
        0.45 * train["macro_family_accuracy"]
        + 0.45 * validation["macro_family_accuracy"]
        + 0.10 * validation["halt_efficiency"]
        - 0.015 * ((mean_steps - 1.0) / max(1.0, MAX_STEPS - 1.0))
    )


def search_eval(candidate: dict[str, Any], task: dict[str, Any], all_splits: bool = False) -> dict[str, Any]:
    splits = task if all_splits else {"train": task["train"], "validation": task["validation"]}
    evals = {split: evaluate_prediction(predict_candidate(candidate, data), data, 0) for split, data in splits.items()}
    return {"candidate": candidate, "evals": evals, "fitness": round_float(fitness_from_evals(evals))}


def parameter_diff(system: str, initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    before = {path: value for path, value in e2.flatten_paths(initial) if not (path and path[0] == "spec")}
    after = {path: value for path, value in e2.flatten_paths(final) if not (path and path[0] == "spec")}
    changed = {}
    l2 = 0.0
    for path in sorted(before):
        delta = after[path] - before[path]
        if abs(delta) > 1e-12:
            key = ".".join(str(part) for part in path)
            changed[key] = {"before": before[path], "after": after[path], "delta": round_float(delta)}
            l2 += delta * delta
    return {
        "schema_version": f"e7a_parameter_diff_{system}_v1",
        "system": system,
        "before_hash": vector_hash(initial),
        "after_hash": vector_hash(final),
        "actual_parameter_diff_found": bool(changed),
        "changed_parameter_count": len(changed),
        "parameter_diff_l2": round_float(math.sqrt(l2)),
        "changed_parameters_sample": dict(list(changed.items())[:80]),
    }


def generation_metric(system: str, evals: dict[str, Any], accepted: int, rejected: int, rollback: int, state_hash: str) -> dict[str, Any]:
    return {
        "system": system,
        "train_accuracy": evals["train"]["metrics"]["accuracy"],
        "validation_accuracy": evals["validation"]["metrics"]["accuracy"],
        "train_macro_family_accuracy": evals["train"]["metrics"]["macro_family_accuracy"],
        "validation_macro_family_accuracy": evals["validation"]["metrics"]["macro_family_accuracy"],
        "validation_mean_steps": evals["validation"]["metrics"]["mean_steps"],
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "state_hash": state_hash,
    }


def mutation_history_artifact(system: str, attempts: int, accepted: int, rejected: int, rollback: int, history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": f"e7a_mutation_history_{system}_v1",
        "system": system,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "history": history,
    }


def run_mutation_system(system: str, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"search-{system}-{settings.seeds}"))
    np_rng = np.random.default_rng(stable_seed(f"init-{system}-{settings.seeds}"))
    initial = candidate_template(system, np_rng)
    population = [initial]
    for _ in range(settings.population_size - 1):
        population.append(mutate(initial, rng, settings.mutation_sigma))
    scored = [search_eval(candidate, task) for candidate in population]
    scored.sort(key=lambda row: row["fitness"], reverse=True)
    best = scored[0]
    history: list[dict[str, Any]] = []
    generation_metrics: list[dict[str, Any]] = []
    attempts = accepted = rejected = rollback = 0
    start = time.perf_counter()
    for generation in range(1, settings.generations + 1):
        elites = scored[: settings.elite_count]
        for attempt in range(settings.population_size):
            parent = elites[attempt % len(elites)]
            attempts += 1
            child = mutate(parent["candidate"], rng, settings.mutation_sigma)
            child_eval = search_eval(child, task) if finite_candidate(child) else {"candidate": child, "evals": {}, "fitness": -1_000_000.0}
            better = child_eval["fitness"] > parent["fitness"]
            neutral = child_eval["fitness"] == parent["fitness"]
            accepted_flag = better or (neutral and rng.random() < 0.25)
            if accepted_flag:
                accepted += 1
                scored.append(child_eval)
                scored.sort(key=lambda row: row["fitness"], reverse=True)
                scored = scored[: settings.population_size]
                if child_eval["fitness"] >= best["fitness"]:
                    best = child_eval
            else:
                rejected += 1
                rollback += 1
            history.append(
                {
                    "system": system,
                    "generation": generation,
                    "attempt": attempts,
                    "accepted": bool(accepted_flag),
                    "parent_hash": vector_hash(parent["candidate"]),
                    "candidate_hash": vector_hash(child),
                    "parent_fitness": parent["fitness"],
                    "candidate_fitness": child_eval["fitness"],
                    "rollback_performed": not accepted_flag,
                }
            )
        full = evaluate_candidate(best["candidate"], task, sample_limit=0)
        row = generation_metric(system, full, accepted, rejected, rollback, vector_hash(best["candidate"]))
        generation_metrics.append(row)
        if out is not None:
            e2.append_progress(out, "generation_complete", system=system, generation=generation, metrics=row)
            e2.write_json(out / f"e7a_mutation_history_{system}.json", mutation_history_artifact(system, attempts, accepted, rejected, rollback, history))
    final_eval = evaluate_candidate(best["candidate"], task, sample_limit=8)
    diff = parameter_diff(system, initial, best["candidate"])
    parameter_count = len(mutable_paths(initial))
    return {
        "system": system,
        "training_mode": "mutation_only",
        "spec": system_spec(system),
        "runtime_seconds": round_float(time.perf_counter() - start),
        "parameter_count": parameter_count,
        "initial_state": {"state_hash": vector_hash(initial), "parameter_count": parameter_count},
        "final_state": diff | {"state_hash": vector_hash(best["candidate"]), "parameter_count": parameter_count},
        "final_eval": {"evals": final_eval},
        "history": history,
        "generation_metrics": generation_metrics,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "_initial": initial,
        "_candidate": best["candidate"],
    }


def run_random(task: dict[str, Any]) -> dict[str, Any]:
    evals = {}
    for split, data in task.items():
        evals[split] = evaluate_prediction(random_predictions(data, stable_seed(f"random-{split}")), data, sample_limit=8)
    return {
        "system": "random_classifier",
        "training_mode": "none",
        "spec": {"mode": "random"},
        "runtime_seconds": 0.0,
        "parameter_count": 0,
        "initial_state": {"state_hash": e2.payload_sha256({"system": "random_classifier"}), "parameter_count": 0},
        "final_state": {"state_hash": e2.payload_sha256({"system": "random_classifier"}), "parameter_count": 0},
        "final_eval": {"evals": evals},
    }


def run_single_system(system: str, task: dict[str, Any], settings: Settings, out_raw: str | None) -> tuple[str, dict[str, Any]]:
    out = Path(out_raw) if out_raw else None
    started = time.perf_counter()
    if out is not None:
        e2.append_progress(out, "system_start", system=system)
    if system == "random_classifier":
        result = run_random(task)
    else:
        result = run_mutation_system(system, task, settings, out)
    if out is not None:
        e2.append_progress(out, "system_complete", system=system, runtime_seconds=round_float(time.perf_counter() - started))
    return system, result


def run_systems_serial(task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    results = {}
    for system in SYSTEMS:
        returned, result = run_single_system(system, task, settings, out.as_posix() if out else None)
        results[returned] = result
    return results


def run_systems_parallel(task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    worker_count = settings.parallel_workers if settings.parallel_workers > 0 else min(len(SYSTEMS), max(1, (os.cpu_count() or 4) - 2))
    worker_count = min(max(1, worker_count), len(SYSTEMS))
    out_raw = out.as_posix() if out is not None else None
    if out is not None:
        e2.append_progress(out, "parallel_systems_start", systems=list(SYSTEMS), worker_count=worker_count)
    results: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_single_system, system, task, settings, out_raw): system for system in SYSTEMS}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
            if not done:
                if out is not None:
                    e2.append_progress(out, "parallel_heartbeat", completed_systems=sorted(results), pending_systems=sorted(futures[future] for future in pending))
                continue
            for future in done:
                expected = futures[future]
                system, result = future.result()
                if system != expected:
                    raise RuntimeError(f"system mismatch: expected {expected}, got {system}")
                results[system] = result
                if out is not None:
                    e2.append_progress(out, "parallel_system_result_received", system=system, completed_count=len(results), pending_count=len(pending))
    if out is not None:
        e2.append_progress(out, "parallel_systems_complete", completed_systems=sorted(results))
    return results


def collapse_linear_candidate(candidate: dict[str, Any], x: np.ndarray) -> np.ndarray:
    arr = arrays(candidate)
    a = arr["w_in"]
    b = arr["b_in"]
    for layer in arr["layers"]:
        b = b @ layer["w"] + layer["b"]
        a = a @ layer["w"]
    out_w = a @ arr["w_out"]
    out_b = b @ arr["w_out"] + arr["b_out"]
    return x @ out_w + out_b


def collapse_audit(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    x = task["heldout"]["x"]
    for system in ("linear_matrix_depth1", "linear_matrix_depth3", "linear_matrix_depth6"):
        candidate = searches[system]["_candidate"]
        stacked = predict_candidate(candidate, task["heldout"])
        collapsed_logits = collapse_linear_candidate(candidate, x)
        collapsed_pred = np.argmax(collapsed_logits, axis=1).astype(np.int64)
        rows[system] = {
            "stack_equals_collapsed_predictions": bool(np.all(stacked["pred"] == collapsed_pred)),
            "prediction_mismatch_count": int(np.sum(stacked["pred"] != collapsed_pred)),
        }
    return {
        "schema_version": "e7a_collapse_audit_v1",
        "linear_depth_systems_are_functionally_collapsible": all(row["stack_equals_collapsed_predictions"] for row in rows.values()),
        "systems": rows,
    }


def system_metrics(search: dict[str, Any]) -> dict[str, Any]:
    evals = search["final_eval"]["evals"]
    heldout = evals["heldout"]["metrics"]
    train = evals["train"]["metrics"]
    validation = evals["validation"]["metrics"]
    result = {
        "system": search["system"],
        "training_mode": search["training_mode"],
        "spec": search["spec"],
        "parameter_count": search["parameter_count"],
        "train_macro_family_accuracy": train["macro_family_accuracy"],
        "validation_macro_family_accuracy": validation["macro_family_accuracy"],
        "heldout_macro_family_accuracy": heldout["macro_family_accuracy"],
        "ood_macro_family_accuracy": evals["ood"]["metrics"]["macro_family_accuracy"],
        "counterfactual_macro_family_accuracy": evals["counterfactual"]["metrics"]["macro_family_accuracy"],
        "adversarial_macro_family_accuracy": evals["adversarial"]["metrics"]["macro_family_accuracy"],
        "heldout_accuracy": heldout["accuracy"],
        "heldout_per_family_accuracy": heldout["per_family_accuracy"],
        "heldout_mean_steps": heldout["mean_steps"],
        "heldout_halt_efficiency": heldout["halt_efficiency"],
        "heldout_state_mean_abs": heldout["state_mean_abs"],
        "heldout_state_std": heldout["state_std"],
        "generalization_gap": round_float(train["macro_family_accuracy"] - heldout["macro_family_accuracy"]),
    }
    for key in ("mutation_attempt_count", "accepted_mutation_count", "rejected_mutation_count", "rollback_count"):
        if key in search:
            result[key] = search[key]
    return result


def halting_audit(searches: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for system in (
        "c19_fixed_recurrent_fixed6",
        "c19_c_rho_mut_recurrent_fixed6",
        "c19_fixed_recurrent_halting6",
        "c19_c_rho_mut_recurrent_halting6",
        "c19_c_rho_mut_recurrent_halting_restart6",
    ):
        metrics = searches[system]["final_eval"]["evals"]["heldout"]["metrics"]
        rows[system] = {
            "accuracy": metrics["accuracy"],
            "macro_family_accuracy": metrics["macro_family_accuracy"],
            "mean_steps": metrics["mean_steps"],
            "halt_efficiency": metrics["halt_efficiency"],
            "halted_by": metrics["halted_by"],
        }
    fixed = rows["c19_fixed_recurrent_fixed6"]
    fixed_mut = rows["c19_c_rho_mut_recurrent_fixed6"]
    halt = rows["c19_c_rho_mut_recurrent_halting6"]
    restart = rows["c19_c_rho_mut_recurrent_halting_restart6"]
    return {
        "schema_version": "e7a_halting_audit_v1",
        "systems": rows,
        "halting_reduced_steps_without_accuracy_loss": bool(halt["mean_steps"] < fixed["mean_steps"] and halt["macro_family_accuracy"] >= fixed["macro_family_accuracy"] - 0.02),
        "restart_reduced_steps_without_accuracy_loss": bool(restart["mean_steps"] < fixed["mean_steps"] and restart["macro_family_accuracy"] >= fixed["macro_family_accuracy"] - 0.02),
        "mutable_c_rho_recurrent_beats_fixed_recurrent": bool(fixed_mut["macro_family_accuracy"] > fixed["macro_family_accuracy"] + 0.02),
    }


def c19_parameter_mode_audit(aggregate: dict[str, Any]) -> dict[str, Any]:
    matrix_systems = (
        "c19_fixed_matrix_depth3",
        "c19_rho0_matrix_depth3",
        "c19_c_mut_matrix_depth3",
        "c19_rho_mut_matrix_depth3",
        "c19_c_rho_mut_matrix_depth3",
    )
    recurrent_systems = (
        "c19_fixed_recurrent_fixed6",
        "c19_c_rho_mut_recurrent_fixed6",
        "c19_fixed_recurrent_halting6",
        "c19_c_rho_mut_recurrent_halting6",
        "c19_c_rho_mut_recurrent_halting_restart6",
    )
    systems = aggregate["systems"]
    matrix_rows = {
        system: {
            "c19_mode": systems[system]["spec"]["c19_mode"],
            "heldout_macro_family_accuracy": systems[system]["heldout_macro_family_accuracy"],
            "ood_macro_family_accuracy": systems[system]["ood_macro_family_accuracy"],
            "adversarial_macro_family_accuracy": systems[system]["adversarial_macro_family_accuracy"],
            "parameter_count": systems[system]["parameter_count"],
        }
        for system in matrix_systems
    }
    recurrent_rows = {
        system: {
            "c19_mode": systems[system]["spec"]["c19_mode"],
            "heldout_macro_family_accuracy": systems[system]["heldout_macro_family_accuracy"],
            "heldout_mean_steps": systems[system]["heldout_mean_steps"],
            "heldout_halt_efficiency": systems[system]["heldout_halt_efficiency"],
            "parameter_count": systems[system]["parameter_count"],
        }
        for system in recurrent_systems
    }
    best_matrix = max(matrix_systems, key=lambda system: systems[system]["heldout_macro_family_accuracy"])
    best_recurrent = max(recurrent_systems, key=lambda system: systems[system]["heldout_macro_family_accuracy"])
    return {
        "schema_version": "e7a_c19_parameter_mode_audit_v1",
        "matrix_depth3_systems": matrix_rows,
        "recurrent_systems": recurrent_rows,
        "best_c19_matrix_mode_system": best_matrix,
        "best_c19_recurrent_mode_system": best_recurrent,
        "rho_off_beats_fixed_on_heldout": bool(systems["c19_rho0_matrix_depth3"]["heldout_macro_family_accuracy"] > systems["c19_fixed_matrix_depth3"]["heldout_macro_family_accuracy"] + 0.02),
        "c_mut_beats_fixed_on_heldout": bool(systems["c19_c_mut_matrix_depth3"]["heldout_macro_family_accuracy"] > systems["c19_fixed_matrix_depth3"]["heldout_macro_family_accuracy"] + 0.02),
        "rho_mut_beats_fixed_on_heldout": bool(systems["c19_rho_mut_matrix_depth3"]["heldout_macro_family_accuracy"] > systems["c19_fixed_matrix_depth3"]["heldout_macro_family_accuracy"] + 0.02),
        "c_rho_mut_beats_fixed_on_heldout": bool(systems["c19_c_rho_mut_matrix_depth3"]["heldout_macro_family_accuracy"] > systems["c19_fixed_matrix_depth3"]["heldout_macro_family_accuracy"] + 0.02),
    }


def aggregate_metrics(searches: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    systems = {system: system_metrics(searches[system]) for system in SYSTEMS}
    best = max(SYSTEMS, key=lambda system: systems[system]["heldout_macro_family_accuracy"])
    return {
        "schema_version": "e7a_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "observational_only": True,
        "systems": systems,
        "best_heldout_macro_family_system": best,
        "deterministic_replay_passed": deterministic["internal_replay_passed"],
    }


def primitive_scan_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a_primitive_scan_report_v1",
        "milestone": MILESTONE,
        "scope": "lowest_abstraction_layer_only_no_final_e7_verdict",
        "systems": aggregate["systems"],
        "best_heldout_macro_family_system": aggregate["best_heldout_macro_family_system"],
    }


def decision_artifact(aggregate: dict[str, Any], collapse: dict[str, Any], c19_audit: dict[str, Any], halting: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a_decision_v1",
        "milestone": MILESTONE,
        "decision": "e7a_observational_primitive_scan_complete",
        "final_e7_verdict_intentionally_deferred": True,
        "best_heldout_macro_family_system": aggregate["best_heldout_macro_family_system"],
        "best_c19_matrix_mode_system": c19_audit["best_c19_matrix_mode_system"],
        "best_c19_recurrent_mode_system": c19_audit["best_c19_recurrent_mode_system"],
        "linear_collapse_audit_passed": collapse["linear_depth_systems_are_functionally_collapsible"],
        "halting_reduced_steps_without_accuracy_loss": halting["halting_reduced_steps_without_accuracy_loss"],
        "restart_reduced_steps_without_accuracy_loss": halting["restart_reduced_steps_without_accuracy_loss"],
        "deterministic_replay_passed": aggregate["deterministic_replay_passed"],
        "next": "E7B_MATRIX_MEDIUM_ROUTING_PROXY_IF_PRIMITIVES_ARE_INTERESTING",
    }


def row_samples(searches: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "schema_version": f"e7a_row_level_eval_sample_{split}_v1",
        "split": split,
        "samples": {system: searches[system]["final_eval"]["evals"][split]["row_level_samples"] for system in SYSTEMS},
    }


def backend_manifest(settings: Settings, git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a_backend_manifest_v1",
        "milestone": MILESTONE,
        "systems": list(SYSTEMS),
        "mutation_systems": list(MUTATION_SYSTEMS),
        "matrix_medium_primitive_scan": True,
        "row_level_eval_used": True,
        "settings": settings.__dict__,
        "git_preflight": git,
    }


def task_report(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a_task_report_v1",
        "milestone": MILESTONE,
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "class_count": CLASS_COUNT,
        "families": list(FAMILIES),
        "splits": {split: {"row_count": len(data["rows"])} for split, data in task.items()},
    }


def deterministic_stub(passed: bool, comparisons: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a_deterministic_replay_report_v1",
        "internal_replay_executed": True,
        "internal_replay_passed": passed,
        "deterministic_replay_passed": passed,
        "hash_artifacts": list(HASH_ARTIFACTS),
        "hash_comparisons": comparisons,
    }


def strip_search(search: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in search.items() if not key.startswith("_") and key != "runtime_seconds"}


def summary(decision: dict[str, Any], aggregate: dict[str, Any], git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "best_heldout_macro_family_system": decision["best_heldout_macro_family_system"],
        "final_e7_verdict_intentionally_deferred": True,
        "deterministic_replay_passed": decision["deterministic_replay_passed"],
        "git_status": git["git_status"],
    }


def report_md(decision: dict[str, Any], aggregate: dict[str, Any], halting: dict[str, Any]) -> str:
    lines = [
        f"# {MILESTONE} Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_heldout_macro_family_system = {decision['best_heldout_macro_family_system']}",
        f"best_c19_matrix_mode_system = {decision['best_c19_matrix_mode_system']}",
        f"best_c19_recurrent_mode_system = {decision['best_c19_recurrent_mode_system']}",
        "final_e7_verdict = deferred",
        "```",
        "",
        "## Primitive Systems",
        "",
    ]
    for system, metrics in aggregate["systems"].items():
        lines.append(
            f"- {system}: heldout_macro={metrics['heldout_macro_family_accuracy']} "
            f"ood={metrics['ood_macro_family_accuracy']} steps={metrics['heldout_mean_steps']}"
        )
    lines.extend(
        [
            "",
            "## Halting",
            "",
            f"- halting_reduced_steps_without_accuracy_loss: {halting['halting_reduced_steps_without_accuracy_loss']}",
            f"- restart_reduced_steps_without_accuracy_loss: {halting['restart_reduced_steps_without_accuracy_loss']}",
            "",
            "## Boundary",
            "",
            "E7A is a primitive matrix-medium scan only. It is not a final E7 routing result and makes no AGI, consciousness, natural-language, or model-scale claims.",
            "",
        ]
    )
    return "\n".join(lines)


def compose_artifacts(core: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    searches = core["searches"]
    task = core["task"]
    aggregate = aggregate_metrics(searches, deterministic)
    collapse = collapse_audit(searches, task)
    c19_audit = c19_parameter_mode_audit(aggregate)
    halting = halting_audit(searches)
    decision = decision_artifact(aggregate, collapse, c19_audit, halting)
    artifacts: dict[str, Any] = {
        "e7a_backend_manifest.json": backend_manifest(core["settings"], core["git"]),
        "e7a_task_report.json": task_report(task),
        "e7a_primitive_scan_report.json": primitive_scan_report(aggregate),
        "e7a_collapse_audit.json": collapse,
        "e7a_c19_parameter_mode_audit.json": c19_audit,
        "e7a_halting_audit.json": halting,
        "e7a_deterministic_replay_report.json": deterministic,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": summary(decision, aggregate, core["git"]),
        "report.md": report_md(decision, aggregate, halting),
    }
    for system in SYSTEMS:
        artifacts[f"e7a_candidate_{system}_summary.json"] = strip_search(searches[system])
        if system in MUTATION_SYSTEMS:
            artifacts[f"e7a_parameter_diff_{system}.json"] = searches[system]["final_state"]
            artifacts[f"e7a_mutation_history_{system}.json"] = mutation_history_artifact(
                system,
                searches[system]["mutation_attempt_count"],
                searches[system]["accepted_mutation_count"],
                searches[system]["rejected_mutation_count"],
                searches[system]["rollback_count"],
                searches[system]["history"],
            )
    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        artifacts[f"e7a_row_level_eval_sample_{split}.json"] = row_samples(searches, split)
    return artifacts


def write_artifacts(out: Path, core: dict[str, Any], deterministic: dict[str, Any]) -> None:
    artifacts = compose_artifacts(core, deterministic)
    for name, payload in artifacts.items():
        if isinstance(payload, str):
            e2.write_text(out / name, payload)
        else:
            e2.write_json(out / name, payload)
    e2.append_progress(out, "final_artifacts_written", artifact_count=len(artifacts))


def compare_core(primary: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    primary_artifacts = compose_artifacts(primary, deterministic_stub(True, {}))
    replay_artifacts = compose_artifacts(replay, deterministic_stub(True, {}))
    comparisons = {}
    for name in HASH_ARTIFACTS:
        primary_hash = e2.payload_sha256(primary_artifacts[name])
        replay_hash = e2.payload_sha256(replay_artifacts[name])
        comparisons[name] = {"primary_hash": primary_hash, "replay_hash": replay_hash, "match": primary_hash == replay_hash}
    return deterministic_stub(all(row["match"] for row in comparisons.values()), comparisons)


def run_core(settings: Settings, out: Path | None = None) -> dict[str, Any]:
    task = generate_task(settings)
    git = e2.git_preflight()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        e2.append_progress(out, "startup", milestone=MILESTONE, settings=settings.__dict__, systems=list(SYSTEMS))
    searches = run_systems_parallel(task, settings, out) if settings.execution_mode == "parallel" else run_systems_serial(task, settings, out)
    return {"settings": settings, "task": task, "git": git, "searches": searches}


def build_settings(args: argparse.Namespace) -> Settings:
    return Settings(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=512)
    parser.add_argument("--validation-rows-per-seed", type=int, default=256)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=256)
    parser.add_argument("--ood-rows-per-seed", type=int, default=256)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=256)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=256)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--elite-count", type=int, default=6)
    parser.add_argument("--mutation-sigma", type=float, default=0.08)
    parser.add_argument("--execution-mode", default="serial", choices=("serial", "parallel"))
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = build_settings(args)
    out = resolve_out(args.out)
    core = run_core(settings, out)
    replay = run_core(settings, out / "_internal_replay")
    deterministic = compare_core(core, replay)
    write_artifacts(out, core, deterministic)
    decision = compose_artifacts(core, deterministic)["decision.json"]
    print(json.dumps({"decision": decision["decision"], "best": decision["best_heldout_macro_family_system"], "replay": decision["deterministic_replay_passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
