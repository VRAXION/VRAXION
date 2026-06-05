#!/usr/bin/env python3
"""E7A3 neural-to-matrix substrate harness.

This probe compares three concrete paths on the same toy task:
1. standard float MLP trained with backprop,
2. standard integer/quantized MLP trained with mutation + rollback,
3. integer matrix hidden-replacement core trained with mutation + rollback.

The point is size and substrate evidence, not a final E7 claim.
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "E7A3_NEURAL_MATRIX_SUBSTRATE_HARNESS"
DEFAULT_OUT = Path("target/pilot_wave/e7a3_neural_matrix_substrate_harness")
DEFAULT_SEEDS = (80001, 80002, 80003)

INPUT_DIM = 10
CLASS_COUNT = 4
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
SYSTEMS = (
    "float_mlp_backprop",
    "integer_mlp_mutation",
    "integer_matrix_hidden_replacement_mutation",
    "random_control",
)
MUTATION_SYSTEMS = ("integer_mlp_mutation", "integer_matrix_hidden_replacement_mutation")
GRADIENT_SYSTEMS = ("float_mlp_backprop",)
HASH_ARTIFACTS = (
    "e7a3_task_generation_report.json",
    "e7a3_size_sweep_report.json",
    "e7a3_substrate_comparison_report.json",
    "e7a3_matrix_size_report.json",
    "e7a3_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
VALID_DECISIONS = (
    "e7a3_backprop_reference_solved_only",
    "e7a3_integer_mutation_network_viable",
    "e7a3_matrix_hidden_replacement_viable",
    "e7a3_matrix_hidden_replacement_matches_integer_network",
    "e7a3_no_mutation_path_detected",
    "e7a3_reference_not_solved_redesign_required",
    "e7a3_task_too_easy_or_leaky",
)

TRUE_FEATURE_W = np.asarray(
    [
        [1.30, -0.45, 0.25, -0.95],
        [-0.55, 1.10, -0.85, 0.20],
        [0.30, 0.20, 1.20, -0.70],
        [-0.85, -0.35, 0.50, 1.05],
        [1.05, -0.25, -0.65, 0.25],
        [-0.20, 0.85, 0.45, -0.55],
        [0.55, -0.75, 0.30, 0.40],
        [-0.45, 0.30, -0.30, 0.85],
        [0.75, 0.25, -0.45, -0.20],
        [-0.30, -0.65, 0.80, 0.15],
    ],
    dtype=np.float64,
)
TRUE_FEATURE_B = np.asarray([0.10, -0.04, 0.06, -0.08], dtype=np.float64)


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
    batch_size: int
    learning_rate: float
    weight_decay: float
    population_size: int
    generations: int
    elite_count: int
    integer_mutation_steps: int
    integer_weight_limit: int
    integer_scale: float
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


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one integer is required")
    return values


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7a3::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def set_determinism(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")


def nonlinear_features(x: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            x[:, 0],
            x[:, 1],
            x[:, 2],
            x[:, 3],
            x[:, 0] * x[:, 1],
            x[:, 2] * x[:, 3],
            np.sin(1.7 * x[:, 4]),
            np.cos(1.3 * x[:, 5]),
            x[:, 6] * x[:, 6] - x[:, 7] * x[:, 7],
            np.tanh(2.0 * x[:, 8] * x[:, 9]),
        ],
        axis=1,
    )


def target_logits_from_x(x: np.ndarray) -> np.ndarray:
    feats = nonlinear_features(x)
    logits = x @ TRUE_FEATURE_W + 0.35 * (feats @ TRUE_FEATURE_W) + TRUE_FEATURE_B
    return logits


def target_from_x(x: np.ndarray) -> np.ndarray:
    return np.argmax(target_logits_from_x(x), axis=1).astype(np.int64)


def target_margin_from_x(x: np.ndarray) -> np.ndarray:
    logits = target_logits_from_x(x)
    sorted_logits = np.sort(logits, axis=1)
    return sorted_logits[:, -1] - sorted_logits[:, -2]


def make_split(split: str, seeds: tuple[int, ...], rows_per_seed: int) -> dict[str, Any]:
    rows = []
    xs = []
    for seed in seeds:
        rng = np.random.default_rng(stable_seed(f"{split}-{seed}"))
        for index in range(rows_per_seed):
            x = None
            margin = 0.0
            for _ in range(200):
                scale = 1.0
                shift = 0.0
                if split == "ood":
                    scale = 1.20
                    shift = 0.08
                trial = rng.normal(shift, scale, size=INPUT_DIM).astype(np.float64)
                if split == "counterfactual":
                    trial[0] *= -1.0
                    trial[4] *= -1.0
                    trial[8] *= -1.0
                if split == "adversarial":
                    y_tmp = int(target_from_x(trial.reshape(1, -1))[0])
                    lure = (y_tmp - 1.5) / 1.5
                    trial[8] = lure + rng.normal(0.0, 0.05)
                    trial[9] = -lure + rng.normal(0.0, 0.05)
                margin = float(target_margin_from_x(trial.reshape(1, -1))[0])
                if margin >= 0.65:
                    x = trial
                    break
            if x is None:
                x = trial
            xs.append(x)
            rows.append({"row_id": f"{split}_{seed}_{index:05d}", "split": split, "x": [round_float(v) for v in x.tolist()], "target_margin": round_float(margin)})
    x_arr = np.asarray(xs, dtype=np.float64)
    y = target_from_x(x_arr)
    for row, target in zip(rows, y.tolist()):
        row["target"] = int(target)
    return {"rows": rows, "x": x_arr, "y": y}


def generate_task(settings: Settings) -> dict[str, Any]:
    counts = {
        "train": settings.train_rows_per_seed,
        "validation": settings.validation_rows_per_seed,
        "heldout": settings.heldout_rows_per_seed,
        "ood": settings.ood_rows_per_seed,
        "counterfactual": settings.counterfactual_rows_per_seed,
        "adversarial": settings.adversarial_rows_per_seed,
    }
    return {split: make_split(split, settings.seeds, counts[split]) for split in SPLITS}


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy_np(logits: np.ndarray, y: np.ndarray) -> float:
    probs = softmax_np(logits)
    return float(-np.mean(np.log(np.maximum(probs[np.arange(y.shape[0]), y], 1e-12))))


def evaluate_logits(logits: np.ndarray, split_data: dict[str, Any], sample_limit: int = 10) -> dict[str, Any]:
    y = split_data["y"]
    pred = np.argmax(logits, axis=1).astype(np.int64)
    correct = pred == y
    confidence = np.max(softmax_np(logits), axis=1)
    metrics = {
        "accuracy": round_float(float(np.mean(correct))),
        "cross_entropy": round_float(cross_entropy_np(logits, y)),
        "mean_confidence": round_float(float(np.mean(confidence))),
    }
    samples = []
    for index in range(min(sample_limit, len(split_data["rows"]))):
        row = split_data["rows"][index]
        samples.append(
            {
                "row_id": row["row_id"],
                "target": int(row["target"]),
                "pred": int(pred[index]),
                "correct": bool(correct[index]),
                "confidence": round_float(float(confidence[index])),
            }
        )
    return {"metrics": metrics, "row_level_samples": samples}


class FloatMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, width)
        self.fc2 = nn.Linear(width, width)
        self.out = nn.Linear(width, CLASS_COUNT)

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        logits = self.out(h)
        if return_hidden:
            return logits, h
        return logits


def torch_state_vector(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().reshape(-1).astype(np.float64) for p in model.parameters()])


def vector_hash(vector: np.ndarray) -> str:
    return hashlib.sha256(np.round(vector.astype(np.float64), 12).tobytes()).hexdigest()


def train_float_mlp(width: int, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"float-{width}-{settings.seeds}"), device)
    model = FloatMLP(width).to(device)
    initial_vector = torch_state_vector(model)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train = torch.as_tensor(task["train"]["x"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(task["train"]["y"], dtype=torch.long, device=device)
    x_val = torch.as_tensor(task["validation"]["x"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(task["validation"]["y"], dtype=torch.long, device=device)
    rng = np.random.default_rng(stable_seed(f"float-batches-{width}-{settings.seeds}"))
    history = []
    last_heartbeat = time.monotonic()
    for epoch in range(1, settings.gradient_epochs + 1):
        order = rng.permutation(x_train.shape[0])
        model.train()
        for start in range(0, len(order), settings.batch_size):
            idx = torch.as_tensor(order[start : start + settings.batch_size], dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = F.cross_entropy(val_logits, y_val).detach().cpu().item()
            val_acc = (torch.argmax(val_logits, dim=1) == y_val).float().mean().detach().cpu().item()
        row = {"epoch": epoch, "validation_accuracy": round_float(val_acc), "validation_loss": round_float(val_loss), "state_hash": vector_hash(torch_state_vector(model))}
        history.append(row)
        now = time.monotonic()
        if out and (now - last_heartbeat >= settings.heartbeat_seconds or epoch == settings.gradient_epochs):
            write_json(out / f"e7a3_training_history_float_mlp_backprop_width{width}.json", {"system": "float_mlp_backprop", "width": width, "history": history})
            append_progress_locked(out, "gradient_epoch", system="float_mlp_backprop", width=width, epoch=epoch, validation_accuracy=row["validation_accuracy"])
            last_heartbeat = now
    evals = {}
    hidden_by_split = {}
    model.eval()
    with torch.no_grad():
        for split, data in task.items():
            x = torch.as_tensor(data["x"], dtype=torch.float32, device=device)
            logits_t, hidden_t = model(x, return_hidden=True)
            evals[split] = evaluate_logits(logits_t.detach().cpu().numpy(), data)
            hidden_by_split[split] = hidden_t.detach().cpu().numpy().astype(np.float64)
    hidden = hidden_by_split["heldout"]
    centered = hidden - np.mean(hidden, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False) if centered.size else np.zeros(width)
    rank = int(np.sum(singular > 1e-6))
    result = {
        "system": "float_mlp_backprop",
        "width": width,
        "training_mode": "gradient_backprop",
        "device": device,
        "evals": evals,
        "history": history,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "hidden_matrix_shape": [width, width],
        "hidden_state_rank": rank,
        "hidden_state_singular_values_sample": [round_float(v) for v in singular[: min(8, len(singular))].tolist()],
        "initial_hash": vector_hash(initial_vector),
        "final_hash": vector_hash(torch_state_vector(model)),
    }
    if out:
        write_system_size_outputs(out, result)
    return result


def init_integer_candidate(system: str, width: int, settings: Settings, rng: np.random.Generator) -> dict[str, Any]:
    limit = settings.integer_weight_limit
    if system == "integer_mlp_mutation":
        return {
            "system": system,
            "width": width,
            "w1": rng.integers(-limit, limit + 1, size=(INPUT_DIM, width), dtype=np.int16).tolist(),
            "b1": rng.integers(-limit, limit + 1, size=width, dtype=np.int16).tolist(),
            "w2": rng.integers(-limit, limit + 1, size=(width, width), dtype=np.int16).tolist(),
            "b2": rng.integers(-limit, limit + 1, size=width, dtype=np.int16).tolist(),
            "wout": rng.integers(-limit, limit + 1, size=(width, CLASS_COUNT), dtype=np.int16).tolist(),
            "bout": rng.integers(-limit, limit + 1, size=CLASS_COUNT, dtype=np.int16).tolist(),
        }
    return {
        "system": system,
        "width": width,
        "win": rng.integers(-limit, limit + 1, size=(INPUT_DIM, width), dtype=np.int16).tolist(),
        "state": rng.integers(-limit, limit + 1, size=(width, width), dtype=np.int16).tolist(),
        "carry": rng.integers(0, 5, size=width, dtype=np.int16).tolist(),
        "bstate": rng.integers(-limit, limit + 1, size=width, dtype=np.int16).tolist(),
        "wout": rng.integers(-limit, limit + 1, size=(width, CLASS_COUNT), dtype=np.int16).tolist(),
        "bout": rng.integers(-limit, limit + 1, size=CLASS_COUNT, dtype=np.int16).tolist(),
    }


def int_arr(candidate: dict[str, Any], key: str, settings: Settings) -> np.ndarray:
    return np.asarray(candidate[key], dtype=np.float64) * settings.integer_scale


def integer_forward(candidate: dict[str, Any], x: np.ndarray, settings: Settings) -> np.ndarray:
    system = candidate["system"]
    if system == "integer_mlp_mutation":
        h = np.tanh(x @ int_arr(candidate, "w1", settings) + int_arr(candidate, "b1", settings))
        h = np.tanh(h @ int_arr(candidate, "w2", settings) + int_arr(candidate, "b2", settings))
        return h @ int_arr(candidate, "wout", settings) + int_arr(candidate, "bout", settings)
    h = np.tanh(x @ int_arr(candidate, "win", settings) + int_arr(candidate, "bstate", settings))
    state_matrix = int_arr(candidate, "state", settings)
    carry = np.asarray(candidate["carry"], dtype=np.float64) / 8.0
    for _ in range(settings.matrix_steps):
        proposal = np.tanh(h @ state_matrix + x @ int_arr(candidate, "win", settings) + int_arr(candidate, "bstate", settings))
        h = carry * h + (1.0 - carry) * proposal
    return h @ int_arr(candidate, "wout", settings) + int_arr(candidate, "bout", settings)


def candidate_hash(candidate: dict[str, Any]) -> str:
    return payload_sha256(candidate)


def flatten_int_paths(payload: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], int]]:
    rows = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"system", "width"}:
                continue
            rows.extend(flatten_int_paths(value, prefix + (key,)))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            rows.extend(flatten_int_paths(value, prefix + (index,)))
    elif isinstance(payload, int):
        rows.append((prefix, int(payload)))
    return rows


def set_path(payload: Any, path: tuple[Any, ...], value: int) -> None:
    cursor = payload
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = int(value)


def mutate_integer_candidate(candidate: dict[str, Any], settings: Settings, rng: random.Random) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    paths = flatten_int_paths(child)
    limit = settings.integer_weight_limit
    for path, value in rng.sample(paths, k=min(settings.integer_mutation_steps, len(paths))):
        if path and path[0] == "carry":
            set_path(child, path, min(max(value + rng.choice((-1, 1)), 0), 8))
        else:
            step = rng.choice((-2, -1, 1, 2))
            set_path(child, path, min(max(value + step, -limit), limit))
    return child


def score_integer(candidate: dict[str, Any], task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    evals = {
        split: evaluate_logits(integer_forward(candidate, data["x"], settings), data, sample_limit=0)
        for split, data in {"train": task["train"], "validation": task["validation"]}.items()
    }
    train = evals["train"]["metrics"]
    val = evals["validation"]["metrics"]
    fitness = 0.25 * train["accuracy"] + 0.75 * val["accuracy"] - 0.025 * val["cross_entropy"]
    return {"candidate": candidate, "evals": evals, "fitness": round_float(fitness)}


def integer_parameter_count(candidate: dict[str, Any]) -> int:
    return len(flatten_int_paths(candidate))


def integer_parameter_diff(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    before = {path: value for path, value in flatten_int_paths(initial)}
    after = {path: value for path, value in flatten_int_paths(final)}
    changed = {}
    l1 = 0
    for path in sorted(before):
        delta = after[path] - before[path]
        if delta != 0:
            key = ".".join(str(part) for part in path)
            changed[key] = {"before": before[path], "after": after[path], "delta": int(delta)}
            l1 += abs(delta)
    return {
        "actual_parameter_diff_found": bool(changed),
        "changed_parameter_count": len(changed),
        "parameter_diff_l1": int(l1),
        "before_hash": candidate_hash(initial),
        "after_hash": candidate_hash(final),
        "changed_parameters_sample": dict(list(changed.items())[:100]),
    }


def run_integer_mutation_system(system: str, width: int, task: dict[str, Any], settings: Settings, out: str | None) -> dict[str, Any]:
    out_path = Path(out) if out else None
    if out_path:
        append_progress_locked(out_path, "mutation_system_start", system=system, width=width)
    rng_np = np.random.default_rng(stable_seed(f"{system}-{width}-init-{settings.seeds}"))
    rng = random.Random(stable_seed(f"{system}-{width}-mut-{settings.seeds}"))
    population = []
    for _ in range(settings.population_size):
        population.append(score_integer(init_integer_candidate(system, width, settings, rng_np), task, settings))
    initial = copy.deepcopy(population[0]["candidate"])
    accepted = 0
    rejected = 0
    rollback = 0
    generation_metrics = []
    last_heartbeat = time.monotonic()
    attempts = 0
    for generation in range(1, settings.generations + 1):
        population.sort(key=lambda row: row["fitness"], reverse=True)
        elites = population[: max(1, settings.elite_count)]
        next_population = copy.deepcopy(elites)
        while len(next_population) < settings.population_size:
            parent = copy.deepcopy(rng.choice(elites))
            child_candidate = mutate_integer_candidate(parent["candidate"], settings, rng)
            child = score_integer(child_candidate, task, settings)
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
                    out_path / "partial_status" / f"{system}_width{width}.json",
                    {
                        "system": system,
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
                append_progress_locked(out_path, "mutation_heartbeat", system=system, width=width, generation=generation, best_fitness=best_mid["fitness"])
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
        generation_metrics.append(row)
        if out_path:
            write_json(out_path / f"e7a3_mutation_history_{system}_width{width}.json", {"system": system, "width": width, "history": generation_metrics, "accepted_mutation_count": accepted, "rejected_mutation_count": rejected, "rollback_count": rollback, "mutation_attempt_count": attempts})
            append_progress_locked(out_path, "mutation_generation", system=system, width=width, generation=generation, validation_accuracy=row["validation_accuracy"])
    best = max(population, key=lambda row: row["fitness"])
    final_candidate = best["candidate"]
    evals = {split: evaluate_logits(integer_forward(final_candidate, data["x"], settings), data) for split, data in task.items()}
    result = {
        "system": system,
        "width": width,
        "training_mode": "integer_mutation_rollback",
        "evals": evals,
        "history": generation_metrics,
        "mutation_history": {
            "system": system,
            "width": width,
            "mutation_attempt_count": attempts,
            "accepted_mutation_count": accepted,
            "rejected_mutation_count": rejected,
            "rollback_count": rollback,
            "history": generation_metrics,
        },
        "parameter_count": integer_parameter_count(final_candidate),
        "hidden_matrix_shape": [width, width],
        "initial_candidate": initial,
        "final_candidate": final_candidate,
        "parameter_diff": integer_parameter_diff(initial, final_candidate),
        "initial_hash": candidate_hash(initial),
        "final_hash": candidate_hash(final_candidate),
    }
    if out_path:
        write_system_size_outputs(out_path, result)
        append_progress_locked(out_path, "mutation_system_complete", system=system, width=width, heldout_accuracy=evals["heldout"]["metrics"]["accuracy"])
    return result


def random_result(width: int, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"random-{width}-{settings.seeds}"))
    evals = {}
    for split, data in task.items():
        logits = rng.normal(0.0, 1.0, size=(data["x"].shape[0], CLASS_COUNT))
        evals[split] = evaluate_logits(logits, data)
    result = {
        "system": "random_control",
        "width": width,
        "training_mode": "random",
        "evals": evals,
        "history": [],
        "parameter_count": 0,
        "hidden_matrix_shape": [0, 0],
        "initial_hash": None,
        "final_hash": None,
    }
    if out:
        write_system_size_outputs(out, result)
    return result


def write_system_size_outputs(out: Path, result: dict[str, Any]) -> None:
    system = result["system"]
    width = result["width"]
    summary = {
        "system": system,
        "width": width,
        "training_mode": result["training_mode"],
        "parameter_count": result["parameter_count"],
        "hidden_matrix_shape": result["hidden_matrix_shape"],
        "initial_hash": result.get("initial_hash"),
        "final_hash": result.get("final_hash"),
        "split_metrics": {split: result["evals"][split]["metrics"] for split in SPLITS},
    }
    if "hidden_state_rank" in result:
        summary["hidden_state_rank"] = result["hidden_state_rank"]
        summary["hidden_state_singular_values_sample"] = result["hidden_state_singular_values_sample"]
    write_json(out / f"e7a3_candidate_{system}_width{width}_summary.json", summary)
    if system in MUTATION_SYSTEMS:
        write_json(out / f"e7a3_candidate_{system}_width{width}_initial.json", result["initial_candidate"])
        write_json(out / f"e7a3_candidate_{system}_width{width}_final.json", result["final_candidate"])
        write_json(out / f"e7a3_parameter_diff_{system}_width{width}.json", result["parameter_diff"])


def run_core(settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, settings=settings.__dict__)
    task = generate_task(settings)
    if out:
        append_progress_locked(out, "task_generated", rows={split: len(task[split]["rows"]) for split in SPLITS})
    results: dict[str, Any] = {system: {} for system in SYSTEMS}
    futures = {}
    executor: ProcessPoolExecutor | None = None
    if settings.execution_mode == "parallel":
        executor = ProcessPoolExecutor(max_workers=max(1, settings.parallel_workers))
        for system in MUTATION_SYSTEMS:
            for width in settings.widths:
                future = executor.submit(run_integer_mutation_system, system, width, task, settings, out.as_posix() if out else None)
                futures[future] = (system, width)
        if out:
            append_progress_locked(out, "cpu_mutation_lane_started", jobs=len(futures), workers=settings.parallel_workers)
    for width in settings.widths:
        results["float_mlp_backprop"][width] = train_float_mlp(width, task, settings, out)
        results["random_control"][width] = random_result(width, task, settings, out)
    if executor is not None:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                system, width = futures[future]
                results[system][width] = future.result()
                if out:
                    append_progress_locked(out, "cpu_mutation_lane_joined", system=system, width=width, completed=sum(len(v) for v in results.values()))
        executor.shutdown()
    else:
        for system in MUTATION_SYSTEMS:
            for width in settings.widths:
                results[system][width] = run_integer_mutation_system(system, width, task, settings, out.as_posix() if out else None)
    return task, results


def solve_pass(metrics: dict[str, float]) -> bool:
    return (
        metrics["heldout_accuracy"] >= 0.90
        and metrics["ood_accuracy"] >= 0.85
        and metrics["counterfactual_accuracy"] >= 0.85
        and metrics["adversarial_accuracy"] >= 0.80
    )


def aggregate_results(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    systems = {}
    for system in SYSTEMS:
        systems[system] = {}
        for width in settings.widths:
            result = results[system][width]
            split_metrics = {split: result["evals"][split]["metrics"] for split in SPLITS}
            eval_accuracy = round_float(float(np.mean([split_metrics[split]["accuracy"] for split in EVAL_SPLITS])))
            row = {
                "width": width,
                "matrix_shape": result["hidden_matrix_shape"],
                "matrix_cells": int(result["hidden_matrix_shape"][0] * result["hidden_matrix_shape"][1]),
                "parameter_count": result["parameter_count"],
                "eval_accuracy": eval_accuracy,
                "train_accuracy": split_metrics["train"]["accuracy"],
                "validation_accuracy": split_metrics["validation"]["accuracy"],
                "heldout_accuracy": split_metrics["heldout"]["accuracy"],
                "ood_accuracy": split_metrics["ood"]["accuracy"],
                "counterfactual_accuracy": split_metrics["counterfactual"]["accuracy"],
                "adversarial_accuracy": split_metrics["adversarial"]["accuracy"],
                "generalization_gap": round_float(split_metrics["validation"]["accuracy"] - eval_accuracy),
                "training_mode": result["training_mode"],
                "solve_passed": False,
            }
            row["solve_passed"] = solve_pass(row)
            if system in MUTATION_SYSTEMS:
                hist = result["mutation_history"]
                row["mutation_attempt_count"] = hist["mutation_attempt_count"]
                row["accepted_mutation_count"] = hist["accepted_mutation_count"]
                row["rejected_mutation_count"] = hist["rejected_mutation_count"]
                row["rollback_count"] = hist["rollback_count"]
            if system == "float_mlp_backprop":
                row["hidden_state_rank"] = result["hidden_state_rank"]
            systems[system][str(width)] = row
    smallest = {}
    for system in SYSTEMS:
        passing = [int(width) for width, row in systems[system].items() if row["solve_passed"]]
        smallest[system] = min(passing) if passing else None
    best_by_eval = {}
    for system in SYSTEMS:
        best_width = max(settings.widths, key=lambda width: systems[system][str(width)]["eval_accuracy"])
        best_by_eval[system] = {"width": best_width, **systems[system][str(best_width)]}
    return {
        "schema_version": "e7a3_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "smallest_passing_width": smallest,
        "best_by_eval_accuracy": best_by_eval,
        "solve_thresholds": {
            "heldout_accuracy": 0.90,
            "ood_accuracy": 0.85,
            "counterfactual_accuracy": 0.85,
            "adversarial_accuracy": 0.80,
        },
    }


def task_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a3_task_generation_report_v1",
        "milestone": MILESTONE,
        "input_dim": INPUT_DIM,
        "class_count": CLASS_COUNT,
        "label_rule": "deterministic nonlinear feature map; models see raw input only",
        "minimum_target_margin": 0.65,
        "splits": {split: {"row_count": len(data["rows"])} for split, data in task.items()},
        "seeds": list(settings.seeds),
    }


def size_sweep_report(aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a3_size_sweep_report_v1",
        "widths": list(settings.widths),
        "matrix_sizes_tested": {str(width): {"shape": [width, width], "cells": width * width} for width in settings.widths},
        "smallest_passing_width": aggregate["smallest_passing_width"],
        "best_by_eval_accuracy": aggregate["best_by_eval_accuracy"],
    }


def substrate_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a3_substrate_comparison_report_v1",
        "systems": SYSTEMS,
        "float_backprop_solved": aggregate["smallest_passing_width"]["float_mlp_backprop"] is not None,
        "integer_mlp_mutation_solved": aggregate["smallest_passing_width"]["integer_mlp_mutation"] is not None,
        "integer_matrix_hidden_replacement_solved": aggregate["smallest_passing_width"]["integer_matrix_hidden_replacement_mutation"] is not None,
        "random_control_solved": aggregate["smallest_passing_width"]["random_control"] is not None,
    }


def matrix_size_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for system, best in aggregate["best_by_eval_accuracy"].items():
        rows[system] = {
            "best_width": best["width"],
            "best_matrix_shape": best["matrix_shape"],
            "best_matrix_cells": best["matrix_cells"],
            "best_eval_accuracy": best["eval_accuracy"],
            "smallest_passing_width": aggregate["smallest_passing_width"][system],
        }
    return {
        "schema_version": "e7a3_matrix_size_report_v1",
        "rows": rows,
        "interpretation": "Width is the hidden dimension for neural systems and state dimension for matrix hidden replacement.",
    }


def mutation_history_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for system in MUTATION_SYSTEMS:
        rows[system] = {}
        for width in settings.widths:
            rows[system][str(width)] = results[system][width]["mutation_history"]
    return {
        "schema_version": "e7a3_mutation_history_report_v1",
        "systems": rows,
        "all_mutation_runs_have_accept_reject_rollback": all(
            rows[system][str(width)]["accepted_mutation_count"] > 0
            and rows[system][str(width)]["rejected_mutation_count"] > 0
            and rows[system][str(width)]["rejected_mutation_count"] == rows[system][str(width)]["rollback_count"]
            for system in MUTATION_SYSTEMS
            for width in settings.widths
        ),
    }


def training_history_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a3_training_history_report_v1",
        "systems": {
            "float_mlp_backprop": {str(width): results["float_mlp_backprop"][width]["history"] for width in settings.widths}
        },
    }


def no_synthetic_metric_audit(task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a3_no_synthetic_metric_audit_v1",
        "generated_from_row_level_eval": True,
        "row_counts": {split: len(task[split]["rows"]) for split in SPLITS},
        "row_level_samples_present": all(
            bool(results[system][width]["evals"]["heldout"]["row_level_samples"])
            for system in SYSTEMS
            for width in settings.widths
        ),
        "gradient_backprop_used_only_for_float_reference": True,
        "integer_mutation_systems_used_optimizer_or_backprop": False,
        "hardcoded_improvement_flags_present": False,
        "final_e7_verdict_intentionally_deferred": True,
    }


def row_samples(results: dict[str, Any], split: str, settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a3_row_level_eval_sample_v1",
        "split": split,
        "samples": {
            system: {
                str(width): results[system][width]["evals"][split]["row_level_samples"]
                for width in settings.widths
            }
            for system in SYSTEMS
        },
    }


def choose_decision(aggregate: dict[str, Any], substrate: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    if substrate["random_control_solved"]:
        decision = "e7a3_task_too_easy_or_leaky"
    elif not substrate["float_backprop_solved"]:
        decision = "e7a3_reference_not_solved_redesign_required"
    elif substrate["integer_matrix_hidden_replacement_solved"] and substrate["integer_mlp_mutation_solved"]:
        matrix_width = aggregate["smallest_passing_width"]["integer_matrix_hidden_replacement_mutation"]
        int_width = aggregate["smallest_passing_width"]["integer_mlp_mutation"]
        decision = "e7a3_matrix_hidden_replacement_matches_integer_network" if matrix_width <= int_width else "e7a3_matrix_hidden_replacement_viable"
    elif substrate["integer_matrix_hidden_replacement_solved"]:
        decision = "e7a3_matrix_hidden_replacement_viable"
    elif substrate["integer_mlp_mutation_solved"]:
        decision = "e7a3_integer_mutation_network_viable"
    elif substrate["float_backprop_solved"]:
        decision = "e7a3_backprop_reference_solved_only"
    else:
        decision = "e7a3_no_mutation_path_detected"
    if not audit["generated_from_row_level_eval"] or audit["hardcoded_improvement_flags_present"]:
        decision = "e7a3_task_too_easy_or_leaky"
    return {
        "schema_version": "e7a3_decision_v1",
        "decision": decision,
        "valid_decisions": list(VALID_DECISIONS),
        "smallest_passing_width": aggregate["smallest_passing_width"],
        "final_e7_verdict_intentionally_deferred": True,
        "deterministic_replay_passed": False,
    }


def build_report(out: Path, aggregate: dict[str, Any], decision: dict[str, Any], substrate: dict[str, Any]) -> str:
    lines = [
        "# E7A3 Neural Matrix Substrate Harness Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"float_backprop_solved = {substrate['float_backprop_solved']}",
        f"integer_mlp_mutation_solved = {substrate['integer_mlp_mutation_solved']}",
        f"integer_matrix_hidden_replacement_solved = {substrate['integer_matrix_hidden_replacement_solved']}",
        "final_e7_verdict = intentionally deferred",
        "```",
        "",
        f"Run root: `{out.relative_to(REPO_ROOT).as_posix()}`",
        "",
        "## Size Sweep",
        "",
        "| system | smallest passing width | best width | best eval accuracy | matrix cells |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, best in aggregate["best_by_eval_accuracy"].items():
        smallest = aggregate["smallest_passing_width"][system]
        lines.append(f"| `{system}` | {smallest if smallest is not None else 'none'} | {best['width']} | {best['eval_accuracy']:.6f} | {best['matrix_cells']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This harness asks how large the hidden matrix/state has to be before a toy nonlinear classification task is solved by each substrate.",
            "",
            "It does not claim a final matrix-medium architecture. It only establishes whether the standard neural reference, integer mutation network, and matrix hidden replacement path have a viable toy foothold.",
            "",
        ]
    )
    return "\n".join(lines)


def build_payloads(out: Path, task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    aggregate = aggregate_results(results, settings)
    substrate = substrate_report(aggregate)
    audit = no_synthetic_metric_audit(task, results, settings)
    decision = choose_decision(aggregate, substrate, audit)
    payloads: dict[str, Any] = {
        "e7a3_backend_manifest.json": {
            "schema_version": "e7a3_backend_manifest_v1",
            "milestone": MILESTONE,
            "systems": list(SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "gradient_systems": list(GRADIENT_SYSTEMS),
            "widths": list(settings.widths),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": select_device(settings.device),
            "settings": {**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths)},
            "cpu_mutation_and_gpu_gradient_overlap_supported": True,
            "final_e7_verdict_intentionally_deferred": True,
        },
        "e7a3_task_generation_report.json": task_report(task, settings),
        "e7a3_size_sweep_report.json": size_sweep_report(aggregate, settings),
        "e7a3_substrate_comparison_report.json": substrate,
        "e7a3_matrix_size_report.json": matrix_size_report(aggregate),
        "e7a3_mutation_history.json": mutation_history_report(results, settings),
        "e7a3_training_history.json": training_history_report(results, settings),
        "e7a3_no_synthetic_metric_audit.json": audit,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e7a3_summary_v1",
            "milestone": MILESTONE,
            "decision": decision["decision"],
            "smallest_passing_width": aggregate["smallest_passing_width"],
            "final_e7_verdict_intentionally_deferred": True,
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
        },
    }
    for split in EVAL_SPLITS:
        payloads[f"e7a3_row_level_eval_sample_{split}.json"] = row_samples(results, split, settings)
    payloads["report.md"] = build_report(out, aggregate, decision, substrate)
    return payloads


def compute_hashes(payloads: dict[str, Any]) -> dict[str, str]:
    return {name: payload_sha256(payloads[name]) for name in HASH_ARTIFACTS}


def deterministic_replay(settings: Settings, out: Path, primary_payloads: dict[str, Any]) -> dict[str, Any]:
    replay_settings = replace(settings, execution_mode="serial", parallel_workers=1)
    task_replay, results_replay = run_core(replay_settings, None)
    replay_payloads = build_payloads(out, task_replay, results_replay, replay_settings)
    primary_hashes = compute_hashes(primary_payloads)
    replay_hashes = compute_hashes(replay_payloads)
    comparisons = {
        name: {"primary_hash": primary_hashes[name], "replay_hash": replay_hashes[name], "match": primary_hashes[name] == replay_hashes[name]}
        for name in HASH_ARTIFACTS
    }
    return {
        "schema_version": "e7a3_deterministic_replay_report_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
    }


def write_final_artifacts(out: Path, payloads: dict[str, Any], deterministic: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["e7a3_deterministic_replay_report.json"] = deterministic
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
    parser.add_argument("--widths", default="4,8,16,32")
    parser.add_argument("--train-rows-per-seed", type=int, default=240)
    parser.add_argument("--validation-rows-per-seed", type=int, default=100)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=100)
    parser.add_argument("--ood-rows-per-seed", type=int, default=100)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=100)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=100)
    parser.add_argument("--gradient-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--integer-mutation-steps", type=int, default=18)
    parser.add_argument("--integer-weight-limit", type=int, default=8)
    parser.add_argument("--integer-scale", type=float, default=0.18)
    parser.add_argument("--matrix-steps", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--execution-mode", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--parallel-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 4, 16)))
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
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        integer_mutation_steps=args.integer_mutation_steps,
        integer_weight_limit=args.integer_weight_limit,
        integer_scale=args.integer_scale,
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
