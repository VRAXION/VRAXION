#!/usr/bin/env python3
"""E7A6 quantization stress and mutation repair limit probe.

E7A4 showed that an E7A-style plain matrix-core survives ordinary quantization.
E7A6 maps how far that core can be compressed before it breaks, and whether
mutation-only repair becomes useful after the break.
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
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
E7A3_PATH = Path(__file__).with_name("run_e7a3_neural_matrix_substrate_harness.py")
MILESTONE = "E7A6_QUANTIZATION_STRESS_AND_REPAIR_LIMIT"
DEFAULT_OUT = Path("target/pilot_wave/e7a6_quantization_stress_and_repair_limit")
DEFAULT_SEEDS = (80001, 80002, 80003)

QUANT_LEVELS = ("int8", "int4", "int3", "ternary", "binary")
QUANT_CONFIGS: dict[str, dict[str, Any]] = {
    "int8": {"kind": "symmetric_int", "q_limit": 127, "nominal_bits": 8},
    "int4": {"kind": "symmetric_int", "q_limit": 7, "nominal_bits": 4},
    "int3": {"kind": "symmetric_int", "q_limit": 3, "nominal_bits": 3},
    "ternary": {"kind": "ternary", "q_limit": 1, "nominal_bits": 2},
    "binary": {"kind": "binary", "q_limit": 1, "nominal_bits": 1},
}
SYSTEMS = ("float32_matrix_core", "quantized_no_repair", "quantized_mutation_repair", "random_control")
HASH_ARTIFACTS = (
    "e7a6_task_generation_report.json",
    "e7a6_float_training_report.json",
    "e7a6_quantization_stress_report.json",
    "e7a6_mutation_repair_report.json",
    "e7a6_frontier_report.json",
    "e7a6_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
VALID_DECISIONS = (
    "e7a6_int8_only_stable",
    "e7a6_int4_stable_without_repair",
    "e7a6_int3_or_lower_stable_without_repair",
    "e7a6_mutation_repair_recovers_low_bit_core",
    "e7a6_mutation_repair_partial_low_bit_recovery",
    "e7a6_mutation_repair_not_useful",
    "e7a6_quantization_breakpoint_mapped",
    "e7a6_invalid_artifact_detected",
)


def load_e7a3_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7a3_neural_matrix_substrate_harness", E7A3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7A3 from {E7A3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7a3 = load_e7a3_module()


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
    return int(hashlib.sha256(f"e7a6::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def e7a3_settings(settings: Settings) -> Any:
    return e7a3.Settings(
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
        generations=settings.repair_generations,
        elite_count=settings.elite_count,
        integer_mutation_steps=settings.quant_mutation_steps,
        integer_weight_limit=127,
        integer_scale=1.0,
        matrix_steps=settings.matrix_steps,
        device=settings.device,
        execution_mode=settings.execution_mode,
        parallel_workers=settings.parallel_workers,
        heartbeat_seconds=settings.heartbeat_seconds,
    )


class FloatMatrixCore(nn.Module):
    def __init__(self, width: int, matrix_steps: int) -> None:
        super().__init__()
        self.width = width
        self.matrix_steps = matrix_steps
        scale = 1.0 / math.sqrt(max(1, width))
        self.win = nn.Parameter(torch.empty(e7a3.INPUT_DIM, width).uniform_(-scale, scale))
        self.state = nn.Parameter(torch.empty(width, width).uniform_(-scale, scale))
        self.carry_raw = nn.Parameter(torch.zeros(width))
        self.bstate = nn.Parameter(torch.zeros(width))
        self.wout = nn.Parameter(torch.empty(width, e7a3.CLASS_COUNT).uniform_(-scale, scale))
        self.bout = nn.Parameter(torch.zeros(e7a3.CLASS_COUNT))

    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        drive = x @ self.win + self.bstate
        h = torch.tanh(drive)
        carry = torch.sigmoid(self.carry_raw).unsqueeze(0)
        for _ in range(self.matrix_steps):
            proposal = torch.tanh(h @ self.state + drive)
            h = carry * h + (1.0 - carry) * proposal
        logits = h @ self.wout + self.bout
        if return_hidden:
            return logits, h
        return logits


def torch_state_vector(model: nn.Module) -> np.ndarray:
    return np.concatenate([param.detach().cpu().numpy().reshape(-1).astype(np.float64) for param in model.parameters()])


def vector_hash(vector: np.ndarray) -> str:
    return hashlib.sha256(np.round(vector.astype(np.float64), 12).tobytes()).hexdigest()


def candidate_hash(candidate: dict[str, Any]) -> str:
    return payload_sha256(candidate)


def serializable_state_dict(state_dict: dict[str, np.ndarray]) -> dict[str, Any]:
    payload = {}
    for key, value in state_dict.items():
        if value.ndim == 2:
            payload[key] = [[round_float(v) for v in row] for row in value.tolist()]
        else:
            payload[key] = [round_float(v) for v in value.tolist()]
    return payload


def task_splits() -> tuple[str, ...]:
    return e7a3.SPLITS


def eval_splits() -> tuple[str, ...]:
    return e7a3.EVAL_SPLITS


def train_float_core(width: int, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = e7a3.select_device(settings.device)
    e7a3.set_determinism(stable_seed(f"train-float-core-{width}-{settings.seeds}"), device)
    model = FloatMatrixCore(width, settings.matrix_steps).to(device)
    initial_vector = torch_state_vector(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train = torch.as_tensor(task["train"]["x"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(task["train"]["y"], dtype=torch.long, device=device)
    x_val = torch.as_tensor(task["validation"]["x"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(task["validation"]["y"], dtype=torch.long, device=device)
    rng = np.random.default_rng(stable_seed(f"batches-float-core-{width}-{settings.seeds}"))
    history = []
    last_heartbeat = time.monotonic()
    for epoch in range(1, settings.gradient_epochs + 1):
        order = rng.permutation(x_train.shape[0])
        model.train()
        for start in range(0, len(order), settings.batch_size):
            idx = torch.as_tensor(order[start : start + settings.batch_size], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            val_loss = F.cross_entropy(logits, y_val).detach().cpu().item()
            val_acc = (torch.argmax(logits, dim=1) == y_val).float().mean().detach().cpu().item()
        row = {
            "epoch": epoch,
            "validation_accuracy": round_float(val_acc),
            "validation_loss": round_float(val_loss),
            "state_hash": vector_hash(torch_state_vector(model)),
        }
        history.append(row)
        now = time.monotonic()
        if out and (now - last_heartbeat >= settings.heartbeat_seconds or epoch == settings.gradient_epochs):
            write_json(out / f"e7a6_training_history_float32_width{width}.json", {"width": width, "history": history})
            append_progress_locked(out, "gradient_epoch", width=width, epoch=epoch, validation_accuracy=row["validation_accuracy"])
            last_heartbeat = now
    evals = {}
    hidden_by_split = {}
    model.eval()
    with torch.no_grad():
        for split, data in task.items():
            x = torch.as_tensor(data["x"], dtype=torch.float32, device=device)
            logits_t, hidden_t = model(x, return_hidden=True)
            evals[split] = e7a3.evaluate_logits(logits_t.detach().cpu().numpy(), data)
            hidden_by_split[split] = hidden_t.detach().cpu().numpy().astype(np.float64)
    hidden = hidden_by_split["heldout"]
    centered = hidden - np.mean(hidden, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False) if centered.size else np.zeros(width)
    state_dict = {key: value.detach().cpu().numpy().astype(np.float64) for key, value in model.state_dict().items()}
    result = {
        "system": "float32_matrix_core",
        "width": width,
        "training_mode": "gradient_backprop_float32",
        "device": device,
        "evals": evals,
        "history": history,
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
        "hidden_matrix_shape": [width, width],
        "hidden_state_rank": int(np.sum(singular > 1e-6)),
        "hidden_state_singular_values_sample": [round_float(v) for v in singular[: min(8, len(singular))].tolist()],
        "initial_hash": vector_hash(initial_vector),
        "final_hash": vector_hash(torch_state_vector(model)),
        "state_dict": state_dict,
    }
    if out:
        write_float_summary(out, result)
        write_json(out / f"e7a6_float_state_width{width}.json", serializable_state_dict(state_dict))
    return result


def quantize_array(value: np.ndarray, level: str) -> tuple[np.ndarray, float]:
    config = QUANT_CONFIGS[level]
    if config["kind"] == "symmetric_int":
        q_limit = int(config["q_limit"])
        max_abs = float(np.max(np.abs(value))) if value.size else 0.0
        scale = max_abs / q_limit if max_abs > 0 else 1.0 / q_limit
        q = np.clip(np.rint(value / scale), -q_limit, q_limit).astype(np.int16)
        return q, scale
    nonzero = np.abs(value) > 1e-12
    if not bool(np.any(nonzero)):
        return np.zeros_like(value, dtype=np.int16), 1.0
    scale = float(np.mean(np.abs(value[nonzero])))
    if config["kind"] == "ternary":
        threshold = 0.5 * scale
        q = np.where(value > threshold, 1, np.where(value < -threshold, -1, 0)).astype(np.int16)
        return q, scale
    if config["kind"] == "binary":
        q = np.where(np.abs(value) <= 1e-12, 0, np.where(value >= 0.0, 1, -1)).astype(np.int16)
        return q, scale
    raise ValueError(f"unknown quant kind: {config['kind']}")


def quantize_state_dict(level: str, state_dict: dict[str, np.ndarray], width: int) -> dict[str, Any]:
    q: dict[str, Any] = {}
    scales: dict[str, float] = {}
    zero_counts: dict[str, int] = {}
    for key, value in state_dict.items():
        quantized, scale = quantize_array(value, level)
        q[key] = quantized.tolist()
        scales[key] = round_float(scale)
        zero_counts[key] = int(np.sum(quantized == 0))
    return {
        "schema_version": "e7a6_quantized_plain_matrix_core_candidate_v1",
        "quant_level": level,
        "quant_config": QUANT_CONFIGS[level],
        "width": width,
        "q": q,
        "scales": scales,
        "zero_counts": zero_counts,
    }


def dequant(candidate: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(candidate["q"][key], dtype=np.float64) * float(candidate["scales"][key])


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


def evaluate_quantized_candidate(candidate: dict[str, Any], task: dict[str, Any], matrix_steps: int, sample_limit: int = 10) -> dict[str, Any]:
    return {
        split: e7a3.evaluate_logits(quantized_forward(candidate, data["x"], matrix_steps), data, sample_limit)
        for split, data in task.items()
    }


def quantized_parameter_count(candidate: dict[str, Any]) -> int:
    return sum(np.asarray(value).size for value in candidate["q"].values())


def quantized_nonzero_count(candidate: dict[str, Any]) -> int:
    return sum(int(np.sum(np.asarray(value) != 0)) for value in candidate["q"].values())


def quantized_paths(candidate: dict[str, Any]) -> list[tuple[tuple[Any, ...], int]]:
    rows: list[tuple[tuple[Any, ...], int]] = []
    for key, value in candidate["q"].items():
        arr = np.asarray(value, dtype=np.int16)
        for index in np.ndindex(arr.shape):
            rows.append((("q", key, *index), int(arr[index])))
    return rows


def set_quantized_path(candidate: dict[str, Any], path: tuple[Any, ...], value: int) -> None:
    _, key, *indices = path
    cursor = candidate["q"][key]
    for index in indices[:-1]:
        cursor = cursor[index]
    cursor[indices[-1]] = int(value)


def mutate_quantized(candidate: dict[str, Any], settings: Settings, rng: random.Random) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    level = child["quant_level"]
    config = QUANT_CONFIGS[level]
    paths = quantized_paths(child)
    for path, value in rng.sample(paths, k=min(settings.quant_mutation_steps, len(paths))):
        if config["kind"] == "binary":
            if value == 0:
                new_value = rng.choice((-1, 1))
            else:
                new_value = -value
        elif config["kind"] == "ternary":
            choices = [-1, 0, 1]
            choices.remove(value)
            new_value = rng.choice(choices)
        else:
            q_limit = int(config["q_limit"])
            step = rng.choice((-3, -2, -1, 1, 2, 3))
            new_value = min(max(value + step, -q_limit), q_limit)
        set_quantized_path(child, path, new_value)
    return child


def score_quantized(candidate: dict[str, Any], task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    evals = {
        split: e7a3.evaluate_logits(quantized_forward(candidate, data["x"], settings.matrix_steps), data, sample_limit=0)
        for split, data in {"train": task["train"], "validation": task["validation"]}.items()
    }
    train = evals["train"]["metrics"]
    val = evals["validation"]["metrics"]
    nonzero_ratio = quantized_nonzero_count(candidate) / max(1, quantized_parameter_count(candidate))
    fitness = 0.20 * train["accuracy"] + 0.80 * val["accuracy"] - 0.02 * val["cross_entropy"] - 0.001 * nonzero_ratio
    return {"candidate": candidate, "evals": evals, "fitness": round_float(fitness)}


def quantized_diff(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    before = {path: value for path, value in quantized_paths(initial)}
    after = {path: value for path, value in quantized_paths(final)}
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
        "changed_parameters_sample": dict(list(changed.items())[:120]),
    }


def split_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {split: result["evals"][split]["metrics"] for split in task_splits()}


def write_float_summary(out: Path, result: dict[str, Any]) -> None:
    width = result["width"]
    write_json(
        out / f"e7a6_candidate_float32_width{width}_summary.json",
        {
            "system": result["system"],
            "width": width,
            "training_mode": result["training_mode"],
            "parameter_count": result["parameter_count"],
            "hidden_matrix_shape": result["hidden_matrix_shape"],
            "hidden_state_rank": result["hidden_state_rank"],
            "hidden_state_singular_values_sample": result["hidden_state_singular_values_sample"],
            "initial_hash": result["initial_hash"],
            "final_hash": result["final_hash"],
            "device": result["device"],
            "split_metrics": split_metrics(result),
        },
    )


def quantized_no_repair_result(level: str, width: int, candidate: dict[str, Any], task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    evals = evaluate_quantized_candidate(candidate, task, settings.matrix_steps)
    result = {
        "system": "quantized_no_repair",
        "quant_level": level,
        "width": width,
        "training_mode": "post_backprop_quantized_no_repair",
        "evals": evals,
        "history": [],
        "parameter_count": quantized_parameter_count(candidate),
        "nonzero_parameter_count": quantized_nonzero_count(candidate),
        "hidden_matrix_shape": [width, width],
        "initial_hash": candidate_hash(candidate),
        "final_hash": candidate_hash(candidate),
        "candidate": candidate,
    }
    if out:
        write_json(out / f"e7a6_candidate_{level}_width{width}.json", candidate)
        write_json(
            out / f"e7a6_candidate_{level}_no_repair_width{width}_summary.json",
            {
                "system": result["system"],
                "quant_level": level,
                "width": width,
                "training_mode": result["training_mode"],
                "parameter_count": result["parameter_count"],
                "nonzero_parameter_count": result["nonzero_parameter_count"],
                "hidden_matrix_shape": result["hidden_matrix_shape"],
                "initial_hash": result["initial_hash"],
                "final_hash": result["final_hash"],
                "split_metrics": split_metrics(result),
            },
        )
    return result


def run_quantized_repair(level: str, width: int, initial_candidate: dict[str, Any], task: dict[str, Any], settings: Settings, out: str | None) -> dict[str, Any]:
    out_path = Path(out) if out else None
    if out_path:
        append_progress_locked(out_path, "repair_start", quant_level=level, width=width)
    rng = random.Random(stable_seed(f"repair-{level}-{width}-{settings.seeds}"))
    population = [score_quantized(copy.deepcopy(initial_candidate), task, settings)]
    for _ in range(settings.population_size - 1):
        population.append(score_quantized(mutate_quantized(initial_candidate, settings, rng), task, settings))
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
            child_candidate = mutate_quantized(parent["candidate"], settings, rng)
            child = score_quantized(child_candidate, task, settings)
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
                    out_path / "partial_status" / f"repair_{level}_width{width}.json",
                    {
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
                append_progress_locked(out_path, "repair_heartbeat", quant_level=level, width=width, generation=generation, best_fitness=best_mid["fitness"])
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
                out_path / f"e7a6_mutation_history_{level}_width{width}.json",
                {"quant_level": level, "width": width, "history": history, "accepted_mutation_count": accepted, "rejected_mutation_count": rejected, "rollback_count": rollback, "mutation_attempt_count": attempts},
            )
            append_progress_locked(out_path, "repair_generation", quant_level=level, width=width, generation=generation, validation_accuracy=row["validation_accuracy"])
    best = max(population, key=lambda row: row["fitness"])
    final_candidate = best["candidate"]
    evals = evaluate_quantized_candidate(final_candidate, task, settings.matrix_steps)
    result = {
        "system": "quantized_mutation_repair",
        "quant_level": level,
        "width": width,
        "training_mode": "quantized_mutation_repair",
        "evals": evals,
        "history": history,
        "mutation_history": {
            "quant_level": level,
            "width": width,
            "mutation_attempt_count": attempts,
            "accepted_mutation_count": accepted,
            "rejected_mutation_count": rejected,
            "rollback_count": rollback,
            "history": history,
        },
        "parameter_count": quantized_parameter_count(final_candidate),
        "nonzero_parameter_count": quantized_nonzero_count(final_candidate),
        "hidden_matrix_shape": [width, width],
        "initial_candidate": initial_candidate,
        "final_candidate": final_candidate,
        "parameter_diff": quantized_diff(initial_candidate, final_candidate),
        "initial_hash": candidate_hash(initial_candidate),
        "final_hash": candidate_hash(final_candidate),
    }
    if out_path:
        write_json(out_path / f"e7a6_candidate_{level}_mutation_repair_width{width}_initial.json", initial_candidate)
        write_json(out_path / f"e7a6_candidate_{level}_mutation_repair_width{width}_final.json", final_candidate)
        write_json(out_path / f"e7a6_parameter_diff_{level}_mutation_repair_width{width}.json", result["parameter_diff"])
        write_json(
            out_path / f"e7a6_candidate_{level}_mutation_repair_width{width}_summary.json",
            {
                "system": result["system"],
                "quant_level": level,
                "width": width,
                "training_mode": result["training_mode"],
                "parameter_count": result["parameter_count"],
                "nonzero_parameter_count": result["nonzero_parameter_count"],
                "hidden_matrix_shape": result["hidden_matrix_shape"],
                "initial_hash": result["initial_hash"],
                "final_hash": result["final_hash"],
                "split_metrics": split_metrics(result),
                "mutation_history": {key: value for key, value in result["mutation_history"].items() if key != "history"},
            },
        )
        append_progress_locked(out_path, "repair_complete", quant_level=level, width=width, heldout_accuracy=evals["heldout"]["metrics"]["accuracy"])
    return result


def random_result(width: int, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"random-{width}-{settings.seeds}"))
    evals = {}
    for split, data in task.items():
        logits = rng.normal(0.0, 1.0, size=(data["x"].shape[0], e7a3.CLASS_COUNT))
        evals[split] = e7a3.evaluate_logits(logits, data)
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
        write_json(
            out / f"e7a6_candidate_random_control_width{width}_summary.json",
            {
                "system": "random_control",
                "width": width,
                "training_mode": "random",
                "parameter_count": 0,
                "hidden_matrix_shape": [0, 0],
                "split_metrics": split_metrics(result),
            },
        )
    return result


def run_core(settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, settings=settings.__dict__)
    task = e7a3.generate_task(e7a3_settings(settings))
    if out:
        append_progress_locked(out, "task_generated", rows={split: len(task[split]["rows"]) for split in task_splits()})
    results: dict[str, Any] = {"float32": {}, "quantized": {}, "repair": {}, "random_control": {}}
    executor: ProcessPoolExecutor | None = None
    futures = {}
    if settings.execution_mode == "parallel":
        executor = ProcessPoolExecutor(max_workers=max(1, settings.parallel_workers))
        if out:
            append_progress_locked(out, "repair_lane_ready", workers=settings.parallel_workers)
    for width in settings.widths:
        results["random_control"][width] = random_result(width, task, settings, out)
        trained = train_float_core(width, task, settings, out)
        results["float32"][width] = trained
        for level in QUANT_LEVELS:
            candidate = quantize_state_dict(level, trained["state_dict"], width)
            quantized = quantized_no_repair_result(level, width, candidate, task, settings, out)
            results["quantized"][(level, width)] = quantized
            if executor is not None:
                future = executor.submit(run_quantized_repair, level, width, candidate, task, settings, out.as_posix() if out else None)
                futures[future] = (level, width)
            else:
                results["repair"][(level, width)] = run_quantized_repair(level, width, candidate, task, settings, out.as_posix() if out else None)
    if executor is not None:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                level, width = futures[future]
                results["repair"][(level, width)] = future.result()
                if out:
                    append_progress_locked(out, "repair_lane_joined", quant_level=level, width=width, completed=len(results["repair"]))
        executor.shutdown()
    return task, results


def solve_pass(row: dict[str, float]) -> bool:
    return (
        row["heldout_accuracy"] >= 0.90
        and row["ood_accuracy"] >= 0.85
        and row["counterfactual_accuracy"] >= 0.85
        and row["adversarial_accuracy"] >= 0.80
    )


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = split_metrics(result)
    eval_accuracy = round_float(float(np.mean([metrics[split]["accuracy"] for split in eval_splits()])))
    row = {
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
    row["solve_passed"] = solve_pass(row)
    if "quant_level" in result:
        row["quant_level"] = result["quant_level"]
    if "nonzero_parameter_count" in result:
        row["nonzero_parameter_count"] = result["nonzero_parameter_count"]
        row["nonzero_ratio"] = round_float(result["nonzero_parameter_count"] / max(1, result["parameter_count"]))
    if "mutation_history" in result:
        hist = result["mutation_history"]
        row["mutation_attempt_count"] = hist["mutation_attempt_count"]
        row["accepted_mutation_count"] = hist["accepted_mutation_count"]
        row["rejected_mutation_count"] = hist["rejected_mutation_count"]
        row["rollback_count"] = hist["rollback_count"]
    return row


def aggregate_results(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    systems: dict[str, Any] = {"float32": {}, "quantized": {}, "repair": {}, "random_control": {}}
    for width in settings.widths:
        systems["float32"][str(width)] = result_row(results["float32"][width])
        systems["random_control"][str(width)] = result_row(results["random_control"][width])
    for level in QUANT_LEVELS:
        systems["quantized"][level] = {}
        systems["repair"][level] = {}
        for width in settings.widths:
            systems["quantized"][level][str(width)] = result_row(results["quantized"][(level, width)])
            systems["repair"][level][str(width)] = result_row(results["repair"][(level, width)])
    best: dict[str, Any] = {
        "float32": max((systems["float32"][str(width)] for width in settings.widths), key=lambda row: row["eval_accuracy"]),
        "random_control": max((systems["random_control"][str(width)] for width in settings.widths), key=lambda row: row["eval_accuracy"]),
        "quantized": {},
        "repair": {},
    }
    smallest: dict[str, Any] = {"float32": None, "random_control": None, "quantized": {}, "repair": {}}
    float_passing = [width for width in settings.widths if systems["float32"][str(width)]["solve_passed"]]
    smallest["float32"] = min(float_passing) if float_passing else None
    random_passing = [width for width in settings.widths if systems["random_control"][str(width)]["solve_passed"]]
    smallest["random_control"] = min(random_passing) if random_passing else None
    for level in QUANT_LEVELS:
        best["quantized"][level] = max((systems["quantized"][level][str(width)] for width in settings.widths), key=lambda row: row["eval_accuracy"])
        best["repair"][level] = max((systems["repair"][level][str(width)] for width in settings.widths), key=lambda row: row["eval_accuracy"])
        q_passing = [width for width in settings.widths if systems["quantized"][level][str(width)]["solve_passed"]]
        r_passing = [width for width in settings.widths if systems["repair"][level][str(width)]["solve_passed"]]
        smallest["quantized"][level] = min(q_passing) if q_passing else None
        smallest["repair"][level] = min(r_passing) if r_passing else None
    return {
        "schema_version": "e7a6_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "best_by_eval_accuracy": best,
        "smallest_passing_width": smallest,
        "solve_thresholds": {
            "heldout_accuracy": 0.90,
            "ood_accuracy": 0.85,
            "counterfactual_accuracy": 0.85,
            "adversarial_accuracy": 0.80,
        },
    }


def quantization_stress_report(aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for level in QUANT_LEVELS:
        rows[level] = {}
        for width in settings.widths:
            float_row = aggregate["systems"]["float32"][str(width)]
            q_row = aggregate["systems"]["quantized"][level][str(width)]
            r_row = aggregate["systems"]["repair"][level][str(width)]
            q_drop = round_float(float_row["eval_accuracy"] - q_row["eval_accuracy"])
            r_drop = round_float(float_row["eval_accuracy"] - r_row["eval_accuracy"])
            rows[level][str(width)] = {
                "float_eval_accuracy": float_row["eval_accuracy"],
                "quantized_eval_accuracy": q_row["eval_accuracy"],
                "repair_eval_accuracy": r_row["eval_accuracy"],
                "quantization_drop": q_drop,
                "repair_drop": r_drop,
                "repair_delta_vs_quantized": round_float(r_row["eval_accuracy"] - q_row["eval_accuracy"]),
                "quantized_solved": q_row["solve_passed"],
                "repair_solved": r_row["solve_passed"],
                "stable_without_repair": bool(q_row["solve_passed"] and q_drop <= 0.02),
                "stable_after_repair": bool(r_row["solve_passed"] and r_drop <= 0.02),
            }
    return {
        "schema_version": "e7a6_quantization_stress_report_v1",
        "levels": list(QUANT_LEVELS),
        "stability_drop_threshold": 0.02,
        "rows": rows,
    }


def mutation_repair_report(results: dict[str, Any], aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for level in QUANT_LEVELS:
        rows[level] = {}
        for width in settings.widths:
            rows[level][str(width)] = {
                "mutation_history": results["repair"][(level, width)]["mutation_history"],
                "aggregate_metrics": aggregate["systems"]["repair"][level][str(width)],
            }
    return {
        "schema_version": "e7a6_mutation_repair_report_v1",
        "repair_generations": settings.repair_generations,
        "rows": rows,
        "all_repair_runs_have_accept_reject_rollback": all(
            results["repair"][(level, width)]["mutation_history"]["accepted_mutation_count"] > 0
            and results["repair"][(level, width)]["mutation_history"]["rejected_mutation_count"] > 0
            and results["repair"][(level, width)]["mutation_history"]["rejected_mutation_count"]
            == results["repair"][(level, width)]["mutation_history"]["rollback_count"]
            for level in QUANT_LEVELS
            for width in settings.widths
        ),
    }


def frontier_report(aggregate: dict[str, Any], stress: dict[str, Any], settings: Settings) -> dict[str, Any]:
    stable_no_repair_by_width = {}
    stable_repair_by_width = {}
    for width in settings.widths:
        stable_no_repair_by_width[str(width)] = [level for level in QUANT_LEVELS if stress["rows"][level][str(width)]["stable_without_repair"]]
        stable_repair_by_width[str(width)] = [level for level in QUANT_LEVELS if stress["rows"][level][str(width)]["stable_after_repair"]]
    best_level_rows = {}
    for level in QUANT_LEVELS:
        best_q = aggregate["best_by_eval_accuracy"]["quantized"][level]
        best_r = aggregate["best_by_eval_accuracy"]["repair"][level]
        float_for_best_q = aggregate["systems"]["float32"][str(best_q["width"])]
        float_for_best_r = aggregate["systems"]["float32"][str(best_r["width"])]
        best_level_rows[level] = {
            "best_quantized": best_q,
            "best_repair": best_r,
            "best_quantized_drop": round_float(float_for_best_q["eval_accuracy"] - best_q["eval_accuracy"]),
            "best_repair_drop": round_float(float_for_best_r["eval_accuracy"] - best_r["eval_accuracy"]),
            "best_quantized_stable": bool(best_q["solve_passed"] and float_for_best_q["eval_accuracy"] - best_q["eval_accuracy"] <= 0.02),
            "best_repair_stable": bool(best_r["solve_passed"] and float_for_best_r["eval_accuracy"] - best_r["eval_accuracy"] <= 0.02),
        }
    return {
        "schema_version": "e7a6_frontier_report_v1",
        "stable_without_repair_by_width": stable_no_repair_by_width,
        "stable_after_repair_by_width": stable_repair_by_width,
        "best_level_rows": best_level_rows,
    }


def no_synthetic_metric_audit(task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a6_no_synthetic_metric_audit_v1",
        "generated_from_row_level_eval": True,
        "row_counts": {split: len(task[split]["rows"]) for split in task_splits()},
        "row_level_samples_present": all(
            bool(results["float32"][width]["evals"]["heldout"]["row_level_samples"])
            for width in settings.widths
        )
        and all(
            bool(results["quantized"][(level, width)]["evals"]["heldout"]["row_level_samples"])
            and bool(results["repair"][(level, width)]["evals"]["heldout"]["row_level_samples"])
            for level in QUANT_LEVELS
            for width in settings.widths
        ),
        "backprop_used_for_float32_matrix_core": True,
        "mutation_repair_used_optimizer_or_backprop": False,
        "hardcoded_improvement_flags_present": False,
        "final_e7_verdict_intentionally_deferred": True,
    }


def choose_decision(aggregate: dict[str, Any], stress: dict[str, Any], repair: dict[str, Any], audit: dict[str, Any], frontier: dict[str, Any]) -> dict[str, Any]:
    if not audit["generated_from_row_level_eval"] or audit["hardcoded_improvement_flags_present"]:
        decision = "e7a6_invalid_artifact_detected"
    elif aggregate["smallest_passing_width"]["random_control"] is not None:
        decision = "e7a6_invalid_artifact_detected"
    elif not repair["all_repair_runs_have_accept_reject_rollback"]:
        decision = "e7a6_invalid_artifact_detected"
    else:
        repair_recovery = False
        repair_value = False
        for level in QUANT_LEVELS:
            for width, row in stress["rows"][level].items():
                if (not row["stable_without_repair"]) and row["stable_after_repair"]:
                    repair_recovery = True
                if row["repair_delta_vs_quantized"] >= 0.02:
                    repair_value = True
        best_rows = frontier["best_level_rows"]
        low_stable = any(best_rows[level]["best_quantized_stable"] for level in ("int3", "ternary", "binary"))
        int4_stable = best_rows["int4"]["best_quantized_stable"]
        int8_stable = best_rows["int8"]["best_quantized_stable"]
        if repair_recovery or repair_value:
            decision = "e7a6_mutation_repair_recovers_low_bit_core" if repair_recovery else "e7a6_mutation_repair_partial_low_bit_recovery"
        elif low_stable:
            decision = "e7a6_int3_or_lower_stable_without_repair"
        elif int4_stable:
            decision = "e7a6_int4_stable_without_repair"
        elif int8_stable:
            decision = "e7a6_int8_only_stable"
        elif any(best_rows[level]["best_quantized_stable"] or best_rows[level]["best_repair_stable"] for level in QUANT_LEVELS):
            decision = "e7a6_quantization_breakpoint_mapped"
        else:
            decision = "e7a6_mutation_repair_not_useful"
    return {
        "schema_version": "e7a6_decision_v1",
        "decision": decision,
        "valid_decisions": list(VALID_DECISIONS),
        "final_e7_verdict_intentionally_deferred": True,
        "deterministic_replay_passed": False,
    }


def task_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a6_task_generation_report_v1",
        "milestone": MILESTONE,
        "inherits_task_from": "E7A3_NEURAL_MATRIX_SUBSTRATE_HARNESS",
        "input_dim": e7a3.INPUT_DIM,
        "class_count": e7a3.CLASS_COUNT,
        "minimum_target_margin": 0.65,
        "splits": {split: {"row_count": len(data["rows"])} for split, data in task.items()},
        "seeds": list(settings.seeds),
    }


def training_history_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a6_float_training_report_v1",
        "system": "float32_matrix_core",
        "widths": {
            str(width): {
                "history": results["float32"][width]["history"],
                "final_hash": results["float32"][width]["final_hash"],
                "parameter_count": results["float32"][width]["parameter_count"],
            }
            for width in settings.widths
        },
    }


def mutation_history_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a6_mutation_history_report_v1",
        "levels": {
            level: {str(width): results["repair"][(level, width)]["mutation_history"] for width in settings.widths}
            for level in QUANT_LEVELS
        },
    }


def row_samples(results: dict[str, Any], split: str, settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a6_row_level_eval_sample_v1",
        "split": split,
        "samples": {
            str(width): {
                "float32": results["float32"][width]["evals"][split]["row_level_samples"],
                "quantized": {
                    level: results["quantized"][(level, width)]["evals"][split]["row_level_samples"]
                    for level in QUANT_LEVELS
                },
                "repair": {
                    level: results["repair"][(level, width)]["evals"][split]["row_level_samples"]
                    for level in QUANT_LEVELS
                },
            }
            for width in settings.widths
        },
    }


def build_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any], frontier: dict[str, Any]) -> str:
    lines = [
        "# E7A6 Quantization Stress And Repair Limit Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        "final_e7_verdict = intentionally deferred",
        "```",
        "",
        f"Run root: `{out.relative_to(REPO_ROOT).as_posix()}`",
        "",
        "## Best Level Frontier",
        "",
        "| level | best no-repair width | no-repair eval | no-repair drop | stable no-repair | best repair width | repair eval | repair drop | stable repair |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for level in QUANT_LEVELS:
        row = frontier["best_level_rows"][level]
        q = row["best_quantized"]
        r = row["best_repair"]
        lines.append(
            f"| `{level}` | {q['width']} | {q['eval_accuracy']:.6f} | {row['best_quantized_drop']:.6f} | {row['best_quantized_stable']} | "
            f"{r['width']} | {r['eval_accuracy']:.6f} | {row['best_repair_drop']:.6f} | {row['best_repair_stable']} |"
        )
    float_best = aggregate["best_by_eval_accuracy"]["float32"]
    lines.extend(
        [
            "",
            "## Float Baseline",
            "",
            f"Best float32 matrix-core: width `{float_best['width']}`, eval `{float_best['eval_accuracy']:.6f}`.",
            "",
            "This probe only maps the controlled symbolic/numeric proxy. It does not make final E7 claims.",
            "",
        ]
    )
    return "\n".join(lines)


def build_payloads(out: Path, task: dict[str, Any], results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    aggregate = aggregate_results(results, settings)
    stress = quantization_stress_report(aggregate, settings)
    repair = mutation_repair_report(results, aggregate, settings)
    frontier = frontier_report(aggregate, stress, settings)
    no_synth = no_synthetic_metric_audit(task, results, settings)
    decision = choose_decision(aggregate, stress, repair, no_synth, frontier)
    payloads: dict[str, Any] = {
        "e7a6_backend_manifest.json": {
            "schema_version": "e7a6_backend_manifest_v1",
            "milestone": MILESTONE,
            "systems": list(SYSTEMS),
            "quant_levels": list(QUANT_LEVELS),
            "quant_configs": QUANT_CONFIGS,
            "widths": list(settings.widths),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": e7a3.select_device(settings.device),
            "settings": {**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths)},
            "cpu_repair_and_gpu_gradient_overlap_supported": True,
            "parallel_replay_supported": True,
            "final_e7_verdict_intentionally_deferred": True,
        },
        "e7a6_task_generation_report.json": task_report(task, settings),
        "e7a6_float_training_report.json": training_history_report(results, settings),
        "e7a6_quantization_stress_report.json": stress,
        "e7a6_mutation_repair_report.json": repair,
        "e7a6_frontier_report.json": frontier,
        "e7a6_mutation_history.json": mutation_history_report(results, settings),
        "e7a6_no_synthetic_metric_audit.json": no_synth,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e7a6_summary_v1",
            "milestone": MILESTONE,
            "decision": decision["decision"],
            "best_float_width": aggregate["best_by_eval_accuracy"]["float32"]["width"],
            "best_float_eval_accuracy": aggregate["best_by_eval_accuracy"]["float32"]["eval_accuracy"],
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
            "final_e7_verdict_intentionally_deferred": True,
        },
    }
    for split in eval_splits():
        payloads[f"e7a6_row_level_eval_sample_{split}.json"] = row_samples(results, split, settings)
    payloads["report.md"] = build_report(out, decision, aggregate, frontier)
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
        "schema_version": "e7a6_deterministic_replay_report_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
        "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix(),
    }
    append_progress_locked(out, "deterministic_replay_complete", internal_replay_passed=report["internal_replay_passed"])
    return report


def write_final_artifacts(out: Path, payloads: dict[str, Any], deterministic: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["e7a6_deterministic_replay_report.json"] = deterministic
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
    parser.add_argument("--gradient-epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
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
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
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
