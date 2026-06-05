#!/usr/bin/env python3
"""E7A4 quantized bridge from backprop to mutation matrix.

This probe tests whether the E7A3 matrix hidden replacement is learnable when
trained as a differentiable float matrix-core, then whether that learned core
survives quantization and mutation repair.
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
import sys
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
E7A3_PATH = Path(__file__).with_name("run_e7a3_neural_matrix_substrate_harness.py")
MILESTONE = "E7A4_QUANTIZED_BRIDGE_FROM_BACKPROP_TO_MUTATION_MATRIX"
DEFAULT_OUT = Path("target/pilot_wave/e7a4_quantized_bridge_from_backprop_to_mutation_matrix")
DEFAULT_SEEDS = (80001, 80002, 80003)

SYSTEMS = (
    "float_mlp_backprop_reference",
    "float_matrix_core_backprop",
    "quantized_matrix_core_no_repair",
    "quantized_matrix_core_mutation_repair",
    "random_control",
)
GRADIENT_SYSTEMS = ("float_mlp_backprop_reference", "float_matrix_core_backprop")
MUTATION_SYSTEMS = ("quantized_matrix_core_mutation_repair",)
HASH_ARTIFACTS = (
    "e7a4_task_generation_report.json",
    "e7a4_bridge_report.json",
    "e7a4_quantization_report.json",
    "e7a4_mutation_repair_report.json",
    "e7a4_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
VALID_DECISIONS = (
    "e7a4_reference_not_solved_redesign_required",
    "e7a4_float_matrix_core_not_learned",
    "e7a4_quantized_matrix_core_preserved_without_repair",
    "e7a4_mutation_repair_improves_quantized_matrix_core",
    "e7a4_mutation_repair_recovers_quantized_matrix_core",
    "e7a4_quantization_breaks_and_mutation_repair_failed",
    "e7a4_task_too_easy_or_leaky",
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
    quant_limit: int
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
    return int(hashlib.sha256(f"e7a4::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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
        integer_weight_limit=settings.quant_limit,
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


def train_float_matrix_core(width: int, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = e7a3.select_device(settings.device)
    e7a3.set_determinism(stable_seed(f"float-matrix-core-{width}-{settings.seeds}"), device)
    model = FloatMatrixCore(width, settings.matrix_steps).to(device)
    initial_vector = torch_state_vector(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train = torch.as_tensor(task["train"]["x"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(task["train"]["y"], dtype=torch.long, device=device)
    x_val = torch.as_tensor(task["validation"]["x"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(task["validation"]["y"], dtype=torch.long, device=device)
    rng = np.random.default_rng(stable_seed(f"matrix-core-batches-{width}-{settings.seeds}"))
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
            val_logits = model(x_val)
            val_loss = F.cross_entropy(val_logits, y_val).detach().cpu().item()
            val_acc = (torch.argmax(val_logits, dim=1) == y_val).float().mean().detach().cpu().item()
        row = {
            "epoch": epoch,
            "validation_accuracy": round_float(val_acc),
            "validation_loss": round_float(val_loss),
            "state_hash": vector_hash(torch_state_vector(model)),
        }
        history.append(row)
        now = time.monotonic()
        if out and (now - last_heartbeat >= settings.heartbeat_seconds or epoch == settings.gradient_epochs):
            write_json(out / f"e7a4_training_history_float_matrix_core_backprop_width{width}.json", {"system": "float_matrix_core_backprop", "width": width, "history": history})
            append_progress_locked(out, "matrix_core_gradient_epoch", width=width, epoch=epoch, validation_accuracy=row["validation_accuracy"])
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
    rank = int(np.sum(singular > 1e-6))
    state_dict = {key: value.detach().cpu().numpy().astype(np.float64) for key, value in model.state_dict().items()}
    result = {
        "system": "float_matrix_core_backprop",
        "width": width,
        "training_mode": "gradient_backprop",
        "device": device,
        "evals": evals,
        "history": history,
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
        "hidden_matrix_shape": [width, width],
        "hidden_state_rank": rank,
        "hidden_state_singular_values_sample": [round_float(v) for v in singular[: min(8, len(singular))].tolist()],
        "initial_hash": vector_hash(initial_vector),
        "final_hash": vector_hash(torch_state_vector(model)),
        "state_dict": state_dict,
    }
    if out:
        write_system_width_summary(out, result)
        write_json(out / f"e7a4_float_matrix_core_state_width{width}.json", serializable_state_dict(state_dict))
    return result


def serializable_state_dict(state_dict: dict[str, np.ndarray]) -> dict[str, Any]:
    return {key: [[round_float(v) for v in row] for row in value.tolist()] if value.ndim == 2 else [round_float(v) for v in value.tolist()] for key, value in state_dict.items()}


def quantize_state_dict(state_dict: dict[str, np.ndarray], width: int, quant_limit: int) -> dict[str, Any]:
    q: dict[str, Any] = {}
    scales: dict[str, float] = {}
    for key, value in state_dict.items():
        max_abs = float(np.max(np.abs(value))) if value.size else 0.0
        scale = max_abs / quant_limit if max_abs > 0 else 1.0 / quant_limit
        quantized = np.clip(np.rint(value / scale), -quant_limit, quant_limit).astype(np.int16)
        q[key] = quantized.tolist()
        scales[key] = round_float(scale)
    return {
        "schema_version": "e7a4_quantized_matrix_core_candidate_v1",
        "system": "quantized_matrix_core",
        "width": width,
        "quant_limit": quant_limit,
        "q": q,
        "scales": scales,
    }


def dequant(candidate: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(candidate["q"][key], dtype=np.float64) * float(candidate["scales"][key])


def quantized_forward(candidate: dict[str, Any], x: np.ndarray, matrix_steps: int) -> np.ndarray:
    win = dequant(candidate, "win")
    state = dequant(candidate, "state")
    carry_raw = dequant(candidate, "carry_raw")
    bstate = dequant(candidate, "bstate")
    wout = dequant(candidate, "wout")
    bout = dequant(candidate, "bout")
    drive = x @ win + bstate
    h = np.tanh(drive)
    carry = 1.0 / (1.0 + np.exp(-np.clip(carry_raw, -40.0, 40.0)))
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
    paths = quantized_paths(child)
    for path, value in rng.sample(paths, k=min(settings.quant_mutation_steps, len(paths))):
        step = rng.choice((-3, -2, -1, 1, 2, 3))
        set_quantized_path(child, path, min(max(value + step, -settings.quant_limit), settings.quant_limit))
    return child


def score_quantized(candidate: dict[str, Any], task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    evals = {
        split: e7a3.evaluate_logits(quantized_forward(candidate, data["x"], settings.matrix_steps), data, sample_limit=0)
        for split, data in {"train": task["train"], "validation": task["validation"]}.items()
    }
    train = evals["train"]["metrics"]
    val = evals["validation"]["metrics"]
    fitness = 0.20 * train["accuracy"] + 0.80 * val["accuracy"] - 0.02 * val["cross_entropy"]
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


def run_quantized_repair(width: int, initial_candidate: dict[str, Any], task: dict[str, Any], settings: Settings, out: str | None) -> dict[str, Any]:
    out_path = Path(out) if out else None
    if out_path:
        append_progress_locked(out_path, "repair_start", width=width)
    rng = random.Random(stable_seed(f"repair-{width}-{settings.seeds}"))
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
            parent = copy.deepcopy(rng.choice(elites))
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
                    out_path / "partial_status" / f"repair_width{width}.json",
                    {
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
                append_progress_locked(out_path, "repair_heartbeat", width=width, generation=generation, best_fitness=best_mid["fitness"])
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
            write_json(out_path / f"e7a4_mutation_history_quantized_matrix_core_mutation_repair_width{width}.json", {"width": width, "history": history, "accepted_mutation_count": accepted, "rejected_mutation_count": rejected, "rollback_count": rollback, "mutation_attempt_count": attempts})
            append_progress_locked(out_path, "repair_generation", width=width, generation=generation, validation_accuracy=row["validation_accuracy"])
    best = max(population, key=lambda row: row["fitness"])
    final_candidate = best["candidate"]
    evals = evaluate_quantized_candidate(final_candidate, task, settings.matrix_steps)
    result = {
        "system": "quantized_matrix_core_mutation_repair",
        "width": width,
        "training_mode": "quantized_mutation_repair",
        "evals": evals,
        "history": history,
        "mutation_history": {
            "width": width,
            "mutation_attempt_count": attempts,
            "accepted_mutation_count": accepted,
            "rejected_mutation_count": rejected,
            "rollback_count": rollback,
            "history": history,
        },
        "parameter_count": quantized_parameter_count(final_candidate),
        "hidden_matrix_shape": [width, width],
        "initial_candidate": initial_candidate,
        "final_candidate": final_candidate,
        "parameter_diff": quantized_diff(initial_candidate, final_candidate),
        "initial_hash": candidate_hash(initial_candidate),
        "final_hash": candidate_hash(final_candidate),
    }
    if out_path:
        write_system_width_summary(out_path, result)
        write_json(out_path / f"e7a4_candidate_quantized_matrix_core_mutation_repair_width{width}_initial.json", initial_candidate)
        write_json(out_path / f"e7a4_candidate_quantized_matrix_core_mutation_repair_width{width}_final.json", final_candidate)
        write_json(out_path / f"e7a4_parameter_diff_quantized_matrix_core_mutation_repair_width{width}.json", result["parameter_diff"])
        append_progress_locked(out_path, "repair_complete", width=width, heldout_accuracy=evals["heldout"]["metrics"]["accuracy"])
    return result


def split_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {split: result["evals"][split]["metrics"] for split in e7a3.SPLITS}


def write_system_width_summary(out: Path, result: dict[str, Any]) -> None:
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
        "split_metrics": split_metrics(result),
    }
    for key in ("hidden_state_rank", "hidden_state_singular_values_sample", "device"):
        if key in result:
            summary[key] = result[key]
    write_json(out / f"e7a4_candidate_{system}_width{width}_summary.json", summary)


def quantized_no_repair_result(width: int, candidate: dict[str, Any], task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    evals = evaluate_quantized_candidate(candidate, task, settings.matrix_steps)
    result = {
        "system": "quantized_matrix_core_no_repair",
        "width": width,
        "training_mode": "post_backprop_quantized_no_repair",
        "evals": evals,
        "history": [],
        "parameter_count": quantized_parameter_count(candidate),
        "hidden_matrix_shape": [width, width],
        "initial_hash": candidate_hash(candidate),
        "final_hash": candidate_hash(candidate),
        "candidate": candidate,
    }
    if out:
        write_system_width_summary(out, result)
        write_json(out / f"e7a4_candidate_quantized_matrix_core_no_repair_width{width}.json", candidate)
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
        write_system_width_summary(out, result)
    return result


def rename_reference_result(result: dict[str, Any]) -> dict[str, Any]:
    renamed = copy.deepcopy(result)
    renamed["system"] = "float_mlp_backprop_reference"
    return renamed


def run_core(settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, settings=settings.__dict__)
    task = e7a3.generate_task(e7a3_settings(settings))
    if out:
        append_progress_locked(out, "task_generated", rows={split: len(task[split]["rows"]) for split in e7a3.SPLITS})
    results: dict[str, dict[int, Any]] = {system: {} for system in SYSTEMS}
    futures = {}
    executor: ProcessPoolExecutor | None = None
    if settings.execution_mode == "parallel":
        executor = ProcessPoolExecutor(max_workers=max(1, settings.parallel_workers))
        if out:
            append_progress_locked(out, "repair_lane_ready", workers=settings.parallel_workers)
    for width in settings.widths:
        reference = rename_reference_result(e7a3.train_float_mlp(width, task, e7a3_settings(settings), None))
        results["float_mlp_backprop_reference"][width] = reference
        if out:
            write_system_width_summary(out, reference)
            append_progress_locked(out, "reference_complete", width=width, heldout_accuracy=reference["evals"]["heldout"]["metrics"]["accuracy"])
        matrix_core = train_float_matrix_core(width, task, settings, out)
        results["float_matrix_core_backprop"][width] = matrix_core
        quantized = quantize_state_dict(matrix_core["state_dict"], width, settings.quant_limit)
        no_repair = quantized_no_repair_result(width, quantized, task, settings, out)
        results["quantized_matrix_core_no_repair"][width] = no_repair
        results["random_control"][width] = random_result(width, task, settings, out)
        if executor is not None:
            future = executor.submit(run_quantized_repair, width, quantized, task, settings, out.as_posix() if out else None)
            futures[future] = width
        else:
            results["quantized_matrix_core_mutation_repair"][width] = run_quantized_repair(width, quantized, task, settings, out.as_posix() if out else None)
    if executor is not None:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                width = futures[future]
                results["quantized_matrix_core_mutation_repair"][width] = future.result()
                if out:
                    append_progress_locked(out, "repair_lane_joined", width=width, completed=len(results["quantized_matrix_core_mutation_repair"]))
        executor.shutdown()
    return task, results


def solve_pass(row: dict[str, float]) -> bool:
    return (
        row["heldout_accuracy"] >= 0.90
        and row["ood_accuracy"] >= 0.85
        and row["counterfactual_accuracy"] >= 0.85
        and row["adversarial_accuracy"] >= 0.80
    )


def aggregate_results(results: dict[str, dict[int, Any]], settings: Settings) -> dict[str, Any]:
    systems = {}
    for system in SYSTEMS:
        systems[system] = {}
        for width in settings.widths:
            result = results[system][width]
            metrics = split_metrics(result)
            eval_accuracy = round_float(float(np.mean([metrics[split]["accuracy"] for split in e7a3.EVAL_SPLITS])))
            row = {
                "width": width,
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
            if system == "float_matrix_core_backprop":
                row["hidden_state_rank"] = result["hidden_state_rank"]
            if system in MUTATION_SYSTEMS:
                hist = result["mutation_history"]
                row["mutation_attempt_count"] = hist["mutation_attempt_count"]
                row["accepted_mutation_count"] = hist["accepted_mutation_count"]
                row["rejected_mutation_count"] = hist["rejected_mutation_count"]
                row["rollback_count"] = hist["rollback_count"]
            systems[system][str(width)] = row
    smallest = {}
    best = {}
    for system in SYSTEMS:
        passing = [int(width) for width, row in systems[system].items() if row["solve_passed"]]
        smallest[system] = min(passing) if passing else None
        best_width = max(settings.widths, key=lambda width: systems[system][str(width)]["eval_accuracy"])
        best[system] = {"width": best_width, **systems[system][str(best_width)]}
    return {
        "schema_version": "e7a4_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "smallest_passing_width": smallest,
        "best_by_eval_accuracy": best,
        "solve_thresholds": {
            "heldout_accuracy": 0.90,
            "ood_accuracy": 0.85,
            "counterfactual_accuracy": 0.85,
            "adversarial_accuracy": 0.80,
        },
    }


def quantization_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for width, float_row in aggregate["systems"]["float_matrix_core_backprop"].items():
        q_row = aggregate["systems"]["quantized_matrix_core_no_repair"][width]
        repair_row = aggregate["systems"]["quantized_matrix_core_mutation_repair"][width]
        rows[width] = {
            "float_eval_accuracy": float_row["eval_accuracy"],
            "quantized_eval_accuracy": q_row["eval_accuracy"],
            "repair_eval_accuracy": repair_row["eval_accuracy"],
            "quantization_delta": round_float(q_row["eval_accuracy"] - float_row["eval_accuracy"]),
            "repair_delta_vs_quantized": round_float(repair_row["eval_accuracy"] - q_row["eval_accuracy"]),
            "float_solved": float_row["solve_passed"],
            "quantized_solved": q_row["solve_passed"],
            "repair_solved": repair_row["solve_passed"],
        }
    return {
        "schema_version": "e7a4_quantization_report_v1",
        "quant_limit": "symmetric per-tensor integer limit from settings",
        "rows": rows,
    }


def mutation_repair_report(results: dict[str, dict[int, Any]], aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a4_mutation_repair_report_v1",
        "repair_generations": settings.repair_generations,
        "mutation_system": "quantized_matrix_core_mutation_repair",
        "rows": {
            str(width): {
                "mutation_history": results["quantized_matrix_core_mutation_repair"][width]["mutation_history"],
                "aggregate_metrics": aggregate["systems"]["quantized_matrix_core_mutation_repair"][str(width)],
            }
            for width in settings.widths
        },
        "all_repair_runs_have_accept_reject_rollback": all(
            results["quantized_matrix_core_mutation_repair"][width]["mutation_history"]["accepted_mutation_count"] > 0
            and results["quantized_matrix_core_mutation_repair"][width]["mutation_history"]["rejected_mutation_count"] > 0
            and results["quantized_matrix_core_mutation_repair"][width]["mutation_history"]["rejected_mutation_count"]
            == results["quantized_matrix_core_mutation_repair"][width]["mutation_history"]["rollback_count"]
            for width in settings.widths
        ),
    }


def bridge_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a4_bridge_report_v1",
        "systems": list(SYSTEMS),
        "smallest_passing_width": aggregate["smallest_passing_width"],
        "best_by_eval_accuracy": aggregate["best_by_eval_accuracy"],
        "float_matrix_core_learned": aggregate["smallest_passing_width"]["float_matrix_core_backprop"] is not None,
        "quantized_matrix_core_preserved": aggregate["smallest_passing_width"]["quantized_matrix_core_no_repair"] is not None,
        "mutation_repair_solved": aggregate["smallest_passing_width"]["quantized_matrix_core_mutation_repair"] is not None,
    }


def no_synthetic_metric_audit(task: dict[str, Any], results: dict[str, dict[int, Any]], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a4_no_synthetic_metric_audit_v1",
        "generated_from_row_level_eval": True,
        "row_counts": {split: len(task[split]["rows"]) for split in e7a3.SPLITS},
        "row_level_samples_present": all(
            bool(results[system][width]["evals"]["heldout"]["row_level_samples"])
            for system in SYSTEMS
            for width in settings.widths
        ),
        "backprop_used_for_float_systems": True,
        "mutation_repair_used_optimizer_or_backprop": False,
        "hardcoded_improvement_flags_present": False,
        "final_e7_verdict_intentionally_deferred": True,
    }


def training_history_report(results: dict[str, dict[int, Any]], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a4_training_history_report_v1",
        "systems": {
            "float_mlp_backprop_reference": {str(width): results["float_mlp_backprop_reference"][width]["history"] for width in settings.widths},
            "float_matrix_core_backprop": {str(width): results["float_matrix_core_backprop"][width]["history"] for width in settings.widths},
        },
    }


def mutation_history_report(results: dict[str, dict[int, Any]], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a4_mutation_history_report_v1",
        "systems": {
            "quantized_matrix_core_mutation_repair": {
                str(width): results["quantized_matrix_core_mutation_repair"][width]["mutation_history"]
                for width in settings.widths
            }
        },
    }


def row_samples(results: dict[str, dict[int, Any]], split: str, settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a4_row_level_eval_sample_v1",
        "split": split,
        "samples": {
            system: {str(width): results[system][width]["evals"][split]["row_level_samples"] for width in settings.widths}
            for system in SYSTEMS
        },
    }


def choose_decision(aggregate: dict[str, Any], bridge: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    if aggregate["smallest_passing_width"]["random_control"] is not None:
        decision = "e7a4_task_too_easy_or_leaky"
    elif aggregate["smallest_passing_width"]["float_mlp_backprop_reference"] is None:
        decision = "e7a4_reference_not_solved_redesign_required"
    elif aggregate["smallest_passing_width"]["float_matrix_core_backprop"] is None:
        decision = "e7a4_float_matrix_core_not_learned"
    elif aggregate["smallest_passing_width"]["quantized_matrix_core_no_repair"] is not None:
        q_best = bridge["best_by_eval_accuracy"]["quantized_matrix_core_no_repair"]
        r_best = bridge["best_by_eval_accuracy"]["quantized_matrix_core_mutation_repair"]
        if r_best["eval_accuracy"] >= q_best["eval_accuracy"] + 0.01:
            decision = "e7a4_mutation_repair_improves_quantized_matrix_core"
        else:
            decision = "e7a4_quantized_matrix_core_preserved_without_repair"
    elif aggregate["smallest_passing_width"]["quantized_matrix_core_mutation_repair"] is not None:
        decision = "e7a4_mutation_repair_recovers_quantized_matrix_core"
    else:
        decision = "e7a4_quantization_breaks_and_mutation_repair_failed"
    if not audit["generated_from_row_level_eval"] or audit["hardcoded_improvement_flags_present"]:
        decision = "e7a4_task_too_easy_or_leaky"
    return {
        "schema_version": "e7a4_decision_v1",
        "decision": decision,
        "valid_decisions": list(VALID_DECISIONS),
        "smallest_passing_width": aggregate["smallest_passing_width"],
        "final_e7_verdict_intentionally_deferred": True,
        "deterministic_replay_passed": False,
    }


def task_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a4_task_generation_report_v1",
        "milestone": MILESTONE,
        "inherits_task_from": "E7A3_NEURAL_MATRIX_SUBSTRATE_HARNESS",
        "input_dim": e7a3.INPUT_DIM,
        "class_count": e7a3.CLASS_COUNT,
        "minimum_target_margin": 0.65,
        "splits": {split: {"row_count": len(data["rows"])} for split, data in task.items()},
        "seeds": list(settings.seeds),
    }


def build_report(out: Path, aggregate: dict[str, Any], decision: dict[str, Any], bridge: dict[str, Any]) -> str:
    lines = [
        "# E7A4 Quantized Bridge From Backprop To Mutation Matrix Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"float_matrix_core_learned = {bridge['float_matrix_core_learned']}",
        f"quantized_matrix_core_preserved = {bridge['quantized_matrix_core_preserved']}",
        f"mutation_repair_solved = {bridge['mutation_repair_solved']}",
        "final_e7_verdict = intentionally deferred",
        "```",
        "",
        f"Run root: `{out.relative_to(REPO_ROOT).as_posix()}`",
        "",
        "## Width Sweep",
        "",
        "| system | smallest passing width | best width | best eval accuracy |",
        "|---|---:|---:|---:|",
    ]
    for system, best in aggregate["best_by_eval_accuracy"].items():
        smallest = aggregate["smallest_passing_width"][system]
        lines.append(f"| `{system}` | {smallest if smallest is not None else 'none'} | {best['width']} | {best['eval_accuracy']:.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe tests whether the matrix-core idea is learnable as a float differentiable system, whether quantization preserves it, and whether mutation repair adds value after quantization.",
            "",
            "It does not claim a final matrix-medium architecture or model-scale behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def build_payloads(out: Path, task: dict[str, Any], results: dict[str, dict[int, Any]], settings: Settings) -> dict[str, Any]:
    aggregate = aggregate_results(results, settings)
    bridge = bridge_report(aggregate)
    quant = quantization_report(aggregate)
    repair = mutation_repair_report(results, aggregate, settings)
    audit = no_synthetic_metric_audit(task, results, settings)
    decision = choose_decision(aggregate, bridge, audit)
    payloads: dict[str, Any] = {
        "e7a4_backend_manifest.json": {
            "schema_version": "e7a4_backend_manifest_v1",
            "milestone": MILESTONE,
            "systems": list(SYSTEMS),
            "gradient_systems": list(GRADIENT_SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "widths": list(settings.widths),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": e7a3.select_device(settings.device),
            "settings": {**settings.__dict__, "seeds": list(settings.seeds), "widths": list(settings.widths)},
            "cpu_repair_and_gpu_gradient_overlap_supported": True,
            "final_e7_verdict_intentionally_deferred": True,
        },
        "e7a4_task_generation_report.json": task_report(task, settings),
        "e7a4_bridge_report.json": bridge,
        "e7a4_quantization_report.json": quant,
        "e7a4_mutation_repair_report.json": repair,
        "e7a4_training_history.json": training_history_report(results, settings),
        "e7a4_mutation_history.json": mutation_history_report(results, settings),
        "e7a4_no_synthetic_metric_audit.json": audit,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e7a4_summary_v1",
            "milestone": MILESTONE,
            "decision": decision["decision"],
            "smallest_passing_width": aggregate["smallest_passing_width"],
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
            "final_e7_verdict_intentionally_deferred": True,
        },
    }
    for split in e7a3.EVAL_SPLITS:
        payloads[f"e7a4_row_level_eval_sample_{split}.json"] = row_samples(results, split, settings)
    payloads["report.md"] = build_report(out, aggregate, decision, bridge)
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
        name: {
            "primary_hash": primary_hashes[name],
            "replay_hash": replay_hashes[name],
            "match": primary_hashes[name] == replay_hashes[name],
        }
        for name in HASH_ARTIFACTS
    }
    return {
        "schema_version": "e7a4_deterministic_replay_report_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
    }


def write_final_artifacts(out: Path, payloads: dict[str, Any], deterministic: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["e7a4_deterministic_replay_report.json"] = deterministic
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
    parser.add_argument("--gradient-epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--repair-generations", type=int, default=60)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--quant-mutation-steps", type=int, default=20)
    parser.add_argument("--quant-limit", type=int, default=127)
    parser.add_argument("--matrix-steps", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--execution-mode", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--parallel-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 4, 12)))
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
        quant_limit=args.quant_limit,
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
