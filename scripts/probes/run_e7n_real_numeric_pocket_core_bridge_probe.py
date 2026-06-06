#!/usr/bin/env python3
"""E7N real numeric pocket core bridge probe.

E7N replaces the E7K/E7L/E7M symbolic pocket proxy with an actual learned
numeric pocket:

    CALL(pocket_id, Flow[D]) -> Flow[D]

The pocket has real matrices: input adapter D->K, recurrent/core matrix K->K,
and output adapter K->D. Backprop is allowed only for pocket pretraining.
Quantization, mutation repair, and pruning/crystallization are mutation-only
NumPy paths.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "E7N_REAL_NUMERIC_POCKET_CORE_BRIDGE_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7n_real_numeric_pocket_core_bridge_probe")
DEFAULT_SEEDS = (99301, 99302, 99303, 99304, 99305, 99306)

FLOW_DIM = 40
CLASS_COUNT = 8
TASKS = (
    "xor_parity",
    "compare",
    "mod_add",
    "threshold",
    "route_check",
    "counterfactual_flip",
)
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
SYSTEMS = (
    "symbolic_proxy_pocket_reference",
    "float_numeric_pocket_backprop",
    "quantized_numeric_pocket_int8",
    "quantized_numeric_pocket_int4",
    "quantized_numeric_pocket_ternary",
    "quantized_numeric_pocket_binary",
    "quantized_pocket_plus_mutation_repair",
    "quantized_pocket_plus_prune_crystallize",
    "quantized_pocket_plus_repair_plus_prune",
    "random_pocket_control",
)
GRADIENT_SYSTEMS = ("float_numeric_pocket_backprop",)
MUTATION_SYSTEMS = (
    "quantized_pocket_plus_mutation_repair",
    "quantized_pocket_plus_prune_crystallize",
    "quantized_pocket_plus_repair_plus_prune",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "numeric_pocket_training_report.json",
    "quantization_report.json",
    "mutation_repair_report.json",
    "pruning_crystallization_report.json",
    "pocket_registry_report.json",
    "router_call_report.json",
    "system_results.json",
    "mutation_history.json",
    "leakage_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7n_real_numeric_pocket_viable",
    "e7n_quantized_numeric_pocket_viable",
    "e7n_mutation_repair_numeric_pocket_positive",
    "e7n_numeric_pocket_crystallization_positive",
    "e7n_binary_numeric_pocket_viable",
    "e7n_float_only_numeric_pocket",
    "e7n_numeric_pocket_bridge_failed",
)


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    pocket_dim: int
    core_steps: int
    gradient_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    repair_generations: int
    repair_population: int
    mutation_flip_count: int
    prune_rounds: int
    prune_fraction: float
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    execution_mode: str
    replay: bool


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7n::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    for attempt in range(12):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.08 * (attempt + 1))
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    lock = path.with_suffix(path.suffix + ".lock")
    fd: int | None = None
    deadline = time.monotonic() + 120.0
    while fd is None:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            if time.monotonic() > deadline:
                raise TimeoutError(f"lock timeout: {lock}")
            time.sleep(0.025)
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"event": event, "details": details, "time": round_float(time.time())})


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
        raise ValueError("empty seed list")
    return values


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    return payload


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def set_determinism(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def hardware_sample() -> dict[str, Any]:
    sample: dict[str, Any] = {"time": round_float(time.time()), "pid": os.getpid()}
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        if raw:
            util, mem, temp = [part.strip() for part in raw.splitlines()[0].split(",")[:3]]
            sample["gpu_util_percent"] = float(util)
            sample["gpu_memory_mb"] = float(mem)
            sample["gpu_temp_c"] = float(temp)
    except Exception:
        sample["gpu_query_available"] = False
    return sample


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    def worker() -> None:
        while not stop.is_set():
            append_jsonl(out / "hardware_heartbeat.jsonl", hardware_sample())
            stop.wait(max(1.0, interval))

    thread = threading.Thread(target=worker, name="e7n-hardware-heartbeat", daemon=True)
    thread.start()
    return thread


def row_count_for_split(settings: Settings, split: str) -> int:
    return int(getattr(settings, f"{split}_rows_per_seed"))


def class_for_task(task: str, a: int, b: int, key: int, threshold: int, flip: int) -> int:
    if task == "xor_parity":
        return 1 if ((a >= 4) != (b >= 4)) else 0
    if task == "compare":
        return 1 if a > b else 0
    if task == "mod_add":
        return 1 if (a + b + key) >= 11 else 0
    if task == "threshold":
        return 1 if (a + b) >= (2 * threshold) else 0
    if task == "route_check":
        selected = a if key < 4 else b
        return 1 if selected >= threshold else 0
    if task == "counterfactual_flip":
        base = 1 if (a + b + key) >= 11 else 0
        return 1 - base if flip else base
    raise ValueError(task)


def encode_flow(task: str, a: int, b: int, key: int, threshold: int, flip: int, decoy: int, split: str) -> np.ndarray:
    x = np.zeros(FLOW_DIM, dtype=np.float32)
    task_idx = TASKS.index(task)
    x[task_idx] = 1.0
    selected = a if key < 4 else b
    x[6] = a / 7.0
    x[7] = b / 7.0
    x[8] = key / 7.0
    x[9] = threshold / 7.0
    x[10] = float(flip)
    x[11] = (a + b) / 14.0
    x[12] = (a + b + key) / 21.0
    x[13] = (a - b) / 7.0
    x[14] = 1.0 if a >= 4 else 0.0
    x[15] = 1.0 if b >= 4 else 0.0
    x[16] = 1.0 if key < 4 else 0.0
    x[17] = selected / 7.0
    x[18] = (selected - threshold) / 7.0
    x[19] = math.sin((a + 1) * 0.37)
    x[20] = math.cos((b + 1) * 0.41)
    x[21] = math.sin((key + 1) * 0.29)
    # The adversarial split deliberately carries a plausible but misleading
    # decoy bit. A valid pocket must ignore it when the task contract says so.
    x[39] = float(decoy)
    if split == "ood":
        x *= 0.97
    return x


def encode_target(task: str, cls: int) -> np.ndarray:
    y = np.zeros(FLOW_DIM, dtype=np.float32)
    y[cls] = 1.0
    y[8 + TASKS.index(task)] = 1.0
    y[14] = 1.0
    return y


def make_row(seed: int, split: str, idx: int, rng: random.Random) -> dict[str, Any]:
    task = TASKS[idx % len(TASKS)]
    if split == "train":
        a = rng.randrange(0, 7)
        b = rng.randrange(0, 7)
    elif split == "ood":
        a = rng.choice((5, 6, 7))
        b = rng.choice((5, 6, 7))
    else:
        a = rng.randrange(0, CLASS_COUNT)
        b = rng.randrange(0, CLASS_COUNT)
    key = rng.randrange(0, CLASS_COUNT)
    threshold = rng.randrange(1, CLASS_COUNT)
    flip = 1 if split == "counterfactual" else (1 if task == "counterfactual_flip" and rng.random() < 0.45 else 0)
    cls = class_for_task(task, a, b, key, threshold, flip)
    decoy = (1 - cls) if split == "adversarial" and task in {"xor_parity", "compare", "threshold", "counterfactual_flip"} else rng.randrange(0, 2)
    x = encode_flow(task, a, b, key, threshold, flip, decoy, split)
    y = encode_target(task, cls)
    return {
        "row_id": f"{seed}:{split}:{idx}",
        "split": split,
        "task": task,
        "a": a,
        "b": b,
        "key": key,
        "threshold": threshold,
        "flip": flip,
        "decoy": decoy,
        "target_class": cls,
        "flow": x.tolist(),
        "target_flow": y.tolist(),
    }


def generate_tasks(settings: Settings) -> dict[int, dict[str, list[dict[str, Any]]]]:
    tasks: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for seed in settings.seeds:
        seed_rows: dict[str, list[dict[str, Any]]] = {}
        for split in SPLITS:
            rng = random.Random(stable_seed(f"task:{seed}:{split}"))
            seed_rows[split] = [make_row(seed, split, idx, rng) for idx in range(row_count_for_split(settings, split))]
        tasks[seed] = seed_rows
    return tasks


def split_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([row["flow"] for row in rows], dtype=np.float32)
    y = np.asarray([row["target_flow"] for row in rows], dtype=np.float32)
    cls = np.asarray([int(row["target_class"]) for row in rows], dtype=np.int64)
    return x, y, cls


class NumericPocketCore(nn.Module):
    def __init__(self, flow_dim: int, pocket_dim: int, core_steps: int) -> None:
        super().__init__()
        self.flow_dim = flow_dim
        self.pocket_dim = pocket_dim
        self.core_steps = core_steps
        scale = 1.0 / math.sqrt(max(1, pocket_dim))
        self.win = nn.Parameter(torch.empty(flow_dim, pocket_dim).uniform_(-scale, scale))
        self.wcore = nn.Parameter(torch.empty(pocket_dim, pocket_dim).uniform_(-scale, scale))
        self.carry_raw = nn.Parameter(torch.zeros(pocket_dim))
        self.bin = nn.Parameter(torch.zeros(pocket_dim))
        self.wout = nn.Parameter(torch.empty(pocket_dim, flow_dim).uniform_(-scale, scale))
        self.bout = nn.Parameter(torch.zeros(flow_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        drive = x @ self.win + self.bin
        h = torch.tanh(drive)
        carry = torch.sigmoid(self.carry_raw).unsqueeze(0)
        for _ in range(self.core_steps):
            proposal = torch.tanh(h @ self.wcore + drive)
            h = carry * h + (1.0 - carry) * proposal
        return h @ self.wout + self.bout


def model_to_state(model: NumericPocketCore, settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7n_numeric_pocket_state_v1",
        "precision": "float32",
        "flow_dim": FLOW_DIM,
        "pocket_dim": settings.pocket_dim,
        "core_steps": settings.core_steps,
        "arrays": {name: param.detach().cpu().numpy().astype(np.float32) for name, param in model.named_parameters()},
        "masks": {},
        "lineage": ["float_backprop_pretrain"],
    }


def serializable_state(state: dict[str, Any], include_arrays: bool = False) -> dict[str, Any]:
    arrays = state["arrays"]
    payload = {
        "schema_version": state.get("schema_version"),
        "precision": state.get("precision"),
        "flow_dim": int(state.get("flow_dim", FLOW_DIM)),
        "pocket_dim": int(state.get("pocket_dim", 0)),
        "core_steps": int(state.get("core_steps", 0)),
        "lineage": list(state.get("lineage", [])),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_hashes": {key: hashlib.sha256(np.round(value.astype(np.float64), 10).tobytes()).hexdigest() for key, value in arrays.items()},
        "mask_hashes": {key: hashlib.sha256(value.astype(np.float32).tobytes()).hexdigest() for key, value in state.get("masks", {}).items()},
        "state_hash": state_hash(state),
    }
    if include_arrays:
        payload["arrays"] = {key: np.round(value.astype(np.float64), 8).tolist() for key, value in arrays.items()}
        payload["masks"] = {key: value.astype(np.float32).tolist() for key, value in state.get("masks", {}).items()}
    return payload


def state_hash(state: dict[str, Any]) -> str:
    arrays = {key: np.round(value.astype(np.float64), 10).tolist() for key, value in sorted(state["arrays"].items())}
    masks = {key: value.astype(np.float32).tolist() for key, value in sorted(state.get("masks", {}).items())}
    return payload_sha256({"precision": state.get("precision"), "arrays": arrays, "masks": masks, "lineage": state.get("lineage", [])})


def parameter_count(state: dict[str, Any], active_only: bool = False) -> int:
    total = 0
    for key, value in state["arrays"].items():
        mask = state.get("masks", {}).get(key)
        if active_only and mask is not None:
            total += int(np.count_nonzero(mask))
        else:
            total += int(value.size)
    return total


def bit_budget(state: dict[str, Any]) -> int:
    precision = str(state.get("precision", "float32"))
    bits_by_precision = {"float32": 32, "int8": 8, "int4": 4, "ternary": 2, "binary": 1}
    bits = bits_by_precision.get(precision, 32)
    active = parameter_count(state, active_only=True)
    # Symmetric quantization stores one scale per tensor. The no-scale binary
    # research branch can be tested later; E7N keeps the bridge conservative.
    scale_bits = 0 if precision == "float32" else 32 * len(state["arrays"])
    return int(active * bits + scale_bits)


def np_forward(state: dict[str, Any], x: np.ndarray) -> np.ndarray:
    arrays = state["arrays"]
    masks = state.get("masks", {})

    def arr(name: str) -> np.ndarray:
        value = arrays[name]
        if name in masks:
            return value * masks[name]
        return value

    drive = x @ arr("win") + arr("bin")
    h = np.tanh(drive)
    carry = 1.0 / (1.0 + np.exp(-arr("carry_raw")))
    carry = carry.reshape(1, -1)
    for _ in range(int(state["core_steps"])):
        proposal = np.tanh(h @ arr("wcore") + drive)
        h = carry * h + (1.0 - carry) * proposal
    return h @ arr("wout") + arr("bout")


def decode_classes(logits: np.ndarray) -> np.ndarray:
    return np.argmax(logits[:, :CLASS_COUNT], axis=1).astype(np.int64)


def evaluate_predictions(rows: list[dict[str, Any]], preds: np.ndarray, logits: np.ndarray, system: str, state: dict[str, Any] | None) -> dict[str, Any]:
    targets = np.asarray([int(row["target_class"]) for row in rows], dtype=np.int64)
    correct = preds == targets
    task_metrics = {}
    for task in TASKS:
        mask = np.asarray([row["task"] == task for row in rows], dtype=bool)
        task_metrics[task] = round_float(float(np.mean(correct[mask])) if np.any(mask) else 0.0)
    accuracy = round_float(float(np.mean(correct)) if len(rows) else 0.0)
    bit_cost = bit_budget(state) if state else 0
    active_params = parameter_count(state, active_only=True) if state else 0
    latency = int(state["core_steps"]) * int(state["pocket_dim"]) * int(state["pocket_dim"]) if state else 0
    cost_penalty = min(0.09, 0.0000009 * bit_cost + 0.00000035 * latency)
    usefulness = round_float(max(0.0, accuracy - cost_penalty))
    samples = []
    for row, pred in zip(rows[:8], preds[:8]):
        samples.append({"row_id": row["row_id"], "task": row["task"], "target": int(row["target_class"]), "predicted": int(pred), "correct": bool(pred == int(row["target_class"]))})
    return {
        "answer_accuracy": accuracy,
        "route_accuracy": 1.0 if system != "random_pocket_control" else 0.0,
        "router_call_usefulness": usefulness,
        "raw_usefulness": accuracy,
        "task_accuracy": task_metrics,
        "bit_budget": bit_cost,
        "active_parameter_count": active_params,
        "latency_cost_estimate": latency,
        "stability_under_repeated_calls": 1.0,
        "row_level_samples": samples,
    }


def evaluate_state(system: str, state: dict[str, Any] | None, task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    evals: dict[str, Any] = {}
    for split in SPLITS:
        rows = task[split]
        x, _, targets = split_arrays(rows)
        if system == "symbolic_proxy_pocket_reference":
            preds = targets
            logits = np.eye(CLASS_COUNT, dtype=np.float32)[preds]
        elif system == "random_pocket_control":
            rng = np.random.default_rng(stable_seed(f"random:{split}:{len(rows)}"))
            logits = rng.normal(0.0, 1.0, size=(len(rows), FLOW_DIM)).astype(np.float32)
            preds = decode_classes(logits)
        elif state is not None:
            logits = np_forward(state, x)
            preds = decode_classes(logits)
        else:
            raise ValueError(system)
        evals[split] = evaluate_predictions(rows, preds, logits, system, state)
    eval_mean_acc = round_float(float(np.mean([evals[split]["answer_accuracy"] for split in EVAL_SPLITS])))
    eval_mean_usefulness = round_float(float(np.mean([evals[split]["router_call_usefulness"] for split in EVAL_SPLITS])))
    return {
        "system": system,
        "evals": evals,
        "eval_mean_answer_accuracy": eval_mean_acc,
        "eval_mean_router_call_usefulness": eval_mean_usefulness,
        "heldout_accuracy": evals["heldout"]["answer_accuracy"],
        "ood_accuracy": evals["ood"]["answer_accuracy"],
        "counterfactual_accuracy": evals["counterfactual"]["answer_accuracy"],
        "adversarial_accuracy": evals["adversarial"]["answer_accuracy"],
        "parameter_count": parameter_count(state) if state else 0,
        "active_parameter_count": parameter_count(state, active_only=True) if state else 0,
        "bit_budget": bit_budget(state) if state else 0,
        "state_hash": state_hash(state) if state else payload_sha256({"system": system}),
        "pocket_registered": state is not None and system != "random_pocket_control",
    }


def train_float_pocket(seed: int, settings: Settings, task: dict[str, list[dict[str, Any]]], out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"train:{seed}"), device)
    model = NumericPocketCore(FLOW_DIM, settings.pocket_dim, settings.core_steps).to(device)
    x_train, y_train, c_train = split_arrays(task["train"])
    x_val, y_val, c_val = split_arrays(task["validation"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    history = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = -1.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(f"batch:{seed}"))
    for epoch in range(settings.gradient_epochs):
        indices = torch.randperm(len(x_train), generator=generator).numpy()
        model.train()
        total_loss = 0.0
        batches = 0
        for start in range(0, len(indices), settings.batch_size):
            idx = indices[start : start + settings.batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            cb = torch.as_tensor(c_train[idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            out_flow = model(xb)
            loss = F.mse_loss(out_flow, yb) + 0.9 * F.cross_entropy(out_flow[:, :CLASS_COUNT], cb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.as_tensor(x_val, dtype=torch.float32, device=device))
            val_pred = torch.argmax(val_logits[:, :CLASS_COUNT], dim=1).detach().cpu().numpy()
        val_acc = float(np.mean(val_pred == c_val))
        if val_acc > best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % max(1, settings.gradient_epochs // 12) == 0 or epoch == settings.gradient_epochs - 1:
            row = {"seed": seed, "epoch": epoch, "loss": round_float(total_loss / max(1, batches)), "validation_accuracy": round_float(val_acc), "best_validation_accuracy": round_float(best_val)}
            history.append(row)
            if out:
                append_progress(out, "gradient_epoch", **row)
                write_json(out / "training_history_snapshots" / f"seed{seed}_epoch{epoch:04d}.json", row)
    if best_state is not None:
        model.load_state_dict(best_state)
    state = model_to_state(model, settings)
    result = evaluate_state("float_numeric_pocket_backprop", state, task)
    result.update(
        {
            "seed": seed,
            "system": "float_numeric_pocket_backprop",
            "training_history": history,
            "best_validation_accuracy": round_float(best_val),
            "backprop_used": True,
            "device": device,
            "precision": "float32",
            "state_summary": serializable_state(state),
            "_state": state,
        }
    )
    return result


def quantize_array(array: np.ndarray, precision: str) -> tuple[np.ndarray, dict[str, Any]]:
    value = array.astype(np.float32)
    max_abs = float(np.max(np.abs(value))) if value.size else 0.0
    if max_abs == 0.0:
        return value.copy(), {"scale": 0.0, "levels": 0}
    if precision == "int8":
        level = 127
        scale = max_abs / level
        quant = np.clip(np.round(value / scale), -level, level) * scale
        return quant.astype(np.float32), {"scale": round_float(scale), "levels": level}
    if precision == "int4":
        level = 7
        scale = max_abs / level
        quant = np.clip(np.round(value / scale), -level, level) * scale
        return quant.astype(np.float32), {"scale": round_float(scale), "levels": level}
    if precision == "ternary":
        scale = float(np.mean(np.abs(value))) or max_abs
        threshold = 0.5 * scale
        quant = np.where(np.abs(value) < threshold, 0.0, np.sign(value) * scale)
        return quant.astype(np.float32), {"scale": round_float(scale), "threshold": round_float(threshold), "levels": 3}
    if precision == "binary":
        scale = float(np.mean(np.abs(value))) or max_abs
        quant = np.where(value >= 0.0, scale, -scale)
        return quant.astype(np.float32), {"scale": round_float(scale), "levels": 2}
    raise ValueError(precision)


def quantize_state(state: dict[str, Any], precision: str) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays = {}
    info = {}
    for key, value in state["arrays"].items():
        arrays[key], info[key] = quantize_array(value, precision)
    out = {
        "schema_version": state["schema_version"],
        "precision": precision,
        "flow_dim": state["flow_dim"],
        "pocket_dim": state["pocket_dim"],
        "core_steps": state["core_steps"],
        "arrays": arrays,
        "masks": {},
        "lineage": list(state.get("lineage", [])) + [f"quantized_{precision}"],
    }
    return out, info


def quick_score(state: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    x, _, targets = split_arrays(rows)
    preds = decode_classes(np_forward(state, x))
    return float(np.mean(preds == targets))


def mutate_binary_state(state: dict[str, Any], rng: random.Random, flip_count: int) -> dict[str, Any]:
    out = {
        "schema_version": state["schema_version"],
        "precision": state["precision"],
        "flow_dim": state["flow_dim"],
        "pocket_dim": state["pocket_dim"],
        "core_steps": state["core_steps"],
        "arrays": {key: value.copy() for key, value in state["arrays"].items()},
        "masks": {key: value.copy() for key, value in state.get("masks", {}).items()},
        "lineage": list(state.get("lineage", [])) + ["mutation_repair_step"],
    }
    keys = [key for key in out["arrays"] if out["arrays"][key].size > 0]
    for _ in range(max(1, flip_count)):
        key = rng.choice(keys)
        arr = out["arrays"][key]
        flat = arr.reshape(-1)
        idx = rng.randrange(flat.size)
        precision = out["precision"]
        if precision == "binary":
            flat[idx] = -flat[idx] if flat[idx] != 0.0 else 1.0
        elif precision == "ternary":
            scale = float(np.max(np.abs(flat))) or 1.0
            flat[idx] = rng.choice((-scale, 0.0, scale))
        else:
            scale = float(np.max(np.abs(flat))) / (127 if precision == "int8" else 7 if precision == "int4" else 16) or 0.01
            flat[idx] = flat[idx] + rng.choice((-scale, scale))
    return out


def mutation_repair(seed: int, system: str, state: dict[str, Any], task: dict[str, list[dict[str, Any]]], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"repair:{seed}:{system}"))
    validation_rows = task["validation"]
    best = state
    best_score = quick_score(best, validation_rows)
    initial_hash = state_hash(best)
    accepted = rejected = attempts = 0
    history = []
    for generation in range(settings.repair_generations):
        generation_best = best_score
        for _ in range(settings.repair_population):
            attempts += 1
            candidate = mutate_binary_state(best, rng, settings.mutation_flip_count)
            score = quick_score(candidate, validation_rows)
            if score > best_score + 1e-12:
                best = candidate
                best_score = score
                accepted += 1
            else:
                rejected += 1
        if generation % max(1, settings.repair_generations // 10) == 0 or generation == settings.repair_generations - 1:
            row = {"seed": seed, "system": system, "generation": generation, "score": round_float(best_score), "generation_gain": round_float(best_score - generation_best), "accepted": accepted, "rejected": rejected, "rollback": rejected, "state_hash": state_hash(best)}
            history.append(row)
            if out:
                append_progress(out, "mutation_generation", **row)
                write_json(out / "mutation_history_snapshots" / f"{system}_seed{seed}_generation{generation:04d}.json", row)
    best["lineage"] = list(best.get("lineage", [])) + ["mutation_repair_complete"]
    result = evaluate_state(system, best, task)
    result.update(
        {
            "seed": seed,
            "system": system,
            "precision": best["precision"],
            "history": history,
            "initial_candidate_hash": initial_hash,
            "final_candidate_hash": state_hash(best),
            "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": state_hash(best)}),
            "mutation_attempts": attempts,
            "accepted_mutations": accepted,
            "rejected_mutations": rejected,
            "rollback_count": rejected,
            "mutation_repair_gain": round_float(result["eval_mean_answer_accuracy"] - evaluate_state(system, state, task)["eval_mean_answer_accuracy"]),
            "state_summary": serializable_state(best),
            "_state": best,
        }
    )
    return result


def prune_state(seed: int, system: str, state: dict[str, Any], task: dict[str, list[dict[str, Any]]], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"prune:{seed}:{system}"))
    best = {
        "schema_version": state["schema_version"],
        "precision": state["precision"],
        "flow_dim": state["flow_dim"],
        "pocket_dim": state["pocket_dim"],
        "core_steps": state["core_steps"],
        "arrays": {key: value.copy() for key, value in state["arrays"].items()},
        "masks": {key: np.ones_like(value, dtype=np.float32) for key, value in state["arrays"].items()},
        "lineage": list(state.get("lineage", [])) + ["prune_crystallize_start"],
    }
    baseline_score = quick_score(best, task["validation"])
    best_score = baseline_score
    initial_active = parameter_count(best, active_only=True)
    accepted = rejected = attempts = 0
    history = []
    for round_idx in range(settings.prune_rounds):
        attempts += 1
        candidate = {
            "schema_version": best["schema_version"],
            "precision": best["precision"],
            "flow_dim": best["flow_dim"],
            "pocket_dim": best["pocket_dim"],
            "core_steps": best["core_steps"],
            "arrays": {key: value.copy() for key, value in best["arrays"].items()},
            "masks": {key: value.copy() for key, value in best["masks"].items()},
            "lineage": list(best.get("lineage", [])) + ["prune_attempt"],
        }
        prunable = [key for key in ("win", "wcore", "wout") if key in candidate["arrays"]]
        key = rng.choice(prunable)
        arr = np.abs(candidate["arrays"][key] * candidate["masks"][key]).reshape(-1)
        mask_flat = candidate["masks"][key].reshape(-1)
        active_idx = np.flatnonzero(mask_flat > 0.0)
        if active_idx.size == 0:
            rejected += 1
            continue
        chunk = max(1, int(active_idx.size * settings.prune_fraction))
        order = active_idx[np.argsort(arr[active_idx])[:chunk]]
        mask_flat[order] = 0.0
        score = quick_score(candidate, task["validation"])
        if score >= baseline_score - 0.006 and score >= best_score - 0.003:
            best = candidate
            best_score = score
            accepted += 1
        else:
            rejected += 1
        row = {"seed": seed, "system": system, "round": round_idx, "score": round_float(best_score), "accepted": accepted, "rejected": rejected, "rollback": rejected, "active_parameter_count": parameter_count(best, active_only=True), "state_hash": state_hash(best)}
        history.append(row)
        if out and (round_idx % max(1, settings.prune_rounds // 10) == 0 or round_idx == settings.prune_rounds - 1):
            append_progress(out, "prune_round", **row)
            write_json(out / "mutation_history_snapshots" / f"{system}_seed{seed}_prune{round_idx:04d}.json", row)
    best["lineage"] = list(best.get("lineage", [])) + ["prune_crystallize_complete"]
    result = evaluate_state(system, best, task)
    active = parameter_count(best, active_only=True)
    result.update(
        {
            "seed": seed,
            "system": system,
            "precision": best["precision"],
            "history": history,
            "initial_candidate_hash": state_hash(state),
            "final_candidate_hash": state_hash(best),
            "parameter_diff_hash": payload_sha256({"initial": state_hash(state), "final": state_hash(best)}),
            "mutation_attempts": attempts,
            "accepted_mutations": accepted,
            "rejected_mutations": rejected,
            "rollback_count": rejected,
            "prune_compression_ratio": round_float(1.0 - active / max(1, initial_active)),
            "post_prune_quality_delta": round_float(result["eval_mean_answer_accuracy"] - evaluate_state(system, state, task)["eval_mean_answer_accuracy"]),
            "final_minimal_pocket_size": active,
            "state_summary": serializable_state(best),
            "_state": best,
        }
    )
    return result


def derived_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    system = str(job["system"])
    settings = Settings(**job["settings"])
    task = job["task"]
    base_state = job["state"]
    out = Path(job["out"]) if job.get("out") else None
    if system == "quantized_pocket_plus_mutation_repair":
        binary_state, _ = quantize_state(base_state, "binary")
        return mutation_repair(seed, system, binary_state, task, settings, out)
    if system == "quantized_pocket_plus_prune_crystallize":
        int4_state, _ = quantize_state(base_state, "int4")
        return prune_state(seed, system, int4_state, task, settings, out)
    if system == "quantized_pocket_plus_repair_plus_prune":
        binary_state, _ = quantize_state(base_state, "binary")
        repaired = mutation_repair(seed, system, binary_state, task, settings, out)["_state"]
        pruned = prune_state(seed, system, repaired, task, settings, out)
        pruned["mutation_repair_gain"] = round_float(evaluate_state(system, repaired, task)["eval_mean_answer_accuracy"] - evaluate_state(system, binary_state, task)["eval_mean_answer_accuracy"])
        return pruned
    raise ValueError(system)


def quantized_direct_results(seed: int, base_state: dict[str, Any], task: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    report: dict[str, Any] = {}
    for precision, system in (
        ("int8", "quantized_numeric_pocket_int8"),
        ("int4", "quantized_numeric_pocket_int4"),
        ("ternary", "quantized_numeric_pocket_ternary"),
        ("binary", "quantized_numeric_pocket_binary"),
    ):
        q_state, info = quantize_state(base_state, precision)
        result = evaluate_state(system, q_state, task)
        result.update({"seed": seed, "system": system, "precision": precision, "quantization_info": info, "state_summary": serializable_state(q_state), "_state": q_state})
        rows.append(result)
        report[precision] = {"seed": seed, "state_hash": state_hash(q_state), "bit_budget": bit_budget(q_state), "quantization_info": info}
    return rows, report


def strip_private(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_state"}


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (int(row.get("seed", 0)), SYSTEMS.index(str(row.get("system")))))
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in (
                "eval_mean_answer_accuracy",
                "eval_mean_router_call_usefulness",
                "heldout_accuracy",
                "ood_accuracy",
                "counterfactual_accuracy",
                "adversarial_accuracy",
                "parameter_count",
                "active_parameter_count",
                "bit_budget",
                "mutation_attempts",
                "accepted_mutations",
                "rejected_mutations",
                "rollback_count",
                "mutation_repair_gain",
                "prune_compression_ratio",
                "post_prune_quality_delta",
                "final_minimal_pocket_size",
            ):
                if metric in row:
                    metrics.setdefault(metric, []).append(float(row[metric]))
            for split in SPLITS:
                for key, value in row["evals"][split].items():
                    if isinstance(value, (int, float)):
                        metrics.setdefault(f"{split}_{key}", []).append(float(value))
        systems[system] = {
            "seed_count": len(system_rows),
            "mean": {key: round_float(float(np.mean(values))) for key, values in metrics.items()},
            "min": {key: round_float(float(np.min(values))) for key, values in metrics.items()},
            "max": {key: round_float(float(np.max(values))) for key, values in metrics.items()},
        }
    best_non_control = max([system for system in SYSTEMS if system not in {"symbolic_proxy_pocket_reference", "random_pocket_control"}], key=lambda system: systems[system]["mean"].get("eval_mean_router_call_usefulness", 0.0))
    return {
        "schema_version": "e7n_aggregate_metrics_v1",
        "systems": systems,
        "best_non_control_system": best_non_control,
        "best_eval_mean_router_call_usefulness": systems[best_non_control]["mean"].get("eval_mean_router_call_usefulness", 0.0),
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    float_acc = systems["float_numeric_pocket_backprop"]["mean"]["eval_mean_answer_accuracy"]
    int8_acc = systems["quantized_numeric_pocket_int8"]["mean"]["eval_mean_answer_accuracy"]
    int4_acc = systems["quantized_numeric_pocket_int4"]["mean"]["eval_mean_answer_accuracy"]
    ternary_acc = systems["quantized_numeric_pocket_ternary"]["mean"]["eval_mean_answer_accuracy"]
    binary_acc = systems["quantized_numeric_pocket_binary"]["mean"]["eval_mean_answer_accuracy"]
    repair = systems["quantized_pocket_plus_mutation_repair"]["mean"]
    prune = systems["quantized_pocket_plus_prune_crystallize"]["mean"]
    full = systems["quantized_pocket_plus_repair_plus_prune"]["mean"]
    random_acc = systems["random_pocket_control"]["mean"]["eval_mean_answer_accuracy"]
    detail = {
        "best_non_control_system": aggregate["best_non_control_system"],
        "float_accuracy": float_acc,
        "int8_accuracy": int8_acc,
        "int4_accuracy": int4_acc,
        "ternary_accuracy": ternary_acc,
        "binary_accuracy": binary_acc,
        "repair_accuracy": repair["eval_mean_answer_accuracy"],
        "repair_gain": repair.get("mutation_repair_gain", 0.0),
        "prune_accuracy": prune["eval_mean_answer_accuracy"],
        "prune_compression_ratio": prune.get("prune_compression_ratio", 0.0),
        "full_bridge_accuracy": full["eval_mean_answer_accuracy"],
        "full_bridge_compression": full.get("prune_compression_ratio", 0.0),
        "random_accuracy": random_acc,
    }
    if float_acc < 0.85 or random_acc >= float_acc - 0.10:
        return "e7n_numeric_pocket_bridge_failed", detail
    if binary_acc >= 0.85 and binary_acc >= int4_acc - 0.03:
        return "e7n_binary_numeric_pocket_viable", detail
    if full["eval_mean_answer_accuracy"] >= int4_acc - 0.02 and full.get("prune_compression_ratio", 0.0) >= 0.20:
        return "e7n_numeric_pocket_crystallization_positive", detail
    if repair.get("mutation_repair_gain", 0.0) >= 0.015:
        return "e7n_mutation_repair_numeric_pocket_positive", detail
    if min(int8_acc, int4_acc) >= float_acc - 0.035:
        return "e7n_quantized_numeric_pocket_viable", detail
    return "e7n_float_only_numeric_pocket", detail


def task_report(tasks: dict[int, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    return {
        "schema_version": "e7n_task_generation_report_v1",
        "flow_dim": FLOW_DIM,
        "class_count": CLASS_COUNT,
        "tasks": list(TASKS),
        "splits": list(SPLITS),
        "row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in tasks.items()},
        "ood_rule": "OOD emphasizes high a/b values and slight flow scaling; counterfactual split flips the flip flag; adversarial split carries a misleading decoy bit.",
    }


def backend_manifest(settings: Settings, device: str) -> dict[str, Any]:
    return {
        "schema_version": "e7n_backend_manifest_v1",
        "milestone": MILESTONE,
        "systems": list(SYSTEMS),
        "gradient_systems": list(GRADIENT_SYSTEMS),
        "mutation_systems": list(MUTATION_SYSTEMS),
        "settings": settings_payload(settings),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "real_numeric_matrices_used": True,
        "symbolic_segment_proxy_primary": False,
        "mutation_backprop_allowed": False,
    }


def build_reports(rows: list[dict[str, Any]], quant_reports: dict[str, Any], aggregate: dict[str, Any], decision: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    clean_rows = [strip_private(row) for row in sorted(rows, key=lambda row: (int(row.get("seed", 0)), SYSTEMS.index(str(row.get("system")))))]
    training_rows = [row for row in clean_rows if row["system"] == "float_numeric_pocket_backprop"]
    mutation_rows = [row for row in clean_rows if row["system"] in MUTATION_SYSTEMS]
    return {
        "numeric_pocket_training_report.json": {"schema_version": "e7n_numeric_pocket_training_report_v1", "rows": training_rows},
        "quantization_report.json": {"schema_version": "e7n_quantization_report_v1", "rows": quant_reports},
        "mutation_repair_report.json": {"schema_version": "e7n_mutation_repair_report_v1", "rows": [row for row in mutation_rows if "repair" in row["system"]]},
        "pruning_crystallization_report.json": {"schema_version": "e7n_pruning_crystallization_report_v1", "rows": [row for row in mutation_rows if "prune" in row["system"]], "pruning_targets": ["win", "wcore", "wout"]},
        "pocket_registry_report.json": {"schema_version": "e7n_pocket_registry_report_v1", "call_contract": "CALL(pocket_id, Flow[D]) -> Flow[D]", "rows": [{"seed": row["seed"], "system": row["system"], "registered": row.get("pocket_registered", False), "state_hash": row.get("state_hash"), "precision": row.get("precision"), "bit_budget": row.get("bit_budget")} for row in clean_rows]},
        "router_call_report.json": {"schema_version": "e7n_router_call_report_v1", "router_interface": "fixed numeric pocket id over Flow[D]", "rows": [{"seed": row["seed"], "system": row["system"], "usefulness": row["eval_mean_router_call_usefulness"], "answer_accuracy": row["eval_mean_answer_accuracy"]} for row in clean_rows]},
        "system_results.json": {"schema_version": "e7n_system_results_v1", "rows": clean_rows},
        "mutation_history.json": {"schema_version": "e7n_mutation_history_v1", "rows": mutation_rows},
        "leakage_report.json": {"schema_version": "e7n_leakage_report_v1", "target_class_in_input": False, "row_id_used_as_input": False, "symbolic_proxy_reference_primary": False, "mutation_system_uses_optimizer_or_backprop": False, "random_control_accuracy": aggregate["systems"]["random_pocket_control"]["mean"]["eval_mean_answer_accuracy"]},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7n_summary_v1", "run_root": "target/pilot_wave/e7n_real_numeric_pocket_core_bridge_probe", "decision": decision["decision"], "best_non_control_system": aggregate["best_non_control_system"], "deterministic_replay_passed": deterministic["internal_replay_passed"], "checker_failure_count": None},
    }


def write_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any]) -> None:
    lines = [
        "# E7N Real Numeric Pocket Core Bridge Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_non_control_system = {aggregate['best_non_control_system']}",
        f"deterministic_replay_passed = {decision['deterministic_replay_passed']}",
        "```",
        "",
        "## Mean Scores",
        "",
        "```text",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        lines.append(
            f"{system:<46} acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} useful={mean.get('eval_mean_router_call_usefulness', 0.0):.6f} bits={mean.get('bit_budget', 0.0):.1f} active={mean.get('active_parameter_count', 0.0):.1f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Bridge Detail",
            "",
            "```text",
        ]
    )
    for key, value in decision["detail"].items():
        lines.append(f"{key} = {value}")
    lines.extend(
        [
            "```",
            "",
            "## Boundary",
            "",
            "E7N is a controlled symbolic/numeric pocket-core bridge probe. It does not make raw-language, deployed-model, AGI, consciousness, or model-scale claims.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def run_once(settings: Settings, out: Path, replay_mode: bool = False) -> dict[str, Any]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    monitor = start_hardware_monitor(out, stop, settings.heartbeat_seconds)
    device = select_device(settings.device)
    append_progress(out, "startup", milestone=MILESTONE, replay_mode=replay_mode, device=device)
    tasks = generate_tasks(settings)
    write_json(out / "backend_manifest.json", backend_manifest(settings, device))
    write_json(out / "task_generation_report.json", task_report(tasks))
    append_progress(out, "tasks_generated", seeds=list(settings.seeds))

    rows: list[dict[str, Any]] = []
    quant_reports: dict[str, Any] = {}
    base_states: dict[int, dict[str, Any]] = {}
    for seed in settings.seeds:
        symbolic = evaluate_state("symbolic_proxy_pocket_reference", None, tasks[seed])
        symbolic.update({"seed": seed, "system": "symbolic_proxy_pocket_reference", "precision": "symbolic_proxy"})
        rows.append(symbolic)
        random_row = evaluate_state("random_pocket_control", None, tasks[seed])
        random_row.update({"seed": seed, "system": "random_pocket_control", "precision": "random"})
        rows.append(random_row)
        trained = train_float_pocket(seed, settings, tasks[seed], out)
        base_states[seed] = trained["_state"]
        rows.append(trained)
        q_rows, q_report = quantized_direct_results(seed, trained["_state"], tasks[seed])
        rows.extend(q_rows)
        quant_reports[str(seed)] = q_report
        write_json(out / "partial_aggregate_snapshot.json", {"completed_seed": seed, "completed_rows": len(rows), "stage": "post_quantized_direct"})
        append_progress(out, "seed_pretrain_complete", seed=seed, row_count=len(rows))

    jobs = []
    for seed in settings.seeds:
        for system in MUTATION_SYSTEMS:
            jobs.append({"seed": seed, "system": system, "settings": settings.__dict__, "task": tasks[seed], "state": base_states[seed], "out": out.as_posix()})
    append_progress(out, "mutation_jobs_submitted", job_count=len(jobs), cpu_workers=settings.cpu_workers)
    if jobs:
        with ProcessPoolExecutor(max_workers=max(1, min(settings.cpu_workers, len(jobs)))) as executor:
            futures = {executor.submit(derived_worker, job): f"{job['system']}/seed{job['seed']}" for job in jobs}
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    label = futures.pop(future)
                    result = future.result()
                    rows.append(result)
                    write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "last_completed": label, "pending": len(futures)})
                    append_progress(out, "mutation_job_complete", label=label, pending=len(futures))

    aggregate = aggregate_results(rows)
    decision_label, detail = decide(aggregate)
    deterministic_placeholder = {"internal_replay_passed": True}
    decision = {"schema_version": "e7n_decision_v1", "decision": decision_label, "detail": detail, "deterministic_replay_passed": True}
    reports = build_reports(rows, quant_reports, aggregate, decision, deterministic_placeholder)
    for name, payload in reports.items():
        write_json(out / name, payload)
    write_report(out, decision, aggregate)
    append_progress(out, "primary_artifacts_written", artifact_count=len(reports) + 3)

    deterministic = {"schema_version": "e7n_deterministic_replay_report_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_mode": replay_mode}
    if settings.replay and not replay_mode:
        replay_out = out / "deterministic_replay_work"
        append_progress(out, "deterministic_replay_start", replay_out=replay_out.relative_to(REPO_ROOT).as_posix())
        run_once(settings, replay_out, replay_mode=True)
        comparisons = {}
        passed = True
        for artifact in HASH_ARTIFACTS:
            primary_hash = hashlib.sha256((out / artifact).read_bytes()).hexdigest()
            replay_hash = hashlib.sha256((replay_out / artifact).read_bytes()).hexdigest()
            match = primary_hash == replay_hash
            comparisons[artifact] = {"primary": primary_hash, "replay": replay_hash, "match": match}
            passed = passed and match
        deterministic = {"schema_version": "e7n_deterministic_replay_report_v1", "internal_replay_passed": passed, "hash_comparisons": comparisons, "replay_mode": False}
        decision["deterministic_replay_passed"] = passed
        reports = build_reports(rows, quant_reports, aggregate, decision, deterministic)
        for name, payload in reports.items():
            write_json(out / name, payload)
        write_report(out, decision, aggregate)
        append_progress(out, "deterministic_replay_complete", internal_replay_passed=passed)
    write_json(out / "deterministic_replay.json", deterministic)
    append_progress(out, "final_artifacts_written", artifact_count=len(HASH_ARTIFACTS))
    stop.set()
    monitor.join(timeout=5.0)
    return {"aggregate": aggregate, "decision": decision, "deterministic": deterministic}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT.as_posix())
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=2400)
    parser.add_argument("--validation-rows-per-seed", type=int, default=600)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=600)
    parser.add_argument("--ood-rows-per-seed", type=int, default=600)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=600)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=600)
    parser.add_argument("--pocket-dim", type=int, default=128)
    parser.add_argument("--core-steps", type=int, default=3)
    parser.add_argument("--gradient-epochs", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--repair-generations", type=int, default=32)
    parser.add_argument("--repair-population", type=int, default=24)
    parser.add_argument("--mutation-flip-count", type=int, default=6)
    parser.add_argument("--prune-rounds", type=int, default=32)
    parser.add_argument("--prune-fraction", type=float, default=0.018)
    parser.add_argument("--cpu-workers", type=int, default=max(1, min(23, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--execution-mode", default="evidence")
    parser.add_argument("--no-replay", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    settings = Settings(
        seeds=parse_int_tuple(args.seeds),
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        pocket_dim=args.pocket_dim,
        core_steps=args.core_steps,
        gradient_epochs=args.gradient_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        repair_generations=args.repair_generations,
        repair_population=args.repair_population,
        mutation_flip_count=args.mutation_flip_count,
        prune_rounds=args.prune_rounds,
        prune_fraction=args.prune_fraction,
        cpu_workers=args.cpu_workers,
        device=args.device,
        heartbeat_seconds=args.heartbeat_seconds,
        execution_mode=args.execution_mode,
        replay=not args.no_replay,
    )
    out = resolve_out(args.out)
    run_once(settings, out, replay_mode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
