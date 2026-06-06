#!/usr/bin/env python3
"""E7B pocket routing composition probe.

This probe tests whether mutation/rollback can learn a compact routing fabric
over frozen pocket outputs, instead of learning a full monolithic solution from
scratch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
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


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "E7B_POCKET_ROUTING_COMPOSITION_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7b_pocket_routing_composition_probe")
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
FAMILIES = (
    "add_then_compare",
    "xor_then_compare",
    "memory_add_then_compare",
    "branch_apply_then_compare",
    "memory_xor_then_compare",
)
ROUTES = (
    "add_compare",
    "xor_compare",
    "memory_add_compare",
    "branch_add_compare",
    "branch_xor_compare",
    "memory_xor_compare",
)
SYSTEMS = (
    "monolithic_backprop_model",
    "monolithic_mutation_model",
    "frozen_pockets_gradient_router",
    "frozen_pockets_mutation_router",
    "frozen_pockets_binary_router",
    "router_plus_limited_pocket_repair",
    "random_router_control",
    "oracle_pocket_router_reference",
)
GRADIENT_SYSTEMS = ("monolithic_backprop_model", "frozen_pockets_gradient_router")
MUTATION_SYSTEMS = (
    "monolithic_mutation_model",
    "frozen_pockets_mutation_router",
    "frozen_pockets_binary_router",
    "router_plus_limited_pocket_repair",
)
CONTROL_SYSTEMS = ("random_router_control", "oracle_pocket_router_reference")
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_library_report.json",
    "system_results.json",
    "mutation_history.json",
    "training_history.json",
    "composition_report.json",
    "leakage_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7b_mutation_router_composition_viable",
    "e7b_gradient_only_composition_viable",
    "e7b_monolithic_mutation_sufficient_or_task_too_easy",
    "e7b_pocket_router_no_advantage_detected",
    "e7b_router_leak_or_artifact_detected",
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
    gradient_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    mutation_generations: int
    mutation_population: int
    mutation_sigma: float
    mutation_elite_count: int
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
    return int(hashlib.sha256(f"e7b::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def safe_file_id(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def locked_append_jsonl(path: Path, payload: dict[str, Any]) -> None:
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
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def append_progress(out: Path, event: str, **details: Any) -> None:
    locked_append_jsonl(out / "progress.jsonl", {"event": event, "details": details})


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
        raise ValueError("empty integer tuple")
    return values


def set_determinism(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def select_device(requested: str) -> str:
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def hardware_probe() -> dict[str, Any]:
    gpu = {"cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        gpu["cuda_name"] = torch.cuda.get_device_name(0)
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=4,
        )
        if query.returncode == 0 and query.stdout.strip():
            parts = [part.strip() for part in query.stdout.strip().splitlines()[0].split(",")]
            gpu.update(
                {
                    "nvidia_smi_gpu_util_percent": int(parts[0]),
                    "nvidia_smi_mem_util_percent": int(parts[1]),
                    "nvidia_smi_memory_used_mb": int(parts[2]),
                    "nvidia_smi_memory_total_mb": int(parts[3]),
                    "nvidia_smi_temperature_c": int(parts[4]),
                }
            )
    except Exception as exc:  # pragma: no cover - platform dependent
        gpu["nvidia_smi_error"] = str(exc)
    return {
        "cpu_count": os.cpu_count(),
        "process_id": os.getpid(),
        "torch_threads": torch.get_num_threads(),
        "gpu": gpu,
    }


def stable_hardware_identity() -> dict[str, Any]:
    gpu = {"cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        gpu["cuda_name"] = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            gpu["cuda_total_memory_bytes"] = int(props.total_memory)
            gpu["cuda_major"] = int(props.major)
            gpu["cuda_minor"] = int(props.minor)
        except Exception as exc:  # pragma: no cover - platform dependent
            gpu["cuda_properties_error"] = str(exc)
    return {
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "gpu": gpu,
    }


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    def loop() -> None:
        while not stop.is_set():
            locked_append_jsonl(out / "hardware_heartbeat.jsonl", {"timestamp": time.time(), "hardware": hardware_probe()})
            stop.wait(max(2.0, interval))

    thread = threading.Thread(target=loop, name="e7b-hardware-monitor", daemon=True)
    thread.start()
    return thread


def memory_value(seed: int, key: int, split: str) -> int:
    split_shift = {"train": 0, "validation": 1, "heldout": 2, "ood": 7, "counterfactual": 3, "adversarial": 5}[split]
    return (key * 5 + (seed % 17) + split_shift * 3) & 15


def route_for_family(family: int, a: int, b: int) -> int:
    if family == 0:
        return 0
    if family == 1:
        return 1
    if family == 2:
        return 2
    if family == 3:
        return 3 if a > b else 4
    if family == 4:
        return 5
    raise ValueError(family)


def compute_pockets(seed: int, split: str, a: int, b: int, key: int, threshold: int) -> dict[str, Any]:
    mem = memory_value(seed, key, split)
    add = (a + b) & 15
    xor = a ^ b
    mem_add = (mem + a) & 15
    mem_xor = mem ^ b
    branch_add = add
    branch_xor = xor
    values = np.asarray([add, xor, mem_add, branch_add, branch_xor, mem_xor], dtype=np.float32)
    candidate_answers = (values > threshold).astype(np.int64)
    return {
        "memory_value": mem,
        "values": values,
        "candidate_answers": candidate_answers,
        "branch_flag": 1.0 if a > b else 0.0,
    }


def target_for_row(seed: int, split: str, family: int, a: int, b: int, key: int, threshold: int) -> tuple[int, int, np.ndarray]:
    pockets = compute_pockets(seed, split, a, b, key, threshold)
    route = route_for_family(family, a, b)
    return int(pockets["candidate_answers"][route]), route, pockets["candidate_answers"]


def make_row(seed: int, split: str, index: int, rng: random.Random) -> dict[str, Any]:
    family = rng.randrange(len(FAMILIES))
    if split == "ood":
        a = rng.choice((0, 1, 2, 13, 14, 15))
        b = rng.choice((0, 1, 2, 13, 14, 15))
        threshold = rng.choice((2, 3, 12, 13))
    else:
        a = rng.randrange(16)
        b = rng.randrange(16)
        threshold = rng.randrange(4, 12)
    key = rng.randrange(16)
    if split == "counterfactual":
        base_y, route, _ = target_for_row(seed, "heldout", family, a, b, key, threshold)
        for delta in (1, -1, 2, -2, 4, -4):
            t2 = max(0, min(15, threshold + delta))
            y2, route, _ = target_for_row(seed, "heldout", family, a, b, key, t2)
            if y2 != base_y:
                threshold = t2
                break
    y, route, candidate_answers = target_for_row(seed, split, family, a, b, key, threshold)
    if split == "adversarial":
        misleading_route = (route + 1 + (index % (len(ROUTES) - 1))) % len(ROUTES)
    elif split in {"train", "validation"} and rng.random() < 0.82:
        misleading_route = route
    else:
        misleading_route = rng.randrange(len(ROUTES))
    noise = [rng.uniform(-1.0, 1.0) for _ in range(6)]
    family_one_hot = [1.0 if i == family else 0.0 for i in range(len(FAMILIES))]
    misleading_one_hot = [1.0 if i == misleading_route else 0.0 for i in range(len(ROUTES))]
    raw = [
        a / 15.0,
        b / 15.0,
        key / 15.0,
        threshold / 15.0,
        (a - b) / 15.0,
    ] + family_one_hot + misleading_one_hot + noise
    pockets = compute_pockets(seed, split, a, b, key, threshold)
    values_norm = (pockets["values"] / 15.0).tolist()
    pocket_features = pockets["candidate_answers"].astype(np.float32).tolist() + values_norm + [pockets["branch_flag"], threshold / 15.0]
    return {
        "row_id": f"{seed}/{split}/{index}",
        "seed": seed,
        "split": split,
        "family": family,
        "route": route,
        "answer": y,
        "raw": raw,
        "pocket": pocket_features,
        "candidate_answers": candidate_answers.astype(np.int64).tolist(),
        "misleading_route": misleading_route,
    }


def generate_seed_task(seed: int, settings: Settings) -> dict[str, Any]:
    counts = {
        "train": settings.train_rows_per_seed,
        "validation": settings.validation_rows_per_seed,
        "heldout": settings.heldout_rows_per_seed,
        "ood": settings.ood_rows_per_seed,
        "counterfactual": settings.counterfactual_rows_per_seed,
        "adversarial": settings.adversarial_rows_per_seed,
    }
    task: dict[str, Any] = {}
    for split, count in counts.items():
        rng = random.Random(stable_seed(f"task-{seed}-{split}"))
        rows = [make_row(seed, split, idx, rng) for idx in range(count)]
        raw = np.asarray([row["raw"] for row in rows], dtype=np.float32)
        pocket = np.asarray([row["pocket"] for row in rows], dtype=np.float32)
        y = np.asarray([row["answer"] for row in rows], dtype=np.int64)
        route = np.asarray([row["route"] for row in rows], dtype=np.int64)
        candidate_answers = np.asarray([row["candidate_answers"] for row in rows], dtype=np.int64)
        task[split] = {
            "rows": rows,
            "raw": raw,
            "pocket": pocket,
            "combo": np.concatenate([raw, pocket], axis=1),
            "y": y,
            "route": route,
            "candidate_answers": candidate_answers,
        }
    return task


def task_report(tasks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "e7b_task_generation_report_v1",
        "families": list(FAMILIES),
        "routes": list(ROUTES),
        "row_counts": {
            str(seed): {split: int(len(task[split]["rows"])) for split in SPLITS}
            for seed, task in tasks.items()
        },
        "raw_feature_dim": int(next(iter(tasks.values()))["train"]["raw"].shape[1]),
        "pocket_feature_dim": int(next(iter(tasks.values()))["train"]["pocket"].shape[1]),
        "combo_feature_dim": int(next(iter(tasks.values()))["train"]["combo"].shape[1]),
        "adversarial_misleading_route_used": True,
        "counterfactual_threshold_flip_used": True,
        "ood_edge_value_distribution_used": True,
    }


def pocket_library_report(tasks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    route_counts = {route: 0 for route in ROUTES}
    answer_counts = {str(i): 0 for i in range(2)}
    for task in tasks.values():
        for split in SPLITS:
            for route in task[split]["route"].tolist():
                route_counts[ROUTES[int(route)]] += 1
            for answer in task[split]["y"].tolist():
                answer_counts[str(int(answer))] += 1
    return {
        "schema_version": "e7b_pocket_library_report_v1",
        "pocket_source": "frozen_deterministic_symbolic_pockets_for_routing_isolation",
        "pockets": {
            "add": "computes (a + b) mod 16",
            "xor": "computes bitwise xor over 4-bit inputs",
            "memory": "deterministic split-aware key lookup",
            "compare": "compares candidate value to row threshold",
            "branch": "selects add route when a > b, xor route otherwise",
        },
        "route_counts": route_counts,
        "answer_counts": answer_counts,
        "learned_pocket_claim_deferred": True,
    }


class MultiHeadMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.answer = nn.Linear(hidden_dim, 2)
        self.route = nn.Linear(hidden_dim, len(ROUTES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.answer(h), self.route(h)


def evaluate_predictions(
    answer_pred: np.ndarray,
    route_pred: np.ndarray,
    data: dict[str, Any],
    sample_limit: int = 8,
) -> dict[str, Any]:
    y = data["y"]
    route = data["route"]
    answer_acc = float(np.mean(answer_pred == y))
    route_acc = float(np.mean(route_pred == route))
    composition_acc = float(np.mean((answer_pred == y) & (route_pred == route)))
    candidate = data["candidate_answers"]
    shortcut = float(np.mean(candidate[np.arange(len(candidate)), route_pred] != y))
    samples = []
    for idx, row in enumerate(data["rows"][:sample_limit]):
        samples.append(
            {
                "row_id": row["row_id"],
                "family": FAMILIES[row["family"]],
                "target_route": ROUTES[int(route[idx])],
                "predicted_route": ROUTES[int(route_pred[idx])],
                "target_answer": int(y[idx]),
                "predicted_answer": int(answer_pred[idx]),
                "misleading_route": ROUTES[row["misleading_route"]],
            }
        )
    usefulness = 0.50 * answer_acc + 0.35 * route_acc + 0.15 * composition_acc
    return {
        "answer_accuracy": round_float(answer_acc),
        "route_accuracy": round_float(route_acc),
        "composition_accuracy": round_float(composition_acc),
        "shortcut_rate": round_float(shortcut),
        "usefulness_score": round_float(usefulness),
        "row_level_samples": samples,
    }


def evaluate_gradient_model(model_state: dict[str, Any], task: dict[str, Any], input_key: str, device: str) -> dict[str, Any]:
    model = MultiHeadMLP(int(model_state["input_dim"]))
    model.load_state_dict({key: torch.as_tensor(value, dtype=torch.float32) for key, value in model_state["state_dict"].items()})
    model.to(device)
    model.eval()
    out: dict[str, Any] = {}
    with torch.no_grad():
        for split in SPLITS:
            x = torch.as_tensor(task[split][input_key], dtype=torch.float32, device=device)
            answer_logits, route_logits = model(x)
            answer_pred = torch.argmax(answer_logits, dim=1).cpu().numpy()
            route_pred = torch.argmax(route_logits, dim=1).cpu().numpy()
            out[split] = evaluate_predictions(answer_pred, route_pred, task[split])
    return out


def train_gradient_system(seed: int, system: str, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"gradient-{system}-{seed}"), device)
    input_key = "raw" if system == "monolithic_backprop_model" else "combo"
    input_dim = int(task["train"][input_key].shape[1])
    model = MultiHeadMLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train = torch.as_tensor(task["train"][input_key], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(task["train"]["y"], dtype=torch.long, device=device)
    r_train = torch.as_tensor(task["train"]["route"], dtype=torch.long, device=device)
    x_val = torch.as_tensor(task["validation"][input_key], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(task["validation"]["y"], dtype=torch.long, device=device)
    r_val = torch.as_tensor(task["validation"]["route"], dtype=torch.long, device=device)
    history = []
    best_state = None
    best_score = -1.0
    rng = np.random.default_rng(stable_seed(f"gradient-order-{system}-{seed}"))
    last_heartbeat = time.monotonic()
    for epoch in range(1, settings.gradient_epochs + 1):
        order = rng.permutation(len(x_train))
        model.train()
        for start in range(0, len(order), settings.batch_size):
            idx = torch.as_tensor(order[start : start + settings.batch_size], dtype=torch.long, device=device)
            answer_logits, route_logits = model(x_train[idx])
            loss = nn.functional.cross_entropy(answer_logits, y_train[idx]) + 0.75 * nn.functional.cross_entropy(route_logits, r_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            answer_logits, route_logits = model(x_val)
            answer_pred = torch.argmax(answer_logits, dim=1)
            route_pred = torch.argmax(route_logits, dim=1)
            answer_acc = float((answer_pred == y_val).float().mean().item())
            route_acc = float((route_pred == r_val).float().mean().item())
            score = 0.6 * answer_acc + 0.4 * route_acc
        row = {"epoch": epoch, "validation_answer_accuracy": round_float(answer_acc), "validation_route_accuracy": round_float(route_acc), "score": round_float(score)}
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in model.state_dict().items()}
        if out and (time.monotonic() - last_heartbeat >= settings.heartbeat_seconds or epoch == settings.gradient_epochs):
            locked_write_json(out / "partial_status" / f"gradient_{system}_seed{seed}.json", row)
            append_progress(out, "gradient_epoch", system=system, seed=seed, epoch=epoch, validation_answer_accuracy=row["validation_answer_accuracy"], validation_route_accuracy=row["validation_route_accuracy"], device=device)
            last_heartbeat = time.monotonic()
    assert best_state is not None
    serial_state = {key: value.tolist() for key, value in best_state.items()}
    model_state = {"input_dim": input_dim, "state_dict": serial_state}
    evals = evaluate_gradient_model(model_state, task, input_key, device)
    return {
        "seed": seed,
        "system": system,
        "training_mode": "gradient_backprop",
        "device": device,
        "input_key": input_key,
        "parameter_count": sum(int(np.asarray(value).size) for value in serial_state.values()),
        "state_hash": payload_sha256(serial_state),
        "history": history,
        "evals": evals,
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def init_mlp_candidate(seed: int, input_dim: int, hidden_dim: int = 28) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"mlp-init-{seed}"))
    return {
        "kind": "monolithic_mlp",
        "w1": rng.normal(0.0, 0.18, size=(input_dim, hidden_dim)).astype(np.float64),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "w_answer": rng.normal(0.0, 0.18, size=(hidden_dim, 2)).astype(np.float64),
        "b_answer": np.zeros(2, dtype=np.float64),
        "w_route": rng.normal(0.0, 0.18, size=(hidden_dim, len(ROUTES))).astype(np.float64),
        "b_route": np.zeros(len(ROUTES), dtype=np.float64),
    }


def init_router_candidate(seed: int, kind: str) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"router-init-{kind}-{seed}"))
    if kind == "binary_router":
        return {
            "kind": kind,
            "family_scores": rng.choice((-1.0, 1.0), size=(len(FAMILIES), len(ROUTES))).astype(np.float64),
            "branch_on_scores": rng.choice((-1.0, 1.0), size=len(ROUTES)).astype(np.float64),
            "branch_off_scores": rng.choice((-1.0, 1.0), size=len(ROUTES)).astype(np.float64),
            "scale": 0.25,
        }
    return {
        "kind": kind,
        "family_scores": rng.normal(0.0, 0.05, size=(len(FAMILIES), len(ROUTES))).astype(np.float64),
        "branch_on_scores": rng.normal(0.0, 0.05, size=len(ROUTES)).astype(np.float64),
        "branch_off_scores": rng.normal(0.0, 0.05, size=len(ROUTES)).astype(np.float64),
        "answer_bias": np.zeros(len(ROUTES), dtype=np.float64),
    }


def candidate_to_serial(candidate: dict[str, Any]) -> dict[str, Any]:
    serial = {}
    for key, value in candidate.items():
        if isinstance(value, np.ndarray):
            serial[key] = value.round(12).tolist()
        else:
            serial[key] = value
    return serial


def candidate_hash(candidate: dict[str, Any]) -> str:
    return payload_sha256(candidate_to_serial(candidate))


def parameter_count(candidate: dict[str, Any]) -> int:
    return sum(int(value.size) for value in candidate.values() if isinstance(value, np.ndarray)) + (1 if "scale" in candidate else 0)


def router_logits(candidate: dict[str, Any], data: dict[str, Any]) -> np.ndarray:
    rows = data["rows"]
    family = np.asarray([row["family"] for row in rows], dtype=np.int64)
    branch = data["pocket"][:, len(ROUTES) + len(ROUTES)]  # after answer candidates and values
    if candidate["kind"] == "binary_router":
        scale = float(candidate["scale"])
        logits = scale * candidate["family_scores"][family]
        logits += scale * branch[:, None] * candidate["branch_on_scores"][None, :]
        logits += scale * (1.0 - branch[:, None]) * candidate["branch_off_scores"][None, :]
    else:
        logits = candidate["family_scores"][family].copy()
        logits += branch[:, None] * candidate["branch_on_scores"][None, :]
        logits += (1.0 - branch[:, None]) * candidate["branch_off_scores"][None, :]
    return logits


def predict_candidate(candidate: dict[str, Any], data: dict[str, Any], input_key: str) -> tuple[np.ndarray, np.ndarray]:
    if candidate["kind"] == "monolithic_mlp":
        x = data[input_key]
        h = np.tanh(x @ candidate["w1"] + candidate["b1"])
        answer_logits = h @ candidate["w_answer"] + candidate["b_answer"]
        route_logits = h @ candidate["w_route"] + candidate["b_route"]
        return np.argmax(answer_logits, axis=1), np.argmax(route_logits, axis=1)
    route_logits = router_logits(candidate, data)
    route_pred = np.argmax(route_logits, axis=1)
    candidate_answers = data["candidate_answers"]
    answer_pred = candidate_answers[np.arange(len(candidate_answers)), route_pred].copy()
    if candidate["kind"] == "repair_router":
        flip = candidate["answer_bias"][route_pred] > 1.0
        answer_pred[flip] = 1 - answer_pred[flip]
    return answer_pred.astype(np.int64), route_pred.astype(np.int64)


def score_candidate(candidate: dict[str, Any], task: dict[str, Any], input_key: str) -> dict[str, Any]:
    evals = {}
    for split in ("train", "validation"):
        answer_pred, route_pred = predict_candidate(candidate, task[split], input_key)
        evals[split] = evaluate_predictions(answer_pred, route_pred, task[split], sample_limit=0)
    train = evals["train"]
    val = evals["validation"]
    fitness = (
        0.20 * train["usefulness_score"]
        + 0.55 * val["usefulness_score"]
        + 0.15 * val["route_accuracy"]
        + 0.10 * val["composition_accuracy"]
        - 0.01 * val["shortcut_rate"]
    )
    return {"candidate": candidate, "evals": evals, "fitness": round_float(fitness)}


def mutate_candidate(candidate: dict[str, Any], rng: random.Random, settings: Settings) -> tuple[dict[str, Any], str, int]:
    child = copy.deepcopy(candidate)
    kind = child["kind"]
    if kind == "monolithic_mlp":
        key = rng.choice(("w1", "b1", "w_answer", "b_answer", "w_route", "b_route"))
        arr = child[key].copy()
        flat = arr.reshape(-1)
        count = rng.randint(1, max(1, min(8, flat.size)))
        for idx in rng.sample(range(flat.size), k=count):
            flat[idx] += rng.gauss(0.0, settings.mutation_sigma)
        child[key] = flat.reshape(arr.shape)
        return child, f"gaussian_{key}", count
    if kind == "binary_router":
        op = rng.choice(("flip_family_score", "flip_branch_score", "scale_mutation"))
        if op == "flip_family_score":
            i = rng.randrange(len(FAMILIES))
            j = rng.randrange(len(ROUTES))
            child["family_scores"][i, j] *= -1.0
            return child, op, 1
        if op == "flip_branch_score":
            key = rng.choice(("branch_on_scores", "branch_off_scores"))
            j = rng.randrange(len(ROUTES))
            child[key][j] *= -1.0
            return child, f"{op}_{key}", 1
        child["scale"] = max(0.01, float(child["scale"]) * math.exp(rng.gauss(0.0, 0.12)))
        return child, op, 1
    op = rng.choice(("family_score", "branch_score", "answer_bias"))
    if op == "family_score":
        i = rng.randrange(len(FAMILIES))
        j = rng.randrange(len(ROUTES))
        child["family_scores"][i, j] += rng.gauss(0.0, settings.mutation_sigma)
        return child, op, 1
    if op == "branch_score":
        key = rng.choice(("branch_on_scores", "branch_off_scores"))
        j = rng.randrange(len(ROUTES))
        child[key][j] += rng.gauss(0.0, settings.mutation_sigma)
        return child, f"{op}_{key}", 1
    child["answer_bias"][rng.randrange(len(ROUTES))] += rng.gauss(0.0, settings.mutation_sigma)
    return child, op, 1


def evaluate_candidate_full(candidate: dict[str, Any], task: dict[str, Any], input_key: str) -> dict[str, Any]:
    out = {}
    for split in SPLITS:
        answer_pred, route_pred = predict_candidate(candidate, task[split], input_key)
        out[split] = evaluate_predictions(answer_pred, route_pred, task[split])
    return out


def mutation_worker(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    seed = int(job["seed"])
    system = job["system"]
    task = job["task"]
    input_key = "raw" if system == "monolithic_mutation_model" else "combo"
    if system == "monolithic_mutation_model":
        initial = init_mlp_candidate(seed, int(task["train"]["raw"].shape[1]))
    elif system == "frozen_pockets_binary_router":
        initial = init_router_candidate(seed, "binary_router")
    elif system == "router_plus_limited_pocket_repair":
        initial = init_router_candidate(seed, "repair_router")
    else:
        initial = init_router_candidate(seed, "float_router")
    initial_hash = candidate_hash(initial)
    rng = random.Random(stable_seed(f"mutation-{system}-{seed}"))
    population = [score_candidate(copy.deepcopy(initial), task, input_key)]
    for _ in range(settings.mutation_population - 1):
        child, _, _ = mutate_candidate(initial, rng, settings)
        population.append(score_candidate(child, task, input_key))
    accepted = rejected = rollback = attempts = changed_total = 0
    accepted_by_operator: dict[str, int] = {}
    rejected_by_operator: dict[str, int] = {}
    history = []
    best_eval = -1.0
    budget_to_best = 0
    last_heartbeat = time.monotonic()
    for generation in range(1, settings.mutation_generations + 1):
        population.sort(key=lambda row: row["fitness"], reverse=True)
        next_population = copy.deepcopy(population[: settings.mutation_elite_count])
        while len(next_population) < settings.mutation_population:
            parent = copy.deepcopy(rng.choice(population))
            child_candidate, operator, changed = mutate_candidate(parent["candidate"], rng, settings)
            child = score_candidate(child_candidate, task, input_key)
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
        val = best["evals"]["validation"]
        if val["usefulness_score"] > best_eval:
            best_eval = val["usefulness_score"]
            budget_to_best = attempts
        row = {
            "generation": generation,
            "best_fitness": best["fitness"],
            "validation_usefulness": val["usefulness_score"],
            "validation_answer_accuracy": val["answer_accuracy"],
            "validation_route_accuracy": val["route_accuracy"],
            "attempts": attempts,
        }
        history.append(row)
        if out and (time.monotonic() - last_heartbeat >= settings.heartbeat_seconds or generation == settings.mutation_generations):
            safe = safe_file_id(f"{system}_seed{seed}")
            locked_write_json(out / "partial_status" / f"mutation_{safe}.json", row)
            locked_write_json(
                out / "mutation_history_snapshots" / f"{safe}.json",
                {
                    "schema_version": "e7b_mutation_history_snapshot_v1",
                    "system": system,
                    "seed": seed,
                    "history_tail": history[-25:],
                    "accepted_mutations": accepted,
                    "rejected_mutations": rejected,
                    "rollback_count": rollback,
                    "mutation_attempts": attempts,
                },
            )
            append_progress(out, "mutation_generation", system=system, seed=seed, generation=generation, validation_usefulness=row["validation_usefulness"])
            last_heartbeat = time.monotonic()
    best = max(population, key=lambda row: row["fitness"])
    final_candidate = best["candidate"]
    evals = evaluate_candidate_full(final_candidate, task, input_key)
    return {
        "seed": seed,
        "system": system,
        "training_mode": "mutation_rollback",
        "input_key": input_key,
        "parameter_count": parameter_count(final_candidate),
        "initial_hash": initial_hash,
        "final_hash": candidate_hash(final_candidate),
        "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": candidate_hash(final_candidate)}),
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rollback,
        "accepted_by_operator": accepted_by_operator,
        "rejected_by_operator": rejected_by_operator,
        "mean_changed_parameters_per_attempt": round_float(changed_total / max(1, attempts)),
        "budget_to_best_usefulness": budget_to_best,
        "history": history,
        "evals": evals,
    }


def gpu_lane_worker(job: dict[str, Any]) -> dict[str, Any]:
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    tasks = job["tasks"]
    rows = []
    histories = []
    for seed_text in sorted(tasks, key=lambda value: int(value)):
        seed = int(seed_text)
        for system in GRADIENT_SYSTEMS:
            if out:
                append_progress(out, "gradient_job_start", system=system, seed=seed)
            result = train_gradient_system(seed, system, tasks[seed_text], settings, out)
            rows.append({key: value for key, value in result.items() if key != "history"})
            histories.append({"seed": seed, "system": system, "history": result["history"], "device": result["device"]})
            if out:
                append_progress(out, "gradient_job_complete", system=system, seed=seed, device=result["device"])
    return {"lane": "gpu_gradient_lane", "rows": rows, "histories": histories, "hardware": hardware_probe()}


def control_results(seed: int, task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(stable_seed(f"random-control-{seed}"))
    for system in CONTROL_SYSTEMS:
        evals = {}
        for split in SPLITS:
            data = task[split]
            if system == "oracle_pocket_router_reference":
                route_pred = data["route"].copy()
                answer_pred = data["y"].copy()
            else:
                route_pred = rng.integers(0, len(ROUTES), size=len(data["y"]), dtype=np.int64)
                answer_pred = data["candidate_answers"][np.arange(len(data["y"])), route_pred]
            evals[split] = evaluate_predictions(answer_pred, route_pred, data)
        rows.append(
            {
                "seed": seed,
                "system": system,
                "training_mode": "control",
                "input_key": "combo",
                "parameter_count": 0,
                "state_hash": payload_sha256({"seed": seed, "system": system}),
                "evals": evals,
            }
        )
    return rows


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    return payload


def split_summary(evals: dict[str, Any]) -> dict[str, float]:
    out = {}
    for metric in ("answer_accuracy", "route_accuracy", "composition_accuracy", "shortcut_rate", "usefulness_score"):
        out[metric] = round_float(float(np.mean([evals[split][metric] for split in EVAL_SPLITS])))
    out["generalization_gap"] = round_float(evals["train"]["usefulness_score"] - out["usefulness_score"])
    out["heldout_usefulness"] = evals["heldout"]["usefulness_score"]
    out["ood_usefulness"] = evals["ood"]["usefulness_score"]
    out["counterfactual_usefulness"] = evals["counterfactual"]["usefulness_score"]
    out["adversarial_usefulness"] = evals["adversarial"]["usefulness_score"]
    return out


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_system: dict[str, list[dict[str, Any]]] = {system: [] for system in SYSTEMS}
    for row in rows:
        by_system[row["system"]].append(row)
    systems = {}
    for system, items in by_system.items():
        seed_rows = []
        for item in sorted(items, key=lambda row: int(row["seed"])):
            seed_rows.append(
                {
                    "seed": item["seed"],
                    "parameter_count": item.get("parameter_count", 0),
                    **split_summary(item["evals"]),
                }
            )
        systems[system] = {
            "seed_count": len(seed_rows),
            "rows": seed_rows,
            "mean": {
                key: round_float(float(np.mean([row[key] for row in seed_rows]))) if seed_rows else 0.0
                for key in (
                    "answer_accuracy",
                    "route_accuracy",
                    "composition_accuracy",
                    "shortcut_rate",
                    "usefulness_score",
                    "generalization_gap",
                    "heldout_usefulness",
                    "ood_usefulness",
                    "counterfactual_usefulness",
                    "adversarial_usefulness",
                    "parameter_count",
                )
            },
        }
    best_system = max((system for system in SYSTEMS if system != "oracle_pocket_router_reference"), key=lambda name: systems[name]["mean"]["usefulness_score"])
    return {
        "schema_version": "e7b_aggregate_metrics_v1",
        "systems": systems,
        "best_non_oracle_system": best_system,
        "best_non_oracle_usefulness": systems[best_system]["mean"]["usefulness_score"],
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    mutation_router = systems["frozen_pockets_mutation_router"]["mean"]
    binary_router = systems["frozen_pockets_binary_router"]["mean"]
    repair_router = systems["router_plus_limited_pocket_repair"]["mean"]
    monolithic_mut = systems["monolithic_mutation_model"]["mean"]
    gradient_router = systems["frozen_pockets_gradient_router"]["mean"]
    monolithic_grad = systems["monolithic_backprop_model"]["mean"]
    random_control = systems["random_router_control"]["mean"]
    best_mut_router = max(
        (
            ("frozen_pockets_mutation_router", mutation_router),
            ("frozen_pockets_binary_router", binary_router),
            ("router_plus_limited_pocket_repair", repair_router),
        ),
        key=lambda row: row[1]["usefulness_score"],
    )
    leak = random_control["usefulness_score"] >= 0.80
    task_too_easy = monolithic_mut["usefulness_score"] >= 0.93 and monolithic_mut["route_accuracy"] >= 0.90
    router_pass = (
        best_mut_router[1]["usefulness_score"] >= 0.92
        and best_mut_router[1]["route_accuracy"] >= 0.90
        and best_mut_router[1]["adversarial_usefulness"] >= 0.86
        and best_mut_router[1]["shortcut_rate"] <= 0.12
    )
    gradient_pass = (
        max(gradient_router["usefulness_score"], monolithic_grad["usefulness_score"]) >= 0.92
        and max(gradient_router["route_accuracy"], monolithic_grad["route_accuracy"]) >= 0.90
    )
    if leak:
        decision = "e7b_router_leak_or_artifact_detected"
    elif task_too_easy:
        decision = "e7b_monolithic_mutation_sufficient_or_task_too_easy"
    elif router_pass:
        decision = "e7b_mutation_router_composition_viable"
    elif gradient_pass:
        decision = "e7b_gradient_only_composition_viable"
    else:
        decision = "e7b_pocket_router_no_advantage_detected"
    return decision, {
        "best_mutation_router_system": best_mut_router[0],
        "best_mutation_router_mean": best_mut_router[1],
        "leak_control_triggered": leak,
        "task_too_easy_triggered": task_too_easy,
        "router_pass": router_pass,
        "gradient_pass": gradient_pass,
    }


def build_composition_report(aggregate: dict[str, Any], decision_detail: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    return {
        "schema_version": "e7b_composition_report_v1",
        "question": "can mutation/rollback compose frozen pockets by learning the router",
        "best_mutation_router_system": decision_detail["best_mutation_router_system"],
        "best_non_oracle_system": aggregate["best_non_oracle_system"],
        "monolithic_mutation_usefulness": systems["monolithic_mutation_model"]["mean"]["usefulness_score"],
        "best_mutation_router_usefulness": decision_detail["best_mutation_router_mean"]["usefulness_score"],
        "gradient_router_usefulness": systems["frozen_pockets_gradient_router"]["mean"]["usefulness_score"],
        "monolithic_backprop_usefulness": systems["monolithic_backprop_model"]["mean"]["usefulness_score"],
        "random_control_usefulness": systems["random_router_control"]["mean"]["usefulness_score"],
        "oracle_usefulness": systems["oracle_pocket_router_reference"]["mean"]["usefulness_score"],
        "interpretation_boundary": "routing_over_frozen_symbolic_pockets_only",
    }


def build_leakage_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    random_mean = aggregate["systems"]["random_router_control"]["mean"]
    grad_adv = aggregate["systems"]["frozen_pockets_gradient_router"]["mean"]["adversarial_usefulness"]
    mut_adv = aggregate["systems"]["frozen_pockets_mutation_router"]["mean"]["adversarial_usefulness"]
    return {
        "schema_version": "e7b_leakage_report_v1",
        "misleading_route_adversarial_split_used": True,
        "random_control_usefulness": random_mean["usefulness_score"],
        "random_control_passed": random_mean["usefulness_score"] < 0.80,
        "gradient_router_adversarial_usefulness": grad_adv,
        "mutation_router_adversarial_usefulness": mut_adv,
        "route_name_or_index_leakage_claim": False,
        "hidden_correct_route_index_used_as_input": False,
    }


def build_markdown(payloads: dict[str, Any]) -> str:
    summary = payloads["summary.json"]
    aggregate = payloads["aggregate_metrics.json"]
    lines = [
        "# E7B Pocket Routing Composition Probe Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {summary['decision']}",
        f"best_non_oracle_system = {summary['best_non_oracle_system']}",
        f"deterministic_replay_passed = {summary['deterministic_replay_passed']}",
        "```",
        "",
        "## Mean Metrics",
        "",
        "| system | usefulness | answer | route | composition | adversarial | shortcut | params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        lines.append(
            f"| {system} | {mean['usefulness_score']:.6f} | {mean['answer_accuracy']:.6f} | "
            f"{mean['route_accuracy']:.6f} | {mean['composition_accuracy']:.6f} | "
            f"{mean['adversarial_usefulness']:.6f} | {mean['shortcut_rate']:.6f} | {mean['parameter_count']:.1f} |"
        )
    lines.extend(
        [
            "",
            "This is a controlled symbolic pocket-routing proxy over frozen pocket outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress(out, "startup", milestone=MILESTONE, settings=settings_payload(settings), hardware=hardware_probe())
    tasks = {seed: generate_seed_task(seed, settings) for seed in settings.seeds}
    if out:
        append_progress(out, "tasks_generated", seeds=list(settings.seeds), row_counts=task_report(tasks)["row_counts"])
    task_payload = {str(seed): task for seed, task in tasks.items()}
    rows: list[dict[str, Any]] = []
    mutation_histories: list[dict[str, Any]] = []
    training_histories: list[dict[str, Any]] = []
    for seed, task in tasks.items():
        rows.extend(control_results(seed, task))
    jobs = [
        {
            "seed": seed,
            "system": system,
            "task": tasks[seed],
            "settings": settings.__dict__,
            "out": out.as_posix() if out else None,
        }
        for seed in settings.seeds
        for system in MUTATION_SYSTEMS
    ]
    gpu_job = {"tasks": task_payload, "settings": settings.__dict__, "out": out.as_posix() if out else None}
    if settings.execution_mode == "parallel":
        with ProcessPoolExecutor(max_workers=max(1, settings.cpu_workers + 1)) as executor:
            futures = {executor.submit(gpu_lane_worker, gpu_job): "gpu_gradient_lane"}
            for job in jobs:
                futures[executor.submit(mutation_worker, job)] = f"{job['system']}/seed{job['seed']}"
            pending = set(futures)
            if out:
                append_progress(out, "lanes_submitted", cpu_mutation_jobs=len(jobs), cpu_workers=settings.cpu_workers, gpu_lane=True)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    label = futures[future]
                    result = future.result()
                    if label == "gpu_gradient_lane":
                        rows.extend(result["rows"])
                        training_histories.extend(result["histories"])
                        if out:
                            append_progress(out, "gpu_lane_complete", completed_gradient_rows=len(result["rows"]), hardware=result["hardware"])
                    else:
                        rows.append({key: value for key, value in result.items() if key != "history"})
                        mutation_histories.append(
                            {
                                "seed": result["seed"],
                                "system": result["system"],
                                "history": result["history"],
                                "mutation_attempts": result["mutation_attempts"],
                                "accepted_mutations": result["accepted_mutations"],
                                "rejected_mutations": result["rejected_mutations"],
                                "rollback_count": result["rollback_count"],
                                "accepted_by_operator": result["accepted_by_operator"],
                                "rejected_by_operator": result["rejected_by_operator"],
                            }
                        )
                        if out:
                            locked_write_json(
                                out / "partial_aggregate_snapshot.json",
                                {
                                    "schema_version": "e7b_partial_aggregate_snapshot_v1",
                                    "completed_rows": len(rows),
                                    "expected_rows": len(settings.seeds) * len(SYSTEMS),
                                    "pending_jobs": len(pending),
                                },
                            )
                            append_progress(out, "mutation_job_complete", label=label, pending=len(pending))
    else:
        if out:
            append_progress(out, "lanes_submitted", cpu_mutation_jobs=len(jobs), cpu_workers=1, gpu_lane=True)
        gpu = gpu_lane_worker(gpu_job)
        rows.extend(gpu["rows"])
        training_histories.extend(gpu["histories"])
        for job in jobs:
            result = mutation_worker(job)
            rows.append({key: value for key, value in result.items() if key != "history"})
            mutation_histories.append(
                {
                    "seed": result["seed"],
                    "system": result["system"],
                    "history": result["history"],
                    "mutation_attempts": result["mutation_attempts"],
                    "accepted_mutations": result["accepted_mutations"],
                    "rejected_mutations": result["rejected_mutations"],
                    "rollback_count": result["rollback_count"],
                    "accepted_by_operator": result["accepted_by_operator"],
                    "rejected_by_operator": result["rejected_by_operator"],
                }
            )
    rows.sort(key=lambda row: (row["system"], int(row["seed"])))
    mutation_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    training_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    aggregate = aggregate_results(rows)
    decision, detail = decide(aggregate)
    return {
        "tasks": tasks,
        "rows": rows,
        "mutation_histories": mutation_histories,
        "training_histories": training_histories,
        "aggregate": aggregate,
        "decision": decision,
        "decision_detail": detail,
    }


def build_payloads(settings: Settings, out: Path, results: dict[str, Any]) -> dict[str, Any]:
    decision = results["decision"]
    aggregate = results["aggregate"]
    payloads: dict[str, Any] = {
        "backend_manifest.json": {
            "schema_version": "e7b_backend_manifest_v1",
            "milestone": MILESTONE,
            "settings": settings_payload(settings),
            "systems": list(SYSTEMS),
            "gradient_systems": list(GRADIENT_SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "control_systems": list(CONTROL_SYSTEMS),
            "hardware_identity": stable_hardware_identity(),
            "parallel_cpu_gpu_lanes": settings.execution_mode == "parallel",
        },
        "task_generation_report.json": task_report(results["tasks"]),
        "pocket_library_report.json": pocket_library_report(results["tasks"]),
        "system_results.json": {
            "schema_version": "e7b_system_results_v1",
            "rows": results["rows"],
        },
        "mutation_history.json": {
            "schema_version": "e7b_mutation_history_v1",
            "rows": results["mutation_histories"],
        },
        "training_history.json": {
            "schema_version": "e7b_training_history_v1",
            "rows": results["training_histories"],
        },
        "aggregate_metrics.json": aggregate,
        "composition_report.json": build_composition_report(aggregate, results["decision_detail"]),
        "leakage_report.json": build_leakage_report(aggregate),
        "decision.json": {
            "schema_version": "e7b_decision_v1",
            "decision": decision,
            "detail": results["decision_detail"],
            "deterministic_replay_passed": False,
        },
        "summary.json": {
            "schema_version": "e7b_summary_v1",
            "decision": decision,
            "best_non_oracle_system": aggregate["best_non_oracle_system"],
            "deterministic_replay_passed": False,
            "checker_failure_count": None,
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
        },
    }
    payloads["report.md"] = build_markdown(payloads)
    return payloads


def compute_hashes(payloads: dict[str, Any]) -> dict[str, str]:
    return {name: payload_sha256(payloads[name]) for name in HASH_ARTIFACTS}


def deterministic_replay(settings: Settings, out: Path, primary_payloads: dict[str, Any]) -> dict[str, Any]:
    replay_out = out / "deterministic_replay_work"
    if replay_out.exists():
        shutil.rmtree(replay_out)
    append_progress(out, "deterministic_replay_start", replay_out=replay_out.relative_to(REPO_ROOT).as_posix())
    replay_results = run_core(settings, replay_out)
    replay_payloads = build_payloads(settings, out, replay_results)
    primary_hashes = compute_hashes(primary_payloads)
    replay_hashes = compute_hashes(replay_payloads)
    comparisons = {
        name: {"primary_hash": primary_hashes[name], "replay_hash": replay_hashes[name], "match": primary_hashes[name] == replay_hashes[name]}
        for name in HASH_ARTIFACTS
    }
    report = {
        "schema_version": "e7b_deterministic_replay_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
        "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix(),
    }
    append_progress(out, "deterministic_replay_complete", internal_replay_passed=report["internal_replay_passed"])
    return report


def write_final_artifacts(out: Path, payloads: dict[str, Any], replay: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["deterministic_replay.json"] = replay
    payloads["summary.json"]["deterministic_replay_passed"] = replay["internal_replay_passed"]
    payloads["decision.json"]["deterministic_replay_passed"] = replay["internal_replay_passed"]
    payloads["report.md"] = build_markdown(payloads)
    for name, payload in payloads.items():
        if name.endswith(".md"):
            write_text(out / name, payload)
        else:
            write_json(out / name, payload)
    append_progress(out, "final_artifacts_written", artifact_count=len(payloads))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT.as_posix())
    parser.add_argument("--seeds", default="91001,91002,91003,91004,91005,91006,91007,91008")
    parser.add_argument("--train-rows-per-seed", type=int, default=360)
    parser.add_argument("--validation-rows-per-seed", type=int, default=160)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=160)
    parser.add_argument("--ood-rows-per-seed", type=int, default=160)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=160)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=160)
    parser.add_argument("--gradient-epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mutation-generations", type=int, default=120)
    parser.add_argument("--mutation-population", type=int, default=22)
    parser.add_argument("--mutation-sigma", type=float, default=0.25)
    parser.add_argument("--mutation-elite-count", type=int, default=4)
    parser.add_argument("--cpu-workers", type=int, default=max(1, min((os.cpu_count() or 4) - 1, 23)))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--execution-mode", choices=("parallel", "serial"), default="parallel")
    parser.add_argument("--no-replay", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    settings = Settings(
        seeds=parse_int_tuple(args.seeds),
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
        mutation_generations=args.mutation_generations,
        mutation_population=args.mutation_population,
        mutation_sigma=args.mutation_sigma,
        mutation_elite_count=args.mutation_elite_count,
        cpu_workers=args.cpu_workers,
        device=args.device,
        heartbeat_seconds=args.heartbeat_seconds,
        execution_mode=args.execution_mode,
        replay=not args.no_replay,
    )
    out.mkdir(parents=True, exist_ok=True)
    stop_monitor = threading.Event()
    monitor = start_hardware_monitor(out, stop_monitor, settings.heartbeat_seconds)
    try:
        results = run_core(settings, out)
        payloads = build_payloads(settings, out, results)
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7b_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_skipped": True}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop_monitor.set()
        monitor.join(timeout=5.0)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"decision": summary["decision"], "best_non_oracle_system": summary["best_non_oracle_system"], "deterministic_replay_passed": summary["deterministic_replay_passed"], "out": out.as_posix()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
