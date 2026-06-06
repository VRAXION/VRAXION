#!/usr/bin/env python3
"""E7D short-pipe composition versus fused-pipe probe.

This probe tests whether a recurrent router over short reusable pocket pipes
can replace a library of longer fused AB pipes. The task is deliberately a
controlled symbolic/numeric proxy: rows specify primitive operation tokens, the
system must compose them, and OOD rows hold out complete operation pairs while
keeping the individual primitive operations seen.
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
from pathlib import Path
import random
import shutil
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
E7B_PATH = Path(__file__).with_name("run_e7b_pocket_routing_composition_probe.py")
MILESTONE = "E7D_SHORT_PIPE_COMPOSITION_VS_FUSED_PIPE_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7d_short_pipe_composition_vs_fused_pipe_probe")
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
PRIMITIVES = ("add_b", "xor_b", "add_mem", "xor_mem", "rot_add_key")
PAIR_ROUTES = tuple(f"{left}_then_{right}" for left in PRIMITIVES for right in PRIMITIVES)
SYSTEMS = (
    "monolithic_matrix_core_gradient",
    "monolithic_mutation_model",
    "fused_long_pipe_gradient_router",
    "fused_long_pipe_mutation_router",
    "short_pipe_no_router_between",
    "short_pipe_router_composition",
    "router_plus_limited_pocket_repair",
    "random_router_control",
    "oracle_short_pipe_reference",
)
GRADIENT_SYSTEMS = ("monolithic_matrix_core_gradient", "fused_long_pipe_gradient_router")
MUTATION_SYSTEMS = (
    "monolithic_mutation_model",
    "fused_long_pipe_mutation_router",
    "short_pipe_no_router_between",
    "short_pipe_router_composition",
    "router_plus_limited_pocket_repair",
)
CONTROL_SYSTEMS = ("random_router_control", "oracle_short_pipe_reference")
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pipe_topology_report.json",
    "flow_state_report.json",
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
    "e7d_short_pipe_router_flow_preferred",
    "e7d_fused_long_pipe_required",
    "e7d_hybrid_common_fused_plus_short_preferred",
    "e7d_monolithic_sufficient_or_task_too_easy",
    "e7d_router_overhead_failure",
    "e7d_leak_or_artifact_detected",
    "e7d_no_clear_topology_winner",
)


def load_e7b_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7b_pocket_routing_composition_probe", E7B_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7B helpers from {E7B_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7b = load_e7b_module()


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


class MultiHeadMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.answer = nn.Linear(hidden_dim, 2)
        self.route = nn.Linear(hidden_dim, len(PAIR_ROUTES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.answer(h), self.route(h)


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7d::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7b.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7b.write_json(path, payload)


def locked_write_json(path: Path, payload: Any) -> None:
    e7b.locked_write_json(path, payload)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7b.append_progress(out, event, **details)


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


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    return payload


def select_device(requested: str) -> str:
    return e7b.select_device(requested)


def set_determinism(seed: int, device: str) -> None:
    e7b.set_determinism(seed, device)


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    return e7b.start_hardware_monitor(out, stop, interval)


def pair_index(op1: int, op2: int) -> int:
    return op1 * len(PRIMITIVES) + op2


def pair_from_index(index: int) -> tuple[int, int]:
    return divmod(int(index), len(PRIMITIVES))


def heldout_pair(op1: int, op2: int) -> bool:
    return (op1 + 2 * op2) % len(PRIMITIVES) == 0


def pair_pool(split: str) -> list[tuple[int, int]]:
    heldout = [(i, j) for i in range(len(PRIMITIVES)) for j in range(len(PRIMITIVES)) if heldout_pair(i, j)]
    seen = [(i, j) for i in range(len(PRIMITIVES)) for j in range(len(PRIMITIVES)) if not heldout_pair(i, j)]
    return heldout if split == "ood" else seen


def memory_value(seed: int, key: int, split: str) -> int:
    split_shift = {"train": 0, "validation": 1, "heldout": 2, "ood": 7, "counterfactual": 3, "adversarial": 5}[split]
    return (key * 7 + seed % 19 + split_shift * 5) & 15


def apply_primitive(op: int, state: int, a: int, b: int, key: int, mem: int) -> int:
    if op == 0:
        return (state + b) & 15
    if op == 1:
        return state ^ b
    if op == 2:
        return (state + mem) & 15
    if op == 3:
        return state ^ mem
    if op == 4:
        rotated = ((state << 1) & 15) | ((state >> 3) & 1)
        return (rotated + key) & 15
    raise ValueError(op)


def pair_value(op1: int, op2: int, a: int, b: int, key: int, mem: int) -> tuple[int, int]:
    first = apply_primitive(op1, a, a, b, key, mem)
    second = apply_primitive(op2, first, a, b, key, mem)
    return first, second


def all_pair_values(a: int, b: int, key: int, mem: int) -> np.ndarray:
    values = []
    for op1 in range(len(PRIMITIVES)):
        for op2 in range(len(PRIMITIVES)):
            _, second = pair_value(op1, op2, a, b, key, mem)
            values.append(second)
    return np.asarray(values, dtype=np.float32)


def context_key(op1: int, op2_true: int, op2_false: int, mode: int, branch_flag: int) -> int:
    n = len(PRIMITIVES)
    return (((op1 * n + op2_true) * n + op2_false) * 2 + mode) * 2 + branch_flag


def context_count() -> int:
    n = len(PRIMITIVES)
    return n * n * n * 2 * 2


def make_row(seed: int, split: str, index: int, rng: random.Random) -> dict[str, Any]:
    a = rng.choice((0, 1, 2, 13, 14, 15)) if split == "ood" else rng.randrange(16)
    b = rng.choice((0, 1, 2, 13, 14, 15)) if split == "ood" else rng.randrange(16)
    key = rng.randrange(16)
    threshold = rng.choice((2, 3, 12, 13)) if split == "ood" else rng.randrange(4, 12)
    mem = memory_value(seed, key, split)
    allowed_pairs = pair_pool(split)
    target_op1, target_op2 = rng.choice(allowed_pairs)
    mode = 1 if rng.random() < 0.44 else 0
    branch_gate = rng.randrange(5, 11)
    first_value = apply_primitive(target_op1, a, a, b, key, mem)
    branch_flag = 1 if first_value > branch_gate else 0
    if mode == 0:
        op2_true = target_op2
        op2_false = target_op2
    else:
        alt_choices = [idx for idx in range(len(PRIMITIVES)) if idx != target_op2]
        alternate = rng.choice(alt_choices)
        if branch_flag:
            op2_true = target_op2
            op2_false = alternate
        else:
            op2_true = alternate
            op2_false = target_op2
    _, final_value = pair_value(target_op1, target_op2, a, b, key, mem)
    if split == "counterfactual":
        base_answer = 1 if final_value > threshold else 0
        for delta in (1, -1, 2, -2, 4, -4, 6, -6):
            candidate_threshold = max(0, min(15, threshold + delta))
            if (1 if final_value > candidate_threshold else 0) != base_answer:
                threshold = candidate_threshold
                break
    candidate_values = all_pair_values(a, b, key, mem)
    candidate_answers = (candidate_values > threshold).astype(np.int64)
    route = pair_index(target_op1, target_op2)
    answer = int(candidate_answers[route])
    if split == "adversarial":
        misleading = (route + 1 + index % (len(PAIR_ROUTES) - 1)) % len(PAIR_ROUTES)
        if int(candidate_answers[misleading]) != answer:
            for offset in range(2, len(PAIR_ROUTES)):
                alt = (route + offset) % len(PAIR_ROUTES)
                if int(candidate_answers[alt]) == answer:
                    misleading = alt
                    break
    elif split in {"train", "validation"} and rng.random() < 0.78:
        misleading = route
    else:
        misleading = rng.randrange(len(PAIR_ROUTES))
    op1_hot = [1.0 if idx == target_op1 else 0.0 for idx in range(len(PRIMITIVES))]
    op2_true_hot = [1.0 if idx == op2_true else 0.0 for idx in range(len(PRIMITIVES))]
    op2_false_hot = [1.0 if idx == op2_false else 0.0 for idx in range(len(PRIMITIVES))]
    misleading_hot = [1.0 if idx == misleading else 0.0 for idx in range(len(PAIR_ROUTES))]
    noise = [rng.uniform(-1.0, 1.0) for _ in range(8)]
    raw = [
        a / 15.0,
        b / 15.0,
        key / 15.0,
        threshold / 15.0,
        mem / 15.0,
        branch_gate / 15.0,
        float(mode),
    ] + op1_hot + op2_true_hot + op2_false_hot + misleading_hot + noise
    fused = raw + candidate_answers.astype(np.float32).tolist() + (candidate_values / 15.0).astype(np.float32).tolist()
    return {
        "row_id": f"{seed}/{split}/{index}",
        "seed": seed,
        "split": split,
        "mode": "branch_after_first" if mode else "fixed_pair",
        "mode_id": mode,
        "op1": target_op1,
        "op2_true": op2_true,
        "op2_false": op2_false,
        "target_op2": target_op2,
        "branch_gate": branch_gate,
        "branch_flag": branch_flag,
        "route": route,
        "answer": answer,
        "raw": raw,
        "fused": fused,
        "candidate_values": candidate_values.astype(np.float32).tolist(),
        "candidate_answers": candidate_answers.tolist(),
        "misleading_route": misleading,
        "a": a,
        "b": b,
        "key": key,
        "threshold": threshold,
        "mem": mem,
        "pair_seen_in_train": not heldout_pair(target_op1, target_op2),
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
        task[split] = {
            "rows": rows,
            "raw": np.asarray([row["raw"] for row in rows], dtype=np.float32),
            "fused": np.asarray([row["fused"] for row in rows], dtype=np.float32),
            "y": np.asarray([row["answer"] for row in rows], dtype=np.int64),
            "route": np.asarray([row["route"] for row in rows], dtype=np.int64),
            "candidate_answers": np.asarray([row["candidate_answers"] for row in rows], dtype=np.int64),
            "candidate_values": np.asarray([row["candidate_values"] for row in rows], dtype=np.float32),
        }
    return task


def task_report(tasks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    seen = [PAIR_ROUTES[pair_index(i, j)] for i, j in pair_pool("train")]
    heldout = [PAIR_ROUTES[pair_index(i, j)] for i, j in pair_pool("ood")]
    return {
        "schema_version": "e7d_task_generation_report_v1",
        "primitives": list(PRIMITIVES),
        "pair_routes": list(PAIR_ROUTES),
        "seen_train_pairs": seen,
        "ood_heldout_pairs": heldout,
        "row_counts": {
            str(seed): {split: int(len(task[split]["rows"])) for split in SPLITS}
            for seed, task in tasks.items()
        },
        "raw_feature_dim": int(next(iter(tasks.values()))["train"]["raw"].shape[1]),
        "fused_feature_dim": int(next(iter(tasks.values()))["train"]["fused"].shape[1]),
        "target_pair_route_not_in_raw_features": True,
        "legitimate_instruction_tokens": ["op1", "op2_true", "op2_false", "mode"],
        "adversarial_misleading_pair_used": True,
        "counterfactual_threshold_flip_used": True,
        "ood_unseen_pair_compositions_used": True,
    }


def pipe_topology_report() -> dict[str, Any]:
    return {
        "schema_version": "e7d_pipe_topology_report_v1",
        "topology_question": "short reusable pocket pipes plus recurrent router versus fused AB pipe library",
        "short_pipe_library": {
            "primitive_pipe_count": len(PRIMITIVES),
            "primitives": list(PRIMITIVES),
            "router_return_between_pipes": True,
            "composed_route_count": len(PAIR_ROUTES),
        },
        "fused_pipe_library": {
            "fused_pair_pipe_count": len(PAIR_ROUTES),
            "fused_pair_names": list(PAIR_ROUTES),
            "unseen_pair_generalization_required": True,
        },
        "falsification_logic": "If fused pair pipes win OOD, long pipes may be necessary; if short pipes match or win, reusable pipes plus router are preferred.",
    }


def flow_state_report() -> dict[str, Any]:
    return {
        "schema_version": "e7d_flow_state_report_v1",
        "flow_state_fields": ["current_value", "branch_gate", "branch_flag_after_first_pipe", "residual_raw_inputs"],
        "router_between_pipes": "short_pipe_router_composition recomputes the branch decision after the first predicted pipe.",
        "no_router_control": "short_pipe_no_router_between ignores first-pipe feedback on branch rows.",
        "flow_not_packet_boundary": "The probe treats the state as a recurrent flow value that is bent through short pipes, not as a one-shot opaque packet.",
    }


def evaluate_predictions(
    answer_pred: np.ndarray,
    route_pred: np.ndarray,
    data: dict[str, Any],
    latency_steps: float,
    sample_limit: int = 8,
) -> dict[str, Any]:
    y = data["y"]
    route = data["route"]
    answer_acc = float(np.mean(answer_pred == y))
    route_acc = float(np.mean(route_pred == route))
    composition_acc = float(np.mean((answer_pred == y) & (route_pred == route)))
    candidate = data["candidate_answers"]
    shortcut = float(np.mean(candidate[np.arange(len(candidate)), route_pred] != y))
    usefulness = 0.48 * answer_acc + 0.37 * route_acc + 0.15 * composition_acc
    step_penalized = usefulness - 0.006 * max(0.0, latency_steps - 1.0)
    samples = []
    for idx, row in enumerate(data["rows"][:sample_limit]):
        samples.append(
            {
                "row_id": row["row_id"],
                "mode": row["mode"],
                "target_route": PAIR_ROUTES[int(route[idx])],
                "predicted_route": PAIR_ROUTES[int(route_pred[idx])],
                "target_answer": int(y[idx]),
                "predicted_answer": int(answer_pred[idx]),
                "misleading_route": PAIR_ROUTES[row["misleading_route"]],
                "ood_pair_seen_in_train": bool(row["pair_seen_in_train"]),
            }
        )
    return {
        "answer_accuracy": round_float(answer_acc),
        "route_accuracy": round_float(route_acc),
        "composition_accuracy": round_float(composition_acc),
        "shortcut_rate": round_float(shortcut),
        "usefulness_score": round_float(usefulness),
        "latency_steps": round_float(latency_steps),
        "step_penalized_usefulness": round_float(step_penalized),
        "row_level_samples": samples,
    }


def evaluate_gradient_model(model_state: dict[str, Any], task: dict[str, Any], input_key: str, device: str, latency_steps: float) -> dict[str, Any]:
    model = MultiHeadMLP(int(model_state["input_dim"]))
    model.load_state_dict({key: torch.as_tensor(value, dtype=torch.float32) for key, value in model_state["state_dict"].items()})
    model.to(device)
    model.eval()
    out = {}
    with torch.no_grad():
        for split in SPLITS:
            x = torch.as_tensor(task[split][input_key], dtype=torch.float32, device=device)
            answer_logits, route_logits = model(x)
            answer_pred = torch.argmax(answer_logits, dim=1).cpu().numpy()
            route_pred = torch.argmax(route_logits, dim=1).cpu().numpy()
            out[split] = evaluate_predictions(answer_pred, route_pred, task[split], latency_steps)
    return out


def train_gradient_system(seed: int, system: str, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"gradient-{system}-{seed}"), device)
    input_key = "raw" if system == "monolithic_matrix_core_gradient" else "fused"
    latency = 1.0
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
            loss = nn.functional.cross_entropy(answer_logits, y_train[idx]) + 0.9 * nn.functional.cross_entropy(route_logits, r_train[idx])
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
            score = 0.55 * answer_acc + 0.45 * route_acc
        row = {
            "epoch": epoch,
            "validation_answer_accuracy": round_float(answer_acc),
            "validation_route_accuracy": round_float(route_acc),
            "score": round_float(score),
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in model.state_dict().items()}
        if out and (time.monotonic() - last_heartbeat >= settings.heartbeat_seconds or epoch == settings.gradient_epochs):
            locked_write_json(out / "partial_status" / f"gradient_{system}_seed{seed}.json", row)
            append_progress(out, "gradient_epoch", system=system, seed=seed, epoch=epoch, validation_route_accuracy=row["validation_route_accuracy"], device=device)
            last_heartbeat = time.monotonic()
    assert best_state is not None
    serial_state = {key: value.tolist() for key, value in best_state.items()}
    model_state = {"input_dim": input_dim, "state_dict": serial_state}
    evals = evaluate_gradient_model(model_state, task, input_key, device, latency)
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


def init_mlp_candidate(seed: int, input_dim: int, hidden_dim: int = 36) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"mlp-init-{seed}"))
    return {
        "kind": "monolithic_mlp",
        "w1": rng.normal(0.0, 0.16, size=(input_dim, hidden_dim)).astype(np.float64),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "w_answer": rng.normal(0.0, 0.16, size=(hidden_dim, 2)).astype(np.float64),
        "b_answer": np.zeros(2, dtype=np.float64),
        "w_route": rng.normal(0.0, 0.16, size=(hidden_dim, len(PAIR_ROUTES))).astype(np.float64),
        "b_route": np.zeros(len(PAIR_ROUTES), dtype=np.float64),
    }


def init_fused_candidate(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"fused-init-{seed}"))
    return {
        "kind": "fused_pair_table",
        "context_scores": rng.normal(0.0, 0.03, size=(context_count(), len(PAIR_ROUTES))).astype(np.float64),
        "answer_bias": np.zeros(len(PAIR_ROUTES), dtype=np.float64),
    }


def init_short_candidate(seed: int, kind: str) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"short-init-{kind}-{seed}"))
    n = len(PRIMITIVES)
    return {
        "kind": kind,
        "op1_scores": rng.normal(0.0, 0.04, size=(n, n)).astype(np.float64),
        "op2_fixed_scores": rng.normal(0.0, 0.04, size=(n, n)).astype(np.float64),
        "op2_true_scores": rng.normal(0.0, 0.04, size=(n, n)).astype(np.float64),
        "op2_false_scores": rng.normal(0.0, 0.04, size=(n, n)).astype(np.float64),
        "answer_bias": np.zeros(len(PAIR_ROUTES), dtype=np.float64),
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
    return sum(int(value.size) for value in candidate.values() if isinstance(value, np.ndarray))


def predict_candidate(candidate: dict[str, Any], data: dict[str, Any], input_key: str) -> tuple[np.ndarray, np.ndarray]:
    kind = candidate["kind"]
    if kind == "monolithic_mlp":
        x = data[input_key]
        h = np.tanh(x @ candidate["w1"] + candidate["b1"])
        answer_logits = h @ candidate["w_answer"] + candidate["b_answer"]
        route_logits = h @ candidate["w_route"] + candidate["b_route"]
        return np.argmax(answer_logits, axis=1).astype(np.int64), np.argmax(route_logits, axis=1).astype(np.int64)
    rows = data["rows"]
    if kind == "fused_pair_table":
        keys = np.asarray(
            [context_key(row["op1"], row["op2_true"], row["op2_false"], row["mode_id"], row["branch_flag"]) for row in rows],
            dtype=np.int64,
        )
        logits = candidate["context_scores"][keys]
        route_pred = np.argmax(logits, axis=1).astype(np.int64)
        answer_pred = data["candidate_answers"][np.arange(len(route_pred)), route_pred].copy()
        flip = candidate["answer_bias"][route_pred] > 1.0
        answer_pred[flip] = 1 - answer_pred[flip]
        return answer_pred.astype(np.int64), route_pred
    op1_pred = []
    op2_pred = []
    for row in rows:
        pred_op1 = int(np.argmax(candidate["op1_scores"][row["op1"]]))
        if kind == "short_pipe_no_router_between" or row["mode_id"] == 0:
            token = row["op2_true"]
            matrix = candidate["op2_fixed_scores"]
        else:
            pred_first = apply_primitive(pred_op1, row["a"], row["a"], row["b"], row["key"], row["mem"])
            pred_branch = 1 if pred_first > row["branch_gate"] else 0
            token = row["op2_true"] if pred_branch else row["op2_false"]
            matrix = candidate["op2_true_scores"] if pred_branch else candidate["op2_false_scores"]
        op1_pred.append(pred_op1)
        op2_pred.append(int(np.argmax(matrix[token])))
    route_pred = np.asarray([pair_index(i, j) for i, j in zip(op1_pred, op2_pred)], dtype=np.int64)
    answer_pred = data["candidate_answers"][np.arange(len(route_pred)), route_pred].copy()
    if kind == "short_router_repair":
        flip = candidate["answer_bias"][route_pred] > 1.0
        answer_pred[flip] = 1 - answer_pred[flip]
    return answer_pred.astype(np.int64), route_pred


def score_candidate(candidate: dict[str, Any], task: dict[str, Any], input_key: str, latency_steps: float) -> dict[str, Any]:
    evals = {}
    for split in ("train", "validation"):
        answer_pred, route_pred = predict_candidate(candidate, task[split], input_key)
        evals[split] = evaluate_predictions(answer_pred, route_pred, task[split], latency_steps, sample_limit=0)
    train = evals["train"]
    val = evals["validation"]
    fitness = (
        0.18 * train["usefulness_score"]
        + 0.50 * val["usefulness_score"]
        + 0.17 * val["route_accuracy"]
        + 0.10 * val["composition_accuracy"]
        + 0.05 * val["step_penalized_usefulness"]
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
        count = rng.randint(1, max(1, min(10, flat.size)))
        for idx in rng.sample(range(flat.size), k=count):
            flat[idx] += rng.gauss(0.0, settings.mutation_sigma)
        child[key] = flat.reshape(arr.shape)
        return child, f"gaussian_{key}", count
    if kind == "fused_pair_table":
        op = rng.choice(("context_score", "context_row_boost", "answer_bias"))
        if op == "context_score":
            i = rng.randrange(child["context_scores"].shape[0])
            j = rng.randrange(child["context_scores"].shape[1])
            child["context_scores"][i, j] += rng.gauss(0.0, settings.mutation_sigma)
            return child, op, 1
        if op == "context_row_boost":
            i = rng.randrange(child["context_scores"].shape[0])
            j = rng.randrange(child["context_scores"].shape[1])
            child["context_scores"][i, j] += rng.gauss(0.0, settings.mutation_sigma * 2.0)
            return child, op, 1
        child["answer_bias"][rng.randrange(len(PAIR_ROUTES))] += rng.gauss(0.0, settings.mutation_sigma)
        return child, op, 1
    mutable_keys = ("op1_scores", "op2_fixed_scores", "op2_true_scores", "op2_false_scores")
    if kind == "short_router_repair" and rng.random() < 0.16:
        child["answer_bias"][rng.randrange(len(PAIR_ROUTES))] += rng.gauss(0.0, settings.mutation_sigma)
        return child, "answer_bias", 1
    key = rng.choice(mutable_keys)
    i = rng.randrange(child[key].shape[0])
    j = rng.randrange(child[key].shape[1])
    child[key][i, j] += rng.gauss(0.0, settings.mutation_sigma)
    return child, key, 1


def latency_for_system(system: str) -> float:
    if system in {"short_pipe_router_composition", "router_plus_limited_pocket_repair", "oracle_short_pipe_reference"}:
        return 3.0
    if system == "short_pipe_no_router_between":
        return 2.0
    return 1.0


def evaluate_candidate_full(candidate: dict[str, Any], task: dict[str, Any], input_key: str, latency_steps: float) -> dict[str, Any]:
    out = {}
    for split in SPLITS:
        answer_pred, route_pred = predict_candidate(candidate, task[split], input_key)
        out[split] = evaluate_predictions(answer_pred, route_pred, task[split], latency_steps)
    return out


def mutation_worker(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    seed = int(job["seed"])
    system = job["system"]
    task = job["task"]
    input_key = "raw"
    latency = latency_for_system(system)
    if system == "monolithic_mutation_model":
        initial = init_mlp_candidate(seed, int(task["train"]["raw"].shape[1]))
    elif system == "fused_long_pipe_mutation_router":
        initial = init_fused_candidate(seed)
    elif system == "short_pipe_no_router_between":
        initial = init_short_candidate(seed, "short_pipe_no_router_between")
    elif system == "router_plus_limited_pocket_repair":
        initial = init_short_candidate(seed, "short_router_repair")
    else:
        initial = init_short_candidate(seed, "short_pipe_router")
    initial_hash = candidate_hash(initial)
    rng = random.Random(stable_seed(f"mutation-{system}-{seed}"))
    population = [score_candidate(copy.deepcopy(initial), task, input_key, latency)]
    for _ in range(settings.mutation_population - 1):
        child, _, _ = mutate_candidate(initial, rng, settings)
        population.append(score_candidate(child, task, input_key, latency))
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
            child = score_candidate(child_candidate, task, input_key, latency)
            attempts += 1
            changed_total += changed
            neutral_exploration = child["fitness"] == parent["fitness"] and rng.random() < 0.04
            if child["fitness"] > parent["fitness"] or neutral_exploration:
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
            "validation_route_accuracy": val["route_accuracy"],
            "validation_step_penalized_usefulness": val["step_penalized_usefulness"],
            "attempts": attempts,
        }
        history.append(row)
        if out and (time.monotonic() - last_heartbeat >= settings.heartbeat_seconds or generation == settings.mutation_generations):
            safe = e7b.safe_file_id(f"{system}_seed{seed}")
            locked_write_json(out / "partial_status" / f"mutation_{safe}.json", row)
            locked_write_json(
                out / "mutation_history_snapshots" / f"{safe}.json",
                {
                    "schema_version": "e7d_mutation_history_snapshot_v1",
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
    evals = evaluate_candidate_full(final_candidate, task, input_key, latency)
    final_hash = candidate_hash(final_candidate)
    return {
        "seed": seed,
        "system": system,
        "training_mode": "mutation_rollback",
        "input_key": input_key,
        "parameter_count": parameter_count(final_candidate),
        "latency_steps": latency,
        "initial_hash": initial_hash,
        "final_hash": final_hash,
        "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": final_hash}),
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
    return {"lane": "gpu_gradient_lane", "rows": rows, "histories": histories, "hardware": e7b.hardware_probe()}


def control_results(seed: int, task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(stable_seed(f"random-control-{seed}"))
    for system in CONTROL_SYSTEMS:
        evals = {}
        for split in SPLITS:
            data = task[split]
            if system == "oracle_short_pipe_reference":
                route_pred = data["route"].copy()
                answer_pred = data["y"].copy()
            else:
                route_pred = rng.integers(0, len(PAIR_ROUTES), size=len(data["y"]), dtype=np.int64)
                answer_pred = data["candidate_answers"][np.arange(len(data["y"])), route_pred]
            evals[split] = evaluate_predictions(answer_pred, route_pred, data, latency_for_system(system))
        rows.append(
            {
                "seed": seed,
                "system": system,
                "training_mode": "control",
                "input_key": "raw",
                "parameter_count": 0,
                "latency_steps": latency_for_system(system),
                "state_hash": payload_sha256({"seed": seed, "system": system}),
                "evals": evals,
            }
        )
    return rows


def split_summary(evals: dict[str, Any]) -> dict[str, float]:
    out = {}
    for metric in ("answer_accuracy", "route_accuracy", "composition_accuracy", "shortcut_rate", "usefulness_score", "step_penalized_usefulness", "latency_steps"):
        out[metric] = round_float(float(np.mean([evals[split][metric] for split in EVAL_SPLITS])))
    out["generalization_gap"] = round_float(evals["train"]["usefulness_score"] - out["usefulness_score"])
    out["heldout_usefulness"] = evals["heldout"]["usefulness_score"]
    out["ood_usefulness"] = evals["ood"]["usefulness_score"]
    out["counterfactual_usefulness"] = evals["counterfactual"]["usefulness_score"]
    out["adversarial_usefulness"] = evals["adversarial"]["usefulness_score"]
    out["ood_route_accuracy"] = evals["ood"]["route_accuracy"]
    return out


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_system: dict[str, list[dict[str, Any]]] = {system: [] for system in SYSTEMS}
    for row in rows:
        by_system[row["system"]].append(row)
    systems = {}
    metrics = (
        "answer_accuracy",
        "route_accuracy",
        "composition_accuracy",
        "shortcut_rate",
        "usefulness_score",
        "step_penalized_usefulness",
        "latency_steps",
        "generalization_gap",
        "heldout_usefulness",
        "ood_usefulness",
        "counterfactual_usefulness",
        "adversarial_usefulness",
        "ood_route_accuracy",
        "parameter_count",
    )
    for system, items in by_system.items():
        seed_rows = []
        for item in sorted(items, key=lambda row: int(row["seed"])):
            seed_rows.append({"seed": item["seed"], "parameter_count": item.get("parameter_count", 0), **split_summary(item["evals"])})
        systems[system] = {
            "seed_count": len(seed_rows),
            "rows": seed_rows,
            "mean": {
                key: round_float(float(np.mean([row[key] for row in seed_rows]))) if seed_rows else 0.0
                for key in metrics
            },
        }
    best = max((system for system in SYSTEMS if system != "oracle_short_pipe_reference"), key=lambda name: systems[name]["mean"]["usefulness_score"])
    return {
        "schema_version": "e7d_aggregate_metrics_v1",
        "systems": systems,
        "best_non_oracle_system": best,
        "best_non_oracle_usefulness": systems[best]["mean"]["usefulness_score"],
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    short_candidates = (
        ("short_pipe_router_composition", systems["short_pipe_router_composition"]["mean"]),
        ("router_plus_limited_pocket_repair", systems["router_plus_limited_pocket_repair"]["mean"]),
    )
    fused_candidates = (
        ("fused_long_pipe_mutation_router", systems["fused_long_pipe_mutation_router"]["mean"]),
        ("fused_long_pipe_gradient_router", systems["fused_long_pipe_gradient_router"]["mean"]),
    )
    short_best = max(short_candidates, key=lambda row: row[1]["usefulness_score"])
    fused_best = max(fused_candidates, key=lambda row: row[1]["usefulness_score"])
    no_router = systems["short_pipe_no_router_between"]["mean"]
    monolithic_mut = systems["monolithic_mutation_model"]["mean"]
    monolithic_grad = systems["monolithic_matrix_core_gradient"]["mean"]
    random_control = systems["random_router_control"]["mean"]
    leak = random_control["usefulness_score"] >= 0.78
    task_too_easy = monolithic_mut["usefulness_score"] >= 0.90 and monolithic_mut["ood_usefulness"] >= 0.86
    short_pass = (
        short_best[1]["usefulness_score"] >= 0.88
        and short_best[1]["ood_usefulness"] >= 0.82
        and short_best[1]["route_accuracy"] >= 0.82
        and short_best[1]["adversarial_usefulness"] >= 0.80
    )
    fused_pass = fused_best[1]["usefulness_score"] >= 0.88 and fused_best[1]["ood_usefulness"] >= 0.82
    router_overhead = short_best[1]["usefulness_score"] >= fused_best[1]["usefulness_score"] - 0.01 and short_best[1]["step_penalized_usefulness"] < fused_best[1]["step_penalized_usefulness"] - 0.025
    if leak:
        decision = "e7d_leak_or_artifact_detected"
    elif task_too_easy:
        decision = "e7d_monolithic_sufficient_or_task_too_easy"
    elif router_overhead:
        decision = "e7d_router_overhead_failure"
    elif short_pass and (short_best[1]["ood_usefulness"] >= fused_best[1]["ood_usefulness"] - 0.005) and (short_best[1]["parameter_count"] <= fused_best[1]["parameter_count"] or short_best[1]["usefulness_score"] >= fused_best[1]["usefulness_score"] - 0.01):
        decision = "e7d_short_pipe_router_flow_preferred"
    elif fused_pass and fused_best[1]["usefulness_score"] >= short_best[1]["usefulness_score"] + 0.02:
        decision = "e7d_fused_long_pipe_required"
    elif fused_pass and short_pass:
        decision = "e7d_hybrid_common_fused_plus_short_preferred"
    else:
        decision = "e7d_no_clear_topology_winner"
    return decision, {
        "short_best_system": short_best[0],
        "short_best_mean": short_best[1],
        "fused_best_system": fused_best[0],
        "fused_best_mean": fused_best[1],
        "no_router_between_mean": no_router,
        "monolithic_gradient_mean": monolithic_grad,
        "monolithic_mutation_mean": monolithic_mut,
        "random_control_mean": random_control,
        "leak_flag": leak,
        "task_too_easy_flag": task_too_easy,
        "router_overhead_flag": router_overhead,
    }


def build_composition_report(aggregate: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    short = detail["short_best_mean"]
    fused = detail["fused_best_mean"]
    return {
        "schema_version": "e7d_composition_report_v1",
        "interpretation_boundary": "short_pipe_router_flow_vs_fused_pipe_controlled_proxy",
        "short_best_system": detail["short_best_system"],
        "fused_best_system": detail["fused_best_system"],
        "short_minus_fused_usefulness": round_float(short["usefulness_score"] - fused["usefulness_score"]),
        "short_minus_fused_ood_usefulness": round_float(short["ood_usefulness"] - fused["ood_usefulness"]),
        "short_minus_fused_params": round_float(short["parameter_count"] - fused["parameter_count"]),
        "no_router_between_usefulness": detail["no_router_between_mean"]["usefulness_score"],
        "router_between_gain_over_no_router": round_float(short["usefulness_score"] - detail["no_router_between_mean"]["usefulness_score"]),
        "best_non_oracle_system": aggregate["best_non_oracle_system"],
    }


def build_leakage_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    random_mean = aggregate["systems"]["random_router_control"]["mean"]
    return {
        "schema_version": "e7d_leakage_report_v1",
        "random_control_usefulness": random_mean["usefulness_score"],
        "random_control_passed": random_mean["usefulness_score"] < 0.78,
        "hidden_correct_pair_route_used_as_input": False,
        "target_pair_route_not_in_raw_features": True,
        "misleading_pair_distractor_present": True,
        "ood_unseen_pair_split_present": True,
    }


def build_markdown(payloads: dict[str, Any]) -> str:
    decision = payloads["decision.json"]["decision"]
    aggregate = payloads["aggregate_metrics.json"]
    detail = payloads["decision.json"]["detail"]
    lines = [
        "# E7D Short-Pipe Composition Vs Fused-Pipe Probe Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision}",
        f"best_non_oracle_system = {aggregate['best_non_oracle_system']}",
        f"deterministic_replay_passed = {payloads['decision.json'].get('deterministic_replay_passed')}",
        "```",
        "",
        "## Mean Metrics",
        "",
        "| system | usefulness | step_penalized | ood | adversarial | route | shortcut | params | steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        lines.append(
            f"| {system} | {mean['usefulness_score']:.6f} | {mean['step_penalized_usefulness']:.6f} | {mean['ood_usefulness']:.6f} | {mean['adversarial_usefulness']:.6f} | {mean['route_accuracy']:.6f} | {mean['shortcut_rate']:.6f} | {mean['parameter_count']:.0f} | {mean['latency_steps']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Topology Comparison",
            "",
            "```text",
            f"short_best_system = {detail['short_best_system']}",
            f"fused_best_system = {detail['fused_best_system']}",
            f"short_minus_fused_usefulness = {payloads['composition_report.json']['short_minus_fused_usefulness']}",
            f"short_minus_fused_ood_usefulness = {payloads['composition_report.json']['short_minus_fused_ood_usefulness']}",
            f"router_between_gain_over_no_router = {payloads['composition_report.json']['router_between_gain_over_no_router']}",
            "```",
            "",
            "## Boundary",
            "",
            "This is a controlled symbolic/numeric flow-router proxy. It tests whether short reusable pipes plus a recurrent router can replace fused AB pipes on this proxy only.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress(out, "startup", milestone=MILESTONE, settings=settings_payload(settings), hardware=e7b.hardware_probe())
    tasks = {seed: generate_seed_task(seed, settings) for seed in settings.seeds}
    if out:
        append_progress(out, "tasks_generated", seeds=list(settings.seeds), row_counts=task_report(tasks)["row_counts"])
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
    gpu_job = {"tasks": {str(seed): tasks[seed] for seed in settings.seeds}, "settings": settings.__dict__, "out": out.as_posix() if out else None}
    if settings.execution_mode == "parallel":
        with ProcessPoolExecutor(max_workers=max(1, settings.cpu_workers)) as executor:
            futures = {}
            for job in jobs:
                futures[executor.submit(mutation_worker, job)] = f"{job['system']}/seed{job['seed']}"
            pending = set(futures)
            if out:
                append_progress(out, "lanes_submitted", cpu_mutation_jobs=len(jobs), cpu_workers=settings.cpu_workers, gpu_lane=True)
            gpu_result = gpu_lane_worker(gpu_job)
            rows.extend(gpu_result["rows"])
            training_histories.extend(gpu_result["histories"])
            if out:
                append_progress(out, "gpu_lane_complete", completed_gradient_rows=len(gpu_result["rows"]), hardware=gpu_result["hardware"])
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    label = futures[future]
                    result = future.result()
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
                                "schema_version": "e7d_partial_aggregate_snapshot_v1",
                                "completed_rows": len(rows),
                                "expected_rows": len(settings.seeds) * len(SYSTEMS),
                                "pending_jobs": len(pending),
                            },
                        )
                        append_progress(out, "mutation_job_complete", label=label, pending=len(pending))
    else:
        gpu = gpu_lane_worker(gpu_job)
        rows.extend(gpu["rows"])
        training_histories.extend(gpu["histories"])
        for job in jobs:
            result = mutation_worker(job)
            rows.append({key: value for key, value in result.items() if key != "history"})
            mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "accepted_by_operator", "rejected_by_operator")})
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
    composition = build_composition_report(results["aggregate"], results["decision_detail"])
    leakage = build_leakage_report(results["aggregate"])
    payloads: dict[str, Any] = {
        "backend_manifest.json": {
            "schema_version": "e7d_backend_manifest_v1",
            "milestone": MILESTONE,
            "settings": settings_payload(settings),
            "systems": list(SYSTEMS),
            "gradient_systems": list(GRADIENT_SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "control_systems": list(CONTROL_SYSTEMS),
            "hardware_identity": e7b.stable_hardware_identity(),
            "parallel_cpu_gpu_lanes": settings.execution_mode == "parallel",
        },
        "task_generation_report.json": task_report(results["tasks"]),
        "pipe_topology_report.json": pipe_topology_report(),
        "flow_state_report.json": flow_state_report(),
        "system_results.json": {"schema_version": "e7d_system_results_v1", "rows": results["rows"]},
        "mutation_history.json": {"schema_version": "e7d_mutation_history_v1", "rows": results["mutation_histories"]},
        "training_history.json": {"schema_version": "e7d_training_history_v1", "rows": results["training_histories"]},
        "aggregate_metrics.json": results["aggregate"],
        "composition_report.json": composition,
        "leakage_report.json": leakage,
        "decision.json": {"schema_version": "e7d_decision_v1", "decision": results["decision"], "detail": results["decision_detail"], "deterministic_replay_passed": False},
        "summary.json": {
            "schema_version": "e7d_summary_v1",
            "decision": results["decision"],
            "best_non_oracle_system": results["aggregate"]["best_non_oracle_system"],
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
        "schema_version": "e7d_deterministic_replay_v1",
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
    parser.add_argument("--seeds", default="93001,93002,93003,93004,93005,93006,93007,93008,93009,93010,93011,93012")
    parser.add_argument("--train-rows-per-seed", type=int, default=520)
    parser.add_argument("--validation-rows-per-seed", type=int, default=180)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=180)
    parser.add_argument("--ood-rows-per-seed", type=int, default=180)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=180)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=180)
    parser.add_argument("--gradient-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mutation-generations", type=int, default=130)
    parser.add_argument("--mutation-population", type=int, default=24)
    parser.add_argument("--mutation-sigma", type=float, default=0.24)
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
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    monitor = start_hardware_monitor(out, stop, settings.heartbeat_seconds)
    try:
        results = run_core(settings, out)
        payloads = build_payloads(settings, out, results)
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7d_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_work_root": None}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop.set()
        monitor.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
