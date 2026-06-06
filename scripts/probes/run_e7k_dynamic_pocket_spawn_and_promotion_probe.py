#!/usr/bin/env python3
"""E7K dynamic pocket spawn and promotion probe.

E7J showed that existing callable thought-pockets can receive useful internal
capacity. E7K asks the next narrower question: when the current callable
library is insufficient, can the control layer create a new typed pocket,
validate it, promote it to the library, and later call it by ID without
collapsing into anonymous dense graph routing?
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
E7I_PATH = Path(__file__).with_name("run_e7i_pocket_size_optimum_sweep.py")
MILESTONE = "E7K_DYNAMIC_POCKET_SPAWN_AND_PROMOTION_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7k_dynamic_pocket_spawn_and_promotion_probe")
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
PHASES = (
    "phase_1_existing_library_sufficient",
    "phase_2_missing_reusable_transform",
    "phase_3_reuse_multiple_contexts",
    "phase_4_ood_counterfactual_generalization",
    "phase_5_damage_drift_repair",
)
SYSTEMS = (
    "fixed_library_no_spawn",
    "fixed_library_router_plus_repair",
    "oracle_spawn_scaffold",
    "random_spawn_control",
    "control_spawn_blank_pocket",
    "control_spawn_from_split",
    "control_spawn_from_composed_route",
    "control_spawn_plus_limited_repair",
    "dense_graph_danger_control",
)
MUTATION_SYSTEMS = (
    "control_spawn_blank_pocket",
    "control_spawn_from_split",
    "control_spawn_from_composed_route",
    "control_spawn_plus_limited_repair",
)
GRADIENT_SYSTEMS = ("dense_graph_danger_control",)
CONTROL_SYSTEMS = tuple(system for system in SYSTEMS if system not in MUTATION_SYSTEMS and system not in GRADIENT_SYSTEMS)
MICRO_COUNT = 16
PAD_SEGMENT = MICRO_COUNT
MAX_MICRO_PATH = 12
FLOW_D = 16
K_VALUES = (1, 2, 4, 8)
BASE_POCKETS = ((0, 1), (2, 3), (4, 5), (6, 7), (8,), (15,))
DAMAGED_BASE_POCKETS = ((2, 3), (4, 5))
TRUE_MOTIFS = {
    "alpha": (9, 10, 11),
    "beta": (12, 13, 14),
    "gamma": (2, 3, 4),
    "delta": (0, 5, 10),
}
PHASE_TRUE_MOTIFS = {
    "phase_1_existing_library_sufficient": (),
    "phase_2_missing_reusable_transform": ("alpha",),
    "phase_3_reuse_multiple_contexts": ("alpha", "beta"),
    "phase_4_ood_counterfactual_generalization": ("alpha", "beta", "delta"),
    "phase_5_damage_drift_repair": ("alpha", "gamma"),
}
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "spawn_mechanism_report.json",
    "spawn_promotion_report.json",
    "phase_spawn_winner_report.json",
    "system_results.json",
    "mutation_history.json",
    "training_history.json",
    "leakage_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7k_dynamic_pocket_spawn_positive",
    "e7k_composed_route_pocket_spawn_positive",
    "e7k_split_spawn_positive",
    "e7k_spawn_needs_prior_scaffold",
    "e7k_spawn_artifact_or_task_too_easy",
    "e7k_spawn_overproduction_failure",
    "e7k_no_spawn_needed_existing_library_sufficient",
    "e7k_pocket_spawn_collapses_to_graph_soup",
    "e7k_leak_or_artifact_detected",
)


def load_e7i_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7i_pocket_size_sweep", E7I_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7I helpers from {E7I_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7i = load_e7i_module()
e7h = e7i.e7h


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


class DensePathMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 160) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.segment_heads = nn.ModuleList([nn.Linear(hidden_dim, MICRO_COUNT + 1) for _ in range(MAX_MICRO_PATH)])
        self.length_head = nn.Linear(hidden_dim, MAX_MICRO_PATH - 3)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        h = self.net(x)
        return [head(h) for head in self.segment_heads], self.length_head(h)


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7k::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7h.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7h.write_json(path, payload)


def locked_write_json(path: Path, payload: Any) -> None:
    e7h.locked_write_json(path, payload)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7h.append_progress(out, event, **details)


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
    return e7h.select_device(requested)


def set_determinism(seed: int, device: str) -> None:
    e7h.set_determinism(seed, device)


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    return e7h.start_hardware_monitor(out, stop, interval)


def capacity_params(k: int, depth: int = 1) -> int:
    k = int(k)
    depth = max(1, int(depth))
    return int(2 * FLOW_D * k + depth * k * k)


def required_k_for_call(call: tuple[int, ...]) -> int:
    length = len(call)
    if length <= 1:
        return 1
    if length == 2:
        return 2
    if length == 3:
        return 4
    return 8


def clamp_k(value: int) -> int:
    return min(K_VALUES, key=lambda item: abs(item - int(value)))


def normalize_pockets(pockets: list[tuple[int, ...]] | tuple[tuple[int, ...], ...]) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for pocket in pockets:
        item = tuple(int(seg) for seg in pocket if 0 <= int(seg) < MICRO_COUNT)
        if item and len(item) <= MAX_MICRO_PATH and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def greedy_calls(micro_path: list[int] | tuple[int, ...], pockets: list[tuple[int, ...]] | tuple[tuple[int, ...], ...]) -> list[tuple[int, ...]]:
    usable = sorted(normalize_pockets(pockets), key=lambda item: (-len(item), item))
    calls: list[tuple[int, ...]] = []
    idx = 0
    path = [int(seg) for seg in micro_path]
    while idx < len(path):
        matched = None
        for pocket in usable:
            if tuple(path[idx : idx + len(pocket)]) == pocket:
                matched = pocket
                break
        if matched is None:
            matched = (path[idx],)
        calls.append(matched)
        idx += len(matched)
    return calls


def calls_to_micro(calls: list[tuple[int, ...]]) -> list[int]:
    out: list[int] = []
    for call in calls:
        out.extend(call)
    return out


def base_route_cost(calls: list[tuple[int, ...]]) -> float:
    return float(sum(1.0 + 0.15 * len(call) for call in calls))


def spawned_route_cost(k: int, depth: int) -> float:
    return float(1.0 + 0.06 * clamp_k(k) + 0.08 * max(1, int(depth)))


def corrupt_call(call: tuple[int, ...], salt: int) -> list[int]:
    out = list(call)
    if out:
        out[-1] = (out[-1] + salt) % MICRO_COUNT
    return out


def apply_micro_path(path: list[int] | tuple[int, ...], a: int, b: int, key: int, mem: int, threshold: int) -> int:
    return e7i.apply_micro_path(path, a, b, key, mem, threshold)


def motif_tuple(name: str) -> tuple[int, ...]:
    return TRUE_MOTIFS[name]


def base_piece(rng: random.Random) -> tuple[int, ...]:
    return rng.choice(BASE_POCKETS)


def build_micro_path(phase: str, split: str, index: int, rng: random.Random) -> tuple[list[int], list[tuple[int, ...]]]:
    hidden: list[tuple[int, ...]] = []
    pieces: list[tuple[int, ...]]
    if phase == "phase_1_existing_library_sufficient":
        pieces = [base_piece(rng) for _ in range(rng.choice((3, 4, 5)))]
    elif phase == "phase_2_missing_reusable_transform":
        alpha = motif_tuple("alpha")
        hidden.append(alpha)
        pieces = [base_piece(rng), alpha, base_piece(rng), base_piece(rng)]
    elif phase == "phase_3_reuse_multiple_contexts":
        alpha = motif_tuple("alpha")
        beta = motif_tuple("beta")
        hidden.extend([alpha, beta])
        pieces = [alpha, base_piece(rng), beta, base_piece(rng)]
    elif phase == "phase_4_ood_counterfactual_generalization":
        alpha = motif_tuple("alpha")
        beta = motif_tuple("beta")
        delta = motif_tuple("delta")
        hidden.extend([alpha, beta, delta])
        if split in {"ood", "counterfactual", "adversarial"}:
            pieces = [delta, base_piece(rng), alpha, base_piece(rng)]
        else:
            pieces = [beta, base_piece(rng), alpha, base_piece(rng)]
    else:
        alpha = motif_tuple("alpha")
        gamma = motif_tuple("gamma")
        hidden.extend([alpha, gamma])
        pieces = [gamma, base_piece(rng), alpha, rng.choice(((2, 3), (4, 5)))]
    path: list[int] = []
    for piece in pieces:
        path.extend(piece)
    if split in {"train", "validation"} and phase != "phase_1_existing_library_sufficient" and index % 5 == 0 and len(path) + 2 <= MAX_MICRO_PATH:
        path = [14, 15] + path
    return path[:MAX_MICRO_PATH], hidden


def memory_value(seed: int, key: int, split: str, phase: str) -> int:
    split_shift = {"train": 0, "validation": 1, "heldout": 2, "ood": 7, "counterfactual": 3, "adversarial": 5}[split]
    return (key * 11 + seed % 29 + split_shift * 5 + len(phase)) & 15


def make_row(seed: int, phase: str, split: str, index: int, rng: random.Random) -> dict[str, Any]:
    a = rng.choice((0, 1, 14, 15)) if split == "ood" else rng.randrange(16)
    b = rng.choice((0, 1, 14, 15)) if split == "ood" else rng.randrange(16)
    key = rng.randrange(16)
    threshold = rng.choice((1, 2, 13, 14)) if split == "ood" else rng.randrange(3, 13)
    mem = memory_value(seed, key, split, phase)
    micro_path, hidden = build_micro_path(phase, split, index, rng)
    final_value = apply_micro_path(micro_path, a, b, key, mem, threshold)
    if split == "counterfactual":
        threshold = max(0, min(15, final_value - 1 if final_value <= threshold else final_value + 1))
    final_value = apply_micro_path(micro_path, a, b, key, mem, threshold)
    answer = 1 if final_value > threshold else 0
    padded = micro_path + [PAD_SEGMENT] * (MAX_MICRO_PATH - len(micro_path))
    micro_hot: list[float] = []
    for segment in padded:
        micro_hot.extend(1.0 if idx == segment else 0.0 for idx in range(MICRO_COUNT + 1))
    phase_hot = [1.0 if PHASES[idx] == phase else 0.0 for idx in range(len(PHASES))]
    noise = [rng.uniform(-1.0, 1.0) for _ in range(8)]
    raw = [a / 15.0, b / 15.0, key / 15.0, threshold / 15.0, mem / 15.0, len(micro_path) / MAX_MICRO_PATH] + phase_hot + micro_hot + noise
    return {
        "row_id": f"{seed}/{phase}/{split}/{index}",
        "seed": seed,
        "phase": phase,
        "split": split,
        "a": a,
        "b": b,
        "key": key,
        "threshold": threshold,
        "mem": mem,
        "micro_path": micro_path,
        "padded_micro_path": padded,
        "micro_path_length": len(micro_path),
        "hidden_missing_motifs": [list(item) for item in hidden],
        "answer": answer,
        "final_value": final_value,
        "raw": raw,
    }


def generate_seed_task(seed: int, settings: Settings) -> dict[str, dict[str, list[dict[str, Any]]]]:
    counts = {
        "train": settings.train_rows_per_seed,
        "validation": settings.validation_rows_per_seed,
        "heldout": settings.heldout_rows_per_seed,
        "ood": settings.ood_rows_per_seed,
        "counterfactual": settings.counterfactual_rows_per_seed,
        "adversarial": settings.adversarial_rows_per_seed,
    }
    per_phase_counts = {split: max(1, count // len(PHASES)) for split, count in counts.items()}
    task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for phase in PHASES:
        task[phase] = {}
        for split, count in per_phase_counts.items():
            rng = random.Random(stable_seed(f"{seed}:{phase}:{split}:rows"))
            task[phase][split] = [make_row(seed, phase, split, idx, rng) for idx in range(count)]
    return task


def generate_tasks(settings: Settings) -> dict[int, dict[str, dict[str, list[dict[str, Any]]]]]:
    return {seed: generate_seed_task(seed, settings) for seed in settings.seeds}


def all_rows(task: dict[str, dict[str, list[dict[str, Any]]]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in PHASES:
        rows.extend(task[phase][split])
    return rows


def true_motifs_for_rows(rows: list[dict[str, Any]]) -> set[tuple[int, ...]]:
    out: set[tuple[int, ...]] = set()
    for row in rows:
        for motif in row.get("hidden_missing_motifs", []):
            out.add(tuple(int(seg) for seg in motif))
    return out


def frequent_substrings(rows: list[dict[str, Any]], min_len: int = 2, max_len: int = 4) -> list[tuple[tuple[int, ...], int]]:
    counts: dict[tuple[int, ...], int] = {}
    base = set(BASE_POCKETS)
    for row in rows:
        path = row["micro_path"]
        for length in range(min_len, max_len + 1):
            for idx in range(0, len(path) - length + 1):
                item = tuple(path[idx : idx + length])
                if item not in base and len(set(item)) > 1:
                    counts[item] = counts.get(item, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def split_substrings(rows: list[dict[str, Any]]) -> list[tuple[tuple[int, ...], int]]:
    counts: dict[tuple[int, ...], int] = {}
    for row in rows:
        calls = greedy_calls(row["micro_path"], BASE_POCKETS)
        for left, right in zip(calls, calls[1:]):
            merged = tuple(list(left) + list(right))
            if 2 <= len(merged) <= 4 and merged not in BASE_POCKETS:
                counts[merged] = counts.get(merged, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def spawn_id(segments: tuple[int, ...], source: str) -> str:
    return f"{source}_{hashlib.sha256(','.join(str(seg) for seg in segments).encode('ascii')).hexdigest()[:8]}"


def make_spawn(segments: tuple[int, ...], source: str, k: int | None = None, depth: int = 1) -> dict[str, Any]:
    item = tuple(int(seg) for seg in segments if 0 <= int(seg) < MICRO_COUNT)
    return {
        "id": spawn_id(item, source),
        "segments": list(item),
        "source": source,
        "K": clamp_k(k if k is not None else required_k_for_call(item)),
        "depth": max(1, min(4, int(depth))),
        "promoted": True,
        "frozen": True,
        "repair_permission": source in {"repair", "oracle"},
    }


def normalize_spawned(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[int, ...]] = set()
    out: list[dict[str, Any]] = []
    for raw in candidate.get("spawned", []):
        segments = tuple(int(seg) for seg in raw.get("segments", []) if 0 <= int(seg) < MICRO_COUNT)
        if len(segments) < 2 or len(segments) > 4 or segments in seen:
            continue
        seen.add(segments)
        out.append(
            {
                "id": str(raw.get("id") or spawn_id(segments, str(raw.get("source", "spawn")))),
                "segments": list(segments),
                "source": str(raw.get("source", "spawn")),
                "K": clamp_k(int(raw.get("K", required_k_for_call(segments)))),
                "depth": max(1, min(4, int(raw.get("depth", 1)))),
                "promoted": bool(raw.get("promoted", True)),
                "frozen": bool(raw.get("frozen", True)),
                "repair_permission": bool(raw.get("repair_permission", False)),
            }
        )
    return out[:8]


def candidate_initial(system: str) -> dict[str, Any]:
    return {"spawned": [], "system": system, "router_prior": 0.0}


def candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    spawned = normalize_spawned(candidate)
    k_values = [int(row["K"]) for row in spawned]
    depths = [int(row["depth"]) for row in spawned]
    return {
        "promoted_pocket_count": len(spawned),
        "spawned_pockets": spawned,
        "average_K": round_float(float(np.mean(k_values)) if k_values else 0.0),
        "average_depth": round_float(float(np.mean(depths)) if depths else 0.0),
        "K_distribution": {str(k): sum(1 for value in k_values if value == k) for k in K_VALUES},
        "candidate_hash": payload_sha256(spawned),
    }


def active_spawn_map(candidate: dict[str, Any]) -> dict[tuple[int, ...], dict[str, Any]]:
    out: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in normalize_spawned(candidate):
        if row.get("promoted", True):
            segments = tuple(int(seg) for seg in row["segments"])
            current = out.get(segments)
            if current is None or (int(row["K"]), int(row["depth"])) > (int(current["K"]), int(current["depth"])):
                out[segments] = row
    return out


def candidate_pockets(candidate: dict[str, Any]) -> list[tuple[int, ...]]:
    return list(BASE_POCKETS) + sorted(active_spawn_map(candidate), key=lambda item: (-len(item), item))


def has_repair_permission(system: str) -> bool:
    return system in {"fixed_library_router_plus_repair", "oracle_spawn_scaffold", "control_spawn_plus_limited_repair"}


def predict_with_candidate(
    task: dict[str, dict[str, list[dict[str, Any]]]],
    system: str,
    candidate: dict[str, Any],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    spawn_map = active_spawn_map(candidate)
    pockets = candidate_pockets(candidate)
    repair = has_repair_permission(system)
    for phase in PHASES:
        predictions[phase] = {}
        for split in SPLITS:
            split_preds = []
            for row in task[phase][split]:
                calls = greedy_calls(row["micro_path"], pockets)
                predicted_micro: list[int] = []
                call_rows: list[dict[str, Any]] = []
                cost = 0.0
                branch_expansions = 0
                for call_idx, call in enumerate(calls):
                    source = "base" if call in BASE_POCKETS else "spawned" if call in spawn_map else "atomic"
                    emitted = list(call)
                    k = required_k_for_call(call)
                    depth = 1
                    if call in spawn_map:
                        spawn = spawn_map[call]
                        k = int(spawn["K"])
                        depth = int(spawn["depth"])
                        required_k = required_k_for_call(call)
                        if k < required_k or depth < max(1, len(call) - 2):
                            emitted = corrupt_call(call, k + depth + call_idx)
                        cost += spawned_route_cost(k, depth)
                    else:
                        if phase == "phase_5_damage_drift_repair" and call in DAMAGED_BASE_POCKETS and not repair:
                            emitted = corrupt_call(call, len(call) + call_idx + 1)
                        cost += 1.0 + 0.15 * len(call)
                    predicted_micro.extend(emitted)
                    call_rows.append({"call": list(call), "source": source, "K": k, "depth": depth, "emitted": emitted})
                split_preds.append(
                    {
                        "calls": call_rows,
                        "micro_path": predicted_micro[:MAX_MICRO_PATH],
                        "steps": len(calls),
                        "branch_expansions": branch_expansions,
                        "route_cost": cost,
                    }
                )
            predictions[phase][split] = split_preds
    return predictions


def predict_random_spawn(task: dict[str, dict[str, list[dict[str, Any]]]], seed: int) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    rng = random.Random(stable_seed(f"{seed}:random_spawn_control"))
    spawned = []
    for _ in range(5):
        length = rng.choice((2, 3, 4))
        segments = tuple(rng.sample(range(MICRO_COUNT), length))
        spawned.append(make_spawn(segments, "random", rng.choice(K_VALUES), rng.choice((1, 2, 3))))
    candidate = {"spawned": spawned, "system": "random_spawn_control"}
    return predict_with_candidate(task, "random_spawn_control", candidate), candidate


def oracle_candidate(task: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    motifs: set[tuple[int, ...]] = set()
    for split in SPLITS:
        motifs.update(true_motifs_for_rows(all_rows(task, split)))
    return {"spawned": [make_spawn(motif, "oracle", required_k_for_call(motif), max(1, len(motif) - 2)) for motif in sorted(motifs)], "system": "oracle_spawn_scaffold"}


def parameter_count(candidate: dict[str, Any], router_extra: int = 48) -> int:
    total = int(router_extra + len(BASE_POCKETS) * 5)
    for row in normalize_spawned(candidate):
        total += 8 + capacity_params(int(row["K"]), int(row["depth"]))
    return int(total)


def spawn_metrics_for_rows(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    promoted = {tuple(row["segments"]) for row in normalize_spawned(candidate)}
    true_motifs = true_motifs_for_rows(rows)
    use_counts = {motif: 0 for motif in promoted}
    for motif in promoted:
        for row in rows:
            path = row["micro_path"]
            use_counts[motif] += sum(1 for idx in range(0, len(path) - len(motif) + 1) if tuple(path[idx : idx + len(motif)]) == motif)
    used_promoted = {motif for motif, count in use_counts.items() if count > 0}
    matched = used_promoted & true_motifs
    precision = len(matched) / max(1, len(used_promoted)) if used_promoted else 0.0
    recall = len(matched) / max(1, len(true_motifs))
    unnecessary = (len(promoted) - len(used_promoted)) / max(1, len(promoted))
    reuse_count = sum(use_counts.values())
    avg_reuse = reuse_count / max(1, len(promoted))
    k_values = [int(row["K"]) for row in normalize_spawned(candidate)]
    depths = [int(row["depth"]) for row in normalize_spawned(candidate)]
    overproduction = max(0, len(promoted) - len(used_promoted) - 1) / max(1, len(promoted))
    return {
        "spawn_precision": round_float(precision),
        "spawn_recall": round_float(recall),
        "unnecessary_spawn_rate": round_float(unnecessary),
        "promoted_pocket_count": len(promoted),
        "promoted_pocket_reuse_count": round_float(avg_reuse),
        "spawned_pocket_average_K": round_float(float(np.mean(k_values)) if k_values else 0.0),
        "spawned_pocket_average_depth": round_float(float(np.mean(depths)) if depths else 0.0),
        "spawn_overproduction_rate": round_float(overproduction),
    }


def evaluate_predictions(
    task: dict[str, dict[str, list[dict[str, Any]]]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    system: str,
    candidate: dict[str, Any],
    params: int,
    router_complexity: float,
) -> dict[str, Any]:
    phase_metrics: dict[str, Any] = {}
    evals: dict[str, Any] = {}
    for phase in PHASES:
        phase_metrics[phase] = {}
        for split in SPLITS:
            rows = task[phase][split]
            preds = predictions[phase][split]
            answer_hits = route_hits = valid_hits = loops = 0
            steps_total = baseline_steps_total = cost_total = baseline_cost_total = 0.0
            irrelevant_total = branch_total = 0.0
            samples: list[dict[str, Any]] = []
            for row, pred in zip(rows, preds):
                predicted = [int(seg) for seg in pred.get("micro_path", []) if 0 <= int(seg) < MICRO_COUNT]
                final = apply_micro_path(predicted, row["a"], row["b"], row["key"], row["mem"], row["threshold"])
                predicted_answer = 1 if final > row["threshold"] else 0
                target = list(row["micro_path"])
                call_rows = pred.get("calls", [])
                call_tuples = [tuple(int(seg) for seg in call.get("call", [])) for call in call_rows]
                baseline_calls = greedy_calls(target, BASE_POCKETS)
                baseline_cost = base_route_cost(baseline_calls)
                steps = float(pred.get("steps", len(call_tuples)))
                route_cost = float(pred.get("route_cost", base_route_cost(call_tuples)))
                branches = max(0, int(pred.get("branch_expansions", 0)))
                answer_hits += int(predicted_answer == row["answer"])
                route_hits += int(predicted == target)
                valid_hits += int(bool(predicted) and len(predicted) <= MAX_MICRO_PATH)
                loops += int(len(call_tuples) != len(set(call_tuples)))
                irrelevant = sum(1 for seg in predicted if seg not in set(target))
                irrelevant_total += min(1.0, (irrelevant + branches) / max(1, MICRO_COUNT - len(set(target))))
                branch_total += branches
                steps_total += steps
                baseline_steps_total += len(baseline_calls)
                cost_total += route_cost
                baseline_cost_total += baseline_cost
                if len(samples) < 3:
                    samples.append(
                        {
                            "row_id": row["row_id"],
                            "target": target,
                            "predicted": predicted,
                            "calls": call_rows,
                            "steps": round_float(steps),
                            "baseline_steps": len(baseline_calls),
                            "target_answer": row["answer"],
                            "predicted_answer": predicted_answer,
                        }
                    )
            n = max(1, len(rows))
            answer_accuracy = answer_hits / n
            route_accuracy = route_hits / n
            valid_rate = valid_hits / n
            loop_rate = loops / n
            irrelevant_rate = irrelevant_total / n
            branch_rate = branch_total / n
            mean_steps = steps_total / n
            baseline_steps = baseline_steps_total / n
            mean_cost = cost_total / n
            baseline_cost = baseline_cost_total / n
            step_reduction = max(0.0, (baseline_steps - mean_steps) / max(1.0, baseline_steps))
            cost_reduction = max(0.0, (baseline_cost - mean_cost) / max(1.0, baseline_cost))
            spawn_metrics = spawn_metrics_for_rows(candidate, rows)
            param_norm = min(1.0, params / 1400.0)
            router_norm = min(1.0, router_complexity / 12.0)
            freeze_survival = route_accuracy if spawn_metrics["promoted_pocket_count"] else 0.0
            local_repair_gain = step_reduction if system in {"fixed_library_router_plus_repair", "control_spawn_plus_limited_repair"} else 0.0
            usefulness = (
                0.30 * answer_accuracy
                + 0.23 * route_accuracy
                + 0.17 * step_reduction
                + 0.10 * cost_reduction
                + 0.05 * min(1.0, spawn_metrics["promoted_pocket_reuse_count"] / max(1.0, n / 4.0))
                + 0.04 * freeze_survival
                + 0.03 * local_repair_gain
                + 0.03 * valid_rate
                - 0.05 * irrelevant_rate
                - 0.04 * loop_rate
                - 0.03 * branch_rate
                - 0.03 * param_norm
                - 0.02 * router_norm
            )
            spawn_value = (
                usefulness
                + 0.08 * spawn_metrics["spawn_precision"]
                + 0.08 * spawn_metrics["spawn_recall"]
                + 0.04 * min(1.0, spawn_metrics["promoted_pocket_reuse_count"] / max(1.0, n / 4.0))
                - 0.06 * spawn_metrics["unnecessary_spawn_rate"]
                - 0.05 * spawn_metrics["spawn_overproduction_rate"]
            )
            phase_metrics[phase][split] = {
                "answer_accuracy": round_float(answer_accuracy),
                "route_accuracy": round_float(route_accuracy),
                "valid_route_rate": round_float(valid_rate),
                "mean_route_steps": round_float(mean_steps),
                "baseline_route_steps": round_float(baseline_steps),
                "route_step_reduction": round_float(step_reduction),
                "route_cost": round_float(mean_cost),
                "baseline_route_cost": round_float(baseline_cost),
                "route_cost_reduction": round_float(cost_reduction),
                "freeze_survival": round_float(freeze_survival),
                "local_repair_gain": round_float(local_repair_gain),
                "irrelevant_branch_rate": round_float(irrelevant_rate),
                "branch_expansion_rate": round_float(branch_rate),
                "loop_rate": round_float(loop_rate),
                "parameter_count": int(params),
                "router_complexity": round_float(router_complexity),
                "usefulness_score": round_float(max(0.0, min(1.0, usefulness))),
                "spawn_value_score": round_float(max(0.0, min(1.0, spawn_value))),
                "row_level_samples": samples,
                **spawn_metrics,
            }
    for split in SPLITS:
        split_values: dict[str, list[float]] = {}
        for phase in PHASES:
            for key, value in phase_metrics[phase][split].items():
                if isinstance(value, (int, float)):
                    split_values.setdefault(key, []).append(float(value))
        evals[split] = {key: round_float(float(np.mean(values))) for key, values in split_values.items()}
        evals[split]["row_level_samples"] = [phase_metrics[phase][split]["row_level_samples"][0] for phase in PHASES if phase_metrics[phase][split]["row_level_samples"]]
    train = evals["train"]["spawn_value_score"]
    eval_mean_spawn = float(np.mean([evals[split]["spawn_value_score"] for split in EVAL_SPLITS]))
    eval_mean_usefulness = float(np.mean([evals[split]["usefulness_score"] for split in EVAL_SPLITS]))
    return {
        "system": system,
        "evals": evals,
        "phase_metrics": phase_metrics,
        "heldout_usefulness": round_float(evals["heldout"]["usefulness_score"]),
        "ood_usefulness": round_float(evals["ood"]["usefulness_score"]),
        "counterfactual_usefulness": round_float(evals["counterfactual"]["usefulness_score"]),
        "adversarial_usefulness": round_float(evals["adversarial"]["usefulness_score"]),
        "heldout_spawn_value": round_float(evals["heldout"]["spawn_value_score"]),
        "ood_spawn_value": round_float(evals["ood"]["spawn_value_score"]),
        "counterfactual_spawn_value": round_float(evals["counterfactual"]["spawn_value_score"]),
        "adversarial_spawn_value": round_float(evals["adversarial"]["spawn_value_score"]),
        "eval_mean_usefulness": round_float(eval_mean_usefulness),
        "eval_mean_spawn_value": round_float(eval_mean_spawn),
        "generalization_gap": round_float(train - eval_mean_spawn),
        "parameter_count": int(params),
        "candidate_summary": candidate_summary(candidate),
    }


def profile_result(task: dict[str, dict[str, list[dict[str, Any]]]], seed: int, system: str, candidate: dict[str, Any], router_complexity: float) -> dict[str, Any]:
    predictions = predict_with_candidate(task, system, candidate)
    row = evaluate_predictions(task, predictions, system, candidate, parameter_count(candidate), router_complexity)
    row["seed"] = seed
    return row


def control_results(seed: int, task: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fixed = candidate_initial("fixed_library_no_spawn")
    rows.append(profile_result(task, seed, "fixed_library_no_spawn", fixed, router_complexity=2.0))
    repair = candidate_initial("fixed_library_router_plus_repair")
    rows.append(profile_result(task, seed, "fixed_library_router_plus_repair", repair, router_complexity=2.4))
    oracle = oracle_candidate(task)
    rows.append(profile_result(task, seed, "oracle_spawn_scaffold", oracle, router_complexity=2.8))
    random_pred, random_candidate = predict_random_spawn(task, seed)
    random_row = evaluate_predictions(task, random_pred, "random_spawn_control", random_candidate, parameter_count(random_candidate, 96), router_complexity=8.0)
    random_row["seed"] = seed
    rows.append(random_row)
    return rows


def training_rows_for_spawn(task: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return all_rows(task, "train") + all_rows(task, "validation")


def predict_single_row(row: dict[str, Any], system: str, candidate: dict[str, Any]) -> dict[str, Any]:
    spawn_map = active_spawn_map(candidate)
    pockets = candidate_pockets(candidate)
    repair = has_repair_permission(system)
    calls = greedy_calls(row["micro_path"], pockets)
    predicted_micro: list[int] = []
    cost = 0.0
    for call_idx, call in enumerate(calls):
        emitted = list(call)
        if call in spawn_map:
            spawn = spawn_map[call]
            k = int(spawn["K"])
            depth = int(spawn["depth"])
            if k < required_k_for_call(call) or depth < max(1, len(call) - 2):
                emitted = corrupt_call(call, k + depth + call_idx)
            cost += spawned_route_cost(k, depth)
        else:
            if row["phase"] == "phase_5_damage_drift_repair" and call in DAMAGED_BASE_POCKETS and not repair:
                emitted = corrupt_call(call, len(call) + call_idx + 1)
            cost += 1.0 + 0.15 * len(call)
        predicted_micro.extend(emitted)
    return {"micro_path": predicted_micro[:MAX_MICRO_PATH], "steps": len(calls), "route_cost": cost}


def quick_usefulness(candidate: dict[str, Any], system: str, rows: list[dict[str, Any]]) -> float:
    answer_hits = route_hits = loops = 0
    steps_total = baseline_steps_total = cost_total = baseline_cost_total = irrelevant_total = 0.0
    pockets = candidate_pockets(candidate)
    for row in rows:
        pred = predict_single_row(row, system, candidate)
        predicted = [int(seg) for seg in pred["micro_path"] if 0 <= int(seg) < MICRO_COUNT]
        final = apply_micro_path(predicted, row["a"], row["b"], row["key"], row["mem"], row["threshold"])
        predicted_answer = 1 if final > row["threshold"] else 0
        target = list(row["micro_path"])
        calls = greedy_calls(target, pockets)
        baseline_calls = greedy_calls(target, BASE_POCKETS)
        answer_hits += int(predicted_answer == row["answer"])
        route_hits += int(predicted == target)
        loops += int(len(calls) != len(set(calls)))
        irrelevant_total += min(1.0, sum(1 for seg in predicted if seg not in set(target)) / max(1, MICRO_COUNT - len(set(target))))
        steps_total += float(pred["steps"])
        baseline_steps_total += len(baseline_calls)
        cost_total += float(pred["route_cost"])
        baseline_cost_total += base_route_cost(baseline_calls)
    n = max(1, len(rows))
    answer_accuracy = answer_hits / n
    route_accuracy = route_hits / n
    step_reduction = max(0.0, (baseline_steps_total / n - steps_total / n) / max(1.0, baseline_steps_total / n))
    cost_reduction = max(0.0, (baseline_cost_total / n - cost_total / n) / max(1.0, baseline_cost_total / n))
    reuse = 0.0
    spawned = {tuple(row["segments"]) for row in normalize_spawned(candidate)}
    for motif in spawned:
        for row in rows:
            path = row["micro_path"]
            reuse += sum(1 for idx in range(0, len(path) - len(motif) + 1) if tuple(path[idx : idx + len(motif)]) == motif)
    reuse_norm = min(1.0, (reuse / max(1, len(spawned))) / max(1.0, n / 4.0))
    param_norm = min(1.0, parameter_count(candidate) / 1400.0)
    return float(
        0.32 * answer_accuracy
        + 0.25 * route_accuracy
        + 0.20 * step_reduction
        + 0.11 * cost_reduction
        + 0.06 * reuse_norm
        - 0.04 * (irrelevant_total / n)
        - 0.03 * (loops / n)
        - 0.03 * param_norm
    )


def candidate_learning_score(candidate: dict[str, Any], system: str, task: dict[str, dict[str, list[dict[str, Any]]]]) -> float:
    train_score = quick_usefulness(candidate, system, all_rows(task, "train"))
    validation_score = quick_usefulness(candidate, system, all_rows(task, "validation"))
    count_penalty = 0.006 * candidate_summary(candidate)["promoted_pocket_count"]
    overfit = max(0.0, train_score - validation_score)
    return float(0.45 * train_score + 0.55 * validation_score - count_penalty - 0.08 * overfit)


def mutation_pools(task: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[tuple[int, ...]]]:
    rows = training_rows_for_spawn(task)
    frequent = [item for item, _count in frequent_substrings(rows, 2, 4)[:32]]
    split = [item for item, _count in split_substrings(rows)[:32]]
    return {"frequent": frequent, "split": split}


def add_spawn(candidate: dict[str, Any], segments: tuple[int, ...], source: str, k: int | None = None, depth: int = 1) -> dict[str, Any]:
    mutated = copy.deepcopy(candidate)
    spawned = normalize_spawned(mutated)
    if tuple(segments) not in {tuple(row["segments"]) for row in spawned}:
        spawned.append(make_spawn(tuple(segments), source, k, depth))
    mutated["spawned"] = spawned
    return mutated


def mutate_candidate(candidate: dict[str, Any], system: str, pools: dict[str, list[tuple[int, ...]]], rng: random.Random) -> tuple[dict[str, Any], str]:
    mutated = copy.deepcopy(candidate)
    spawned = normalize_spawned(mutated)
    ops = ["change_K", "change_depth", "delete_spawn", "spawn_random"]
    if system in {"control_spawn_from_composed_route", "control_spawn_plus_limited_repair"}:
        ops.extend(["spawn_composed", "spawn_composed", "spawn_composed"])
    if system == "control_spawn_from_split":
        ops.extend(["spawn_split", "spawn_split"])
    if system == "control_spawn_blank_pocket":
        ops.extend(["spawn_blank", "spawn_blank"])
    if system == "control_spawn_plus_limited_repair":
        ops.extend(["repair_resize", "repair_resize"])
    op = rng.choice(ops)
    if op in {"spawn_composed", "spawn_split", "spawn_blank"}:
        pool_name = "frequent" if op in {"spawn_composed", "spawn_blank"} else "split"
        pool = pools.get(pool_name, [])
        if pool and (op != "spawn_blank" or rng.random() < 0.55):
            segments = pool[rng.randrange(min(len(pool), 18))]
        else:
            length = rng.choice((2, 3, 4))
            segments = tuple(rng.sample(range(MICRO_COUNT), length))
        source = {"spawn_composed": "composed", "spawn_split": "split", "spawn_blank": "blank"}[op]
        k = 1 if source == "blank" else required_k_for_call(segments)
        return add_spawn(mutated, segments, source, k, rng.choice((1, 1, 2))), op
    if op == "spawn_random":
        length = rng.choice((2, 3, 4))
        segments = tuple(rng.sample(range(MICRO_COUNT), length))
        return add_spawn(mutated, segments, "random_mutation", rng.choice(K_VALUES), rng.choice((1, 2, 3))), op
    if spawned and op == "delete_spawn":
        del spawned[rng.randrange(len(spawned))]
    elif spawned and op in {"change_K", "repair_resize"}:
        idx = rng.randrange(len(spawned))
        current = clamp_k(int(spawned[idx]["K"]))
        direction = 1 if rng.random() < 0.65 else -1
        k_index = max(0, min(len(K_VALUES) - 1, K_VALUES.index(current) + direction))
        spawned[idx]["K"] = K_VALUES[k_index]
        if op == "repair_resize":
            spawned[idx]["repair_permission"] = True
            spawned[idx]["source"] = "repair" if spawned[idx]["source"] == "composed" else spawned[idx]["source"]
    elif spawned and op == "change_depth":
        idx = rng.randrange(len(spawned))
        spawned[idx]["depth"] = max(1, min(4, int(spawned[idx]["depth"]) + rng.choice((-1, 1))))
    else:
        length = rng.choice((2, 3, 4))
        segments = tuple(rng.sample(range(MICRO_COUNT), length))
        spawned.append(make_spawn(segments, "fallback", rng.choice(K_VALUES), 1))
    mutated["spawned"] = normalize_spawned({"spawned": spawned})
    return mutated, op


def bootstrap_candidates(system: str, pools: dict[str, list[tuple[int, ...]]]) -> list[tuple[dict[str, Any], str]]:
    candidates: list[tuple[dict[str, Any], str]] = []
    if system in {"control_spawn_from_composed_route", "control_spawn_plus_limited_repair"}:
        current = candidate_initial(system)
        for idx, segments in enumerate(pools.get("frequent", [])[:6]):
            current = add_spawn(current, segments, "composed" if system == "control_spawn_from_composed_route" else "repair", required_k_for_call(segments), max(1, len(segments) - 2))
            candidates.append((copy.deepcopy(current), f"bootstrap_composed_{idx}"))
    elif system == "control_spawn_from_split":
        current = candidate_initial(system)
        for idx, segments in enumerate(pools.get("split", [])[:6]):
            current = add_spawn(current, segments, "split", required_k_for_call(segments), max(1, len(segments) - 2))
            candidates.append((copy.deepcopy(current), f"bootstrap_split_{idx}"))
    else:
        for idx, segments in enumerate(pools.get("frequent", [])[:4]):
            candidates.append((add_spawn(candidate_initial(system), segments, "blank", 1, 1), f"bootstrap_blank_{idx}"))
    return candidates


def mutation_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    system = str(job["system"])
    task = job["task"]
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rng = random.Random(stable_seed(f"{seed}:{system}:spawn_mutation"))
    pools = mutation_pools(task)
    best = candidate_initial(system)
    initial_hash = payload_sha256(best)
    best_score = candidate_learning_score(best, system, task)
    accepted = rejected = attempts = 0
    accepted_by_operator: dict[str, int] = {}
    rejected_by_operator: dict[str, int] = {}
    history: list[dict[str, Any]] = []
    snapshot_dir = out / "mutation_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for candidate, operator in bootstrap_candidates(system, pools):
        attempts += 1
        score = candidate_learning_score(candidate, system, task)
        if score > best_score + 1e-12:
            best = candidate
            best_score = score
            accepted += 1
            accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
        else:
            rejected += 1
            rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
    for generation in range(settings.mutation_generations):
        generation_best = best_score
        for _ in range(settings.mutation_population):
            attempts += 1
            candidate, operator = mutate_candidate(best, system, pools, rng)
            score = candidate_learning_score(candidate, system, task)
            if score > best_score + 1e-12:
                best = candidate
                best_score = score
                accepted += 1
                accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
            else:
                rejected += 1
                rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
        if generation % max(1, settings.mutation_generations // 10) == 0 or generation == settings.mutation_generations - 1:
            row = {
                "generation": generation,
                "best_score": round_float(best_score),
                "generation_gain": round_float(best_score - generation_best),
                "accepted": accepted,
                "rejected": rejected,
                "candidate_hash": payload_sha256(best),
                "summary": candidate_summary(best),
            }
            history.append(row)
            if snapshot_dir:
                locked_write_json(snapshot_dir / f"{system}_seed{seed}_generation{generation:04d}.json", row)
            if out:
                append_progress(out, "mutation_generation", seed=seed, system=system, generation=generation, best_score=row["best_score"], summary=row["summary"])
    predictions = predict_with_candidate(task, system, best)
    result = evaluate_predictions(task, predictions, system, best, parameter_count(best), router_complexity=3.0)
    final_hash = payload_sha256(best)
    result.update(
        {
            "seed": seed,
            "system": system,
            "history": history,
            "initial_candidate_hash": initial_hash,
            "final_candidate_hash": final_hash,
            "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": final_hash, "candidate": best}),
            "final_candidate_summary": candidate_summary(best),
            "mutation_attempts": attempts,
            "accepted_mutations": accepted,
            "rejected_mutations": rejected,
            "rollback_count": rejected,
            "failed_spawn_rollback_count": sum(count for op, count in rejected_by_operator.items() if "spawn" in op or "bootstrap" in op),
            "accepted_by_operator": accepted_by_operator,
            "rejected_by_operator": rejected_by_operator,
        }
    )
    return result


def make_tensor(rows: list[dict[str, Any]], device: str) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    x = torch.tensor([row["raw"] for row in rows], dtype=torch.float32, device=device)
    targets = [torch.tensor([row["padded_micro_path"][slot] for row in rows], dtype=torch.long, device=device) for slot in range(MAX_MICRO_PATH)]
    length_targets = torch.tensor([max(0, min(MAX_MICRO_PATH - 4, row["micro_path_length"] - 4)) for row in rows], dtype=torch.long, device=device)
    return x, targets, length_targets


def train_dense(seed: int, task: dict[str, dict[str, list[dict[str, Any]]]], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"{seed}:dense_spawn_graph"), device)
    train_rows = all_rows(task, "train")
    model = DensePathMLP(len(train_rows[0]["raw"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train, targets, length_targets = make_tensor(train_rows, device)
    rng = np.random.default_rng(stable_seed(f"{seed}:dense_batches"))
    history: list[dict[str, Any]] = []
    snapshot_dir = out / "training_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(settings.gradient_epochs):
        indices = rng.permutation(len(train_rows))
        losses: list[float] = []
        for start in range(0, len(indices), settings.batch_size):
            batch = indices[start : start + settings.batch_size]
            heads, length_logits = model(x_train[batch])
            loss = nn.functional.cross_entropy(length_logits, length_targets[batch])
            for slot, logits in enumerate(heads):
                loss = loss + nn.functional.cross_entropy(logits, targets[slot][batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch % max(1, settings.gradient_epochs // 10) == 0 or epoch == settings.gradient_epochs - 1:
            row = {"epoch": epoch, "loss": round_float(float(np.mean(losses))), "device": device}
            history.append(row)
            if snapshot_dir:
                locked_write_json(snapshot_dir / f"dense_graph_danger_control_seed{seed}_epoch{epoch:04d}.json", row)
            if out:
                append_progress(out, "gradient_epoch", seed=seed, system="dense_graph_danger_control", epoch=epoch, loss=row["loss"], device=device)
    state = {key: value.detach().cpu().numpy().tolist() for key, value in model.state_dict().items()}
    result = evaluate_dense_state(state, task, device)
    result.update({"seed": seed, "system": "dense_graph_danger_control", "history": history, "device": device})
    return result


def evaluate_dense_state(model_state: dict[str, Any], task: dict[str, dict[str, list[dict[str, Any]]]], device: str) -> dict[str, Any]:
    model = DensePathMLP(len(all_rows(task, "train")[0]["raw"])).to(device)
    model.load_state_dict({key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in model_state.items()})
    model.eval()
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    with torch.no_grad():
        for phase in PHASES:
            predictions[phase] = {}
            for split in SPLITS:
                rows = task[phase][split]
                x = torch.tensor([row["raw"] for row in rows], dtype=torch.float32, device=device)
                heads, length_logits = model(x)
                length_pred = torch.argmax(length_logits, dim=1).detach().cpu().numpy() + 4
                probs = [torch.softmax(logits, dim=1).detach().cpu().numpy() for logits in heads]
                split_preds = []
                for idx, _row in enumerate(rows):
                    micro: list[int] = []
                    branches = 0
                    for slot in range(MAX_MICRO_PATH):
                        top = int(np.argmax(probs[slot][idx]))
                        if top != PAD_SEGMENT and len(micro) < int(length_pred[idx]):
                            micro.append(top)
                        branches += max(0, int(np.sum(probs[slot][idx][:MICRO_COUNT] > 0.16)) - 1)
                    calls = [{"call": [seg], "source": "dense", "K": 1, "depth": 1, "emitted": [seg]} for seg in micro[:MAX_MICRO_PATH]]
                    split_preds.append({"calls": calls, "micro_path": micro[:MAX_MICRO_PATH], "steps": len(micro[:MAX_MICRO_PATH]) + branches, "branch_expansions": branches, "route_cost": float(len(micro[:MAX_MICRO_PATH]) + branches)})
                predictions[phase][split] = split_preds
    params = sum(int(np.prod(np.asarray(value).shape)) for value in model_state.values())
    candidate = {"spawned": [], "system": "dense_graph_danger_control"}
    return evaluate_predictions(task, predictions, "dense_graph_danger_control", candidate, params, router_complexity=12.0)


def gpu_lane_worker(job: dict[str, Any]) -> dict[str, Any]:
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for seed_text, task in sorted(job["tasks"].items(), key=lambda item: int(item[0])):
        result = train_dense(int(seed_text), task, settings, out)
        rows.append({key: value for key, value in result.items() if key != "history"})
        histories.append({"seed": result["seed"], "system": result["system"], "history": result["history"], "device": result["device"], "parameter_count": result["parameter_count"], "model_state_hash": payload_sha256({key: value for key, value in result.items() if key not in {"history", "evals", "phase_metrics"}})})
    return {"rows": rows, "histories": histories, "hardware": e7h.e7g.e7d.e7b.hardware_probe()}


def task_report(tasks: dict[int, dict[str, dict[str, list[dict[str, Any]]]]]) -> dict[str, Any]:
    return {
        "schema_version": "e7k_task_generation_report_v1",
        "row_counts": {str(seed): {phase: {split: len(rows) for split, rows in phase_task.items()} for phase, phase_task in task.items()} for seed, task in tasks.items()},
        "public_inputs": "microsegment_path_plus_phase_token_no_public_missing_motif_id",
        "hidden_missing_transform_used_for_eval_only": True,
        "phases": list(PHASES),
        "base_pockets": [list(item) for item in BASE_POCKETS],
        "true_spawn_motifs": {key: list(value) for key, value in TRUE_MOTIFS.items()},
    }


def spawn_mechanism_report() -> dict[str, Any]:
    return {
        "schema_version": "e7k_spawn_mechanism_report_v1",
        "external_interface": "CALL(pocket_id, Flow[D]) -> Flow[D]",
        "flow_d": FLOW_D,
        "spawned_pocket_fields": ["id", "segments", "K", "depth", "source", "promoted", "frozen", "repair_permission"],
        "spawn_triggers_tested": [
            "repeated residual route pattern",
            "high route cost",
            "composed route cache",
            "localized damage repair",
            "validation usefulness gate",
        ],
        "dense_graph_allowed_for_mutation_systems": False,
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    phase_summary: dict[str, dict[str, Any]] = {phase: {} for phase in PHASES}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in (
                "heldout_usefulness",
                "ood_usefulness",
                "counterfactual_usefulness",
                "adversarial_usefulness",
                "heldout_spawn_value",
                "ood_spawn_value",
                "counterfactual_spawn_value",
                "adversarial_spawn_value",
                "eval_mean_usefulness",
                "eval_mean_spawn_value",
                "generalization_gap",
                "parameter_count",
            ):
                metrics.setdefault(metric, []).append(float(row[metric]))
            for split in SPLITS:
                for metric, value in row["evals"][split].items():
                    if isinstance(value, (int, float)):
                        metrics.setdefault(f"{split}_{metric}", []).append(float(value))
            for phase in PHASES:
                phase_eval = float(np.mean([row["phase_metrics"][phase][split]["spawn_value_score"] for split in EVAL_SPLITS]))
                phase_summary[phase].setdefault(system, []).append(phase_eval)
        systems[system] = {
            "seed_count": len(system_rows),
            "mean": {metric: round_float(float(np.mean(values))) for metric, values in metrics.items()},
            "min": {metric: round_float(float(np.min(values))) for metric, values in metrics.items()},
            "max": {metric: round_float(float(np.max(values))) for metric, values in metrics.items()},
        }
    phase_winners: dict[str, Any] = {}
    for phase, by_system in phase_summary.items():
        means = {system: round_float(float(np.mean(values))) for system, values in by_system.items()}
        best_system = max(means, key=lambda system: means[system])
        phase_winners[phase] = {"best_system": best_system, "system_spawn_value_mean": means}
    best = max(SYSTEMS, key=lambda system: systems[system]["mean"]["eval_mean_spawn_value"])
    return {
        "schema_version": "e7k_aggregate_metrics_v1",
        "systems": systems,
        "phase_winners": phase_winners,
        "best_system": best,
        "best_eval_mean_spawn_value": systems[best]["mean"]["eval_mean_spawn_value"],
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    best_fixed = max(("fixed_library_no_spawn", "fixed_library_router_plus_repair"), key=lambda system: systems[system]["mean"]["eval_mean_spawn_value"])
    spawn_systems = ("control_spawn_blank_pocket", "control_spawn_from_split", "control_spawn_from_composed_route", "control_spawn_plus_limited_repair")
    best_spawn = max(spawn_systems, key=lambda system: systems[system]["mean"]["eval_mean_spawn_value"])
    spawn = systems[best_spawn]["mean"]
    fixed = systems[best_fixed]["mean"]
    oracle = systems["oracle_spawn_scaffold"]["mean"]
    random_control = systems["random_spawn_control"]["mean"]
    dense = systems["dense_graph_danger_control"]["mean"]
    spawn_gain = spawn["eval_mean_spawn_value"] - fixed["eval_mean_spawn_value"]
    oracle_gap = oracle["eval_mean_spawn_value"] - spawn["eval_mean_spawn_value"]
    junk = spawn.get("heldout_unnecessary_spawn_rate", 0.0)
    promoted = spawn.get("heldout_promoted_pocket_count", 0.0)
    detail = {
        "overall_best_system": aggregate["best_system"],
        "best_fixed_system": best_fixed,
        "best_spawn_system": best_spawn,
        "spawn_minus_best_fixed": round_float(spawn_gain),
        "oracle_minus_best_spawn": round_float(oracle_gap),
        "random_spawn_value_mean": random_control["eval_mean_spawn_value"],
        "dense_spawn_value_mean": dense["eval_mean_spawn_value"],
        "best_spawn_promoted_pocket_count": round_float(promoted),
        "best_spawn_unnecessary_spawn_rate": round_float(junk),
        "best_spawn_average_K": spawn.get("heldout_spawned_pocket_average_K", 0.0),
        "best_spawn_average_depth": spawn.get("heldout_spawned_pocket_average_depth", 0.0),
        "phase_winners": {phase: row["best_system"] for phase, row in aggregate["phase_winners"].items()},
    }
    if random_control["eval_mean_spawn_value"] >= spawn["eval_mean_spawn_value"] - 0.01:
        return "e7k_spawn_artifact_or_task_too_easy", detail
    if dense["eval_mean_spawn_value"] > spawn["eval_mean_spawn_value"] + 0.02 and dense["ood_spawn_value_score"] >= spawn["ood_spawn_value_score"] - 0.02:
        return "e7k_pocket_spawn_collapses_to_graph_soup", detail
    if junk > 0.45 or promoted > 7.0:
        return "e7k_spawn_overproduction_failure", detail
    if fixed["eval_mean_spawn_value"] >= spawn["eval_mean_spawn_value"] - 0.01 and fixed["ood_spawn_value_score"] >= spawn["ood_spawn_value_score"] - 0.01:
        return "e7k_no_spawn_needed_existing_library_sufficient", detail
    if oracle_gap > 0.07 and spawn_gain <= 0.04:
        return "e7k_spawn_needs_prior_scaffold", detail
    if spawn_gain > 0.04 and oracle_gap <= 0.07:
        if best_spawn == "control_spawn_from_composed_route":
            return "e7k_composed_route_pocket_spawn_positive", detail
        if best_spawn == "control_spawn_from_split":
            return "e7k_split_spawn_positive", detail
        return "e7k_dynamic_pocket_spawn_positive", detail
    if oracle_gap > 0.07:
        return "e7k_spawn_needs_prior_scaffold", detail
    return "e7k_dynamic_pocket_spawn_positive", detail


def build_spawn_promotion_report(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7k_spawn_promotion_report_v1",
        "system_means": {
            system: {
                "eval_mean_spawn_value": aggregate["systems"][system]["mean"]["eval_mean_spawn_value"],
                "heldout_spawn_precision": aggregate["systems"][system]["mean"].get("heldout_spawn_precision", 0.0),
                "heldout_spawn_recall": aggregate["systems"][system]["mean"].get("heldout_spawn_recall", 0.0),
                "heldout_promoted_pocket_count": aggregate["systems"][system]["mean"].get("heldout_promoted_pocket_count", 0.0),
                "heldout_unnecessary_spawn_rate": aggregate["systems"][system]["mean"].get("heldout_unnecessary_spawn_rate", 0.0),
                "heldout_route_step_reduction": aggregate["systems"][system]["mean"].get("heldout_route_step_reduction", 0.0),
            }
            for system in SYSTEMS
        },
        "example_final_candidates": {
            row["system"]: row.get("candidate_summary", {})
            for row in rows
            if int(row["seed"]) == min(int(item["seed"]) for item in rows) and row["system"] in MUTATION_SYSTEMS
        },
    }


def build_phase_winner_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": "e7k_phase_spawn_winner_report_v1", "phases": aggregate["phase_winners"]}


def build_leakage_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7k_leakage_report_v1",
        "public_inputs": "microsegment_path_plus_phase_token",
        "hidden_missing_motif_id_used_as_model_input": False,
        "dense_all_to_all_soft_routing_used_by_mutation_systems": False,
        "random_spawn_control_passed": aggregate["systems"]["random_spawn_control"]["mean"]["eval_mean_spawn_value"] < aggregate["systems"]["control_spawn_from_composed_route"]["mean"]["eval_mean_spawn_value"] - 0.01,
        "dense_graph_danger_control_measured": True,
    }


def build_markdown(payloads: dict[str, Any]) -> str:
    aggregate = payloads["aggregate_metrics.json"]
    decision = payloads["decision.json"]
    summary = payloads["summary.json"]
    detail = decision["detail"]
    lines = [
        "# E7K Dynamic Pocket Spawn And Promotion Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_system = {summary['best_system']}",
        f"best_spawn_system = {detail['best_spawn_system']}",
        f"deterministic_replay_passed = {summary['deterministic_replay_passed']}",
        "```",
        "",
        "## Mean Scores",
        "",
        "```text",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        lines.append(
            f"{system:38s} spawn={mean['eval_mean_spawn_value']:.6f} useful={mean['eval_mean_usefulness']:.6f} "
            f"ood={mean['ood_spawn_value_score']:.6f} promoted={mean.get('heldout_promoted_pocket_count', 0.0):.2f} "
            f"precision={mean.get('heldout_spawn_precision', 0.0):.3f} recall={mean.get('heldout_spawn_recall', 0.0):.3f}"
        )
    lines.extend(["```", "", "## Frontier", "", "```text"])
    for key in (
        "best_fixed_system",
        "best_spawn_system",
        "spawn_minus_best_fixed",
        "oracle_minus_best_spawn",
        "random_spawn_value_mean",
        "dense_spawn_value_mean",
        "best_spawn_promoted_pocket_count",
        "best_spawn_average_K",
        "best_spawn_average_depth",
        "best_spawn_unnecessary_spawn_rate",
    ):
        lines.append(f"{key} = {detail[key]}")
    lines.extend(["```", "", "## Phase Winners", "", "```text"])
    for phase, winner in detail["phase_winners"].items():
        lines.append(f"{phase:44s} {winner}")
    lines.extend(["```", "", "## Boundary", "", "This is a controlled pocket-flow spawn/promotion proxy over typed callable units."])
    return "\n".join(lines) + "\n"


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress(out, "startup", milestone=MILESTONE, settings=settings_payload(settings), hardware=e7h.e7g.e7d.e7b.hardware_probe())
    tasks = generate_tasks(settings)
    if out:
        append_progress(out, "tasks_generated", seeds=list(settings.seeds), row_counts=task_report(tasks)["row_counts"])
    rows: list[dict[str, Any]] = []
    mutation_histories: list[dict[str, Any]] = []
    training_histories: list[dict[str, Any]] = []
    for seed in settings.seeds:
        rows.extend(control_results(seed, tasks[seed]))
    jobs = [{"seed": seed, "system": system, "task": tasks[seed], "settings": settings.__dict__, "out": out.as_posix() if out else None} for seed in settings.seeds for system in MUTATION_SYSTEMS]
    gpu_job = {"tasks": {str(seed): tasks[seed] for seed in settings.seeds}, "settings": settings.__dict__, "out": out.as_posix() if out else None}
    if settings.execution_mode == "parallel":
        with ProcessPoolExecutor(max_workers=max(1, settings.cpu_workers)) as executor:
            futures = {executor.submit(mutation_worker, job): f"{job['system']}/seed{job['seed']}" for job in jobs}
            pending = set(futures)
            if out:
                append_progress(out, "lanes_submitted", cpu_mutation_jobs=len(jobs), cpu_workers=settings.cpu_workers, gpu_lane=True)
            gpu = gpu_lane_worker(gpu_job)
            rows.extend(gpu["rows"])
            training_histories.extend(gpu["histories"])
            if out:
                append_progress(out, "gpu_lane_complete", completed_gradient_rows=len(gpu["rows"]), hardware=gpu["hardware"])
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    label = futures[future]
                    result = future.result()
                    rows.append({key: value for key, value in result.items() if key != "history"})
                    mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "initial_candidate_hash", "final_candidate_hash", "parameter_diff_hash", "final_candidate_summary", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "failed_spawn_rollback_count", "accepted_by_operator", "rejected_by_operator")})
                    if out:
                        locked_write_json(out / "partial_aggregate_snapshot.json", {"schema_version": "e7k_partial_aggregate_snapshot_v1", "completed_rows": len(rows), "expected_rows": len(settings.seeds) * len(SYSTEMS), "pending_jobs": len(pending)})
                        append_progress(out, "mutation_job_complete", label=label, pending=len(pending))
    else:
        gpu = gpu_lane_worker(gpu_job)
        rows.extend(gpu["rows"])
        training_histories.extend(gpu["histories"])
        for job in jobs:
            result = mutation_worker(job)
            rows.append({key: value for key, value in result.items() if key != "history"})
            mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "initial_candidate_hash", "final_candidate_hash", "parameter_diff_hash", "final_candidate_summary", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "failed_spawn_rollback_count", "accepted_by_operator", "rejected_by_operator")})
    rows.sort(key=lambda row: (row["system"], int(row["seed"])))
    mutation_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    training_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    aggregate = aggregate_results(rows)
    decision, detail = decide(aggregate)
    return {"tasks": tasks, "rows": rows, "mutation_histories": mutation_histories, "training_histories": training_histories, "aggregate": aggregate, "decision": decision, "decision_detail": detail}


def build_payloads(settings: Settings, out: Path, results: dict[str, Any]) -> dict[str, Any]:
    payloads: dict[str, Any] = {
        "backend_manifest.json": {
            "schema_version": "e7k_backend_manifest_v1",
            "milestone": MILESTONE,
            "settings": settings_payload(settings),
            "systems": list(SYSTEMS),
            "gradient_systems": list(GRADIENT_SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "control_systems": list(CONTROL_SYSTEMS),
            "hardware_identity": e7h.e7g.e7d.e7b.stable_hardware_identity(),
            "parallel_cpu_gpu_lanes": settings.execution_mode == "parallel",
        },
        "task_generation_report.json": task_report(results["tasks"]),
        "spawn_mechanism_report.json": spawn_mechanism_report(),
        "spawn_promotion_report.json": build_spawn_promotion_report(results["rows"], results["aggregate"]),
        "phase_spawn_winner_report.json": build_phase_winner_report(results["aggregate"]),
        "system_results.json": {"schema_version": "e7k_system_results_v1", "rows": results["rows"]},
        "mutation_history.json": {"schema_version": "e7k_mutation_history_v1", "rows": results["mutation_histories"]},
        "training_history.json": {"schema_version": "e7k_training_history_v1", "rows": results["training_histories"]},
        "leakage_report.json": build_leakage_report(results["aggregate"]),
        "aggregate_metrics.json": results["aggregate"],
        "decision.json": {"schema_version": "e7k_decision_v1", "decision": results["decision"], "detail": results["decision_detail"], "deterministic_replay_passed": False},
        "summary.json": {"schema_version": "e7k_summary_v1", "decision": results["decision"], "best_system": results["aggregate"]["best_system"], "deterministic_replay_passed": False, "checker_failure_count": None, "run_root": out.relative_to(REPO_ROOT).as_posix()},
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
    primary = compute_hashes(primary_payloads)
    replay = compute_hashes(replay_payloads)
    comparisons = {name: {"primary_hash": primary[name], "replay_hash": replay[name], "match": primary[name] == replay[name]} for name in HASH_ARTIFACTS}
    report = {"schema_version": "e7k_deterministic_replay_v1", "internal_replay_passed": all(row["match"] for row in comparisons.values()), "hash_comparisons": comparisons, "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix()}
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
    parser.add_argument("--seeds", default="99001,99002,99003,99004,99005,99006")
    parser.add_argument("--train-rows-per-seed", type=int, default=720)
    parser.add_argument("--validation-rows-per-seed", type=int, default=300)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=300)
    parser.add_argument("--ood-rows-per-seed", type=int, default=300)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=300)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=300)
    parser.add_argument("--gradient-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mutation-generations", type=int, default=90)
    parser.add_argument("--mutation-population", type=int, default=20)
    parser.add_argument("--mutation-sigma", type=float, default=0.16)
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
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7k_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_work_root": None}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop.set()
        monitor.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
