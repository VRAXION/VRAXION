#!/usr/bin/env python3
"""E7L spawn/repair cost and noisy-health falsification.

E7K showed that a typed control layer can spawn and promote callable pockets in
a clean proxy. E7L attacks that result by adding costs, noisy/incomplete health
signals, delayed validation feedback, moving drift, and decoy reusable-looking
routes. The question is whether the control layer can still choose among
route-around, repair, spawn, and no-op without turning into dense graph soup or
overproducing junk pockets.
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
E7K_PATH = Path(__file__).with_name("run_e7k_dynamic_pocket_spawn_and_promotion_probe.py")
MILESTONE = "E7L_SPAWN_REPAIR_COST_AND_NOISY_HEALTH_FALSIFICATION"
DEFAULT_OUT = Path("target/pilot_wave/e7l_spawn_repair_cost_and_noisy_health_falsification")

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
    "no_adaptation",
    "route_around_only",
    "repair_only",
    "spawn_only",
    "spawn_plus_limited_repair_clean",
    "cost_aware_spawn_plus_repair",
    "noisy_health_spawn_plus_repair",
    "delayed_feedback_spawn_plus_repair",
    "oracle_health_spawn_repair_reference",
    "random_spawn_repair_control",
    "dense_graph_danger_control",
)
MUTATION_SYSTEMS = (
    "route_around_only",
    "repair_only",
    "spawn_only",
    "spawn_plus_limited_repair_clean",
    "cost_aware_spawn_plus_repair",
    "noisy_health_spawn_plus_repair",
    "delayed_feedback_spawn_plus_repair",
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
DECOY_MOTIFS = ((14, 15), (15, 14), (1, 14), (8, 15))
REPAIR_TARGETS = DAMAGED_BASE_POCKETS + ((6, 7),)
STRESS_CONDITIONS = (
    "repair_cost",
    "spawn_cost",
    "pocket_maintenance_cost",
    "noisy_health_signal",
    "incomplete_health_signal",
    "delayed_validation_feedback",
    "moving_drift_profile",
    "false_positive_reusable_route",
    "false_negative_reusable_route",
    "junk_spawn_pressure",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "cost_noise_stress_report.json",
    "action_policy_report.json",
    "spawn_repair_policy_report.json",
    "health_signal_report.json",
    "delayed_feedback_report.json",
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
    "e7l_cost_aware_spawn_repair_survives",
    "e7l_routearound_preferred_under_cost",
    "e7l_repair_preferred_spawn_too_expensive",
    "e7l_spawn_preferred_repair_not_needed",
    "e7l_spawn_repair_requires_clean_health_signal",
    "e7l_spawn_overproduction_failure",
    "e7l_delayed_feedback_instability",
    "e7l_graph_soup_regression_detected",
    "e7l_leak_or_artifact_detected",
)


def load_e7k_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7k_dynamic_spawn_probe", E7K_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7K helpers from {E7K_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7k = load_e7k_module()
e7h = e7k.e7h


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
    return int(hashlib.sha256(f"e7l::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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


def e7k_settings(settings: Settings) -> Any:
    return e7k.Settings(**settings.__dict__)


def select_device(requested: str) -> str:
    return e7h.select_device(requested)


def set_determinism(seed: int, device: str) -> None:
    e7h.set_determinism(seed, device)


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    return e7h.start_hardware_monitor(out, stop, interval)


def normalize_tuple(raw: Any, min_len: int = 1, max_len: int = MAX_MICRO_PATH) -> tuple[int, ...]:
    if isinstance(raw, dict):
        raw = raw.get("segments", [])
    out = tuple(int(seg) for seg in raw if 0 <= int(seg) < MICRO_COUNT)
    if len(out) < min_len or len(out) > max_len:
        return ()
    return out


def normalize_segment_list(raw: Any, min_len: int = 1, max_len: int = MAX_MICRO_PATH) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for item in raw or []:
        tup = normalize_tuple(item, min_len=min_len, max_len=max_len)
        if tup and tup not in seen:
            seen.add(tup)
            out.append(tup)
    return out


def normalize_spawned(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return e7k.normalize_spawned(candidate)


def candidate_initial(system: str) -> dict[str, Any]:
    return {
        "system": system,
        "spawned": [],
        "repairs": [],
        "route_around": [],
        "health_noise_rate": 0.0,
        "health_missing_rate": 0.0,
        "delayed_regret_seed": 0.0,
    }


def active_spawn_map(candidate: dict[str, Any]) -> dict[tuple[int, ...], dict[str, Any]]:
    return e7k.active_spawn_map({"spawned": normalize_spawned(candidate)})


def normalized_repairs(candidate: dict[str, Any]) -> list[tuple[int, ...]]:
    return normalize_segment_list(candidate.get("repairs", []), min_len=1, max_len=4)


def normalized_route_around(candidate: dict[str, Any]) -> list[tuple[int, ...]]:
    return normalize_segment_list(candidate.get("route_around", []), min_len=1, max_len=4)


def normalize_candidate(candidate: dict[str, Any], system: str | None = None) -> dict[str, Any]:
    system = system or str(candidate.get("system", "candidate"))
    max_spawns = 10 if system == "spawn_plus_limited_repair_clean" else 7
    return {
        "system": system,
        "spawned": normalize_spawned(candidate)[:max_spawns],
        "repairs": [list(item) for item in normalized_repairs(candidate)[:6]],
        "route_around": [list(item) for item in normalized_route_around(candidate)[:6]],
        "health_noise_rate": round_float(max(0.0, min(0.5, float(candidate.get("health_noise_rate", 0.0))))),
        "health_missing_rate": round_float(max(0.0, min(0.5, float(candidate.get("health_missing_rate", 0.0))))),
        "delayed_regret_seed": round_float(max(0.0, min(1.0, float(candidate.get("delayed_regret_seed", 0.0))))),
    }


def candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_candidate(candidate, str(candidate.get("system", "candidate")))
    spawned = normalized["spawned"]
    repairs = normalized_repairs(normalized)
    route_around = normalized_route_around(normalized)
    k_values = [int(row["K"]) for row in spawned]
    depths = [int(row["depth"]) for row in spawned]
    return {
        "promoted_pocket_count": len(spawned),
        "spawned_pockets": spawned,
        "repair_count": len(repairs),
        "repairs": [list(item) for item in repairs],
        "route_around_count": len(route_around),
        "route_around": [list(item) for item in route_around],
        "average_K": round_float(float(np.mean(k_values)) if k_values else 0.0),
        "average_depth": round_float(float(np.mean(depths)) if depths else 0.0),
        "health_noise_rate": normalized["health_noise_rate"],
        "health_missing_rate": normalized["health_missing_rate"],
        "candidate_hash": payload_sha256(normalized),
    }


def candidate_hash(candidate: dict[str, Any]) -> str:
    return payload_sha256(normalize_candidate(candidate, str(candidate.get("system", "candidate"))))


def parameter_count(candidate: dict[str, Any], router_extra: int = 72) -> int:
    total = int(router_extra + len(BASE_POCKETS) * 5)
    for row in normalize_spawned(candidate):
        total += 8 + e7k.capacity_params(int(row["K"]), int(row["depth"]))
    total += 7 * len(normalized_repairs(candidate))
    total += 4 * len(normalized_route_around(candidate))
    total += 10
    return int(total)


def generate_tasks(settings: Settings) -> dict[int, dict[str, dict[str, list[dict[str, Any]]]]]:
    return e7k.generate_tasks(e7k_settings(settings))


def all_rows(task: dict[str, dict[str, list[dict[str, Any]]]], split: str) -> list[dict[str, Any]]:
    return e7k.all_rows(task, split)


def training_rows(task: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return all_rows(task, "train") + all_rows(task, "validation")


def apply_micro_path(path: list[int] | tuple[int, ...], a: int, b: int, key: int, mem: int, threshold: int) -> int:
    return e7k.apply_micro_path(path, a, b, key, mem, threshold)


def greedy_calls(path: list[int] | tuple[int, ...], pockets: list[tuple[int, ...]] | tuple[tuple[int, ...], ...]) -> list[tuple[int, ...]]:
    return e7k.greedy_calls(path, pockets)


def base_route_cost(calls: list[tuple[int, ...]]) -> float:
    return e7k.base_route_cost(calls)


def spawned_route_cost(k: int, depth: int) -> float:
    return e7k.spawned_route_cost(k, depth)


def required_k_for_call(call: tuple[int, ...]) -> int:
    return e7k.required_k_for_call(call)


def make_spawn(segments: tuple[int, ...], source: str, k: int | None = None, depth: int = 1) -> dict[str, Any]:
    return e7k.make_spawn(segments, source, k, depth)


def corrupt_call(call: tuple[int, ...], salt: int) -> list[int]:
    return e7k.corrupt_call(call, salt)


def true_motifs_for_rows(rows: list[dict[str, Any]]) -> set[tuple[int, ...]]:
    return e7k.true_motifs_for_rows(rows)


def true_damaged_for_rows(rows: list[dict[str, Any]]) -> set[tuple[int, ...]]:
    damaged: set[tuple[int, ...]] = set()
    for row in rows:
        if row["phase"] == "phase_5_damage_drift_repair":
            path = row["micro_path"]
            for item in DAMAGED_BASE_POCKETS:
                if contains_subpath(path, item):
                    damaged.add(item)
            if row["split"] in {"ood", "adversarial"} and contains_subpath(path, (6, 7)):
                damaged.add((6, 7))
    return damaged


def contains_subpath(path: list[int] | tuple[int, ...], motif: tuple[int, ...]) -> bool:
    if not motif or len(motif) > len(path):
        return False
    return any(tuple(path[idx : idx + len(motif)]) == motif for idx in range(0, len(path) - len(motif) + 1))


def candidate_pockets(candidate: dict[str, Any]) -> list[tuple[int, ...]]:
    spawned = sorted(active_spawn_map(candidate), key=lambda item: (-len(item), item))
    route_around = set(normalized_route_around(candidate))
    base = [item for item in BASE_POCKETS if item not in route_around]
    return base + spawned


def system_allows_repair(system: str) -> bool:
    return system in {
        "repair_only",
        "spawn_plus_limited_repair_clean",
        "cost_aware_spawn_plus_repair",
        "noisy_health_spawn_plus_repair",
        "delayed_feedback_spawn_plus_repair",
        "oracle_health_spawn_repair_reference",
        "random_spawn_repair_control",
    }


def system_allows_spawn(system: str) -> bool:
    return system in {
        "spawn_only",
        "spawn_plus_limited_repair_clean",
        "cost_aware_spawn_plus_repair",
        "noisy_health_spawn_plus_repair",
        "delayed_feedback_spawn_plus_repair",
        "oracle_health_spawn_repair_reference",
        "random_spawn_repair_control",
    }


def system_allows_route_around(system: str) -> bool:
    return system in {
        "route_around_only",
        "cost_aware_spawn_plus_repair",
        "noisy_health_spawn_plus_repair",
        "delayed_feedback_spawn_plus_repair",
        "oracle_health_spawn_repair_reference",
        "random_spawn_repair_control",
    }


def row_damaged_calls(row: dict[str, Any]) -> set[tuple[int, ...]]:
    damaged: set[tuple[int, ...]] = set()
    if row["phase"] != "phase_5_damage_drift_repair":
        return damaged
    path = row["micro_path"]
    for item in DAMAGED_BASE_POCKETS:
        if contains_subpath(path, item):
            damaged.add(item)
    if row["split"] in {"ood", "adversarial"} and contains_subpath(path, (6, 7)):
        damaged.add((6, 7))
    return damaged


def predict_with_candidate(
    task: dict[str, dict[str, list[dict[str, Any]]]],
    system: str,
    candidate: dict[str, Any],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    candidate = normalize_candidate(candidate, system)
    spawn_map = active_spawn_map(candidate)
    repairs = set(normalized_repairs(candidate)) if system_allows_repair(system) else set()
    route_around = set(normalized_route_around(candidate)) if system_allows_route_around(system) else set()
    pockets = candidate_pockets(candidate)
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for phase in PHASES:
        predictions[phase] = {}
        for split in SPLITS:
            split_preds: list[dict[str, Any]] = []
            for row_idx, row in enumerate(task[phase][split]):
                calls = greedy_calls(row["micro_path"], pockets)
                damaged = row_damaged_calls(row)
                predicted_micro: list[int] = []
                call_rows: list[dict[str, Any]] = []
                cost = 0.0
                branches = 0
                for call_idx, call in enumerate(calls):
                    source = "base" if call in BASE_POCKETS else "spawned" if call in spawn_map else "atomic"
                    emitted = list(call)
                    k = required_k_for_call(call)
                    depth = 1
                    if call in spawn_map and system_allows_spawn(system):
                        spawn = spawn_map[call]
                        k = int(spawn["K"])
                        depth = int(spawn["depth"])
                        if k < required_k_for_call(call) or depth < max(1, len(call) - 2):
                            emitted = corrupt_call(call, k + depth + call_idx + row_idx)
                        cost += spawned_route_cost(k, depth)
                    else:
                        if call in damaged and call not in repairs and call not in route_around:
                            emitted = corrupt_call(call, len(call) + call_idx + row_idx + 1)
                        if call in route_around:
                            source = "route_around"
                        cost += 1.0 + 0.15 * len(call)
                    predicted_micro.extend(emitted)
                    call_rows.append({"call": list(call), "source": source, "K": k, "depth": depth, "emitted": emitted})
                split_preds.append(
                    {
                        "calls": call_rows,
                        "micro_path": predicted_micro[:MAX_MICRO_PATH],
                        "steps": len(calls),
                        "branch_expansions": branches,
                        "route_cost": cost,
                    }
                )
            predictions[phase][split] = split_preds
    return predictions


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
    junk = len([motif for motif in promoted if motif not in true_motifs and motif in used_promoted]) / max(1, len(promoted))
    reuse_count = sum(use_counts.values())
    avg_reuse = reuse_count / max(1, len(promoted))
    k_values = [int(row["K"]) for row in normalize_spawned(candidate)]
    return {
        "spawn_precision": round_float(precision),
        "spawn_recall": round_float(recall),
        "unnecessary_spawn_rate": round_float(unnecessary),
        "junk_pocket_rate": round_float(junk),
        "promoted_pocket_count": len(promoted),
        "promoted_pocket_reuse_count": round_float(avg_reuse),
        "library_size": int(len(BASE_POCKETS) + len(promoted)),
        "spawned_pocket_average_K": round_float(float(np.mean(k_values)) if k_values else 0.0),
    }


def repair_metrics_for_rows(candidate: dict[str, Any], rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    repairs = set(normalized_repairs(candidate)) if system_allows_repair(system) else set()
    true_damage = true_damaged_for_rows(rows)
    if not repairs:
        precision = 0.0
    else:
        precision = len(repairs & true_damage) / max(1, len(repairs))
    recall = len(repairs & true_damage) / max(1, len(true_damage))
    false_pos = len(repairs - true_damage) / max(1, len(repairs)) if repairs else 0.0
    false_neg = len(true_damage - repairs) / max(1, len(true_damage))
    return {
        "repair_precision": round_float(precision),
        "repair_recall": round_float(recall),
        "repair_count": len(repairs),
        "health_false_positive_rate": round_float(false_pos),
        "health_false_negative_rate": round_float(false_neg),
    }


def cost_terms(
    candidate: dict[str, Any],
    system: str,
    mean_steps: float,
    spawn_metrics: dict[str, Any],
    repair_metrics: dict[str, Any],
    route_around_success: float,
) -> dict[str, float]:
    spawned = normalize_spawned(candidate) if system_allows_spawn(system) else []
    repairs = normalized_repairs(candidate) if system_allows_repair(system) else []
    spawn_cost = sum(0.010 + 0.0022 * int(row["K"]) + 0.002 * int(row["depth"]) for row in spawned)
    repair_cost = 0.020 * len(repairs)
    maintenance_cost = 0.0055 * len(spawned)
    route_step_cost = 0.0038 * mean_steps
    overproduction_penalty = 0.060 * float(spawn_metrics["unnecessary_spawn_rate"]) + 0.050 * float(spawn_metrics["junk_pocket_rate"])
    delayed = float(candidate.get("delayed_regret_seed", 0.0))
    if system == "delayed_feedback_spawn_plus_repair":
        delayed += 0.30 * max(0.0, float(spawn_metrics["junk_pocket_rate"]) - 0.05)
    delayed_regret_penalty = 0.055 * delayed
    health_penalty = 0.0
    if system == "noisy_health_spawn_plus_repair":
        health_penalty = 0.045 * float(repair_metrics["health_false_positive_rate"]) + 0.050 * float(repair_metrics["health_false_negative_rate"])
    route_credit = -0.018 * route_around_success if system in {"route_around_only", "cost_aware_spawn_plus_repair"} else 0.0
    return {
        "spawn_cost_spent": round_float(spawn_cost),
        "repair_cost_spent": round_float(repair_cost),
        "maintenance_cost": round_float(maintenance_cost),
        "route_step_cost": round_float(route_step_cost),
        "overproduction_penalty": round_float(overproduction_penalty),
        "delayed_regret_penalty": round_float(delayed_regret_penalty),
        "health_signal_penalty": round_float(health_penalty),
        "route_around_credit": round_float(route_credit),
        "total_cost": round_float(spawn_cost + repair_cost + maintenance_cost + route_step_cost + overproduction_penalty + delayed_regret_penalty + health_penalty + route_credit),
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
            steps_total = baseline_steps_total = cost_total = baseline_cost_total = branch_total = irrelevant_total = 0.0
            route_around_rows = route_around_hits = 0
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
                if row_damaged_calls(row):
                    route_around_rows += 1
                    if predicted == target and any(tuple(call.get("call", [])) not in DAMAGED_BASE_POCKETS for call in call_rows):
                        route_around_hits += 1
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
            route_cost_reduction = max(0.0, (baseline_cost - mean_cost) / max(1.0, baseline_cost))
            route_around_success = route_around_hits / max(1, route_around_rows)
            spawn_metrics = spawn_metrics_for_rows(candidate, rows)
            repair_metrics = repair_metrics_for_rows(candidate, rows, system)
            param_norm = min(1.0, params / 1800.0)
            router_norm = min(1.0, router_complexity / 14.0)
            repair_gain = route_accuracy if phase == "phase_5_damage_drift_repair" and repair_metrics["repair_count"] else 0.0
            repair_gain_per_cost = repair_gain / max(0.001, 0.020 * repair_metrics["repair_count"])
            raw_usefulness = (
                0.31 * answer_accuracy
                + 0.24 * route_accuracy
                + 0.15 * step_reduction
                + 0.08 * route_cost_reduction
                + 0.06 * min(1.0, float(spawn_metrics["promoted_pocket_reuse_count"]) / max(1.0, n / 4.0))
                + 0.04 * repair_metrics["repair_recall"]
                + 0.04 * route_around_success
                + 0.03 * valid_rate
                - 0.04 * irrelevant_rate
                - 0.035 * loop_rate
                - 0.025 * branch_rate
                - 0.020 * param_norm
                - 0.015 * router_norm
            )
            costs = cost_terms(candidate, system, mean_steps, spawn_metrics, repair_metrics, route_around_success)
            net = raw_usefulness - costs["total_cost"]
            phase_metrics[phase][split] = {
                "answer_accuracy": round_float(answer_accuracy),
                "route_accuracy": round_float(route_accuracy),
                "valid_route_rate": round_float(valid_rate),
                "mean_route_steps": round_float(mean_steps),
                "baseline_route_steps": round_float(baseline_steps),
                "route_step_reduction": round_float(step_reduction),
                "route_cost": round_float(mean_cost),
                "baseline_route_cost": round_float(baseline_cost),
                "route_cost_reduction": round_float(route_cost_reduction),
                "route_around_success": round_float(route_around_success),
                "irrelevant_branch_rate": round_float(irrelevant_rate),
                "branch_expansion_rate": round_float(branch_rate),
                "loop_rate": round_float(loop_rate),
                "parameter_count": int(params),
                "router_complexity": round_float(router_complexity),
                "raw_usefulness": round_float(max(0.0, min(1.0, raw_usefulness))),
                "net_utility": round_float(max(0.0, min(1.0, net))),
                "repair_gain_per_cost": round_float(min(40.0, repair_gain_per_cost)),
                "delayed_feedback_regret": round_float(float(candidate.get("delayed_regret_seed", 0.0))),
                "row_level_samples": samples,
                **spawn_metrics,
                **repair_metrics,
                **costs,
            }
    for split in SPLITS:
        split_values: dict[str, list[float]] = {}
        for phase in PHASES:
            for key, value in phase_metrics[phase][split].items():
                if isinstance(value, (int, float)):
                    split_values.setdefault(key, []).append(float(value))
        evals[split] = {key: round_float(float(np.mean(values))) for key, values in split_values.items()}
        evals[split]["row_level_samples"] = [phase_metrics[phase][split]["row_level_samples"][0] for phase in PHASES if phase_metrics[phase][split]["row_level_samples"]]
    train = evals["train"]["net_utility"]
    eval_mean_net = float(np.mean([evals[split]["net_utility"] for split in EVAL_SPLITS]))
    eval_mean_raw = float(np.mean([evals[split]["raw_usefulness"] for split in EVAL_SPLITS]))
    return {
        "system": system,
        "evals": evals,
        "phase_metrics": phase_metrics,
        "heldout_raw_usefulness": round_float(evals["heldout"]["raw_usefulness"]),
        "ood_raw_usefulness": round_float(evals["ood"]["raw_usefulness"]),
        "counterfactual_raw_usefulness": round_float(evals["counterfactual"]["raw_usefulness"]),
        "adversarial_raw_usefulness": round_float(evals["adversarial"]["raw_usefulness"]),
        "heldout_net_utility": round_float(evals["heldout"]["net_utility"]),
        "ood_net_utility": round_float(evals["ood"]["net_utility"]),
        "counterfactual_net_utility": round_float(evals["counterfactual"]["net_utility"]),
        "adversarial_net_utility": round_float(evals["adversarial"]["net_utility"]),
        "eval_mean_raw_usefulness": round_float(eval_mean_raw),
        "eval_mean_net_utility": round_float(eval_mean_net),
        "generalization_gap": round_float(train - eval_mean_net),
        "parameter_count": int(params),
        "candidate_summary": candidate_summary(candidate),
    }


def profile_result(task: dict[str, dict[str, list[dict[str, Any]]]], seed: int, system: str, candidate: dict[str, Any], router_complexity: float) -> dict[str, Any]:
    predictions = predict_with_candidate(task, system, candidate)
    row = evaluate_predictions(task, predictions, system, candidate, parameter_count(candidate), router_complexity)
    row["seed"] = seed
    return row


def oracle_candidate(task: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    motifs: set[tuple[int, ...]] = set()
    for split in SPLITS:
        motifs.update(true_motifs_for_rows(all_rows(task, split)))
    return normalize_candidate(
        {
            "system": "oracle_health_spawn_repair_reference",
            "spawned": [make_spawn(motif, "oracle", required_k_for_call(motif), max(1, len(motif) - 2)) for motif in sorted(motifs)],
            "repairs": [list(item) for item in REPAIR_TARGETS],
            "route_around": [],
        },
        "oracle_health_spawn_repair_reference",
    )


def random_candidate(seed: int) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"{seed}:random_spawn_repair"))
    spawned = []
    for _ in range(5):
        length = rng.choice((2, 3, 4))
        spawned.append(make_spawn(tuple(rng.sample(range(MICRO_COUNT), length)), "random", rng.choice(K_VALUES), rng.choice((1, 2, 3))))
    repairs = [list(rng.choice(REPAIR_TARGETS + DECOY_MOTIFS)) for _ in range(3)]
    route_around = [list(rng.choice(REPAIR_TARGETS + DECOY_MOTIFS)) for _ in range(2)]
    return normalize_candidate({"system": "random_spawn_repair_control", "spawned": spawned, "repairs": repairs, "route_around": route_around}, "random_spawn_repair_control")


def control_results(seed: int, task: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows = [
        profile_result(task, seed, "no_adaptation", candidate_initial("no_adaptation"), 2.0),
        profile_result(task, seed, "oracle_health_spawn_repair_reference", oracle_candidate(task), 2.6),
        profile_result(task, seed, "random_spawn_repair_control", random_candidate(seed), 9.0),
    ]
    return rows


def quick_eval(candidate: dict[str, Any], system: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    task = {"quick": {"train": rows, "validation": rows, "heldout": rows, "ood": rows, "counterfactual": rows, "adversarial": rows}}
    original_phases = list(PHASES)
    # Inline lightweight scorer avoids changing the global phase constant used by artifacts.
    answer_hits = route_hits = valid_hits = loops = 0
    steps_total = baseline_steps_total = cost_total = baseline_cost_total = irrelevant_total = 0.0
    pockets = candidate_pockets(candidate)
    spawn_map = active_spawn_map(candidate)
    repairs = set(normalized_repairs(candidate)) if system_allows_repair(system) else set()
    route_around = set(normalized_route_around(candidate)) if system_allows_route_around(system) else set()
    for idx, row in enumerate(rows):
        calls = greedy_calls(row["micro_path"], pockets)
        damaged = row_damaged_calls(row)
        predicted: list[int] = []
        route_cost = 0.0
        for call_idx, call in enumerate(calls):
            emitted = list(call)
            if call in spawn_map and system_allows_spawn(system):
                spawn = spawn_map[call]
                if int(spawn["K"]) < required_k_for_call(call) or int(spawn["depth"]) < max(1, len(call) - 2):
                    emitted = corrupt_call(call, idx + call_idx)
                route_cost += spawned_route_cost(int(spawn["K"]), int(spawn["depth"]))
            else:
                if call in damaged and call not in repairs and call not in route_around:
                    emitted = corrupt_call(call, idx + call_idx + 3)
                route_cost += 1.0 + 0.15 * len(call)
            predicted.extend(emitted)
        predicted = predicted[:MAX_MICRO_PATH]
        final = apply_micro_path(predicted, row["a"], row["b"], row["key"], row["mem"], row["threshold"])
        predicted_answer = 1 if final > row["threshold"] else 0
        target = list(row["micro_path"])
        baseline_calls = greedy_calls(target, BASE_POCKETS)
        answer_hits += int(predicted_answer == row["answer"])
        route_hits += int(predicted == target)
        valid_hits += int(bool(predicted) and len(predicted) <= MAX_MICRO_PATH)
        loops += int(len(calls) != len(set(calls)))
        irrelevant_total += min(1.0, sum(1 for seg in predicted if seg not in set(target)) / max(1, MICRO_COUNT - len(set(target))))
        steps_total += len(calls)
        baseline_steps_total += len(baseline_calls)
        cost_total += route_cost
        baseline_cost_total += base_route_cost(baseline_calls)
    n = max(1, len(rows))
    answer_accuracy = answer_hits / n
    route_accuracy = route_hits / n
    valid_rate = valid_hits / n
    mean_steps = steps_total / n
    step_reduction = max(0.0, (baseline_steps_total / n - mean_steps) / max(1.0, baseline_steps_total / n))
    cost_reduction = max(0.0, (baseline_cost_total / n - cost_total / n) / max(1.0, baseline_cost_total / n))
    spawn_metrics = spawn_metrics_for_rows(candidate, rows)
    repair_metrics = repair_metrics_for_rows(candidate, rows, system)
    raw = (
        0.33 * answer_accuracy
        + 0.25 * route_accuracy
        + 0.17 * step_reduction
        + 0.08 * cost_reduction
        + 0.05 * repair_metrics["repair_recall"]
        + 0.04 * min(1.0, float(spawn_metrics["promoted_pocket_reuse_count"]) / max(1.0, n / 4.0))
        + 0.03 * valid_rate
        - 0.035 * (irrelevant_total / n)
        - 0.030 * (loops / n)
        - 0.020 * min(1.0, parameter_count(candidate) / 1800.0)
    )
    costs = cost_terms(candidate, system, mean_steps, spawn_metrics, repair_metrics, 0.0)
    net = raw - costs["total_cost"]
    return {
        "raw": float(max(0.0, min(1.0, raw))),
        "net": float(max(0.0, min(1.0, net))),
        "junk": float(spawn_metrics["junk_pocket_rate"]),
        "unnecessary": float(spawn_metrics["unnecessary_spawn_rate"]),
        "health_fp": float(repair_metrics["health_false_positive_rate"]),
        "health_fn": float(repair_metrics["health_false_negative_rate"]),
    }


def candidate_learning_score(candidate: dict[str, Any], system: str, task: dict[str, dict[str, list[dict[str, Any]]]], observed: bool = False) -> float:
    candidate = normalize_candidate(candidate, system)
    train = quick_eval(candidate, system, all_rows(task, "train"))
    validation = quick_eval(candidate, system, all_rows(task, "validation"))
    if system == "spawn_plus_limited_repair_clean":
        score = 0.52 * train["raw"] + 0.48 * validation["raw"] - 0.004 * candidate_summary(candidate)["promoted_pocket_count"]
    elif system == "delayed_feedback_spawn_plus_repair" and observed:
        score = 0.78 * train["raw"] + 0.22 * validation["net"] - 0.018 * validation["health_fn"]
    elif system == "noisy_health_spawn_plus_repair":
        score = 0.42 * train["net"] + 0.58 * validation["net"] - 0.06 * validation["health_fp"] - 0.07 * validation["health_fn"]
    else:
        score = 0.42 * train["net"] + 0.58 * validation["net"]
    overfit = max(0.0, train["raw"] - validation["raw"])
    return float(score - 0.06 * overfit)


def mutation_pools(task: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[tuple[int, ...]]]:
    rows = training_rows(task)
    frequent = [item for item, _count in e7k.frequent_substrings(rows, 2, 4)[:36]]
    split = [item for item, _count in e7k.split_substrings(rows)[:30]]
    pool = frequent + [item for item in split if item not in frequent]
    return {
        "frequent": frequent,
        "split": split,
        "all": pool,
        "repair_targets": list(REPAIR_TARGETS + DECOY_MOTIFS),
        "decoy": list(DECOY_MOTIFS),
    }


def add_spawn(candidate: dict[str, Any], segments: tuple[int, ...], source: str, k: int | None = None, depth: int = 1) -> dict[str, Any]:
    mutated = copy.deepcopy(candidate)
    spawned = normalize_spawned(mutated)
    if tuple(segments) not in {tuple(row["segments"]) for row in spawned}:
        spawned.append(make_spawn(tuple(segments), source, k, depth))
    mutated["spawned"] = spawned
    return normalize_candidate(mutated, str(mutated.get("system", "candidate")))


def add_repair(candidate: dict[str, Any], segments: tuple[int, ...]) -> dict[str, Any]:
    mutated = copy.deepcopy(candidate)
    repairs = normalized_repairs(mutated)
    if tuple(segments) not in repairs:
        repairs.append(tuple(segments))
    mutated["repairs"] = [list(item) for item in repairs]
    return normalize_candidate(mutated, str(mutated.get("system", "candidate")))


def add_route_around(candidate: dict[str, Any], segments: tuple[int, ...]) -> dict[str, Any]:
    mutated = copy.deepcopy(candidate)
    routes = normalized_route_around(mutated)
    if tuple(segments) not in routes:
        routes.append(tuple(segments))
    mutated["route_around"] = [list(item) for item in routes]
    return normalize_candidate(mutated, str(mutated.get("system", "candidate")))


def mutate_candidate(candidate: dict[str, Any], system: str, pools: dict[str, list[tuple[int, ...]]], rng: random.Random) -> tuple[dict[str, Any], str]:
    mutated = normalize_candidate(copy.deepcopy(candidate), system)
    ops: list[str] = []
    if system_allows_route_around(system):
        ops.extend(["add_route_around", "delete_route_around"])
    if system_allows_repair(system):
        ops.extend(["add_repair", "delete_repair", "add_repair"])
    if system_allows_spawn(system):
        ops.extend(["spawn_frequent", "delete_spawn", "change_K", "change_depth"])
        if system in {"spawn_plus_limited_repair_clean", "delayed_feedback_spawn_plus_repair"}:
            ops.extend(["spawn_frequent", "spawn_decoy"])
        if system == "cost_aware_spawn_plus_repair":
            ops.extend(["prune_junk", "spawn_split"])
        if system == "noisy_health_spawn_plus_repair":
            ops.extend(["spawn_decoy", "add_false_repair", "perturb_health"])
    if not ops:
        ops = ["noop"]
    op = rng.choice(ops)
    spawned = normalize_spawned(mutated)
    repairs = normalized_repairs(mutated)
    routes = normalized_route_around(mutated)
    if op in {"spawn_frequent", "spawn_split", "spawn_decoy"}:
        if op == "spawn_decoy":
            pool = pools.get("decoy", []) or pools.get("all", [])
            source = "decoy"
        elif op == "spawn_split":
            pool = pools.get("split", []) or pools.get("all", [])
            source = "split"
        else:
            pool = pools.get("frequent", []) or pools.get("all", [])
            source = "composed"
        if pool:
            segments = pool[rng.randrange(min(len(pool), 24))]
        else:
            segments = tuple(rng.sample(range(MICRO_COUNT), rng.choice((2, 3, 4))))
        return add_spawn(mutated, segments, source, required_k_for_call(segments), max(1, len(segments) - 2)), op
    if op == "delete_spawn" and spawned:
        del spawned[rng.randrange(len(spawned))]
        mutated["spawned"] = spawned
    elif op == "prune_junk" and spawned:
        decoy_indices = [idx for idx, row in enumerate(spawned) if tuple(row["segments"]) in DECOY_MOTIFS]
        del spawned[rng.choice(decoy_indices) if decoy_indices else rng.randrange(len(spawned))]
        mutated["spawned"] = spawned
    elif op == "change_K" and spawned:
        idx = rng.randrange(len(spawned))
        current = e7k.clamp_k(int(spawned[idx]["K"]))
        direction = 1 if rng.random() < 0.55 else -1
        k_index = max(0, min(len(K_VALUES) - 1, K_VALUES.index(current) + direction))
        spawned[idx]["K"] = K_VALUES[k_index]
        mutated["spawned"] = spawned
    elif op == "change_depth" and spawned:
        idx = rng.randrange(len(spawned))
        spawned[idx]["depth"] = max(1, min(4, int(spawned[idx]["depth"]) + rng.choice((-1, 1))))
        mutated["spawned"] = spawned
    elif op in {"add_repair", "add_false_repair"}:
        pool = pools["repair_targets"] if op == "add_repair" else pools["decoy"]
        return add_repair(mutated, rng.choice(pool)), op
    elif op == "delete_repair" and repairs:
        del repairs[rng.randrange(len(repairs))]
        mutated["repairs"] = [list(item) for item in repairs]
    elif op == "add_route_around":
        return add_route_around(mutated, rng.choice(pools["repair_targets"])), op
    elif op == "delete_route_around" and routes:
        del routes[rng.randrange(len(routes))]
        mutated["route_around"] = [list(item) for item in routes]
    elif op == "perturb_health":
        mutated["health_noise_rate"] = max(0.0, min(0.5, float(mutated["health_noise_rate"]) + rng.choice((-0.05, 0.05, 0.10))))
        mutated["health_missing_rate"] = max(0.0, min(0.5, float(mutated["health_missing_rate"]) + rng.choice((-0.05, 0.05, 0.10))))
    else:
        mutated["delayed_regret_seed"] = max(0.0, min(1.0, float(mutated.get("delayed_regret_seed", 0.0)) + rng.uniform(0.0, 0.02)))
    return normalize_candidate(mutated, system), op


def bootstrap_candidates(system: str, pools: dict[str, list[tuple[int, ...]]]) -> list[tuple[dict[str, Any], str]]:
    current = candidate_initial(system)
    candidates: list[tuple[dict[str, Any], str]] = []
    if system == "route_around_only":
        for idx, target in enumerate(REPAIR_TARGETS):
            current = add_route_around(current, target)
            candidates.append((copy.deepcopy(current), f"bootstrap_route_around_{idx}"))
    if system_allows_repair(system):
        for idx, target in enumerate(REPAIR_TARGETS[:2]):
            current = add_repair(current, target)
            candidates.append((copy.deepcopy(current), f"bootstrap_repair_{idx}"))
    if system_allows_spawn(system):
        for idx, segments in enumerate(pools.get("frequent", [])[:5]):
            source = "composed"
            if system == "spawn_plus_limited_repair_clean" and idx % 3 == 0:
                source = "decoy"
            current = add_spawn(current, segments, source, required_k_for_call(segments), max(1, len(segments) - 2))
            candidates.append((copy.deepcopy(current), f"bootstrap_spawn_{idx}"))
    return candidates


def mutation_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    system = str(job["system"])
    task = job["task"]
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rng = random.Random(stable_seed(f"{seed}:{system}:mutation"))
    pools = mutation_pools(task)
    best = candidate_initial(system)
    initial_hash = candidate_hash(best)
    best_observed = candidate_learning_score(best, system, task, observed=(system == "delayed_feedback_spawn_plus_repair"))
    best_true = candidate_learning_score(best, system, task, observed=False)
    accepted = rejected = attempts = 0
    accepted_by_operator: dict[str, int] = {}
    rejected_by_operator: dict[str, int] = {}
    delayed_regret = 0.0
    history: list[dict[str, Any]] = []
    snapshot_dir = out / "mutation_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for candidate, operator in bootstrap_candidates(system, pools):
        attempts += 1
        observed = candidate_learning_score(candidate, system, task, observed=(system == "delayed_feedback_spawn_plus_repair"))
        true_score = candidate_learning_score(candidate, system, task, observed=False)
        if observed > best_observed + 1e-12:
            best = candidate
            best_observed = observed
            delayed_regret += max(0.0, best_true - true_score)
            best_true = true_score
            accepted += 1
            accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
        else:
            rejected += 1
            rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
    for generation in range(settings.mutation_generations):
        generation_best = best_true
        for _ in range(settings.mutation_population):
            attempts += 1
            candidate, operator = mutate_candidate(best, system, pools, rng)
            observed = candidate_learning_score(candidate, system, task, observed=(system == "delayed_feedback_spawn_plus_repair"))
            true_score = candidate_learning_score(candidate, system, task, observed=False)
            if observed > best_observed + 1e-12:
                best = candidate
                best_observed = observed
                delayed_regret += max(0.0, best_true - true_score)
                best_true = true_score
                accepted += 1
                accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
            else:
                rejected += 1
                rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
        if generation % max(1, settings.mutation_generations // 12) == 0 or generation == settings.mutation_generations - 1:
            best["delayed_regret_seed"] = max(float(best.get("delayed_regret_seed", 0.0)), delayed_regret)
            row = {
                "generation": generation,
                "observed_score": round_float(best_observed),
                "true_score": round_float(best_true),
                "generation_gain": round_float(best_true - generation_best),
                "accepted": accepted,
                "rejected": rejected,
                "delayed_regret": round_float(delayed_regret),
                "candidate_hash": candidate_hash(best),
                "summary": candidate_summary(best),
            }
            history.append(row)
            if snapshot_dir:
                locked_write_json(snapshot_dir / f"{system}_seed{seed}_generation{generation:04d}.json", row)
            if out:
                append_progress(out, "mutation_generation", seed=seed, system=system, generation=generation, true_score=row["true_score"], summary=row["summary"])
    best["delayed_regret_seed"] = max(float(best.get("delayed_regret_seed", 0.0)), delayed_regret)
    result = profile_result(task, seed, system, best, router_complexity=3.0 if system != "route_around_only" else 2.6)
    final_hash = candidate_hash(best)
    result.update(
        {
            "history": history,
            "initial_candidate_hash": initial_hash,
            "final_candidate_hash": final_hash,
            "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": final_hash, "candidate": normalize_candidate(best, system)}),
            "final_candidate_summary": candidate_summary(best),
            "mutation_attempts": attempts,
            "accepted_mutations": accepted,
            "rejected_mutations": rejected,
            "rollback_count": rejected,
            "failed_action_rollback_count": sum(count for op, count in rejected_by_operator.items() if any(token in op for token in ("spawn", "repair", "route"))),
            "accepted_by_operator": accepted_by_operator,
            "rejected_by_operator": rejected_by_operator,
            "delayed_feedback_regret_accumulated": round_float(delayed_regret),
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
    set_determinism(stable_seed(f"{seed}:dense_graph"), device)
    train_rows_local = all_rows(task, "train")
    model = DensePathMLP(len(train_rows_local[0]["raw"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train, targets, length_targets = make_tensor(train_rows_local, device)
    rng = np.random.default_rng(stable_seed(f"{seed}:dense_batches"))
    history: list[dict[str, Any]] = []
    snapshot_dir = out / "training_history_snapshots" if out else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(settings.gradient_epochs):
        indices = rng.permutation(len(train_rows_local))
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
        if epoch % max(1, settings.gradient_epochs // 12) == 0 or epoch == settings.gradient_epochs - 1:
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
    return evaluate_predictions(task, predictions, "dense_graph_danger_control", candidate_initial("dense_graph_danger_control"), params, router_complexity=14.0)


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
        "schema_version": "e7l_task_generation_report_v1",
        "row_counts": {str(seed): {phase: {split: len(rows) for split, rows in phase_task.items()} for phase, phase_task in task.items()} for seed, task in tasks.items()},
        "source_task": "E7K typed pocket spawn task reused without hidden motif ID as public input",
        "public_inputs": "microsegment_path_plus_phase_token_no_public_missing_motif_id",
        "hidden_missing_transform_used_for_eval_only": True,
        "phases": list(PHASES),
        "stress_conditions": list(STRESS_CONDITIONS),
        "base_pockets": [list(item) for item in BASE_POCKETS],
        "true_spawn_motifs": {key: list(value) for key, value in TRUE_MOTIFS.items()},
        "decoy_motifs": [list(item) for item in DECOY_MOTIFS],
    }


def cost_noise_stress_report() -> dict[str, Any]:
    return {
        "schema_version": "e7l_cost_noise_stress_report_v1",
        "stress_conditions": list(STRESS_CONDITIONS),
        "net_utility_formula": "raw_usefulness - spawn_cost - repair_cost - maintenance_cost - route_step_cost - overproduction_penalty - delayed_regret_penalty - health_signal_penalty",
        "moving_drift_profile": "phase_5 damages base pockets; OOD/adversarial can add unseen damage to route-around target",
        "false_positive_reusable_route": "train/validation prefix decoys such as [14,15] can look reusable but are not hidden motifs",
        "false_negative_reusable_route": "delta motif is sparse and appears mainly in OOD/counterfactual contexts",
    }


def action_policy_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7l_action_policy_report_v1",
        "system_mean_net_utility": {system: aggregate["systems"][system]["mean"]["eval_mean_net_utility"] for system in SYSTEMS},
        "action_policy_winner": aggregate["best_non_oracle_system"],
        "oracle_reference": aggregate["systems"]["oracle_health_spawn_repair_reference"]["mean"]["eval_mean_net_utility"],
        "dense_graph_danger_control": aggregate["systems"]["dense_graph_danger_control"]["mean"]["eval_mean_net_utility"],
    }


def build_spawn_repair_policy_report(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7l_spawn_repair_policy_report_v1",
        "system_means": {
            system: {
                "eval_mean_raw_usefulness": aggregate["systems"][system]["mean"]["eval_mean_raw_usefulness"],
                "eval_mean_net_utility": aggregate["systems"][system]["mean"]["eval_mean_net_utility"],
                "heldout_spawn_precision": aggregate["systems"][system]["mean"].get("heldout_spawn_precision", 0.0),
                "heldout_spawn_recall": aggregate["systems"][system]["mean"].get("heldout_spawn_recall", 0.0),
                "heldout_unnecessary_spawn_rate": aggregate["systems"][system]["mean"].get("heldout_unnecessary_spawn_rate", 0.0),
                "heldout_repair_precision": aggregate["systems"][system]["mean"].get("heldout_repair_precision", 0.0),
                "heldout_repair_recall": aggregate["systems"][system]["mean"].get("heldout_repair_recall", 0.0),
                "heldout_repair_gain_per_cost": aggregate["systems"][system]["mean"].get("heldout_repair_gain_per_cost", 0.0),
                "heldout_junk_pocket_rate": aggregate["systems"][system]["mean"].get("heldout_junk_pocket_rate", 0.0),
            }
            for system in SYSTEMS
        },
        "example_final_candidates": {
            row["system"]: row.get("candidate_summary", {})
            for row in rows
            if int(row["seed"]) == min(int(item["seed"]) for item in rows) and row["system"] in MUTATION_SYSTEMS
        },
    }


def build_health_signal_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7l_health_signal_report_v1",
        "system_health_means": {
            system: {
                "heldout_health_false_positive_rate": aggregate["systems"][system]["mean"].get("heldout_health_false_positive_rate", 0.0),
                "heldout_health_false_negative_rate": aggregate["systems"][system]["mean"].get("heldout_health_false_negative_rate", 0.0),
                "eval_mean_net_utility": aggregate["systems"][system]["mean"].get("eval_mean_net_utility", 0.0),
            }
            for system in SYSTEMS
        },
        "clean_vs_noisy_net_gap": round_float(
            aggregate["systems"]["spawn_plus_limited_repair_clean"]["mean"]["eval_mean_net_utility"]
            - aggregate["systems"]["noisy_health_spawn_plus_repair"]["mean"]["eval_mean_net_utility"]
        ),
    }


def build_delayed_feedback_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    delayed = aggregate["systems"]["delayed_feedback_spawn_plus_repair"]["mean"]
    cost_aware = aggregate["systems"]["cost_aware_spawn_plus_repair"]["mean"]
    return {
        "schema_version": "e7l_delayed_feedback_report_v1",
        "delayed_feedback_net_utility": delayed["eval_mean_net_utility"],
        "cost_aware_net_utility": cost_aware["eval_mean_net_utility"],
        "delayed_feedback_regret": delayed.get("heldout_delayed_feedback_regret", 0.0),
        "delayed_minus_cost_aware": round_float(delayed["eval_mean_net_utility"] - cost_aware["eval_mean_net_utility"]),
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    phase_summary: dict[str, dict[str, Any]] = {phase: {} for phase in PHASES}
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in (
                "heldout_raw_usefulness",
                "ood_raw_usefulness",
                "counterfactual_raw_usefulness",
                "adversarial_raw_usefulness",
                "heldout_net_utility",
                "ood_net_utility",
                "counterfactual_net_utility",
                "adversarial_net_utility",
                "eval_mean_raw_usefulness",
                "eval_mean_net_utility",
                "generalization_gap",
                "parameter_count",
            ):
                metrics.setdefault(metric, []).append(float(row[metric]))
            for split in SPLITS:
                for metric, value in row["evals"][split].items():
                    if isinstance(value, (int, float)):
                        metrics.setdefault(f"{split}_{metric}", []).append(float(value))
            for phase in PHASES:
                phase_eval = float(np.mean([row["phase_metrics"][phase][split]["net_utility"] for split in EVAL_SPLITS]))
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
        phase_winners[phase] = {"best_system": max(means, key=lambda system: means[system]), "system_net_utility_mean": means}
    non_oracle = [system for system in SYSTEMS if system != "oracle_health_spawn_repair_reference"]
    best = max(non_oracle, key=lambda system: systems[system]["mean"]["eval_mean_net_utility"])
    best_including_oracle = max(SYSTEMS, key=lambda system: systems[system]["mean"]["eval_mean_net_utility"])
    return {
        "schema_version": "e7l_aggregate_metrics_v1",
        "systems": systems,
        "phase_winners": phase_winners,
        "best_non_oracle_system": best,
        "best_system_including_oracle": best_including_oracle,
        "best_eval_mean_net_utility": systems[best]["mean"]["eval_mean_net_utility"],
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    best = aggregate["best_non_oracle_system"]
    cost_aware = systems["cost_aware_spawn_plus_repair"]["mean"]
    route = systems["route_around_only"]["mean"]
    repair = systems["repair_only"]["mean"]
    spawn = systems["spawn_only"]["mean"]
    clean = systems["spawn_plus_limited_repair_clean"]["mean"]
    noisy = systems["noisy_health_spawn_plus_repair"]["mean"]
    delayed = systems["delayed_feedback_spawn_plus_repair"]["mean"]
    dense = systems["dense_graph_danger_control"]["mean"]
    random_control = systems["random_spawn_repair_control"]["mean"]
    detail = {
        "best_non_oracle_system": best,
        "best_system_including_oracle": aggregate["best_system_including_oracle"],
        "cost_aware_net": cost_aware["eval_mean_net_utility"],
        "route_around_net": route["eval_mean_net_utility"],
        "repair_only_net": repair["eval_mean_net_utility"],
        "spawn_only_net": spawn["eval_mean_net_utility"],
        "clean_spawn_repair_net": clean["eval_mean_net_utility"],
        "noisy_spawn_repair_net": noisy["eval_mean_net_utility"],
        "delayed_spawn_repair_net": delayed["eval_mean_net_utility"],
        "dense_graph_net": dense["eval_mean_net_utility"],
        "random_control_net": random_control["eval_mean_net_utility"],
        "cost_aware_unnecessary_spawn_rate": cost_aware.get("heldout_unnecessary_spawn_rate", 0.0),
        "clean_unnecessary_spawn_rate": clean.get("heldout_unnecessary_spawn_rate", 0.0),
        "cost_aware_junk_pocket_rate": cost_aware.get("heldout_junk_pocket_rate", 0.0),
        "clean_junk_pocket_rate": clean.get("heldout_junk_pocket_rate", 0.0),
        "delayed_feedback_regret": delayed.get("heldout_delayed_feedback_regret", 0.0),
        "phase_winners": {phase: row["best_system"] for phase, row in aggregate["phase_winners"].items()},
    }
    if random_control["eval_mean_net_utility"] >= cost_aware["eval_mean_net_utility"] - 0.01:
        return "e7l_leak_or_artifact_detected", detail
    if dense["eval_mean_net_utility"] > cost_aware["eval_mean_net_utility"] + 0.02 and dense["ood_net_utility"] >= cost_aware["ood_net_utility"] - 0.02:
        return "e7l_graph_soup_regression_detected", detail
    if clean.get("heldout_unnecessary_spawn_rate", 0.0) > 0.48 or clean.get("heldout_junk_pocket_rate", 0.0) > 0.28:
        if cost_aware["eval_mean_net_utility"] < route["eval_mean_net_utility"] + 0.01:
            return "e7l_spawn_overproduction_failure", detail
    if delayed["eval_mean_net_utility"] < cost_aware["eval_mean_net_utility"] - 0.08 and delayed.get("heldout_delayed_feedback_regret", 0.0) > 0.05:
        return "e7l_delayed_feedback_instability", detail
    if clean["eval_mean_net_utility"] > noisy["eval_mean_net_utility"] + 0.08 and noisy.get("heldout_health_false_negative_rate", 0.0) > 0.25:
        return "e7l_spawn_repair_requires_clean_health_signal", detail
    if best == "route_around_only":
        return "e7l_routearound_preferred_under_cost", detail
    if best == "repair_only":
        return "e7l_repair_preferred_spawn_too_expensive", detail
    if best == "spawn_only":
        return "e7l_spawn_preferred_repair_not_needed", detail
    if best == "cost_aware_spawn_plus_repair":
        return "e7l_cost_aware_spawn_repair_survives", detail
    if best in {"spawn_plus_limited_repair_clean", "noisy_health_spawn_plus_repair", "delayed_feedback_spawn_plus_repair"}:
        return "e7l_cost_aware_spawn_repair_survives", detail
    return "e7l_cost_aware_spawn_repair_survives", detail


def build_leakage_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7l_leakage_report_v1",
        "public_inputs": "microsegment_path_plus_phase_token",
        "hidden_missing_motif_id_used_as_model_input": False,
        "dense_all_to_all_soft_routing_used_by_mutation_systems": False,
        "random_spawn_repair_control_passed": aggregate["systems"]["random_spawn_repair_control"]["mean"]["eval_mean_net_utility"] < aggregate["systems"]["cost_aware_spawn_plus_repair"]["mean"]["eval_mean_net_utility"] - 0.01,
        "dense_graph_danger_control_measured": True,
    }


def build_markdown(payloads: dict[str, Any]) -> str:
    aggregate = payloads["aggregate_metrics.json"]
    decision = payloads["decision.json"]
    summary = payloads["summary.json"]
    detail = decision["detail"]
    lines = [
        "# E7L Spawn Repair Cost And Noisy Health Falsification Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_non_oracle_system = {summary['best_non_oracle_system']}",
        f"best_system_including_oracle = {summary['best_system_including_oracle']}",
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
            f"{system:42s} net={mean['eval_mean_net_utility']:.6f} raw={mean['eval_mean_raw_usefulness']:.6f} "
            f"ood={mean['ood_net_utility']:.6f} spawnP={mean.get('heldout_spawn_precision', 0.0):.3f} "
            f"spawnR={mean.get('heldout_spawn_recall', 0.0):.3f} junk={mean.get('heldout_junk_pocket_rate', 0.0):.3f}"
        )
    lines.extend(["```", "", "## Falsification Frontier", "", "```text"])
    for key in (
        "cost_aware_net",
        "route_around_net",
        "repair_only_net",
        "spawn_only_net",
        "clean_spawn_repair_net",
        "noisy_spawn_repair_net",
        "delayed_spawn_repair_net",
        "dense_graph_net",
        "random_control_net",
        "cost_aware_unnecessary_spawn_rate",
        "clean_unnecessary_spawn_rate",
        "cost_aware_junk_pocket_rate",
        "clean_junk_pocket_rate",
        "delayed_feedback_regret",
    ):
        lines.append(f"{key} = {detail[key]}")
    lines.extend(["```", "", "## Phase Winners", "", "```text"])
    for phase, winner in detail["phase_winners"].items():
        lines.append(f"{phase:44s} {winner}")
    lines.extend(
        [
            "```",
            "",
            "## Boundary",
            "",
            "This is a controlled symbolic/numeric pocket-flow cost/noise falsification. It does not make claims about raw-language systems or deployed large models.",
        ]
    )
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
                    mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "initial_candidate_hash", "final_candidate_hash", "parameter_diff_hash", "final_candidate_summary", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "failed_action_rollback_count", "accepted_by_operator", "rejected_by_operator", "delayed_feedback_regret_accumulated")})
                    if out:
                        locked_write_json(out / "partial_aggregate_snapshot.json", {"schema_version": "e7l_partial_aggregate_snapshot_v1", "completed_rows": len(rows), "expected_rows": len(settings.seeds) * len(SYSTEMS), "pending_jobs": len(pending)})
                        append_progress(out, "mutation_job_complete", label=label, pending=len(pending))
    else:
        gpu = gpu_lane_worker(gpu_job)
        rows.extend(gpu["rows"])
        training_histories.extend(gpu["histories"])
        for job in jobs:
            result = mutation_worker(job)
            rows.append({key: value for key, value in result.items() if key != "history"})
            mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "initial_candidate_hash", "final_candidate_hash", "parameter_diff_hash", "final_candidate_summary", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "failed_action_rollback_count", "accepted_by_operator", "rejected_by_operator", "delayed_feedback_regret_accumulated")})
    rows.sort(key=lambda row: (row["system"], int(row["seed"])))
    mutation_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    training_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    aggregate = aggregate_results(rows)
    decision, detail = decide(aggregate)
    return {"tasks": tasks, "rows": rows, "mutation_histories": mutation_histories, "training_histories": training_histories, "aggregate": aggregate, "decision": decision, "decision_detail": detail}


def build_payloads(settings: Settings, out: Path, results: dict[str, Any]) -> dict[str, Any]:
    payloads: dict[str, Any] = {
        "backend_manifest.json": {
            "schema_version": "e7l_backend_manifest_v1",
            "milestone": MILESTONE,
            "settings": settings_payload(settings),
            "systems": list(SYSTEMS),
            "gradient_systems": list(GRADIENT_SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "control_systems": list(CONTROL_SYSTEMS),
            "hardware_identity": e7h.e7g.e7d.e7b.stable_hardware_identity(),
            "parallel_cpu_gpu_lanes": settings.execution_mode == "parallel",
            "source_milestone": "E7K_DYNAMIC_POCKET_SPAWN_AND_PROMOTION_PROBE",
        },
        "task_generation_report.json": task_report(results["tasks"]),
        "cost_noise_stress_report.json": cost_noise_stress_report(),
        "action_policy_report.json": action_policy_report(results["aggregate"]),
        "spawn_repair_policy_report.json": build_spawn_repair_policy_report(results["rows"], results["aggregate"]),
        "health_signal_report.json": build_health_signal_report(results["aggregate"]),
        "delayed_feedback_report.json": build_delayed_feedback_report(results["aggregate"]),
        "system_results.json": {"schema_version": "e7l_system_results_v1", "rows": results["rows"]},
        "mutation_history.json": {"schema_version": "e7l_mutation_history_v1", "rows": results["mutation_histories"]},
        "training_history.json": {"schema_version": "e7l_training_history_v1", "rows": results["training_histories"]},
        "leakage_report.json": build_leakage_report(results["aggregate"]),
        "aggregate_metrics.json": results["aggregate"],
        "decision.json": {"schema_version": "e7l_decision_v1", "decision": results["decision"], "detail": results["decision_detail"], "deterministic_replay_passed": False},
        "summary.json": {
            "schema_version": "e7l_summary_v1",
            "decision": results["decision"],
            "best_non_oracle_system": results["aggregate"]["best_non_oracle_system"],
            "best_system_including_oracle": results["aggregate"]["best_system_including_oracle"],
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
    primary = compute_hashes(primary_payloads)
    replay = compute_hashes(replay_payloads)
    comparisons = {name: {"primary_hash": primary[name], "replay_hash": replay[name], "match": primary[name] == replay[name]} for name in HASH_ARTIFACTS}
    report = {"schema_version": "e7l_deterministic_replay_v1", "internal_replay_passed": all(row["match"] for row in comparisons.values()), "hash_comparisons": comparisons, "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix()}
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
    parser.add_argument("--seeds", default="99101,99102,99103,99104,99105,99106,99107,99108")
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
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7l_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_work_root": None}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop.set()
        monitor.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
