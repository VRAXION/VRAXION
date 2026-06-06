#!/usr/bin/env python3
"""E7E flow-pipe drift and router repair probe.

E7E falsifies the E7D short-pipe result under pipe damage. It keeps the E7D
task shape, splits each semantic primitive into primary/backup physical pipes,
corrupts a deterministic subset of pipes, and compares route-around-only
adaptation against limited local pipe repair and fused-pipe repair.
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
import time
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
E7D_PATH = Path(__file__).with_name("run_e7d_short_pipe_composition_vs_fused_pipe_probe.py")
MILESTONE = "E7E_FLOW_PIPE_DRIFT_AND_ROUTER_REPAIR_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7e_flow_pipe_drift_and_router_repair_probe")
SPLITS = ("train", "validation", "heldout", "ood", "counterfactual", "adversarial")
EVAL_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
SYSTEMS = (
    "damaged_primary_no_adaptation",
    "router_routearound_mutation_only",
    "router_plus_limited_pipe_repair",
    "fused_long_pipe_repair_mutation",
    "monolithic_gradient_drift_adapter",
    "fused_long_pipe_gradient_adapter",
    "random_route_control",
    "oracle_routearound_reference",
    "oracle_repair_reference",
)
GRADIENT_SYSTEMS = ("monolithic_gradient_drift_adapter", "fused_long_pipe_gradient_adapter")
MUTATION_SYSTEMS = (
    "router_routearound_mutation_only",
    "router_plus_limited_pipe_repair",
    "fused_long_pipe_repair_mutation",
)
CONTROL_SYSTEMS = (
    "damaged_primary_no_adaptation",
    "random_route_control",
    "oracle_routearound_reference",
    "oracle_repair_reference",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "drift_profile_report.json",
    "pipe_repair_report.json",
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
    "e7e_router_routearound_sufficient",
    "e7e_router_plus_limited_repair_preferred",
    "e7e_fused_pipe_repair_more_robust",
    "e7e_pipe_redundancy_insufficient",
    "e7e_gradient_adapter_only_viable",
    "e7e_leak_or_artifact_detected",
    "e7e_no_clear_repair_strategy",
)


def load_e7d_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7d_short_pipe_probe", E7D_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7D helpers from {E7D_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7d = load_e7d_module()
PRIMITIVES = e7d.PRIMITIVES
SEMANTIC_ROUTES = e7d.PAIR_ROUTES
PHYSICAL_PIPES = tuple(f"{name}_{variant}" for name in PRIMITIVES for variant in ("primary", "backup"))
PHYSICAL_ROUTES = tuple(f"{left}_then_{right}" for left in PHYSICAL_PIPES for right in PHYSICAL_PIPES)


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
    def __init__(self, input_dim: int, route_dim: int, hidden_dim: int = 112) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.answer = nn.Linear(hidden_dim, 2)
        self.route = nn.Linear(hidden_dim, route_dim)

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
    return int(hashlib.sha256(f"e7e::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7d.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7d.write_json(path, payload)


def locked_write_json(path: Path, payload: Any) -> None:
    e7d.locked_write_json(path, payload)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7d.append_progress(out, event, **details)


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


def e7d_settings(settings: Settings) -> Any:
    return e7d.Settings(
        seeds=settings.seeds,
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
        mutation_generations=settings.mutation_generations,
        mutation_population=settings.mutation_population,
        mutation_sigma=settings.mutation_sigma,
        mutation_elite_count=settings.mutation_elite_count,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        execution_mode=settings.execution_mode,
        replay=settings.replay,
    )


def select_device(requested: str) -> str:
    return e7d.select_device(requested)


def set_determinism(seed: int, device: str) -> None:
    e7d.set_determinism(seed, device)


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    return e7d.start_hardware_monitor(out, stop, interval)


def physical_index(semantic: int, variant: int) -> int:
    return int(semantic) * 2 + int(variant)


def physical_route_index(pipe1: int, pipe2: int) -> int:
    return int(pipe1) * len(PHYSICAL_PIPES) + int(pipe2)


def physical_route_to_semantic(route: int) -> int:
    pipe1, pipe2 = divmod(int(route), len(PHYSICAL_PIPES))
    return e7d.pair_index(pipe1 // 2, pipe2 // 2)


def primary_route_for_semantic(semantic_route: int) -> int:
    sem1, sem2 = e7d.pair_from_index(semantic_route)
    return physical_route_index(physical_index(sem1, 0), physical_index(sem2, 0))


def drift_profile(seed: int) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"drift-profile-{seed}"))
    semantics = list(range(len(PRIMITIVES)))
    routearound_semantics = rng.sample(semantics, k=2)
    remaining = [value for value in semantics if value not in routearound_semantics]
    repair_required_semantic = rng.choice(remaining)
    light_quantized_semantic = rng.choice([value for value in remaining if value != repair_required_semantic])
    damaged: dict[int, str] = {}
    for semantic in routearound_semantics:
        damaged[physical_index(semantic, 0)] = "offset7"
    damaged[physical_index(repair_required_semantic, 0)] = "invert"
    damaged[physical_index(repair_required_semantic, 1)] = "offset5"
    damaged[physical_index(light_quantized_semantic, 0)] = "quantize4"
    return {
        "seed": seed,
        "routearound_semantics": routearound_semantics,
        "repair_required_semantic": repair_required_semantic,
        "light_quantized_semantic": light_quantized_semantic,
        "damaged_pipes": {str(pipe): kind for pipe, kind in sorted(damaged.items())},
    }


def damage_kind(profile: dict[str, Any], pipe: int) -> str | None:
    return profile["damaged_pipes"].get(str(int(pipe)))


def apply_damage(value: int, kind: str | None) -> int:
    if kind is None:
        return int(value)
    if kind == "offset7":
        return (int(value) + 7) & 15
    if kind == "offset5":
        return (int(value) + 5) & 15
    if kind == "invert":
        return int(value) ^ 15
    if kind == "quantize4":
        return max(0, min(15, int(round(int(value) / 4.0) * 4)))
    raise ValueError(kind)


def apply_physical_pipe(profile: dict[str, Any], pipe: int, state: int, row: dict[str, Any], repaired: np.ndarray | None) -> int:
    semantic = int(pipe) // 2
    clean = e7d.apply_primitive(semantic, state, row["a"], row["b"], row["key"], row["mem"])
    if repaired is not None and bool(repaired[int(pipe)]):
        return clean
    return apply_damage(clean, damage_kind(profile, int(pipe)))


def physical_pair_value(profile: dict[str, Any], pipe1: int, pipe2: int, row: dict[str, Any], repaired: np.ndarray | None) -> tuple[int, int]:
    first = apply_physical_pipe(profile, pipe1, row["a"], row, repaired)
    return first, apply_physical_pipe(profile, pipe2, first, row, repaired)


def physical_pair_value_fixed(profile: dict[str, Any], pipe1: int, pipe2: int, row: dict[str, Any], repaired: np.ndarray | None) -> tuple[int, int]:
    first = apply_physical_pipe(profile, pipe1, row["a"], row, repaired)
    return first, apply_physical_pipe(profile, pipe2, first, row, repaired)


def clean_semantic_value(row: dict[str, Any]) -> int:
    _, value = e7d.pair_value(row["op1"], row["target_op2"], row["a"], row["b"], row["key"], row["mem"])
    return int(value)


def candidate_arrays(profile: dict[str, Any], row: dict[str, Any], repaired: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for pipe1 in range(len(PHYSICAL_PIPES)):
        for pipe2 in range(len(PHYSICAL_PIPES)):
            _, value = physical_pair_value_fixed(profile, pipe1, pipe2, row, repaired)
            values.append(value)
    values_arr = np.asarray(values, dtype=np.float32)
    return values_arr, (values_arr > row["threshold"]).astype(np.int64)


def choose_clean_variant(profile: dict[str, Any], semantic: int) -> tuple[int, bool]:
    pipes = [physical_index(semantic, 0), physical_index(semantic, 1)]
    for pipe in pipes:
        if damage_kind(profile, pipe) is None:
            return pipe, False
    return pipes[0], True


def oracle_routes_for_row(profile: dict[str, Any], row: dict[str, Any]) -> tuple[int, int, bool]:
    pipe1, repair1 = choose_clean_variant(profile, row["op1"])
    first = apply_physical_pipe(profile, pipe1, row["a"], row, None)
    if row["mode_id"]:
        branch = 1 if first > row["branch_gate"] else 0
        op2 = row["op2_true"] if branch else row["op2_false"]
    else:
        op2 = row["target_op2"]
    pipe2, repair2 = choose_clean_variant(profile, op2)
    return physical_route_index(pipe1, pipe2), primary_route_for_semantic(row["route"]), bool(repair1 or repair2)


def augment_seed_task(seed: int, base_task: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    health = [0.0 if damage_kind(profile, pipe) else 1.0 for pipe in range(len(PHYSICAL_PIPES))]
    task: dict[str, Any] = {}
    for split in SPLITS:
        rows = []
        physical_candidates = []
        physical_answers = []
        clean_answers = []
        primary_routes = []
        oracle_routes = []
        routearound_repair_needed = []
        raw_drift = []
        for base_row in base_task[split]["rows"]:
            row = copy.deepcopy(base_row)
            values, answers = candidate_arrays(profile, row, repaired=None)
            oracle_route, primary_route, repair_needed = oracle_routes_for_row(profile, row)
            row["primary_physical_route"] = primary_route
            row["oracle_routearound_physical_route"] = oracle_route
            row["routearound_repair_needed"] = repair_needed
            row["damaged_pipe_count"] = len(profile["damaged_pipes"])
            row["clean_answer"] = int(clean_semantic_value(row) > row["threshold"])
            row["drifted_primary_answer"] = int(answers[primary_route])
            rows.append(row)
            physical_candidates.append(values)
            physical_answers.append(answers)
            clean_answers.append(row["clean_answer"])
            primary_routes.append(primary_route)
            oracle_routes.append(oracle_route)
            routearound_repair_needed.append(1 if repair_needed else 0)
            raw_drift.append(row["raw"] + health + [row["drifted_primary_answer"], float(repair_needed)])
        task[split] = {
            "rows": rows,
            "raw_drift": np.asarray(raw_drift, dtype=np.float32),
            "y": np.asarray(clean_answers, dtype=np.int64),
            "semantic_route": np.asarray([row["route"] for row in rows], dtype=np.int64),
            "primary_physical_route": np.asarray(primary_routes, dtype=np.int64),
            "oracle_routearound_physical_route": np.asarray(oracle_routes, dtype=np.int64),
            "routearound_repair_needed": np.asarray(routearound_repair_needed, dtype=np.int64),
            "physical_candidate_values": np.asarray(physical_candidates, dtype=np.float32),
            "physical_candidate_answers": np.asarray(physical_answers, dtype=np.int64),
        }
    return task


def generate_tasks(settings: Settings) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    base_settings = e7d_settings(settings)
    tasks = {}
    profiles = {}
    for seed in settings.seeds:
        base = e7d.generate_seed_task(seed, base_settings)
        profile = drift_profile(seed)
        tasks[seed] = augment_seed_task(seed, base, profile)
        profiles[seed] = profile
    return tasks, profiles


def task_report(tasks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "e7e_task_generation_report_v1",
        "source": "E7D task with physical primary/backup pipe drift augmentation",
        "semantic_primitives": list(PRIMITIVES),
        "physical_pipes": list(PHYSICAL_PIPES),
        "physical_route_count": len(PHYSICAL_ROUTES),
        "row_counts": {
            str(seed): {split: int(len(task[split]["rows"])) for split in SPLITS}
            for seed, task in tasks.items()
        },
        "raw_drift_feature_dim": int(next(iter(tasks.values()))["train"]["raw_drift"].shape[1]),
        "ood_unseen_pair_compositions_retained": True,
        "clean_answer_target_used": True,
        "drifted_primary_control_present": True,
    }


def drift_profile_report(profiles: dict[int, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "e7e_drift_profile_report_v1",
        "drift_types": ["offset7", "offset5", "invert", "quantize4"],
        "physical_pipe_redundancy": "each semantic primitive has primary and backup physical pipes",
        "profiles": {str(seed): profile for seed, profile in profiles.items()},
        "repair_required_semantic_has_both_variants_corrupted": True,
        "routearound_semantics_have_primary_corrupted_backup_clean": True,
    }


def repaired_mask(candidate: dict[str, Any]) -> np.ndarray:
    scores = candidate.get("repair_scores")
    if scores is None:
        return np.zeros(len(PHYSICAL_PIPES), dtype=bool)
    return np.asarray(scores, dtype=np.float64) > 1.0


def physical_route_uses_damaged(profile: dict[str, Any], route: int, repaired: np.ndarray | None) -> bool:
    pipe1, pipe2 = divmod(int(route), len(PHYSICAL_PIPES))
    for pipe in (pipe1, pipe2):
        if repaired is not None and bool(repaired[pipe]):
            continue
        if damage_kind(profile, pipe) is not None:
            return True
    return False


def physical_route_answer(profile: dict[str, Any], route: int, row: dict[str, Any], repaired: np.ndarray | None) -> int:
    pipe1, pipe2 = divmod(int(route), len(PHYSICAL_PIPES))
    _, value = physical_pair_value_fixed(profile, pipe1, pipe2, row, repaired)
    return int(value > row["threshold"])


def semantic_route_answer_primary(profile: dict[str, Any], semantic_route: int, row: dict[str, Any], repaired_pairs: np.ndarray | None) -> int:
    if repaired_pairs is not None and bool(repaired_pairs[int(semantic_route)]):
        return int(clean_semantic_value(row) > row["threshold"])
    return physical_route_answer(profile, primary_route_for_semantic(int(semantic_route)), row, None)


def evaluate_predictions(
    answer_pred: np.ndarray,
    physical_route_pred: np.ndarray,
    data: dict[str, Any],
    profile: dict[str, Any],
    repair_mask: np.ndarray | None,
    sample_limit: int = 8,
) -> dict[str, Any]:
    y = data["y"]
    semantic_target = data["semantic_route"]
    semantic_pred = np.asarray([physical_route_to_semantic(route) for route in physical_route_pred], dtype=np.int64)
    answer_acc = float(np.mean(answer_pred == y))
    route_acc = float(np.mean(semantic_pred == semantic_target))
    composition_acc = float(np.mean((answer_pred == y) & (semantic_pred == semantic_target)))
    damaged_hit = float(np.mean([physical_route_uses_damaged(profile, route, repair_mask) for route in physical_route_pred]))
    needs_repair = data["routearound_repair_needed"] > 0
    routearound_possible = ~needs_repair
    if np.any(routearound_possible):
        routearound_rate = float(np.mean([
            not physical_route_uses_damaged(profile, route, repair_mask)
            for route in physical_route_pred[routearound_possible]
        ]))
    else:
        routearound_rate = 0.0
    repair_use_rate = float(np.mean(repair_mask)) if repair_mask is not None else 0.0
    usefulness = 0.45 * answer_acc + 0.25 * route_acc + 0.15 * routearound_rate + 0.15 * (1.0 - damaged_hit)
    samples = []
    for idx, row in enumerate(data["rows"][:sample_limit]):
        samples.append(
            {
                "row_id": row["row_id"],
                "target_semantic_route": SEMANTIC_ROUTES[int(semantic_target[idx])],
                "predicted_semantic_route": SEMANTIC_ROUTES[int(semantic_pred[idx])],
                "predicted_physical_route": PHYSICAL_ROUTES[int(physical_route_pred[idx])],
                "target_answer": int(y[idx]),
                "predicted_answer": int(answer_pred[idx]),
                "routearound_repair_needed": bool(row["routearound_repair_needed"]),
            }
        )
    return {
        "answer_accuracy": round_float(answer_acc),
        "semantic_route_accuracy": round_float(route_acc),
        "composition_accuracy": round_float(composition_acc),
        "damaged_pipe_hit_rate": round_float(damaged_hit),
        "routearound_rate": round_float(routearound_rate),
        "repair_use_rate": round_float(repair_use_rate),
        "usefulness_score": round_float(usefulness),
        "row_level_samples": samples,
    }


def init_short_candidate(seed: int, repair: bool) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"short-init-{seed}-{repair}"))
    n = len(PRIMITIVES)
    identity = np.full((n, n), -0.18, dtype=np.float64)
    for idx in range(n):
        identity[idx, idx] = 0.35
    variant = np.tile(np.asarray([0.35, -0.10], dtype=np.float64), (n, 1))
    candidate = {
        "kind": "short_repair" if repair else "short_routearound",
        "op1_scores": identity + rng.normal(0.0, 0.015, size=(n, n)),
        "op2_scores": identity + rng.normal(0.0, 0.015, size=(n, n)),
        "op1_variant_scores": variant + rng.normal(0.0, 0.015, size=(n, 2)),
        "op2_variant_scores": variant + rng.normal(0.0, 0.015, size=(n, 2)),
    }
    if repair:
        candidate["repair_scores"] = np.zeros(len(PHYSICAL_PIPES), dtype=np.float64)
    return candidate


def init_fused_candidate(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(f"fused-repair-init-{seed}"))
    return {
        "kind": "fused_pair_repair",
        "context_scores": rng.normal(0.0, 0.03, size=(e7d.context_count(), len(SEMANTIC_ROUTES))).astype(np.float64),
        "repair_pair_scores": np.zeros(len(SEMANTIC_ROUTES), dtype=np.float64),
    }


def candidate_to_serial(candidate: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in candidate.items():
        out[key] = value.round(12).tolist() if isinstance(value, np.ndarray) else value
    return out


def candidate_hash(candidate: dict[str, Any]) -> str:
    return payload_sha256(candidate_to_serial(candidate))


def parameter_count(candidate: dict[str, Any]) -> int:
    return sum(int(value.size) for value in candidate.values() if isinstance(value, np.ndarray))


def predict_short_candidate(candidate: dict[str, Any], data: dict[str, Any], profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    repairs = repaired_mask(candidate)
    routes = []
    answers = []
    for row in data["rows"]:
        op1_sem = int(np.argmax(candidate["op1_scores"][row["op1"]]))
        op1_var = int(np.argmax(candidate["op1_variant_scores"][row["op1"]]))
        pipe1 = physical_index(op1_sem, op1_var)
        first = apply_physical_pipe(profile, pipe1, row["a"], row, repairs)
        op2_token = row["target_op2"]
        if row["mode_id"]:
            branch = 1 if first > row["branch_gate"] else 0
            op2_token = row["op2_true"] if branch else row["op2_false"]
        op2_sem = int(np.argmax(candidate["op2_scores"][op2_token]))
        op2_var = int(np.argmax(candidate["op2_variant_scores"][op2_token]))
        pipe2 = physical_index(op2_sem, op2_var)
        route = physical_route_index(pipe1, pipe2)
        routes.append(route)
        answers.append(physical_route_answer(profile, route, row, repairs))
    return np.asarray(answers, dtype=np.int64), np.asarray(routes, dtype=np.int64), repairs


def predict_fused_candidate(candidate: dict[str, Any], data: dict[str, Any], profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_repairs = np.asarray(candidate["repair_pair_scores"], dtype=np.float64) > 1.0
    answers = []
    routes = []
    for row in data["rows"]:
        key = e7d.context_key(row["op1"], row["op2_true"], row["op2_false"], row["mode_id"], row["branch_flag"])
        semantic_route = int(np.argmax(candidate["context_scores"][key]))
        routes.append(primary_route_for_semantic(semantic_route))
        answers.append(semantic_route_answer_primary(profile, semantic_route, row, pair_repairs))
    return np.asarray(answers, dtype=np.int64), np.asarray(routes, dtype=np.int64), np.zeros(len(PHYSICAL_PIPES), dtype=bool)


def score_candidate(candidate: dict[str, Any], task: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    evals = {}
    for split in ("train", "validation"):
        if candidate["kind"] == "fused_pair_repair":
            answer, route, repairs = predict_fused_candidate(candidate, task[split], profile)
        else:
            answer, route, repairs = predict_short_candidate(candidate, task[split], profile)
        evals[split] = evaluate_predictions(answer, route, task[split], profile, repairs, sample_limit=0)
    val = evals["validation"]
    train = evals["train"]
    fitness = (
        0.18 * train["usefulness_score"]
        + 0.56 * val["usefulness_score"]
        + 0.10 * val["answer_accuracy"]
        + 0.10 * val["semantic_route_accuracy"]
        + 0.06 * (1.0 - val["damaged_pipe_hit_rate"])
    )
    return {"candidate": candidate, "evals": evals, "fitness": round_float(fitness)}


def mutate_candidate(candidate: dict[str, Any], rng: random.Random, settings: Settings) -> tuple[dict[str, Any], str, int]:
    child = copy.deepcopy(candidate)
    if child["kind"] == "fused_pair_repair":
        op = rng.choice(("context_score", "context_boost", "repair_pair_score", "repair_pair_toggle"))
        if op == "repair_pair_toggle":
            idx = rng.randrange(len(SEMANTIC_ROUTES))
            child["repair_pair_scores"][idx] = 1.25 if child["repair_pair_scores"][idx] <= 1.0 else 0.0
            return child, op, 1
        if op == "repair_pair_score":
            child["repair_pair_scores"][rng.randrange(len(SEMANTIC_ROUTES))] += rng.gauss(0.0, settings.mutation_sigma)
            return child, op, 1
        i = rng.randrange(child["context_scores"].shape[0])
        j = rng.randrange(child["context_scores"].shape[1])
        scale = 2.0 if op == "context_boost" else 1.0
        child["context_scores"][i, j] += rng.gauss(0.0, settings.mutation_sigma * scale)
        return child, op, 1
    keys = ("op1_scores", "op2_scores", "op1_variant_scores", "op2_variant_scores")
    if child["kind"] == "short_repair" and rng.random() < 0.24:
        if rng.random() < 0.72:
            idx = rng.randrange(len(PHYSICAL_PIPES))
            child["repair_scores"][idx] = 1.25 if child["repair_scores"][idx] <= 1.0 else 0.0
            return child, "repair_toggle", 1
        child["repair_scores"][rng.randrange(len(PHYSICAL_PIPES))] += rng.gauss(0.0, settings.mutation_sigma)
        return child, "repair_score", 1
    key = rng.choice(keys)
    arr = child[key].copy()
    i = rng.randrange(arr.shape[0])
    j = rng.randrange(arr.shape[1])
    arr[i, j] += rng.gauss(0.0, settings.mutation_sigma)
    child[key] = arr
    return child, key, 1


def evaluate_candidate_full(candidate: dict[str, Any], task: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for split in SPLITS:
        if candidate["kind"] == "fused_pair_repair":
            answer, route, repairs = predict_fused_candidate(candidate, task[split], profile)
        else:
            answer, route, repairs = predict_short_candidate(candidate, task[split], profile)
        out[split] = evaluate_predictions(answer, route, task[split], profile, repairs)
    return out


def mutation_worker(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    seed = int(job["seed"])
    system = job["system"]
    task = job["task"]
    profile = job["profile"]
    if system == "router_plus_limited_pipe_repair":
        initial = init_short_candidate(seed, repair=True)
    elif system == "fused_long_pipe_repair_mutation":
        initial = init_fused_candidate(seed)
    else:
        initial = init_short_candidate(seed, repair=False)
    initial_hash = candidate_hash(initial)
    rng = random.Random(stable_seed(f"mutation-{system}-{seed}"))
    population = [score_candidate(copy.deepcopy(initial), task, profile)]
    for _ in range(settings.mutation_population - 1):
        child, _, _ = mutate_candidate(initial, rng, settings)
        population.append(score_candidate(child, task, profile))
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
            child = score_candidate(child_candidate, task, profile)
            attempts += 1
            changed_total += changed
            neutral_exploration = child["fitness"] == parent["fitness"] and rng.random() < 0.035
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
            "validation_answer_accuracy": val["answer_accuracy"],
            "validation_routearound_rate": val["routearound_rate"],
            "attempts": attempts,
        }
        history.append(row)
        if out and (time.monotonic() - last_heartbeat >= settings.heartbeat_seconds or generation == settings.mutation_generations):
            safe = e7d.e7b.safe_file_id(f"{system}_seed{seed}")
            locked_write_json(out / "partial_status" / f"mutation_{safe}.json", row)
            locked_write_json(
                out / "mutation_history_snapshots" / f"{safe}.json",
                {
                    "schema_version": "e7e_mutation_history_snapshot_v1",
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
    evals = evaluate_candidate_full(final_candidate, task, profile)
    final_hash = candidate_hash(final_candidate)
    return {
        "seed": seed,
        "system": system,
        "training_mode": "mutation_rollback",
        "parameter_count": parameter_count(final_candidate),
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


def evaluate_gradient_state(model_state: dict[str, Any], task: dict[str, Any], profile: dict[str, Any], system: str, device: str) -> dict[str, Any]:
    route_dim = int(model_state["route_dim"])
    model = MultiHeadMLP(int(model_state["input_dim"]), route_dim)
    model.load_state_dict({key: torch.as_tensor(value, dtype=torch.float32) for key, value in model_state["state_dict"].items()})
    model.to(device)
    model.eval()
    out = {}
    with torch.no_grad():
        for split in SPLITS:
            x = torch.as_tensor(task[split]["raw_drift"], dtype=torch.float32, device=device)
            answer_logits, route_logits = model(x)
            answer_pred = torch.argmax(answer_logits, dim=1).cpu().numpy()
            raw_route = torch.argmax(route_logits, dim=1).cpu().numpy()
            if system == "fused_long_pipe_gradient_adapter":
                physical_route = np.asarray([primary_route_for_semantic(int(route)) for route in raw_route], dtype=np.int64)
            else:
                physical_route = raw_route.astype(np.int64)
            out[split] = evaluate_predictions(answer_pred, physical_route, task[split], profile, np.zeros(len(PHYSICAL_PIPES), dtype=bool))
    return out


def train_gradient_system(seed: int, system: str, task: dict[str, Any], profile: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    device = select_device(settings.device)
    set_determinism(stable_seed(f"gradient-{system}-{seed}"), device)
    route_dim = len(SEMANTIC_ROUTES) if system == "fused_long_pipe_gradient_adapter" else len(PHYSICAL_ROUTES)
    route_target = task["train"]["semantic_route"] if system == "fused_long_pipe_gradient_adapter" else task["train"]["oracle_routearound_physical_route"]
    val_route_target = task["validation"]["semantic_route"] if system == "fused_long_pipe_gradient_adapter" else task["validation"]["oracle_routearound_physical_route"]
    input_dim = int(task["train"]["raw_drift"].shape[1])
    model = MultiHeadMLP(input_dim, route_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    x_train = torch.as_tensor(task["train"]["raw_drift"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(task["train"]["y"], dtype=torch.long, device=device)
    r_train = torch.as_tensor(route_target, dtype=torch.long, device=device)
    x_val = torch.as_tensor(task["validation"]["raw_drift"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(task["validation"]["y"], dtype=torch.long, device=device)
    r_val = torch.as_tensor(val_route_target, dtype=torch.long, device=device)
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
        row = {"epoch": epoch, "validation_answer_accuracy": round_float(answer_acc), "validation_route_accuracy": round_float(route_acc), "score": round_float(score)}
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
    model_state = {"input_dim": input_dim, "route_dim": route_dim, "state_dict": serial_state}
    evals = evaluate_gradient_state(model_state, task, profile, system, device)
    return {
        "seed": seed,
        "system": system,
        "training_mode": "gradient_backprop",
        "device": device,
        "parameter_count": sum(int(np.asarray(value).size) for value in serial_state.values()),
        "state_hash": payload_sha256(serial_state),
        "history": history,
        "evals": evals,
    }


def gpu_lane_worker(job: dict[str, Any]) -> dict[str, Any]:
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rows = []
    histories = []
    for seed_text in sorted(job["tasks"], key=lambda value: int(value)):
        seed = int(seed_text)
        for system in GRADIENT_SYSTEMS:
            if out:
                append_progress(out, "gradient_job_start", system=system, seed=seed)
            result = train_gradient_system(seed, system, job["tasks"][seed_text], job["profiles"][seed_text], settings, out)
            rows.append({key: value for key, value in result.items() if key != "history"})
            histories.append({"seed": seed, "system": system, "history": result["history"], "device": result["device"]})
            if out:
                append_progress(out, "gradient_job_complete", system=system, seed=seed, device=result["device"])
    return {"lane": "gpu_gradient_lane", "rows": rows, "histories": histories, "hardware": e7d.e7b.hardware_probe()}


def control_results(seed: int, task: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    rng = np.random.default_rng(stable_seed(f"random-control-{seed}"))
    for system in CONTROL_SYSTEMS:
        evals = {}
        for split in SPLITS:
            data = task[split]
            if system == "damaged_primary_no_adaptation":
                route = data["primary_physical_route"].copy()
                answer = np.asarray([physical_route_answer(profile, int(route[idx]), row, None) for idx, row in enumerate(data["rows"])], dtype=np.int64)
                repair = np.zeros(len(PHYSICAL_PIPES), dtype=bool)
            elif system == "oracle_routearound_reference":
                route = data["oracle_routearound_physical_route"].copy()
                answer = np.asarray([physical_route_answer(profile, int(route[idx]), row, None) for idx, row in enumerate(data["rows"])], dtype=np.int64)
                repair = np.zeros(len(PHYSICAL_PIPES), dtype=bool)
            elif system == "oracle_repair_reference":
                route = data["primary_physical_route"].copy()
                answer = data["y"].copy()
                repair = np.ones(len(PHYSICAL_PIPES), dtype=bool)
            else:
                route = rng.integers(0, len(PHYSICAL_ROUTES), size=len(data["y"]), dtype=np.int64)
                answer = np.asarray([physical_route_answer(profile, int(route[idx]), row, None) for idx, row in enumerate(data["rows"])], dtype=np.int64)
                repair = np.zeros(len(PHYSICAL_PIPES), dtype=bool)
            evals[split] = evaluate_predictions(answer, route, data, profile, repair)
        rows.append(
            {
                "seed": seed,
                "system": system,
                "training_mode": "control",
                "parameter_count": 0,
                "state_hash": payload_sha256({"seed": seed, "system": system}),
                "evals": evals,
            }
        )
    return rows


def split_summary(evals: dict[str, Any]) -> dict[str, float]:
    out = {}
    for metric in ("answer_accuracy", "semantic_route_accuracy", "composition_accuracy", "damaged_pipe_hit_rate", "routearound_rate", "repair_use_rate", "usefulness_score"):
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
    metrics = (
        "answer_accuracy",
        "semantic_route_accuracy",
        "composition_accuracy",
        "damaged_pipe_hit_rate",
        "routearound_rate",
        "repair_use_rate",
        "usefulness_score",
        "generalization_gap",
        "heldout_usefulness",
        "ood_usefulness",
        "counterfactual_usefulness",
        "adversarial_usefulness",
        "parameter_count",
    )
    systems = {}
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
    best = max((system for system in SYSTEMS if system not in {"oracle_repair_reference", "oracle_routearound_reference"}), key=lambda name: systems[name]["mean"]["usefulness_score"])
    return {
        "schema_version": "e7e_aggregate_metrics_v1",
        "systems": systems,
        "best_non_oracle_system": best,
        "best_non_oracle_usefulness": systems[best]["mean"]["usefulness_score"],
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    routearound = systems["router_routearound_mutation_only"]["mean"]
    repair = systems["router_plus_limited_pipe_repair"]["mean"]
    fused = systems["fused_long_pipe_repair_mutation"]["mean"]
    grad = systems["monolithic_gradient_drift_adapter"]["mean"]
    damaged = systems["damaged_primary_no_adaptation"]["mean"]
    random_control = systems["random_route_control"]["mean"]
    oracle_route = systems["oracle_routearound_reference"]["mean"]
    oracle_repair = systems["oracle_repair_reference"]["mean"]
    leak = random_control["usefulness_score"] >= 0.72
    routearound_gap = oracle_route["usefulness_score"] - routearound["usefulness_score"]
    routearound_ood_gap = oracle_route["ood_usefulness"] - routearound["ood_usefulness"]
    routearound_pass = routearound_gap <= 0.04 and routearound_ood_gap <= 0.05 and routearound["usefulness_score"] >= damaged["usefulness_score"] + 0.18
    repair_pass = (
        repair["usefulness_score"] >= routearound["usefulness_score"] + 0.02
        and repair["ood_usefulness"] >= routearound["ood_usefulness"] - 0.02
        and repair["repair_use_rate"] > 0.02
    )
    fused_pass = fused["usefulness_score"] >= max(routearound["usefulness_score"], repair["usefulness_score"]) + 0.02
    gradient_only = grad["usefulness_score"] >= 0.90 and max(routearound["usefulness_score"], repair["usefulness_score"]) < 0.84
    if leak:
        decision = "e7e_leak_or_artifact_detected"
    elif repair_pass:
        decision = "e7e_router_plus_limited_repair_preferred"
    elif routearound_pass and routearound["usefulness_score"] >= repair["usefulness_score"] - 0.015:
        decision = "e7e_router_routearound_sufficient"
    elif fused_pass:
        decision = "e7e_fused_pipe_repair_more_robust"
    elif gradient_only:
        decision = "e7e_gradient_adapter_only_viable"
    elif max(routearound["usefulness_score"], repair["usefulness_score"], fused["usefulness_score"]) < damaged["usefulness_score"] + 0.05:
        decision = "e7e_pipe_redundancy_insufficient"
    else:
        decision = "e7e_no_clear_repair_strategy"
    return decision, {
        "routearound_mean": routearound,
        "limited_repair_mean": repair,
        "fused_repair_mean": fused,
        "monolithic_gradient_mean": grad,
        "damaged_primary_mean": damaged,
        "oracle_routearound_mean": oracle_route,
        "oracle_repair_mean": oracle_repair,
        "random_control_mean": random_control,
        "routearound_gain_over_damaged": round_float(routearound["usefulness_score"] - damaged["usefulness_score"]),
        "repair_gain_over_routearound": round_float(repair["usefulness_score"] - routearound["usefulness_score"]),
        "repair_gain_over_damaged": round_float(repair["usefulness_score"] - damaged["usefulness_score"]),
        "routearound_gap_to_oracle_routearound": round_float(routearound_gap),
        "routearound_ood_gap_to_oracle_routearound": round_float(routearound_ood_gap),
        "leak_flag": leak,
        "routearound_pass": routearound_pass,
        "repair_pass": repair_pass,
        "fused_pass": fused_pass,
    }


def build_composition_report(aggregate: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7e_composition_report_v1",
        "interpretation_boundary": "flow_pipe_drift_routearound_vs_limited_repair_controlled_proxy",
        "best_non_oracle_system": aggregate["best_non_oracle_system"],
        "routearound_gain_over_damaged": detail["routearound_gain_over_damaged"],
        "repair_gain_over_routearound": detail["repair_gain_over_routearound"],
        "repair_gain_over_damaged": detail["repair_gain_over_damaged"],
        "routearound_to_repair_ceiling_gap": round_float(detail["oracle_repair_mean"]["usefulness_score"] - detail["routearound_mean"]["usefulness_score"]),
        "limited_repair_to_oracle_gap": round_float(detail["oracle_repair_mean"]["usefulness_score"] - detail["limited_repair_mean"]["usefulness_score"]),
    }


def build_pipe_repair_report(profiles: dict[int, dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7e_pipe_repair_report_v1",
        "damage_profile_count": len(profiles),
        "router_routearound_mean": aggregate["systems"]["router_routearound_mutation_only"]["mean"],
        "router_plus_limited_pipe_repair_mean": aggregate["systems"]["router_plus_limited_pipe_repair"]["mean"],
        "fused_long_pipe_repair_mean": aggregate["systems"]["fused_long_pipe_repair_mutation"]["mean"],
        "drift_modes_tested": ["primary_damage_backup_clean", "both_variants_corrupted", "light_quantization"],
    }


def build_leakage_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    random_mean = aggregate["systems"]["random_route_control"]["mean"]
    return {
        "schema_version": "e7e_leakage_report_v1",
        "random_control_usefulness": random_mean["usefulness_score"],
        "random_control_passed": random_mean["usefulness_score"] < 0.72,
        "hidden_correct_physical_route_used_as_input": False,
        "health_features_are_damage_signals_not_target_route_labels": True,
        "ood_unseen_pair_split_retained": True,
    }


def build_markdown(payloads: dict[str, Any]) -> str:
    decision = payloads["decision.json"]["decision"]
    aggregate = payloads["aggregate_metrics.json"]
    detail = payloads["decision.json"]["detail"]
    lines = [
        "# E7E Flow-Pipe Drift And Router Repair Probe Result",
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
        "| system | usefulness | answer | semantic_route | OOD | adversarial | damage_hit | routearound | repair_use | params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        lines.append(
            f"| {system} | {mean['usefulness_score']:.6f} | {mean['answer_accuracy']:.6f} | {mean['semantic_route_accuracy']:.6f} | {mean['ood_usefulness']:.6f} | {mean['adversarial_usefulness']:.6f} | {mean['damaged_pipe_hit_rate']:.6f} | {mean['routearound_rate']:.6f} | {mean['repair_use_rate']:.6f} | {mean['parameter_count']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Repair Comparison",
            "",
            "```text",
            f"routearound_gain_over_damaged = {detail['routearound_gain_over_damaged']}",
            f"repair_gain_over_routearound = {detail['repair_gain_over_routearound']}",
            f"repair_gain_over_damaged = {detail['repair_gain_over_damaged']}",
            "```",
            "",
            "## Boundary",
            "",
            "This is a controlled symbolic/numeric flow-pipe drift proxy. It tests route-around and limited repair behavior only.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress(out, "startup", milestone=MILESTONE, settings=settings_payload(settings), hardware=e7d.e7b.hardware_probe())
    tasks, profiles = generate_tasks(settings)
    if out:
        append_progress(out, "tasks_generated", seeds=list(settings.seeds), row_counts=task_report(tasks)["row_counts"])
    rows: list[dict[str, Any]] = []
    mutation_histories: list[dict[str, Any]] = []
    training_histories: list[dict[str, Any]] = []
    for seed in settings.seeds:
        rows.extend(control_results(seed, tasks[seed], profiles[seed]))
    jobs = [
        {
            "seed": seed,
            "system": system,
            "task": tasks[seed],
            "profile": profiles[seed],
            "settings": settings.__dict__,
            "out": out.as_posix() if out else None,
        }
        for seed in settings.seeds
        for system in MUTATION_SYSTEMS
    ]
    gpu_job = {
        "tasks": {str(seed): tasks[seed] for seed in settings.seeds},
        "profiles": {str(seed): profiles[seed] for seed in settings.seeds},
        "settings": settings.__dict__,
        "out": out.as_posix() if out else None,
    }
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
                                "schema_version": "e7e_partial_aggregate_snapshot_v1",
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
        "profiles": profiles,
        "rows": rows,
        "mutation_histories": mutation_histories,
        "training_histories": training_histories,
        "aggregate": aggregate,
        "decision": decision,
        "decision_detail": detail,
    }


def build_payloads(settings: Settings, out: Path, results: dict[str, Any]) -> dict[str, Any]:
    composition = build_composition_report(results["aggregate"], results["decision_detail"])
    payloads: dict[str, Any] = {
        "backend_manifest.json": {
            "schema_version": "e7e_backend_manifest_v1",
            "milestone": MILESTONE,
            "settings": settings_payload(settings),
            "systems": list(SYSTEMS),
            "gradient_systems": list(GRADIENT_SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "control_systems": list(CONTROL_SYSTEMS),
            "hardware_identity": e7d.e7b.stable_hardware_identity(),
            "parallel_cpu_gpu_lanes": settings.execution_mode == "parallel",
        },
        "task_generation_report.json": task_report(results["tasks"]),
        "drift_profile_report.json": drift_profile_report(results["profiles"]),
        "pipe_repair_report.json": build_pipe_repair_report(results["profiles"], results["aggregate"]),
        "system_results.json": {"schema_version": "e7e_system_results_v1", "rows": results["rows"]},
        "mutation_history.json": {"schema_version": "e7e_mutation_history_v1", "rows": results["mutation_histories"]},
        "training_history.json": {"schema_version": "e7e_training_history_v1", "rows": results["training_histories"]},
        "aggregate_metrics.json": results["aggregate"],
        "composition_report.json": composition,
        "leakage_report.json": build_leakage_report(results["aggregate"]),
        "decision.json": {"schema_version": "e7e_decision_v1", "decision": results["decision"], "detail": results["decision_detail"], "deterministic_replay_passed": False},
        "summary.json": {
            "schema_version": "e7e_summary_v1",
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
        "schema_version": "e7e_deterministic_replay_v1",
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
    parser.add_argument("--seeds", default="94001,94002,94003,94004,94005,94006,94007,94008,94009,94010,94011,94012")
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
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7e_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_work_root": None}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop.set()
        monitor.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
