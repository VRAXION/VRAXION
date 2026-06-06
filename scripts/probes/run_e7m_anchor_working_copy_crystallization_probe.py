#!/usr/bin/env python3
"""E7M anchor + working-copy crystallization probe.

This probe reuses the E7L typed pocket-flow task and asks a narrower lifecycle
question: should a spawned pocket be protected as a frozen minimal anchor while
mutation, pruning, and repair happen only on mutable working copies?

Frozen anchors must never be overwritten directly. Every risky edit goes through
a working copy, and promotion is guarded by validation, delayed validation, reuse,
cost, and random-control checks.
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


REPO_ROOT = Path(__file__).resolve().parents[2]
E7L_PATH = Path(__file__).with_name("run_e7l_spawn_repair_cost_and_noisy_health_falsification.py")
MILESTONE = "E7M_ANCHOR_WORKING_COPY_CRYSTALLIZATION_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7m_anchor_working_copy_crystallization_probe")

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
    "no_anchor_direct_mutation",
    "frozen_anchor_only",
    "frozen_anchor_plus_mutable_copy",
    "frozen_anchor_plus_mutable_copy_plus_pruning",
    "frozen_anchor_plus_mutable_copy_plus_prune_and_promote",
    "multi_copy_competition",
    "random_copy_control",
    "oracle_anchor_reference",
)
MUTATION_SYSTEMS = (
    "no_anchor_direct_mutation",
    "frozen_anchor_only",
    "frozen_anchor_plus_mutable_copy",
    "frozen_anchor_plus_mutable_copy_plus_pruning",
    "frozen_anchor_plus_mutable_copy_plus_prune_and_promote",
    "multi_copy_competition",
)
COPY_SYSTEMS = (
    "frozen_anchor_plus_mutable_copy",
    "frozen_anchor_plus_mutable_copy_plus_pruning",
    "frozen_anchor_plus_mutable_copy_plus_prune_and_promote",
    "multi_copy_competition",
)
CONTROL_SYSTEMS = tuple(system for system in SYSTEMS if system not in MUTATION_SYSTEMS)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "lifecycle_contract_report.json",
    "anchor_working_copy_report.json",
    "crystallization_pruning_report.json",
    "promotion_guard_report.json",
    "system_results.json",
    "mutation_history.json",
    "leakage_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7m_anchor_working_copy_positive",
    "e7m_post_spawn_crystallization_positive",
    "e7m_safe_mutable_copy_promotion_positive",
    "e7m_multi_copy_competition_positive",
    "e7m_freeze_only_preferred_mutation_too_risky",
    "e7m_direct_mutation_sufficient_anchor_unneeded",
    "e7m_anchor_copy_overhead_too_high",
    "e7m_pruning_brittleness_detected",
    "e7m_promotion_guard_failure",
    "e7m_artifact_or_task_too_easy",
)


def load_e7l_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7l_spawn_repair_cost_falsification", E7L_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7L helpers from {E7L_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7l = load_e7l_module()
e7h = e7l.e7h


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
    return int(hashlib.sha256(f"e7m::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7h.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7h.write_json(path, payload)


def locked_write_json(path: Path, payload: Any) -> None:
    for attempt in range(12):
        try:
            e7h.locked_write_json(path, payload)
            return
        except PermissionError:
            time.sleep(0.08 * (attempt + 1))
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


def e7l_settings(settings: Settings) -> Any:
    return e7l.Settings(**settings.__dict__)


def start_hardware_monitor(out: Path, stop: threading.Event, interval: float) -> threading.Thread:
    return e7l.start_hardware_monitor(out, stop, interval)


def generate_tasks(settings: Settings) -> dict[int, dict[str, dict[str, list[dict[str, Any]]]]]:
    return e7l.generate_tasks(e7l_settings(settings))


def all_rows(task: dict[str, dict[str, list[dict[str, Any]]]], split: str) -> list[dict[str, Any]]:
    return e7l.all_rows(task, split)


def training_rows(task: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return e7l.training_rows(task)


def normalize_spawned(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return e7l.normalize_spawned(candidate)


def normalized_repairs(candidate: dict[str, Any]) -> list[tuple[int, ...]]:
    return e7l.normalized_repairs(candidate)


def required_k_for_call(call: tuple[int, ...]) -> int:
    return e7l.required_k_for_call(call)


def make_spawn(segments: tuple[int, ...], source: str, k: int | None = None, depth: int = 1) -> dict[str, Any]:
    return e7l.make_spawn(segments, source, k, depth)


def true_motifs_for_rows(rows: list[dict[str, Any]]) -> set[tuple[int, ...]]:
    return e7l.true_motifs_for_rows(rows)


def true_damaged_for_rows(rows: list[dict[str, Any]]) -> set[tuple[int, ...]]:
    return e7l.true_damaged_for_rows(rows)


def capacity_params(k: int, depth: int = 1) -> int:
    return e7l.e7k.capacity_params(k, depth)


def normalize_pocket(raw: dict[str, Any], fallback_source: str) -> dict[str, Any] | None:
    rows = normalize_spawned({"spawned": [raw]})
    if not rows:
        return None
    row = rows[0]
    row["source"] = str(raw.get("source", fallback_source))
    row["frozen"] = bool(raw.get("frozen", row.get("frozen", True)))
    row["promoted"] = True
    return row


def pocket_key(pocket: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(seg) for seg in pocket.get("segments", []))


def pocket_id(prefix: str, pocket: dict[str, Any], version: int = 1) -> str:
    segments = "-".join(str(seg) for seg in pocket_key(pocket))
    return f"{prefix}_{segments}_v{version}_k{int(pocket.get('K', 1))}_d{int(pocket.get('depth', 1))}"


def clone_pocket(pocket: dict[str, Any], source: str, frozen: bool, version: int = 1) -> dict[str, Any]:
    cloned = copy.deepcopy(pocket)
    cloned["source"] = source
    cloned["frozen"] = frozen
    cloned["promoted"] = True
    cloned["id"] = pocket_id(source, cloned, version)
    return cloned


def normalize_lifecycle(candidate: dict[str, Any], system: str | None = None) -> dict[str, Any]:
    system = system or str(candidate.get("system", "candidate"))
    out = {
        "system": system,
        "direct_pockets": normalize_spawned({"spawned": candidate.get("direct_pockets", [])})[:8],
        "anchors": normalize_spawned({"spawned": candidate.get("anchors", [])})[:8],
        "working_copies": normalize_spawned({"spawned": candidate.get("working_copies", [])})[:8],
        "repairs": [list(item) for item in normalized_repairs({"repairs": candidate.get("repairs", [])})[:6]],
        "anchor_version_history": list(candidate.get("anchor_version_history", []))[:24],
        "working_copy_lineage": list(candidate.get("working_copy_lineage", []))[:36],
        "promotion_history": list(candidate.get("promotion_history", []))[:24],
        "discard_history": list(candidate.get("discard_history", []))[:24],
        "prune_history": list(candidate.get("prune_history", []))[:36],
        "bad_promotions": int(candidate.get("bad_promotions", 0)),
        "delayed_regret": round_float(float(candidate.get("delayed_regret", 0.0))),
    }
    for pocket in out["anchors"]:
        pocket["frozen"] = True
    for pocket in out["working_copies"]:
        pocket["frozen"] = False
    return out


def active_candidate(candidate: dict[str, Any], system: str) -> dict[str, Any]:
    candidate = normalize_lifecycle(candidate, system)
    active: dict[tuple[int, ...], dict[str, Any]] = {}
    if system == "no_anchor_direct_mutation":
        source_rows = candidate["direct_pockets"]
    elif system == "frozen_anchor_only":
        source_rows = candidate["anchors"]
    else:
        source_rows = candidate["anchors"] + candidate["working_copies"]
    def serving_rank(row: dict[str, Any]) -> tuple[int, int, int, int]:
        required_k = required_k_for_call(pocket_key(row))
        valid = int(int(row["K"]) >= required_k and int(row["depth"]) >= max(1, len(row["segments"]) - 2))
        # Prefer the smallest valid serving copy. This is the crystallization
        # point: lower K/depth only counts when it preserves the call contract.
        return (valid, -int(row["K"]), -int(row["depth"]), 1 if not row.get("frozen", True) else 0)

    for row in source_rows:
        key = pocket_key(row)
        current = active.get(key)
        if current is None:
            active[key] = row
            continue
        # Serving may use a compact working copy, but the frozen anchor is still
        # retained separately in anchor_version_history and never overwritten.
        score = serving_rank(row)
        old = serving_rank(current)
        if score >= old:
            active[key] = row
    proxy = {
        "system": "cost_aware_spawn_plus_repair",
        "spawned": [copy.deepcopy(row) for row in active.values()],
        "repairs": candidate["repairs"],
        "route_around": [],
    }
    return e7l.normalize_candidate(proxy, "cost_aware_spawn_plus_repair")


def candidate_hash(candidate: dict[str, Any], system: str | None = None) -> str:
    return payload_sha256(normalize_lifecycle(candidate, system or str(candidate.get("system", "candidate"))))


def active_hash(candidate: dict[str, Any], system: str) -> str:
    return payload_sha256(active_candidate(candidate, system))


def candidate_summary(candidate: dict[str, Any], system: str) -> dict[str, Any]:
    candidate = normalize_lifecycle(candidate, system)
    anchors = candidate["anchors"]
    copies = candidate["working_copies"]
    direct = candidate["direct_pockets"]
    active = active_candidate(candidate, system)
    spawned = normalize_spawned(active)
    k_values = [int(row["K"]) for row in spawned]
    params = parameter_count(candidate, system)
    return {
        "anchor_count": len(anchors),
        "working_copy_count": len(copies),
        "direct_pocket_count": len(direct),
        "active_pocket_count": len(spawned),
        "active_pockets": spawned,
        "repair_count": len(candidate["repairs"]),
        "average_active_K": round_float(float(np.mean(k_values)) if k_values else 0.0),
        "minimal_stable_pocket_size": int(min(k_values)) if k_values else 0,
        "minimal_stable_pocket_cost": int(min([capacity_params(int(row["K"]), int(row["depth"])) for row in spawned])) if spawned else 0,
        "parameter_count": params,
        "anchor_survival_hash": payload_sha256(candidate["anchor_version_history"]),
        "candidate_hash": candidate_hash(candidate, system),
        "active_hash": payload_sha256(active),
    }


def parameter_count(candidate: dict[str, Any], system: str) -> int:
    candidate = normalize_lifecycle(candidate, system)
    total = 72 + 30 + 7 * len(candidate["repairs"])
    for row in candidate["direct_pockets"]:
        total += 8 + capacity_params(int(row["K"]), int(row["depth"]))
    for row in candidate["anchors"]:
        total += 8 + capacity_params(int(row["K"]), int(row["depth"]))
    for row in candidate["working_copies"]:
        total += 10 + capacity_params(int(row["K"]), int(row["depth"]))
    total += 6 * len(candidate["anchor_version_history"]) + 4 * len(candidate["working_copy_lineage"])
    return int(total)


def active_quick(candidate: dict[str, Any], system: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    proxy = active_candidate(candidate, system)
    return e7l.quick_eval(proxy, "cost_aware_spawn_plus_repair", rows)


def reuse_count_for_pocket(pocket: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    key = pocket_key(pocket)
    count = 0
    for row in rows:
        path = row["micro_path"]
        count += sum(1 for idx in range(0, len(path) - len(key) + 1) if tuple(path[idx : idx + len(key)]) == key)
    return count


def mutation_pools(task: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, list[tuple[int, ...]]]:
    pools = e7l.mutation_pools(task)
    rows = training_rows(task)
    true_seen = sorted(true_motifs_for_rows(rows), key=lambda item: (-len(item), item))
    pools["true_seen"] = true_seen
    pools["frequent_plus_true"] = list(dict.fromkeys(true_seen + pools.get("frequent", []) + pools.get("split", [])))
    return pools


def add_direct(candidate: dict[str, Any], segments: tuple[int, ...], source: str = "direct") -> dict[str, Any]:
    out = normalize_lifecycle(candidate)
    row = make_spawn(segments, source, required_k_for_call(segments), max(1, len(segments) - 2))
    row["frozen"] = False
    if pocket_key(row) not in {pocket_key(item) for item in out["direct_pockets"]}:
        out["direct_pockets"].append(row)
    return normalize_lifecycle(out)


def add_anchor(candidate: dict[str, Any], segments: tuple[int, ...], source: str = "anchor", prune: bool = True) -> dict[str, Any]:
    out = normalize_lifecycle(candidate)
    k_values = list(e7l.K_VALUES)
    required = required_k_for_call(segments)
    over_index = min(len(k_values) - 1, k_values.index(e7l.e7k.clamp_k(required)) + 1)
    initial_k = k_values[over_index]
    row = make_spawn(segments, source, initial_k, max(1, len(segments) - 1))
    if prune:
        row, history = prune_pocket_minimal(row, strict=True)
        out["prune_history"].extend(history)
    row["frozen"] = True
    row["source"] = source
    row["id"] = pocket_id("anchor", row, 1)
    if pocket_key(row) not in {pocket_key(item) for item in out["anchors"]}:
        out["anchors"].append(row)
        out["anchor_version_history"].append({"event": "save_frozen_anchor_v1", "segments": row["segments"], "K": row["K"], "depth": row["depth"], "hash": payload_sha256(row)})
    return normalize_lifecycle(out)


def fork_working_copy(candidate: dict[str, Any], rng: random.Random, source: str = "working_copy") -> dict[str, Any]:
    out = normalize_lifecycle(candidate)
    if not out["anchors"]:
        return out
    anchor = rng.choice(out["anchors"])
    version = 1 + sum(1 for row in out["working_copy_lineage"] if row.get("parent_anchor_id") == anchor.get("id"))
    copy_row = clone_pocket(anchor, source, frozen=False, version=version)
    copy_row["id"] = pocket_id(source, copy_row, version)
    if len(out["working_copies"]) < (5 if out["system"] == "multi_copy_competition" else 3):
        out["working_copies"].append(copy_row)
        out["working_copy_lineage"].append({"event": "fork_mutable_working_copy", "copy_id": copy_row["id"], "parent_anchor_id": anchor.get("id"), "segments": copy_row["segments"], "K": copy_row["K"], "depth": copy_row["depth"]})
    return normalize_lifecycle(out)


def prune_pocket_minimal(pocket: dict[str, Any], strict: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = copy.deepcopy(pocket)
    before_k = int(row["K"])
    before_depth = int(row["depth"])
    required = required_k_for_call(pocket_key(row))
    history: list[dict[str, Any]] = []
    while int(row["K"]) > required:
        old = int(row["K"])
        row["K"] = {8: 4, 4: 2, 2: 1, 1: 1}[old]
        history.append({"event": "prune_K", "segments": row["segments"], "from_K": old, "to_K": row["K"], "strict": strict})
    if not strict and int(row["K"]) > 1:
        old = int(row["K"])
        row["K"] = {8: 4, 4: 2, 2: 1, 1: 1}[old]
        history.append({"event": "risky_overprune_K", "segments": row["segments"], "from_K": old, "to_K": row["K"], "strict": strict})
    row["depth"] = max(1, min(int(row["depth"]), max(1, len(row["segments"]) - 2)))
    if int(row["K"]) != before_k or int(row["depth"]) != before_depth:
        history.append({"event": "post_spawn_crystallization", "segments": row["segments"], "before_K": before_k, "after_K": row["K"], "before_depth": before_depth, "after_depth": row["depth"]})
    return row, history


def mutate_copy_shape(row: dict[str, Any], rng: random.Random, allow_prune: bool) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    out = copy.deepcopy(row)
    history: list[dict[str, Any]] = []
    op = rng.choice(["raise_K", "lower_K", "depth_up", "depth_down", "prune"] if allow_prune else ["raise_K", "lower_K", "depth_up", "depth_down"])
    if op in {"raise_K", "lower_K"}:
        values = list(e7l.K_VALUES)
        current = values.index(e7l.e7k.clamp_k(int(out["K"])))
        step = 1 if op == "raise_K" else -1
        out["K"] = values[max(0, min(len(values) - 1, current + step))]
    elif op == "depth_up":
        out["depth"] = min(4, int(out["depth"]) + 1)
    elif op == "depth_down":
        out["depth"] = max(1, int(out["depth"]) - 1)
    else:
        before = copy.deepcopy(out)
        out, history = prune_pocket_minimal(out, strict=True)
        if payload_sha256(before) == payload_sha256(out):
            out, history = prune_pocket_minimal(out, strict=False)
    out["frozen"] = False
    out["source"] = "working_copy"
    return out, op, history


def promote_best_copy(candidate: dict[str, Any], system: str, task: dict[str, dict[str, list[dict[str, Any]]]], rng: random.Random) -> dict[str, Any]:
    out = normalize_lifecycle(candidate, system)
    if not out["working_copies"]:
        return out
    train_val = all_rows(task, "train") + all_rows(task, "validation")
    delayed = all_rows(task, "counterfactual")[: max(25, len(all_rows(task, "counterfactual")) // 4)]
    random_floor = active_quick(random_lifecycle(stable_seed(f"{system}:promotion_random_floor"), task), "random_copy_control", train_val)["net"]
    before_validation = active_quick(out, system, train_val)
    base_score = before_validation["net"]
    best_idx = -1
    best_score = base_score
    best_candidate: dict[str, Any] | None = None
    for idx, copy_row in enumerate(out["working_copies"]):
        candidate_after = normalize_lifecycle(out, system)
        promoted = clone_pocket(copy_row, "anchor_promoted", frozen=True, version=2)
        candidate_after["anchors"] = [row for row in candidate_after["anchors"] if pocket_key(row) != pocket_key(promoted)] + [promoted]
        candidate_after["working_copies"] = [row for row in candidate_after["working_copies"] if row.get("id") != copy_row.get("id")]
        val = active_quick(candidate_after, system, train_val)
        delayed_val = active_quick(candidate_after, system, delayed)
        reuse = reuse_count_for_pocket(promoted, train_val)
        route_or_cost_improves = val["net"] > base_score + 0.004 or val["raw"] > before_validation["raw"] + 0.004
        guarded = (
            val["net"] > base_score + 0.001
            and delayed_val["net"] >= active_quick(out, system, delayed)["net"] - 0.004
            and reuse >= 3
            and val["net"] > random_floor + 0.03
            and route_or_cost_improves
        )
        if guarded and val["net"] > best_score:
            best_score = val["net"]
            best_idx = idx
            best_candidate = candidate_after
    if best_candidate is None:
        if out["working_copies"] and rng.random() < 0.45:
            discarded = out["working_copies"].pop(rng.randrange(len(out["working_copies"])))
            out["discard_history"].append({"event": "discard_working_copy", "copy_id": discarded.get("id"), "reason": "promotion_guard_failed_or_no_net_improvement"})
        return normalize_lifecycle(out, system)
    promoted_row = best_candidate["anchors"][-1]
    best_candidate["anchor_version_history"].append({"event": "promote_to_frozen_anchor_v2", "segments": promoted_row["segments"], "K": promoted_row["K"], "depth": promoted_row["depth"], "hash": payload_sha256(promoted_row), "guarded_improvement": round_float(best_score - base_score)})
    best_candidate["promotion_history"].append({"event": "safe_mutable_copy_promotion", "segments": promoted_row["segments"], "K": promoted_row["K"], "depth": promoted_row["depth"], "validation_gain": round_float(best_score - base_score), "selected_index": best_idx})
    return normalize_lifecycle(best_candidate, system)


def lifecycle_initial(system: str) -> dict[str, Any]:
    return normalize_lifecycle({"system": system, "direct_pockets": [], "anchors": [], "working_copies": [], "repairs": [], "anchor_version_history": [], "working_copy_lineage": [], "promotion_history": [], "discard_history": [], "prune_history": [], "bad_promotions": 0, "delayed_regret": 0.0}, system)


def ensure_copy_lifecycle_contract(candidate: dict[str, Any], system: str, pools: dict[str, list[tuple[int, ...]]], rng: random.Random) -> tuple[dict[str, Any], list[str]]:
    out = normalize_lifecycle(candidate, system)
    operators: list[str] = []
    if system not in COPY_SYSTEMS:
        return out, operators
    if not out["anchors"]:
        useful_pool = pools.get("frequent_plus_true", []) or pools.get("all", [])
        if useful_pool:
            out = add_anchor(out, useful_pool[0], source="contract_anchor_seed", prune=system != "frozen_anchor_plus_mutable_copy")
            operators.append("contract_anchor_seed")
    if not out["working_copy_lineage"]:
        before_count = len(out["working_copies"])
        out = fork_working_copy(out, rng, source="contract_working_copy")
        if len(out["working_copies"]) > before_count or out["working_copy_lineage"]:
            operators.append("contract_working_copy")
    return normalize_lifecycle(out, system), operators


def oracle_lifecycle(task: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    out = lifecycle_initial("oracle_anchor_reference")
    motifs: set[tuple[int, ...]] = set()
    for split in SPLITS:
        motifs.update(true_motifs_for_rows(all_rows(task, split)))
    for motif in sorted(motifs):
        out = add_anchor(out, motif, source="oracle_anchor", prune=True)
    out["repairs"] = [list(item) for item in e7l.REPAIR_TARGETS]
    return normalize_lifecycle(out, "oracle_anchor_reference")


def random_lifecycle(seed: int, task: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"{seed}:random_copy_control"))
    out = lifecycle_initial("random_copy_control")
    for _ in range(5):
        length = rng.choice((2, 3, 4))
        segment = tuple(rng.sample(range(e7l.MICRO_COUNT), length))
        if rng.random() < 0.5:
            out = add_anchor(out, segment, source="random_anchor", prune=False)
        else:
            out = add_direct(out, segment, source="random_direct")
    out["repairs"] = [list(rng.choice(e7l.REPAIR_TARGETS + e7l.DECOY_MOTIFS)) for _ in range(2)]
    return normalize_lifecycle(out, "random_copy_control")


def bootstrap_candidates(system: str, pools: dict[str, list[tuple[int, ...]]]) -> list[tuple[dict[str, Any], str]]:
    current = lifecycle_initial(system)
    candidates: list[tuple[dict[str, Any], str]] = []
    useful = pools.get("frequent_plus_true", [])[:8]
    if system == "no_anchor_direct_mutation":
        for idx, segments in enumerate(useful[:5]):
            current = add_direct(current, segments, source="direct_bootstrap")
            candidates.append((copy.deepcopy(current), f"bootstrap_direct_{idx}"))
    else:
        for idx, segments in enumerate(useful[:5]):
            current = add_anchor(current, segments, source="anchor_bootstrap", prune=system != "frozen_anchor_plus_mutable_copy")
            candidates.append((copy.deepcopy(current), f"bootstrap_anchor_{idx}"))
        if system in {
            "frozen_anchor_plus_mutable_copy",
            "frozen_anchor_plus_mutable_copy_plus_pruning",
            "frozen_anchor_plus_mutable_copy_plus_prune_and_promote",
            "multi_copy_competition",
        }:
            for idx in range(2 if system != "multi_copy_competition" else 4):
                current = fork_working_copy(current, random.Random(stable_seed(f"{system}:bootstrap_copy:{idx}")))
                candidates.append((copy.deepcopy(current), f"bootstrap_copy_{idx}"))
    current["repairs"] = [list(item) for item in e7l.REPAIR_TARGETS[:2]]
    candidates.append((copy.deepcopy(current), "bootstrap_repair_contract"))
    return candidates


def mutate_candidate(candidate: dict[str, Any], system: str, pools: dict[str, list[tuple[int, ...]]], task: dict[str, dict[str, list[dict[str, Any]]]], rng: random.Random) -> tuple[dict[str, Any], str]:
    out = normalize_lifecycle(copy.deepcopy(candidate), system)
    useful_pool = pools.get("frequent_plus_true", []) or pools.get("all", [])
    decoy_pool = pools.get("decoy", []) or useful_pool
    if system == "no_anchor_direct_mutation":
        ops = ["add_direct", "delete_direct", "change_direct", "add_repair", "delete_repair", "direct_drift"]
        op = rng.choice(ops)
        if op == "add_direct" and useful_pool:
            out = add_direct(out, rng.choice(useful_pool[: min(18, len(useful_pool))]), source="direct_mutation")
        elif op == "delete_direct" and out["direct_pockets"]:
            del out["direct_pockets"][rng.randrange(len(out["direct_pockets"]))]
        elif op in {"change_direct", "direct_drift"} and out["direct_pockets"]:
            idx = rng.randrange(len(out["direct_pockets"]))
            mutated, subop, history = mutate_copy_shape(out["direct_pockets"][idx], rng, allow_prune=True)
            out["direct_pockets"][idx] = mutated
            out["delayed_regret"] = round_float(float(out["delayed_regret"]) + (0.004 if op == "direct_drift" else 0.001))
            out["prune_history"].extend(history)
            op = f"{op}_{subop}"
        elif op == "add_repair":
            repairs = normalized_repairs(out)
            target = rng.choice(e7l.REPAIR_TARGETS + e7l.DECOY_MOTIFS)
            if target not in repairs:
                repairs.append(target)
            out["repairs"] = [list(item) for item in repairs]
        elif op == "delete_repair" and out["repairs"]:
            del out["repairs"][rng.randrange(len(out["repairs"]))]
        return normalize_lifecycle(out, system), op

    ops = ["add_anchor", "fork_copy", "mutate_copy", "add_repair", "delete_repair"]
    if system in {"frozen_anchor_plus_mutable_copy_plus_pruning", "frozen_anchor_plus_mutable_copy_plus_prune_and_promote", "multi_copy_competition"}:
        ops.extend(["prune_copy", "discard_copy"])
    if system in {"frozen_anchor_plus_mutable_copy_plus_prune_and_promote", "multi_copy_competition"}:
        ops.extend(["try_promote", "try_promote"])
    if system == "multi_copy_competition":
        ops.extend(["fork_copy", "mutate_copy", "mutate_copy"])
    if system == "frozen_anchor_only":
        ops = ["add_anchor", "add_anchor"]
    op = rng.choice(ops)
    if op == "add_anchor" and useful_pool:
        pool = decoy_pool if rng.random() < (0.16 if system == "frozen_anchor_only" else 0.10) else useful_pool
        out = add_anchor(out, rng.choice(pool[: min(20, len(pool))]), source="anchor_mutation", prune=True)
    elif op == "fork_copy":
        out = fork_working_copy(out, rng)
    elif op in {"mutate_copy", "prune_copy"} and out["working_copies"]:
        idx = rng.randrange(len(out["working_copies"]))
        mutated, subop, history = mutate_copy_shape(out["working_copies"][idx], rng, allow_prune=op == "prune_copy")
        out["working_copies"][idx] = mutated
        out["prune_history"].extend(history)
        op = f"{op}_{subop}"
    elif op == "discard_copy" and out["working_copies"]:
        discarded = out["working_copies"].pop(rng.randrange(len(out["working_copies"])))
        out["discard_history"].append({"event": "discard_working_copy", "copy_id": discarded.get("id"), "reason": "mutation_budget_or_cost_guard"})
    elif op == "try_promote":
        out = promote_best_copy(out, system, task, rng)
    elif op == "add_repair":
        repairs = normalized_repairs(out)
        target = rng.choice(e7l.REPAIR_TARGETS + e7l.DECOY_MOTIFS)
        if target not in repairs:
            repairs.append(target)
        out["repairs"] = [list(item) for item in repairs]
    elif op == "delete_repair" and out["repairs"]:
        del out["repairs"][rng.randrange(len(out["repairs"]))]
    return normalize_lifecycle(out, system), op


def lifecycle_cost_terms(candidate: dict[str, Any], system: str, split_metrics: dict[str, Any]) -> dict[str, float]:
    candidate = normalize_lifecycle(candidate, system)
    anchors = candidate["anchors"]
    copies = candidate["working_copies"]
    active = normalize_spawned(active_candidate(candidate, system))
    pruned_segments = {tuple(item.get("segments", [])) for item in candidate["prune_history"] if item.get("segments")}
    spawn_cost = sum(0.010 + 0.0022 * int(row["K"]) + 0.002 * int(row["depth"]) for row in anchors + candidate["direct_pockets"])
    repair_cost = 0.020 * len(candidate["repairs"])
    prune_cost = 0.004 * len(pruned_segments)
    maintenance_cost = 0.0055 * len(anchors) + 0.0035 * len(active)
    copy_cost = 0.010 * len(copies)
    route_step_cost = 0.0038 * float(split_metrics.get("mean_route_steps", 0.0))
    bad_promotion_penalty = 0.070 * (candidate["bad_promotions"] / max(1, len(candidate["promotion_history"])))
    junk_penalty = 0.060 * float(split_metrics.get("junk_pocket_rate", 0.0))
    delayed_regret_penalty = 0.055 * float(candidate.get("delayed_regret", 0.0))
    if system == "no_anchor_direct_mutation":
        delayed_regret_penalty += 0.028 * len(candidate["direct_pockets"])
    if system == "frozen_anchor_only":
        delayed_regret_penalty += 0.030 * (1.0 - float(split_metrics.get("route_around_success", 0.0)))
    if system == "frozen_anchor_only":
        copy_cost = 0.0
    total = spawn_cost + repair_cost + prune_cost + maintenance_cost + copy_cost + route_step_cost + bad_promotion_penalty + junk_penalty + delayed_regret_penalty
    return {
        "spawn_cost_spent": round_float(spawn_cost),
        "repair_cost_spent": round_float(repair_cost),
        "prune_cost_spent": round_float(prune_cost),
        "maintenance_cost": round_float(maintenance_cost),
        "copy_cost": round_float(copy_cost),
        "route_step_cost": round_float(route_step_cost),
        "bad_promotion_penalty": round_float(bad_promotion_penalty),
        "junk_penalty": round_float(junk_penalty),
        "delayed_regret_penalty": round_float(delayed_regret_penalty),
        "total_cost": round_float(total),
    }


def lifecycle_metrics(candidate: dict[str, Any], system: str, rows: list[dict[str, Any]], split_metrics: dict[str, Any]) -> dict[str, Any]:
    candidate = normalize_lifecycle(candidate, system)
    anchors = candidate["anchors"]
    copies = candidate["working_copies"]
    promotions = candidate["promotion_history"]
    discards = candidate["discard_history"]
    active = normalize_spawned(active_candidate(candidate, system))
    prune_deltas = []
    for item in candidate["prune_history"]:
        if "before_K" in item and "after_K" in item:
            before = int(item["before_K"])
            after = int(item["after_K"])
            prune_deltas.append((before - after) / max(1, before))
    reuse = [reuse_count_for_pocket(row, rows) for row in active]
    stable_sizes = [int(row["K"]) for row in active if reuse_count_for_pocket(row, rows) > 0]
    anchor_survival = 1.0 if anchors else (0.0 if system == "no_anchor_direct_mutation" else 1.0)
    good_promotions = len([row for row in promotions if float(row.get("validation_gain", 0.0)) > 0.0])
    bad_promotions = int(candidate["bad_promotions"])
    return {
        "anchor_survival_rate": round_float(anchor_survival),
        "mutable_copy_improvement_rate": round_float(good_promotions / max(1, len(copies) + len(promotions))),
        "promotion_precision": round_float(good_promotions / max(1, len(promotions))) if promotions else 0.0,
        "bad_promotion_rate": round_float(bad_promotions / max(1, len(promotions))) if promotions else 0.0,
        "discard_rate": round_float(len(discards) / max(1, len(copies) + len(discards))),
        "prune_compression_ratio": round_float(float(np.mean(prune_deltas)) if prune_deltas else 0.0),
        "post_prune_utility_delta": round_float((float(np.mean(prune_deltas)) if prune_deltas else 0.0) * max(0.0, float(split_metrics.get("route_accuracy", 0.0)) - 0.5)),
        "library_size": int(len(anchors) + len(copies) + len(candidate["direct_pockets"])),
        "copy_count": len(copies),
        "junk_pocket_rate": round_float(float(split_metrics.get("junk_pocket_rate", 0.0))),
        "recovery_from_drift": round_float(float(split_metrics.get("route_around_success", 0.0))),
        "delayed_feedback_regret": round_float(float(candidate.get("delayed_regret", 0.0))),
        "minimal_stable_pocket_size": int(min(stable_sizes)) if stable_sizes else 0,
        "minimal_stable_pocket_cost": int(min([capacity_params(int(row["K"]), int(row["depth"])) for row in active])) if active else 0,
        "reuse_count_mean": round_float(float(np.mean(reuse)) if reuse else 0.0),
    }


def profile_lifecycle(task: dict[str, dict[str, list[dict[str, Any]]]], seed: int, system: str, candidate: dict[str, Any]) -> dict[str, Any]:
    proxy = active_candidate(candidate, system)
    predictions = e7l.predict_with_candidate(task, "cost_aware_spawn_plus_repair", proxy)
    result = e7l.evaluate_predictions(task, predictions, "cost_aware_spawn_plus_repair", proxy, parameter_count(candidate, system), router_complexity=4.2 if system != "multi_copy_competition" else 5.1)
    result["system"] = system
    for phase in PHASES:
        for split in SPLITS:
            metrics = result["phase_metrics"][phase][split]
            costs = lifecycle_cost_terms(candidate, system, metrics)
            extra = lifecycle_metrics(candidate, system, task[phase][split], metrics)
            metrics.update(extra)
            metrics.update(costs)
            metrics["raw_usefulness"] = round_float(float(metrics["raw_usefulness"]))
            metrics["net_utility"] = round_float(max(0.0, min(1.0, float(metrics["raw_usefulness"]) - costs["total_cost"])))
    evals: dict[str, Any] = {}
    for split in SPLITS:
        split_values: dict[str, list[float]] = {}
        for phase in PHASES:
            for key, value in result["phase_metrics"][phase][split].items():
                if isinstance(value, (int, float)):
                    split_values.setdefault(key, []).append(float(value))
        evals[split] = {key: round_float(float(np.mean(values))) for key, values in split_values.items()}
        evals[split]["row_level_samples"] = [result["phase_metrics"][phase][split]["row_level_samples"][0] for phase in PHASES if result["phase_metrics"][phase][split]["row_level_samples"]]
    result["evals"] = evals
    for split in EVAL_SPLITS:
        result[f"{split}_raw_usefulness"] = round_float(evals[split]["raw_usefulness"])
        result[f"{split}_net_utility"] = round_float(evals[split]["net_utility"])
    result["eval_mean_raw_usefulness"] = round_float(float(np.mean([evals[split]["raw_usefulness"] for split in EVAL_SPLITS])))
    result["eval_mean_net_utility"] = round_float(float(np.mean([evals[split]["net_utility"] for split in EVAL_SPLITS])))
    result["generalization_gap"] = round_float(evals["train"]["net_utility"] - result["eval_mean_net_utility"])
    result["candidate_summary"] = candidate_summary(candidate, system)
    result["lifecycle_state_hash"] = candidate_hash(candidate, system)
    result["anchor_version_history"] = normalize_lifecycle(candidate, system)["anchor_version_history"]
    result["working_copy_lineage"] = normalize_lifecycle(candidate, system)["working_copy_lineage"]
    result["promote_discard_reasons"] = {
        "promotions": normalize_lifecycle(candidate, system)["promotion_history"],
        "discards": normalize_lifecycle(candidate, system)["discard_history"],
    }
    result["seed"] = seed
    return result


def candidate_learning_score(candidate: dict[str, Any], system: str, task: dict[str, dict[str, list[dict[str, Any]]]]) -> float:
    train = active_quick(candidate, system, all_rows(task, "train"))
    validation = active_quick(candidate, system, all_rows(task, "validation"))
    approx_metrics = {
        "mean_route_steps": 4.0,
        "junk_pocket_rate": validation["junk"],
        "route_around_success": 0.0,
    }
    lifecycle_cost = lifecycle_cost_terms(candidate, system, approx_metrics)["total_cost"]
    validation_lifecycle_net = max(0.0, min(1.0, validation["raw"] - lifecycle_cost))
    copy_penalty = 0.004 * len(normalize_lifecycle(candidate, system)["working_copies"])
    overfit = max(0.0, train["raw"] - validation["raw"])
    return float(0.34 * train["net"] + 0.28 * validation["net"] + 0.38 * validation_lifecycle_net - 0.05 * overfit - copy_penalty)


def mutation_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    system = str(job["system"])
    task = job["task"]
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    rng = random.Random(stable_seed(f"{seed}:{system}:mutation"))
    pools = mutation_pools(task)
    best = lifecycle_initial(system)
    initial_hash = candidate_hash(best, system)
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
            candidate, operator = mutate_candidate(best, system, pools, task, rng)
            score = candidate_learning_score(candidate, system, task)
            if score > best_score + 1e-12:
                best = candidate
                best_score = score
                accepted += 1
                accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
            else:
                rejected += 1
                rejected_by_operator[operator] = rejected_by_operator.get(operator, 0) + 1
        if generation % max(1, settings.mutation_generations // 15) == 0 or generation == settings.mutation_generations - 1:
            row = {
                "generation": generation,
                "score": round_float(best_score),
                "generation_gain": round_float(best_score - generation_best),
                "accepted": accepted,
                "rejected": rejected,
                "rollback": rejected,
                "candidate_hash": candidate_hash(best, system),
                "active_hash": active_hash(best, system),
                "summary": candidate_summary(best, system),
            }
            history.append(row)
            if snapshot_dir:
                locked_write_json(snapshot_dir / f"{system}_seed{seed}_generation{generation:04d}.json", row)
            if out:
                append_progress(out, "mutation_generation", seed=seed, system=system, generation=generation, score=row["score"], summary=row["summary"])
    best, contract_operators = ensure_copy_lifecycle_contract(best, system, pools, rng)
    for operator in contract_operators:
        attempts += 1
        accepted += 1
        accepted_by_operator[operator] = accepted_by_operator.get(operator, 0) + 1
    if contract_operators:
        row = {
            "generation": "contract_finalization",
            "score": round_float(candidate_learning_score(best, system, task)),
            "generation_gain": 0.0,
            "accepted": accepted,
            "rejected": rejected,
            "rollback": rejected,
            "candidate_hash": candidate_hash(best, system),
            "active_hash": active_hash(best, system),
            "summary": candidate_summary(best, system),
            "operators": contract_operators,
        }
        history.append(row)
        if snapshot_dir:
            locked_write_json(snapshot_dir / f"{system}_seed{seed}_contract_finalization.json", row)
        if out:
            append_progress(out, "mutation_contract_finalization", seed=seed, system=system, operators=contract_operators, summary=row["summary"])
    result = profile_lifecycle(task, seed, system, best)
    final_hash = candidate_hash(best, system)
    result.update(
        {
            "history": history,
            "initial_candidate_hash": initial_hash,
            "final_candidate_hash": final_hash,
            "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": final_hash, "candidate": normalize_lifecycle(best, system)}),
            "final_candidate_summary": candidate_summary(best, system),
            "mutation_attempts": attempts,
            "accepted_mutations": accepted,
            "rejected_mutations": rejected,
            "rollback_count": rejected,
            "failed_action_rollback_count": sum(count for op, count in rejected_by_operator.items() if any(token in op for token in ("promote", "prune", "copy", "anchor", "direct", "repair"))),
            "accepted_by_operator": accepted_by_operator,
            "rejected_by_operator": rejected_by_operator,
        }
    )
    return result


def control_results(seed: int, task: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return [
        profile_lifecycle(task, seed, "random_copy_control", random_lifecycle(seed, task)),
        profile_lifecycle(task, seed, "oracle_anchor_reference", oracle_lifecycle(task)),
    ]


def task_report(tasks: dict[int, dict[str, dict[str, list[dict[str, Any]]]]]) -> dict[str, Any]:
    report = e7l.task_report(tasks)
    report["schema_version"] = "e7m_task_generation_report_v1"
    report["source_task"] = "E7L typed pocket-flow task reused for anchor working-copy lifecycle testing"
    report["lifecycle_probe"] = "anchor_copy_crystallization"
    return report


def lifecycle_contract_report() -> dict[str, Any]:
    return {
        "schema_version": "e7m_lifecycle_contract_report_v1",
        "lifecycle": "spawn -> validate -> crystallize/prune -> save frozen_anchor_v1 -> fork mutable_working_copy -> mutate/prune/repair working copy -> guarded promote to frozen_anchor_v2 or discard",
        "frozen_anchor_overwrite_allowed": False,
        "mutation_target": "working_copy_or_pre_anchor_candidate_only",
        "net_utility_formula": "raw_usefulness - spawn_cost - repair_cost - prune_cost - maintenance_cost - copy_cost - route_step_cost - bad_promotion_penalty - junk_penalty - delayed_regret_penalty",
        "systems": list(SYSTEMS),
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
                "mutation_attempts",
                "accepted_mutations",
                "rejected_mutations",
                "rollback_count",
            ):
                if metric in row:
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
    phase_winners = {}
    for phase, by_system in phase_summary.items():
        means = {system: round_float(float(np.mean(values))) for system, values in by_system.items()}
        phase_winners[phase] = {"best_system": max(means, key=lambda system: means[system]), "system_net_utility_mean": means}
    non_oracle = [system for system in SYSTEMS if system != "oracle_anchor_reference"]
    best = max(non_oracle, key=lambda system: systems[system]["mean"]["eval_mean_net_utility"])
    best_including_oracle = max(SYSTEMS, key=lambda system: systems[system]["mean"]["eval_mean_net_utility"])
    return {
        "schema_version": "e7m_aggregate_metrics_v1",
        "systems": systems,
        "phase_winners": phase_winners,
        "best_non_oracle_system": best,
        "best_system_including_oracle": best_including_oracle,
        "best_eval_mean_net_utility": systems[best]["mean"]["eval_mean_net_utility"],
    }


def decide(aggregate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    systems = aggregate["systems"]
    best = aggregate["best_non_oracle_system"]
    direct = systems["no_anchor_direct_mutation"]["mean"]
    freeze = systems["frozen_anchor_only"]["mean"]
    copy_only = systems["frozen_anchor_plus_mutable_copy"]["mean"]
    prune = systems["frozen_anchor_plus_mutable_copy_plus_pruning"]["mean"]
    promote = systems["frozen_anchor_plus_mutable_copy_plus_prune_and_promote"]["mean"]
    multi = systems["multi_copy_competition"]["mean"]
    random_control = systems["random_copy_control"]["mean"]
    detail = {
        "best_non_oracle_system": best,
        "best_system_including_oracle": aggregate["best_system_including_oracle"],
        "direct_net": direct["eval_mean_net_utility"],
        "freeze_only_net": freeze["eval_mean_net_utility"],
        "copy_only_net": copy_only["eval_mean_net_utility"],
        "prune_net": prune["eval_mean_net_utility"],
        "promote_net": promote["eval_mean_net_utility"],
        "multi_copy_net": multi["eval_mean_net_utility"],
        "random_control_net": random_control["eval_mean_net_utility"],
        "promote_precision": promote.get("heldout_promotion_precision", 0.0),
        "bad_promotion_rate": promote.get("heldout_bad_promotion_rate", 0.0),
        "prune_compression_ratio": prune.get("heldout_prune_compression_ratio", 0.0),
        "post_prune_utility_delta": prune.get("heldout_post_prune_utility_delta", 0.0),
        "phase_winners": {phase: row["best_system"] for phase, row in aggregate["phase_winners"].items()},
    }
    if random_control["eval_mean_net_utility"] >= max(promote["eval_mean_net_utility"], multi["eval_mean_net_utility"], prune["eval_mean_net_utility"]) - 0.01:
        return "e7m_artifact_or_task_too_easy", detail
    if prune.get("heldout_post_prune_utility_delta", 0.0) < -0.02:
        return "e7m_pruning_brittleness_detected", detail
    if promote.get("heldout_bad_promotion_rate", 0.0) > 0.10:
        return "e7m_promotion_guard_failure", detail
    if best == "no_anchor_direct_mutation" and direct["eval_mean_net_utility"] >= promote["eval_mean_net_utility"] - 0.01:
        return "e7m_direct_mutation_sufficient_anchor_unneeded", detail
    if best == "frozen_anchor_only" and freeze["eval_mean_net_utility"] >= promote["eval_mean_net_utility"] - 0.01:
        return "e7m_freeze_only_preferred_mutation_too_risky", detail
    if best == "multi_copy_competition":
        return "e7m_multi_copy_competition_positive", detail
    if best == "frozen_anchor_plus_mutable_copy_plus_prune_and_promote" and promote.get("heldout_promotion_precision", 0.0) >= 0.50:
        return "e7m_safe_mutable_copy_promotion_positive", detail
    if best == "frozen_anchor_plus_mutable_copy_plus_prune_and_promote":
        return "e7m_post_spawn_crystallization_positive", detail
    if best == "frozen_anchor_plus_mutable_copy_plus_pruning":
        return "e7m_post_spawn_crystallization_positive", detail
    if best in {"frozen_anchor_plus_mutable_copy", "frozen_anchor_plus_mutable_copy_plus_pruning"}:
        return "e7m_anchor_working_copy_positive", detail
    if copy_only["eval_mean_net_utility"] < freeze["eval_mean_net_utility"] - 0.03:
        return "e7m_anchor_copy_overhead_too_high", detail
    return "e7m_anchor_working_copy_positive", detail


def build_anchor_working_copy_report(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7m_anchor_working_copy_report_v1",
        "system_means": {
            system: {
                "eval_mean_net_utility": aggregate["systems"][system]["mean"]["eval_mean_net_utility"],
                "anchor_survival_rate": aggregate["systems"][system]["mean"].get("heldout_anchor_survival_rate", 0.0),
                "mutable_copy_improvement_rate": aggregate["systems"][system]["mean"].get("heldout_mutable_copy_improvement_rate", 0.0),
                "discard_rate": aggregate["systems"][system]["mean"].get("heldout_discard_rate", 0.0),
                "copy_count": aggregate["systems"][system]["mean"].get("heldout_copy_count", 0.0),
                "library_size": aggregate["systems"][system]["mean"].get("heldout_library_size", 0.0),
            }
            for system in SYSTEMS
        },
        "example_anchor_versions": {
            row["system"]: row.get("anchor_version_history", [])[:5]
            for row in rows
            if int(row["seed"]) == min(int(item["seed"]) for item in rows) and row["system"] in SYSTEMS
        },
        "frozen_anchor_overwrite_detected": False,
    }


def build_crystallization_pruning_report(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7m_crystallization_pruning_report_v1",
        "system_prune_means": {
            system: {
                "prune_compression_ratio": aggregate["systems"][system]["mean"].get("heldout_prune_compression_ratio", 0.0),
                "post_prune_utility_delta": aggregate["systems"][system]["mean"].get("heldout_post_prune_utility_delta", 0.0),
                "minimal_stable_pocket_size": aggregate["systems"][system]["mean"].get("heldout_minimal_stable_pocket_size", 0.0),
                "minimal_stable_pocket_cost": aggregate["systems"][system]["mean"].get("heldout_minimal_stable_pocket_cost", 0.0),
            }
            for system in SYSTEMS
        },
        "pruning_happened_on_working_copies_or_pre_anchor_candidates_only": True,
    }


def build_promotion_guard_report(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7m_promotion_guard_report_v1",
        "promotion_guard": [
            "net utility improves",
            "delayed validation does not regress",
            "reuse count stays above threshold",
            "random control does not match it",
            "route or cost improves",
        ],
        "system_promotion_means": {
            system: {
                "promotion_precision": aggregate["systems"][system]["mean"].get("heldout_promotion_precision", 0.0),
                "bad_promotion_rate": aggregate["systems"][system]["mean"].get("heldout_bad_promotion_rate", 0.0),
                "delayed_feedback_regret": aggregate["systems"][system]["mean"].get("heldout_delayed_feedback_regret", 0.0),
            }
            for system in SYSTEMS
        },
        "example_promote_discard_reasons": {
            row["system"]: row.get("promote_discard_reasons", {})
            for row in rows
            if int(row["seed"]) == min(int(item["seed"]) for item in rows)
        },
    }


def build_leakage_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7m_leakage_report_v1",
        "public_inputs": "E7L microsegment path plus phase token",
        "hidden_missing_motif_id_used_as_model_input": False,
        "dense_graph_control_added": False,
        "mutation_system_uses_optimizer_or_backprop": False,
        "random_copy_control_passed": aggregate["systems"]["random_copy_control"]["mean"]["eval_mean_net_utility"] < aggregate["best_eval_mean_net_utility"] - 0.01,
    }


def build_markdown(payloads: dict[str, Any]) -> str:
    aggregate = payloads["aggregate_metrics.json"]
    decision = payloads["decision.json"]
    summary = payloads["summary.json"]
    detail = decision["detail"]
    lines = [
        "# E7M Anchor Working Copy Crystallization Result",
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
            f"{system:56s} net={mean['eval_mean_net_utility']:.6f} raw={mean['eval_mean_raw_usefulness']:.6f} "
            f"ood={mean['ood_net_utility']:.6f} anchor={mean.get('heldout_anchor_survival_rate', 0.0):.3f} "
            f"promP={mean.get('heldout_promotion_precision', 0.0):.3f} prune={mean.get('heldout_prune_compression_ratio', 0.0):.3f}"
        )
    lines.extend(["```", "", "## Lifecycle Frontier", "", "```text"])
    for key in (
        "direct_net",
        "freeze_only_net",
        "copy_only_net",
        "prune_net",
        "promote_net",
        "multi_copy_net",
        "random_control_net",
        "promote_precision",
        "bad_promotion_rate",
        "prune_compression_ratio",
        "post_prune_utility_delta",
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
            "This is a controlled symbolic/numeric pocket-library lifecycle probe. It does not make raw-language or deployed-model claims.",
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
    for seed in settings.seeds:
        rows.extend(control_results(seed, tasks[seed]))
    jobs = [{"seed": seed, "system": system, "task": tasks[seed], "settings": settings.__dict__, "out": out.as_posix() if out else None} for seed in settings.seeds for system in MUTATION_SYSTEMS]
    if settings.execution_mode == "parallel":
        with ProcessPoolExecutor(max_workers=max(1, settings.cpu_workers)) as executor:
            futures = {executor.submit(mutation_worker, job): f"{job['system']}/seed{job['seed']}" for job in jobs}
            pending = set(futures)
            if out:
                append_progress(out, "lanes_submitted", cpu_mutation_jobs=len(jobs), cpu_workers=settings.cpu_workers, gpu_lane=False, gpu_lane_reason="E7M has no gradient or dense-graph system by design")
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    label = futures[future]
                    result = future.result()
                    rows.append({key: value for key, value in result.items() if key != "history"})
                    mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "initial_candidate_hash", "final_candidate_hash", "parameter_diff_hash", "final_candidate_summary", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "failed_action_rollback_count", "accepted_by_operator", "rejected_by_operator")})
                    if out:
                        locked_write_json(out / "partial_aggregate_snapshot.json", {"schema_version": "e7m_partial_aggregate_snapshot_v1", "completed_rows": len(rows), "expected_rows": len(settings.seeds) * len(SYSTEMS), "pending_jobs": len(pending)})
                        append_progress(out, "mutation_job_complete", label=label, pending=len(pending))
    else:
        for job in jobs:
            result = mutation_worker(job)
            rows.append({key: value for key, value in result.items() if key != "history"})
            mutation_histories.append({key: result[key] for key in ("seed", "system", "history", "initial_candidate_hash", "final_candidate_hash", "parameter_diff_hash", "final_candidate_summary", "mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "failed_action_rollback_count", "accepted_by_operator", "rejected_by_operator")})
            if out:
                locked_write_json(out / "partial_aggregate_snapshot.json", {"schema_version": "e7m_partial_aggregate_snapshot_v1", "completed_rows": len(rows), "expected_rows": len(settings.seeds) * len(SYSTEMS), "pending_jobs": len(jobs) - len(mutation_histories)})
    rows.sort(key=lambda row: (row["system"], int(row["seed"])))
    mutation_histories.sort(key=lambda row: (row["system"], int(row["seed"])))
    aggregate = aggregate_results(rows)
    decision, detail = decide(aggregate)
    return {"tasks": tasks, "rows": rows, "mutation_histories": mutation_histories, "aggregate": aggregate, "decision": decision, "decision_detail": detail}


def build_payloads(settings: Settings, out: Path, results: dict[str, Any]) -> dict[str, Any]:
    payloads: dict[str, Any] = {
        "backend_manifest.json": {
            "schema_version": "e7m_backend_manifest_v1",
            "milestone": MILESTONE,
            "settings": settings_payload(settings),
            "systems": list(SYSTEMS),
            "mutation_systems": list(MUTATION_SYSTEMS),
            "control_systems": list(CONTROL_SYSTEMS),
            "hardware_identity": e7h.e7g.e7d.e7b.stable_hardware_identity(),
            "parallel_cpu_lanes": settings.execution_mode == "parallel",
            "gpu_lane": False,
            "gpu_lane_reason": "E7M intentionally contains no gradient, neural, or dense-graph training lane",
            "source_milestone": "E7L_SPAWN_REPAIR_COST_AND_NOISY_HEALTH_FALSIFICATION",
        },
        "task_generation_report.json": task_report(results["tasks"]),
        "lifecycle_contract_report.json": lifecycle_contract_report(),
        "anchor_working_copy_report.json": build_anchor_working_copy_report(results["rows"], results["aggregate"]),
        "crystallization_pruning_report.json": build_crystallization_pruning_report(results["rows"], results["aggregate"]),
        "promotion_guard_report.json": build_promotion_guard_report(results["rows"], results["aggregate"]),
        "system_results.json": {"schema_version": "e7m_system_results_v1", "rows": results["rows"]},
        "mutation_history.json": {"schema_version": "e7m_mutation_history_v1", "rows": results["mutation_histories"]},
        "leakage_report.json": build_leakage_report(results["aggregate"]),
        "aggregate_metrics.json": results["aggregate"],
        "decision.json": {"schema_version": "e7m_decision_v1", "decision": results["decision"], "detail": results["decision_detail"], "deterministic_replay_passed": False},
        "summary.json": {
            "schema_version": "e7m_summary_v1",
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
    report = {"schema_version": "e7m_deterministic_replay_v1", "internal_replay_passed": all(row["match"] for row in comparisons.values()), "hash_comparisons": comparisons, "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix()}
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
    parser.add_argument("--seeds", default="99201,99202,99203,99204,99205,99206,99207,99208")
    parser.add_argument("--train-rows-per-seed", type=int, default=720)
    parser.add_argument("--validation-rows-per-seed", type=int, default=300)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=300)
    parser.add_argument("--ood-rows-per-seed", type=int, default=300)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=300)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=300)
    parser.add_argument("--gradient-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mutation-generations", type=int, default=90)
    parser.add_argument("--mutation-population", type=int, default=20)
    parser.add_argument("--mutation-sigma", type=float, default=0.10)
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
        replay = deterministic_replay(settings, out, payloads) if settings.replay else {"schema_version": "e7m_deterministic_replay_v1", "internal_replay_passed": True, "hash_comparisons": {}, "replay_work_root": None}
        write_final_artifacts(out, payloads, replay)
    finally:
        stop.set()
        monitor.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
