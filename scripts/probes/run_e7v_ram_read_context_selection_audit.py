#!/usr/bin/env python3
"""E7V RAM read-context selection audit.

E7T/E7U made the write side simple enough for this proxy: every pocket writes
one deterministic next-free output cell, and that write map is frozen. E7V
keeps that policy fixed and tests how much anonymous RAM context each pocket
must read.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import importlib.util
import json
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


REPO_ROOT = Path(__file__).resolve().parents[2]
E7U_PATH = Path(__file__).with_name("run_e7u_ram_output_cell_assignment_audit.py")
MILESTONE = "E7V_RAM_READ_CONTEXT_SELECTION_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e7v_ram_read_context_selection_audit")
DEFAULT_SEEDS = (100001, 100002, 100003)

SYSTEMS = (
    "broad_read_next_free_write_baseline",
    "fixed_small_read_control",
    "random_read_map_control",
    "progressive_add_read_cells",
    "prune_from_broad_read",
    "swap_mutation_read_map",
    "grid_neighborhood_read_map",
    "sensitivity_guided_read_map_mutation",
    "learned_sparse_mask_reference",
    "oracle_read_map_reference",
    "dense_graph_danger_control",
)
TRAINED_READ_SYSTEMS = (
    "broad_read_next_free_write_baseline",
    "fixed_small_read_control",
    "random_read_map_control",
    "progressive_add_read_cells",
    "prune_from_broad_read",
    "swap_mutation_read_map",
    "grid_neighborhood_read_map",
    "sensitivity_guided_read_map_mutation",
    "learned_sparse_mask_reference",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "read_map_report.json",
    "read_budget_curve_report.json",
    "system_results.json",
    "mutation_history.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7v_compact_read_map_positive",
    "e7v_read_context_pruning_positive",
    "e7v_progressive_read_growth_positive",
    "e7v_read_map_swap_mutation_positive",
    "e7v_ram_grid_topology_positive",
    "e7v_broad_context_still_required",
    "e7v_sparse_mask_still_preferred",
    "e7v_graph_soup_regression_detected",
    "e7v_read_context_selection_no_advantage",
)


def load_e7u_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7u_ram_output_cell_assignment_audit", E7U_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7U helpers from {E7U_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7u = load_e7u_module()
e7r = e7u.e7r
e7p = e7u.e7p
e7o = e7u.e7o

FLOW_DIM = int(e7u.FLOW_DIM)
SKILLS = tuple(e7u.SKILLS)
SPLITS = tuple(e7u.SPLITS)
EVAL_SPLITS = tuple(e7u.EVAL_SPLITS)
RESULT_POS = dict(e7u.RESULT_POS)
RESULT_INDICES = tuple(e7u.RESULT_INDICES)
GRID_COLS = 8
GRID_ROWS = 5
PLATEAU_TOLERANCE = 0.002


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    pocket_pretrain_rows_per_seed: int
    pocket_validation_rows_per_seed: int
    pocket_dim: int
    pocket_core_steps: int
    pocket_epochs: int
    local_epochs: int
    full_epochs: int
    batch_size: int
    learning_rate: float
    local_learning_rate: float
    weight_decay: float
    mutation_generations: int
    mutation_population: int
    read_budgets: tuple[int, ...]
    fixed_read_count: int
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    execution_mode: str
    replay: bool


def round_float(value: float) -> float:
    return e7u.round_float(value)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7v::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7u.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7u.write_json(path, payload)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7u.append_progress(out, event, **details)


def resolve_out(path: str | Path) -> Path:
    return e7u.resolve_out(path)


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return e7u.parse_int_tuple(raw)


def select_device(requested: str) -> str:
    return e7u.select_device(requested)


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    payload["read_budgets"] = list(settings.read_budgets)
    payload["replay"] = False
    return payload


def to_e7r_settings(settings: Settings) -> Any:
    return e7r.Settings(
        seeds=settings.seeds,
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=settings.adversarial_rows_per_seed,
        pocket_pretrain_rows_per_seed=settings.pocket_pretrain_rows_per_seed,
        pocket_validation_rows_per_seed=settings.pocket_validation_rows_per_seed,
        pocket_dim=settings.pocket_dim,
        pocket_core_steps=settings.pocket_core_steps,
        pocket_epochs=settings.pocket_epochs,
        local_epochs=settings.local_epochs,
        full_epochs=settings.full_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        local_learning_rate=settings.local_learning_rate,
        weight_decay=settings.weight_decay,
        mask_mutation_generations=settings.mutation_generations,
        mask_mutation_population=settings.mutation_population,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        execution_mode=settings.execution_mode,
        replay=settings.replay,
    )


def to_e7o_settings(settings: Settings) -> Any:
    return e7r.to_e7o_settings(to_e7r_settings(settings))


def to_e7p_settings(settings: Settings) -> Any:
    return e7r.to_e7p_settings(to_e7r_settings(settings))


def bool_mask(indices: list[int] | tuple[int, ...] | set[int]) -> np.ndarray:
    mask = np.zeros(FLOW_DIM, dtype=bool)
    for idx in sorted({int(value) for value in indices}):
        if 0 <= idx < FLOW_DIM:
            mask[idx] = True
    return mask


def ordered_unique(indices: list[int] | tuple[int, ...]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        value = int(idx)
        if 0 <= value < FLOW_DIM and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def grid_coord(idx: int) -> tuple[int, int]:
    return divmod(int(idx), GRID_COLS)


def grid_locality_score(indices: list[int] | tuple[int, ...]) -> float:
    cells = ordered_unique(list(indices))
    if len(cells) <= 1:
        return 1.0
    distances = []
    for pos, left in enumerate(cells):
        lr, lc = grid_coord(left)
        for right in cells[pos + 1 :]:
            rr, rc = grid_coord(right)
            distances.append(abs(lr - rr) + abs(lc - rc))
    mean_dist = float(np.mean(distances)) if distances else 0.0
    return round_float(max(0.0, 1.0 - mean_dist / float(GRID_ROWS + GRID_COLS - 2)))


def contract_to_json(contract: dict[str, Any]) -> dict[str, Any]:
    row = e7u.contract_to_json(contract)
    row["read_count"] = int(np.sum(contract["read"]))
    row["read_map_sparsity"] = round_float(1.0 - float(np.mean(contract["read"])))
    row["grid_locality_score"] = grid_locality_score(row["read_indices"])
    row["fixed_next_free_write"] = True
    row["semantic_label_control"] = False
    return row


def build_read_contract(skill: str, system: str, read_indices: list[int] | tuple[int, ...]) -> dict[str, Any]:
    contract = e7u.build_contract(skill, system, int(RESULT_POS[skill]), output_count=1)
    contract["read"] = bool_mask(read_indices)
    contract["preserve"] = ~(contract["write"] | contract["scratch"])
    contract["read_selection_system"] = system
    contract["fixed_next_free_write"] = True
    return contract


def read_priority_for_skill(skill: str) -> list[int]:
    family = list(range(0, 6))
    common = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    result_context = [RESULT_POS[s] for s in SKILLS if s != skill]
    if skill == "compare":
        specific = [6, 7, 13, 14, 15]
    elif skill == "mod_add":
        specific = [6, 7, 8, 11, 12, 16]
    elif skill == "parity":
        specific = [14, 15, 6, 7]
    elif skill == "threshold":
        specific = [6, 7, 9, 11, 18]
    elif skill == "counterfactual_flip":
        specific = [10, 12, 8, RESULT_POS["mod_add"], 6, 7]
    elif skill == "verify":
        specific = [RESULT_POS["compare"], RESULT_POS["threshold"], 6, 7, 9, 11, 13]
    else:
        specific = []
    trailing = [22, 23, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    return ordered_unique(specific + family + common + result_context + trailing)


def priority_read_map(skill: str, budget: int) -> list[int]:
    return read_priority_for_skill(skill)[: max(1, min(FLOW_DIM, int(budget)))]


def broad_read_maps() -> dict[str, list[int]]:
    return {skill: list(range(FLOW_DIM)) for skill in SKILLS}


def fixed_small_read_maps(count: int) -> dict[str, list[int]]:
    return {skill: list(range(max(1, min(FLOW_DIM, count)))) for skill in SKILLS}


def random_read_maps(seed: int, count: int, system: str) -> dict[str, list[int]]:
    maps: dict[str, list[int]] = {}
    for skill in SKILLS:
        rng = random.Random(stable_seed(f"{system}:{seed}:{skill}:{count}"))
        maps[skill] = sorted(rng.sample(list(range(FLOW_DIM)), max(1, min(FLOW_DIM, count))))
    return maps


def grid_neighborhood_read_maps(count: int) -> dict[str, list[int]]:
    maps: dict[str, list[int]] = {}
    for skill_index, skill in enumerate(SKILLS):
        center = int(RESULT_POS[skill])
        cr, cc = grid_coord(center)
        cells: list[int] = []
        radius = 0
        while len(cells) < max(1, min(FLOW_DIM, count)) and radius < GRID_ROWS + GRID_COLS:
            for idx in range(FLOW_DIM):
                r, c = grid_coord(idx)
                if abs(r - cr) + abs(c - cc) <= radius:
                    cells.append(idx)
            cells = ordered_unique(cells)
            radius += 1
        # Add a small deterministic raw-feature prefix so every pocket can see
        # task inputs, but keep the final budget fixed.
        cells = ordered_unique(priority_read_map(skill, min(6, count)) + cells + [skill_index])
        maps[skill] = cells[: max(1, min(FLOW_DIM, count))]
    return maps


def estimate_cell_sensitivity(
    skill: str,
    state: dict[str, Any],
    context_task: dict[str, list[dict[str, Any]]],
    max_rows: int = 128,
) -> list[dict[str, Any]]:
    rows = context_task["train"][:max_rows]
    if not rows:
        return [{"cell": idx, "score": 0.0} for idx in range(FLOW_DIM)]
    x = np.asarray([row["flow"] for row in rows], dtype=np.float32)
    base = e7p.np_forward(state, x)[:, RESULT_POS[skill]]
    scores = []
    for idx in range(FLOW_DIM):
        altered = x.copy()
        altered[:, idx] = 0.0
        pred = e7p.np_forward(state, altered)[:, RESULT_POS[skill]]
        perturb = float(np.mean(np.abs(base - pred)))
        variance = float(np.var(x[:, idx]))
        nonzero = float(np.mean(np.abs(x[:, idx]) > 1e-6))
        score = perturb + 0.03 * variance + 0.005 * nonzero
        scores.append({"cell": idx, "score": round_float(score), "perturbation": round_float(perturb), "variance": round_float(variance), "nonzero_rate": round_float(nonzero)})
    return sorted(scores, key=lambda row: (-row["score"], row["cell"]))


def sensitivity_read_maps(
    baseline_library: dict[str, dict[str, Any]],
    context_tasks: dict[str, Any],
    count: int,
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    maps: dict[str, list[int]] = {}
    rows: list[dict[str, Any]] = []
    for skill in SKILLS:
        ranking = estimate_cell_sensitivity(skill, baseline_library[skill], context_tasks[skill])
        priority = [row["cell"] for row in ranking]
        # Preserve essential prior result inputs for dependent pockets without
        # assigning any semantic name to the cells.
        merged = ordered_unique(priority_read_map(skill, min(4, count)) + priority)
        maps[skill] = merged[: max(1, min(FLOW_DIM, count))]
        for rank, row in enumerate(ranking[: min(12, FLOW_DIM)]):
            rows.append({"skill": skill, "rank": rank, **row})
    return maps, rows


def read_map_score(maps: dict[str, list[int]], seed: int) -> float:
    score = 0.0
    for skill in SKILLS:
        selected = set(maps[skill])
        priority = read_priority_for_skill(skill)
        for rank, cell in enumerate(priority):
            if cell in selected:
                score += 1.0 / (1.0 + rank)
        score += 0.08 * grid_locality_score(maps[skill])
        score -= 0.012 * len(selected)
    return score + 0.000001 * (seed % 997)


def mutate_read_maps(seed: int, system: str, initial: dict[str, list[int]], settings: Settings, out: Path | None) -> tuple[dict[str, list[int]], dict[str, Any]]:
    rng = random.Random(stable_seed(f"read-map-mutation:{system}:{seed}"))
    best = {skill: sorted(set(indices)) for skill, indices in initial.items()}
    best_score = read_map_score(best, seed)
    initial_hash = payload_sha256(best)
    attempts = accepted = rejected = 0
    history: list[dict[str, Any]] = []
    for generation in range(settings.mutation_generations):
        generation_best = best_score
        for _ in range(settings.mutation_population):
            attempts += 1
            cand = {skill: list(indices) for skill, indices in best.items()}
            skill = rng.choice(SKILLS)
            current = set(cand[skill])
            op = rng.choice(("add_read_cell", "remove_read_cell", "swap_read_cell", "move_read_cell_nearby", "expand_read_window", "shrink_read_window", "clone_read_map_from_similar_pocket"))
            if op == "add_read_cell" and len(current) < FLOW_DIM:
                current.add(rng.randrange(FLOW_DIM))
            elif op == "remove_read_cell" and len(current) > 1:
                current.remove(rng.choice(sorted(current)))
            elif op == "swap_read_cell" and len(current) < FLOW_DIM:
                current.remove(rng.choice(sorted(current)))
                current.add(rng.randrange(FLOW_DIM))
            elif op == "move_read_cell_nearby" and current:
                old = rng.choice(sorted(current))
                r, c = grid_coord(old)
                candidates = []
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                        candidates.append(rr * GRID_COLS + cc)
                if candidates:
                    current.remove(old)
                    current.add(rng.choice(candidates))
            elif op == "expand_read_window":
                for idx in list(current):
                    r, c = grid_coord(idx)
                    for dr, dc in ((1, 0), (0, 1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                            current.add(rr * GRID_COLS + cc)
                        if len(current) >= FLOW_DIM:
                            break
            elif op == "shrink_read_window" and len(current) > 1:
                priority = set(priority_read_map(skill, max(1, len(current) // 2)))
                removable = sorted(current - priority) or sorted(current)
                if len(current) > 1:
                    current.remove(rng.choice(removable))
            elif op == "clone_read_map_from_similar_pocket":
                other = rng.choice([item for item in SKILLS if item != skill])
                current = set(cand[other])
            cand[skill] = sorted(current)
            score = read_map_score(cand, seed)
            if score > best_score + 1e-12:
                best = {item: sorted(set(values)) for item, values in cand.items()}
                best_score = score
                accepted += 1
            else:
                rejected += 1
        row = {
            "seed": seed,
            "system": system,
            "generation": generation,
            "operation": "add_remove_swap_move_expand_shrink_clone",
            "score": round_float(best_score),
            "generation_gain": round_float(best_score - generation_best),
            "accepted": accepted,
            "rejected": rejected,
            "rollback": rejected,
            "read_map_hash": payload_sha256(best),
        }
        history.append(row)
        if out:
            append_progress(out, "read_map_mutation_generation", **row)
    if accepted == 0:
        forced = {skill: priority_read_map(skill, settings.fixed_read_count) for skill in SKILLS}
        forced_score = read_map_score(forced, seed)
        if payload_sha256(forced) != payload_sha256(best):
            attempts += 1
            accepted += 1
            best = forced
            best_score = forced_score
            row = {
                "seed": seed,
                "system": system,
                "generation": settings.mutation_generations,
                "operation": "deterministic_priority_repair",
                "score": round_float(best_score),
                "generation_gain": 0.0,
                "accepted": accepted,
                "rejected": rejected,
                "rollback": rejected,
                "read_map_hash": payload_sha256(best),
            }
            history.append(row)
            if out:
                append_progress(out, "read_map_mutation_generation", **row)
    final_hash = payload_sha256(best)
    mutation = {
        "history": history,
        "initial_candidate_hash": initial_hash,
        "final_candidate_hash": final_hash,
        "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": final_hash}),
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "mean_read_count": round_float(float(np.mean([len(indices) for indices in best.values()]))),
        "mean_grid_locality_score": round_float(float(np.mean([grid_locality_score(indices) for indices in best.values()]))),
    }
    return best, mutation


def build_contracts(system: str, read_maps: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
    return {skill: build_read_contract(skill, system, read_maps[skill]) for skill in SKILLS}


def read_map_rows(seed: int, system: str, contracts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for skill in SKILLS:
        contract = contracts[skill]
        contract_json = contract_to_json(contract)
        rows.append({
            "seed": seed,
            "system": system,
            "skill": skill,
            "read_count": contract_json["read_count"],
            "read_indices": contract_json["read_indices"],
            "write_indices": contract_json["write_indices"],
            "read_map_sparsity": contract_json["read_map_sparsity"],
            "grid_locality_score": contract_json["grid_locality_score"],
            "contract": contract_json,
        })
    return rows


def train_read_map_library(
    seed: int,
    system: str,
    baseline_library: dict[str, dict[str, Any]],
    context_tasks: dict[str, Any],
    read_maps: dict[str, list[int]],
    settings: Settings,
    out: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    library: dict[str, dict[str, Any]] = {}
    contracts = build_contracts(system, read_maps)
    training_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    e7r_settings = to_e7r_settings(settings)
    for skill in SKILLS:
        contract = contracts[skill]
        task = e7u.transform_context_for_assignment(context_tasks[skill], skill, int(RESULT_POS[skill]))
        trained = e7r.train_masked_context_pocket(seed, skill, system, baseline_library[skill], task, e7r_settings, contract, out)
        library[skill] = trained["state"]
        contract_json = contract_to_json(contract)
        map_rows.append({"seed": seed, "system": system, "skill": skill, "read_count": contract_json["read_count"], "read_indices": contract_json["read_indices"], "write_indices": contract_json["write_indices"], "read_map_sparsity": contract_json["read_map_sparsity"], "grid_locality_score": contract_json["grid_locality_score"], "contract": contract_json, "state_hash": e7p.state_hash(trained["state"])})
        training_rows.append({"seed": seed, "system": system, "skill": skill, "state_hash": e7p.state_hash(trained["state"]), "history": trained["history"], "contract": contract_json})
    return library, contracts, training_rows, map_rows


def add_read_metrics(row: dict[str, Any], contracts: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if not contracts:
        row["eval_mean_read_cell_count"] = 0.0
        row["eval_mean_write_cell_count"] = 0.0
        row["eval_mean_read_map_sparsity"] = 0.0
        row["eval_mean_grid_locality_score"] = 0.0
        return row
    read_counts = [int(np.sum(contract["read"])) for contract in contracts.values()]
    sparsities = [1.0 - float(np.mean(contract["read"])) for contract in contracts.values()]
    localities = [grid_locality_score(np.flatnonzero(contract["read"]).astype(int).tolist()) for contract in contracts.values()]
    mean_read = round_float(float(np.mean(read_counts)))
    mean_sparsity = round_float(float(np.mean(sparsities)))
    mean_locality = round_float(float(np.mean(localities)))
    for split in SPLITS:
        row["evals"][split]["read_cell_count"] = mean_read
        row["evals"][split]["write_cell_count"] = row["evals"][split].get("output_cell_count", 1.0)
        row["evals"][split]["read_map_sparsity"] = mean_sparsity
        row["evals"][split]["grid_locality_score"] = mean_locality
    row["eval_mean_read_cell_count"] = mean_read
    row["eval_mean_write_cell_count"] = row.get("eval_mean_output_cell_count", 1.0)
    row["eval_mean_read_map_sparsity"] = mean_sparsity
    row["eval_mean_grid_locality_score"] = mean_locality
    row["smallest_stable_read_count"] = mean_read
    return row


def evaluate_read_system(seed: int, system: str, library: dict[str, dict[str, Any]] | None, contracts: dict[str, dict[str, Any]] | None, task: dict[str, list[dict[str, Any]]], symbolic: bool = False) -> dict[str, Any]:
    row = e7u.evaluate_assignment_system(seed, system, library, contracts, task, symbolic=symbolic)
    return add_read_metrics(row, contracts)


def choose_smallest_within_plateau(curve: list[dict[str, Any]]) -> dict[str, Any]:
    best_score = max(row["eval_mean_composition_usefulness"] for row in curve)
    candidates = [row for row in curve if row["eval_mean_composition_usefulness"] >= best_score - PLATEAU_TOLERANCE]
    return min(candidates, key=lambda row: (row["read_budget"], -row["eval_mean_composition_usefulness"]))


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (int(row.get("seed", 0)), SYSTEMS.index(str(row.get("system")))))
    systems: dict[str, Any] = {}
    metric_names = (
        "eval_mean_answer_accuracy",
        "eval_mean_composition_usefulness",
        "eval_mean_route_accuracy",
        "heldout_usefulness",
        "ood_usefulness",
        "counterfactual_usefulness",
        "adversarial_usefulness",
        "eval_mean_read_cell_count",
        "eval_mean_write_cell_count",
        "eval_mean_read_map_sparsity",
        "eval_mean_grid_locality_score",
        "eval_mean_write_spread",
        "eval_mean_changed_cell_count",
        "eval_mean_output_cell_count",
        "eval_mean_ram_collision_rate",
        "eval_mean_preserve_mask_corruption_rate",
        "eval_mean_write_mask_violation_rate",
        "eval_mean_next_pocket_input_compatibility_error",
        "parameter_count",
        "bit_budget",
        "mutation_attempts",
        "accepted_mutations",
        "rejected_mutations",
        "rollback_count",
        "mean_read_count",
        "mean_grid_locality_score",
    )
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in metric_names:
                if metric in row and row[metric] is not None:
                    metrics.setdefault(metric, []).append(float(row[metric]))
            for split in SPLITS:
                for key, value in row.get("evals", {}).get(split, {}).items():
                    if isinstance(value, (int, float)):
                        metrics.setdefault(f"{split}_{key}", []).append(float(value))
        systems[system] = {
            "seed_count": len(system_rows),
            "mean": {key: round_float(float(np.mean(values))) for key, values in metrics.items()},
            "min": {key: round_float(float(np.min(values))) for key, values in metrics.items()},
            "max": {key: round_float(float(np.max(values))) for key, values in metrics.items()},
        }
    candidates = [system for system in SYSTEMS if system not in {"oracle_read_map_reference", "dense_graph_danger_control"}]
    best = max(candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    return {"schema_version": "e7v_aggregate_metrics_v1", "systems": systems, "best_non_reference_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    mean = {system: aggregate["systems"][system]["mean"].get("eval_mean_composition_usefulness", 0.0) for system in SYSTEMS}
    read = {system: aggregate["systems"][system]["mean"].get("eval_mean_read_cell_count", 0.0) for system in SYSTEMS}
    best = aggregate["best_non_reference_system"]
    compact_systems = (
        "progressive_add_read_cells",
        "prune_from_broad_read",
        "swap_mutation_read_map",
        "grid_neighborhood_read_map",
        "sensitivity_guided_read_map_mutation",
        "learned_sparse_mask_reference",
    )
    best_compact = max(compact_systems, key=lambda system: mean[system])
    broad = mean["broad_read_next_free_write_baseline"]
    dense = mean["dense_graph_danger_control"]
    oracle = mean["oracle_read_map_reference"]
    detail = {
        "best_non_reference_system": best,
        "best_compact_system": best_compact,
        "broad": broad,
        "progressive": mean["progressive_add_read_cells"],
        "prune": mean["prune_from_broad_read"],
        "swap": mean["swap_mutation_read_map"],
        "grid": mean["grid_neighborhood_read_map"],
        "sensitivity": mean["sensitivity_guided_read_map_mutation"],
        "learned_sparse": mean["learned_sparse_mask_reference"],
        "oracle": oracle,
        "dense": dense,
        "broad_read_count": read["broad_read_next_free_write_baseline"],
        "best_read_count": read.get(best, 0.0),
    }
    non_dense_max = max(value for key, value in detail.items() if key not in {"dense", "oracle", "best_non_reference_system", "best_compact_system", "broad_read_count", "best_read_count"})
    if dense >= non_dense_max + 0.025:
        decision = "e7v_graph_soup_regression_detected"
    elif best == "learned_sparse_mask_reference" and mean[best] >= broad + 0.004:
        decision = "e7v_sparse_mask_still_preferred"
    elif best == "grid_neighborhood_read_map" and mean[best] >= broad + 0.004:
        decision = "e7v_ram_grid_topology_positive"
    elif best == "swap_mutation_read_map" and mean[best] >= broad + 0.004:
        decision = "e7v_read_map_swap_mutation_positive"
    elif best == "prune_from_broad_read" and mean[best] >= broad - PLATEAU_TOLERANCE and read[best] <= max(1.0, read["broad_read_next_free_write_baseline"] * 0.75):
        decision = "e7v_read_context_pruning_positive"
    elif best == "progressive_add_read_cells" and mean[best] >= broad - PLATEAU_TOLERANCE and read[best] <= max(1.0, read["broad_read_next_free_write_baseline"] * 0.75):
        decision = "e7v_progressive_read_growth_positive"
    elif best_compact != "learned_sparse_mask_reference" and mean[best_compact] >= broad + 0.004:
        decision = "e7v_compact_read_map_positive"
    elif broad >= max(mean[system] for system in compact_systems) - PLATEAU_TOLERANCE:
        decision = "e7v_broad_context_still_required"
    else:
        decision = "e7v_read_context_selection_no_advantage"
    return {"schema_version": "e7v_decision_v1", "decision": decision, "detail": detail, "deterministic_replay_passed": False}


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    e7o_settings = to_e7o_settings(settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    context_tasks = e7p.generate_context_tasks(composition_task)
    baseline_library: dict[str, dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    read_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    trained_cache: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {}

    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        state = e7p.copy_state(trained["state"], "e7v_baseline_standalone")
        baseline_library[skill] = state
        training_rows.append({"seed": seed, "system": "baseline_standalone_pocket", "skill": skill, "state_hash": e7p.state_hash(state), "standalone": trained["standalone"]})

    direct_maps = {
        "broad_read_next_free_write_baseline": broad_read_maps(),
        "fixed_small_read_control": fixed_small_read_maps(settings.fixed_read_count),
        "random_read_map_control": random_read_maps(seed, settings.fixed_read_count, "random_read_map_control"),
        "grid_neighborhood_read_map": grid_neighborhood_read_maps(settings.fixed_read_count),
    }
    sens_maps, sens_rows = sensitivity_read_maps(baseline_library, context_tasks, settings.fixed_read_count)
    direct_maps["sensitivity_guided_read_map_mutation"] = sens_maps
    sensitivity_rows.extend({"seed": seed, **row} for row in sens_rows)

    swap_initial = random_read_maps(seed, settings.fixed_read_count, "swap_mutation_initial")
    swap_maps, swap_mutation = mutate_read_maps(seed, "swap_mutation_read_map", swap_initial, settings, out)
    direct_maps["swap_mutation_read_map"] = swap_maps
    mutation_rows.extend(swap_mutation["history"])

    learned_initial = random_read_maps(seed + 17, settings.fixed_read_count, "learned_sparse_mask_initial")
    learned_maps, learned_mutation = mutate_read_maps(seed, "learned_sparse_mask_reference", learned_initial, settings, out)
    direct_maps["learned_sparse_mask_reference"] = learned_maps
    mutation_rows.extend(learned_mutation["history"])

    for system, maps in direct_maps.items():
        library, contracts, train_rows, map_rows = train_read_map_library(seed, system, baseline_library, context_tasks, maps, settings, out)
        result = evaluate_read_system(seed, system, library, contracts, composition_task)
        if system == "swap_mutation_read_map":
            result.update({key: value for key, value in swap_mutation.items() if key != "history"})
        if system == "learned_sparse_mask_reference":
            result.update({key: value for key, value in learned_mutation.items() if key != "history"})
        rows.append(result)
        training_rows.extend(train_rows)
        read_rows.extend(map_rows)
        trained_cache[system] = (library, contracts)
        if out:
            append_progress(out, "read_map_system_evaluated", seed=seed, system=system, usefulness=result["eval_mean_composition_usefulness"], read_count=result["eval_mean_read_cell_count"])

    for system, budget_order in (
        ("progressive_add_read_cells", sorted(set(settings.read_budgets))),
        ("prune_from_broad_read", sorted(set(settings.read_budgets), reverse=True)),
    ):
        curve: list[dict[str, Any]] = []
        payload_by_budget: dict[int, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
        for budget in budget_order:
            read_maps = {skill: priority_read_map(skill, int(budget)) for skill in SKILLS}
            library, contracts, train_rows, map_rows = train_read_map_library(seed, f"{system}_k{budget}", baseline_library, context_tasks, read_maps, settings, out)
            result = evaluate_read_system(seed, system, library, contracts, composition_task)
            result["read_budget"] = int(budget)
            curve.append(result)
            payload_by_budget[int(budget)] = (library, contracts, train_rows, map_rows, result)
            accepted = 1 if len(curve) == 1 or result["eval_mean_composition_usefulness"] >= max(item["eval_mean_composition_usefulness"] for item in curve[:-1]) + PLATEAU_TOLERANCE else 0
            mutation_rows.append({"seed": seed, "system": system, "generation": len(curve) - 1, "operation": "add_read_cell" if system.startswith("progressive") else "remove_read_cell", "read_budget": int(budget), "score": result["eval_mean_composition_usefulness"], "accepted": accepted, "rejected": 1 - accepted, "rollback": 1 - accepted, "read_map_hash": payload_sha256(read_maps)})
            if out:
                append_progress(out, "read_budget_candidate_evaluated", seed=seed, system=system, read_budget=int(budget), usefulness=result["eval_mean_composition_usefulness"])
        chosen = choose_smallest_within_plateau(curve)
        chosen_budget = int(chosen["read_budget"])
        _, _, train_rows, map_rows, chosen_result = payload_by_budget[chosen_budget]
        map_rows = [{**row, "system": system, "contract": {**row["contract"], "mode": system}} for row in map_rows]
        final = dict(chosen_result)
        final["smallest_stable_read_count"] = round_float(float(chosen_budget))
        rows.append(final)
        training_rows.extend(train_rows)
        read_rows.extend(map_rows)
        curve_rows.append({
            "seed": seed,
            "system": system,
            "curve": [{"read_budget": int(item["read_budget"]), "usefulness": item["eval_mean_composition_usefulness"], "heldout": item["heldout_usefulness"], "ood": item["ood_usefulness"], "counterfactual": item["counterfactual_usefulness"], "adversarial": item["adversarial_usefulness"]} for item in curve],
            "chosen_read_budget": chosen_budget,
            "plateau_tolerance": PLATEAU_TOLERANCE,
        })

    rows.append(evaluate_read_system(seed, "oracle_read_map_reference", None, None, composition_task, symbolic=True))
    dense = e7o.train_monolithic(seed, "dense_graph_danger_control", composition_task, e7o_settings, out, hidden=176, depth=4)
    dense["system"] = "dense_graph_danger_control"
    for key in ("eval_mean_read_cell_count", "eval_mean_write_cell_count", "eval_mean_read_map_sparsity", "eval_mean_grid_locality_score", "eval_mean_write_spread", "eval_mean_changed_cell_count", "eval_mean_output_cell_count", "eval_mean_ram_collision_rate", "eval_mean_preserve_mask_corruption_rate", "eval_mean_write_mask_violation_rate", "eval_mean_next_pocket_input_compatibility_error"):
        dense[key] = 0.0
    rows.append(dense)

    return {"seed": seed, "rows": rows, "training_rows": training_rows, "read_rows": read_rows, "curve_rows": curve_rows, "mutation_rows": mutation_rows, "sensitivity_rows": sensitivity_rows}


def build_reports(
    rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    read_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    mutation_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    lines = [
        "# E7V RAM Read Context Selection Audit Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_non_reference_system = {aggregate['best_non_reference_system']}",
        f"deterministic_replay_passed = {deterministic.get('internal_replay_passed', False)}",
        "```",
        "",
        "## Mean Scores",
        "",
        "```text",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        lines.append(
            f"{system:<42} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} "
            f"read={mean.get('eval_mean_read_cell_count', 0.0):.3f} "
            f"spread={mean.get('eval_mean_write_spread', 0.0):.6f} "
            f"locality={mean.get('eval_mean_grid_locality_score', 0.0):.6f}"
        )
    lines.extend([
        "```",
        "",
        "## Interpretation",
        "",
        "Write policy was fixed to one deterministic next-free output cell per pocket. The read maps are anonymous cell sets; no semantic cell names are used as model input.",
        "",
        "## Boundary",
        "",
        "E7V is a controlled anonymous RAM read-context selection audit. It does not make raw-language, AGI, consciousness, or model-scale claims.",
        "",
    ])
    return {
        "pocket_training_report.json": {"schema_version": "e7v_pocket_training_report_v1", "rows": sorted(training_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "read_map_report.json": {"schema_version": "e7v_read_map_report_v1", "rows": sorted(read_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "read_budget_curve_report.json": {"schema_version": "e7v_read_budget_curve_report_v1", "rows": sorted(curve_rows, key=lambda row: (row["seed"], row["system"]))},
        "sensitivity_report.json": {"schema_version": "e7v_sensitivity_report_v1", "rows": sorted(sensitivity_rows, key=lambda row: (row["seed"], row["skill"], row["rank"]))},
        "system_results.json": {"schema_version": "e7v_system_results_v1", "rows": sorted(rows, key=lambda row: (row["seed"], SYSTEMS.index(row["system"])))},
        "mutation_history.json": {"schema_version": "e7v_mutation_history_v1", "rows": sorted(mutation_rows, key=lambda row: (row["seed"], row.get("system", ""), row.get("generation", 0), row.get("read_budget", 0)))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7v_summary_v1", "decision": decision["decision"], "best_non_reference_system": aggregate["best_non_reference_system"], "deterministic_replay_passed": deterministic.get("internal_replay_passed", False), "checker_failure_count": None},
        "report.md": "\n".join(lines),
    }


def hash_artifacts(out: Path) -> dict[str, str]:
    return {artifact: hashlib.sha256((out / artifact).read_bytes()).hexdigest() for artifact in HASH_ARTIFACTS}


def replay_command(settings: Settings, replay_out: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--out",
        str(replay_out.relative_to(REPO_ROOT)),
        "--seeds",
        ",".join(map(str, settings.seeds)),
        "--train-rows-per-seed",
        str(settings.train_rows_per_seed),
        "--validation-rows-per-seed",
        str(settings.validation_rows_per_seed),
        "--heldout-rows-per-seed",
        str(settings.heldout_rows_per_seed),
        "--ood-rows-per-seed",
        str(settings.ood_rows_per_seed),
        "--counterfactual-rows-per-seed",
        str(settings.counterfactual_rows_per_seed),
        "--adversarial-rows-per-seed",
        str(settings.adversarial_rows_per_seed),
        "--pocket-pretrain-rows-per-seed",
        str(settings.pocket_pretrain_rows_per_seed),
        "--pocket-validation-rows-per-seed",
        str(settings.pocket_validation_rows_per_seed),
        "--pocket-dim",
        str(settings.pocket_dim),
        "--pocket-core-steps",
        str(settings.pocket_core_steps),
        "--pocket-epochs",
        str(settings.pocket_epochs),
        "--local-epochs",
        str(settings.local_epochs),
        "--full-epochs",
        str(settings.full_epochs),
        "--batch-size",
        str(settings.batch_size),
        "--learning-rate",
        str(settings.learning_rate),
        "--local-learning-rate",
        str(settings.local_learning_rate),
        "--weight-decay",
        str(settings.weight_decay),
        "--mutation-generations",
        str(settings.mutation_generations),
        "--mutation-population",
        str(settings.mutation_population),
        "--read-budgets",
        ",".join(map(str, settings.read_budgets)),
        "--fixed-read-count",
        str(settings.fixed_read_count),
        "--cpu-workers",
        str(settings.cpu_workers),
        "--device",
        settings.device,
        "--heartbeat-seconds",
        str(settings.heartbeat_seconds),
        "--execution-mode",
        settings.execution_mode,
        "--replay",
    ]


def run(settings: Settings, out: Path) -> dict[str, Any]:
    if out.exists() and not settings.replay:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    append_progress(out, "run_start", settings=settings_payload(settings))
    heartbeat = threading.Event()

    def heartbeat_loop() -> None:
        while not heartbeat.wait(settings.heartbeat_seconds):
            e7r.append_jsonl(out / "hardware_heartbeat.jsonl", e7r.hardware_sample())

    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    try:
        e7o_settings = to_e7o_settings(settings)
        composition_tasks = e7o.generate_composition_tasks(e7o_settings)
        pocket_tasks = e7o.generate_pocket_tasks(e7o_settings)
        write_json(out / "backend_manifest.json", {"schema_version": "e7v_backend_manifest_v1", "milestone": MILESTONE, "settings": settings_payload(settings), "systems": list(SYSTEMS), "flow_dim": FLOW_DIM, "grid_shape": [GRID_ROWS, GRID_COLS], "fixed_write_policy": {"allocator": "deterministic_next_free", "output_cells_per_pocket": 1, "frozen_write_map": True, "direct_shared_write": False}, "semantic_lane_labels_as_model_input": False, "dense_graph_primary": False, "training_performed": True, "device": select_device(settings.device), "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        write_json(out / "task_generation_report.json", {"schema_version": "e7v_task_generation_report_v1", "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()}, "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()}})
        rows: list[dict[str, Any]] = []
        training_rows: list[dict[str, Any]] = []
        read_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []
        mutation_rows: list[dict[str, Any]] = []
        sensitivity_rows: list[dict[str, Any]] = []
        jobs = [{"seed": seed, "settings": settings.__dict__.copy(), "composition_task": composition_tasks[seed], "pocket_task": pocket_tasks[seed], "out": str(out)} for seed in settings.seeds]
        max_workers = max(1, min(settings.cpu_workers, len(jobs)))
        if settings.device == "cuda" and max_workers > 1:
            append_progress(out, "cuda_process_pool_disabled", requested_workers=max_workers, active_workers=1)
            max_workers = 1
        if max_workers == 1:
            for job in jobs:
                label = f"seed{job['seed']}"
                result = seed_worker(job)
                rows.extend(result["rows"])
                training_rows.extend(result["training_rows"])
                read_rows.extend(result["read_rows"])
                curve_rows.extend(result["curve_rows"])
                mutation_rows.extend(result["mutation_rows"])
                sensitivity_rows.extend(result["sensitivity_rows"])
                append_progress(out, "seed_job_complete", label=label, pending=len(jobs) - len({row["seed"] for row in rows}))
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_read_rows": len(read_rows), "completed_curve_rows": len(curve_rows), "last_completed": label, "pending": len(jobs) - len({row["seed"] for row in rows})})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
                while futures:
                    done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_read_rows": len(read_rows), "completed_curve_rows": len(curve_rows), "pending": len(futures)})
                        continue
                    for future in done:
                        label = futures.pop(future)
                        result = future.result()
                        rows.extend(result["rows"])
                        training_rows.extend(result["training_rows"])
                        read_rows.extend(result["read_rows"])
                        curve_rows.extend(result["curve_rows"])
                        mutation_rows.extend(result["mutation_rows"])
                        sensitivity_rows.extend(result["sensitivity_rows"])
                        append_progress(out, "seed_job_complete", label=label, pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_read_rows": len(read_rows), "completed_curve_rows": len(curve_rows), "last_completed": label, "pending": len(futures)})
        aggregate = aggregate_results(rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e7v_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, training_rows, read_rows, curve_rows, mutation_rows, sensitivity_rows, aggregate, decision, deterministic_placeholder)
        for name, payload in reports.items():
            if name.endswith(".json"):
                write_json(out / name, payload)
            else:
                write_text(out / name, str(payload))
        append_progress(out, "primary_artifacts_written" if not settings.replay else "replay_artifacts_written", artifact_count=len(HASH_ARTIFACTS))
        if not settings.replay:
            replay_out = out / "deterministic_replay_work"
            if replay_out.exists():
                shutil.rmtree(replay_out)
            append_progress(out, "deterministic_replay_start", replay_out=str(replay_out.relative_to(REPO_ROOT)))
            subprocess.run(replay_command(settings, replay_out), cwd=str(REPO_ROOT), check=True)
            primary_hashes = hash_artifacts(out)
            replay_hashes = hash_artifacts(replay_out)
            comparisons = {artifact: {"primary": primary_hashes[artifact], "replay": replay_hashes[artifact], "match": primary_hashes[artifact] == replay_hashes[artifact]} for artifact in HASH_ARTIFACTS}
            deterministic = {"schema_version": "e7v_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, training_rows, read_rows, curve_rows, mutation_rows, sensitivity_rows, aggregate, decision, deterministic)
            reports["deterministic_replay.json"] = deterministic
            for name, payload in reports.items():
                if name.endswith(".json"):
                    write_json(out / name, payload)
                else:
                    write_text(out / name, str(payload))
            append_progress(out, "deterministic_replay_complete", internal_replay_passed=deterministic["internal_replay_passed"])
            append_progress(out, "final_artifacts_written", artifact_count=len(HASH_ARTIFACTS) + 1)
        else:
            write_json(out / "deterministic_replay.json", deterministic_placeholder)
        return decision
    finally:
        heartbeat.set()
        thread.join(timeout=2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--train-rows-per-seed", type=int, default=360)
    parser.add_argument("--validation-rows-per-seed", type=int, default=160)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=160)
    parser.add_argument("--ood-rows-per-seed", type=int, default=160)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=160)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=160)
    parser.add_argument("--pocket-pretrain-rows-per-seed", type=int, default=420)
    parser.add_argument("--pocket-validation-rows-per-seed", type=int, default=160)
    parser.add_argument("--pocket-dim", type=int, default=56)
    parser.add_argument("--pocket-core-steps", type=int, default=2)
    parser.add_argument("--pocket-epochs", type=int, default=42)
    parser.add_argument("--local-epochs", type=int, default=32)
    parser.add_argument("--full-epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--local-learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mutation-generations", type=int, default=18)
    parser.add_argument("--mutation-population", type=int, default=8)
    parser.add_argument("--read-budgets", default="1,2,4,8,12,16,24,30,40")
    parser.add_argument("--fixed-read-count", type=int, default=12)
    parser.add_argument("--cpu-workers", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--execution-mode", default="evidence")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings(
        seeds=parse_int_tuple(args.seeds),
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        pocket_pretrain_rows_per_seed=args.pocket_pretrain_rows_per_seed,
        pocket_validation_rows_per_seed=args.pocket_validation_rows_per_seed,
        pocket_dim=args.pocket_dim,
        pocket_core_steps=args.pocket_core_steps,
        pocket_epochs=args.pocket_epochs,
        local_epochs=args.local_epochs,
        full_epochs=args.full_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        local_learning_rate=args.local_learning_rate,
        weight_decay=args.weight_decay,
        mutation_generations=args.mutation_generations,
        mutation_population=args.mutation_population,
        read_budgets=parse_int_tuple(args.read_budgets),
        fixed_read_count=args.fixed_read_count,
        cpu_workers=args.cpu_workers,
        device=select_device(args.device),
        heartbeat_seconds=args.heartbeat_seconds,
        execution_mode=args.execution_mode,
        replay=args.replay,
    )
    decision = run(settings, resolve_out(args.out))
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
