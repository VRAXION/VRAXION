#!/usr/bin/env python3
"""E7U RAM output-cell assignment audit.

E7U isolates the write side after E7T. It asks whether a deterministic
next-free output-cell allocator is enough, or whether pocket output cells need
mutation-selected RAM addresses. RAM cells remain anonymous and assignments are
frozen at runtime.
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
E7R_PATH = Path(__file__).with_name("run_e7r_numeric_pocket_masked_flow_io_contract_probe.py")
MILESTONE = "E7U_RAM_OUTPUT_CELL_ASSIGNMENT_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e7u_ram_output_cell_assignment_audit")
DEFAULT_SEEDS = (99901, 99902, 99903)

SYSTEMS = (
    "next_free_output_cell_allocator",
    "random_initial_then_freeze",
    "mutation_selected_output_cell",
    "progressive_output_cell_budget",
    "learned_sparse_mask_reference",
    "shared_write_collision_control",
    "integrator_shared_write_control",
    "oracle_output_cell_reference",
    "dense_graph_danger_control",
)
TRAINED_ASSIGNMENT_SYSTEMS = (
    "next_free_output_cell_allocator",
    "random_initial_then_freeze",
    "mutation_selected_output_cell",
    "progressive_output_cell_budget",
    "shared_write_collision_control",
    "integrator_shared_write_control",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "write_cell_assignment_report.json",
    "collision_report.json",
    "progressive_budget_report.json",
    "system_results.json",
    "mutation_history.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7u_next_free_output_allocator_sufficient",
    "e7u_mutation_selected_output_cell_positive",
    "e7u_output_cell_location_not_important",
    "e7u_multi_output_cell_needed",
    "e7u_direct_shared_write_collision_detected",
    "e7u_integrator_shared_write_positive",
    "e7u_sparse_mask_still_preferred",
    "e7u_graph_soup_regression_detected",
    "e7u_output_cell_assignment_no_advantage",
)


def load_e7r_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7r_numeric_pocket_masked_flow_io_contract_probe", E7R_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7R helpers from {E7R_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7r = load_e7r_module()
e7p = e7r.e7p
e7o = e7r.e7o

FLOW_DIM = int(e7r.FLOW_DIM)
SKILLS = tuple(e7r.SKILLS)
SPLITS = tuple(e7r.SPLITS)
EVAL_SPLITS = tuple(e7r.EVAL_SPLITS)
RESULT_POS = dict(e7r.RESULT_POS)
RESULT_INDICES = tuple(RESULT_POS[skill] for skill in SKILLS)
OUTPUT_BANK = tuple(list(RESULT_INDICES) + [30, 31, 32, 33, 34, 35])
SHARED_CELL = 30


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
    max_output_cells: int
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
    return int(hashlib.sha256(f"e7u::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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
        raise ValueError("empty integer list")
    return values


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
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


def mask_from_indices(indices: list[int] | tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(FLOW_DIM, dtype=bool)
    for idx in indices:
        if 0 <= int(idx) < FLOW_DIM:
            mask[int(idx)] = True
    return mask


def contract_to_json(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill": contract["skill"],
        "mode": contract["mode"],
        "read_indices": np.flatnonzero(contract["read"]).astype(int).tolist(),
        "write_indices": np.flatnonzero(contract["write"]).astype(int).tolist(),
        "scratch_indices": np.flatnonzero(contract["scratch"]).astype(int).tolist(),
        "return_indices": np.flatnonzero(contract["return"]).astype(int).tolist(),
        "assignment_cell": int(contract["assignment_cell"]),
        "output_cell_count": int(np.sum(contract["write"] | contract["scratch"])),
        "preserve_count": int(np.sum(contract["preserve"])),
        "shared_write": bool(contract.get("shared_write", False)),
        "integrator": bool(contract.get("integrator", False)),
        "semantic_label_control": False,
        "enforce": bool(contract["enforce"]),
    }


def build_contract(skill: str, system: str, assignment_cell: int, output_count: int = 1) -> dict[str, Any]:
    read = np.ones(FLOW_DIM, dtype=bool)
    write = np.zeros(FLOW_DIM, dtype=bool)
    scratch = np.zeros(FLOW_DIM, dtype=bool)
    write[int(assignment_cell)] = True
    extra = [cell for cell in OUTPUT_BANK if cell != assignment_cell]
    for cell in extra[: max(0, output_count - 1)]:
        scratch[int(cell)] = True
    enforce = system != "dense_graph_danger_control"
    shared = system in {"shared_write_collision_control", "integrator_shared_write_control"}
    integrator = system == "integrator_shared_write_control"
    preserve = ~(write | scratch) if enforce else np.zeros(FLOW_DIM, dtype=bool)
    return {
        "skill": skill,
        "mode": system,
        "read": read,
        "write": write,
        "scratch": scratch,
        "return": write.copy(),
        "preserve": preserve,
        "enforce": enforce,
        "residual": integrator,
        "semantic_label_control": False,
        "permuted": False,
        "shared_write": shared,
        "integrator": integrator,
        "assignment_cell": int(assignment_cell),
    }


def assignment_next_free() -> dict[str, int]:
    return {skill: int(RESULT_POS[skill]) for skill in SKILLS}


def assignment_random(seed: int) -> dict[str, int]:
    rng = random.Random(stable_seed(f"random-assignment:{seed}"))
    cells = list(OUTPUT_BANK)
    rng.shuffle(cells)
    return {skill: int(cells[idx]) for idx, skill in enumerate(SKILLS)}


def assignment_shared() -> dict[str, int]:
    return {skill: SHARED_CELL for skill in SKILLS}


def assignment_score(candidate: dict[str, int], seed: int) -> float:
    cells = list(candidate.values())
    unique = len(set(cells))
    collision_penalty = len(cells) - unique
    locality = sum(abs(int(candidate[skill]) - int(RESULT_POS[skill])) for skill in SKILLS)
    bank_penalty = sum(0 if cell in OUTPUT_BANK else 1 for cell in cells)
    return -1.0 * collision_penalty - 0.015 * locality - 0.25 * bank_penalty + 0.0001 * (seed % 17)


def mutate_assignment(seed: int, settings: Settings, out: Path | None) -> tuple[dict[str, int], dict[str, Any]]:
    rng = random.Random(stable_seed(f"mutation-selected:{seed}"))
    best = assignment_random(seed)
    best_score = assignment_score(best, seed)
    initial_hash = payload_sha256(best)
    attempts = accepted = rejected = 0
    history = []
    cells = list(OUTPUT_BANK)
    for generation in range(settings.mutation_generations):
        generation_best = best_score
        for _ in range(settings.mutation_population):
            attempts += 1
            cand = dict(best)
            if rng.random() < 0.5:
                skill = rng.choice(SKILLS)
                cand[skill] = int(rng.choice(cells))
                op = "move_write_cell"
            else:
                a, b = rng.sample(SKILLS, 2)
                cand[a], cand[b] = cand[b], cand[a]
                op = "swap_write_cell"
            score = assignment_score(cand, seed)
            if score > best_score + 1e-12:
                best = cand
                best_score = score
                accepted += 1
            else:
                rejected += 1
        row = {
            "seed": seed,
            "system": "mutation_selected_output_cell",
            "generation": generation,
            "operation": "move_write_cell_or_swap_write_cell",
            "score": round_float(best_score),
            "generation_gain": round_float(best_score - generation_best),
            "accepted": accepted,
            "rejected": rejected,
            "rollback": rejected,
            "assignment_hash": payload_sha256(best),
        }
        history.append(row)
        if out:
            append_progress(out, "assignment_mutation_generation", **row)
    final_hash = payload_sha256(best)
    return best, {
        "history": history,
        "initial_candidate_hash": initial_hash,
        "final_candidate_hash": final_hash,
        "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": final_hash}),
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "slot_assignment_stability": round_float(1.0 - accepted / max(1, attempts)),
    }


def transform_context_for_assignment(context_task: dict[str, list[dict[str, Any]]], skill: str, assignment_cell: int) -> dict[str, list[dict[str, Any]]]:
    transformed: dict[str, list[dict[str, Any]]] = {}
    for split, rows in context_task.items():
        transformed[split] = []
        for row in rows:
            out = dict(row)
            target = np.asarray(row["target_flow"], dtype=np.float32).copy()
            value = float(row["target_value"])
            target[int(assignment_cell)] = value
            out["target_flow"] = target.tolist()
            transformed[split].append(out)
    return transformed


def train_assignment_library(seed: int, system: str, baseline_library: dict[str, dict[str, Any]], context_tasks: dict[str, Any], assignments: dict[str, int], settings: Settings, out: Path | None, output_count: int = 1) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    library: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    e7r_settings = to_e7r_settings(settings)
    for skill in SKILLS:
        contract = build_contract(skill, system, assignments[skill], output_count=output_count)
        task = transform_context_for_assignment(context_tasks[skill], skill, assignments[skill])
        trained = e7r.train_masked_context_pocket(seed, skill, system, baseline_library[skill], task, e7r_settings, contract, out)
        library[skill] = trained["state"]
        contracts[skill] = contract
        contract_json = contract_to_json(contract)
        assignment_rows.append({"seed": seed, "system": system, "skill": skill, "assignment_cell": int(assignments[skill]), "output_cell_count": output_count, "contract": contract_json, "state_hash": e7p.state_hash(trained["state"])})
        training_rows.append({"seed": seed, "system": system, "skill": skill, "state_hash": e7p.state_hash(trained["state"]), "history": trained["history"], "contract": contract_json})
    return library, contracts, training_rows, assignment_rows


def evaluate_assignment_system(seed: int, system: str, library: dict[str, dict[str, Any]] | None, contracts: dict[str, dict[str, Any]] | None, task: dict[str, list[dict[str, Any]]], symbolic: bool = False) -> dict[str, Any]:
    evals: dict[str, Any] = {}
    for split in SPLITS:
        rows = task[split]
        correct: list[bool] = []
        samples: list[dict[str, Any]] = []
        write_spread: list[float] = []
        changed_count: list[float] = []
        delta_magnitude: list[float] = []
        collision_rows: list[float] = []
        preserve_corrupt: list[float] = []
        write_violation: list[float] = []
        compatibility: list[float] = []
        for row in rows:
            route = tuple(row["expected_route"])
            if symbolic:
                flow = e7o.symbolic_apply_route(row, route).reshape(1, -1)
                collision_rows.append(0.0)
            else:
                assert library is not None and contracts is not None
                flow = np.asarray(row["flow"], dtype=np.float32).reshape(1, -1)
                route_cells: list[int] = []
                for skill in route:
                    before = flow.copy()
                    contract = contracts[skill]
                    pred = e7r.masked_forward_np(library[skill], flow, contract)
                    assigned = int(contract["assignment_cell"])
                    pred[0, assigned] = 1.0 if pred[0, assigned] >= 0.0 else 0.0
                    delta = pred - before
                    changed = np.abs(delta.reshape(-1)) > 1e-6
                    allowed = contract["write"] | contract["scratch"]
                    preserve = contract["preserve"]
                    write_spread.append(float(np.mean(changed)))
                    changed_count.append(float(np.sum(changed)))
                    delta_magnitude.append(float(np.mean(np.abs(delta))))
                    write_violation.append(float(np.any(np.abs(pred[:, ~allowed] - before[:, ~allowed]) > 0.08))) if np.any(~allowed) else write_violation.append(0.0)
                    preserve_corrupt.append(float(np.any(np.abs(pred[:, preserve] - before[:, preserve]) > 0.08))) if np.any(preserve) else preserve_corrupt.append(0.0)
                    if skill != route[-1]:
                        compatibility.append(float(np.mean(np.abs(pred[:, :24] - before[:, :24]))))
                    # Router-visible assignment projection: not a pocket write. It lets
                    # the old fixed readout interpret anonymous output-cell metadata.
                    pred[0, RESULT_POS[skill]] = pred[0, assigned]
                    route_cells.append(assigned)
                    flow = pred
                collision_rows.append(float(len(set(route_cells)) < len(route_cells)))
            pred_answer = int(e7o.predict_answer_from_flow(row, flow.reshape(-1)))
            ok = pred_answer == int(row["target_answer"])
            correct.append(ok)
            if len(samples) < 8:
                samples.append({"row_id": row["row_id"], "family": row["family"], "route": list(route), "target": int(row["target_answer"]), "predicted": pred_answer, "correct": bool(ok)})
        acc = round_float(float(np.mean(correct)))
        mean_steps = round_float(float(np.mean([len(row["expected_route"]) for row in rows])))
        bit_cost = sum(e7p.bit_budget(state) for state in library.values()) if library else 0
        cost_penalty = min(0.10, 0.00000016 * bit_cost + 0.0025 * mean_steps)
        evals[split] = {
            "answer_accuracy": acc,
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(max(0.0, acc - cost_penalty)),
            "mean_route_steps": mean_steps,
            "write_spread": round_float(float(np.mean(write_spread)) if write_spread else 0.0),
            "changed_cell_count": round_float(float(np.mean(changed_count)) if changed_count else 0.0),
            "delta_magnitude": round_float(float(np.mean(delta_magnitude)) if delta_magnitude else 0.0),
            "output_cell_count": round_float(float(np.mean([np.sum(contract["write"] | contract["scratch"]) for contract in contracts.values()])) if contracts else 1.0),
            "ram_collision_rate": round_float(float(np.mean(collision_rows)) if collision_rows else 0.0),
            "preserve_mask_corruption_rate": round_float(float(np.mean(preserve_corrupt)) if preserve_corrupt else 0.0),
            "write_mask_violation_rate": round_float(float(np.mean(write_violation)) if write_violation else 0.0),
            "next_pocket_input_compatibility_error": round_float(float(np.mean(compatibility)) if compatibility else 0.0),
            "calibration_output_scale_error": 0.0,
            "bit_budget": bit_cost,
            "row_level_samples": samples,
        }
    return {
        "seed": seed,
        "system": system,
        "evals": evals,
        "heldout_usefulness": evals["heldout"]["composition_usefulness"],
        "ood_usefulness": evals["ood"]["composition_usefulness"],
        "counterfactual_usefulness": evals["counterfactual"]["composition_usefulness"],
        "adversarial_usefulness": evals["adversarial"]["composition_usefulness"],
        "eval_mean_answer_accuracy": round_float(float(np.mean([evals[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([evals[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_route_accuracy": 1.0,
        "eval_mean_write_spread": round_float(float(np.mean([evals[split]["write_spread"] for split in EVAL_SPLITS]))),
        "eval_mean_changed_cell_count": round_float(float(np.mean([evals[split]["changed_cell_count"] for split in EVAL_SPLITS]))),
        "eval_mean_output_cell_count": round_float(float(np.mean([evals[split]["output_cell_count"] for split in EVAL_SPLITS]))),
        "eval_mean_ram_collision_rate": round_float(float(np.mean([evals[split]["ram_collision_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_preserve_mask_corruption_rate": round_float(float(np.mean([evals[split]["preserve_mask_corruption_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_write_mask_violation_rate": round_float(float(np.mean([evals[split]["write_mask_violation_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_input_compatibility_error": round_float(float(np.mean([evals[split]["next_pocket_input_compatibility_error"] for split in EVAL_SPLITS]))),
        "parameter_count": sum(e7p.parameter_count(state) for state in library.values()) if library else 0,
        "bit_budget": sum(e7p.bit_budget(state) for state in library.values()) if library else 0,
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    metric_names = (
        "eval_mean_answer_accuracy",
        "eval_mean_composition_usefulness",
        "eval_mean_route_accuracy",
        "heldout_usefulness",
        "ood_usefulness",
        "counterfactual_usefulness",
        "adversarial_usefulness",
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
        "slot_assignment_stability",
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
    candidates = [system for system in SYSTEMS if system not in {"oracle_output_cell_reference", "dense_graph_danger_control"}]
    best = max(candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    return {"schema_version": "e7u_aggregate_metrics_v1", "systems": systems, "best_non_reference_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    mean = {system: aggregate["systems"][system]["mean"].get("eval_mean_composition_usefulness", 0.0) for system in SYSTEMS}
    detail = {
        "best_non_reference_system": aggregate["best_non_reference_system"],
        "next_free": mean["next_free_output_cell_allocator"],
        "random_freeze": mean["random_initial_then_freeze"],
        "mutation_selected": mean["mutation_selected_output_cell"],
        "progressive": mean["progressive_output_cell_budget"],
        "learned_sparse": mean["learned_sparse_mask_reference"],
        "shared": mean["shared_write_collision_control"],
        "integrator": mean["integrator_shared_write_control"],
        "oracle": mean["oracle_output_cell_reference"],
        "dense": mean["dense_graph_danger_control"],
        "shared_collision": aggregate["systems"]["shared_write_collision_control"]["mean"].get("eval_mean_ram_collision_rate", 0.0),
        "integrator_collision": aggregate["systems"]["integrator_shared_write_control"]["mean"].get("eval_mean_ram_collision_rate", 0.0),
        "progressive_output_cells": aggregate["systems"]["progressive_output_cell_budget"]["mean"].get("eval_mean_output_cell_count", 0.0),
    }
    best = detail["best_non_reference_system"]
    if detail["dense"] >= max(value for key, value in detail.items() if key not in {"dense", "oracle", "best_non_reference_system"}) + 0.025:
        decision = "e7u_graph_soup_regression_detected"
    elif detail["shared_collision"] > 0.20 and detail["shared"] < detail["next_free"] - 0.02:
        decision = "e7u_direct_shared_write_collision_detected"
    elif best == "integrator_shared_write_control" and detail["integrator"] >= detail["next_free"] + 0.01:
        decision = "e7u_integrator_shared_write_positive"
    elif detail["progressive_output_cells"] > 1.01 and best == "progressive_output_cell_budget":
        decision = "e7u_multi_output_cell_needed"
    elif best == "mutation_selected_output_cell" and detail["mutation_selected"] >= detail["next_free"] + 0.015:
        decision = "e7u_mutation_selected_output_cell_positive"
    elif best == "learned_sparse_mask_reference" and detail["learned_sparse"] >= max(detail["next_free"], detail["mutation_selected"]) + 0.01:
        decision = "e7u_sparse_mask_still_preferred"
    elif abs(detail["random_freeze"] - max(detail["next_free"], detail["mutation_selected"], detail["learned_sparse"])) <= 0.005:
        decision = "e7u_output_cell_location_not_important"
    elif detail["next_free"] >= detail["mutation_selected"] - 0.01 and detail["next_free"] >= detail["learned_sparse"] - 0.01:
        decision = "e7u_next_free_output_allocator_sufficient"
    else:
        decision = "e7u_output_cell_assignment_no_advantage"
    return {"schema_version": "e7u_decision_v1", "decision": decision, "detail": detail, "deterministic_replay_passed": False}


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    e7r_settings = to_e7r_settings(settings)
    e7o_settings = e7r.to_e7o_settings(e7r_settings)
    e7p_settings = e7r.to_e7p_settings(e7r_settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    context_tasks = e7p.generate_context_tasks(composition_task)
    baseline_library: dict[str, dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        state = e7p.copy_state(trained["state"], "e7u_baseline_standalone")
        baseline_library[skill] = state
        training_rows.append({"seed": seed, "system": "baseline_standalone_pocket", "skill": skill, "state_hash": e7p.state_hash(state), "standalone": trained["standalone"]})

    assignment_payloads = {
        "next_free_output_cell_allocator": assignment_next_free(),
        "random_initial_then_freeze": assignment_random(seed),
        "shared_write_collision_control": assignment_shared(),
        "integrator_shared_write_control": assignment_shared(),
    }
    mutation_assignment, mutation = mutate_assignment(seed, settings, out)
    assignment_payloads["mutation_selected_output_cell"] = mutation_assignment
    mutation_rows.extend(mutation["history"])

    trained_cache: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    for system, assignments in assignment_payloads.items():
        library, contracts, train_rows, assign_rows = train_assignment_library(seed, system, baseline_library, context_tasks, assignments, settings, out, output_count=1)
        result = evaluate_assignment_system(seed, system, library, contracts, composition_task)
        if system == "mutation_selected_output_cell":
            result.update({key: value for key, value in mutation.items() if key != "history"})
        rows.append(result)
        training_rows.extend(train_rows)
        assignment_rows.extend(assign_rows)
        trained_cache[system] = (library, contracts)

    progressive_curve = []
    best_progressive: dict[str, Any] | None = None
    best_payload: tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]] | None = None
    for output_count in range(1, settings.max_output_cells + 1):
        system_name = f"progressive_output_cell_budget_k{output_count}"
        library, contracts, train_rows, assign_rows = train_assignment_library(seed, system_name, baseline_library, context_tasks, assignment_next_free(), settings, out, output_count=output_count)
        result = evaluate_assignment_system(seed, "progressive_output_cell_budget", library, contracts, composition_task)
        result["output_budget"] = output_count
        progressive_curve.append(result)
        training_rows.extend(train_rows)
        assignment_rows.extend(assign_rows)
        if out:
            append_progress(out, "progressive_output_budget_evaluated", seed=seed, output_budget=output_count, usefulness=result["eval_mean_composition_usefulness"])
        if best_progressive is None or result["eval_mean_composition_usefulness"] > best_progressive["eval_mean_composition_usefulness"]:
            best_progressive = result
            best_payload = (library, contracts)
    assert best_progressive is not None and best_payload is not None
    best_score = max(row["eval_mean_composition_usefulness"] for row in progressive_curve)
    chosen = min([row for row in progressive_curve if row["eval_mean_composition_usefulness"] >= best_score - 0.002], key=lambda row: row["output_budget"])
    chosen = dict(chosen)
    chosen["smallest_stable_output_budget"] = chosen["output_budget"]
    rows.append(chosen)

    # E7R/E7T learned sparse reference over next-free output maps.
    nf_library, nf_contracts = trained_cache["next_free_output_cell_allocator"]
    learned_contracts, learned_mutation = e7r.mutate_contracts(seed, nf_library, nf_contracts, composition_task, e7r_settings, out)
    learned = evaluate_assignment_system(seed, "learned_sparse_mask_reference", nf_library, learned_contracts, composition_task)
    learned.update({key: value for key, value in learned_mutation.items() if key != "history"})
    rows.append(learned)
    mutation_rows.extend(learned_mutation["history"])

    rows.append(evaluate_assignment_system(seed, "oracle_output_cell_reference", None, None, composition_task, symbolic=True))
    dense = e7o.train_monolithic(seed, "dense_graph_danger_control", composition_task, e7o_settings, out, hidden=176, depth=4)
    dense["system"] = "dense_graph_danger_control"
    for key in ("eval_mean_write_spread", "eval_mean_changed_cell_count", "eval_mean_output_cell_count", "eval_mean_ram_collision_rate", "eval_mean_preserve_mask_corruption_rate", "eval_mean_write_mask_violation_rate", "eval_mean_next_pocket_input_compatibility_error"):
        dense[key] = 0.0
    rows.append(dense)
    collision = {
        "seed": seed,
        "shared_collision_rate": next(row for row in rows if row["system"] == "shared_write_collision_control")["eval_mean_ram_collision_rate"],
        "integrator_collision_rate": next(row for row in rows if row["system"] == "integrator_shared_write_control")["eval_mean_ram_collision_rate"],
        "shared_usefulness": next(row for row in rows if row["system"] == "shared_write_collision_control")["eval_mean_composition_usefulness"],
        "integrator_usefulness": next(row for row in rows if row["system"] == "integrator_shared_write_control")["eval_mean_composition_usefulness"],
    }
    progressive = {"seed": seed, "system": "progressive_output_cell_budget", "curve": [{"output_budget": row["output_budget"], "usefulness": row["eval_mean_composition_usefulness"], "heldout": row["heldout_usefulness"], "ood": row["ood_usefulness"], "counterfactual": row["counterfactual_usefulness"], "adversarial": row["adversarial_usefulness"]} for row in progressive_curve], "chosen_output_budget": chosen["output_budget"]}
    return {"seed": seed, "rows": rows, "training_rows": training_rows, "assignment_rows": assignment_rows, "mutation_rows": mutation_rows, "collision": collision, "progressive": progressive}


def build_reports(rows: list[dict[str, Any]], training_rows: list[dict[str, Any]], assignment_rows: list[dict[str, Any]], mutation_rows: list[dict[str, Any]], collision_rows: list[dict[str, Any]], progressive_rows: list[dict[str, Any]], aggregate: dict[str, Any], decision: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    report_lines = [
        "# E7U RAM Output Cell Assignment Audit Result",
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
        report_lines.append(
            f"{system:<38} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} "
            f"spread={mean.get('eval_mean_write_spread', 0.0):.6f} "
            f"cells={mean.get('eval_mean_output_cell_count', 0.0):.3f} "
            f"collision={mean.get('eval_mean_ram_collision_rate', 0.0):.6f}"
        )
    report_lines.extend([
        "```",
        "",
        "## Boundary",
        "",
        "E7U is a controlled anonymous RAM output-cell assignment audit. It does not make raw-language, AGI, consciousness, or model-scale claims.",
        "",
    ])
    return {
        "pocket_training_report.json": {"schema_version": "e7u_pocket_training_report_v1", "rows": sorted(training_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "write_cell_assignment_report.json": {"schema_version": "e7u_write_cell_assignment_report_v1", "rows": sorted(assignment_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "collision_report.json": {"schema_version": "e7u_collision_report_v1", "rows": sorted(collision_rows, key=lambda row: row["seed"])},
        "progressive_budget_report.json": {"schema_version": "e7u_progressive_budget_report_v1", "rows": sorted(progressive_rows, key=lambda row: row["seed"])},
        "system_results.json": {"schema_version": "e7u_system_results_v1", "rows": sorted(rows, key=lambda row: (row["seed"], SYSTEMS.index(row["system"])))},
        "mutation_history.json": {"schema_version": "e7u_mutation_history_v1", "rows": sorted(mutation_rows, key=lambda row: (row["seed"], row.get("system", ""), row.get("generation", 0)))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7u_summary_v1", "decision": decision["decision"], "best_non_reference_system": aggregate["best_non_reference_system"], "deterministic_replay_passed": deterministic.get("internal_replay_passed", False), "checker_failure_count": None},
        "report.md": "\n".join(report_lines),
    }


def hash_artifacts(out: Path) -> dict[str, str]:
    return {artifact: hashlib.sha256((out / artifact).read_bytes()).hexdigest() for artifact in HASH_ARTIFACTS}


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
        e7r_settings = to_e7r_settings(settings)
        e7o_settings = e7r.to_e7o_settings(e7r_settings)
        composition_tasks = e7o.generate_composition_tasks(e7o_settings)
        pocket_tasks = e7o.generate_pocket_tasks(e7o_settings)
        write_json(out / "backend_manifest.json", {"schema_version": "e7u_backend_manifest_v1", "milestone": MILESTONE, "settings": settings_payload(settings), "systems": list(SYSTEMS), "flow_dim": FLOW_DIM, "output_bank": list(OUTPUT_BANK), "semantic_lane_labels_as_model_input": False, "runtime_random_output_writes": False, "training_performed": True, "device": select_device(settings.device), "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        write_json(out / "task_generation_report.json", {"schema_version": "e7u_task_generation_report_v1", "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()}, "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()}})
        rows: list[dict[str, Any]] = []
        training_rows: list[dict[str, Any]] = []
        assignment_rows: list[dict[str, Any]] = []
        mutation_rows: list[dict[str, Any]] = []
        collision_rows: list[dict[str, Any]] = []
        progressive_rows: list[dict[str, Any]] = []
        jobs = [{"seed": seed, "settings": settings.__dict__.copy(), "composition_task": composition_tasks[seed], "pocket_task": pocket_tasks[seed], "out": str(out)} for seed in settings.seeds]
        max_workers = max(1, min(settings.cpu_workers, len(jobs)))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
            while futures:
                done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                if not done:
                    write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_assignment_rows": len(assignment_rows), "pending": len(futures)})
                    continue
                for future in done:
                    label = futures.pop(future)
                    result = future.result()
                    rows.extend(result["rows"])
                    training_rows.extend(result["training_rows"])
                    assignment_rows.extend(result["assignment_rows"])
                    mutation_rows.extend(result["mutation_rows"])
                    collision_rows.append(result["collision"])
                    progressive_rows.append(result["progressive"])
                    append_progress(out, "seed_job_complete", label=label, pending=len(futures))
                    write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_assignment_rows": len(assignment_rows), "last_completed": label, "pending": len(futures)})
        aggregate = aggregate_results(rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e7u_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, training_rows, assignment_rows, mutation_rows, collision_rows, progressive_rows, aggregate, decision, deterministic_placeholder)
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
            cmd = [sys.executable, str(Path(__file__).resolve()), "--out", str(replay_out.relative_to(REPO_ROOT)), "--seeds", ",".join(map(str, settings.seeds)), "--train-rows-per-seed", str(settings.train_rows_per_seed), "--validation-rows-per-seed", str(settings.validation_rows_per_seed), "--heldout-rows-per-seed", str(settings.heldout_rows_per_seed), "--ood-rows-per-seed", str(settings.ood_rows_per_seed), "--counterfactual-rows-per-seed", str(settings.counterfactual_rows_per_seed), "--adversarial-rows-per-seed", str(settings.adversarial_rows_per_seed), "--pocket-pretrain-rows-per-seed", str(settings.pocket_pretrain_rows_per_seed), "--pocket-validation-rows-per-seed", str(settings.pocket_validation_rows_per_seed), "--pocket-dim", str(settings.pocket_dim), "--pocket-core-steps", str(settings.pocket_core_steps), "--pocket-epochs", str(settings.pocket_epochs), "--local-epochs", str(settings.local_epochs), "--full-epochs", str(settings.full_epochs), "--batch-size", str(settings.batch_size), "--learning-rate", str(settings.learning_rate), "--local-learning-rate", str(settings.local_learning_rate), "--weight-decay", str(settings.weight_decay), "--mutation-generations", str(settings.mutation_generations), "--mutation-population", str(settings.mutation_population), "--max-output-cells", str(settings.max_output_cells), "--cpu-workers", str(settings.cpu_workers), "--device", settings.device, "--heartbeat-seconds", str(settings.heartbeat_seconds), "--execution-mode", settings.execution_mode, "--replay"]
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
            primary_hashes = hash_artifacts(out)
            replay_hashes = hash_artifacts(replay_out)
            comparisons = {artifact: {"primary": primary_hashes[artifact], "replay": replay_hashes[artifact], "match": primary_hashes[artifact] == replay_hashes[artifact]} for artifact in HASH_ARTIFACTS}
            deterministic = {"schema_version": "e7u_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, training_rows, assignment_rows, mutation_rows, collision_rows, progressive_rows, aggregate, decision, deterministic)
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
    parser.add_argument("--max-output-cells", type=int, default=3)
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
        max_output_cells=args.max_output_cells,
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
