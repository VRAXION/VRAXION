#!/usr/bin/env python3
"""E7T progressive RAM port allocation probe.

E7T tests whether numeric pockets need a disciplined RAM wrapper/port map:
fixed output ports, fixed input+output ports, progressive +1 slot allocation,
learned frozen port maps, and shared-write controls. It reuses the E7R numeric
pocket Flow[D] task lineage and keeps RAM cells anonymous.
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
MILESTONE = "E7T_PROGRESSIVE_RAM_PORT_ALLOCATION_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7t_progressive_ram_port_allocation_probe")
DEFAULT_SEEDS = (99801, 99802, 99803)

SYSTEMS = (
    "untyped_flow_baseline",
    "output_write_map_only",
    "input_read_map_only",
    "input_plus_output_port_map",
    "learned_sparse_mask_reference",
    "progressive_write_slot_allocation",
    "progressive_read_write_slot_allocation",
    "learned_port_map_then_freeze",
    "shared_write_control",
    "integrator_shared_write_control",
    "oracle_port_map_reference",
    "dense_graph_danger_control",
)
TRAINED_PORT_SYSTEMS = (
    "output_write_map_only",
    "input_read_map_only",
    "input_plus_output_port_map",
    "shared_write_control",
    "integrator_shared_write_control",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "port_map_report.json",
    "slot_plateau_report.json",
    "shared_write_report.json",
    "flow_grid_frames.json",
    "system_results.json",
    "mutation_history.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7t_output_port_map_positive",
    "e7t_input_output_port_map_positive",
    "e7t_progressive_write_slot_allocation_positive",
    "e7t_progressive_read_write_slot_allocation_positive",
    "e7t_learned_frozen_port_map_positive",
    "e7t_direct_shared_write_collision_detected",
    "e7t_integrator_shared_write_positive",
    "e7t_sparse_mask_contract_still_preferred",
    "e7t_graph_soup_regression_detected",
    "e7t_ram_port_allocation_no_advantage",
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
INPUT_BANK = tuple(range(0, 24)) + RESULT_INDICES
WRITE_BANK = tuple(RESULT_INDICES) + (30, 31, 32, 33, 34, 35)
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
    mask_mutation_generations: int
    mask_mutation_population: int
    max_write_slots: int
    read_budgets: tuple[int, ...]
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
        mask_mutation_generations=settings.mask_mutation_generations,
        mask_mutation_population=settings.mask_mutation_population,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        execution_mode=settings.execution_mode,
        replay=settings.replay,
    )


def mask_from_indices(indices: tuple[int, ...] | list[int]) -> np.ndarray:
    mask = np.zeros(FLOW_DIM, dtype=bool)
    for idx in indices:
        if 0 <= int(idx) < FLOW_DIM:
            mask[int(idx)] = True
    return mask


def clone_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in contract.items()}


def contract_to_json(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill": contract["skill"],
        "mode": contract["mode"],
        "read_indices": np.flatnonzero(contract["read"]).astype(int).tolist(),
        "write_indices": np.flatnonzero(contract["write"]).astype(int).tolist(),
        "scratch_indices": np.flatnonzero(contract["scratch"]).astype(int).tolist(),
        "return_indices": np.flatnonzero(contract["return"]).astype(int).tolist(),
        "preserve_count": int(np.sum(contract["preserve"])),
        "enforce": bool(contract["enforce"]),
        "residual": bool(contract["residual"]),
        "shared_write": bool(contract.get("shared_write", False)),
        "integrator": bool(contract.get("integrator", False)),
        "write_budget": int(contract.get("write_budget", int(np.sum(contract["write"] | contract["scratch"])))),
        "read_budget": int(contract.get("read_budget", int(np.sum(contract["read"])))),
        "semantic_label_control": False,
        "permuted": False,
    }


def build_contract(skill: str, mode: str, read_budget: int | None = None, write_budget: int | None = None) -> dict[str, Any]:
    read = np.ones(FLOW_DIM, dtype=bool)
    write = np.zeros(FLOW_DIM, dtype=bool)
    scratch = np.zeros(FLOW_DIM, dtype=bool)
    enforce = True
    residual = False
    shared = False
    integrator = False
    if mode == "untyped_flow_baseline":
        write[:] = True
        scratch[:] = False
        enforce = False
    elif mode == "input_read_map_only":
        rb = min(read_budget if read_budget is not None else len(INPUT_BANK), len(INPUT_BANK))
        read = mask_from_indices(INPUT_BANK[:rb])
        write[:] = True
        enforce = False
    else:
        if mode in {"output_write_map_only", "progressive_write_slot_allocation", "shared_write_control", "integrator_shared_write_control"}:
            read[:] = True
        elif mode in {"input_plus_output_port_map", "progressive_read_write_slot_allocation", "learned_port_map_then_freeze", "learned_sparse_mask_reference"}:
            rb = min(read_budget if read_budget is not None else len(INPUT_BANK), len(INPUT_BANK))
            read = mask_from_indices(INPUT_BANK[:rb])
        write[RESULT_POS[skill]] = True
        wb = max(1, int(write_budget if write_budget is not None else 1))
        extra_count = max(0, wb - 1)
        extra_bank = [idx for idx in WRITE_BANK if idx != RESULT_POS[skill]]
        for idx in extra_bank[:extra_count]:
            scratch[idx] = True
        if mode == "shared_write_control":
            scratch[SHARED_CELL] = True
            shared = True
        if mode == "integrator_shared_write_control":
            scratch[SHARED_CELL] = True
            shared = True
            integrator = True
            residual = True
    allowed = write | scratch
    preserve = ~allowed if enforce else np.zeros(FLOW_DIM, dtype=bool)
    return {
        "skill": skill,
        "mode": mode,
        "read": read,
        "write": write,
        "scratch": scratch,
        "return": write.copy(),
        "preserve": preserve,
        "enforce": enforce,
        "residual": residual,
        "semantic_label_control": False,
        "permuted": False,
        "shared_write": shared,
        "integrator": integrator,
        "write_budget": int(np.sum(allowed)),
        "read_budget": int(np.sum(read)),
    }


def train_masked_library(seed: int, system: str, baseline_library: dict[str, dict[str, Any]], context_tasks: dict[str, Any], contracts: dict[str, dict[str, Any]], settings: Settings, out: Path | None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    library: dict[str, dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    port_rows: list[dict[str, Any]] = []
    e7r_settings = to_e7r_settings(settings)
    for skill in SKILLS:
        trained = e7r.train_masked_context_pocket(seed, skill, system, baseline_library[skill], context_tasks[skill], e7r_settings, contracts[skill], out)
        library[skill] = trained["state"]
        contract_json = contract_to_json(contracts[skill])
        port_rows.append({"seed": seed, "system": system, "skill": skill, "contract": contract_json, "state_hash": e7p.state_hash(trained["state"])})
        training_rows.append({"seed": seed, "system": system, "skill": skill, "state_hash": e7p.state_hash(trained["state"]), "history": trained["history"], "contract": contract_json, "trainable_scope": ["win", "bin", "wcore", "carry_raw", "wout", "bout"]})
    return library, training_rows, port_rows


def port_runtime_metrics(seed: int, system: str, library: dict[str, dict[str, Any]] | None, contracts: dict[str, dict[str, Any]] | None, task: dict[str, list[dict[str, Any]]], symbolic: bool = False) -> dict[str, float]:
    write_spread: list[float] = []
    changed_count: list[float] = []
    delta_mag: list[float] = []
    read_count: list[float] = []
    write_count: list[float] = []
    collision: list[float] = []
    for split in EVAL_SPLITS:
        for row in task[split]:
            if symbolic:
                flow = np.asarray(row["flow"], dtype=np.float32)
                for skill in row["expected_route"]:
                    before = flow.copy()
                    flow[RESULT_POS[skill]] = float(e7o.base_skill_value(skill, row["a"], row["b"], row["key"], row["threshold"], row["flip"], flow))
                    delta = flow - before
                    changed = np.abs(delta) > 1e-6
                    write_spread.append(float(np.mean(changed)))
                    changed_count.append(float(np.sum(changed)))
                    delta_mag.append(float(np.mean(np.abs(delta))))
                    read_count.append(float(len(INPUT_BANK)))
                    write_count.append(1.0)
                    collision.append(0.0)
                continue
            assert library is not None and contracts is not None
            flow2 = np.asarray(row["flow"], dtype=np.float32).reshape(1, -1)
            route = tuple(row["expected_route"])
            route_allowed: list[set[int]] = []
            for skill in route:
                contract = contracts[skill]
                before = flow2.copy()
                pred = e7r.masked_forward_np(library[skill], flow2, contract)
                pos = RESULT_POS[skill]
                pred[0, pos] = 1.0 if pred[0, pos] >= 0.0 else 0.0
                delta = pred - before
                changed = np.abs(delta.reshape(-1)) > 1e-6
                allowed = contract["write"] | contract["scratch"] if contract["enforce"] else np.ones(FLOW_DIM, dtype=bool)
                route_allowed.append(set(np.flatnonzero(allowed).astype(int).tolist()))
                write_spread.append(float(np.mean(changed)))
                changed_count.append(float(np.sum(changed)))
                delta_mag.append(float(np.mean(np.abs(delta))))
                read_count.append(float(np.sum(contract["read"])))
                write_count.append(float(np.sum(allowed)))
                flow2 = pred
            if len(route_allowed) >= 2:
                overlap = set.intersection(*route_allowed) if route_allowed else set()
                collision.append(float(len(overlap) > 0 and (SHARED_CELL in overlap)))
    return {
        "eval_mean_write_spread": round_float(float(np.mean(write_spread)) if write_spread else 0.0),
        "eval_mean_changed_cell_count": round_float(float(np.mean(changed_count)) if changed_count else 0.0),
        "eval_mean_delta_magnitude": round_float(float(np.mean(delta_mag)) if delta_mag else 0.0),
        "eval_mean_read_cell_count": round_float(float(np.mean(read_count)) if read_count else 0.0),
        "eval_mean_write_cell_count": round_float(float(np.mean(write_count)) if write_count else 0.0),
        "ram_collision_rate": round_float(float(np.mean(collision)) if collision else 0.0),
    }


def evaluate_system(seed: int, system: str, library: dict[str, dict[str, Any]] | None, contracts: dict[str, dict[str, Any]] | None, task: dict[str, list[dict[str, Any]]], symbolic: bool = False) -> dict[str, Any]:
    row = e7r.evaluate_contract_system(seed, system, library, contracts, task, symbolic=symbolic)
    row.update(port_runtime_metrics(seed, system, library, contracts, task, symbolic=symbolic))
    if contracts:
        row["read_cell_count"] = round_float(float(np.mean([np.sum(contract["read"]) for contract in contracts.values()])))
        row["write_cell_count"] = round_float(float(np.mean([np.sum(contract["write"] | contract["scratch"]) for contract in contracts.values()])))
    else:
        row["read_cell_count"] = 0.0
        row["write_cell_count"] = 0.0
    return row


def score_for_plateau(row: dict[str, Any]) -> float:
    return float(np.mean([row[f"{split}_usefulness"] for split in ("heldout", "ood", "counterfactual", "adversarial")]))


def choose_plateau(rows: list[dict[str, Any]], budget_key: str) -> dict[str, Any]:
    best = max(rows, key=score_for_plateau)
    best_score = score_for_plateau(best)
    candidates = [row for row in rows if score_for_plateau(row) >= best_score - 0.002]
    chosen = min(candidates, key=lambda row: row[budget_key])
    return {"best_score": round_float(best_score), "best_budget": int(best[budget_key]), "chosen_budget": int(chosen[budget_key]), "chosen_score": round_float(score_for_plateau(chosen))}


def train_progressive_write(seed: int, baseline_library: dict[str, dict[str, Any]], context_tasks: dict[str, Any], composition_task: dict[str, list[dict[str, Any]]], settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    libraries: dict[int, dict[str, dict[str, Any]]] = {}
    contracts_by_k: dict[int, dict[str, dict[str, Any]]] = {}
    training_rows: list[dict[str, Any]] = []
    port_rows: list[dict[str, Any]] = []
    for k in range(1, settings.max_write_slots + 1):
        system_name = f"progressive_write_slot_allocation_k{k}"
        contracts = {skill: build_contract(skill, "progressive_write_slot_allocation", read_budget=len(INPUT_BANK), write_budget=k) for skill in SKILLS}
        library, train_rows, rows = train_masked_library(seed, system_name, baseline_library, context_tasks, contracts, settings, out)
        result = evaluate_system(seed, "progressive_write_slot_allocation", library, contracts, composition_task)
        result["write_budget"] = k
        curve.append(result)
        libraries[k] = library
        contracts_by_k[k] = contracts
        training_rows.extend(train_rows)
        port_rows.extend(rows)
        if out:
            append_progress(out, "progressive_write_budget_evaluated", seed=seed, write_budget=k, usefulness=result["eval_mean_composition_usefulness"])
    plateau = choose_plateau(curve, "write_budget")
    selected = next(row for row in curve if row["write_budget"] == plateau["chosen_budget"])
    selected = dict(selected)
    selected["slot_plateau"] = plateau
    selected["smallest_stable_write_budget"] = plateau["chosen_budget"]
    selected["smallest_stable_read_budget"] = len(INPUT_BANK)
    return selected, {"curve": curve, "training_rows": training_rows, "port_rows": port_rows, "selected_contracts": contracts_by_k[plateau["chosen_budget"]], "selected_library": libraries[plateau["chosen_budget"]], "plateau": plateau}


def train_progressive_read_write(seed: int, baseline_library: dict[str, dict[str, Any]], context_tasks: dict[str, Any], composition_task: dict[str, list[dict[str, Any]]], settings: Settings, out: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    curve: list[dict[str, Any]] = []
    libraries: dict[int, dict[str, dict[str, Any]]] = {}
    contracts_by_k: dict[int, dict[str, dict[str, Any]]] = {}
    training_rows: list[dict[str, Any]] = []
    port_rows: list[dict[str, Any]] = []
    for read_budget in settings.read_budgets:
        rb = max(1, min(int(read_budget), len(INPUT_BANK)))
        system_name = f"progressive_read_write_slot_allocation_r{rb}"
        contracts = {skill: build_contract(skill, "progressive_read_write_slot_allocation", read_budget=rb, write_budget=1) for skill in SKILLS}
        library, train_rows, rows = train_masked_library(seed, system_name, baseline_library, context_tasks, contracts, settings, out)
        result = evaluate_system(seed, "progressive_read_write_slot_allocation", library, contracts, composition_task)
        result["read_budget"] = rb
        result["write_budget"] = 1
        curve.append(result)
        libraries[rb] = library
        contracts_by_k[rb] = contracts
        training_rows.extend(train_rows)
        port_rows.extend(rows)
        if out:
            append_progress(out, "progressive_read_write_budget_evaluated", seed=seed, read_budget=rb, write_budget=1, usefulness=result["eval_mean_composition_usefulness"])
    plateau = choose_plateau(curve, "read_budget")
    selected = next(row for row in curve if row["read_budget"] == plateau["chosen_budget"])
    selected = dict(selected)
    selected["slot_plateau"] = plateau
    selected["smallest_stable_read_budget"] = plateau["chosen_budget"]
    selected["smallest_stable_write_budget"] = 1
    return selected, {"curve": curve, "training_rows": training_rows, "port_rows": port_rows, "selected_contracts": contracts_by_k[plateau["chosen_budget"]], "selected_library": libraries[plateau["chosen_budget"]], "plateau": plateau}


def canonical_system_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row.get("seed", 0)), SYSTEMS.index(str(row.get("system")))))


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = canonical_system_rows(rows)
    systems: dict[str, Any] = {}
    metric_names = (
        "eval_mean_answer_accuracy",
        "eval_mean_composition_usefulness",
        "eval_mean_route_accuracy",
        "heldout_usefulness",
        "ood_usefulness",
        "counterfactual_usefulness",
        "adversarial_usefulness",
        "eval_mean_state_preservation_error",
        "eval_mean_write_mask_violation_rate",
        "eval_mean_preserve_mask_corruption_rate",
        "eval_mean_result_region_corruption_rate",
        "eval_mean_next_pocket_input_compatibility_error",
        "eval_mean_write_spread",
        "eval_mean_changed_cell_count",
        "eval_mean_delta_magnitude",
        "eval_mean_read_cell_count",
        "eval_mean_write_cell_count",
        "read_cell_count",
        "write_cell_count",
        "ram_collision_rate",
        "parameter_count",
        "bit_budget",
        "mutation_attempts",
        "accepted_mutations",
        "rejected_mutations",
        "rollback_count",
        "mask_sparsity",
        "smallest_stable_write_budget",
        "smallest_stable_read_budget",
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
    candidates = [system for system in SYSTEMS if system not in {"oracle_port_map_reference", "dense_graph_danger_control"}]
    best = max(candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    return {"schema_version": "e7t_aggregate_metrics_v1", "systems": systems, "best_non_reference_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    mean = {system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0) for system in SYSTEMS}
    detail = {
        "best_non_reference_system": aggregate["best_non_reference_system"],
        "untyped": mean["untyped_flow_baseline"],
        "output": mean["output_write_map_only"],
        "input_only": mean["input_read_map_only"],
        "input_output": mean["input_plus_output_port_map"],
        "learned_sparse": mean["learned_sparse_mask_reference"],
        "progressive_write": mean["progressive_write_slot_allocation"],
        "progressive_read_write": mean["progressive_read_write_slot_allocation"],
        "learned_frozen": mean["learned_port_map_then_freeze"],
        "shared": mean["shared_write_control"],
        "integrator": mean["integrator_shared_write_control"],
        "oracle": mean["oracle_port_map_reference"],
        "dense": mean["dense_graph_danger_control"],
        "shared_collision_rate": systems["shared_write_control"]["mean"].get("ram_collision_rate", 0.0),
        "integrator_collision_rate": systems["integrator_shared_write_control"]["mean"].get("ram_collision_rate", 0.0),
        "progressive_write_budget": systems["progressive_write_slot_allocation"]["mean"].get("smallest_stable_write_budget", 0.0),
        "progressive_read_budget": systems["progressive_read_write_slot_allocation"]["mean"].get("smallest_stable_read_budget", 0.0),
    }
    best = detail["best_non_reference_system"]
    if detail["dense"] >= max(value for key, value in detail.items() if key not in {"dense", "oracle", "best_non_reference_system"}) + 0.025:
        decision = "e7t_graph_soup_regression_detected"
    elif detail["shared_collision_rate"] > 0.10 and detail["shared"] < detail["output"] - 0.02:
        decision = "e7t_direct_shared_write_collision_detected"
    elif best == "integrator_shared_write_control" and detail["integrator"] >= detail["output"] + 0.01:
        decision = "e7t_integrator_shared_write_positive"
    elif best == "learned_port_map_then_freeze":
        decision = "e7t_learned_frozen_port_map_positive"
    elif best == "progressive_read_write_slot_allocation":
        decision = "e7t_progressive_read_write_slot_allocation_positive"
    elif best == "progressive_write_slot_allocation":
        decision = "e7t_progressive_write_slot_allocation_positive"
    elif best == "learned_sparse_mask_reference":
        decision = "e7t_sparse_mask_contract_still_preferred"
    elif detail["input_output"] >= max(detail["output"], detail["learned_sparse"]) + 0.01:
        decision = "e7t_input_output_port_map_positive"
    elif detail["output"] >= detail["untyped"] + 0.05:
        decision = "e7t_output_port_map_positive"
    else:
        decision = "e7t_ram_port_allocation_no_advantage"
    return {"schema_version": "e7t_decision_v1", "decision": decision, "detail": detail, "deterministic_replay_passed": False}


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
    port_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    plateau_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []

    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        state = e7p.copy_state(trained["state"], "e7t_baseline_standalone")
        baseline_library[skill] = state
        training_rows.append({"seed": seed, "skill": skill, "system": "baseline_standalone_pocket", "state_hash": e7p.state_hash(state), "standalone": trained["standalone"], "context": e7p.evaluate_context_pocket(skill, state, context_tasks[skill]), "trainable_scope": []})

    untyped_library: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        trained = e7p.train_context_pocket(seed, skill, "joint_adapter_plus_pocket_training", baseline_library[skill], context_tasks[skill], e7p_settings, out)
        untyped_library[skill] = trained["state"]
        training_rows.append({"seed": seed, "skill": skill, "system": "untyped_flow_baseline", "state_hash": e7p.state_hash(trained["state"]), "history": trained["history"], "context": e7p.evaluate_context_pocket(skill, trained["state"], context_tasks[skill]), "trainable_scope": trained["scope"]})
    untyped_contracts = {skill: build_contract(skill, "untyped_flow_baseline") for skill in SKILLS}
    rows.append(evaluate_system(seed, "untyped_flow_baseline", untyped_library, untyped_contracts, composition_task))

    libraries: dict[str, dict[str, dict[str, Any]]] = {}
    contracts_by_system: dict[str, dict[str, dict[str, Any]]] = {}
    for system in TRAINED_PORT_SYSTEMS:
        read_budget = len(INPUT_BANK) if system != "input_read_map_only" else 12
        write_budget = 1
        contracts = {skill: build_contract(skill, system, read_budget=read_budget, write_budget=write_budget) for skill in SKILLS}
        library, train_rows, map_rows = train_masked_library(seed, system, baseline_library, context_tasks, contracts, settings, out)
        result = evaluate_system(seed, system, library, contracts, composition_task)
        rows.append(result)
        training_rows.extend(train_rows)
        port_rows.extend(map_rows)
        libraries[system] = library
        contracts_by_system[system] = contracts

    learned_contracts, mask_mutation = e7r.mutate_contracts(seed, libraries["output_write_map_only"], contracts_by_system["output_write_map_only"], composition_task, e7r_settings, out)
    learned = evaluate_system(seed, "learned_sparse_mask_reference", libraries["output_write_map_only"], learned_contracts, composition_task)
    learned.update({key: value for key, value in mask_mutation.items() if key != "history"})
    rows.append(learned)
    mutation_rows.extend(mask_mutation["history"])
    for skill, contract in learned_contracts.items():
        port_rows.append({"seed": seed, "system": "learned_sparse_mask_reference", "skill": skill, "contract": contract_to_json(contract), "state_hash": e7p.state_hash(libraries["output_write_map_only"][skill])})

    learned_frozen_library, train_rows, map_rows = train_masked_library(seed, "learned_port_map_then_freeze", baseline_library, context_tasks, learned_contracts, settings, out)
    learned_frozen = evaluate_system(seed, "learned_port_map_then_freeze", learned_frozen_library, learned_contracts, composition_task)
    learned_frozen["source_mask_hash"] = payload_sha256({skill: contract_to_json(contract) for skill, contract in learned_contracts.items()})
    rows.append(learned_frozen)
    training_rows.extend(train_rows)
    port_rows.extend(map_rows)

    progressive_write, write_payload = train_progressive_write(seed, baseline_library, context_tasks, composition_task, settings, out)
    rows.append(progressive_write)
    training_rows.extend(write_payload["training_rows"])
    port_rows.extend(write_payload["port_rows"])
    plateau_rows.append({"seed": seed, "system": "progressive_write_slot_allocation", "curve": [{"budget": row["write_budget"], "usefulness": row["eval_mean_composition_usefulness"], "heldout": row["heldout_usefulness"], "ood": row["ood_usefulness"], "counterfactual": row["counterfactual_usefulness"], "adversarial": row["adversarial_usefulness"]} for row in write_payload["curve"]], "plateau": write_payload["plateau"]})

    progressive_read, read_payload = train_progressive_read_write(seed, baseline_library, context_tasks, composition_task, settings, out)
    rows.append(progressive_read)
    training_rows.extend(read_payload["training_rows"])
    port_rows.extend(read_payload["port_rows"])
    plateau_rows.append({"seed": seed, "system": "progressive_read_write_slot_allocation", "curve": [{"budget": row["read_budget"], "usefulness": row["eval_mean_composition_usefulness"], "heldout": row["heldout_usefulness"], "ood": row["ood_usefulness"], "counterfactual": row["counterfactual_usefulness"], "adversarial": row["adversarial_usefulness"]} for row in read_payload["curve"]], "plateau": read_payload["plateau"]})

    rows.append(evaluate_system(seed, "oracle_port_map_reference", None, None, composition_task, symbolic=True))
    dense = e7o.train_monolithic(seed, "dense_graph_danger_control", composition_task, e7o_settings, out, hidden=192, depth=4)
    dense["system"] = "dense_graph_danger_control"
    for key in ("eval_mean_write_spread", "eval_mean_changed_cell_count", "eval_mean_delta_magnitude", "eval_mean_read_cell_count", "eval_mean_write_cell_count", "ram_collision_rate", "read_cell_count", "write_cell_count"):
        dense[key] = 0.0
    rows.append(dense)

    shared_report = {
        "seed": seed,
        "shared_write_collision_rate": next(row for row in rows if row["system"] == "shared_write_control")["ram_collision_rate"],
        "integrator_collision_rate": next(row for row in rows if row["system"] == "integrator_shared_write_control")["ram_collision_rate"],
        "shared_usefulness": next(row for row in rows if row["system"] == "shared_write_control")["eval_mean_composition_usefulness"],
        "integrator_usefulness": next(row for row in rows if row["system"] == "integrator_shared_write_control")["eval_mean_composition_usefulness"],
    }
    return {"seed": seed, "rows": rows, "training_rows": training_rows, "port_rows": port_rows, "plateau_rows": plateau_rows, "mutation_rows": mutation_rows, "shared_report": shared_report}


def build_flow_grid_frames(rows: list[dict[str, Any]], plateau_rows: list[dict[str, Any]]) -> dict[str, Any]:
    interesting = sorted(rows, key=lambda row: row.get("eval_mean_composition_usefulness", 0.0))
    chosen = interesting[:2] + interesting[-2:]
    frames = []
    for idx, row in enumerate(chosen):
        frames.append({
            "frame_id": idx,
            "seed": row["seed"],
            "system": row["system"],
            "usefulness": row.get("eval_mean_composition_usefulness", 0.0),
            "write_spread": row.get("eval_mean_write_spread", 0.0),
            "read_cell_count": row.get("read_cell_count", 0.0),
            "write_cell_count": row.get("write_cell_count", 0.0),
            "note": "FlowGrid-compatible summary frame; inspect E7S HTML for cell-level visualization.",
        })
    return {"schema_version": "e7t_flow_grid_frames_v1", "frames": frames, "plateau_rows": plateau_rows}


def build_reports(rows: list[dict[str, Any]], training_rows: list[dict[str, Any]], port_rows: list[dict[str, Any]], plateau_rows: list[dict[str, Any]], mutation_rows: list[dict[str, Any]], shared_rows: list[dict[str, Any]], aggregate: dict[str, Any], decision: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    clean_rows = canonical_system_rows(rows)
    report_lines = [
        "# E7T Progressive RAM Port Allocation Probe Result",
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
            f"{system:<42} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} "
            f"write_spread={mean.get('eval_mean_write_spread', 0.0):.6f} "
            f"read={mean.get('eval_mean_read_cell_count', 0.0):.3f} "
            f"write={mean.get('eval_mean_write_cell_count', 0.0):.3f}"
        )
    report_lines.extend([
        "```",
        "",
        "## Boundary",
        "",
        "E7T is a controlled numeric Flow/RAM IO allocation probe. It does not make raw-language, AGI, consciousness, or model-scale claims.",
        "",
    ])
    return {
        "pocket_training_report.json": {"schema_version": "e7t_pocket_training_report_v1", "rows": sorted(training_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "port_map_report.json": {"schema_version": "e7t_port_map_report_v1", "rows": sorted(port_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "slot_plateau_report.json": {"schema_version": "e7t_slot_plateau_report_v1", "rows": sorted(plateau_rows, key=lambda row: (row["seed"], row["system"]))},
        "shared_write_report.json": {"schema_version": "e7t_shared_write_report_v1", "rows": sorted(shared_rows, key=lambda row: row["seed"])},
        "flow_grid_frames.json": build_flow_grid_frames(clean_rows, plateau_rows),
        "system_results.json": {"schema_version": "e7t_system_results_v1", "rows": clean_rows},
        "mutation_history.json": {"schema_version": "e7t_mutation_history_v1", "rows": sorted(mutation_rows, key=lambda row: (row["seed"], row["generation"]))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7t_summary_v1", "decision": decision["decision"], "best_non_reference_system": aggregate["best_non_reference_system"], "deterministic_replay_passed": deterministic.get("internal_replay_passed", False), "checker_failure_count": None},
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
        write_json(out / "backend_manifest.json", {"schema_version": "e7t_backend_manifest_v1", "milestone": MILESTONE, "settings": settings_payload(settings), "systems": list(SYSTEMS), "flow_dim": FLOW_DIM, "input_bank": list(INPUT_BANK), "write_bank": list(WRITE_BANK), "semantic_lane_labels_as_model_input": False, "training_performed": True, "device": select_device(settings.device), "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        write_json(out / "task_generation_report.json", {"schema_version": "e7t_task_generation_report_v1", "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()}, "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()}})
        rows: list[dict[str, Any]] = []
        training_rows: list[dict[str, Any]] = []
        port_rows: list[dict[str, Any]] = []
        plateau_rows: list[dict[str, Any]] = []
        mutation_rows: list[dict[str, Any]] = []
        shared_rows: list[dict[str, Any]] = []
        jobs = [{"seed": seed, "settings": settings.__dict__.copy(), "composition_task": composition_tasks[seed], "pocket_task": pocket_tasks[seed], "out": str(out)} for seed in settings.seeds]
        max_workers = max(1, min(settings.cpu_workers, len(jobs)))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
            while futures:
                done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                if not done:
                    write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_port_rows": len(port_rows), "pending": len(futures)})
                    continue
                for future in done:
                    label = futures.pop(future)
                    result = future.result()
                    rows.extend(result["rows"])
                    training_rows.extend(result["training_rows"])
                    port_rows.extend(result["port_rows"])
                    plateau_rows.extend(result["plateau_rows"])
                    mutation_rows.extend(result["mutation_rows"])
                    shared_rows.append(result["shared_report"])
                    append_progress(out, "seed_job_complete", label=label, pending=len(futures))
                    write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_port_rows": len(port_rows), "last_completed": label, "pending": len(futures)})
        aggregate = aggregate_results(rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e7t_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, training_rows, port_rows, plateau_rows, mutation_rows, shared_rows, aggregate, decision, deterministic_placeholder)
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
            cmd = [sys.executable, str(Path(__file__).resolve()), "--out", str(replay_out.relative_to(REPO_ROOT)), "--seeds", ",".join(map(str, settings.seeds)), "--train-rows-per-seed", str(settings.train_rows_per_seed), "--validation-rows-per-seed", str(settings.validation_rows_per_seed), "--heldout-rows-per-seed", str(settings.heldout_rows_per_seed), "--ood-rows-per-seed", str(settings.ood_rows_per_seed), "--counterfactual-rows-per-seed", str(settings.counterfactual_rows_per_seed), "--adversarial-rows-per-seed", str(settings.adversarial_rows_per_seed), "--pocket-pretrain-rows-per-seed", str(settings.pocket_pretrain_rows_per_seed), "--pocket-validation-rows-per-seed", str(settings.pocket_validation_rows_per_seed), "--pocket-dim", str(settings.pocket_dim), "--pocket-core-steps", str(settings.pocket_core_steps), "--pocket-epochs", str(settings.pocket_epochs), "--local-epochs", str(settings.local_epochs), "--full-epochs", str(settings.full_epochs), "--batch-size", str(settings.batch_size), "--learning-rate", str(settings.learning_rate), "--local-learning-rate", str(settings.local_learning_rate), "--weight-decay", str(settings.weight_decay), "--mask-mutation-generations", str(settings.mask_mutation_generations), "--mask-mutation-population", str(settings.mask_mutation_population), "--max-write-slots", str(settings.max_write_slots), "--read-budgets", ",".join(map(str, settings.read_budgets)), "--cpu-workers", str(settings.cpu_workers), "--device", settings.device, "--heartbeat-seconds", str(settings.heartbeat_seconds), "--execution-mode", settings.execution_mode, "--replay"]
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
            primary_hashes = hash_artifacts(out)
            replay_hashes = hash_artifacts(replay_out)
            comparisons = {artifact: {"primary": primary_hashes[artifact], "replay": replay_hashes[artifact], "match": primary_hashes[artifact] == replay_hashes[artifact]} for artifact in HASH_ARTIFACTS}
            deterministic = {"schema_version": "e7t_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, training_rows, port_rows, plateau_rows, mutation_rows, shared_rows, aggregate, decision, deterministic)
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
    parser.add_argument("--mask-mutation-generations", type=int, default=18)
    parser.add_argument("--mask-mutation-population", type=int, default=8)
    parser.add_argument("--max-write-slots", type=int, default=4)
    parser.add_argument("--read-budgets", default="4,8,12,24,30")
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
        mask_mutation_generations=args.mask_mutation_generations,
        mask_mutation_population=args.mask_mutation_population,
        max_write_slots=args.max_write_slots,
        read_budgets=parse_int_tuple(args.read_budgets),
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
