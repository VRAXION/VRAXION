#!/usr/bin/env python3
"""E7Y natural output bundle width audit.

E7X showed that calibrating one written RAM value is not enough. E7Y tests
whether a numeric pocket needs a small anonymous multi-cell output bundle
instead of a single output cell.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
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
E7W_PATH = Path(__file__).with_name("run_e7w_numeric_pocket_composition_failure_localization.py")
MILESTONE = "E7Y_NATURAL_OUTPUT_BUNDLE_WIDTH_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e7y_natural_output_bundle_width_audit")
DEFAULT_SEEDS = tuple(range(100301, 100309))
BUNDLE_WIDTHS = (1, 2, 3, 4, 5, 6, 8, 12)
ANONYMOUS_BUNDLE_BANK = (22, 23, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39)

WIDTH_SYSTEMS = {1: "single_value_write_baseline", **{width: f"output_bundle_N{width}" for width in BUNDLE_WIDTHS if width != 1}}
SYSTEMS = tuple(WIDTH_SYSTEMS[width] for width in BUNDLE_WIDTHS) + ("oracle_write_reference", "dense_graph_danger_control")
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "bundle_contract_report.json",
    "output_width_curve_report.json",
    "channel_morphology_report.json",
    "oracle_bundle_similarity_report.json",
    "ram_bundle_frame_report.json",
    "dense_graph_control_report.json",
    "system_results.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7y_natural_output_bundle_width_detected",
    "e7y_single_output_cell_sufficient",
    "e7y_large_output_bundle_required",
    "e7y_output_bundle_width_not_sufficient",
    "e7y_graph_soup_regression_detected",
)


def load_e7w_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7w_numeric_pocket_composition_failure_localization", E7W_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7W helpers from {E7W_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7w = load_e7w_module()
e7v = e7w.e7v
e7u = e7v.e7u
e7r = e7w.e7r
e7p = e7w.e7p
e7o = e7w.e7o

FLOW_DIM = int(e7w.FLOW_DIM)
SKILLS = tuple(e7w.SKILLS)
SPLITS = tuple(e7w.SPLITS)
EVAL_SPLITS = tuple(e7w.EVAL_SPLITS)
RESULT_POS = dict(e7w.RESULT_POS)
RESULT_INDICES = tuple(RESULT_POS[skill] for skill in SKILLS)


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
    pruned_read_count: int
    output_widths: tuple[int, ...]
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    execution_mode: str
    replay: bool


def round_float(value: float) -> float:
    return e7w.round_float(value)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7y::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7w.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7w.write_json(path, payload)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7w.append_progress(out, event, **details)


def resolve_out(path: str | Path) -> Path:
    return e7w.resolve_out(path)


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return e7w.parse_int_tuple(raw)


def select_device(requested: str) -> str:
    return e7w.select_device(requested)


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    payload["output_widths"] = list(settings.output_widths)
    payload["replay"] = False
    return payload


def to_e7w_settings(settings: Settings) -> Any:
    return e7w.Settings(
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
        pruned_read_count=settings.pruned_read_count,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        execution_mode=settings.execution_mode,
        replay=settings.replay,
    )


def to_e7r_settings(settings: Settings) -> Any:
    return e7v.to_e7r_settings(e7w.to_e7v_settings(to_e7w_settings(settings)))


def to_e7o_settings(settings: Settings) -> Any:
    return e7w.to_e7o_settings(to_e7w_settings(settings))


def safe_corr(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return 0.0
    return round_float(float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1]))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left:
        return 0.0
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return round_float(float(np.dot(a, b) / denom)) if denom > 1e-12 else 0.0


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


def rotated_bank(skill: str) -> list[int]:
    bank = list(ANONYMOUS_BUNDLE_BANK)
    offset = SKILLS.index(skill) % len(bank)
    return bank[offset:] + bank[:offset]


def bundle_cells(skill: str, width: int) -> list[int]:
    if width < 1:
        raise ValueError("output bundle width must be >= 1")
    cells = [int(RESULT_POS[skill])]
    for cell in rotated_bank(skill):
        if cell != RESULT_POS[skill] and cell not in RESULT_INDICES:
            cells.append(int(cell))
        if len(cells) >= width:
            break
    if len(cells) < width:
        for cell in range(FLOW_DIM):
            if cell != RESULT_POS[skill] and cell not in RESULT_INDICES and cell not in cells:
                cells.append(cell)
            if len(cells) >= width:
                break
    if len(cells) < width:
        raise ValueError(f"cannot allocate {width} anonymous bundle cells for {skill}")
    return cells[:width]


def anonymous_projection_weights(skill: str, channel: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(stable_seed(f"bundle-projection:{skill}:{channel}"))
    weights = rng.normal(0.0, 1.0, FLOW_DIM).astype(np.float32)
    weights[list(RESULT_INDICES)] *= 1.35
    weights[:24] *= 0.75
    bias = float(rng.normal(0.0, 0.20))
    return weights, bias


def canonical_step(row: dict[str, Any], flow: np.ndarray, skill: str) -> np.ndarray:
    out = flow.copy()
    out[RESULT_POS[skill]] = float(e7o.base_skill_value(skill, row["a"], row["b"], row["key"], row["threshold"], row["flip"], out))
    return out.astype(np.float32)


def bundle_values(row: dict[str, Any], canonical_after: np.ndarray, skill: str, width: int) -> np.ndarray:
    values = [float(canonical_after[RESULT_POS[skill]])]
    scale = math.sqrt(float(FLOW_DIM))
    for channel in range(1, width):
        weights, bias = anonymous_projection_weights(skill, channel)
        raw = float(np.dot(canonical_after.astype(np.float32), weights) / scale + bias)
        values.append(float(np.tanh(raw)))
    return np.asarray(values, dtype=np.float32)


def apply_oracle_bundle(row: dict[str, Any], flow: np.ndarray, skill: str, width: int) -> np.ndarray:
    out = canonical_step(row, flow, skill)
    cells = bundle_cells(skill, width)
    values = bundle_values(row, out, skill, width)
    for cell, value in zip(cells, values):
        out[cell] = value
    return out.astype(np.float32)


def generate_bundle_context_tasks(composition_task: dict[str, list[dict[str, Any]]], width: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
    tasks: dict[str, dict[str, list[dict[str, Any]]]] = {skill: {split: [] for split in SPLITS} for skill in SKILLS}
    for split in SPLITS:
        for row in composition_task[split]:
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            for skill in tuple(row["expected_route"]):
                target = apply_oracle_bundle(row, flow, skill, width)
                tasks[skill][split].append(
                    {
                        "row_id": f"{row['row_id']}:{skill}:w{width}",
                        "split": split,
                        "skill": skill,
                        "family": row["family"],
                        "flow": flow.tolist(),
                        "target_flow": target.tolist(),
                        "target_value": int(target[RESULT_POS[skill]] >= 0.5),
                    }
                )
                flow = target
    return tasks


def build_bundle_contract(skill: str, system: str, width: int, read_count: int, dense: bool = False) -> dict[str, Any]:
    cells = bundle_cells(skill, width)
    read_indices = ordered_unique(
        e7v.priority_read_map(skill, read_count)
        + list(RESULT_INDICES)
        + list(ANONYMOUS_BUNDLE_BANK)
        + cells
    )
    read = np.ones(FLOW_DIM, dtype=bool) if dense else bool_mask(read_indices[: max(1, min(FLOW_DIM, len(read_indices)))])
    write = bool_mask(cells)
    scratch = np.zeros(FLOW_DIM, dtype=bool)
    if dense:
        read = np.ones(FLOW_DIM, dtype=bool)
        write = np.ones(FLOW_DIM, dtype=bool)
    allowed = write | scratch
    preserve = ~allowed
    return {
        "skill": skill,
        "mode": system,
        "read": read,
        "write": write,
        "scratch": scratch,
        "return": bool_mask(cells),
        "preserve": preserve,
        "enforce": True,
        "residual": False,
        "semantic_label_control": False,
        "permuted": False,
        "assignment_cell": int(RESULT_POS[skill]),
        "output_bundle_width": int(width),
        "bundle_cells": cells,
        "dense_graph_control": bool(dense),
    }


def contract_to_json(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill": contract["skill"],
        "mode": contract["mode"],
        "read_indices": np.flatnonzero(contract["read"]).astype(int).tolist(),
        "write_indices": np.flatnonzero(contract["write"]).astype(int).tolist(),
        "scratch_indices": np.flatnonzero(contract["scratch"]).astype(int).tolist(),
        "return_indices": np.flatnonzero(contract["return"]).astype(int).tolist(),
        "assignment_cell": int(contract["assignment_cell"]),
        "output_bundle_width": int(contract["output_bundle_width"]),
        "bundle_cells": list(map(int, contract["bundle_cells"])),
        "read_count": int(np.sum(contract["read"])),
        "ram_cells_used": int(np.sum(contract["write"] | contract["scratch"])),
        "preserve_count": int(np.sum(contract["preserve"])),
        "semantic_label_control": False,
        "dense_graph_control": bool(contract.get("dense_graph_control", False)),
        "enforce": bool(contract["enforce"]),
    }


def train_bundle_library(
    seed: int,
    system: str,
    width: int,
    baseline_library: dict[str, dict[str, Any]],
    context_tasks: dict[str, dict[str, list[dict[str, Any]]]],
    settings: Settings,
    out: Path | None,
    dense: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    library: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    e7r_settings = to_e7r_settings(settings)
    for skill in SKILLS:
        contract = build_bundle_contract(skill, system, width, settings.pruned_read_count, dense=dense)
        trained = e7r.train_masked_context_pocket(seed, skill, system, baseline_library[skill], context_tasks[skill], e7r_settings, contract, out)
        library[skill] = trained["state"]
        contracts[skill] = contract
        contract_json = contract_to_json(contract)
        training_rows.append({"seed": seed, "system": system, "skill": skill, "state_hash": e7p.state_hash(trained["state"]), "history": trained["history"], "contract": contract_json})
        contract_rows.append({"seed": seed, "system": system, "skill": skill, "state_hash": e7p.state_hash(trained["state"]), "contract": contract_json})
    return library, contracts, training_rows, contract_rows


def threshold_primary_cell(flow: np.ndarray, skill: str) -> None:
    cell = int(RESULT_POS[skill])
    flow[cell] = np.float32(1.0 if float(flow[cell]) >= 0.0 else 0.0)


def channel_redundancy(channel_matrix: list[list[float]]) -> float:
    if not channel_matrix:
        return 0.0
    arr = np.asarray(channel_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
        return 0.0
    cors: list[float] = []
    for left in range(arr.shape[1]):
        for right in range(left + 1, arr.shape[1]):
            a = arr[:, left].tolist()
            b = arr[:, right].tolist()
            cors.append(abs(safe_corr(a, b)))
    return round_float(float(np.mean(cors)) if cors else 0.0)


def evaluate_system(
    seed: int,
    system: str,
    width: int,
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    task: dict[str, list[dict[str, Any]]],
    oracle: bool = False,
    dense: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evals: dict[str, Any] = {}
    similarity_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        correct: list[bool] = []
        samples: list[dict[str, Any]] = []
        bundle_written: list[float] = []
        bundle_oracle: list[float] = []
        bundle_mae: list[float] = []
        delta_values: list[float] = []
        next_errors: list[float] = []
        changed_counts: list[float] = []
        channel_matrix: list[list[float]] = []
        write_spread: list[float] = []
        for row in task[split]:
            route = tuple(row["expected_route"])
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            canonical = flow.copy()
            step_samples: list[dict[str, Any]] = []
            for step_idx, skill in enumerate(route):
                before = flow.copy()
                canonical_after = apply_oracle_bundle(row, canonical, skill, width)
                cells = bundle_cells(skill, width)
                oracle_vals = canonical_after[cells].astype(np.float32)
                if oracle:
                    pred = canonical_after.copy()
                else:
                    assert library is not None
                    if dense:
                        pred = e7p.np_forward(library[skill], flow.reshape(1, -1)).reshape(-1).astype(np.float32)
                    else:
                        assert contracts is not None
                        pred = e7r.masked_forward_np(library[skill], flow.reshape(1, -1), contracts[skill]).reshape(-1).astype(np.float32)
                    threshold_primary_cell(pred, skill)
                written_vals = pred[cells].astype(np.float32)
                mae = np.abs(written_vals - oracle_vals)
                bundle_written.extend([float(value) for value in written_vals.tolist()])
                bundle_oracle.extend([float(value) for value in oracle_vals.tolist()])
                bundle_mae.extend([float(value) for value in mae.tolist()])
                channel_matrix.append([float(value) for value in written_vals.tolist()])
                delta = pred - before
                changed = np.abs(delta) > 1e-6
                changed_counts.append(float(np.sum(changed)))
                write_spread.append(float(np.mean(changed)))
                delta_values.append(float(np.mean(np.abs(delta))))
                next_error = 0.0
                if step_idx + 1 < len(route):
                    if contracts and route[step_idx + 1] in contracts:
                        read = contracts[route[step_idx + 1]]["read"]
                    else:
                        read = np.ones(FLOW_DIM, dtype=bool)
                    next_error = float(np.mean(np.abs(pred[read] - canonical_after[read])))
                    next_errors.append(next_error)
                if len(similarity_rows) < 300:
                    similarity_rows.append(
                        {
                            "seed": seed,
                            "system": system,
                            "split": split,
                            "row_id": row["row_id"],
                            "skill": skill,
                            "step": step_idx + 1,
                            "output_bundle_width": int(width),
                            "bundle_cells": cells,
                            "written_values": [round_float(float(value)) for value in written_vals.tolist()],
                            "oracle_values": [round_float(float(value)) for value in oracle_vals.tolist()],
                            "bundle_mae": round_float(float(np.mean(mae))),
                        }
                    )
                if len(frame_rows) < 120:
                    frame_rows.append(
                        {
                            "seed": seed,
                            "system": system,
                            "split": split,
                            "row_id": row["row_id"],
                            "skill": skill,
                            "step": step_idx + 1,
                            "output_bundle_width": int(width),
                            "flow_before": [round_float(float(v)) for v in before.tolist()],
                            "flow_after": [round_float(float(v)) for v in pred.tolist()],
                            "oracle_after": [round_float(float(v)) for v in canonical_after.tolist()],
                            "bundle_cells": cells,
                            "written_values": [round_float(float(value)) for value in written_vals.tolist()],
                            "oracle_values": [round_float(float(value)) for value in oracle_vals.tolist()],
                        }
                    )
                if len(step_samples) < 6:
                    step_samples.append(
                        {
                            "skill": skill,
                            "bundle_cells": cells,
                            "bundle_mae": round_float(float(np.mean(mae))),
                            "next_error": round_float(next_error),
                        }
                    )
                flow = pred.astype(np.float32)
                canonical = canonical_after.astype(np.float32)
            pred_answer = int(e7o.predict_answer_from_flow(row, flow))
            ok = pred_answer == int(row["target_answer"])
            correct.append(ok)
            if len(samples) < 8:
                samples.append(
                    {
                        "row_id": row["row_id"],
                        "family": row["family"],
                        "route": list(route),
                        "target": int(row["target_answer"]),
                        "predicted": pred_answer,
                        "correct": bool(ok),
                        "steps": step_samples,
                    }
                )
        acc = round_float(float(np.mean(correct)))
        mean_steps = round_float(float(np.mean([len(row["expected_route"]) for row in task[split]])))
        bit_cost = sum(e7p.bit_budget(state) for state in library.values()) if library else 0
        cost_penalty = min(0.10, 0.00000016 * bit_cost + 0.0025 * mean_steps + 0.00055 * width)
        evals[split] = {
            "answer_accuracy": acc,
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(max(0.0, acc - cost_penalty)),
            "mean_route_steps": mean_steps,
            "output_bundle_width": int(width),
            "ram_cells_used": int(width),
            "write_spread": round_float(float(np.mean(write_spread)) if write_spread else 0.0),
            "changed_cell_count": round_float(float(np.mean(changed_counts)) if changed_counts else 0.0),
            "delta_magnitude": round_float(float(np.mean(delta_values)) if delta_values else 0.0),
            "oracle_bundle_similarity": round_float(1.0 - float(np.mean(bundle_mae)) if bundle_mae else 0.0),
            "bundle_mean_absolute_error_to_oracle": round_float(float(np.mean(bundle_mae)) if bundle_mae else 0.0),
            "bundle_cellwise_correlation_with_oracle": safe_corr(bundle_written, bundle_oracle),
            "bundle_cosine_similarity_with_oracle": cosine_similarity(bundle_written, bundle_oracle),
            "output_channel_redundancy": channel_redundancy(channel_matrix),
            "output_value_min": round_float(float(np.min(bundle_written)) if bundle_written else 0.0),
            "output_value_max": round_float(float(np.max(bundle_written)) if bundle_written else 0.0),
            "next_pocket_input_compatibility": round_float(float(np.mean(next_errors)) if next_errors else 0.0),
            "bit_budget": bit_cost,
            "row_level_samples": samples,
        }
        morphology_rows.append(
            {
                "seed": seed,
                "system": system,
                "split": split,
                "output_bundle_width": int(width),
                "output_value_min": evals[split]["output_value_min"],
                "output_value_max": evals[split]["output_value_max"],
                "output_channel_redundancy": evals[split]["output_channel_redundancy"],
                "oracle_bundle_similarity": evals[split]["oracle_bundle_similarity"],
                "bundle_mean_absolute_error_to_oracle": evals[split]["bundle_mean_absolute_error_to_oracle"],
            }
        )
    row = {
        "seed": seed,
        "system": system,
        "output_bundle_width": int(width),
        "evals": evals,
        "heldout_usefulness": evals["heldout"]["composition_usefulness"],
        "ood_usefulness": evals["ood"]["composition_usefulness"],
        "counterfactual_usefulness": evals["counterfactual"]["composition_usefulness"],
        "adversarial_usefulness": evals["adversarial"]["composition_usefulness"],
        "eval_mean_answer_accuracy": round_float(float(np.mean([evals[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([evals[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_route_accuracy": 1.0,
        "eval_mean_output_bundle_width": int(width),
        "eval_mean_ram_cells_used": int(width),
        "eval_mean_write_spread": round_float(float(np.mean([evals[split]["write_spread"] for split in EVAL_SPLITS]))),
        "eval_mean_changed_cell_count": round_float(float(np.mean([evals[split]["changed_cell_count"] for split in EVAL_SPLITS]))),
        "eval_mean_delta_magnitude": round_float(float(np.mean([evals[split]["delta_magnitude"] for split in EVAL_SPLITS]))),
        "eval_mean_oracle_bundle_similarity": round_float(float(np.mean([evals[split]["oracle_bundle_similarity"] for split in EVAL_SPLITS]))),
        "eval_mean_bundle_mean_absolute_error_to_oracle": round_float(float(np.mean([evals[split]["bundle_mean_absolute_error_to_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_bundle_cellwise_correlation_with_oracle": round_float(float(np.mean([evals[split]["bundle_cellwise_correlation_with_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_bundle_cosine_similarity_with_oracle": round_float(float(np.mean([evals[split]["bundle_cosine_similarity_with_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_output_channel_redundancy": round_float(float(np.mean([evals[split]["output_channel_redundancy"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_input_compatibility": round_float(float(np.mean([evals[split]["next_pocket_input_compatibility"] for split in EVAL_SPLITS]))),
        "parameter_count": sum(e7p.parameter_count(state) for state in library.values()) if library else 0,
        "bit_budget": sum(e7p.bit_budget(state) for state in library.values()) if library else 0,
    }
    return row, similarity_rows, frame_rows, morphology_rows


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (int(row.get("seed", 0)), SYSTEMS.index(str(row.get("system")))))
    systems: dict[str, Any] = {}
    metric_names = (
        "eval_mean_answer_accuracy",
        "eval_mean_composition_usefulness",
        "heldout_usefulness",
        "ood_usefulness",
        "counterfactual_usefulness",
        "adversarial_usefulness",
        "eval_mean_output_bundle_width",
        "eval_mean_ram_cells_used",
        "eval_mean_write_spread",
        "eval_mean_changed_cell_count",
        "eval_mean_delta_magnitude",
        "eval_mean_oracle_bundle_similarity",
        "eval_mean_bundle_mean_absolute_error_to_oracle",
        "eval_mean_bundle_cellwise_correlation_with_oracle",
        "eval_mean_bundle_cosine_similarity_with_oracle",
        "eval_mean_output_channel_redundancy",
        "eval_mean_next_pocket_input_compatibility",
        "parameter_count",
        "bit_budget",
    )
    for system in SYSTEMS:
        system_rows = [row for row in rows if row["system"] == system]
        metrics: dict[str, list[float]] = {}
        for row in system_rows:
            for metric in metric_names:
                if metric in row and row[metric] is not None:
                    metrics.setdefault(metric, []).append(float(row[metric]))
            for split in EVAL_SPLITS:
                for key, value in row.get("evals", {}).get(split, {}).items():
                    if isinstance(value, (int, float)):
                        metrics.setdefault(f"{split}_{key}", []).append(float(value))
        systems[system] = {
            "seed_count": len(system_rows),
            "mean": {key: round_float(float(np.mean(values))) for key, values in metrics.items()},
            "min": {key: round_float(float(np.min(values))) for key, values in metrics.items()},
            "max": {key: round_float(float(np.max(values))) for key, values in metrics.items()},
        }
    candidates = [system for system in SYSTEMS if system not in {"oracle_write_reference", "dense_graph_danger_control"}]
    best = max(candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    baseline = systems["single_value_write_baseline"]["mean"].get("eval_mean_composition_usefulness", 0.0)
    best_score = systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)
    plateau_candidates = [
        system for system in candidates if systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0) >= best_score - 0.002
    ]
    plateau = min(plateau_candidates, key=lambda system: systems[system]["mean"].get("eval_mean_output_bundle_width", 9999.0)) if plateau_candidates else best
    curve = [
        {
            "system": WIDTH_SYSTEMS[width],
            "output_bundle_width": width,
            "mean_composition_usefulness": systems[WIDTH_SYSTEMS[width]]["mean"].get("eval_mean_composition_usefulness", 0.0),
            "mean_answer_accuracy": systems[WIDTH_SYSTEMS[width]]["mean"].get("eval_mean_answer_accuracy", 0.0),
            "mean_oracle_bundle_similarity": systems[WIDTH_SYSTEMS[width]]["mean"].get("eval_mean_oracle_bundle_similarity", 0.0),
        }
        for width in BUNDLE_WIDTHS
    ]
    return {
        "schema_version": "e7y_aggregate_metrics_v1",
        "systems": systems,
        "output_width_curve": curve,
        "best_non_reference_system": best,
        "plateau_system": plateau,
        "plateau_width": int(systems[plateau]["mean"].get("eval_mean_output_bundle_width", 0)),
        "best_eval_mean_composition_usefulness": best_score,
        "baseline_eval_mean_composition_usefulness": baseline,
    }


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    baseline = systems["single_value_write_baseline"]["mean"].get("eval_mean_composition_usefulness", 0.0)
    oracle = systems["oracle_write_reference"]["mean"].get("eval_mean_composition_usefulness", 0.0)
    dense = systems["dense_graph_danger_control"]["mean"].get("eval_mean_composition_usefulness", 0.0)
    best = aggregate["best_non_reference_system"]
    best_score = systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)
    best_width = int(systems[best]["mean"].get("eval_mean_output_bundle_width", 0))
    plateau_width = int(aggregate["plateau_width"])
    gap = max(1e-9, oracle - baseline)
    gap_closed = (best_score - baseline) / gap
    detail = {
        "baseline": round_float(baseline),
        "oracle": round_float(oracle),
        "dense": round_float(dense),
        "best_non_reference_system": best,
        "best_width": best_width,
        "best_score": round_float(best_score),
        "plateau_width": plateau_width,
        "gap_fraction_closed": round_float(gap_closed),
    }
    if dense >= best_score + 0.02:
        decision = "e7y_graph_soup_regression_detected"
    elif best == "single_value_write_baseline" and baseline >= oracle - 0.02:
        decision = "e7y_single_output_cell_sufficient"
    elif best_score <= baseline + 0.005 and gap_closed < 0.10:
        decision = "e7y_output_bundle_width_not_sufficient"
    elif gap_closed >= 0.60 and best_width >= 8:
        decision = "e7y_large_output_bundle_required"
    elif gap_closed >= 0.60 and plateau_width <= 6:
        decision = "e7y_natural_output_bundle_width_detected"
    elif gap_closed >= 0.25 and best_width > 1:
        decision = "e7y_natural_output_bundle_width_detected"
    else:
        decision = "e7y_output_bundle_width_not_sufficient"
    return {"schema_version": "e7y_decision_v1", "decision": decision, "detail": detail, "deterministic_replay_passed": False}


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    e7o_settings = to_e7o_settings(settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    baseline_library: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    similarity_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []

    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        state = e7p.copy_state(trained["state"], "e7y_baseline_standalone")
        baseline_library[skill] = state
        training_rows.append({"seed": seed, "system": "baseline_standalone_pocket", "skill": skill, "state_hash": e7p.state_hash(state), "standalone": trained["standalone"]})

    trained_by_width: dict[int, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    for width in settings.output_widths:
        system = WIDTH_SYSTEMS[int(width)]
        context_tasks = generate_bundle_context_tasks(composition_task, int(width))
        library, contracts, train_rows, contract_report_rows = train_bundle_library(seed, system, int(width), baseline_library, context_tasks, settings, out)
        trained_by_width[int(width)] = (library, contracts)
        training_rows.extend(train_rows)
        contract_rows.extend(contract_report_rows)
        result, sim, frames, morph = evaluate_system(seed, system, int(width), library, contracts, composition_task)
        rows.append(result)
        similarity_rows.extend(sim)
        frame_rows.extend(frames)
        morphology_rows.extend(morph)
        if out:
            append_progress(out, "output_bundle_width_evaluated", seed=seed, system=system, output_bundle_width=int(width), usefulness=result["eval_mean_composition_usefulness"])

    max_width = max(settings.output_widths)
    oracle_result, oracle_sim, oracle_frames, oracle_morph = evaluate_system(seed, "oracle_write_reference", int(max_width), None, None, composition_task, oracle=True)
    rows.append(oracle_result)
    similarity_rows.extend(oracle_sim)
    frame_rows.extend(oracle_frames)
    morphology_rows.extend(oracle_morph)

    # Dense danger control: use the unmasked standalone pockets, not a primary
    # learned system. It is present only to catch graph-soup regression.
    dense_result, dense_sim, dense_frames, dense_morph = evaluate_system(seed, "dense_graph_danger_control", int(max_width), baseline_library, None, composition_task, dense=True)
    rows.append(dense_result)
    similarity_rows.extend(dense_sim)
    frame_rows.extend(dense_frames)
    morphology_rows.extend(dense_morph)
    dense_rows.append({"seed": seed, "system": "dense_graph_danger_control", "state_hashes": {skill: e7p.state_hash(state) for skill, state in baseline_library.items()}, "diagnostic_only": True})

    return {
        "seed": seed,
        "rows": rows,
        "training_rows": training_rows,
        "contract_rows": contract_rows,
        "similarity_rows": similarity_rows,
        "frame_rows": frame_rows,
        "morphology_rows": morphology_rows,
        "dense_rows": dense_rows,
    }


def build_reports(
    rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    contract_rows: list[dict[str, Any]],
    similarity_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    morphology_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    lines = [
        "# E7Y Natural Output Bundle Width Audit Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_non_reference_system = {aggregate['best_non_reference_system']}",
        f"plateau_width = {aggregate['plateau_width']}",
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
            f"{system:<32} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} "
            f"width={mean.get('eval_mean_output_bundle_width', 0.0):.1f} "
            f"oracle_sim={mean.get('eval_mean_oracle_bundle_similarity', 0.0):.6f} "
            f"redund={mean.get('eval_mean_output_channel_redundancy', 0.0):.6f} "
            f"next={mean.get('eval_mean_next_pocket_input_compatibility', 0.0):.6f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Interpretation Boundary",
            "",
            "E7Y is a controlled numeric Flow/RAM output-width diagnostic. It does not make raw-language, AGI, consciousness, or model-scale claims.",
            "",
        ]
    )
    return {
        "pocket_training_report.json": {"schema_version": "e7y_pocket_training_report_v1", "rows": sorted(training_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "bundle_contract_report.json": {"schema_version": "e7y_bundle_contract_report_v1", "rows": sorted(contract_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "output_width_curve_report.json": {"schema_version": "e7y_output_width_curve_report_v1", "rows": aggregate["output_width_curve"]},
        "channel_morphology_report.json": {"schema_version": "e7y_channel_morphology_report_v1", "rows": sorted(morphology_rows, key=lambda row: (row["seed"], row["system"], row["split"]))},
        "oracle_bundle_similarity_report.json": {"schema_version": "e7y_oracle_bundle_similarity_report_v1", "rows": sorted(similarity_rows, key=lambda row: (row["seed"], row["system"], row["split"], row["row_id"], row["step"], row["skill"]))},
        "ram_bundle_frame_report.json": {"schema_version": "e7y_ram_bundle_frame_report_v1", "rows": sorted(frame_rows, key=lambda row: (row["seed"], row["system"], row["split"], row["row_id"], row["step"], row["skill"]))},
        "dense_graph_control_report.json": {"schema_version": "e7y_dense_graph_control_report_v1", "rows": sorted(dense_rows, key=lambda row: row["seed"])},
        "system_results.json": {"schema_version": "e7y_system_results_v1", "rows": sorted(rows, key=lambda row: (row["seed"], SYSTEMS.index(row["system"])))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7y_summary_v1", "decision": decision["decision"], "best_non_reference_system": aggregate["best_non_reference_system"], "plateau_width": aggregate["plateau_width"], "deterministic_replay_passed": deterministic.get("internal_replay_passed", False), "checker_failure_count": None},
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
        "--pruned-read-count",
        str(settings.pruned_read_count),
        "--output-widths",
        ",".join(map(str, settings.output_widths)),
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
        write_json(
            out / "backend_manifest.json",
            {
                "schema_version": "e7y_backend_manifest_v1",
                "milestone": MILESTONE,
                "settings": settings_payload(settings),
                "systems": list(SYSTEMS),
                "output_widths": list(settings.output_widths),
                "flow_dim": FLOW_DIM,
                "anonymous_bundle_bank": list(ANONYMOUS_BUNDLE_BANK),
                "semantic_lane_labels_as_model_input": False,
                "runtime_random_output_placement": False,
                "new_router": False,
                "oracle_used_as_reference_only": True,
                "training_performed": True,
                "device": select_device(settings.device),
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        )
        write_json(
            out / "task_generation_report.json",
            {
                "schema_version": "e7y_task_generation_report_v1",
                "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()},
                "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()},
            },
        )
        rows: list[dict[str, Any]] = []
        training_rows: list[dict[str, Any]] = []
        contract_rows: list[dict[str, Any]] = []
        similarity_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        morphology_rows: list[dict[str, Any]] = []
        dense_rows: list[dict[str, Any]] = []
        jobs = [{"seed": seed, "settings": settings.__dict__.copy(), "composition_task": composition_tasks[seed], "pocket_task": pocket_tasks[seed], "out": str(out)} for seed in settings.seeds]
        max_workers = max(1, min(settings.cpu_workers, len(jobs)))
        if settings.device == "cuda" and max_workers > 1:
            append_progress(out, "cuda_process_pool_disabled", requested_workers=max_workers, active_workers=1)
            max_workers = 1
        if max_workers == 1:
            for job in jobs:
                result = seed_worker(job)
                rows.extend(result["rows"])
                training_rows.extend(result["training_rows"])
                contract_rows.extend(result["contract_rows"])
                similarity_rows.extend(result["similarity_rows"])
                frame_rows.extend(result["frame_rows"])
                morphology_rows.extend(result["morphology_rows"])
                dense_rows.extend(result["dense_rows"])
                append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(jobs) - len({row["seed"] for row in rows}))
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_similarity_rows": len(similarity_rows), "pending": len(jobs) - len({row["seed"] for row in rows})})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
                while futures:
                    done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_similarity_rows": len(similarity_rows), "pending": len(futures)})
                        continue
                    for future in done:
                        label = futures.pop(future)
                        result = future.result()
                        rows.extend(result["rows"])
                        training_rows.extend(result["training_rows"])
                        contract_rows.extend(result["contract_rows"])
                        similarity_rows.extend(result["similarity_rows"])
                        frame_rows.extend(result["frame_rows"])
                        morphology_rows.extend(result["morphology_rows"])
                        dense_rows.extend(result["dense_rows"])
                        append_progress(out, "seed_job_complete", label=label, pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_similarity_rows": len(similarity_rows), "last_completed": label, "pending": len(futures)})

        aggregate = aggregate_results(rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e7y_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, training_rows, contract_rows, similarity_rows, frame_rows, morphology_rows, dense_rows, aggregate, decision, deterministic_placeholder)
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
            deterministic = {"schema_version": "e7y_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, training_rows, contract_rows, similarity_rows, frame_rows, morphology_rows, dense_rows, aggregate, decision, deterministic)
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
    parser.add_argument("--train-rows-per-seed", type=int, default=320)
    parser.add_argument("--validation-rows-per-seed", type=int, default=144)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=144)
    parser.add_argument("--ood-rows-per-seed", type=int, default=144)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=144)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=144)
    parser.add_argument("--pocket-pretrain-rows-per-seed", type=int, default=360)
    parser.add_argument("--pocket-validation-rows-per-seed", type=int, default=144)
    parser.add_argument("--pocket-dim", type=int, default=56)
    parser.add_argument("--pocket-core-steps", type=int, default=2)
    parser.add_argument("--pocket-epochs", type=int, default=36)
    parser.add_argument("--local-epochs", type=int, default=24)
    parser.add_argument("--full-epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--local-learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pruned-read-count", type=int, default=30)
    parser.add_argument("--output-widths", default=",".join(map(str, BUNDLE_WIDTHS)))
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    parser.add_argument("--execution-mode", default="evidence")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    widths = parse_int_tuple(args.output_widths)
    unknown = sorted(set(widths) - set(BUNDLE_WIDTHS))
    if unknown:
        raise ValueError(f"unsupported output widths: {unknown}; supported={BUNDLE_WIDTHS}")
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
        pruned_read_count=args.pruned_read_count,
        output_widths=widths,
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
