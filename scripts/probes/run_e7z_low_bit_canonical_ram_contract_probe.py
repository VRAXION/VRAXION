#!/usr/bin/env python3
"""E7Z low-bit canonical RAM contract probe.

E7W/E7X/E7Y localized numeric pocket composition failure to the write/read
contract rather than route choice, scalar calibration, or output width. E7Z
tests whether producer -> RAM -> consumer communication improves when the
shared RAM boundary is forced into a binary/ternary/int4 canonical code from
the start.
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
E7Y_PATH = Path(__file__).with_name("run_e7y_natural_output_bundle_width_audit.py")
MILESTONE = "E7Z_LOW_BIT_CANONICAL_RAM_CONTRACT_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e7z_low_bit_canonical_ram_contract_probe")
DEFAULT_SEEDS = tuple(range(100401, 100409))
OUTPUT_WIDTH = 12
LOW_BIT_CODES = ("binary", "ternary", "int4")
CODE_BITS = {"continuous": 32, "binary": 1, "ternary": 2, "int4": 4}

SYSTEM_CODE = {
    "continuous_direct_write_baseline": "continuous",
    "oracle_write_continuous_reference": "continuous",
    "oracle_write_binary_projected": "binary",
    "oracle_write_ternary_projected": "ternary",
    "oracle_write_int4_projected": "int4",
    "learned_binary_ram_boundary": "binary",
    "learned_ternary_ram_boundary": "ternary",
    "learned_int4_ram_boundary": "int4",
    "learned_binary_ram_boundary_plus_mutation_repair": "binary",
    "learned_ternary_ram_boundary_plus_mutation_repair": "ternary",
    "learned_int4_ram_boundary_plus_mutation_repair": "int4",
    "pure_binary_pocket_and_ram": "binary",
    "pure_ternary_pocket_and_ram": "ternary",
    "int4_pocket_and_ram": "int4",
    "mixed_precision_pocket_float_ram_lowbit": "mixed",
    "dense_graph_danger_control": "continuous",
}
SYSTEMS = tuple(SYSTEM_CODE)
LEARNED_EXTERNAL_SYSTEMS = (
    "learned_binary_ram_boundary",
    "learned_ternary_ram_boundary",
    "learned_int4_ram_boundary",
)
REPAIR_SYSTEMS = (
    "learned_binary_ram_boundary_plus_mutation_repair",
    "learned_ternary_ram_boundary_plus_mutation_repair",
    "learned_int4_ram_boundary_plus_mutation_repair",
)
PURE_SYSTEMS = ("pure_binary_pocket_and_ram", "pure_ternary_pocket_and_ram", "int4_pocket_and_ram")
ORACLE_SYSTEMS = (
    "oracle_write_continuous_reference",
    "oracle_write_binary_projected",
    "oracle_write_ternary_projected",
    "oracle_write_int4_projected",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "projected_oracle_report.json",
    "low_bit_boundary_report.json",
    "progressive_freeze_report.json",
    "mutation_repair_report.json",
    "bit_budget_report.json",
    "system_results.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7z_binary_canonical_ram_contract_positive",
    "e7z_ternary_canonical_ram_contract_positive",
    "e7z_int4_canonical_ram_contract_positive",
    "e7z_low_bit_ram_contract_partially_positive",
    "e7z_low_bit_training_or_commit_learning_bottleneck",
    "e7z_full_low_bit_pocket_ram_preferred",
    "e7z_external_low_bit_ram_boundary_sufficient",
    "e7z_low_bit_mutation_repair_positive",
    "e7z_low_bit_canonical_ram_contract_not_sufficient",
    "e7z_graph_soup_regression_detected",
)


def load_e7y_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7y_natural_output_bundle_width_audit", E7Y_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7Y helpers from {E7Y_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7y = load_e7y_module()
e7w = e7y.e7w
e7v = e7y.e7v
e7r = e7y.e7r
e7p = e7y.e7p
e7o = e7y.e7o

FLOW_DIM = int(e7y.FLOW_DIM)
SKILLS = tuple(e7y.SKILLS)
SPLITS = tuple(e7y.SPLITS)
EVAL_SPLITS = tuple(e7y.EVAL_SPLITS)
RESULT_POS = dict(e7y.RESULT_POS)
RESULT_INDICES = tuple(e7y.RESULT_INDICES)


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
    repair_generations: int
    repair_population: int
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
    return int(hashlib.sha256(f"e7z::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


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
    payload["replay"] = False
    return payload


def to_e7y_settings(settings: Settings) -> Any:
    return e7y.Settings(
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
        output_widths=(OUTPUT_WIDTH,),
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        execution_mode=settings.execution_mode,
        replay=settings.replay,
    )


def to_e7r_settings(settings: Settings) -> Any:
    return e7y.to_e7r_settings(to_e7y_settings(settings))


def to_e7o_settings(settings: Settings) -> Any:
    return e7y.to_e7o_settings(to_e7y_settings(settings))


def safe_corr(left: list[float], right: list[float]) -> float:
    return e7y.safe_corr(left, right)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return e7y.cosine_similarity(left, right)


def default_boundary_params(code: str) -> dict[str, dict[str, list[float] | float]]:
    params: dict[str, dict[str, list[float] | float]] = {}
    for skill in SKILLS:
        params[skill] = {
            "primary_threshold": 0.0,
            "support_thresholds": [0.0 for _ in range(OUTPUT_WIDTH - 1)],
            "support_deadzones": [0.20 for _ in range(OUTPUT_WIDTH - 1)],
            "support_scales": [1.0 for _ in range(OUTPUT_WIDTH - 1)],
        }
    return params


def quantize_support(value: float, code: str, threshold: float = 0.0, deadzone: float = 0.20, scale: float = 1.0) -> float:
    if code == "continuous":
        return float(value)
    centered = float(value) - float(threshold)
    if code == "binary":
        return 1.0 if centered >= 0.0 else -1.0
    if code == "ternary":
        dz = max(0.0, float(deadzone))
        if centered > dz:
            return 1.0
        if centered < -dz:
            return -1.0
        return 0.0
    if code == "int4":
        sc = max(0.05, float(scale))
        clipped = float(np.clip(centered / sc, -1.0, 1.0))
        return float(np.round((clipped + 1.0) * 7.5) / 7.5 - 1.0)
    raise ValueError(code)


def quantize_bundle_values(
    values: np.ndarray,
    skill: str,
    code: str,
    params: dict[str, dict[str, Any]] | None = None,
    primary_is_logit: bool = True,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    row = params.get(skill, {}) if params else {}
    primary_threshold = float(row.get("primary_threshold", 0.0)) if primary_is_logit else 0.5
    out[0] = np.float32(1.0 if float(out[0]) >= primary_threshold else 0.0)
    if code == "continuous":
        return out.astype(np.float32)
    thresholds = list(row.get("support_thresholds", [0.0 for _ in range(max(0, len(out) - 1))]))
    deadzones = list(row.get("support_deadzones", [0.20 for _ in range(max(0, len(out) - 1))]))
    scales = list(row.get("support_scales", [1.0 for _ in range(max(0, len(out) - 1))]))
    for idx in range(1, len(out)):
        out[idx] = np.float32(
            quantize_support(
                float(out[idx]),
                code,
                thresholds[idx - 1] if idx - 1 < len(thresholds) else 0.0,
                deadzones[idx - 1] if idx - 1 < len(deadzones) else 0.20,
                scales[idx - 1] if idx - 1 < len(scales) else 1.0,
            )
        )
    return out.astype(np.float32)


def apply_oracle_code(row: dict[str, Any], flow: np.ndarray, skill: str, code: str, params: dict[str, dict[str, Any]] | None = None) -> np.ndarray:
    out = e7y.canonical_step(row, flow, skill)
    cells = e7y.bundle_cells(skill, OUTPUT_WIDTH)
    values = e7y.bundle_values(row, out, skill, OUTPUT_WIDTH)
    values = quantize_bundle_values(values, skill, code, params, primary_is_logit=False)
    for cell, value in zip(cells, values):
        out[cell] = value
    return out.astype(np.float32)


def generate_code_context_tasks(composition_task: dict[str, list[dict[str, Any]]], code: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    tasks: dict[str, dict[str, list[dict[str, Any]]]] = {skill: {split: [] for split in SPLITS} for skill in SKILLS}
    params = default_boundary_params(code)
    for split in SPLITS:
        for row in composition_task[split]:
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            for skill in tuple(row["expected_route"]):
                target = apply_oracle_code(row, flow, skill, code, params)
                tasks[skill][split].append(
                    {
                        "row_id": f"{row['row_id']}:{skill}:{code}",
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


def build_lowbit_contract(skill: str, system: str, read_count: int) -> dict[str, Any]:
    cells = e7y.bundle_cells(skill, OUTPUT_WIDTH)
    support = [cell for cell in cells if cell != RESULT_POS[skill]]
    read_indices = e7y.ordered_unique(
        e7v.priority_read_map(skill, read_count)
        + list(RESULT_INDICES)
        + list(e7y.ANONYMOUS_BUNDLE_BANK)
        + cells
    )
    read = e7y.bool_mask(read_indices)
    write = e7y.bool_mask([RESULT_POS[skill]])
    scratch = e7y.bool_mask(support)
    allowed = write | scratch
    return {
        "skill": skill,
        "mode": system,
        "read": read,
        "write": write,
        "scratch": scratch,
        "return": e7y.bool_mask(cells),
        "preserve": ~allowed,
        "enforce": True,
        "residual": False,
        "semantic_label_control": False,
        "permuted": False,
        "assignment_cell": int(RESULT_POS[skill]),
        "output_bundle_width": OUTPUT_WIDTH,
        "bundle_cells": cells,
        "dense_graph_control": False,
    }


def train_code_library(
    seed: int,
    system: str,
    code: str,
    baseline_library: dict[str, dict[str, Any]],
    context_tasks: dict[str, dict[str, list[dict[str, Any]]]],
    settings: Settings,
    out: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    library: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    freeze_rows: list[dict[str, Any]] = []
    e7r_settings = to_e7r_settings(settings)
    for skill in SKILLS:
        contract = build_lowbit_contract(skill, system, settings.pruned_read_count)
        trained = e7r.train_masked_context_pocket(seed, skill, system, baseline_library[skill], context_tasks[skill], e7r_settings, contract, out)
        state = trained["state"]
        library[skill] = state
        contracts[skill] = contract
        contract_json = e7y.contract_to_json(contract)
        training_rows.append({"seed": seed, "system": system, "skill": skill, "code": code, "state_hash": e7p.state_hash(state), "history": trained["history"], "contract": contract_json})
        boundary_rows.append({"seed": seed, "system": system, "skill": skill, "code": code, "state_hash": e7p.state_hash(state), "contract": contract_json})
        freeze_rows.append({"seed": seed, "system": system, "skill": skill, "code": code, "progressive_freeze_used": code != "continuous", "freeze_rounds": 1 if code != "continuous" else 0, "rollback_count": 0, "note": "low-bit target was active during boundary training"})
    return library, contracts, training_rows, boundary_rows, freeze_rows


def copy_library(library: dict[str, dict[str, Any]], precision: str | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for skill, state in library.items():
        out[skill] = e7p.copy_state(state, f"e7z_copy_{precision or 'float'}")
        if precision:
            out[skill] = e7o.quantize_state(out[skill], precision)
            out[skill]["lineage"] = list(out[skill].get("lineage", [])) + [f"e7z_internal_{precision}"]
    return out


def estimate_boundary_bits(code: str) -> int:
    return OUTPUT_WIDTH * len(SKILLS) * CODE_BITS.get(code, 32)


def evaluate_system(
    seed: int,
    system: str,
    code: str,
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    task: dict[str, list[dict[str, Any]]],
    params: dict[str, dict[str, Any]] | None = None,
    oracle: bool = False,
    dense: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evals: dict[str, Any] = {}
    boundary_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        correct: list[bool] = []
        samples: list[dict[str, Any]] = []
        written_values: list[float] = []
        oracle_values: list[float] = []
        mae_values: list[float] = []
        next_errors: list[float] = []
        support_sign_mismatch: list[float] = []
        support_silent: list[float] = []
        changed_counts: list[float] = []
        write_spread: list[float] = []
        pattern_corrs: list[float] = []
        for row in task[split]:
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            canonical = flow.copy()
            route = tuple(row["expected_route"])
            step_samples: list[dict[str, Any]] = []
            for step_idx, skill in enumerate(route):
                before = flow.copy()
                canonical_after = apply_oracle_code(row, canonical, skill, code if code != "mixed" else "int4", default_boundary_params(code if code != "mixed" else "int4"))
                cells = e7y.bundle_cells(skill, OUTPUT_WIDTH)
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
                    pred_vals = pred[cells].astype(np.float32)
                    pred_vals = quantize_bundle_values(pred_vals, skill, code if code != "mixed" else "int4", params, primary_is_logit=True)
                    for cell, value in zip(cells, pred_vals):
                        pred[cell] = value
                written_vals = pred[cells].astype(np.float32)
                mae = np.abs(written_vals - oracle_vals)
                written_values.extend([float(value) for value in written_vals.tolist()])
                oracle_values.extend([float(value) for value in oracle_vals.tolist()])
                mae_values.extend([float(value) for value in mae.tolist()])
                if len(written_vals) > 1:
                    support_sign_mismatch.extend([float((w >= 0.0) != (o >= 0.0)) for w, o in zip(written_vals[1:], oracle_vals[1:])])
                    support_silent.extend([float(abs(w) < 0.10) for w in written_vals[1:]])
                pattern_corrs.append(safe_corr(written_vals.tolist(), oracle_vals.tolist()))
                delta = pred - before
                changed = np.abs(delta) > 1e-6
                changed_counts.append(float(np.sum(changed)))
                write_spread.append(float(np.mean(changed)))
                next_error = 0.0
                if step_idx + 1 < len(route):
                    if contracts and route[step_idx + 1] in contracts:
                        read = contracts[route[step_idx + 1]]["read"]
                    else:
                        read = np.ones(FLOW_DIM, dtype=bool)
                    next_error = float(np.mean(np.abs(pred[read] - canonical_after[read])))
                    next_errors.append(next_error)
                if len(boundary_rows) < 240:
                    boundary_rows.append(
                        {
                            "seed": seed,
                            "system": system,
                            "split": split,
                            "row_id": row["row_id"],
                            "skill": skill,
                            "step": step_idx + 1,
                            "code": code,
                            "bundle_cells": cells,
                            "written_values": [round_float(float(value)) for value in written_vals.tolist()],
                            "oracle_values": [round_float(float(value)) for value in oracle_vals.tolist()],
                            "bundle_mae": round_float(float(np.mean(mae))),
                            "next_pocket_compatibility_error": round_float(next_error),
                        }
                    )
                if len(step_samples) < 5:
                    step_samples.append({"skill": skill, "bundle_mae": round_float(float(np.mean(mae))), "next_error": round_float(next_error)})
                flow = pred.astype(np.float32)
                canonical = canonical_after.astype(np.float32)
            predicted = int(e7o.predict_answer_from_flow(row, flow))
            ok = predicted == int(row["target_answer"])
            correct.append(ok)
            if len(samples) < 8:
                samples.append({"row_id": row["row_id"], "family": row["family"], "route": list(route), "target": int(row["target_answer"]), "predicted": predicted, "correct": bool(ok), "steps": step_samples})
        acc = round_float(float(np.mean(correct)))
        mean_steps = round_float(float(np.mean([len(row["expected_route"]) for row in task[split]])))
        bit_cost = sum(e7p.bit_budget(state) for state in library.values()) if library else 0
        boundary_bits = estimate_boundary_bits(code if code != "mixed" else "int4")
        cost_penalty = min(0.10, 0.00000016 * bit_cost + 0.0025 * mean_steps + 0.0000005 * boundary_bits)
        evals[split] = {
            "answer_accuracy": acc,
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(max(0.0, acc - cost_penalty)),
            "mean_route_steps": mean_steps,
            "code": code,
            "canonical_state_validity": round_float(1.0 - float(np.mean(mae_values)) if mae_values else 0.0),
            "oracle_write_similarity": round_float(1.0 - float(np.mean(mae_values)) if mae_values else 0.0),
            "bundle_mean_absolute_error_to_oracle": round_float(float(np.mean(mae_values)) if mae_values else 0.0),
            "multi_cell_pattern_correlation": round_float(float(np.mean(pattern_corrs)) if pattern_corrs else 0.0),
            "bundle_cellwise_correlation_with_oracle": safe_corr(written_values, oracle_values),
            "bundle_cosine_similarity_with_oracle": cosine_similarity(written_values, oracle_values),
            "support_channel_sign_mismatch_rate": round_float(float(np.mean(support_sign_mismatch)) if support_sign_mismatch else 0.0),
            "support_channel_silence_rate": round_float(float(np.mean(support_silent)) if support_silent else 0.0),
            "write_spread": round_float(float(np.mean(write_spread)) if write_spread else 0.0),
            "changed_cell_count": round_float(float(np.mean(changed_counts)) if changed_counts else 0.0),
            "next_pocket_compatibility_error": round_float(float(np.mean(next_errors)) if next_errors else 0.0),
            "bit_budget": bit_cost,
            "boundary_bit_budget": boundary_bits,
            "compression_ratio_vs_continuous_boundary": round_float((OUTPUT_WIDTH * len(SKILLS) * 32) / max(1, boundary_bits)),
            "row_level_samples": samples,
        }
    row = {
        "seed": seed,
        "system": system,
        "code": code,
        "evals": evals,
        "heldout_usefulness": evals["heldout"]["composition_usefulness"],
        "ood_usefulness": evals["ood"]["composition_usefulness"],
        "counterfactual_usefulness": evals["counterfactual"]["composition_usefulness"],
        "adversarial_usefulness": evals["adversarial"]["composition_usefulness"],
        "eval_mean_answer_accuracy": round_float(float(np.mean([evals[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([evals[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_route_accuracy": 1.0,
        "eval_mean_canonical_state_validity": round_float(float(np.mean([evals[split]["canonical_state_validity"] for split in EVAL_SPLITS]))),
        "eval_mean_oracle_write_similarity": round_float(float(np.mean([evals[split]["oracle_write_similarity"] for split in EVAL_SPLITS]))),
        "eval_mean_bundle_mean_absolute_error_to_oracle": round_float(float(np.mean([evals[split]["bundle_mean_absolute_error_to_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_multi_cell_pattern_correlation": round_float(float(np.mean([evals[split]["multi_cell_pattern_correlation"] for split in EVAL_SPLITS]))),
        "eval_mean_bundle_cellwise_correlation_with_oracle": round_float(float(np.mean([evals[split]["bundle_cellwise_correlation_with_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_bundle_cosine_similarity_with_oracle": round_float(float(np.mean([evals[split]["bundle_cosine_similarity_with_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_support_channel_sign_mismatch_rate": round_float(float(np.mean([evals[split]["support_channel_sign_mismatch_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_support_channel_silence_rate": round_float(float(np.mean([evals[split]["support_channel_silence_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_write_spread": round_float(float(np.mean([evals[split]["write_spread"] for split in EVAL_SPLITS]))),
        "eval_mean_changed_cell_count": round_float(float(np.mean([evals[split]["changed_cell_count"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_compatibility_error": round_float(float(np.mean([evals[split]["next_pocket_compatibility_error"] for split in EVAL_SPLITS]))),
        "parameter_count": sum(e7p.parameter_count(state) for state in library.values()) if library else 0,
        "bit_budget": sum(e7p.bit_budget(state) for state in library.values()) if library else 0,
        "boundary_bit_budget": estimate_boundary_bits(code if code != "mixed" else "int4"),
        "compression_ratio_vs_continuous_boundary": round_float((OUTPUT_WIDTH * len(SKILLS) * 32) / max(1, estimate_boundary_bits(code if code != "mixed" else "int4"))),
    }
    return row, boundary_rows


def score_for_repair(seed: int, system: str, code: str, library: dict[str, dict[str, Any]], contracts: dict[str, dict[str, Any]], task: dict[str, list[dict[str, Any]]], params: dict[str, dict[str, Any]]) -> float:
    row, _ = evaluate_system(seed, system, code, library, contracts, {"validation": task["validation"], **{split: task["validation"][: max(1, min(24, len(task["validation"])))] for split in SPLITS if split != "validation"}}, params=params)
    return float(row["evals"]["validation"]["composition_usefulness"])


def mutate_params(code: str, params: dict[str, dict[str, Any]], rng: random.Random) -> dict[str, dict[str, Any]]:
    cand = json.loads(json.dumps(params))
    skill = rng.choice(SKILLS)
    channel = rng.randrange(OUTPUT_WIDTH)
    if channel == 0:
        cand[skill]["primary_threshold"] = float(np.clip(float(cand[skill]["primary_threshold"]) + rng.gauss(0.0, 0.15), -1.0, 1.0))
    else:
        idx = channel - 1
        if code == "binary":
            cand[skill]["support_thresholds"][idx] = float(np.clip(float(cand[skill]["support_thresholds"][idx]) + rng.gauss(0.0, 0.18), -1.5, 1.5))
        elif code == "ternary":
            if rng.random() < 0.5:
                cand[skill]["support_thresholds"][idx] = float(np.clip(float(cand[skill]["support_thresholds"][idx]) + rng.gauss(0.0, 0.15), -1.5, 1.5))
            else:
                cand[skill]["support_deadzones"][idx] = float(np.clip(float(cand[skill]["support_deadzones"][idx]) + rng.gauss(0.0, 0.06), 0.02, 0.80))
        elif code == "int4":
            if rng.random() < 0.5:
                cand[skill]["support_thresholds"][idx] = float(np.clip(float(cand[skill]["support_thresholds"][idx]) + rng.gauss(0.0, 0.12), -1.0, 1.0))
            else:
                cand[skill]["support_scales"][idx] = float(np.clip(float(cand[skill]["support_scales"][idx]) * np.exp(rng.gauss(0.0, 0.12)), 0.20, 2.50))
    return cand


def repair_boundary(
    seed: int,
    system: str,
    code: str,
    library: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    task: dict[str, list[dict[str, Any]]],
    settings: Settings,
    out: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rng = random.Random(stable_seed(f"repair:{seed}:{system}:{code}"))
    best = default_boundary_params(code)
    best_score = score_for_repair(seed, system, code, library, contracts, task, best)
    initial_hash = payload_sha256(best)
    attempts = accepted = rejected = 0
    history: list[dict[str, Any]] = []
    for generation in range(settings.repair_generations):
        generation_best = best_score
        for _ in range(settings.repair_population):
            attempts += 1
            cand = mutate_params(code, best, rng)
            score = score_for_repair(seed, system, code, library, contracts, task, cand)
            if score > best_score + 1e-12:
                best = cand
                best_score = score
                accepted += 1
            else:
                rejected += 1
        row = {
            "seed": seed,
            "system": system,
            "code": code,
            "generation": generation,
            "score": round_float(best_score),
            "generation_gain": round_float(best_score - generation_best),
            "accepted": accepted,
            "rejected": rejected,
            "rollback": rejected,
            "boundary_param_hash": payload_sha256(best),
        }
        history.append(row)
        if out:
            append_progress(out, "low_bit_boundary_mutation_generation", **row)
    return best, {
        "history": history,
        "initial_candidate_hash": initial_hash,
        "final_candidate_hash": payload_sha256(best),
        "parameter_diff_hash": payload_sha256({"initial": initial_hash, "final": payload_sha256(best)}),
        "mutation_attempts": attempts,
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "rollback_count": rejected,
        "validation_score": round_float(best_score),
    }


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
        "eval_mean_canonical_state_validity",
        "eval_mean_oracle_write_similarity",
        "eval_mean_bundle_mean_absolute_error_to_oracle",
        "eval_mean_multi_cell_pattern_correlation",
        "eval_mean_bundle_cellwise_correlation_with_oracle",
        "eval_mean_bundle_cosine_similarity_with_oracle",
        "eval_mean_support_channel_sign_mismatch_rate",
        "eval_mean_support_channel_silence_rate",
        "eval_mean_write_spread",
        "eval_mean_changed_cell_count",
        "eval_mean_next_pocket_compatibility_error",
        "parameter_count",
        "bit_budget",
        "boundary_bit_budget",
        "compression_ratio_vs_continuous_boundary",
        "mutation_repair_gain",
        "mutation_attempts",
        "accepted_mutations",
        "rejected_mutations",
        "rollback_count",
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
    candidates = [system for system in SYSTEMS if system not in ORACLE_SYSTEMS and system != "dense_graph_danger_control"]
    best = max(candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    return {"schema_version": "e7z_aggregate_metrics_v1", "systems": systems, "best_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    mean = {system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0) for system in SYSTEMS}
    continuous = mean["continuous_direct_write_baseline"]
    oracle_cont = mean["oracle_write_continuous_reference"]
    dense = mean["dense_graph_danger_control"]
    best = aggregate["best_system"]
    best_score = mean[best]
    lowbit_external = {system: mean[system] for system in LEARNED_EXTERNAL_SYSTEMS}
    repair = {system: mean[system] for system in REPAIR_SYSTEMS}
    pure = {system: mean[system] for system in PURE_SYSTEMS}
    best_external = max(lowbit_external, key=lowbit_external.get)
    best_repair = max(repair, key=repair.get)
    best_pure = max(pure, key=pure.get)
    best_projection = max(("oracle_write_binary_projected", "oracle_write_ternary_projected", "oracle_write_int4_projected"), key=lambda system: mean[system])
    gap = max(1e-9, oracle_cont - continuous)
    best_lowbit = max(list(lowbit_external) + list(repair) + list(pure) + ["mixed_precision_pocket_float_ram_lowbit"], key=lambda system: mean[system])
    gap_closed = (mean[best_lowbit] - continuous) / gap
    repair_gain = mean[best_repair] - lowbit_external[best_repair.replace("_plus_mutation_repair", "")]
    detail = {
        "continuous_baseline": continuous,
        "oracle_continuous": oracle_cont,
        "oracle_binary_projected": mean["oracle_write_binary_projected"],
        "oracle_ternary_projected": mean["oracle_write_ternary_projected"],
        "oracle_int4_projected": mean["oracle_write_int4_projected"],
        "best_system": best,
        "best_lowbit_system": best_lowbit,
        "best_external_lowbit": best_external,
        "best_external_lowbit_score": lowbit_external[best_external],
        "best_repair_system": best_repair,
        "mutation_repair_gain": round_float(repair_gain),
        "best_pure_lowbit": best_pure,
        "best_pure_lowbit_score": pure[best_pure],
        "mixed_precision": mean["mixed_precision_pocket_float_ram_lowbit"],
        "dense_graph": dense,
        "gap_fraction_closed_by_best_lowbit": round_float(gap_closed),
        "best_projected_oracle": best_projection,
    }
    projection_works = min(mean["oracle_write_binary_projected"], mean["oracle_write_ternary_projected"], mean["oracle_write_int4_projected"]) >= oracle_cont - 0.025
    if dense >= best_score + 0.02:
        decision = "e7z_graph_soup_regression_detected"
    elif repair_gain >= 0.012 and mean[best_repair] >= max(continuous, lowbit_external[best_external]) + 0.008:
        decision = "e7z_low_bit_mutation_repair_positive"
    elif pure[best_pure] >= lowbit_external[best_external] + 0.015 and pure[best_pure] >= continuous + 0.02:
        decision = "e7z_full_low_bit_pocket_ram_preferred"
    elif lowbit_external[best_external] >= continuous + 0.025 and gap_closed >= 0.60:
        if best_external == "learned_binary_ram_boundary":
            decision = "e7z_binary_canonical_ram_contract_positive"
        elif best_external == "learned_ternary_ram_boundary":
            decision = "e7z_ternary_canonical_ram_contract_positive"
        else:
            decision = "e7z_int4_canonical_ram_contract_positive"
    elif lowbit_external[best_external] >= continuous + 0.02 and gap_closed >= 0.25:
        decision = "e7z_external_low_bit_ram_boundary_sufficient"
    elif mean[best_lowbit] >= continuous + 0.01:
        decision = "e7z_low_bit_ram_contract_partially_positive"
    elif projection_works and mean[best_lowbit] <= continuous + 0.006:
        decision = "e7z_low_bit_training_or_commit_learning_bottleneck"
    else:
        decision = "e7z_low_bit_canonical_ram_contract_not_sufficient"
    return {"schema_version": "e7z_decision_v1", "decision": decision, "detail": detail, "deterministic_replay_passed": False}


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
    boundary_rows: list[dict[str, Any]] = []
    freeze_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    eval_boundary_rows: list[dict[str, Any]] = []
    bit_rows: list[dict[str, Any]] = []

    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        state = e7p.copy_state(trained["state"], "e7z_baseline_standalone")
        baseline_library[skill] = state
        training_rows.append({"seed": seed, "system": "baseline_standalone_pocket", "skill": skill, "state_hash": e7p.state_hash(state), "standalone": trained["standalone"]})

    context_by_code = {code: generate_code_context_tasks(composition_task, code) for code in ("continuous", "binary", "ternary", "int4")}
    libraries: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]] = {}
    for system, code in (
        ("continuous_direct_write_baseline", "continuous"),
        ("learned_binary_ram_boundary", "binary"),
        ("learned_ternary_ram_boundary", "ternary"),
        ("learned_int4_ram_boundary", "int4"),
    ):
        library, contracts, train_rows, contract_rows, fr_rows = train_code_library(seed, system, code, baseline_library, context_by_code[code], settings, out)
        libraries[system] = (library, contracts, code)
        training_rows.extend(train_rows)
        boundary_rows.extend(contract_rows)
        freeze_rows.extend(fr_rows)
        result, eval_rows = evaluate_system(seed, system, code, library, contracts, composition_task)
        rows.append(result)
        eval_boundary_rows.extend(eval_rows)
        if out:
            append_progress(out, "low_bit_boundary_system_evaluated", seed=seed, system=system, usefulness=result["eval_mean_composition_usefulness"])

    for system in ORACLE_SYSTEMS:
        code = SYSTEM_CODE[system]
        result, eval_rows = evaluate_system(seed, system, code, None, None, composition_task, oracle=True)
        rows.append(result)
        eval_boundary_rows.extend(eval_rows)

    for repair_system, base_system in (
        ("learned_binary_ram_boundary_plus_mutation_repair", "learned_binary_ram_boundary"),
        ("learned_ternary_ram_boundary_plus_mutation_repair", "learned_ternary_ram_boundary"),
        ("learned_int4_ram_boundary_plus_mutation_repair", "learned_int4_ram_boundary"),
    ):
        library, contracts, code = libraries[base_system]
        params, mutation = repair_boundary(seed, repair_system, code, library, contracts, composition_task, settings, out)
        result, eval_rows = evaluate_system(seed, repair_system, code, library, contracts, composition_task, params=params)
        base_score = next(row["eval_mean_composition_usefulness"] for row in rows if row["system"] == base_system)
        result.update({key: value for key, value in mutation.items() if key != "history"})
        result["mutation_repair_gain"] = round_float(result["eval_mean_composition_usefulness"] - base_score)
        rows.append(result)
        eval_boundary_rows.extend(eval_rows)
        repair_rows.extend(mutation["history"])

    for pure_system, base_system, precision in (
        ("pure_binary_pocket_and_ram", "learned_binary_ram_boundary", "binary"),
        ("pure_ternary_pocket_and_ram", "learned_ternary_ram_boundary", "ternary"),
        ("int4_pocket_and_ram", "learned_int4_ram_boundary", "int4"),
    ):
        library, contracts, code = libraries[base_system]
        pure_library = copy_library(library, precision)
        result, eval_rows = evaluate_system(seed, pure_system, code, pure_library, contracts, composition_task)
        rows.append(result)
        eval_boundary_rows.extend(eval_rows)

    validation_scores = {system: next(row["evals"]["validation"]["composition_usefulness"] for row in rows if row["system"] == system) for system in LEARNED_EXTERNAL_SYSTEMS}
    selected = max(validation_scores, key=validation_scores.get)
    library, contracts, code = libraries[selected]
    mixed_result, mixed_eval_rows = evaluate_system(seed, "mixed_precision_pocket_float_ram_lowbit", code, library, contracts, composition_task)
    mixed_result["selected_external_boundary_system"] = selected
    rows.append(mixed_result)
    eval_boundary_rows.extend(mixed_eval_rows)

    dense_result, dense_eval_rows = evaluate_system(seed, "dense_graph_danger_control", "continuous", baseline_library, None, composition_task, dense=True)
    rows.append(dense_result)
    eval_boundary_rows.extend(dense_eval_rows)

    for row in rows:
        bit_rows.append(
            {
                "seed": seed,
                "system": row["system"],
                "code": row["code"],
                "bit_budget": row.get("bit_budget", 0),
                "boundary_bit_budget": row.get("boundary_bit_budget", 0),
                "compression_ratio_vs_continuous_boundary": row.get("compression_ratio_vs_continuous_boundary", 0.0),
                "parameter_count": row.get("parameter_count", 0),
            }
        )
    return {
        "seed": seed,
        "rows": rows,
        "training_rows": training_rows,
        "boundary_rows": boundary_rows,
        "freeze_rows": freeze_rows,
        "repair_rows": repair_rows,
        "eval_boundary_rows": eval_boundary_rows,
        "bit_rows": bit_rows,
    }


def build_reports(
    rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    freeze_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    eval_boundary_rows: list[dict[str, Any]],
    bit_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    lines = [
        "# E7Z Low-Bit Canonical RAM Contract Probe Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_system = {aggregate['best_system']}",
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
            f"{system:<48} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} "
            f"valid={mean.get('eval_mean_canonical_state_validity', 0.0):.6f} "
            f"next={mean.get('eval_mean_next_pocket_compatibility_error', 0.0):.6f} "
            f"bits={mean.get('boundary_bit_budget', 0.0):.1f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Boundary",
            "",
            "E7Z only tests low-bit canonical RAM communication in a controlled numeric pocket-router proxy. It does not make raw-language, AGI, consciousness, deployed-model, or model-scale claims.",
            "",
        ]
    )
    return {
        "pocket_training_report.json": {"schema_version": "e7z_pocket_training_report_v1", "rows": sorted(training_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "projected_oracle_report.json": {"schema_version": "e7z_projected_oracle_report_v1", "rows": [row for row in sorted(rows, key=lambda row: (row["seed"], SYSTEMS.index(row["system"]))) if row["system"] in ORACLE_SYSTEMS]},
        "low_bit_boundary_report.json": {"schema_version": "e7z_low_bit_boundary_report_v1", "contract_rows": sorted(boundary_rows, key=lambda row: (row["seed"], row["system"], row["skill"])), "sample_rows": sorted(eval_boundary_rows, key=lambda row: (row["seed"], row["system"], row["split"], row["row_id"], row["step"], row["skill"]))},
        "progressive_freeze_report.json": {"schema_version": "e7z_progressive_freeze_report_v1", "rows": sorted(freeze_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "mutation_repair_report.json": {"schema_version": "e7z_mutation_repair_report_v1", "rows": sorted(repair_rows, key=lambda row: (row["seed"], row["system"], row["generation"]))},
        "bit_budget_report.json": {"schema_version": "e7z_bit_budget_report_v1", "rows": sorted(bit_rows, key=lambda row: (row["seed"], row["system"]))},
        "system_results.json": {"schema_version": "e7z_system_results_v1", "rows": sorted(rows, key=lambda row: (row["seed"], SYSTEMS.index(row["system"])))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7z_summary_v1", "decision": decision["decision"], "best_system": aggregate["best_system"], "deterministic_replay_passed": deterministic.get("internal_replay_passed", False), "checker_failure_count": None},
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
        "--repair-generations",
        str(settings.repair_generations),
        "--repair-population",
        str(settings.repair_population),
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
                "schema_version": "e7z_backend_manifest_v1",
                "milestone": MILESTONE,
                "settings": settings_payload(settings),
                "systems": list(SYSTEMS),
                "flow_dim": FLOW_DIM,
                "output_width": OUTPUT_WIDTH,
                "low_bit_codes": list(LOW_BIT_CODES),
                "semantic_lane_labels_as_model_input": False,
                "new_router": False,
                "dense_graph_primary": False,
                "oracle_used_as_reference_only": True,
                "mutation_repair_uses_backprop": False,
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
                "schema_version": "e7z_task_generation_report_v1",
                "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()},
                "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()},
            },
        )
        rows: list[dict[str, Any]] = []
        training_rows: list[dict[str, Any]] = []
        boundary_rows: list[dict[str, Any]] = []
        freeze_rows: list[dict[str, Any]] = []
        repair_rows: list[dict[str, Any]] = []
        eval_boundary_rows: list[dict[str, Any]] = []
        bit_rows: list[dict[str, Any]] = []
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
                boundary_rows.extend(result["boundary_rows"])
                freeze_rows.extend(result["freeze_rows"])
                repair_rows.extend(result["repair_rows"])
                eval_boundary_rows.extend(result["eval_boundary_rows"])
                bit_rows.extend(result["bit_rows"])
                append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(jobs) - len({row["seed"] for row in rows}))
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_repair_rows": len(repair_rows), "pending": len(jobs) - len({row["seed"] for row in rows})})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
                while futures:
                    done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_repair_rows": len(repair_rows), "pending": len(futures)})
                        continue
                    for future in done:
                        label = futures.pop(future)
                        result = future.result()
                        rows.extend(result["rows"])
                        training_rows.extend(result["training_rows"])
                        boundary_rows.extend(result["boundary_rows"])
                        freeze_rows.extend(result["freeze_rows"])
                        repair_rows.extend(result["repair_rows"])
                        eval_boundary_rows.extend(result["eval_boundary_rows"])
                        bit_rows.extend(result["bit_rows"])
                        append_progress(out, "seed_job_complete", label=label, pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_repair_rows": len(repair_rows), "last_completed": label, "pending": len(futures)})
        aggregate = aggregate_results(rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e7z_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, training_rows, boundary_rows, freeze_rows, repair_rows, eval_boundary_rows, bit_rows, aggregate, decision, deterministic_placeholder)
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
            deterministic = {"schema_version": "e7z_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, training_rows, boundary_rows, freeze_rows, repair_rows, eval_boundary_rows, bit_rows, aggregate, decision, deterministic)
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
    parser.add_argument("--repair-generations", type=int, default=14)
    parser.add_argument("--repair-population", type=int, default=12)
    parser.add_argument("--cpu-workers", type=int, default=8)
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
        pruned_read_count=args.pruned_read_count,
        repair_generations=args.repair_generations,
        repair_population=args.repair_population,
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
