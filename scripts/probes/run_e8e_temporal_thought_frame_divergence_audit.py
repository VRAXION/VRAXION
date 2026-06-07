#!/usr/bin/env python3
"""E8E temporal thought-frame divergence audit.

E8C/E8D showed that producer target decomposition and substrate-first snapshot
pretraining did not close the numeric pocket/RAM composition gap. E8E is a
diagnostic-only trace audit: compare oracle Flow/RAM trajectories against
learned trajectories step by step, then run explicit oracle-reset interventions
to localize whether failure is immediate write divergence, accumulated drift,
consumer-sensitive state mismatch, or a wrong attractor-like trace.
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
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
E8D_PATH = Path(__file__).with_name("run_e8d_substrate_first_ram_language_pretraining_probe.py")
MILESTONE = "E8E_TEMPORAL_THOUGHT_FRAME_DIVERGENCE_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e8e_temporal_thought_frame_divergence_audit")
DEFAULT_SEEDS = (105001, 105002, 105003, 105004, 105005, 105006, 105007, 105008)
OUTPUT_WIDTH = 12
DIVERGENCE_THRESHOLD = 0.18

SYSTEMS = (
    "oracle_trace_reference",
    "current_best_learned_trace",
    "consumer_distill_trace_reference",
    "substrate_first_trace",
    "mutation_only_trace",
    "dense_graph_danger_trace",
)
ORACLE_SYSTEMS = {"oracle_trace_reference", "consumer_distill_trace_reference"}
MUTATION_SYSTEMS = {"mutation_only_trace"}
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "trace_divergence_report.json",
    "intervention_report.json",
    "attractor_report.json",
    "local_editability_report.json",
    "mutation_history_report.json",
    "system_results.json",
    "row_level_samples.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e8e_first_step_write_divergence",
    "e8e_temporal_drift_accumulation",
    "e8e_consumer_sensitive_state_mismatch",
    "e8e_recoverable_state_drift",
    "e8e_wrong_attractor_trace",
    "e8e_answer_shortcut_trace_invalid",
)
INTERVENTIONS = (
    "oracle_reset_after_step_1",
    "oracle_reset_after_each_step",
    "learned_step_1_oracle_rest",
    "oracle_step_1_learned_rest",
    "one_learned_pocket_at_a_time",
    "consumer_sensitive_cell_replacement_only",
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e8d = load_module(E8D_PATH, "e8d_substrate_first_probe")
e8c = e8d.e8c
e8a = e8d.e8a
e7z = e8d.e7z
e7y = e8d.e7y
e7r = e8d.e7r
e7p = e8d.e7p
e7o = e8d.e7o

FLOW_DIM = int(e8d.FLOW_DIM)
SKILLS = tuple(e8d.SKILLS)
SPLITS = tuple(e8d.SPLITS)
EVAL_SPLITS = tuple(e8d.EVAL_SPLITS)
RESULT_POS = dict(e8d.RESULT_POS)


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
    gradient_diagnostic_batch_size: int
    learning_rate: float
    local_learning_rate: float
    weight_decay: float
    pruned_read_count: int
    repair_generations: int
    repair_population: int
    similarity_threshold: float
    trace_sample_limit_per_split: int
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    replay: bool
    execution_mode: str


def round_float(value: float) -> float:
    return e8d.round_float(value)


def write_json(path: Path, payload: Any) -> None:
    e8d.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    e8d.write_text(path, text)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e8d.append_progress(out, event, **details)


def resolve_out(path: str | Path) -> Path:
    return e8d.resolve_out(path)


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return e8d.parse_int_tuple(raw)


def select_device(requested: str) -> str:
    return e8d.select_device(requested)


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    payload["replay"] = False
    return payload


def to_e8d_settings(settings: Settings) -> Any:
    return e8d.Settings(
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
        gradient_diagnostic_batch_size=settings.gradient_diagnostic_batch_size,
        learning_rate=settings.learning_rate,
        local_learning_rate=settings.local_learning_rate,
        weight_decay=settings.weight_decay,
        pruned_read_count=settings.pruned_read_count,
        repair_generations=settings.repair_generations,
        repair_population=settings.repair_population,
        similarity_threshold=settings.similarity_threshold,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        replay=settings.replay,
        execution_mode=settings.execution_mode,
    )


def to_e8c_settings(settings: Settings) -> Any:
    return e8d.to_e8c_settings(to_e8d_settings(settings))


def to_e7o_settings(settings: Settings) -> Any:
    return e8d.to_e7o_settings(to_e8d_settings(settings))


def to_e7z_settings(settings: Settings) -> Any:
    return e8d.to_e7z_settings(to_e8d_settings(settings))


def vec_cos(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return round_float(float(np.dot(left, right) / denom) if denom > 1e-12 else 0.0)


def vec_corr(left: np.ndarray, right: np.ndarray) -> float:
    return e7z.safe_corr(np.asarray(left, dtype=np.float32).reshape(-1).tolist(), np.asarray(right, dtype=np.float32).reshape(-1).tolist())


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    return round_float(float(np.sum(x * (y - float(np.mean(y)))) / denom) if denom > 1e-12 else 0.0)


def read_mask_for(next_skill: str | None, contracts: dict[str, dict[str, Any]] | None) -> np.ndarray:
    if next_skill and contracts and next_skill in contracts:
        return contracts[next_skill]["read"].astype(bool)
    if next_skill:
        return e7z.build_lowbit_contract(next_skill, "e8e_consumer_read_reference", read_count=OUTPUT_WIDTH)["read"].astype(bool)
    return np.zeros(FLOW_DIM, dtype=bool)


def oracle_frames(row: dict[str, Any]) -> list[np.ndarray]:
    frames = [np.asarray(row["flow"], dtype=np.float32).copy()]
    flow = frames[0].copy()
    for skill in tuple(row["expected_route"]):
        flow = e8c.apply_e8c_teacher_code(row, flow, skill, "int4", "current_full_code_teacher")
        frames.append(flow.astype(np.float32).copy())
    return frames


def apply_system_step(
    system: str,
    skill: str,
    flow: np.ndarray,
    oracle_after: np.ndarray,
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    params: dict[str, dict[str, Any]] | None,
    dense: bool,
) -> np.ndarray:
    if system in ORACLE_SYSTEMS:
        return oracle_after.astype(np.float32).copy()
    assert library is not None
    if dense:
        pred = e7p.np_forward(library[skill], flow.reshape(1, -1)).reshape(-1).astype(np.float32)
    else:
        assert contracts is not None
        pred = e7r.masked_forward_np(library[skill], flow.reshape(1, -1), contracts[skill]).reshape(-1).astype(np.float32)
        cells = np.asarray(e7y.bundle_cells(skill, OUTPUT_WIDTH), dtype=np.int64)
        pred[cells] = e7z.quantize_bundle_values(pred[cells], skill, "int4", params, primary_is_logit=True)
    return pred.astype(np.float32)


def run_trace(
    system: str,
    row: dict[str, Any],
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    params: dict[str, dict[str, Any]] | None = None,
    dense: bool = False,
    intervention: str | None = None,
    one_learned_index: int | None = None,
) -> list[np.ndarray]:
    route = tuple(row["expected_route"])
    oracle = oracle_frames(row)
    flow = oracle[0].copy()
    frames = [flow.copy()]
    for step_idx, skill in enumerate(route):
        oracle_after = oracle[step_idx + 1]
        use_oracle = system in ORACLE_SYSTEMS
        if intervention == "learned_step_1_oracle_rest" and step_idx > 0:
            use_oracle = True
        if intervention == "oracle_step_1_learned_rest" and step_idx == 0:
            use_oracle = True
        if intervention == "one_learned_pocket_at_a_time":
            use_oracle = step_idx != one_learned_index
        if use_oracle:
            pred = oracle_after.copy()
        else:
            pred = apply_system_step(system, skill, flow, oracle_after, library, contracts, params, dense)
            if intervention == "consumer_sensitive_cell_replacement_only" and step_idx + 1 < len(route):
                read = read_mask_for(route[step_idx + 1], contracts)
                pred[read] = oracle_after[read]
        frames.append(pred.astype(np.float32).copy())
        flow = pred.astype(np.float32).copy()
        if intervention == "oracle_reset_after_step_1" and step_idx == 0:
            flow = oracle_after.copy()
            frames[-1] = flow.copy()
        elif intervention == "oracle_reset_after_each_step":
            flow = oracle_after.copy()
            frames[-1] = flow.copy()
    return frames


def frame_rows_for_trace(
    seed: int,
    system: str,
    split: str,
    row: dict[str, Any],
    learned: list[np.ndarray],
    oracle: list[np.ndarray],
    contracts: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route = tuple(row["expected_route"])
    step_rows: list[dict[str, Any]] = []
    frame_mae_values: list[float] = []
    read_mae_values: list[float] = []
    delta_mae_values: list[float] = []
    transition_validity_values: list[float] = []
    result_errors: list[float] = []
    sign_mismatches: list[float] = []
    first_divergence = len(route) + 1
    wrong_attractor_hits = 0
    for step_idx, skill in enumerate(route):
        before = learned[step_idx]
        after = learned[step_idx + 1]
        oracle_before = oracle[step_idx]
        oracle_after = oracle[step_idx + 1]
        cells = np.asarray(e7y.bundle_cells(skill, OUTPUT_WIDTH), dtype=np.int64)
        frame_mae = float(np.mean(np.abs(after - oracle_after)))
        delta_mae = float(np.mean(np.abs((after - before) - (oracle_after - oracle_before))))
        next_skill = route[step_idx + 1] if step_idx + 1 < len(route) else None
        read = read_mask_for(next_skill, contracts)
        read_mae = float(np.mean(np.abs(after[read] - oracle_after[read]))) if bool(np.any(read)) else 0.0
        bundle = after[cells]
        oracle_bundle = oracle_after[cells]
        result_error = float(abs(after[RESULT_POS[skill]] - oracle_after[RESULT_POS[skill]]))
        sign_mismatch = float(np.mean((bundle[1:] >= 0.0) != (oracle_bundle[1:] >= 0.0))) if len(bundle) > 1 else 0.0
        transition_validity = max(0.0, 1.0 - delta_mae)
        if frame_mae > DIVERGENCE_THRESHOLD and first_divergence == len(route) + 1:
            first_divergence = step_idx + 1
        local_step = float(np.mean(np.abs(after - before)))
        oracle_step = float(np.mean(np.abs(oracle_after - oracle_before)))
        collapse = bool(local_step < 0.015 and oracle_step > 0.050 and frame_mae > DIVERGENCE_THRESHOLD)
        wrong_attractor_hits += int(collapse)
        frame_mae_values.append(frame_mae)
        delta_mae_values.append(delta_mae)
        read_mae_values.append(read_mae)
        transition_validity_values.append(transition_validity)
        result_errors.append(result_error)
        sign_mismatches.append(sign_mismatch)
        step_rows.append(
            {
                "seed": seed,
                "system": system,
                "split": split,
                "row_id": row["row_id"],
                "family": row["family"],
                "route_step": step_idx,
                "skill": skill,
                "next_skill": next_skill,
                "frame_mae_to_oracle": round_float(frame_mae),
                "frame_cosine_to_oracle": vec_cos(after, oracle_after),
                "frame_correlation_to_oracle": vec_corr(after, oracle_after),
                "delta_mae_to_oracle": round_float(delta_mae),
                "consumer_read_mask_mae": round_float(read_mae),
                "result_cell_error": round_float(result_error),
                "support_cell_sign_mismatch": round_float(sign_mismatch),
                "transition_validity": round_float(transition_validity),
                "learned_step_magnitude": round_float(local_step),
                "oracle_step_magnitude": round_float(oracle_step),
                "collapse_like_step": collapse,
                "consumer_sensitive_cell_count": int(np.sum(read)),
                "bundle_cells": [int(v) for v in cells.tolist()],
            }
        )
    final_pos = RESULT_POS[route[-1]]
    predicted = int(learned[-1][final_pos] >= 0.5)
    answer_ok = predicted == int(row["target_answer"])
    if first_divergence == len(route) + 1:
        first_divergence = 0
    summary = {
        "answer_correct": bool(answer_ok),
        "predicted": predicted,
        "target": int(row["target_answer"]),
        "mean_frame_mae": float(np.mean(frame_mae_values)) if frame_mae_values else 0.0,
        "mean_delta_mae": float(np.mean(delta_mae_values)) if delta_mae_values else 0.0,
        "mean_consumer_read_mask_mae": float(np.mean(read_mae_values)) if read_mae_values else 0.0,
        "mean_transition_validity": float(np.mean(transition_validity_values)) if transition_validity_values else 0.0,
        "mean_result_cell_error": float(np.mean(result_errors)) if result_errors else 0.0,
        "mean_support_sign_mismatch": float(np.mean(sign_mismatches)) if sign_mismatches else 0.0,
        "first_divergence_step": first_divergence,
        "drift_slope": slope(frame_mae_values),
        "wrong_attractor_rate": float(wrong_attractor_hits / max(1, len(route))),
    }
    return step_rows, summary


def evaluate_trace_system(
    seed: int,
    system: str,
    task: dict[str, list[dict[str, Any]]],
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    params: dict[str, dict[str, Any]] | None = None,
    dense: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    split_results: dict[str, Any] = {}
    trace_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    attractor_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        answers: list[float] = []
        frame_mae: list[float] = []
        delta_mae: list[float] = []
        read_mae: list[float] = []
        transition_validity: list[float] = []
        result_error: list[float] = []
        sign_mismatch: list[float] = []
        first_divergence: list[float] = []
        drift_slopes: list[float] = []
        wrong_attractor: list[float] = []
        local_effects: list[float] = []
        for row_idx, row in enumerate(task[split]):
            oracle = oracle_frames(row)
            learned = run_trace(system, row, library, contracts, params=params, dense=dense)
            rows, summary = frame_rows_for_trace(seed, system, split, row, learned, oracle, contracts)
            trace_rows.extend(rows)
            answers.append(float(summary["answer_correct"]))
            frame_mae.append(float(summary["mean_frame_mae"]))
            delta_mae.append(float(summary["mean_delta_mae"]))
            read_mae.append(float(summary["mean_consumer_read_mask_mae"]))
            transition_validity.append(float(summary["mean_transition_validity"]))
            result_error.append(float(summary["mean_result_cell_error"]))
            sign_mismatch.append(float(summary["mean_support_sign_mismatch"]))
            first_divergence.append(float(summary["first_divergence_step"]))
            drift_slopes.append(float(summary["drift_slope"]))
            wrong_attractor.append(float(summary["wrong_attractor_rate"]))
            if row_idx < settings_trace_limit():
                sample_rows.append(sample_trace_row(seed, system, split, row, learned, oracle, rows, summary))
            if row_idx < 18:
                effect = local_editability_effect(seed, system, row, library, contracts, params, dense)
                local_effects.append(effect)
                edit_rows.append({"seed": seed, "system": system, "split": split, "row_id": row["row_id"], "local_perturbation_effect_size": round_float(effect)})
        answer_acc = float(np.mean(answers)) if answers else 0.0
        mean_frame = float(np.mean(frame_mae)) if frame_mae else 0.0
        mean_read = float(np.mean(read_mae)) if read_mae else 0.0
        mean_delta = float(np.mean(delta_mae)) if delta_mae else 0.0
        usefulness = answer_acc - 0.08 * mean_read - 0.05 * mean_frame - 0.03 * mean_delta
        split_results[split] = {
            "answer_accuracy": round_float(answer_acc),
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(usefulness),
            "trace_similarity": round_float(max(0.0, 1.0 - mean_frame)),
            "flow_frame_mae_to_oracle": round_float(mean_frame),
            "flow_delta_mae_to_oracle": round_float(mean_delta),
            "consumer_read_mask_mae": round_float(mean_read),
            "result_cell_error": round_float(float(np.mean(result_error)) if result_error else 0.0),
            "support_cell_sign_mismatch": round_float(float(np.mean(sign_mismatch)) if sign_mismatch else 0.0),
            "first_divergence_step": round_float(float(np.mean(first_divergence)) if first_divergence else 0.0),
            "drift_slope": round_float(float(np.mean(drift_slopes)) if drift_slopes else 0.0),
            "transition_validity": round_float(float(np.mean(transition_validity)) if transition_validity else 0.0),
            "wrong_attractor_rate": round_float(float(np.mean(wrong_attractor)) if wrong_attractor else 0.0),
            "local_perturbation_effect_size": round_float(float(np.mean(local_effects)) if local_effects else 0.0),
            "mean_route_steps": round_float(float(np.mean([len(row["expected_route"]) for row in task[split]])) if task[split] else 0.0),
        }
        attractor_rows.append(
            {
                "seed": seed,
                "system": system,
                "split": split,
                "wrong_attractor_rate": split_results[split]["wrong_attractor_rate"],
                "transition_validity": split_results[split]["transition_validity"],
                "trace_similarity": split_results[split]["trace_similarity"],
            }
        )
    result = {
        "seed": seed,
        "system": system,
        "evals": split_results,
        "eval_mean_answer_accuracy": round_float(float(np.mean([split_results[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([split_results[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_trace_similarity": round_float(float(np.mean([split_results[split]["trace_similarity"] for split in EVAL_SPLITS]))),
        "eval_mean_flow_frame_mae_to_oracle": round_float(float(np.mean([split_results[split]["flow_frame_mae_to_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_flow_delta_mae_to_oracle": round_float(float(np.mean([split_results[split]["flow_delta_mae_to_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_consumer_read_mask_mae": round_float(float(np.mean([split_results[split]["consumer_read_mask_mae"] for split in EVAL_SPLITS]))),
        "eval_mean_result_cell_error": round_float(float(np.mean([split_results[split]["result_cell_error"] for split in EVAL_SPLITS]))),
        "eval_mean_support_cell_sign_mismatch": round_float(float(np.mean([split_results[split]["support_cell_sign_mismatch"] for split in EVAL_SPLITS]))),
        "eval_mean_first_divergence_step": round_float(float(np.mean([split_results[split]["first_divergence_step"] for split in EVAL_SPLITS]))),
        "eval_mean_drift_slope": round_float(float(np.mean([split_results[split]["drift_slope"] for split in EVAL_SPLITS]))),
        "eval_mean_transition_validity": round_float(float(np.mean([split_results[split]["transition_validity"] for split in EVAL_SPLITS]))),
        "eval_mean_wrong_attractor_rate": round_float(float(np.mean([split_results[split]["wrong_attractor_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_local_perturbation_effect_size": round_float(float(np.mean([split_results[split]["local_perturbation_effect_size"] for split in EVAL_SPLITS]))),
        "oracle_write_at_inference": bool(system in ORACLE_SYSTEMS),
        "diagnostic_only": bool(system in ORACLE_SYSTEMS),
    }
    return result, trace_rows, sample_rows, attractor_rows, edit_rows


def settings_trace_limit() -> int:
    return int(os.environ.get("E8E_TRACE_SAMPLE_LIMIT", "3"))


def sample_trace_row(
    seed: int,
    system: str,
    split: str,
    row: dict[str, Any],
    learned: list[np.ndarray],
    oracle: list[np.ndarray],
    step_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    compact_steps = []
    route = tuple(row["expected_route"])
    for idx, skill in enumerate(route):
        cells = np.asarray(e7y.bundle_cells(skill, min(6, OUTPUT_WIDTH)), dtype=np.int64)
        compact_steps.append(
            {
                "route_step": idx,
                "skill": skill,
                "frame_mae_to_oracle": step_rows[idx]["frame_mae_to_oracle"],
                "consumer_read_mask_mae": step_rows[idx]["consumer_read_mask_mae"],
                "learned_bundle_head": [round_float(float(v)) for v in learned[idx + 1][cells].tolist()],
                "oracle_bundle_head": [round_float(float(v)) for v in oracle[idx + 1][cells].tolist()],
            }
        )
    return {
        "seed": seed,
        "system": system,
        "split": split,
        "row_id": row["row_id"],
        "family": row["family"],
        "route": list(route),
        "target": int(row["target_answer"]),
        "predicted": int(summary["predicted"]),
        "correct": bool(summary["answer_correct"]),
        "first_divergence_step": int(summary["first_divergence_step"]),
        "drift_slope": round_float(float(summary["drift_slope"])),
        "steps": compact_steps,
    }


def local_editability_effect(
    seed: int,
    system: str,
    row: dict[str, Any],
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    params: dict[str, dict[str, Any]] | None,
    dense: bool,
) -> float:
    if system in ORACLE_SYSTEMS:
        return 0.0
    route = tuple(row["expected_route"])
    rng = np.random.default_rng(e7z.stable_seed(f"E8E-edit:{seed}:{system}:{row['row_id']}"))
    base = run_trace(system, row, library, contracts, params=params, dense=dense)[-1]
    perturbed_row = dict(row)
    flow = np.asarray(row["flow"], dtype=np.float32).copy()
    cells = rng.choice(FLOW_DIM, size=min(4, FLOW_DIM), replace=False)
    delta = rng.normal(0.0, 0.025, size=len(cells)).astype(np.float32)
    flow[cells] = np.clip(flow[cells] + delta, -1.0, 1.0)
    perturbed_row["flow"] = flow.tolist()
    changed = run_trace(system, perturbed_row, library, contracts, params=params, dense=dense)[-1]
    denom = float(np.mean(np.abs(delta))) + 1e-8
    route_factor = max(1, len(route))
    return float(np.mean(np.abs(changed - base)) / denom / route_factor)


def evaluate_interventions(
    seed: int,
    system: str,
    task: dict[str, list[dict[str, Any]]],
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    params: dict[str, dict[str, Any]] | None = None,
    dense: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if system in ORACLE_SYSTEMS:
        return rows
    for intervention in INTERVENTIONS:
        for split in EVAL_SPLITS:
            answer_ok: list[float] = []
            frame_mae: list[float] = []
            read_mae: list[float] = []
            max_step_drift: dict[int, list[float]] = {}
            for row in task[split]:
                route = tuple(row["expected_route"])
                if intervention == "one_learned_pocket_at_a_time":
                    per_row = []
                    for step_idx in range(len(route)):
                        oracle = oracle_frames(row)
                        frames = run_trace(system, row, library, contracts, params=params, dense=dense, intervention=intervention, one_learned_index=step_idx)
                        _, summary = frame_rows_for_trace(seed, system, split, row, frames, oracle, contracts)
                        per_row.append(float(summary["mean_frame_mae"]))
                        max_step_drift.setdefault(step_idx, []).append(float(summary["mean_frame_mae"]))
                    frame_mae.append(max(per_row) if per_row else 0.0)
                    continue
                oracle = oracle_frames(row)
                frames = run_trace(system, row, library, contracts, params=params, dense=dense, intervention=intervention)
                _, summary = frame_rows_for_trace(seed, system, split, row, frames, oracle, contracts)
                answer_ok.append(float(summary["answer_correct"]))
                frame_mae.append(float(summary["mean_frame_mae"]))
                read_mae.append(float(summary["mean_consumer_read_mask_mae"]))
            drift_by_step = {str(k): round_float(float(np.mean(v))) for k, v in sorted(max_step_drift.items())}
            rows.append(
                {
                    "seed": seed,
                    "system": system,
                    "split": split,
                    "intervention": intervention,
                    "answer_accuracy": round_float(float(np.mean(answer_ok)) if answer_ok else 0.0),
                    "mean_frame_mae": round_float(float(np.mean(frame_mae)) if frame_mae else 0.0),
                    "mean_consumer_read_mask_mae": round_float(float(np.mean(read_mae)) if read_mae else 0.0),
                    "usefulness_proxy": round_float((float(np.mean(answer_ok)) if answer_ok else 0.0) - 0.05 * (float(np.mean(frame_mae)) if frame_mae else 0.0)),
                    "one_learned_step_drift_by_step": drift_by_step,
                }
            )
    return rows


def aggregate_results(rows: list[dict[str, Any]], interventions: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        subset = sorted((row for row in rows if row["system"] == system), key=lambda row: int(row["seed"]))
        if not subset:
            continue
        numeric = sorted({key for row in subset for key, value in row.items() if isinstance(value, (int, float)) and key != "seed"})
        mean = {key: round_float(math.fsum(float(row.get(key, 0.0)) for row in subset) / len(subset)) for key in numeric}
        step_subset = [row for row in trace_rows if row["system"] == system and row["split"] in EVAL_SPLITS]
        if step_subset:
            by_step: dict[int, list[float]] = {}
            by_skill: dict[str, list[float]] = {}
            for row in step_subset:
                by_step.setdefault(int(row["route_step"]), []).append(float(row["frame_mae_to_oracle"]))
                by_skill.setdefault(str(row["skill"]), []).append(float(row["frame_mae_to_oracle"]))
            worst_step = max(by_step, key=lambda idx: float(np.mean(by_step[idx])))
            worst_skill = max(by_skill, key=lambda skill: float(np.mean(by_skill[skill])))
            mean["worst_route_step"] = int(worst_step)
            mean["worst_route_step_frame_mae"] = round_float(float(np.mean(by_step[worst_step])))
            systems[system] = {
                "seed_count": len({row["seed"] for row in subset}),
                "mean": mean,
                "worst_skill_by_trace_drift": worst_skill,
                "worst_skill_frame_mae": round_float(float(np.mean(by_skill[worst_skill]))),
            }
        else:
            systems[system] = {"seed_count": len({row["seed"] for row in subset}), "mean": mean}
    clean_candidates = [system for system in systems if system not in ORACLE_SYSTEMS and system != "dense_graph_danger_trace"]
    best = max(clean_candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", -1e9))
    intervention_summary: dict[str, Any] = {}
    for system in clean_candidates:
        subset = [row for row in interventions if row["system"] == system and row["split"] in EVAL_SPLITS]
        by_intervention: dict[str, list[float]] = {}
        for row in subset:
            by_intervention.setdefault(row["intervention"], []).append(float(row["usefulness_proxy"]))
        intervention_summary[system] = {key: round_float(float(np.mean(values))) for key, values in sorted(by_intervention.items())}
    return {
        "schema_version": "e8e_aggregate_metrics_v1",
        "systems": systems,
        "best_system": best,
        "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0),
        "intervention_summary": intervention_summary,
    }


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    learned = systems.get("current_best_learned_trace", {}).get("mean", {})
    dense = systems.get("dense_graph_danger_trace", {}).get("mean", {})
    best = aggregate["best_system"]
    baseline_useful = learned.get("eval_mean_composition_usefulness", 0.0)
    dense_useful = dense.get("eval_mean_composition_usefulness", 0.0)
    dense_trace = dense.get("eval_mean_trace_similarity", 0.0)
    first = learned.get("eval_mean_first_divergence_step", 0.0)
    slope_v = learned.get("eval_mean_drift_slope", 0.0)
    frame = learned.get("eval_mean_flow_frame_mae_to_oracle", 0.0)
    read = learned.get("eval_mean_consumer_read_mask_mae", 0.0)
    wrong = learned.get("eval_mean_wrong_attractor_rate", 0.0)
    interventions = aggregate.get("intervention_summary", {}).get("current_best_learned_trace", {})
    reset_gain = interventions.get("oracle_reset_after_step_1", 0.0) - baseline_useful
    reset_each_gain = interventions.get("oracle_reset_after_each_step", 0.0) - baseline_useful
    if dense_useful >= baseline_useful + 0.03 and dense_trace < learned.get("eval_mean_trace_similarity", 0.0) - 0.08:
        decision = "e8e_answer_shortcut_trace_invalid"
    elif first > 0.0 and first <= 1.25 and frame > DIVERGENCE_THRESHOLD:
        decision = "e8e_first_step_write_divergence"
    elif read >= frame * 0.78 and read > 0.10:
        decision = "e8e_consumer_sensitive_state_mismatch"
    elif max(reset_gain, reset_each_gain) >= 0.05:
        decision = "e8e_recoverable_state_drift"
    elif slope_v >= 0.015:
        decision = "e8e_temporal_drift_accumulation"
    elif wrong >= 0.08:
        decision = "e8e_wrong_attractor_trace"
    else:
        decision = "e8e_temporal_drift_accumulation"
    return {
        "schema_version": "e8e_decision_v1",
        "decision": decision,
        "detail": {
            "best_system": best,
            "current_best_learned_trace_usefulness": baseline_useful,
            "current_best_first_divergence_step": first,
            "current_best_frame_mae": frame,
            "current_best_consumer_read_mask_mae": read,
            "current_best_drift_slope": slope_v,
            "current_best_wrong_attractor_rate": wrong,
            "oracle_reset_after_step_1_gain": round_float(reset_gain),
            "oracle_reset_after_each_step_gain": round_float(reset_each_gain),
            "dense_graph_usefulness": dense_useful,
            "dense_graph_trace_similarity": dense_trace,
        },
        "deterministic_replay_passed": False,
    }


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    os.environ["E8E_TRACE_SAMPLE_LIMIT"] = str(settings.trace_sample_limit_per_split)
    e7o_settings = to_e7o_settings(settings)
    e7z_settings = to_e7z_settings(settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    baseline_library: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        baseline_library[skill] = e7p.copy_state(trained["state"], "E8E_baseline_standalone")

    base_contexts = e8c.generate_context_tasks(composition_task, "int4", "current_full_code_teacher")
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    attractor_rows: list[dict[str, Any]] = []
    editability_rows: list[dict[str, Any]] = []
    mutation_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []

    baseline_lib, baseline_contracts, dyn, _, _ = e8d.train_library_on_contexts(seed, "current_best_learned_trace", "baseline", baseline_library, base_contexts, settings, out)
    dynamics_rows.extend(dyn)

    substrate, _, _ = e8d.fit_substrate(seed, "substrate_first_trace", "autoencoder", base_contexts)
    substrate_context = e8d.substrate_contexts(composition_task, substrate, "autoencoder")
    substrate_lib, substrate_contracts, dyn, _, _ = e8d.train_library_on_contexts(seed, "substrate_first_trace", "baseline", baseline_library, substrate_context, settings, out)
    dynamics_rows.extend(dyn)

    mutation_contracts = {skill: e7z.build_lowbit_contract(skill, "mutation_only_trace", read_count=OUTPUT_WIDTH) for skill in SKILLS}
    mutation_params, mutation = e7z.repair_boundary(seed, "mutation_only_trace", "int4", baseline_library, mutation_contracts, composition_task, e7z_settings, out)
    mutation_rows.extend(mutation["history"])

    configs: dict[str, dict[str, Any]] = {
        "oracle_trace_reference": {"library": None, "contracts": None, "params": None, "dense": False},
        "consumer_distill_trace_reference": {"library": None, "contracts": None, "params": None, "dense": False},
        "current_best_learned_trace": {"library": baseline_lib, "contracts": baseline_contracts, "params": None, "dense": False},
        "substrate_first_trace": {"library": substrate_lib, "contracts": substrate_contracts, "params": None, "dense": False},
        "mutation_only_trace": {"library": baseline_library, "contracts": mutation_contracts, "params": mutation_params, "dense": False},
        "dense_graph_danger_trace": {"library": baseline_library, "contracts": None, "params": None, "dense": True},
    }
    for system in SYSTEMS:
        cfg = configs[system]
        result, trace, samples, attractors, edits = evaluate_trace_system(seed, system, composition_task, cfg["library"], cfg["contracts"], cfg["params"], cfg["dense"])
        if system == "mutation_only_trace":
            result.update({key: value for key, value in mutation.items() if key != "history"})
        rows.append(result)
        trace_rows.extend(trace)
        sample_rows.extend(samples)
        attractor_rows.extend(attractors)
        editability_rows.extend(edits)
        intervention_rows.extend(evaluate_interventions(seed, system, composition_task, cfg["library"], cfg["contracts"], cfg["params"], cfg["dense"]))
        if out:
            append_progress(out, "e8e_trace_system_evaluated", seed=seed, system=system, usefulness=result["eval_mean_composition_usefulness"], trace_similarity=result["eval_mean_trace_similarity"])

    return {
        "seed": seed,
        "rows": rows,
        "trace_rows": trace_rows,
        "intervention_rows": intervention_rows,
        "attractor_rows": attractor_rows,
        "editability_rows": editability_rows,
        "mutation_rows": mutation_rows,
        "sample_rows": sample_rows,
        "dynamics_rows": dynamics_rows,
    }


def build_report_text(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# E8E Temporal Thought-Frame Divergence Audit Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_system = {aggregate['best_system']}",
        f"deterministic_replay_passed = {decision.get('deterministic_replay_passed', False)}",
        "```",
        "",
        "## Mean Trace Scores",
        "",
        "```text",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"].get(system, {}).get("mean", {})
        lines.append(
            f"{system:<34} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"trace={mean.get('eval_mean_trace_similarity', 0.0):.6f} "
            f"frame_mae={mean.get('eval_mean_flow_frame_mae_to_oracle', 0.0):.6f} "
            f"read_mae={mean.get('eval_mean_consumer_read_mask_mae', 0.0):.6f} "
            f"first_div={mean.get('eval_mean_first_divergence_step', 0.0):.3f} "
            f"slope={mean.get('eval_mean_drift_slope', 0.0):.6f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Interpretation Guard",
            "",
            "E8E is a diagnostic symbolic/numeric Flow/RAM trace audit. Oracle writes are used only for reference and explicit intervention arms. No architecture, router, semantic lane labels, raw-language task, AGI, consciousness, deployed-model, or model-scale claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_reports(
    rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    attractor_rows: list[dict[str, Any]],
    editability_rows: list[dict[str, Any]],
    mutation_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    dynamics_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trace_divergence_report.json": {"schema_version": "e8e_trace_divergence_report_v1", "rows": sorted(trace_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"], r["route_step"]))},
        "intervention_report.json": {"schema_version": "e8e_intervention_report_v1", "rows": sorted(intervention_rows, key=lambda r: (r["seed"], r["system"], r["intervention"], r["split"]))},
        "attractor_report.json": {"schema_version": "e8e_attractor_report_v1", "rows": sorted(attractor_rows, key=lambda r: (r["seed"], r["system"], r["split"]))},
        "local_editability_report.json": {"schema_version": "e8e_local_editability_report_v1", "rows": sorted(editability_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"]))},
        "mutation_history_report.json": {"schema_version": "e8e_mutation_history_report_v1", "rows": sorted(mutation_rows, key=lambda r: (r["seed"], r["system"], r["generation"]))},
        "producer_dynamics_report.json": {"schema_version": "e8e_producer_dynamics_report_v1", "rows": sorted(dynamics_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", "")))},
        "system_results.json": {"schema_version": "e8e_system_results_v1", "rows": sorted(rows, key=lambda r: (r["seed"], SYSTEMS.index(r["system"])))},
        "row_level_samples.json": {"schema_version": "e8e_row_level_samples_v1", "rows": sorted(sample_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"]))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e8e_summary_v1",
            "decision": decision["decision"],
            "best_system": aggregate["best_system"],
            "deterministic_replay_passed": deterministic.get("internal_replay_passed", False),
            "checker_failure_count": None,
            "first_divergence_step": decision["detail"].get("current_best_first_divergence_step"),
            "recommended_next_fix": recommended_fix(decision["decision"]),
        },
        "deterministic_replay.json": deterministic,
        "report.md": build_report_text(decision, aggregate),
    }


def recommended_fix(decision: str) -> str:
    mapping = {
        "e8e_first_step_write_divergence": "fix producer commit/write operator before adding new substrate machinery",
        "e8e_temporal_drift_accumulation": "test temporal stabilizer or integrator on existing Flow transition",
        "e8e_consumer_sensitive_state_mismatch": "target consumer-read cells and IO contract, not global RAM validity",
        "e8e_recoverable_state_drift": "test guarded reset/renormalization at router return boundary",
        "e8e_wrong_attractor_trace": "audit state basin/collapse and transition validity",
        "e8e_answer_shortcut_trace_invalid": "reject answer-only improvement and enforce trace validity gates",
    }
    return mapping.get(decision, "run a narrower trace-local fix probe")


def hash_artifacts(out: Path) -> dict[str, str]:
    return {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in HASH_ARTIFACTS}


def replay_command(settings: Settings, replay_out: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--out",
        str(replay_out.relative_to(REPO_ROOT)),
        "--seeds",
        ",".join(str(seed) for seed in settings.seeds),
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
        "--gradient-diagnostic-batch-size",
        str(settings.gradient_diagnostic_batch_size),
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
        "--similarity-threshold",
        str(settings.similarity_threshold),
        "--trace-sample-limit-per-split",
        str(settings.trace_sample_limit_per_split),
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
    os.environ["E8E_TRACE_SAMPLE_LIMIT"] = str(settings.trace_sample_limit_per_split)
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
                "schema_version": "e8e_backend_manifest_v1",
                "milestone": MILESTONE,
                "settings": settings_payload(settings),
                "systems": list(SYSTEMS),
                "interventions": list(INTERVENTIONS),
                "flow_dim": FLOW_DIM,
                "output_width": OUTPUT_WIDTH,
                "diagnostic_only": True,
                "new_architecture": False,
                "new_router": False,
                "semantic_lane_labels_as_model_input": False,
                "oracle_write_at_inference_for_learned_systems": False,
                "oracle_writes_allowed_only_in_reference_and_intervention_arms": True,
                "row_level_eval": True,
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
                "schema_version": "e8e_task_generation_report_v1",
                "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()},
                "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()},
            },
        )
        rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []
        intervention_rows: list[dict[str, Any]] = []
        attractor_rows: list[dict[str, Any]] = []
        editability_rows: list[dict[str, Any]] = []
        mutation_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        dynamics_rows: list[dict[str, Any]] = []
        jobs = [{"seed": seed, "settings": settings.__dict__.copy(), "composition_task": composition_tasks[seed], "pocket_task": pocket_tasks[seed], "out": str(out)} for seed in settings.seeds]
        max_workers = max(1, min(settings.cpu_workers, len(jobs)))
        if settings.device == "cuda" and max_workers > 1:
            append_progress(out, "cuda_process_pool_disabled", requested_workers=max_workers, active_workers=1)
            max_workers = 1
        if max_workers == 1:
            for job in jobs:
                result = seed_worker(job)
                rows.extend(result["rows"])
                trace_rows.extend(result["trace_rows"])
                intervention_rows.extend(result["intervention_rows"])
                attractor_rows.extend(result["attractor_rows"])
                editability_rows.extend(result["editability_rows"])
                mutation_rows.extend(result["mutation_rows"])
                sample_rows.extend(result["sample_rows"])
                dynamics_rows.extend(result["dynamics_rows"])
                done_seeds = len({row["seed"] for row in rows})
                append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(jobs) - done_seeds)
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_trace_rows": len(trace_rows), "completed_intervention_rows": len(intervention_rows), "completed_dynamics_rows": len(dynamics_rows), "pending": len(jobs) - done_seeds})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(seed_worker, job): job for job in jobs}
                while futures:
                    done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_trace_rows": len(trace_rows), "completed_intervention_rows": len(intervention_rows), "completed_dynamics_rows": len(dynamics_rows), "pending": len(futures)})
                        continue
                    for future in done:
                        job = futures.pop(future)
                        result = future.result()
                        rows.extend(result["rows"])
                        trace_rows.extend(result["trace_rows"])
                        intervention_rows.extend(result["intervention_rows"])
                        attractor_rows.extend(result["attractor_rows"])
                        editability_rows.extend(result["editability_rows"])
                        mutation_rows.extend(result["mutation_rows"])
                        sample_rows.extend(result["sample_rows"])
                        dynamics_rows.extend(result["dynamics_rows"])
                        append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_trace_rows": len(trace_rows), "completed_intervention_rows": len(intervention_rows), "completed_dynamics_rows": len(dynamics_rows), "last_completed": f"seed{job['seed']}", "pending": len(futures)})
        aggregate = aggregate_results(rows, intervention_rows, trace_rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e8e_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, trace_rows, intervention_rows, attractor_rows, editability_rows, mutation_rows, sample_rows, dynamics_rows, aggregate, decision, deterministic_placeholder)
        for name, payload in reports.items():
            if name.endswith(".json"):
                write_json(out / name, payload)
            else:
                write_text(out / name, str(payload))
        append_progress(out, "primary_artifacts_written", artifact_count=len(reports))
        if not settings.replay:
            primary_hashes = hash_artifacts(out)
            replay_out = out / "deterministic_replay_work"
            if replay_out.exists():
                shutil.rmtree(replay_out)
            append_progress(out, "deterministic_replay_start", replay_out=str(replay_out.relative_to(REPO_ROOT)))
            subprocess.run(replay_command(settings, replay_out), cwd=str(REPO_ROOT), check=True)
            replay_hashes = hash_artifacts(replay_out)
            comparisons = {name: {"primary": primary_hashes[name], "replay": replay_hashes.get(name), "match": primary_hashes[name] == replay_hashes.get(name)} for name in HASH_ARTIFACTS}
            deterministic = {"schema_version": "e8e_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, trace_rows, intervention_rows, attractor_rows, editability_rows, mutation_rows, sample_rows, dynamics_rows, aggregate, decision, deterministic)
            for name, payload in reports.items():
                if name.endswith(".json"):
                    write_json(out / name, payload)
                else:
                    write_text(out / name, str(payload))
            append_progress(out, "deterministic_replay_complete", internal_replay_passed=deterministic["internal_replay_passed"])
            append_progress(out, "final_artifacts_written", artifact_count=len(reports))
        else:
            write_json(out / "deterministic_replay.json", deterministic_placeholder)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return decision
    finally:
        heartbeat.set()
        thread.join(timeout=2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=220)
    parser.add_argument("--validation-rows-per-seed", type=int, default=128)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=128)
    parser.add_argument("--ood-rows-per-seed", type=int, default=128)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=128)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=128)
    parser.add_argument("--pocket-pretrain-rows-per-seed", type=int, default=300)
    parser.add_argument("--pocket-validation-rows-per-seed", type=int, default=128)
    parser.add_argument("--pocket-dim", type=int, default=56)
    parser.add_argument("--pocket-core-steps", type=int, default=2)
    parser.add_argument("--pocket-epochs", type=int, default=18)
    parser.add_argument("--local-epochs", type=int, default=14)
    parser.add_argument("--full-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-diagnostic-batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1.1e-3)
    parser.add_argument("--local-learning-rate", type=float, default=8.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pruned-read-count", type=int, default=OUTPUT_WIDTH)
    parser.add_argument("--repair-generations", type=int, default=4)
    parser.add_argument("--repair-population", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--trace-sample-limit-per-split", type=int, default=3)
    parser.add_argument("--cpu-workers", type=int, default=max(1, min(23, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
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
        gradient_diagnostic_batch_size=args.gradient_diagnostic_batch_size,
        learning_rate=args.learning_rate,
        local_learning_rate=args.local_learning_rate,
        weight_decay=args.weight_decay,
        pruned_read_count=args.pruned_read_count,
        repair_generations=args.repair_generations,
        repair_population=args.repair_population,
        similarity_threshold=args.similarity_threshold,
        trace_sample_limit_per_split=args.trace_sample_limit_per_split,
        cpu_workers=args.cpu_workers,
        device=select_device(args.device),
        heartbeat_seconds=args.heartbeat_seconds,
        replay=bool(args.replay),
        execution_mode=args.execution_mode,
    )
    run(settings, resolve_out(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
