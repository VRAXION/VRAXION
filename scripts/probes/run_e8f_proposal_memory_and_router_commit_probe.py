#!/usr/bin/env python3
"""E8F proposal memory and router/commit probe.

E8E localized the current numeric pocket/RAM failure to repeated route-return
drift. E8F tests the next hypothesis without changing task family or adding
semantic lanes: pocket outputs are proposals, proposal memory stores them, and
a router/commit controller decides what becomes the next stable Flow state.
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

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


REPO_ROOT = Path(__file__).resolve().parents[2]
E8E_PATH = Path(__file__).with_name("run_e8e_temporal_thought_frame_divergence_audit.py")
MILESTONE = "E8F_PROPOSAL_MEMORY_AND_ROUTER_COMMIT_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e8f_proposal_memory_and_router_commit_probe")
DEFAULT_SEEDS = (106001, 106002, 106003, 106004, 106005, 106006, 106007, 106008)
OUTPUT_WIDTH = 12
ARTIFACT_FLOAT_DECIMALS = 2

SYSTEMS = (
    "direct_overwrite_baseline",
    "output_feedback_only",
    "proposal_memory_no_commit",
    "proposal_memory_plus_simple_commit",
    "proposal_memory_plus_router_commit_gate",
    "proposal_memory_plus_learned_commit",
    "proposal_memory_plus_per_skill_commit",
    "proposal_memory_ring_buffer",
    "proposal_memory_plus_verifier_pocket",
    "proposal_memory_plus_stepwise_renormalization",
    "oracle_stepwise_commit_reference",
    "dense_graph_danger_control",
)
ORACLE_SYSTEMS = {"oracle_stepwise_commit_reference"}
PRIMARY_CLEAN_SYSTEMS = tuple(system for system in SYSTEMS if system not in ORACLE_SYSTEMS | {"dense_graph_danger_control"})
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "proposal_memory_report.json",
    "commit_controller_report.json",
    "temporal_trace_report.json",
    "dense_graph_control_report.json",
    "system_results.json",
    "row_level_samples.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e8f_proposal_memory_commit_positive",
    "e8f_output_feedback_sufficient",
    "e8f_commit_controller_required",
    "e8f_shared_commit_controller_positive",
    "e8f_per_skill_commit_required",
    "e8f_proposal_trace_memory_positive",
    "e8f_verifier_commit_required",
    "e8f_stepwise_renormalization_positive",
    "e8f_proposal_memory_not_sufficient",
    "e8f_answer_shortcut_trace_invalid",
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e8e = load_module(E8E_PATH, "e8e_temporal_trace_audit")
e8d = e8e.e8d
e8c = e8e.e8c
e8a = e8e.e8a
e7z = e8e.e7z
e7y = e8e.e7y
e7r = e8e.e7r
e7p = e8e.e7p
e7o = e8e.e7o

FLOW_DIM = int(e8e.FLOW_DIM)
SKILLS = tuple(e8e.SKILLS)
SPLITS = tuple(e8e.SPLITS)
EVAL_SPLITS = tuple(e8e.EVAL_SPLITS)
RESULT_POS = dict(e8e.RESULT_POS)


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
    ridge_lambda: float
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    replay: bool
    execution_mode: str


def round_float(value: float) -> float:
    return round(float(value), ARTIFACT_FLOAT_DECIMALS)


def stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stable_payload(item) for item in value]
    if isinstance(value, tuple):
        return [stable_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return stable_payload(value.tolist())
    if isinstance(value, (float, np.floating)):
        return round(float(value), ARTIFACT_FLOAT_DECIMALS)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    e8e.write_json(path, stable_payload(payload))


def write_text(path: Path, text: str) -> None:
    e8e.write_text(path, text)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e8e.append_progress(out, event, **details)


def resolve_out(path: str | Path) -> Path:
    return e8e.resolve_out(path)


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return e8e.parse_int_tuple(raw)


def select_device(requested: str) -> str:
    return e8e.select_device(requested)


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    payload["replay"] = False
    return payload


def to_e8e_settings(settings: Settings) -> Any:
    return e8e.Settings(
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
        trace_sample_limit_per_split=settings.trace_sample_limit_per_split,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        replay=settings.replay,
        execution_mode=settings.execution_mode,
    )


def to_e7o_settings(settings: Settings) -> Any:
    return e8e.to_e7o_settings(to_e8e_settings(settings))


def direct_proposal(
    skill: str,
    stable_flow: np.ndarray,
    library: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]] | None,
    dense: bool = False,
) -> np.ndarray:
    if dense:
        return e7p.np_forward(library[skill], stable_flow.reshape(1, -1)).reshape(-1).astype(np.float32)
    assert contracts is not None
    pred = e7r.masked_forward_np(library[skill], stable_flow.reshape(1, -1), contracts[skill]).reshape(-1).astype(np.float32)
    cells = np.asarray(e7y.bundle_cells(skill, OUTPUT_WIDTH), dtype=np.int64)
    pred[cells] = e7z.quantize_bundle_values(pred[cells], skill, "int4", None, primary_is_logit=True)
    return pred.astype(np.float32)


def clamp_flow(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float32), -1.0, 1.0).astype(np.float32)


def renorm_flow(stable: np.ndarray, proposal: np.ndarray, skill: str, alpha: float) -> np.ndarray:
    out = clamp_flow(stable + alpha * (proposal - stable))
    cells = np.asarray(e7y.bundle_cells(skill, OUTPUT_WIDTH), dtype=np.int64)
    out[cells] = e7z.quantize_bundle_values(out[cells], skill, "int4", None, primary_is_logit=True)
    delta = out - stable
    max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_delta > 0.75:
        out = clamp_flow(stable + delta * (0.75 / max_delta))
        out[cells] = e7z.quantize_bundle_values(out[cells], skill, "int4", None, primary_is_logit=True)
    return out


def feature_row(stable: np.ndarray, proposal: np.ndarray, memory: np.ndarray) -> np.ndarray:
    return np.concatenate([stable, proposal, proposal - stable, memory], axis=0).astype(np.float32)


def fit_ridge(xs: np.ndarray, ys: np.ndarray, ridge_lambda: float) -> dict[str, Any]:
    if len(xs) == 0:
        return {"weights": np.zeros((xs.shape[1] + 1, FLOW_DIM), dtype=np.float32), "train_mae": 1.0, "condition": 0.0}
    x_aug = np.concatenate([xs.astype(np.float32), np.ones((len(xs), 1), dtype=np.float32)], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=np.float32) * np.float32(ridge_lambda)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve((xtx + reg).astype(np.float64), (x_aug.T @ ys).astype(np.float64)).astype(np.float32)
    pred = x_aug @ weights
    return {
        "weights": weights,
        "train_mae": round_float(float(np.mean(np.abs(pred - ys)))),
        "condition": round(float(np.linalg.cond((xtx + reg).astype(np.float64))), 6),
    }


def apply_ridge(model: dict[str, Any], stable: np.ndarray, proposal: np.ndarray, memory: np.ndarray) -> np.ndarray:
    x = feature_row(stable, proposal, memory)
    weights = np.asarray(model["weights"], dtype=np.float32)
    x_aug = np.concatenate([x, np.ones(1, dtype=np.float32)], axis=0)
    return clamp_flow(x_aug @ weights)


def fit_alpha(train_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    num = 0.0
    den = 0.0
    for stable, proposal, target in train_rows:
        delta = proposal - stable
        wanted = target - stable
        num += float(np.sum(delta * wanted))
        den += float(np.sum(delta * delta))
    if den <= 1e-12:
        return 0.5
    return round_float(float(np.clip(num / den, 0.05, 1.20)))


def build_commit_models(
    seed: int,
    library: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    task: dict[str, list[dict[str, Any]]],
    settings: Settings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    alpha_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    shared_x: list[np.ndarray] = []
    shared_y: list[np.ndarray] = []
    per_skill_x: dict[str, list[np.ndarray]] = {skill: [] for skill in SKILLS}
    per_skill_y: dict[str, list[np.ndarray]] = {skill: [] for skill in SKILLS}
    ring_x: dict[int, list[np.ndarray]] = {1: [], 2: [], 4: []}
    ring_y: dict[int, list[np.ndarray]] = {1: [], 2: [], 4: []}
    for row in task["train"]:
        oracle = e8e.oracle_frames(row)
        stable = oracle[0].copy()
        memory_history: list[np.ndarray] = []
        for step_idx, skill in enumerate(tuple(row["expected_route"])):
            proposal = direct_proposal(skill, stable, library, contracts)
            memory = proposal.copy()
            target = oracle[step_idx + 1]
            alpha_pairs.append((stable.copy(), proposal.copy(), target.copy()))
            shared_x.append(feature_row(stable, proposal, memory))
            shared_y.append(target)
            per_skill_x[skill].append(feature_row(stable, proposal, memory))
            per_skill_y[skill].append(target)
            memory_history.append(proposal.copy())
            for n in (1, 2, 4):
                tail = memory_history[-n:]
                memory_n = np.mean(np.asarray(tail, dtype=np.float32), axis=0)
                ring_x[n].append(feature_row(stable, proposal, memory_n))
                ring_y[n].append(target)
            stable = target.copy()
    shared_model = fit_ridge(np.asarray(shared_x, dtype=np.float32), np.asarray(shared_y, dtype=np.float32), settings.ridge_lambda)
    per_skill = {}
    for skill in SKILLS:
        per_skill[skill] = fit_ridge(np.asarray(per_skill_x[skill], dtype=np.float32), np.asarray(per_skill_y[skill], dtype=np.float32), settings.ridge_lambda)
    ring = {}
    for n in (1, 2, 4):
        ring[n] = fit_ridge(np.asarray(ring_x[n], dtype=np.float32), np.asarray(ring_y[n], dtype=np.float32), settings.ridge_lambda)
    alpha = fit_alpha(alpha_pairs)
    models = {
        "seed": seed,
        "schema_version": "e8f_commit_models_v1",
        "router_commit_alpha": alpha,
        "simple_commit_alpha": 0.5,
        "shared_commit": shared_model,
        "per_skill_commit": per_skill,
        "ring_commit": ring,
    }
    rows.append(
        {
            "seed": seed,
            "system": "proposal_memory_plus_router_commit_gate",
            "commit_type": "scalar_alpha",
            "alpha": alpha,
            "train_pairs": len(alpha_pairs),
            "semantic_labels_used": False,
            "oracle_used_at_inference": False,
        }
    )
    rows.append(
        {
            "seed": seed,
            "system": "proposal_memory_plus_learned_commit",
            "commit_type": "shared_ridge_commit",
            "train_mae": shared_model["train_mae"],
            "condition": shared_model["condition"],
            "semantic_labels_used": False,
            "oracle_used_at_inference": False,
        }
    )
    for skill in SKILLS:
        rows.append(
            {
                "seed": seed,
                "system": "proposal_memory_plus_per_skill_commit",
                "commit_type": "per_skill_ridge_commit",
                "skill": skill,
                "train_mae": per_skill[skill]["train_mae"],
                "condition": per_skill[skill]["condition"],
                "semantic_labels_used": False,
                "oracle_used_at_inference": False,
            }
        )
    for n in (1, 2, 4):
        rows.append(
            {
                "seed": seed,
                "system": "proposal_memory_ring_buffer",
                "commit_type": "ring_buffer_shared_ridge_commit",
                "ring_n": n,
                "train_mae": ring[n]["train_mae"],
                "condition": ring[n]["condition"],
                "semantic_labels_used": False,
                "oracle_used_at_inference": False,
            }
        )
    return models, rows


def commit_step(
    system: str,
    skill: str,
    stable: np.ndarray,
    proposal: np.ndarray,
    memory_history: list[np.ndarray],
    models: dict[str, Any],
    library: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, float]]:
    memory = proposal.copy()
    accepted = 1.0
    rejection = 0.0
    correction = 0.0
    if system == "direct_overwrite_baseline":
        committed = proposal
    elif system == "output_feedback_only":
        committed = clamp_flow(0.45 * stable + 0.55 * proposal)
    elif system == "proposal_memory_no_commit":
        committed = clamp_flow(0.85 * stable + 0.15 * memory)
        accepted = 0.15
    elif system == "proposal_memory_plus_simple_commit":
        committed = renorm_flow(stable, proposal, skill, float(models["simple_commit_alpha"]))
        accepted = float(models["simple_commit_alpha"])
    elif system == "proposal_memory_plus_router_commit_gate":
        alpha = float(models["router_commit_alpha"])
        committed = renorm_flow(stable, proposal, skill, alpha)
        accepted = alpha
    elif system == "proposal_memory_plus_learned_commit":
        committed = apply_ridge(models["shared_commit"], stable, proposal, memory)
    elif system == "proposal_memory_plus_per_skill_commit":
        committed = apply_ridge(models["per_skill_commit"][skill], stable, proposal, memory)
    elif system == "proposal_memory_ring_buffer":
        tail = (memory_history + [proposal])[-2:]
        ring_memory = np.mean(np.asarray(tail, dtype=np.float32), axis=0)
        committed = apply_ridge(models["ring_commit"][2], stable, proposal, ring_memory)
    elif system == "proposal_memory_plus_verifier_pocket":
        rough = renorm_flow(stable, proposal, skill, float(models["router_commit_alpha"]))
        verifier = direct_proposal("verify", rough, library, contracts)
        gate = float(np.clip(0.25 + 0.50 * np.mean(np.abs(verifier - rough)), 0.20, 0.80))
        committed = renorm_flow(stable, proposal, skill, gate)
        accepted = gate
    elif system == "proposal_memory_plus_stepwise_renormalization":
        committed = renorm_flow(stable, proposal, skill, float(models["router_commit_alpha"]))
        accepted = float(models["router_commit_alpha"])
    else:
        raise ValueError(system)
    committed = clamp_flow(committed)
    proposal_delta = proposal - stable
    commit_delta = committed - stable
    proposal_norm = float(np.mean(np.abs(proposal_delta)))
    commit_norm = float(np.mean(np.abs(commit_delta)))
    if proposal_norm > 1e-8:
        accepted = round_float(float(np.clip(commit_norm / proposal_norm, 0.0, 1.5)))
    rejection = round_float(max(0.0, 1.0 - min(1.0, accepted)))
    correction = round_float(float(np.mean(np.abs(committed - proposal))))
    return committed.astype(np.float32), {"accepted": accepted, "rejected": rejection, "correction": correction, "proposal_norm": round_float(proposal_norm), "commit_norm": round_float(commit_norm)}


def run_variant_trace(
    system: str,
    row: dict[str, Any],
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    models: dict[str, Any] | None,
    dense: bool = False,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    oracle = e8e.oracle_frames(row)
    if system == "oracle_stepwise_commit_reference":
        return [frame.copy() for frame in oracle], []
    assert library is not None
    route = tuple(row["expected_route"])
    stable = oracle[0].copy()
    frames = [stable.copy()]
    memory_history: list[np.ndarray] = []
    proposal_rows: list[dict[str, Any]] = []
    for step_idx, skill in enumerate(route):
        proposal = direct_proposal(skill, stable, library, contracts, dense=dense)
        if dense:
            committed = proposal.copy()
            stats = {"accepted": 1.0, "rejected": 0.0, "correction": 0.0, "proposal_norm": round_float(float(np.mean(np.abs(proposal - stable)))), "commit_norm": round_float(float(np.mean(np.abs(proposal - stable))))}
        else:
            assert contracts is not None and models is not None
            committed, stats = commit_step(system, skill, stable, proposal, memory_history, models, library, contracts)
        memory_history.append(proposal.copy())
        proposal_rows.append(
            {
                "route_step": step_idx,
                "skill": skill,
                "proposal_memory_utilization": round_float(float(np.mean(np.abs(proposal)))),
                "proposal_delta_magnitude": stats["proposal_norm"],
                "commit_delta_magnitude": stats["commit_norm"],
                "proposal_acceptance_rate": stats["accepted"],
                "proposal_rejection_rate": stats["rejected"],
                "commit_correction_magnitude": stats["correction"],
            }
        )
        stable = committed.copy()
        frames.append(stable.copy())
    return frames, proposal_rows


def trace_metrics(
    seed: int,
    system: str,
    split: str,
    row: dict[str, Any],
    frames: list[np.ndarray],
    contracts: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oracle = e8e.oracle_frames(row)
    return e8e.frame_rows_for_trace(seed, system, split, row, frames, oracle, contracts)


def evaluate_system(
    seed: int,
    system: str,
    task: dict[str, list[dict[str, Any]]],
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    models: dict[str, Any] | None,
    settings: Settings,
    dense: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    split_results: dict[str, Any] = {}
    temporal_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        answer_ok: list[float] = []
        usefulness_values: list[float] = []
        frame_mae_values: list[float] = []
        delta_mae_values: list[float] = []
        read_mae_values: list[float] = []
        first_div_values: list[float] = []
        drift_values: list[float] = []
        accept_values: list[float] = []
        reject_values: list[float] = []
        correction_values: list[float] = []
        overcommit_values: list[float] = []
        undercommit_values: list[float] = []
        for row_idx, row in enumerate(task[split]):
            frames, props = run_variant_trace(system, row, library, contracts, models, dense=dense)
            trace_rows, summary = trace_metrics(seed, system, split, row, frames, contracts)
            temporal_rows.extend(trace_rows)
            for prop in props:
                prop_row = {"seed": seed, "system": system, "split": split, "row_id": row["row_id"], **prop}
                proposal_rows.append(prop_row)
                accept_values.append(float(prop["proposal_acceptance_rate"]))
                reject_values.append(float(prop["proposal_rejection_rate"]))
                correction_values.append(float(prop["commit_correction_magnitude"]))
                overcommit_values.append(float(prop["commit_delta_magnitude"] > prop["proposal_delta_magnitude"] * 1.05))
                undercommit_values.append(float(prop["commit_delta_magnitude"] < prop["proposal_delta_magnitude"] * 0.25))
            answer_ok.append(float(summary["answer_correct"]))
            frame_mae_values.append(float(summary["mean_frame_mae"]))
            delta_mae_values.append(float(summary["mean_delta_mae"]))
            read_mae_values.append(float(summary["mean_consumer_read_mask_mae"]))
            first_div_values.append(float(summary["first_divergence_step"]))
            drift_values.append(float(summary["drift_slope"]))
            usefulness = float(summary["answer_correct"]) - 0.08 * float(summary["mean_consumer_read_mask_mae"]) - 0.05 * float(summary["mean_frame_mae"]) - 0.03 * float(summary["mean_delta_mae"])
            usefulness_values.append(usefulness)
            if row_idx < settings.trace_sample_limit_per_split:
                sample_rows.append(e8e.sample_trace_row(seed, system, split, row, frames, e8e.oracle_frames(row), trace_rows, summary))
        split_results[split] = {
            "answer_accuracy": round_float(float(np.mean(answer_ok)) if answer_ok else 0.0),
            "composition_usefulness": round_float(float(np.mean(usefulness_values)) if usefulness_values else 0.0),
            "trace_validity": round_float(max(0.0, 1.0 - (float(np.mean(frame_mae_values)) if frame_mae_values else 1.0))),
            "frame_mae_to_oracle": round_float(float(np.mean(frame_mae_values)) if frame_mae_values else 0.0),
            "delta_mae_to_oracle": round_float(float(np.mean(delta_mae_values)) if delta_mae_values else 0.0),
            "read_mae_on_next_pocket_cells": round_float(float(np.mean(read_mae_values)) if read_mae_values else 0.0),
            "first_divergence_step": round_float(float(np.mean(first_div_values)) if first_div_values else 0.0),
            "drift_slope": round_float(float(np.mean(drift_values)) if drift_values else 0.0),
            "proposal_acceptance_rate": round_float(float(np.mean(accept_values)) if accept_values else 0.0),
            "proposal_rejection_rate": round_float(float(np.mean(reject_values)) if reject_values else 0.0),
            "commit_correction_magnitude": round_float(float(np.mean(correction_values)) if correction_values else 0.0),
            "overcommit_rate": round_float(float(np.mean(overcommit_values)) if overcommit_values else 0.0),
            "undercommit_rate": round_float(float(np.mean(undercommit_values)) if undercommit_values else 0.0),
            "proposal_memory_utilization": round_float(float(np.mean([row["proposal_memory_utilization"] for row in proposal_rows if row["split"] == split and row["system"] == system])) if proposal_rows else 0.0),
        }
    result = {
        "seed": seed,
        "system": system,
        "evals": split_results,
        "eval_mean_answer_accuracy": round_float(float(np.mean([split_results[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([split_results[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_trace_validity": round_float(float(np.mean([split_results[split]["trace_validity"] for split in EVAL_SPLITS]))),
        "eval_mean_frame_mae_to_oracle": round_float(float(np.mean([split_results[split]["frame_mae_to_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_delta_mae_to_oracle": round_float(float(np.mean([split_results[split]["delta_mae_to_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_read_mae_on_next_pocket_cells": round_float(float(np.mean([split_results[split]["read_mae_on_next_pocket_cells"] for split in EVAL_SPLITS]))),
        "eval_mean_drift_slope": round_float(float(np.mean([split_results[split]["drift_slope"] for split in EVAL_SPLITS]))),
        "eval_mean_first_divergence_step": round_float(float(np.mean([split_results[split]["first_divergence_step"] for split in EVAL_SPLITS]))),
        "eval_mean_proposal_acceptance_rate": round_float(float(np.mean([split_results[split]["proposal_acceptance_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_proposal_rejection_rate": round_float(float(np.mean([split_results[split]["proposal_rejection_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_commit_correction_magnitude": round_float(float(np.mean([split_results[split]["commit_correction_magnitude"] for split in EVAL_SPLITS]))),
        "eval_mean_overcommit_rate": round_float(float(np.mean([split_results[split]["overcommit_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_undercommit_rate": round_float(float(np.mean([split_results[split]["undercommit_rate"] for split in EVAL_SPLITS]))),
        "oracle_write_at_inference": bool(system in ORACLE_SYSTEMS),
    }
    return result, temporal_rows, proposal_rows, sample_rows


def aggregate_results(rows: list[dict[str, Any]], temporal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        subset = sorted((row for row in rows if row["system"] == system), key=lambda row: int(row["seed"]))
        if not subset:
            continue
        numeric_keys = sorted({key for row in subset for key, value in row.items() if isinstance(value, (int, float)) and key != "seed"})
        mean = {key: round_float(math.fsum(float(row.get(key, 0.0)) for row in subset) / len(subset)) for key in numeric_keys}
        steps = [row for row in temporal_rows if row["system"] == system and row["split"] in EVAL_SPLITS]
        if steps:
            by_step: dict[int, list[float]] = {}
            for row in steps:
                by_step.setdefault(int(row["route_step"]), []).append(float(row["frame_mae_to_oracle"]))
            worst_step = max(by_step, key=lambda idx: float(np.mean(by_step[idx])))
            mean["worst_route_step"] = int(worst_step)
            mean["worst_route_step_frame_mae"] = round_float(float(np.mean(by_step[worst_step])))
        systems[system] = {"seed_count": len({row["seed"] for row in subset}), "mean": mean}
    best = max(PRIMARY_CLEAN_SYSTEMS, key=lambda system: systems.get(system, {}).get("mean", {}).get("eval_mean_composition_usefulness", -1e9))
    return {
        "schema_version": "e8f_aggregate_metrics_v1",
        "systems": systems,
        "best_system": best,
        "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0),
    }


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    baseline = systems["direct_overwrite_baseline"]["mean"]
    dense = systems["dense_graph_danger_control"]["mean"]
    best = aggregate["best_system"]
    best_mean = systems[best]["mean"]
    base_use = baseline["eval_mean_composition_usefulness"]
    base_trace = baseline["eval_mean_trace_validity"]
    best_gain = best_mean["eval_mean_composition_usefulness"] - base_use
    best_trace_gain = best_mean["eval_mean_trace_validity"] - base_trace
    dense_answer_gap = dense["eval_mean_composition_usefulness"] - best_mean["eval_mean_composition_usefulness"]
    dense_trace_gap = best_mean["eval_mean_trace_validity"] - dense["eval_mean_trace_validity"]
    if dense_answer_gap >= 0.02 and dense_trace_gap >= 0.08:
        decision = "e8f_answer_shortcut_trace_invalid"
    elif best == "output_feedback_only" and best_gain >= 0.02:
        decision = "e8f_output_feedback_sufficient"
    elif best == "proposal_memory_plus_router_commit_gate" and best_gain >= 0.02 and best_trace_gain >= 0.02:
        decision = "e8f_commit_controller_required"
    elif best == "proposal_memory_plus_learned_commit" and best_gain >= 0.02 and best_trace_gain >= 0.02:
        decision = "e8f_shared_commit_controller_positive"
    elif best == "proposal_memory_plus_per_skill_commit" and best_gain >= 0.02 and best_trace_gain >= 0.02:
        decision = "e8f_per_skill_commit_required"
    elif best == "proposal_memory_ring_buffer" and best_gain >= 0.02 and best_trace_gain >= 0.02:
        decision = "e8f_proposal_trace_memory_positive"
    elif best == "proposal_memory_plus_verifier_pocket" and best_gain >= 0.02:
        decision = "e8f_verifier_commit_required"
    elif best == "proposal_memory_plus_stepwise_renormalization" and best_gain >= 0.02 and best_mean["eval_mean_drift_slope"] < baseline["eval_mean_drift_slope"] - 0.005:
        decision = "e8f_stepwise_renormalization_positive"
    elif best_gain >= 0.03 and best_trace_gain >= 0.02:
        decision = "e8f_proposal_memory_commit_positive"
    elif best_gain >= 0.02 and systems["proposal_memory_no_commit"]["mean"]["eval_mean_composition_usefulness"] < base_use + 0.005:
        decision = "e8f_commit_controller_required"
    else:
        decision = "e8f_proposal_memory_not_sufficient"
    return {
        "schema_version": "e8f_decision_v1",
        "decision": decision,
        "detail": {
            "best_system": best,
            "best_usefulness": best_mean["eval_mean_composition_usefulness"],
            "best_trace_validity": best_mean["eval_mean_trace_validity"],
            "direct_overwrite_usefulness": base_use,
            "direct_overwrite_trace_validity": base_trace,
            "best_gain_over_direct": round_float(best_gain),
            "best_trace_gain_over_direct": round_float(best_trace_gain),
            "dense_graph_usefulness": dense["eval_mean_composition_usefulness"],
            "dense_graph_trace_validity": dense["eval_mean_trace_validity"],
            "remaining_oracle_gap": round_float(systems["oracle_stepwise_commit_reference"]["mean"]["eval_mean_composition_usefulness"] - best_mean["eval_mean_composition_usefulness"]),
        },
        "deterministic_replay_passed": False,
    }


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    e7o_settings = to_e7o_settings(settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    baseline_library: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        baseline_library[skill] = e7p.copy_state(trained["state"], "E8F_baseline_standalone")
    contexts = e8c.generate_context_tasks(composition_task, "int4", "current_full_code_teacher")
    library, contracts, dynamics_rows, _, _ = e8d.train_library_on_contexts(seed, "direct_overwrite_baseline", "baseline", baseline_library, contexts, settings, out)
    models, commit_rows = build_commit_models(seed, library, contracts, composition_task, settings)
    rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        dense = system == "dense_graph_danger_control"
        lib = baseline_library if dense else library
        con = None if dense else contracts
        result, temporal, proposals, samples = evaluate_system(seed, system, composition_task, None if system in ORACLE_SYSTEMS else lib, None if system in ORACLE_SYSTEMS else con, models, settings, dense=dense)
        rows.append(result)
        temporal_rows.extend(temporal)
        proposal_rows.extend({"seed": seed, "system": system, **row} for row in proposals)
        sample_rows.extend(samples)
        if dense:
            dense_rows.extend(temporal)
        if out:
            append_progress(out, "e8f_system_evaluated", seed=seed, system=system, usefulness=result["eval_mean_composition_usefulness"], trace_validity=result["eval_mean_trace_validity"])
    return {
        "seed": seed,
        "rows": rows,
        "temporal_rows": temporal_rows,
        "proposal_rows": proposal_rows,
        "commit_rows": commit_rows,
        "dense_rows": dense_rows,
        "sample_rows": sample_rows,
        "dynamics_rows": dynamics_rows,
    }


def build_report_text(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# E8F Proposal Memory And Router Commit Probe Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_system = {aggregate['best_system']}",
        f"deterministic_replay_passed = {decision.get('deterministic_replay_passed', False)}",
        "```",
        "",
        "## Mean Scores",
        "",
        "```text",
    ]
    for system in SYSTEMS:
        mean = aggregate["systems"].get(system, {}).get("mean", {})
        lines.append(
            f"{system:<48} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"trace={mean.get('eval_mean_trace_validity', 0.0):.6f} "
            f"frame={mean.get('eval_mean_frame_mae_to_oracle', 0.0):.6f} "
            f"drift={mean.get('eval_mean_drift_slope', 0.0):.6f} "
            f"accept={mean.get('eval_mean_proposal_acceptance_rate', 0.0):.6f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Boundary",
            "",
            "E8F is a controlled symbolic/numeric Flow/RAM probe. Pocket outputs are tested as proposals, not automatic truth overwrites. No semantic lanes, raw-language task, image task, AGI, consciousness, deployed-model, or model-scale claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_reports(
    rows: list[dict[str, Any]],
    temporal_rows: list[dict[str, Any]],
    proposal_rows: list[dict[str, Any]],
    commit_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    dynamics_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    return {
        "proposal_memory_report.json": {"schema_version": "e8f_proposal_memory_report_v1", "rows": sorted(proposal_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"], r["route_step"]))},
        "commit_controller_report.json": {"schema_version": "e8f_commit_controller_report_v1", "rows": sorted(commit_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", ""), r.get("ring_n", 0)))},
        "temporal_trace_report.json": {"schema_version": "e8f_temporal_trace_report_v1", "rows": sorted(temporal_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"], r["route_step"]))},
        "dense_graph_control_report.json": {"schema_version": "e8f_dense_graph_control_report_v1", "rows": sorted(dense_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"], r["route_step"]))},
        "producer_dynamics_report.json": {"schema_version": "e8f_producer_dynamics_report_v1", "rows": sorted(dynamics_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", "")))},
        "system_results.json": {"schema_version": "e8f_system_results_v1", "rows": sorted(rows, key=lambda r: (r["seed"], SYSTEMS.index(r["system"])))},
        "row_level_samples.json": {"schema_version": "e8f_row_level_samples_v1", "rows": sorted(sample_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"]))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e8f_summary_v1",
            "decision": decision["decision"],
            "best_system": aggregate["best_system"],
            "deterministic_replay_passed": deterministic.get("internal_replay_passed", False),
            "checker_failure_count": None,
        },
        "deterministic_replay.json": deterministic,
        "report.md": build_report_text(decision, aggregate),
    }


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
        "--ridge-lambda",
        str(settings.ridge_lambda),
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
    e7r.append_jsonl(out / "hardware_heartbeat.jsonl", e7r.hardware_sample())
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
                "schema_version": "e8f_backend_manifest_v1",
                "milestone": MILESTONE,
                "settings": settings_payload(settings),
                "systems": list(SYSTEMS),
                "flow_dim": FLOW_DIM,
                "output_width": OUTPUT_WIDTH,
                "proposal_memory": True,
                "stable_flow_requires_commit": True,
                "new_router_architecture": False,
                "semantic_lane_labels_as_model_input": False,
                "oracle_write_at_inference_for_learned_systems": False,
                "dense_graph_primary_success_allowed": False,
                "row_level_eval": True,
                "device": select_device(settings.device),
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        )
        write_json(
            out / "task_generation_report.json",
            {
                "schema_version": "e8f_task_generation_report_v1",
                "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()},
                "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()},
            },
        )
        rows: list[dict[str, Any]] = []
        temporal_rows: list[dict[str, Any]] = []
        proposal_rows: list[dict[str, Any]] = []
        commit_rows: list[dict[str, Any]] = []
        dense_rows: list[dict[str, Any]] = []
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
                temporal_rows.extend(result["temporal_rows"])
                proposal_rows.extend(result["proposal_rows"])
                commit_rows.extend(result["commit_rows"])
                dense_rows.extend(result["dense_rows"])
                sample_rows.extend(result["sample_rows"])
                dynamics_rows.extend(result["dynamics_rows"])
                done = len({row["seed"] for row in rows})
                append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(jobs) - done)
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_temporal_rows": len(temporal_rows), "completed_proposal_rows": len(proposal_rows), "pending": len(jobs) - done})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(seed_worker, job): job for job in jobs}
                while futures:
                    done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_temporal_rows": len(temporal_rows), "completed_proposal_rows": len(proposal_rows), "pending": len(futures)})
                        continue
                    for future in done:
                        job = futures.pop(future)
                        result = future.result()
                        rows.extend(result["rows"])
                        temporal_rows.extend(result["temporal_rows"])
                        proposal_rows.extend(result["proposal_rows"])
                        commit_rows.extend(result["commit_rows"])
                        dense_rows.extend(result["dense_rows"])
                        sample_rows.extend(result["sample_rows"])
                        dynamics_rows.extend(result["dynamics_rows"])
                        append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_temporal_rows": len(temporal_rows), "completed_proposal_rows": len(proposal_rows), "last_completed": f"seed{job['seed']}", "pending": len(futures)})
        aggregate = aggregate_results(rows, temporal_rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e8f_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, temporal_rows, proposal_rows, commit_rows, dense_rows, sample_rows, dynamics_rows, aggregate, decision, deterministic_placeholder)
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
            deterministic = {"schema_version": "e8f_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, temporal_rows, proposal_rows, commit_rows, dense_rows, sample_rows, dynamics_rows, aggregate, decision, deterministic)
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
    parser.add_argument("--local-epochs", type=int, default=12)
    parser.add_argument("--full-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-diagnostic-batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1.1e-3)
    parser.add_argument("--local-learning-rate", type=float, default=8.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pruned-read-count", type=int, default=OUTPUT_WIDTH)
    parser.add_argument("--repair-generations", type=int, default=0)
    parser.add_argument("--repair-population", type=int, default=0)
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--trace-sample-limit-per-split", type=int, default=3)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
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
        ridge_lambda=args.ridge_lambda,
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
