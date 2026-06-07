#!/usr/bin/env python3
"""E8D substrate-first RAM language pretraining probe.

E8C showed that producer target decomposition reduced some gradient conflict
without closing the downstream consumer gap. E8D keeps the same symbolic numeric
Flow/RAM proxy and tests a stricter hypothesis: learn a shared RAM/substrate
state language first, freeze it, then train pockets as Flow_before -> Flow_after
operators over that language.
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
E8C_PATH = Path(__file__).with_name("run_e8c_producer_target_decomposition_and_consumer_compatibility_probe.py")
MILESTONE = "E8D_SUBSTRATE_FIRST_RAM_LANGUAGE_PRETRAINING_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e8d_substrate_first_ram_language_pretraining_probe")
DEFAULT_SEEDS = (104001, 104002, 104003, 104004, 104005, 104006, 104007, 104008)
OUTPUT_WIDTH = 12

SYSTEMS = (
    "no_substrate_baseline",
    "bridge_only_baseline",
    "substrate_autoencoder",
    "substrate_transition_model",
    "low_bit_substrate_codebook",
    "frozen_substrate_then_producer",
    "frozen_substrate_then_consumer",
    "frozen_substrate_then_pocket_composition",
    "jointly_mutable_substrate_and_pockets",
    "oracle_substrate_reference",
    "dense_graph_danger_control",
)
ORACLE_SYSTEMS = {"oracle_substrate_reference"}
MUTATION_SYSTEMS = {"jointly_mutable_substrate_and_pockets"}
TRAINED_SYSTEMS = tuple(system for system in SYSTEMS if system not in ORACLE_SYSTEMS | {"dense_graph_danger_control"})
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "substrate_pretraining_report.json",
    "ram_validity_report.json",
    "producer_dynamics_report.json",
    "consumer_read_report.json",
    "compatibility_report.json",
    "mutation_repair_report.json",
    "system_results.json",
    "row_level_samples.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e8d_substrate_first_positive",
    "e8d_bridge_adapter_sufficient",
    "e8d_pocket_to_substrate_write_bottleneck",
    "e8d_substrate_consumer_read_bottleneck",
    "e8d_frozen_substrate_too_rigid",
    "e8d_graph_soup_regression_detected",
    "e8d_substrate_language_not_helpful",
)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e8c = load_module(E8C_PATH, "e8c_producer_target_decomposition_probe")
e8a = e8c.e8a
e7z = e8c.e7z
e7y = e8c.e7y
e7r = e8c.e7r
e7p = e8c.e7p
e7o = e8c.e7o

FLOW_DIM = int(e8c.FLOW_DIM)
SKILLS = tuple(e8c.SKILLS)
SPLITS = tuple(e8c.SPLITS)
EVAL_SPLITS = tuple(e8c.EVAL_SPLITS)
RESULT_POS = dict(e8c.RESULT_POS)


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
    cpu_workers: int
    device: str
    heartbeat_seconds: float
    replay: bool
    execution_mode: str


def round_float(value: float) -> float:
    return e8c.round_float(value)


def write_json(path: Path, payload: Any) -> None:
    e8c.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    e8c.write_text(path, text)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e8c.append_progress(out, event, **details)


def resolve_out(path: str | Path) -> Path:
    return e8c.resolve_out(path)


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return e8c.parse_int_tuple(raw)


def select_device(requested: str) -> str:
    return e8c.select_device(requested)


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    payload["replay"] = False
    return payload


def to_e8c_settings(settings: Settings) -> Any:
    return e8c.Settings(
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


def to_e7o_settings(settings: Settings) -> Any:
    return e8c.to_e7o_settings(to_e8c_settings(settings))


def to_e7z_settings(settings: Settings) -> Any:
    return e8c.to_e7z_settings(to_e8c_settings(settings))


def quantile_levels(values: np.ndarray, fixed_low_bit: bool, primary: bool = False) -> list[float]:
    if primary:
        return [0.0, 1.0]
    if fixed_low_bit:
        return [-1.0, -0.5, 0.0, 0.5, 1.0]
    if values.size == 0:
        return [-1.0, -0.5, 0.0, 0.5, 1.0]
    qs = np.quantile(values.astype(np.float32), [0.02, 0.25, 0.50, 0.75, 0.98])
    qs = np.clip(qs, -1.0, 1.0)
    dedup = sorted({round(float(v), 6) for v in qs.tolist()})
    return dedup if len(dedup) >= 2 else [float(dedup[0]), float(dedup[0])]


def nearest(values: np.ndarray, levels: list[float]) -> np.ndarray:
    level_arr = np.asarray(levels, dtype=np.float32)
    idx = np.argmin(np.abs(values.reshape(-1, 1) - level_arr.reshape(1, -1)), axis=1)
    return level_arr[idx].astype(np.float32)


def fit_substrate(
    seed: int,
    system: str,
    mode: str,
    base_contexts: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(e7z.stable_seed(f"E8D-substrate:{seed}:{system}:{mode}"))
    substrate: dict[str, Any] = {"schema_version": "e8d_substrate_v1", "seed": seed, "system": system, "mode": mode, "skills": {}}
    pretrain_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    fixed_low_bit = mode in {"low_bit", "frozen", "joint"}
    for skill in SKILLS:
        cells = np.asarray(e7y.bundle_cells(skill, OUTPUT_WIDTH), dtype=np.int64)
        train_rows = base_contexts[skill]["train"]
        target = np.asarray([row["target_flow"] for row in train_rows], dtype=np.float32) if train_rows else np.zeros((0, FLOW_DIM), dtype=np.float32)
        before = np.asarray([row["flow"] for row in train_rows], dtype=np.float32) if train_rows else np.zeros((0, FLOW_DIM), dtype=np.float32)
        bundle = target[:, cells] if len(target) else np.zeros((0, len(cells)), dtype=np.float32)
        delta = target[:, cells] - before[:, cells] if len(target) else np.zeros((0, len(cells)), dtype=np.float32)
        levels_by_cell: list[list[float]] = []
        delta_levels_by_cell: list[list[float]] = []
        for idx in range(len(cells)):
            levels_by_cell.append(quantile_levels(bundle[:, idx], fixed_low_bit=fixed_low_bit, primary=(idx == 0)))
            delta_levels_by_cell.append(quantile_levels(delta[:, idx], fixed_low_bit=False, primary=False))
        if mode == "autoencoder":
            kind = "value_codebook"
        elif mode == "transition":
            kind = "delta_transition_codebook"
        elif mode in {"low_bit", "frozen", "joint"}:
            kind = "fixed_low_bit_codebook"
        else:
            kind = "none"
        substrate["skills"][skill] = {
            "cells": [int(v) for v in cells.tolist()],
            "kind": kind,
            "levels_by_cell": levels_by_cell,
            "delta_levels_by_cell": delta_levels_by_cell,
            "no_final_answer_objective": True,
            "uses_target_answer_label": False,
            "jitter": round_float(float(rng.normal(0.0, 0.0))),
        }
        for split in SPLITS:
            rows = base_contexts[skill][split]
            if not rows:
                continue
            xs = np.asarray([row["flow"] for row in rows], dtype=np.float32)
            ys = np.asarray([row["target_flow"] for row in rows], dtype=np.float32)
            recon = np.asarray([apply_substrate_to_target(xs[i], ys[i], skill, substrate, mode)[cells] for i in range(len(rows))], dtype=np.float32)
            wanted = ys[:, cells]
            mae = float(np.mean(np.abs(recon - wanted)))
            unique_count = len({tuple(round(float(v), 4) for v in row.tolist()) for row in recon[: min(128, len(recon))]})
            validity_rows.append(
                {
                    "seed": seed,
                    "system": system,
                    "mode": mode,
                    "skill": skill,
                    "split": split,
                    "ram_validity_score": round_float(max(0.0, 1.0 - mae)),
                    "reconstruction_mae": round_float(mae),
                    "code_entropy": e8c.bundle_entropy(recon),
                    "low_bit_code_utilization": round_float(unique_count / max(1, min(128, len(recon)))),
                    "code_collapse": bool(unique_count <= 1 and len(rows) > 1),
                }
            )
        pretrain_rows.append(
            {
                "seed": seed,
                "system": system,
                "mode": mode,
                "skill": skill,
                "substrate_hash": e8c.payload_sha256(substrate["skills"][skill]),
                "cells": [int(v) for v in cells.tolist()],
                "kind": kind,
                "no_final_answer_objective": True,
                "semantic_labels_used": False,
                "uses_target_answer_label": False,
            }
        )
    substrate["substrate_hash"] = e8c.payload_sha256(substrate)
    return substrate, pretrain_rows, validity_rows


def apply_substrate_to_target(
    flow_before: np.ndarray,
    target_after: np.ndarray,
    skill: str,
    substrate: dict[str, Any] | None,
    mode: str,
) -> np.ndarray:
    if substrate is None or mode == "none":
        return target_after.astype(np.float32).copy()
    spec = substrate["skills"][skill]
    cells = np.asarray(spec["cells"], dtype=np.int64)
    out = target_after.astype(np.float32).copy()
    source = target_after[cells].astype(np.float32)
    if mode == "transition":
        delta = target_after[cells].astype(np.float32) - flow_before[cells].astype(np.float32)
        decoded = np.asarray([nearest(delta[idx : idx + 1], spec["delta_levels_by_cell"][idx])[0] for idx in range(len(cells))], dtype=np.float32)
        values = flow_before[cells].astype(np.float32) + decoded
        values[0] = source[0]
    else:
        values = np.asarray([nearest(source[idx : idx + 1], spec["levels_by_cell"][idx])[0] for idx in range(len(cells))], dtype=np.float32)
    out[cells] = np.clip(values, -1.0, 1.0)
    return out.astype(np.float32)


def substrate_contexts(
    composition_task: dict[str, list[dict[str, Any]]],
    substrate: dict[str, Any] | None,
    mode: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    tasks: dict[str, dict[str, list[dict[str, Any]]]] = {skill: {split: [] for split in SPLITS} for skill in SKILLS}
    for split in SPLITS:
        for row in composition_task[split]:
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            canonical = flow.copy()
            route = tuple(row["expected_route"])
            for step_idx, skill in enumerate(route):
                base_target = e8c.apply_e8c_teacher_code(row, canonical, skill, "int4", "current_full_code_teacher")
                target = apply_substrate_to_target(flow, base_target, skill, substrate, mode)
                next_skill = route[step_idx + 1] if step_idx + 1 < len(route) else None
                tasks[skill][split].append(
                    {
                        "row_id": f"{row['row_id']}:{step_idx}:{skill}:e8d:{mode}",
                        "split": split,
                        "skill": skill,
                        "family": row["family"],
                        "route_step": step_idx,
                        "next_skill": next_skill,
                        "flow": flow.tolist(),
                        "target_flow": target.tolist(),
                        "target_value": int(target[RESULT_POS[skill]] >= 0.5),
                    }
                )
                flow = target
                canonical = base_target
    return tasks


def train_library_on_contexts(
    seed: int,
    system: str,
    loss_mode: str,
    baseline_library: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, list[dict[str, Any]]]],
    settings: Settings,
    out: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    library: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    dynamics_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    for skill in SKILLS:
        trained = e8c.train_producer_diagnostic(seed, skill, system, "int4", "current_full_code_teacher", loss_mode, baseline_library[skill], contexts[skill], settings, out)
        library[skill] = trained["state"]
        contracts[skill] = trained["contract"]
        dynamics_rows.append(
            {
                "seed": seed,
                "system": system,
                "skill": skill,
                "code": "int4",
                "loss_mode": loss_mode,
                "state_hash": e7p.state_hash(trained["state"]),
                "summary": trained["summary"],
                "oracle_used_as_teacher_target": True,
                "oracle_used_at_inference": False,
            }
        )
        gradient_rows.extend(trained["history"])
        contract_rows.append(
            {
                "seed": seed,
                "system": system,
                "skill": skill,
                "contract": trained["contract_json"],
                "state_hash": e7p.state_hash(trained["state"]),
                "semantic_labels_used": False,
            }
        )
    return library, contracts, dynamics_rows, gradient_rows, contract_rows


def fit_bridge(
    seed: int,
    system: str,
    library: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    contexts: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    params: dict[str, Any] = {"schema_version": "e8d_bridge_v1", "seed": seed, "system": system, "skills": {}}
    rows: list[dict[str, Any]] = []
    for skill in SKILLS:
        cells = np.asarray(e7y.bundle_cells(skill, OUTPUT_WIDTH), dtype=np.int64)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for row in contexts[skill]["train"]:
            flow = np.asarray(row["flow"], dtype=np.float32).reshape(1, -1)
            pred = e7r.masked_forward_np(library[skill], flow, contracts[skill]).reshape(-1).astype(np.float32)
            xs.append(pred[cells])
            ys.append(np.asarray(row["target_flow"], dtype=np.float32)[cells])
        x = np.asarray(xs, dtype=np.float32) if xs else np.zeros((0, len(cells)), dtype=np.float32)
        y = np.asarray(ys, dtype=np.float32) if ys else np.zeros((0, len(cells)), dtype=np.float32)
        scale: list[float] = []
        bias: list[float] = []
        for idx in range(len(cells)):
            xv = x[:, idx] if len(x) else np.asarray([0.0], dtype=np.float32)
            yv = y[:, idx] if len(y) else np.asarray([0.0], dtype=np.float32)
            var = float(np.var(xv))
            a = float(np.cov(xv, yv, bias=True)[0, 1] / var) if var > 1e-8 else 1.0
            b = float(np.mean(yv) - a * np.mean(xv))
            scale.append(round_float(float(np.clip(a, -3.0, 3.0))))
            bias.append(round_float(float(np.clip(b, -1.0, 1.0))))
        params["skills"][skill] = {"cells": [int(v) for v in cells.tolist()], "scale": scale, "bias": bias, "semantic_labels_used": False}
        calibrated = x * np.asarray(scale, dtype=np.float32).reshape(1, -1) + np.asarray(bias, dtype=np.float32).reshape(1, -1) if len(x) else x
        rows.append(
            {
                "seed": seed,
                "system": system,
                "skill": skill,
                "bridge_train_mae": round_float(float(np.mean(np.abs(calibrated - y))) if len(x) else 1.0),
                "bridge_hash": e8c.payload_sha256(params["skills"][skill]),
                "semantic_labels_used": False,
                "uses_target_answer_label": False,
                "no_final_answer_objective": True,
            }
        )
    params["bridge_hash"] = e8c.payload_sha256(params)
    return params, rows


def evaluate_system(
    seed: int,
    system: str,
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    task: dict[str, list[dict[str, Any]]],
    substrate: dict[str, Any] | None = None,
    substrate_mode: str = "none",
    bridge: dict[str, Any] | None = None,
    oracle_substrate: bool = False,
    dense: bool = False,
    params: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evals: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        correct: list[bool] = []
        mae_values: list[float] = []
        written_values: list[float] = []
        target_values: list[float] = []
        next_errors: list[float] = []
        sign_mismatches: list[float] = []
        drift_values: list[float] = []
        row_samples: list[dict[str, Any]] = []
        for row in task[split]:
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            canonical = flow.copy()
            route = tuple(row["expected_route"])
            step_samples: list[dict[str, Any]] = []
            for step_idx, skill in enumerate(route):
                target_after = e8c.apply_e8c_teacher_code(row, canonical, skill, "int4", "current_full_code_teacher")
                cells = np.asarray(e7y.bundle_cells(skill, OUTPUT_WIDTH), dtype=np.int64)
                target = target_after[cells].astype(np.float32)
                if oracle_substrate:
                    pred = apply_substrate_to_target(flow, target_after, skill, substrate, substrate_mode)
                else:
                    assert library is not None
                    if dense:
                        pred = e7p.np_forward(library[skill], flow.reshape(1, -1)).reshape(-1).astype(np.float32)
                    else:
                        assert contracts is not None
                        pred = e7r.masked_forward_np(library[skill], flow.reshape(1, -1), contracts[skill]).reshape(-1).astype(np.float32)
                    if bridge is not None:
                        spec = bridge["skills"][skill]
                        scale = np.asarray(spec["scale"], dtype=np.float32)
                        bias = np.asarray(spec["bias"], dtype=np.float32)
                        pred[cells] = np.clip(pred[cells] * scale + bias, -1.0, 1.0)
                    pred[cells] = e7z.quantize_bundle_values(pred[cells], skill, "int4", params, primary_is_logit=True)
                written = pred[cells].astype(np.float32)
                mae = np.abs(written - target)
                mae_values.extend(float(v) for v in mae.tolist())
                written_values.extend(float(v) for v in written.tolist())
                target_values.extend(float(v) for v in target.tolist())
                drift_values.append(float(np.mean(np.abs(pred - flow))))
                if len(written) > 1:
                    sign_mismatches.extend(float((w >= 0.0) != (t >= 0.0)) for w, t in zip(written[1:], target[1:]))
                if step_idx + 1 < len(route):
                    read = contracts[route[step_idx + 1]]["read"] if contracts and route[step_idx + 1] in contracts else np.ones(FLOW_DIM, dtype=bool)
                    next_errors.append(float(np.mean(np.abs(pred[read] - target_after[read]))))
                step_samples.append(
                    {
                        "skill": skill,
                        "bundle_mae": round_float(float(np.mean(mae))),
                        "next_error": round_float(next_errors[-1] if step_idx + 1 < len(route) and next_errors else 0.0),
                        "written_values": [round_float(float(v)) for v in written.tolist()],
                        "oracle_values": [round_float(float(v)) for v in target.tolist()],
                    }
                )
                flow = pred.astype(np.float32)
                canonical = target_after.astype(np.float32)
            predicted = int(flow[RESULT_POS[route[-1]]] >= 0.5)
            ok = predicted == int(row["target_answer"])
            correct.append(bool(ok))
            if len(row_samples) < 3:
                row_samples.append({"row_id": row["row_id"], "family": row["family"], "route": list(route), "target": int(row["target_answer"]), "predicted": predicted, "correct": bool(ok), "steps": step_samples})
        answer_acc = float(np.mean(correct)) if correct else 0.0
        bundle_mae = float(np.mean(mae_values)) if mae_values else 0.0
        next_error = float(np.mean(next_errors)) if next_errors else 0.0
        sign_mismatch = float(np.mean(sign_mismatches)) if sign_mismatches else 0.0
        state_drift = float(np.mean(drift_values)) if drift_values else 0.0
        usefulness = answer_acc - 0.10 - 0.12 * next_error - 0.02 * state_drift
        compatibility = max(0.0, 1.0 - next_error)
        evals[split] = {
            "answer_accuracy": round_float(answer_acc),
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(usefulness),
            "consumer_read_compatibility": round_float(compatibility),
            "producer_write_compatibility": round_float(max(0.0, 1.0 - bundle_mae)),
            "oracle_code_similarity": round_float(max(0.0, 1.0 - bundle_mae)),
            "bundle_mean_absolute_error_to_oracle": round_float(bundle_mae),
            "bundle_cellwise_correlation_with_oracle": e7z.safe_corr(written_values, target_values),
            "bundle_cosine_similarity_with_oracle": e7z.cosine_similarity(written_values, target_values),
            "support_channel_sign_mismatch_rate": round_float(sign_mismatch),
            "write_entropy": e8c.bundle_entropy(np.asarray(written_values, dtype=np.float32)),
            "next_pocket_compatibility_error": round_float(next_error),
            "state_drift_per_step": round_float(state_drift),
            "mean_route_steps": round_float(float(np.mean([len(row["expected_route"]) for row in task[split]])) if task[split] else 0.0),
            "row_level_samples": row_samples,
        }
        sample_rows.extend({"seed": seed, "system": system, "split": split, "row_id": sample["row_id"], "correct": sample["correct"], "steps": sample["steps"]} for sample in row_samples)
    result = {
        "seed": seed,
        "system": system,
        "code": "int4",
        "evals": evals,
        "eval_mean_answer_accuracy": round_float(float(np.mean([evals[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([evals[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_oracle_code_similarity": round_float(float(np.mean([evals[split]["oracle_code_similarity"] for split in EVAL_SPLITS]))),
        "eval_mean_producer_write_compatibility": round_float(float(np.mean([evals[split]["producer_write_compatibility"] for split in EVAL_SPLITS]))),
        "eval_mean_consumer_read_compatibility": round_float(float(np.mean([evals[split]["consumer_read_compatibility"] for split in EVAL_SPLITS]))),
        "eval_mean_bundle_mean_absolute_error_to_oracle": round_float(float(np.mean([evals[split]["bundle_mean_absolute_error_to_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_compatibility_error": round_float(float(np.mean([evals[split]["next_pocket_compatibility_error"] for split in EVAL_SPLITS]))),
        "eval_mean_support_channel_sign_mismatch_rate": round_float(float(np.mean([evals[split]["support_channel_sign_mismatch_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_state_drift_per_step": round_float(float(np.mean([evals[split]["state_drift_per_step"] for split in EVAL_SPLITS]))),
        "eval_mean_write_entropy": round_float(float(np.mean([evals[split]["write_entropy"] for split in EVAL_SPLITS]))),
        "bit_budget": e7z.estimate_boundary_bits("int4"),
        "boundary_bit_budget": e7z.estimate_boundary_bits("int4"),
    }
    return result, sample_rows


def aggregate_results(rows: list[dict[str, Any]], dynamics_rows: list[dict[str, Any]], validity_rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        subset = sorted((row for row in rows if row["system"] == system), key=lambda row: int(row.get("seed", -1)))
        if not subset:
            continue
        numeric_keys = sorted({key for row in subset for key, value in row.items() if isinstance(value, (int, float)) and key != "seed"})
        mean = {key: round_float(math.fsum(float(row.get(key, 0.0)) for row in subset) / len(subset)) for key in numeric_keys}
        dyn = [row["summary"] for row in dynamics_rows if row["system"] == system and row.get("summary")]
        for metric in ("validation_code_similarity_best", "gradient_negative_rate_mean", "gradient_cosine_mean", "tail_gain", "tail_range"):
            if dyn:
                mean[metric] = round_float(math.fsum(float(row.get(metric, 0.0)) for row in dyn) / len(dyn))
        valid = [row for row in validity_rows if row["system"] == system and row["split"] in EVAL_SPLITS]
        if valid:
            mean["substrate_validity_score"] = round_float(math.fsum(float(row["ram_validity_score"]) for row in valid) / len(valid))
            mean["substrate_code_entropy"] = round_float(math.fsum(float(row["code_entropy"]) for row in valid) / len(valid))
            mean["substrate_low_bit_utilization"] = round_float(math.fsum(float(row["low_bit_code_utilization"]) for row in valid) / len(valid))
        systems[system] = {"seed_count": len({row["seed"] for row in subset}), "mean": mean}
    candidates = [system for system in systems if system not in ORACLE_SYSTEMS and system != "dense_graph_danger_control"]
    best = max(candidates, key=lambda s: systems[s]["mean"].get("eval_mean_composition_usefulness", -1e9))
    return {"schema_version": "e8d_aggregate_metrics_v1", "systems": systems, "best_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    mean = {system: aggregate["systems"].get(system, {}).get("mean", {}) for system in SYSTEMS}
    useful = {system: mean[system].get("eval_mean_composition_usefulness", 0.0) for system in SYSTEMS}
    write = {system: mean[system].get("eval_mean_producer_write_compatibility", 0.0) for system in SYSTEMS}
    read = {system: mean[system].get("eval_mean_consumer_read_compatibility", 0.0) for system in SYSTEMS}
    valid = {system: mean[system].get("substrate_validity_score", 0.0) for system in SYSTEMS}
    baseline = useful.get("no_substrate_baseline", 0.0)
    bridge = useful.get("bridge_only_baseline", 0.0)
    substrate_systems = ("substrate_autoencoder", "substrate_transition_model", "low_bit_substrate_codebook", "frozen_substrate_then_producer", "frozen_substrate_then_consumer", "frozen_substrate_then_pocket_composition")
    best_substrate = max(substrate_systems, key=lambda system: useful.get(system, -1e9))
    best = aggregate["best_system"]
    best_score = aggregate["best_eval_mean_composition_usefulness"]
    substrate_gain = round_float(useful.get(best_substrate, 0.0) - max(baseline, bridge))
    bridge_gain = round_float(bridge - baseline)
    dense_margin = useful.get("dense_graph_danger_control", 0.0) - best_score
    if dense_margin >= 0.02:
        decision = "e8d_graph_soup_regression_detected"
    elif substrate_gain >= 0.03 and read.get(best_substrate, 0.0) >= read.get("no_substrate_baseline", 0.0) - 0.01 and write.get(best_substrate, 0.0) >= write.get("no_substrate_baseline", 0.0) - 0.01:
        decision = "e8d_substrate_first_positive"
    elif bridge >= useful.get(best_substrate, 0.0) + 0.02 and bridge_gain >= 0.02:
        decision = "e8d_bridge_adapter_sufficient"
    elif max(valid.get(system, 0.0) for system in substrate_systems) >= 0.86 and max(write.get(system, 0.0) for system in substrate_systems) < write.get("no_substrate_baseline", 0.0) - 0.03:
        decision = "e8d_pocket_to_substrate_write_bottleneck"
    elif max(valid.get(system, 0.0) for system in substrate_systems) >= 0.86 and max(read.get(system, 0.0) for system in substrate_systems) < read.get("no_substrate_baseline", 0.0) - 0.03:
        decision = "e8d_substrate_consumer_read_bottleneck"
    elif useful.get("jointly_mutable_substrate_and_pockets", 0.0) >= useful.get(best_substrate, 0.0) + 0.02:
        decision = "e8d_frozen_substrate_too_rigid"
    else:
        decision = "e8d_substrate_language_not_helpful"
    return {
        "schema_version": "e8d_decision_v1",
        "decision": decision,
        "detail": {
            "best_system": best,
            "best_score": best_score,
            "baseline": baseline,
            "bridge_only_baseline": bridge,
            "best_substrate_system": best_substrate,
            "best_substrate_usefulness": useful.get(best_substrate, 0.0),
            "substrate_gain_over_best_baseline": substrate_gain,
            "bridge_gain_over_baseline": bridge_gain,
            "oracle_substrate_reference": useful.get("oracle_substrate_reference", 0.0),
            "dense_graph_danger_control": useful.get("dense_graph_danger_control", 0.0),
            "dense_margin_over_clean_best": round_float(dense_margin),
            "best_substrate_validity": valid.get(best_substrate, 0.0),
            "best_substrate_write_compatibility": write.get(best_substrate, 0.0),
            "best_substrate_read_compatibility": read.get(best_substrate, 0.0),
        },
        "deterministic_replay_passed": False,
    }


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    e7o_settings = e8c.to_e7o_settings(to_e8c_settings(settings))
    e7z_settings = to_e7z_settings(settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    baseline_library: dict[str, dict[str, Any]] = {}
    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        baseline_library[skill] = e7p.copy_state(trained["state"], "E8D_baseline_standalone")
    base_contexts = e8c.generate_context_tasks(composition_task, "int4", "current_full_code_teacher")
    rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    substrate_rows: list[dict[str, Any]] = []
    validity_rows: list[dict[str, Any]] = []
    consumer_rows: list[dict[str, Any]] = []
    compatibility_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    baseline_lib, baseline_contracts, dyn, grad, _ = train_library_on_contexts(seed, "no_substrate_baseline", "baseline", baseline_library, base_contexts, settings, out)
    dynamics_rows.extend(dyn)
    gradient_rows.extend(grad)
    result, samples = evaluate_system(seed, "no_substrate_baseline", baseline_lib, baseline_contracts, composition_task)
    rows.append(result)
    sample_rows.extend(samples)
    if out:
        append_progress(out, "e8d_system_evaluated", seed=seed, system="no_substrate_baseline", usefulness=result["eval_mean_composition_usefulness"])

    bridge, bridge_rows = fit_bridge(seed, "bridge_only_baseline", baseline_lib, baseline_contracts, base_contexts)
    substrate_rows.extend(bridge_rows)
    result, samples = evaluate_system(seed, "bridge_only_baseline", baseline_lib, baseline_contracts, composition_task, bridge=bridge)
    rows.append(result)
    sample_rows.extend(samples)

    substrate_configs = {
        "substrate_autoencoder": ("autoencoder", "baseline"),
        "substrate_transition_model": ("transition", "baseline"),
        "low_bit_substrate_codebook": ("low_bit", "baseline"),
        "frozen_substrate_then_producer": ("frozen", "baseline"),
        "frozen_substrate_then_consumer": ("frozen", "consumer_compatibility"),
        "frozen_substrate_then_pocket_composition": ("frozen", "route_step_local"),
        "jointly_mutable_substrate_and_pockets": ("joint", "consumer_compatibility"),
    }
    learned: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], str]] = {}
    for system, (mode, loss_mode) in substrate_configs.items():
        substrate, pretrain, validity = fit_substrate(seed, system, mode, base_contexts)
        substrate_rows.extend(pretrain)
        validity_rows.extend(validity)
        contexts = substrate_contexts(composition_task, substrate, mode)
        lib, contracts, dyn, grad, _ = train_library_on_contexts(seed, system, loss_mode, baseline_library, contexts, settings, out)
        learned[system] = (lib, contracts, substrate, mode)
        dynamics_rows.extend(dyn)
        gradient_rows.extend(grad)
        if system == "jointly_mutable_substrate_and_pockets":
            params, mutation = e7z.repair_boundary(seed, system, "int4", lib, contracts, composition_task, e7z_settings, out)
            result, samples = evaluate_system(seed, system, lib, contracts, composition_task, substrate, mode, params=params)
            result.update({key: value for key, value in mutation.items() if key != "history"})
            repair_rows.extend(mutation["history"])
        else:
            result, samples = evaluate_system(seed, system, lib, contracts, composition_task, substrate, mode)
        rows.append(result)
        sample_rows.extend(samples)
        read_result, read_rows = e8a.evaluate_consumer_read(seed, f"{system}_consumer_read", "int4", lib, contracts, contexts)
        consumer_rows.extend(read_rows)
        compatibility_rows.append(
            {
                "seed": seed,
                "system": system,
                "producer_write_compatibility": result["eval_mean_producer_write_compatibility"],
                "consumer_read_compatibility": result["eval_mean_consumer_read_compatibility"],
                "consumer_read_accuracy": read_result["eval_mean_answer_accuracy"],
                "next_pocket_compatibility_error": result["eval_mean_next_pocket_compatibility_error"],
                "substrate_hash": substrate["substrate_hash"],
            }
        )
        if out:
            append_progress(out, "e8d_system_evaluated", seed=seed, system=system, usefulness=result["eval_mean_composition_usefulness"])

    ref_lib, ref_contracts, ref_substrate, ref_mode = learned["frozen_substrate_then_pocket_composition"]
    result, samples = evaluate_system(seed, "oracle_substrate_reference", ref_lib, ref_contracts, composition_task, ref_substrate, ref_mode, oracle_substrate=True)
    result["diagnostic_only"] = True
    result["oracle_used_at_inference"] = True
    rows.append(result)
    sample_rows.extend(samples)

    dense_result, dense_samples = evaluate_system(seed, "dense_graph_danger_control", baseline_library, None, composition_task, dense=True)
    rows.append(dense_result)
    sample_rows.extend(dense_samples)

    return {
        "seed": seed,
        "rows": rows,
        "dynamics_rows": dynamics_rows,
        "gradient_rows": gradient_rows,
        "substrate_rows": substrate_rows,
        "validity_rows": validity_rows,
        "consumer_rows": consumer_rows,
        "compatibility_rows": compatibility_rows,
        "repair_rows": repair_rows,
        "sample_rows": sample_rows,
    }


def build_report_text(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# E8D Substrate-First RAM Language Pretraining Probe Result",
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
            f"{system:<46} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"write={mean.get('eval_mean_producer_write_compatibility', 0.0):.6f} "
            f"read={mean.get('eval_mean_consumer_read_compatibility', 0.0):.6f} "
            f"valid={mean.get('substrate_validity_score', 0.0):.6f} "
            f"drift={mean.get('eval_mean_state_drift_per_step', 0.0):.6f} "
            f"grad_neg={mean.get('gradient_negative_rate_mean', 0.0):.6f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Boundary",
            "",
            "E8D is a controlled symbolic/numeric Flow/RAM substrate probe. The substrate is trained on valid intermediate state geometry and transition bundles, not on a RAM-to-final-answer objective. Oracle substrate is diagnostic only. No raw-language, AGI, consciousness, deployed-model, or model-scale claim is made.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_reports(
    rows: list[dict[str, Any]],
    dynamics_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
    substrate_rows: list[dict[str, Any]],
    validity_rows: list[dict[str, Any]],
    consumer_rows: list[dict[str, Any]],
    compatibility_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    return {
        "substrate_pretraining_report.json": {"schema_version": "e8d_substrate_pretraining_report_v1", "rows": sorted(substrate_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", "")))},
        "ram_validity_report.json": {"schema_version": "e8d_ram_validity_report_v1", "rows": sorted(validity_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", ""), r.get("split", "")))},
        "producer_dynamics_report.json": {"schema_version": "e8d_producer_dynamics_report_v1", "rows": sorted(dynamics_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", "")))},
        "consumer_read_report.json": {"schema_version": "e8d_consumer_read_report_v1", "rows": sorted(consumer_rows, key=lambda r: (r["seed"], r["system"], r.get("split", "")))},
        "compatibility_report.json": {"schema_version": "e8d_compatibility_report_v1", "rows": sorted(compatibility_rows, key=lambda r: (r["seed"], r["system"]))},
        "mutation_repair_report.json": {"schema_version": "e8d_mutation_repair_report_v1", "rows": sorted(repair_rows, key=lambda r: (r["seed"], r["system"], r["generation"]))},
        "gradient_diagnostics_report.json": {"schema_version": "e8d_gradient_diagnostics_report_v1", "rows": sorted(gradient_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", ""), r.get("epoch", -1)))},
        "system_results.json": {"schema_version": "e8d_system_results_v1", "rows": sorted(rows, key=lambda r: (r["seed"], SYSTEMS.index(r["system"])))},
        "row_level_samples.json": {"schema_version": "e8d_row_level_samples_v1", "rows": sorted(sample_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"]))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e8d_summary_v1", "decision": decision["decision"], "best_system": aggregate["best_system"], "deterministic_replay_passed": deterministic.get("internal_replay_passed", False), "checker_failure_count": None},
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
                "schema_version": "e8d_backend_manifest_v1",
                "milestone": MILESTONE,
                "settings": settings_payload(settings),
                "systems": list(SYSTEMS),
                "flow_dim": FLOW_DIM,
                "output_width": OUTPUT_WIDTH,
                "substrate_first": True,
                "new_router": False,
                "semantic_lane_labels_as_model_input": False,
                "substrate_final_answer_objective": False,
                "oracle_write_at_inference_for_learned_systems": False,
                "oracle_substrate_reference_diagnostic_only": True,
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
                "schema_version": "e8d_task_generation_report_v1",
                "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()},
                "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()},
            },
        )
        rows: list[dict[str, Any]] = []
        dynamics_rows: list[dict[str, Any]] = []
        gradient_rows: list[dict[str, Any]] = []
        substrate_rows: list[dict[str, Any]] = []
        validity_rows: list[dict[str, Any]] = []
        consumer_rows: list[dict[str, Any]] = []
        compatibility_rows: list[dict[str, Any]] = []
        repair_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        jobs = [{"seed": seed, "settings": settings.__dict__.copy(), "composition_task": composition_tasks[seed], "pocket_task": pocket_tasks[seed], "out": str(out)} for seed in settings.seeds]
        max_workers = max(1, min(settings.cpu_workers, len(jobs)))
        if settings.device == "cuda" and max_workers > 1:
            append_progress(out, "cuda_process_pool_disabled", requested_workers=max_workers, active_workers=1)
            max_workers = 1
        if max_workers == 1:
            for job in jobs:
                result = seed_worker(job)
                rows.extend(result["rows"])
                dynamics_rows.extend(result["dynamics_rows"])
                gradient_rows.extend(result["gradient_rows"])
                substrate_rows.extend(result["substrate_rows"])
                validity_rows.extend(result["validity_rows"])
                consumer_rows.extend(result["consumer_rows"])
                compatibility_rows.extend(result["compatibility_rows"])
                repair_rows.extend(result["repair_rows"])
                sample_rows.extend(result["sample_rows"])
                append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(jobs) - len({row["seed"] for row in rows}))
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_dynamics_rows": len(dynamics_rows), "completed_gradient_rows": len(gradient_rows), "completed_substrate_rows": len(substrate_rows), "pending": len(jobs) - len({row["seed"] for row in rows})})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(seed_worker, job): job for job in jobs}
                while futures:
                    done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_dynamics_rows": len(dynamics_rows), "completed_gradient_rows": len(gradient_rows), "completed_substrate_rows": len(substrate_rows), "pending": len(futures)})
                        continue
                    for future in done:
                        job = futures.pop(future)
                        result = future.result()
                        rows.extend(result["rows"])
                        dynamics_rows.extend(result["dynamics_rows"])
                        gradient_rows.extend(result["gradient_rows"])
                        substrate_rows.extend(result["substrate_rows"])
                        validity_rows.extend(result["validity_rows"])
                        consumer_rows.extend(result["consumer_rows"])
                        compatibility_rows.extend(result["compatibility_rows"])
                        repair_rows.extend(result["repair_rows"])
                        sample_rows.extend(result["sample_rows"])
                        append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_dynamics_rows": len(dynamics_rows), "completed_gradient_rows": len(gradient_rows), "completed_substrate_rows": len(substrate_rows), "last_completed": f"seed{job['seed']}", "pending": len(futures)})
        aggregate = aggregate_results(rows, dynamics_rows, validity_rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e8d_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, dynamics_rows, gradient_rows, substrate_rows, validity_rows, consumer_rows, compatibility_rows, repair_rows, sample_rows, aggregate, decision, deterministic_placeholder)
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
            deterministic = {"schema_version": "e8d_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, dynamics_rows, gradient_rows, substrate_rows, validity_rows, consumer_rows, compatibility_rows, repair_rows, sample_rows, aggregate, decision, deterministic)
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
    parser.add_argument("--cpu-workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
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
