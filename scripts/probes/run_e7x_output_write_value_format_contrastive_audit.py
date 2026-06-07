#!/usr/bin/env python3
"""E7X output write value-format contrastive audit.

E7W localized the main composition break to the value/format written back into
Flow/RAM. E7X compares, on the same pocket calls, low-score real writes against
oracle/canonical writes and simple learned value transforms.
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
MILESTONE = "E7X_OUTPUT_WRITE_VALUE_FORMAT_CONTRASTIVE_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e7x_output_write_value_format_contrastive_audit")
DEFAULT_SEEDS = (100201, 100202, 100203, 100204, 100205, 100206, 100207, 100208)

SYSTEMS = (
    "baseline_real_write",
    "oracle_write_reference",
    "affine_calibrated_write",
    "monotonic_calibrated_write",
    "zscore_normalized_write",
    "codebook_write",
    "sign_or_quantized_write",
    "residual_delta_write",
    "router_integrated_write",
)
TRANSFORM_SYSTEMS = tuple(system for system in SYSTEMS if system != "oracle_write_reference")
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "write_transform_report.json",
    "write_morphology_report.json",
    "write_histogram_report.json",
    "oracle_real_scatter_report.json",
    "ram_grid_frame_report.json",
    "top_failing_rows_report.json",
    "system_results.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7x_output_scale_bias_calibration_bottleneck",
    "e7x_output_nonlinear_calibration_bottleneck",
    "e7x_canonical_value_code_required",
    "e7x_delta_write_format_required",
    "e7x_flow_integrator_required",
    "e7x_output_value_format_not_sufficient",
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
e7r = e7w.e7r
e7p = e7w.e7p
e7o = e7w.e7o

FLOW_DIM = int(e7w.FLOW_DIM)
SKILLS = tuple(e7w.SKILLS)
SPLITS = tuple(e7w.SPLITS)
EVAL_SPLITS = tuple(e7w.EVAL_SPLITS)
RESULT_POS = dict(e7w.RESULT_POS)


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


def to_e7v_settings(settings: Settings) -> Any:
    return e7w.to_e7v_settings(to_e7w_settings(settings))


def to_e7o_settings(settings: Settings) -> Any:
    return e7w.to_e7o_settings(to_e7w_settings(settings))


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


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


def entropy_effective_levels(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rounded = [round(float(value), 3) for value in values]
    uniq, counts = np.unique(np.asarray(rounded), return_counts=True)
    probs = counts.astype(np.float64) / float(np.sum(counts))
    entropy = float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
    return round_float(entropy), round_float(float(len(uniq)))


def histogram(values: list[float], bins: int = 21) -> dict[str, Any]:
    if not values:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins, range=(-1.0, 2.0))
    return {"bin_edges": [round_float(float(edge)) for edge in edges.tolist()], "counts": [int(value) for value in counts.tolist()]}


def collect_raw_pairs(
    library: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    context_tasks: dict[str, Any],
    split: str,
) -> dict[str, list[dict[str, float]]]:
    pairs: dict[str, list[dict[str, float]]] = {}
    for skill in SKILLS:
        contract = contracts[skill]
        rows = []
        for row in context_tasks[skill][split]:
            flow = np.asarray(row["flow"], dtype=np.float32).reshape(1, -1)
            pred = e7r.masked_forward_np(library[skill], flow, contract)
            raw = float(pred[0, contract["assignment_cell"]])
            target = float(row["target_value"])
            before = float(flow[0, RESULT_POS[skill]])
            rows.append({"raw": raw, "target": target, "before": before})
        pairs[skill] = rows
    return pairs


def fit_transforms(pairs: dict[str, list[dict[str, float]]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    transforms: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for skill in SKILLS:
        raw = np.asarray([row["raw"] for row in pairs[skill]], dtype=np.float64)
        target = np.asarray([row["target"] for row in pairs[skill]], dtype=np.float64)
        if len(raw) == 0:
            raw = np.asarray([0.0])
            target = np.asarray([0.0])
        x = np.stack([raw, np.ones_like(raw)], axis=1)
        scale, bias = np.linalg.lstsq(x, target, rcond=None)[0]
        candidates = sorted(set(raw.tolist())) or [0.0]
        candidates = [candidates[0] - 1.0] + candidates + [candidates[-1] + 1.0, 0.0]
        best_threshold = 0.0
        best_acc = -1.0
        for threshold in candidates:
            pred = (raw >= threshold).astype(np.float64)
            acc = float(np.mean(pred == target))
            if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and abs(float(threshold)) < abs(best_threshold)):
                best_acc = acc
                best_threshold = float(threshold)
        mean = float(np.mean(raw))
        std = float(np.std(raw) + 1e-6)
        target0 = raw[target < 0.5]
        target1 = raw[target >= 0.5]
        center0 = float(np.mean(target0)) if len(target0) else mean - std
        center1 = float(np.mean(target1)) if len(target1) else mean + std
        low = float(best_threshold - 0.5 * std)
        high = float(best_threshold + 0.5 * std)
        transforms[skill] = {
            "scale": round_float(scale),
            "bias": round_float(bias),
            "threshold": round_float(best_threshold),
            "temperature": round_float(max(0.05, std)),
            "mean": round_float(mean),
            "std": round_float(std),
            "codebook_centers": [round_float(center0), round_float(center1)],
            "codebook_values": [0.0, 1.0],
            "quant_low": round_float(low),
            "quant_high": round_float(high),
            "train_accuracy_at_threshold": round_float(best_acc),
        }
        affine_pred = np.clip(scale * raw + bias, 0.0, 1.0)
        rows.append({
            "skill": skill,
            **transforms[skill],
            "raw_mean": round_float(mean),
            "raw_std": round_float(std),
            "target_mean": round_float(float(np.mean(target))),
            "affine_mae": round_float(float(np.mean(np.abs(affine_pred - target)))),
            "raw_target_correlation": safe_corr(raw.tolist(), target.tolist()),
        })
    return transforms, rows


def write_value(system: str, raw: float, before: float, oracle: float, transform: dict[str, Any]) -> float:
    if system == "baseline_real_write":
        return 1.0 if raw >= 0.0 else 0.0
    if system == "oracle_write_reference":
        return float(oracle)
    if system == "affine_calibrated_write":
        return clip01(float(transform["scale"]) * raw + float(transform["bias"]))
    if system == "monotonic_calibrated_write":
        return sigmoid((raw - float(transform["threshold"])) / max(0.05, float(transform["temperature"])))
    if system == "zscore_normalized_write":
        return sigmoid((raw - float(transform["mean"])) / max(0.05, float(transform["std"])))
    if system == "codebook_write":
        centers = list(transform["codebook_centers"])
        values = list(transform["codebook_values"])
        idx = int(np.argmin([abs(raw - float(center)) for center in centers]))
        return float(values[idx])
    if system == "sign_or_quantized_write":
        if raw < float(transform["quant_low"]):
            return 0.0
        if raw > float(transform["quant_high"]):
            return 1.0
        return 0.5
    if system == "residual_delta_write":
        proposal = sigmoid((raw - float(transform["threshold"])) / max(0.05, float(transform["temperature"])))
        return clip01(before + 0.85 * (proposal - before))
    if system == "router_integrated_write":
        proposal = sigmoid((raw - float(transform["threshold"])) / max(0.05, float(transform["temperature"])))
        if proposal >= 0.62:
            return 1.0
        if proposal <= 0.38:
            return 0.0
        return clip01(0.5 * before + 0.5 * proposal)
    raise ValueError(system)


def evaluate_system(
    seed: int,
    system: str,
    library: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    task: dict[str, list[dict[str, Any]]],
    transforms: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evals: dict[str, Any] = {}
    scatter_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    failing_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        correct: list[bool] = []
        samples: list[dict[str, Any]] = []
        written_values: list[float] = []
        oracle_values: list[float] = []
        raw_values: list[float] = []
        mae_values: list[float] = []
        delta_values: list[float] = []
        next_errors: list[float] = []
        per_skill: dict[str, dict[str, list[float]]] = {skill: {"write": [], "oracle": [], "raw": [], "mae": []} for skill in SKILLS}
        for row in task[split]:
            route = tuple(row["expected_route"])
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            canonical = flow.copy()
            step_details = []
            for step_idx, skill in enumerate(route):
                before = flow.copy()
                canonical_before = canonical.copy()
                canonical_after = e7w.canonical_step(row, canonical_before, skill)
                contract = contracts[skill]
                pred = e7r.masked_forward_np(library[skill], flow.reshape(1, -1), contract).reshape(-1)
                raw = float(pred[contract["assignment_cell"]])
                oracle = float(canonical_after[RESULT_POS[skill]])
                value = write_value(system, raw, float(before[RESULT_POS[skill]]), oracle, transforms[skill])
                flow = pred.astype(np.float32)
                flow[RESULT_POS[skill]] = np.float32(value)
                canonical = canonical_after
                mae = abs(value - oracle)
                next_error = 0.0
                if step_idx + 1 < len(route):
                    next_contract = contracts[route[step_idx + 1]]
                    next_error = float(np.mean(np.abs(flow[next_contract["read"]] - canonical_after[next_contract["read"]])))
                raw_values.append(raw)
                written_values.append(value)
                oracle_values.append(oracle)
                mae_values.append(mae)
                delta_values.append(abs(value - float(before[RESULT_POS[skill]])))
                next_errors.append(next_error)
                per_skill[skill]["write"].append(value)
                per_skill[skill]["oracle"].append(oracle)
                per_skill[skill]["raw"].append(raw)
                per_skill[skill]["mae"].append(mae)
                if len(scatter_rows) < 300:
                    scatter_rows.append({"seed": seed, "system": system, "split": split, "row_id": row["row_id"], "skill": skill, "step": step_idx + 1, "raw_real_write": round_float(raw), "written_value": round_float(value), "oracle_write": round_float(oracle)})
                if len(frame_rows) < 120:
                    frame_rows.append({"seed": seed, "system": system, "split": split, "row_id": row["row_id"], "skill": skill, "step": step_idx + 1, "flow_before": [round_float(float(v)) for v in before.tolist()], "flow_after": [round_float(float(v)) for v in flow.tolist()], "oracle_after": [round_float(float(v)) for v in canonical_after.tolist()], "write_cell": int(RESULT_POS[skill]), "written_value": round_float(value), "oracle_write": round_float(oracle)})
                step_details.append({"skill": skill, "raw": round_float(raw), "write": round_float(value), "oracle": round_float(oracle), "mae": round_float(mae), "next_error": round_float(next_error)})
            pred_answer = int(e7o.predict_answer_from_flow(row, flow))
            ok = pred_answer == int(row["target_answer"])
            correct.append(ok)
            if not ok and len(failing_rows) < 100:
                failing_rows.append({"seed": seed, "system": system, "split": split, "row_id": row["row_id"], "family": row["family"], "route": list(route), "target": int(row["target_answer"]), "predicted": pred_answer, "steps": step_details})
            if len(samples) < 8:
                samples.append({"row_id": row["row_id"], "family": row["family"], "route": list(route), "target": int(row["target_answer"]), "predicted": pred_answer, "correct": bool(ok), "steps": step_details})
        acc = round_float(float(np.mean(correct)))
        mean_steps = round_float(float(np.mean([len(row["expected_route"]) for row in task[split]])))
        cost_penalty = min(0.10, 0.00000016 * sum(e7p.bit_budget(state) for state in library.values()) + 0.0025 * mean_steps)
        entropy, levels = entropy_effective_levels(written_values)
        evals[split] = {
            "answer_accuracy": acc,
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(max(0.0, acc - cost_penalty)),
            "mean_route_steps": mean_steps,
            "oracle_write_similarity": round_float(1.0 - float(np.mean(mae_values)) if mae_values else 0.0),
            "cellwise_correlation_with_oracle": safe_corr(written_values, oracle_values),
            "cosine_similarity_with_oracle": cosine_similarity(written_values, oracle_values),
            "mean_absolute_error_to_oracle": round_float(float(np.mean(mae_values)) if mae_values else 0.0),
            "scale_ratio": round_float(float(np.std(written_values) / (np.std(oracle_values) + 1e-6)) if written_values else 0.0),
            "bias_offset": round_float(float(np.mean(written_values) - np.mean(oracle_values)) if written_values else 0.0),
            "value_min": round_float(float(np.min(written_values)) if written_values else 0.0),
            "value_max": round_float(float(np.max(written_values)) if written_values else 0.0),
            "saturation_rate": round_float(float(np.mean([(value <= 0.05 or value >= 0.95) for value in written_values])) if written_values else 0.0),
            "sign_mismatch_rate": round_float(float(np.mean([(value >= 0.5) != (oracle >= 0.5) for value, oracle in zip(written_values, oracle_values)])) if written_values else 0.0),
            "entropy": entropy,
            "effective_value_levels": levels,
            "noise_floor": round_float(float(np.mean([min(abs(value), abs(1.0 - value)) for value in written_values])) if written_values else 0.0),
            "delta_magnitude": round_float(float(np.mean(delta_values)) if delta_values else 0.0),
            "next_pocket_input_compatibility": round_float(float(np.mean(next_errors)) if next_errors else 0.0),
            "bit_budget": sum(e7p.bit_budget(state) for state in library.values()),
            "row_level_samples": samples,
        }
        for skill in SKILLS:
            writes = per_skill[skill]["write"]
            oracles = per_skill[skill]["oracle"]
            evals[split][f"{skill}_write_mae"] = round_float(float(np.mean(per_skill[skill]["mae"])) if writes else 0.0)
            evals[split][f"{skill}_write_correlation"] = safe_corr(writes, oracles)
    row = {
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
        "eval_mean_oracle_write_similarity": round_float(float(np.mean([evals[split]["oracle_write_similarity"] for split in EVAL_SPLITS]))),
        "eval_mean_cellwise_correlation_with_oracle": round_float(float(np.mean([evals[split]["cellwise_correlation_with_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_cosine_similarity_with_oracle": round_float(float(np.mean([evals[split]["cosine_similarity_with_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_absolute_error_to_oracle": round_float(float(np.mean([evals[split]["mean_absolute_error_to_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_scale_ratio": round_float(float(np.mean([evals[split]["scale_ratio"] for split in EVAL_SPLITS]))),
        "eval_mean_bias_offset": round_float(float(np.mean([evals[split]["bias_offset"] for split in EVAL_SPLITS]))),
        "eval_mean_saturation_rate": round_float(float(np.mean([evals[split]["saturation_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_sign_mismatch_rate": round_float(float(np.mean([evals[split]["sign_mismatch_rate"] for split in EVAL_SPLITS]))),
        "eval_mean_entropy": round_float(float(np.mean([evals[split]["entropy"] for split in EVAL_SPLITS]))),
        "eval_mean_effective_value_levels": round_float(float(np.mean([evals[split]["effective_value_levels"] for split in EVAL_SPLITS]))),
        "eval_mean_noise_floor": round_float(float(np.mean([evals[split]["noise_floor"] for split in EVAL_SPLITS]))),
        "eval_mean_delta_magnitude": round_float(float(np.mean([evals[split]["delta_magnitude"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_input_compatibility": round_float(float(np.mean([evals[split]["next_pocket_input_compatibility"] for split in EVAL_SPLITS]))),
        "parameter_count": sum(e7p.parameter_count(state) for state in library.values()),
        "bit_budget": sum(e7p.bit_budget(state) for state in library.values()),
    }
    return row, scatter_rows, frame_rows, failing_rows


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
        "eval_mean_oracle_write_similarity",
        "eval_mean_cellwise_correlation_with_oracle",
        "eval_mean_cosine_similarity_with_oracle",
        "eval_mean_absolute_error_to_oracle",
        "eval_mean_scale_ratio",
        "eval_mean_bias_offset",
        "eval_mean_saturation_rate",
        "eval_mean_sign_mismatch_rate",
        "eval_mean_entropy",
        "eval_mean_effective_value_levels",
        "eval_mean_noise_floor",
        "eval_mean_delta_magnitude",
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
    candidates = [system for system in SYSTEMS if system != "oracle_write_reference"]
    best = max(candidates, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    return {"schema_version": "e7x_aggregate_metrics_v1", "systems": systems, "best_non_oracle_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    mean = {system: aggregate["systems"][system]["mean"].get("eval_mean_composition_usefulness", 0.0) for system in SYSTEMS}
    baseline = mean["baseline_real_write"]
    oracle = mean["oracle_write_reference"]
    gap = max(1e-9, oracle - baseline)
    closes = {system: (mean[system] - baseline) / gap for system in SYSTEMS}
    best = aggregate["best_non_oracle_system"]
    detail = {
        "baseline": baseline,
        "oracle": oracle,
        "affine": mean["affine_calibrated_write"],
        "monotonic": mean["monotonic_calibrated_write"],
        "zscore": mean["zscore_normalized_write"],
        "codebook": mean["codebook_write"],
        "sign_or_quantized": mean["sign_or_quantized_write"],
        "residual": mean["residual_delta_write"],
        "router_integrated": mean["router_integrated_write"],
        "best_non_oracle_system": best,
        "best_gap_fraction_closed": round_float(closes[best]),
    }
    threshold = 0.60
    if best == "affine_calibrated_write" and closes[best] >= threshold:
        decision = "e7x_output_scale_bias_calibration_bottleneck"
    elif best in {"monotonic_calibrated_write", "zscore_normalized_write"} and closes[best] >= threshold:
        decision = "e7x_output_nonlinear_calibration_bottleneck"
    elif best in {"codebook_write", "sign_or_quantized_write"} and closes[best] >= threshold:
        decision = "e7x_canonical_value_code_required"
    elif best == "residual_delta_write" and closes[best] >= threshold:
        decision = "e7x_delta_write_format_required"
    elif best == "router_integrated_write" and closes[best] >= threshold:
        decision = "e7x_flow_integrator_required"
    else:
        decision = "e7x_output_value_format_not_sufficient"
    return {"schema_version": "e7x_decision_v1", "decision": decision, "detail": detail, "deterministic_replay_passed": False}


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
    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        state = e7p.copy_state(trained["state"], "e7x_baseline_standalone")
        baseline_library[skill] = state
        training_rows.append({"seed": seed, "system": "baseline_standalone_pocket", "skill": skill, "state_hash": e7p.state_hash(state), "standalone": trained["standalone"]})

    read_maps = {skill: e7v.priority_read_map(skill, settings.pruned_read_count) for skill in SKILLS}
    library, contracts, train_rows, read_rows = e7v.train_read_map_library(seed, "e7x_pruned_read_tiny_write", baseline_library, context_tasks, read_maps, to_e7v_settings(settings), out)
    training_rows.extend(train_rows)
    pairs = collect_raw_pairs(library, contracts, context_tasks, "train")
    transforms, transform_rows = fit_transforms(pairs)
    rows: list[dict[str, Any]] = []
    scatter_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    failing_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        result, scatter, frames, failing = evaluate_system(seed, system, library, contracts, composition_task, transforms)
        rows.append(result)
        scatter_rows.extend(scatter)
        frame_rows.extend(frames)
        failing_rows.extend(failing)
        for split in EVAL_SPLITS:
            values = [row["written_value"] for row in scatter if row["split"] == split]
            histogram_rows.append({"seed": seed, "system": system, "split": split, "histogram": histogram(values)})
        mean = result["evals"]["heldout"]
        morphology_rows.append({"seed": seed, "system": system, "value_min": mean["value_min"], "value_max": mean["value_max"], "saturation_rate": mean["saturation_rate"], "entropy": mean["entropy"], "effective_value_levels": mean["effective_value_levels"], "noise_floor": mean["noise_floor"], "scale_ratio": mean["scale_ratio"], "bias_offset": mean["bias_offset"]})
        if out:
            append_progress(out, "write_value_system_evaluated", seed=seed, system=system, usefulness=result["eval_mean_composition_usefulness"])
    return {"seed": seed, "rows": rows, "training_rows": training_rows, "read_rows": read_rows, "transform_rows": [{"seed": seed, **row} for row in transform_rows], "morphology_rows": morphology_rows, "histogram_rows": histogram_rows, "scatter_rows": scatter_rows, "frame_rows": frame_rows, "failing_rows": failing_rows}


def build_reports(
    rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    read_rows: list[dict[str, Any]],
    transform_rows: list[dict[str, Any]],
    morphology_rows: list[dict[str, Any]],
    histogram_rows: list[dict[str, Any]],
    scatter_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    failing_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    lines = [
        "# E7X Output Write Value-Format Contrastive Audit Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"best_non_oracle_system = {aggregate['best_non_oracle_system']}",
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
            f"mae={mean.get('eval_mean_absolute_error_to_oracle', 0.0):.6f} "
            f"corr={mean.get('eval_mean_cellwise_correlation_with_oracle', 0.0):.6f} "
            f"next={mean.get('eval_mean_next_pocket_input_compatibility', 0.0):.6f}"
        )
    lines.extend([
        "```",
        "",
        "## Boundary",
        "",
        "E7X is a controlled diagnostic value-format audit. It does not make raw-language, AGI, consciousness, or model-scale claims.",
        "",
    ])
    return {
        "pocket_training_report.json": {"schema_version": "e7x_pocket_training_report_v1", "rows": sorted(training_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "read_map_report.json": {"schema_version": "e7x_read_map_report_v1", "rows": sorted(read_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "write_transform_report.json": {"schema_version": "e7x_write_transform_report_v1", "rows": sorted(transform_rows, key=lambda row: (row["seed"], row["skill"]))},
        "write_morphology_report.json": {"schema_version": "e7x_write_morphology_report_v1", "rows": sorted(morphology_rows, key=lambda row: (row["seed"], row["system"]))},
        "write_histogram_report.json": {"schema_version": "e7x_write_histogram_report_v1", "rows": sorted(histogram_rows, key=lambda row: (row["seed"], row["system"], row["split"]))},
        "oracle_real_scatter_report.json": {"schema_version": "e7x_oracle_real_scatter_report_v1", "rows": sorted(scatter_rows, key=lambda row: (row["seed"], row["system"], row["split"], row["row_id"], row["step"], row["skill"]))},
        "ram_grid_frame_report.json": {"schema_version": "e7x_ram_grid_frame_report_v1", "rows": sorted(frame_rows, key=lambda row: (row["seed"], row["system"], row["split"], row["row_id"], row["step"], row["skill"]))},
        "top_failing_rows_report.json": {"schema_version": "e7x_top_failing_rows_report_v1", "rows": sorted(failing_rows, key=lambda row: (row["seed"], row["system"], row["split"], row["row_id"]))},
        "system_results.json": {"schema_version": "e7x_system_results_v1", "rows": sorted(rows, key=lambda row: (row["seed"], SYSTEMS.index(row["system"])))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7x_summary_v1", "decision": decision["decision"], "best_non_oracle_system": aggregate["best_non_oracle_system"], "deterministic_replay_passed": deterministic.get("internal_replay_passed", False), "checker_failure_count": None},
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
        write_json(out / "backend_manifest.json", {"schema_version": "e7x_backend_manifest_v1", "milestone": MILESTONE, "settings": settings_payload(settings), "systems": list(SYSTEMS), "flow_dim": FLOW_DIM, "semantic_lane_labels_as_model_input": False, "diagnostic_value_format_audit": True, "new_architecture": False, "oracle_used_as_reference_only": True, "training_performed": True, "device": select_device(settings.device), "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        write_json(out / "task_generation_report.json", {"schema_version": "e7x_task_generation_report_v1", "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()}, "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()}})
        rows: list[dict[str, Any]] = []
        training_rows: list[dict[str, Any]] = []
        read_rows: list[dict[str, Any]] = []
        transform_rows: list[dict[str, Any]] = []
        morphology_rows: list[dict[str, Any]] = []
        histogram_rows: list[dict[str, Any]] = []
        scatter_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        failing_rows: list[dict[str, Any]] = []
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
                read_rows.extend(result["read_rows"])
                transform_rows.extend(result["transform_rows"])
                morphology_rows.extend(result["morphology_rows"])
                histogram_rows.extend(result["histogram_rows"])
                scatter_rows.extend(result["scatter_rows"])
                frame_rows.extend(result["frame_rows"])
                failing_rows.extend(result["failing_rows"])
                append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(jobs) - len({row["seed"] for row in rows}))
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_scatter_rows": len(scatter_rows), "pending": len(jobs) - len({row["seed"] for row in rows})})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
                while futures:
                    done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_scatter_rows": len(scatter_rows), "pending": len(futures)})
                        continue
                    for future in done:
                        label = futures.pop(future)
                        result = future.result()
                        rows.extend(result["rows"])
                        training_rows.extend(result["training_rows"])
                        read_rows.extend(result["read_rows"])
                        transform_rows.extend(result["transform_rows"])
                        morphology_rows.extend(result["morphology_rows"])
                        histogram_rows.extend(result["histogram_rows"])
                        scatter_rows.extend(result["scatter_rows"])
                        frame_rows.extend(result["frame_rows"])
                        failing_rows.extend(result["failing_rows"])
                        append_progress(out, "seed_job_complete", label=label, pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_scatter_rows": len(scatter_rows), "last_completed": label, "pending": len(futures)})
        aggregate = aggregate_results(rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e7x_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, training_rows, read_rows, transform_rows, morphology_rows, histogram_rows, scatter_rows, frame_rows, failing_rows, aggregate, decision, deterministic_placeholder)
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
            deterministic = {"schema_version": "e7x_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, training_rows, read_rows, transform_rows, morphology_rows, histogram_rows, scatter_rows, frame_rows, failing_rows, aggregate, decision, deterministic)
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
    parser.add_argument("--pruned-read-count", type=int, default=30)
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
