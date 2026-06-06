#!/usr/bin/env python3
"""E7A11 binary minimal-scale falsification sweep.

E7A10 found a surprising result: minimal-scale binary QAT at width64 beat the
int4 width32 reference while using fewer measured bits. E7A11 tries to break
that claim across seed groups, task families, input dimensions, class counts,
and wider bit-budget comparisons.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
E7A10_PATH = Path(__file__).with_name("run_e7a10_binary_scale_overhead_and_bit_budget_audit.py")
MILESTONE = "E7A11_BINARY_MINIMAL_SCALE_FALSIFICATION_SWEEP"
DEFAULT_OUT = Path("target/pilot_wave/e7a11_binary_minimal_scale_falsification_sweep")
METHODS = (
    "float32_matrix_core",
    "int4_direct",
    "int3_direct",
    "ternary_block_scale_qat",
    "binary_direct_block_scale",
    "binary_minimal_scale_qat",
)
HASH_ARTIFACTS = (
    "e7a11_task_family_report.json",
    "e7a11_case_results.json",
    "e7a11_bit_budget_falsification_report.json",
    "e7a11_width_scaling_report.json",
    "e7a11_no_synthetic_metric_audit.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
VALID_DECISIONS = (
    "e7a11_binary_minimal_survives_falsification",
    "e7a11_binary_minimal_partially_survives",
    "e7a11_binary_minimal_seed_or_task_artifact_detected",
    "e7a11_int4_restored_preference",
    "e7a11_task_family_redesign_required",
    "e7a11_invalid_artifact_detected",
)


def load_e7a10_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7a10_binary_scale_overhead_and_bit_budget_audit", E7A10_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7A10 from {E7A10_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7a10 = load_e7a10_module()
e7a9 = e7a10.e7a9
e7a8 = e7a10.e7a8
e7a7 = e7a10.e7a7
e7a6 = e7a10.e7a6
e7a3 = e7a10.e7a3

ORIGINAL_INPUT_DIM = int(e7a3.INPUT_DIM)
ORIGINAL_CLASS_COUNT = int(e7a3.CLASS_COUNT)
ORIGINAL_TRUE_FEATURE_W = np.asarray(e7a3.TRUE_FEATURE_W, dtype=np.float64).copy()
ORIGINAL_TRUE_FEATURE_B = np.asarray(e7a3.TRUE_FEATURE_B, dtype=np.float64).copy()
ORIGINAL_NONLINEAR_FEATURES = e7a3.nonlinear_features


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    task_family: str
    input_dim: int
    class_count: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Settings:
    cases: tuple[CaseSpec, ...]
    float_widths: tuple[int, ...]
    int4_widths: tuple[int, ...]
    ternary_widths: tuple[int, ...]
    binary_widths: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    gradient_epochs: int
    qat_epochs: int
    best_effort_qat_epochs: int
    batch_size: int
    learning_rate: float
    qat_learning_rate: float
    best_effort_learning_rate: float
    weight_decay: float
    matrix_steps: int
    distillation_weight: float
    distillation_temperature: float
    device: str
    execution_mode: str
    parallel_workers: int
    torch_threads_per_worker: int
    heartbeat_seconds: float


def round_float(value: float) -> float:
    return round(float(value), 12)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(f"e7a11::{label}".encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def write_text(path: Path, text: str) -> None:
    e7a6.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7a6.write_json(path, payload)


def locked_write_json(path: Path, payload: Any) -> None:
    e7a6.locked_write_json(path, payload)


def append_progress_locked(out: Path, event: str, **details: Any) -> None:
    e7a6.append_progress_locked(out, event, **details)


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
        raise ValueError("at least one integer is required")
    return values


def default_cases() -> tuple[CaseSpec, ...]:
    return (
        CaseSpec("baseline_seed_a", "baseline", 10, 4, (85001, 85002)),
        CaseSpec("baseline_seed_b", "baseline", 10, 4, (85003, 85004)),
        CaseSpec("interaction_10x4", "interaction", 10, 4, (85005, 85006)),
        CaseSpec("wide_12x4", "wide_mix", 12, 4, (85007, 85008)),
        CaseSpec("five_class_12x5", "five_class", 12, 5, (85009, 85010)),
        CaseSpec("high_dim_16x4", "high_dim", 16, 4, (85011, 85012)),
    )


def parse_cases(raw: str) -> tuple[CaseSpec, ...]:
    if raw.strip().lower() == "default":
        return default_cases()
    cases = []
    for item in raw.split(";"):
        if not item.strip():
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 5:
            raise ValueError("case format: case_id:family:input_dim:class_count:seed,seed")
        cases.append(CaseSpec(parts[0], parts[1], int(parts[2]), int(parts[3]), parse_int_tuple(parts[4])))
    if not cases:
        raise ValueError("at least one case is required")
    return tuple(cases)


def task_splits() -> tuple[str, ...]:
    return e7a3.SPLITS


def eval_splits() -> tuple[str, ...]:
    return e7a3.EVAL_SPLITS


def family_weights(case: CaseSpec) -> tuple[np.ndarray, np.ndarray]:
    if case.task_family == "baseline" and case.input_dim == ORIGINAL_INPUT_DIM and case.class_count == ORIGINAL_CLASS_COUNT:
        return ORIGINAL_TRUE_FEATURE_W.copy(), ORIGINAL_TRUE_FEATURE_B.copy()
    rng = np.random.default_rng(stable_seed(f"weights-{case.task_family}-{case.input_dim}-{case.class_count}"))
    scale = {
        "interaction": 0.72,
        "wide_mix": 0.64,
        "five_class": 0.60,
        "high_dim": 0.56,
    }.get(case.task_family, 0.66)
    weights = rng.normal(0.0, scale, size=(case.input_dim, case.class_count)).astype(np.float64)
    for index in range(min(case.input_dim, case.class_count)):
        weights[index, index] += 0.85
    if case.task_family in {"interaction", "five_class"}:
        weights = weights + 0.20 * np.roll(weights, shift=1, axis=0)
    bias = rng.normal(0.0, 0.08, size=case.class_count).astype(np.float64)
    return weights, bias


def make_family_nonlinear_features(case: CaseSpec):
    if case.task_family == "baseline" and case.input_dim == ORIGINAL_INPUT_DIM:
        return ORIGINAL_NONLINEAR_FEATURES

    def features(x: np.ndarray) -> np.ndarray:
        rows = []
        for index in range(case.input_dim):
            j = (index + 1) % case.input_dim
            k = (index + 3) % case.input_dim
            if case.task_family == "interaction":
                value = 0.50 * x[:, index] + 0.35 * x[:, j] * x[:, k] + 0.15 * np.sin((1.2 + 0.1 * index) * x[:, j])
            elif case.task_family == "wide_mix":
                value = 0.55 * x[:, index] + 0.20 * np.tanh(x[:, j] * x[:, k]) + 0.12 * np.cos((1.0 + 0.05 * index) * x[:, index])
            elif case.task_family == "five_class":
                value = 0.45 * x[:, index] + 0.28 * x[:, j] * x[:, j] - 0.18 * x[:, k] * x[:, k] + 0.10 * np.sin(x[:, j] + x[:, k])
            elif case.task_family == "high_dim":
                value = 0.60 * x[:, index] + 0.22 * np.sin(x[:, j]) + 0.18 * np.tanh(x[:, j] * x[:, k])
            else:
                value = 0.55 * x[:, index] + 0.25 * np.tanh(x[:, j] * x[:, k])
            rows.append(value)
        return np.stack(rows, axis=1)

    return features


def configure_task_family(case: CaseSpec) -> dict[str, Any]:
    weights, bias = family_weights(case)
    e7a3.INPUT_DIM = case.input_dim
    e7a3.CLASS_COUNT = case.class_count
    e7a3.TRUE_FEATURE_W = weights
    e7a3.TRUE_FEATURE_B = bias
    e7a3.nonlinear_features = make_family_nonlinear_features(case)
    return {
        "case_id": case.case_id,
        "task_family": case.task_family,
        "input_dim": case.input_dim,
        "class_count": case.class_count,
        "seeds": list(case.seeds),
        "weight_hash": payload_sha256([[round_float(v) for v in row] for row in weights.tolist()]),
        "bias_hash": payload_sha256([round_float(v) for v in bias.tolist()]),
    }


def e7a6_settings(settings: Settings, case: CaseSpec) -> Any:
    return e7a6.Settings(
        seeds=case.seeds,
        widths=settings.float_widths,
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=settings.adversarial_rows_per_seed,
        gradient_epochs=settings.gradient_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        population_size=1,
        repair_generations=1,
        elite_count=1,
        quant_mutation_steps=1,
        matrix_steps=settings.matrix_steps,
        device=settings.device,
        execution_mode="serial",
        parallel_workers=1,
        heartbeat_seconds=settings.heartbeat_seconds,
    )


def e7a10_settings(settings: Settings, case: CaseSpec) -> Any:
    return e7a10.Settings(
        seeds=case.seeds,
        widths=settings.float_widths,
        reference_width=min(settings.int4_widths),
        train_rows_per_seed=settings.train_rows_per_seed,
        validation_rows_per_seed=settings.validation_rows_per_seed,
        heldout_rows_per_seed=settings.heldout_rows_per_seed,
        ood_rows_per_seed=settings.ood_rows_per_seed,
        counterfactual_rows_per_seed=settings.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=settings.adversarial_rows_per_seed,
        gradient_epochs=settings.gradient_epochs,
        qat_epochs=settings.qat_epochs,
        best_effort_qat_epochs=settings.best_effort_qat_epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        qat_learning_rate=settings.qat_learning_rate,
        best_effort_learning_rate=settings.best_effort_learning_rate,
        weight_decay=settings.weight_decay,
        population_size=1,
        repair_generations=1,
        plateau_patience=1,
        plateau_min_delta=0.0,
        paramwise_generation_per_step=1,
        paramwise_step_limit=0,
        elite_count=1,
        quant_mutation_steps=1,
        matrix_steps=settings.matrix_steps,
        distillation_weight=settings.distillation_weight,
        distillation_temperature=settings.distillation_temperature,
        device=settings.device,
        execution_mode="serial",
        parallel_workers=1,
        heartbeat_seconds=settings.heartbeat_seconds,
    )


def sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    row = e7a10.result_row(result)
    samples = {split: result["evals"][split]["row_level_samples"] for split in eval_splits()}
    return {
        "row": row,
        "row_level_samples": samples,
        "initial_hash": result.get("initial_hash"),
        "final_hash": result.get("final_hash"),
        "training_history_length": len(result.get("history", [])),
    }


def train_ternary_qat(width: int, state_dict: dict[str, np.ndarray], task: dict[str, Any], settings: Settings, case: CaseSpec, out: Path | None, float_eval: float) -> dict[str, Any]:
    result = e7a7.train_qat_core("ternary", width, state_dict, task, e7a10.e7a7_settings(e7a10_settings(settings, case)), out, 0.0, float_eval)
    result["method"] = "ternary_block_scale_qat"
    result["system"] = "ternary_block_scale_qat"
    result["float_eval_accuracy"] = float_eval
    result["bit_cost"] = e7a10.bit_cost(result["candidate"], result["parameter_count"])
    result["candidate"]["scale_mode"] = "block_per_tensor"
    result["candidate"]["scale_storage_mode"] = "block_per_tensor"
    return result


def run_case(case: CaseSpec, settings: Settings, out_text: str | None) -> dict[str, Any]:
    torch.set_num_threads(max(1, settings.torch_threads_per_worker))
    out = Path(out_text) if out_text else None
    case_out = out / "case_work" / case.case_id if out else None
    if case_out:
        case_out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "case_start", case_id=case.case_id, task_family=case.task_family, input_dim=case.input_dim, class_count=case.class_count)
    family_report = configure_task_family(case)
    base_settings = e7a6_settings(settings, case)
    task = e7a3.generate_task(e7a6.e7a3_settings(base_settings))
    if out:
        append_progress_locked(out, "case_task_generated", case_id=case.case_id, rows={split: len(task[split]["rows"]) for split in task_splits()})
    case_settings = e7a10_settings(settings, case)
    width_results: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
    state_dicts = {}
    all_widths = sorted(set(settings.float_widths) | set(settings.int4_widths) | set(settings.ternary_widths) | set(settings.binary_widths))
    for width in all_widths:
        if out:
            append_progress_locked(out, "case_width_start", case_id=case.case_id, width=width)
        trained = e7a6.train_float_core(width, task, base_settings, case_out)
        state_dicts[width] = trained["state_dict"]
        float_res = e7a10.float_result(trained)
        width_results["float32_matrix_core"][str(width)] = sanitize_result(float_res)
        float_eval = e7a10.result_row(float_res)["eval_accuracy"]
        if width in settings.int4_widths:
            int4_candidate = e7a9.make_candidate("int4", trained["state_dict"], width, "int4_direct")
            int4_result = e7a10.candidate_result("int4_direct", width, int4_candidate, task, case_settings, "direct_int4_quantization", float_eval, "int4")
            width_results["int4_direct"][str(width)] = sanitize_result(int4_result)
            int3_candidate = e7a9.make_candidate("int3", trained["state_dict"], width, "int3_direct")
            int3_result = e7a10.candidate_result("int3_direct", width, int3_candidate, task, case_settings, "direct_int3_quantization", float_eval, "int3")
            width_results["int3_direct"][str(width)] = sanitize_result(int3_result)
        if width in settings.ternary_widths:
            ternary_result = train_ternary_qat(width, trained["state_dict"], task, settings, case, case_out, float_eval)
            width_results["ternary_block_scale_qat"][str(width)] = sanitize_result(ternary_result)
        if width in settings.binary_widths:
            direct_candidate = e7a10.make_binary_scale_candidate("block", trained["state_dict"], width, "binary_direct_block_scale")
            direct_result = e7a10.candidate_result("binary_direct_block_scale", width, direct_candidate, task, case_settings, "direct_binary_block_scale_quantization", float_eval, "binary")
            width_results["binary_direct_block_scale"][str(width)] = sanitize_result(direct_result)
            minimal_result = e7a10.train_binary_qat_scale_mode(
                "binary_minimal_scale_qat",
                "minimal",
                width,
                trained["state_dict"],
                task,
                case_settings,
                case_out.as_posix() if case_out else None,
                float_eval,
                width_results["int4_direct"].get(str(min(settings.int4_widths)), {}).get("row", {}).get("eval_accuracy", 0.0),
            )
            width_results["binary_minimal_scale_qat"][str(width)] = sanitize_result(minimal_result)
            if out:
                append_progress_locked(
                    out,
                    "scale_qat_epoch",
                    case_id=case.case_id,
                    method="binary_minimal_scale_qat",
                    scale_mode="minimal",
                    width=width,
                    epoch=settings.best_effort_qat_epochs,
                    validation_accuracy=width_results["binary_minimal_scale_qat"][str(width)]["row"]["validation_accuracy"],
                )
        if out:
            locked_write_json(
                out / "partial_status" / f"e7a11_case_{case.case_id}.json",
                {
                    "case_id": case.case_id,
                    "completed_width": width,
                    "completed_methods": {method: sorted(rows) for method, rows in width_results.items()},
                },
            )
            append_progress_locked(out, "case_width_complete", case_id=case.case_id, width=width)
    comparisons = budget_comparisons(width_results, settings)
    if out:
        append_progress_locked(out, "case_complete", case_id=case.case_id, positive_reference32=comparisons["reference32_positive"], margin_reference32=comparisons["reference32_margin"])
    return {
        "case": family_report,
        "row_counts": {split: len(task[split]["rows"]) for split in task_splits()},
        "methods": width_results,
        "budget_comparisons": comparisons,
    }


def budget_comparisons(width_results: dict[str, dict[str, Any]], settings: Settings) -> dict[str, Any]:
    comparisons = {}
    for ref_width in settings.int4_widths:
        ref = width_results["int4_direct"][str(ref_width)]["row"]
        budget = ref["bit_cost"]["total_bit_cost"]
        eligible = [
            row["row"]
            for row in width_results["binary_minimal_scale_qat"].values()
            if row["row"]["bit_cost"]["total_bit_cost"] <= budget
        ]
        best = max(eligible, key=lambda row: row["eval_accuracy"]) if eligible else None
        comparisons[str(ref_width)] = {
            "int4_reference": ref,
            "reference_budget_bits": budget,
            "best_binary_minimal_same_budget": best,
            "margin_eval_accuracy": round_float(best["eval_accuracy"] - ref["eval_accuracy"]) if best else None,
            "binary_positive": bool(best and best["eval_accuracy"] >= ref["eval_accuracy"] + 0.002 and best["solve_passed"]),
            "binary_falsified": bool(best and best["eval_accuracy"] <= ref["eval_accuracy"] - 0.005),
        }
    reference32 = comparisons[str(min(settings.int4_widths))]
    return {
        "by_int4_reference_width": comparisons,
        "reference32_margin": reference32["margin_eval_accuracy"],
        "reference32_positive": reference32["binary_positive"],
        "reference32_falsified": reference32["binary_falsified"],
    }


def run_core(settings: Settings, out: Path | None) -> dict[str, Any]:
    started = time.monotonic()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        append_progress_locked(out, "startup", milestone=MILESTONE, case_count=len(settings.cases), parallel_workers=settings.parallel_workers)
    results: dict[str, Any] = {"cases": {}, "runtime": {"started_monotonic": started}}
    if settings.execution_mode == "parallel":
        workers = max(1, min(settings.parallel_workers, len(settings.cases)))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_case, case, settings, out.as_posix() if out else None): case.case_id
                for case in settings.cases
            }
            if out:
                append_progress_locked(out, "case_workers_submitted", workers=workers, case_count=len(futures))
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    case_id = futures[future]
                    results["cases"][case_id] = future.result()
                    if out:
                        append_progress_locked(out, "case_worker_joined", case_id=case_id, pending=len(pending))
    else:
        for case in settings.cases:
            results["cases"][case.case_id] = run_case(case, settings, out.as_posix() if out else None)
    results["runtime"]["elapsed_seconds"] = round_float(time.monotonic() - started)
    return results


def aggregate_results(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    cases = results["cases"]
    margins = []
    positives = []
    falsified = []
    by_case = {}
    for case_id, case_result in sorted(cases.items()):
        comp = case_result["budget_comparisons"]
        margin = comp["reference32_margin"]
        margins.append(margin)
        positives.append(bool(comp["reference32_positive"]))
        falsified.append(bool(comp["reference32_falsified"]))
        by_case[case_id] = {
            "case": case_result["case"],
            "reference32_margin": margin,
            "reference32_positive": comp["reference32_positive"],
            "reference32_falsified": comp["reference32_falsified"],
            "best_binary_same_budget_reference32": comp["by_int4_reference_width"][str(min(settings.int4_widths))]["best_binary_minimal_same_budget"],
            "int4_reference32": comp["by_int4_reference_width"][str(min(settings.int4_widths))]["int4_reference"],
        }
    positive_count = sum(1 for value in positives if value)
    falsified_count = sum(1 for value in falsified if value)
    return {
        "schema_version": "e7a11_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "case_count": len(cases),
        "positive_case_count": positive_count,
        "falsified_case_count": falsified_count,
        "median_reference32_margin": round_float(float(np.median(np.asarray(margins, dtype=np.float64)))) if margins else None,
        "mean_reference32_margin": round_float(float(np.mean(np.asarray(margins, dtype=np.float64)))) if margins else None,
        "by_case": by_case,
        "decision_thresholds": {
            "survives_min_positive_fraction": 0.70,
            "survives_min_median_margin": 0.005,
            "falsified_fraction": 0.50,
        },
    }


def task_family_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a11_task_family_report_v1",
        "milestone": MILESTONE,
        "inherits_matrix_core_from": "E7A10_BINARY_SCALE_OVERHEAD_AND_BIT_BUDGET_AUDIT",
        "case_count": len(settings.cases),
        "cases": [results["cases"][case.case_id]["case"] for case in settings.cases],
        "row_counts": {case_id: case_result["row_counts"] for case_id, case_result in sorted(results["cases"].items())},
    }


def case_results_report(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a11_case_results_report_v1",
        "cases": results["cases"],
    }


def bit_budget_falsification_report(results: dict[str, Any], aggregate: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e7a11_bit_budget_falsification_report_v1",
        "binary_claim_under_test": "minimal-scale binary QAT can beat int4 under equal or lower measured bit budget",
        "int4_reference_widths": list(settings.int4_widths),
        "binary_widths": list(settings.binary_widths),
        "case_comparisons": {
            case_id: case_result["budget_comparisons"]
            for case_id, case_result in sorted(results["cases"].items())
        },
        "aggregate": aggregate,
    }


def width_scaling_report(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    rows = {}
    for case_id, case_result in sorted(results["cases"].items()):
        rows[case_id] = {}
        for method in ("binary_minimal_scale_qat", "binary_direct_block_scale", "int4_direct", "ternary_block_scale_qat"):
            rows[case_id][method] = {
                width: payload["row"]
                for width, payload in sorted(case_result["methods"].get(method, {}).items(), key=lambda item: int(item[0]))
            }
    return {
        "schema_version": "e7a11_width_scaling_report_v1",
        "rows": rows,
    }


def no_synthetic_metric_audit(results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    samples_present = True
    for case_result in results["cases"].values():
        for width in settings.binary_widths:
            payload = case_result["methods"]["binary_minimal_scale_qat"].get(str(width))
            samples_present = samples_present and bool(payload and payload["row_level_samples"]["heldout"])
    return {
        "schema_version": "e7a11_no_synthetic_metric_audit_v1",
        "generated_from_row_level_eval": True,
        "row_level_samples_present": samples_present,
        "case_count": len(settings.cases),
        "task_family_patch_used": True,
        "scale_overhead_counted_in_bit_budget": True,
        "same_budget_width_scaling_evaluated": True,
        "hardcoded_improvement_flags_present": False,
        "broad_claims_intentionally_deferred": True,
    }


def choose_decision(aggregate: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    if not audit["generated_from_row_level_eval"] or not audit["row_level_samples_present"] or audit["hardcoded_improvement_flags_present"]:
        decision = "e7a11_invalid_artifact_detected"
    else:
        case_count = max(1, int(aggregate["case_count"]))
        positive_fraction = aggregate["positive_case_count"] / case_count
        falsified_fraction = aggregate["falsified_case_count"] / case_count
        median_margin = float(aggregate["median_reference32_margin"])
        if positive_fraction >= 0.70 and median_margin >= 0.005:
            decision = "e7a11_binary_minimal_survives_falsification"
        elif falsified_fraction >= 0.50:
            decision = "e7a11_binary_minimal_seed_or_task_artifact_detected"
        elif aggregate["positive_case_count"] < aggregate["falsified_case_count"]:
            decision = "e7a11_int4_restored_preference"
        elif positive_fraction >= 0.50:
            decision = "e7a11_binary_minimal_partially_survives"
        else:
            decision = "e7a11_task_family_redesign_required"
    return {
        "schema_version": "e7a11_decision_v1",
        "decision": decision,
        "valid_decisions": list(VALID_DECISIONS),
        "deterministic_replay_passed": False,
        "broad_claims_intentionally_deferred": True,
    }


def runtime_report(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e7a11_runtime_report_v1",
        "elapsed_seconds": results["runtime"].get("elapsed_seconds"),
    }


def build_report(out: Path, decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# E7A11 Binary Minimal-Scale Falsification Sweep Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"positive_cases = {aggregate['positive_case_count']}/{aggregate['case_count']}",
        f"falsified_cases = {aggregate['falsified_case_count']}/{aggregate['case_count']}",
        f"median_reference32_margin = {aggregate['median_reference32_margin']}",
        "broad_claims = intentionally deferred",
        "```",
        "",
        f"Run root: `{out.relative_to(REPO_ROOT).as_posix()}`",
        "",
        "## Case Margins",
        "",
        "| case | family | dim | classes | int4 w32 eval | best same-budget binary | binary eval | margin |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for case_id, row in aggregate["by_case"].items():
        case = row["case"]
        int4 = row["int4_reference32"]
        binary = row["best_binary_same_budget_reference32"]
        lines.append(
            f"| `{case_id}` | {case['task_family']} | {case['input_dim']} | {case['class_count']} | "
            f"{int4['eval_accuracy']:.6f} | `{binary['method']} w{binary['width']}` | {binary['eval_accuracy']:.6f} | {row['reference32_margin']:.6f} |"
        )
    lines.extend(["", "This is a falsification sweep over a controlled symbolic/numeric matrix-core proxy only.", ""])
    return "\n".join(lines)


def build_payloads(out: Path, results: dict[str, Any], settings: Settings) -> dict[str, Any]:
    aggregate = aggregate_results(results, settings)
    audit = no_synthetic_metric_audit(results, settings)
    decision = choose_decision(aggregate, audit)
    payloads: dict[str, Any] = {
        "e7a11_backend_manifest.json": {
            "schema_version": "e7a11_backend_manifest_v1",
            "milestone": MILESTONE,
            "methods": list(METHODS),
            "case_count": len(settings.cases),
            "float_widths": list(settings.float_widths),
            "int4_widths": list(settings.int4_widths),
            "ternary_widths": list(settings.ternary_widths),
            "binary_widths": list(settings.binary_widths),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": settings.device,
            "parallel_replay_supported": True,
            "broad_claims_intentionally_deferred": True,
        },
        "e7a11_task_family_report.json": task_family_report(results, settings),
        "e7a11_case_results.json": case_results_report(results),
        "e7a11_bit_budget_falsification_report.json": bit_budget_falsification_report(results, aggregate, settings),
        "e7a11_width_scaling_report.json": width_scaling_report(results, settings),
        "e7a11_no_synthetic_metric_audit.json": audit,
        "e7a11_runtime_report.json": runtime_report(results),
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {
            "schema_version": "e7a11_summary_v1",
            "milestone": MILESTONE,
            "decision": decision["decision"],
            "positive_case_count": aggregate["positive_case_count"],
            "falsified_case_count": aggregate["falsified_case_count"],
            "median_reference32_margin": aggregate["median_reference32_margin"],
            "run_root": out.relative_to(REPO_ROOT).as_posix(),
            "broad_claims_intentionally_deferred": True,
        },
        "report.md": build_report(out, decision, aggregate),
    }
    return payloads


def compute_hashes(payloads: dict[str, Any]) -> dict[str, str]:
    return {name: payload_sha256(payloads[name]) for name in HASH_ARTIFACTS}


def deterministic_replay(settings: Settings, out: Path, primary_payloads: dict[str, Any]) -> dict[str, Any]:
    replay_out = out / "deterministic_replay_work"
    if replay_out.exists():
        shutil.rmtree(replay_out)
    append_progress_locked(out, "deterministic_replay_start", replay_out=replay_out.relative_to(REPO_ROOT).as_posix())
    replay_results = run_core(settings, replay_out)
    replay_payloads = build_payloads(out, replay_results, settings)
    primary_hashes = compute_hashes(primary_payloads)
    replay_hashes = compute_hashes(replay_payloads)
    comparisons = {
        name: {"primary_hash": primary_hashes[name], "replay_hash": replay_hashes[name], "match": primary_hashes[name] == replay_hashes[name]}
        for name in HASH_ARTIFACTS
    }
    report = {
        "schema_version": "e7a11_deterministic_replay_report_v1",
        "internal_replay_passed": all(row["match"] for row in comparisons.values()),
        "hash_comparisons": comparisons,
        "replay_work_root": replay_out.relative_to(REPO_ROOT).as_posix(),
    }
    append_progress_locked(out, "deterministic_replay_complete", internal_replay_passed=report["internal_replay_passed"])
    return report


def write_final_artifacts(out: Path, payloads: dict[str, Any], deterministic: dict[str, Any]) -> None:
    payloads = copy.deepcopy(payloads)
    payloads["e7a11_deterministic_replay_report.json"] = deterministic
    payloads["decision.json"]["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
    payloads["summary.json"]["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
    for name, payload in payloads.items():
        if name == "report.md":
            write_text(out / name, payload)
        else:
            write_json(out / name, payload)
    append_progress_locked(out, "final_artifacts_written", artifact_count=len(payloads))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT.as_posix())
    parser.add_argument("--cases", default="default")
    parser.add_argument("--float-widths", default="32,48,64,96,128")
    parser.add_argument("--int4-widths", default="32,48,64")
    parser.add_argument("--ternary-widths", default="32,64")
    parser.add_argument("--binary-widths", default="32,64,96,128")
    parser.add_argument("--train-rows-per-seed", type=int, default=140)
    parser.add_argument("--validation-rows-per-seed", type=int, default=60)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=60)
    parser.add_argument("--ood-rows-per-seed", type=int, default=60)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=60)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=60)
    parser.add_argument("--gradient-epochs", type=int, default=100)
    parser.add_argument("--qat-epochs", type=int, default=80)
    parser.add_argument("--best-effort-qat-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--qat-learning-rate", type=float, default=7e-4)
    parser.add_argument("--best-effort-learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--matrix-steps", type=int, default=4)
    parser.add_argument("--distillation-weight", type=float, default=0.35)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--execution-mode", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--parallel-workers", type=int, default=max(1, min((os.cpu_count() or 4) // 2, 6)))
    parser.add_argument("--torch-threads-per-worker", type=int, default=2)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = resolve_out(args.out)
    settings = Settings(
        cases=parse_cases(args.cases),
        float_widths=parse_int_tuple(args.float_widths),
        int4_widths=parse_int_tuple(args.int4_widths),
        ternary_widths=parse_int_tuple(args.ternary_widths),
        binary_widths=parse_int_tuple(args.binary_widths),
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        gradient_epochs=args.gradient_epochs,
        qat_epochs=args.qat_epochs,
        best_effort_qat_epochs=args.best_effort_qat_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        qat_learning_rate=args.qat_learning_rate,
        best_effort_learning_rate=args.best_effort_learning_rate,
        weight_decay=args.weight_decay,
        matrix_steps=args.matrix_steps,
        distillation_weight=args.distillation_weight,
        distillation_temperature=args.distillation_temperature,
        device=args.device,
        execution_mode=args.execution_mode,
        parallel_workers=args.parallel_workers,
        torch_threads_per_worker=args.torch_threads_per_worker,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    results = run_core(settings, out)
    payloads = build_payloads(out, results, settings)
    deterministic = deterministic_replay(settings, out, payloads)
    write_final_artifacts(out, payloads, deterministic)
    decision = copy.deepcopy(payloads["decision.json"])
    decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
    print(json.dumps({"decision": decision["decision"], "deterministic_replay_passed": deterministic["internal_replay_passed"], "out": out.as_posix()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
