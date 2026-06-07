#!/usr/bin/env python3
"""E7W numeric pocket composition failure localization.

E7W does not add an architecture. It takes the current typed pocket-flow setup
and applies diagnostic interventions to localize why composed numeric pockets
remain far below the oracle even when routing, standalone pocket training, and
write-side RAM policy are mostly under control.
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
E7V_PATH = Path(__file__).with_name("run_e7v_ram_read_context_selection_audit.py")
MILESTONE = "E7W_NUMERIC_POCKET_COMPOSITION_FAILURE_LOCALIZATION"
DEFAULT_OUT = Path("target/pilot_wave/e7w_numeric_pocket_composition_failure_localization")
DEFAULT_SEEDS = (100101, 100102, 100103, 100104, 100105, 100106, 100107, 100108)

SYSTEMS = (
    "baseline_best_current",
    "oracle_route_only",
    "oracle_intermediate_state_after_each_pocket",
    "one_real_pocket_at_a_time",
    "oracle_read_map_real_write",
    "real_read_map_oracle_write",
    "output_calibration_bridge",
    "residual_delta_integration",
    "broad_read_tiny_write_reference",
    "pruned_read_tiny_write_reference",
)
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "localization_report.json",
    "one_pocket_attribution_report.json",
    "calibration_report.json",
    "step_drift_report.json",
    "system_results.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e7w_intermediate_state_drift_bottleneck",
    "e7w_specific_pocket_bottleneck_detected",
    "e7w_read_context_bottleneck",
    "e7w_output_write_contract_bottleneck",
    "e7w_output_calibration_bottleneck",
    "e7w_flow_integration_bottleneck",
    "e7w_composition_failure_unlocalized",
)


def load_e7v_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7v_ram_read_context_selection_audit", E7V_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7V helpers from {E7V_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7v = load_e7v_module()
e7u = e7v.e7u
e7r = e7v.e7r
e7p = e7v.e7p
e7o = e7v.e7o

FLOW_DIM = int(e7v.FLOW_DIM)
SKILLS = tuple(e7v.SKILLS)
SPLITS = tuple(e7v.SPLITS)
EVAL_SPLITS = tuple(e7v.EVAL_SPLITS)
RESULT_POS = dict(e7v.RESULT_POS)


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
    return e7v.round_float(value)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_text(path: Path, text: str) -> None:
    e7v.write_text(path, text)


def write_json(path: Path, payload: Any) -> None:
    e7v.write_json(path, payload)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7v.append_progress(out, event, **details)


def resolve_out(path: str | Path) -> Path:
    return e7v.resolve_out(path)


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return e7v.parse_int_tuple(raw)


def select_device(requested: str) -> str:
    return e7v.select_device(requested)


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    payload["replay"] = False
    return payload


def to_e7v_settings(settings: Settings) -> Any:
    return e7v.Settings(
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
        mutation_generations=1,
        mutation_population=1,
        read_budgets=(settings.pruned_read_count,),
        fixed_read_count=settings.pruned_read_count,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        execution_mode=settings.execution_mode,
        replay=settings.replay,
    )


def to_e7o_settings(settings: Settings) -> Any:
    return e7v.to_e7o_settings(to_e7v_settings(settings))


def copy_flow(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(row["flow"], dtype=np.float32).copy()


def canonical_step(row: dict[str, Any], flow: np.ndarray, skill: str) -> np.ndarray:
    out = flow.copy()
    out[RESULT_POS[skill]] = float(e7o.base_skill_value(skill, row["a"], row["b"], row["key"], row["threshold"], row["flip"], out))
    return out.astype(np.float32)


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


def train_output_calibration(
    library: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    context_tasks: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    thresholds: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for skill in SKILLS:
        contract = contracts[skill]
        vals: list[float] = []
        targets: list[int] = []
        for row in context_tasks[skill]["validation"]:
            flow = np.asarray(row["flow"], dtype=np.float32).reshape(1, -1)
            pred = e7r.masked_forward_np(library[skill], flow, contract)
            vals.append(float(pred[0, contract["assignment_cell"]]))
            targets.append(int(row["target_value"]))
        candidates = sorted(set(vals)) or [0.0]
        candidates = [candidates[0] - 1.0] + candidates + [candidates[-1] + 1.0, 0.0]
        best_threshold = 0.0
        best_acc = -1.0
        for threshold in candidates:
            pred = [1 if value >= threshold else 0 for value in vals]
            acc = float(np.mean([p == t for p, t in zip(pred, targets)])) if targets else 1.0
            if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and abs(threshold) < abs(best_threshold)):
                best_acc = acc
                best_threshold = float(threshold)
        thresholds[skill] = round_float(best_threshold)
        rows.append({"skill": skill, "threshold": round_float(best_threshold), "validation_accuracy": round_float(best_acc), "sample_count": len(targets)})
    return thresholds, rows


def bit_cost(library: dict[str, dict[str, Any]] | None) -> int:
    return sum(e7p.bit_budget(state) for state in library.values()) if library else 0


def evaluate_variant(
    seed: int,
    system: str,
    library: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    task: dict[str, list[dict[str, Any]]],
    calibration: dict[str, float],
    mode: str,
    one_real_skill: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evals: dict[str, Any] = {}
    step_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        correct: list[bool] = []
        samples: list[dict[str, Any]] = []
        ram_error: list[float] = []
        output_error: list[float] = []
        read_error: list[float] = []
        write_error: list[float] = []
        compatibility: list[float] = []
        step_index_errors: dict[int, list[float]] = {}
        for row in task[split]:
            route = tuple(row["expected_route"])
            flow = copy_flow(row)
            canonical = copy_flow(row)
            route_step_details = []
            for step_idx, skill in enumerate(route):
                before = flow.copy()
                canonical_before = canonical.copy()
                canonical_after = canonical_step(row, canonical_before, skill)
                contract = contracts[skill]
                use_oracle_for_step = mode == "one_real" and skill != one_real_skill
                if use_oracle_for_step:
                    flow = canonical_after.copy()
                    written = float(flow[RESULT_POS[skill]])
                else:
                    pocket_input = flow.copy()
                    if mode == "oracle_read":
                        pocket_input[contract["read"]] = canonical_before[contract["read"]]
                    pred = e7r.masked_forward_np(library[skill], pocket_input.reshape(1, -1), contract).reshape(-1)
                    raw = float(pred[contract["assignment_cell"]])
                    if mode == "oracle_write":
                        written = float(canonical_after[RESULT_POS[skill]])
                    elif mode == "calibration":
                        written = 1.0 if raw >= calibration.get(skill, 0.0) else 0.0
                    elif mode == "residual":
                        prob = sigmoid(raw - calibration.get(skill, 0.0))
                        written = float(before[RESULT_POS[skill]] + 0.85 * (prob - before[RESULT_POS[skill]]))
                    else:
                        written = 1.0 if raw >= 0.0 else 0.0
                    flow = pred.astype(np.float32)
                    flow[RESULT_POS[skill]] = np.float32(written)
                if mode == "oracle_intermediate":
                    flow = canonical_after.copy()
                canonical = canonical_after
                step_ram_error = float(np.mean(np.abs(flow - canonical_after)))
                step_output_error = abs(float(written) - float(canonical_after[RESULT_POS[skill]]))
                next_read_error = 0.0
                if step_idx + 1 < len(route):
                    next_contract = contracts[route[step_idx + 1]]
                    next_read_error = float(np.mean(np.abs(flow[next_contract["read"]] - canonical_after[next_contract["read"]])))
                    compatibility.append(next_read_error)
                ram_error.append(step_ram_error)
                output_error.append(step_output_error)
                write_error.append(step_output_error)
                read_error.append(next_read_error)
                step_index_errors.setdefault(step_idx + 1, []).append(step_ram_error)
                route_step_details.append({"skill": skill, "step": step_idx + 1, "ram_error": round_float(step_ram_error), "output_error": round_float(step_output_error), "next_read_error": round_float(next_read_error)})
                if len(step_rows) < 400:
                    step_rows.append({"seed": seed, "system": system, "split": split, "row_id": row["row_id"], "skill": skill, "step": step_idx + 1, "ram_error": round_float(step_ram_error), "output_calibration_error": round_float(step_output_error), "next_pocket_input_compatibility": round_float(next_read_error)})
            pred_answer = int(e7o.predict_answer_from_flow(row, flow))
            ok = pred_answer == int(row["target_answer"])
            correct.append(ok)
            if len(samples) < 8:
                samples.append({"row_id": row["row_id"], "family": row["family"], "route": list(route), "target": int(row["target_answer"]), "predicted": pred_answer, "correct": bool(ok), "steps": route_step_details})
        acc = round_float(float(np.mean(correct)))
        mean_steps = round_float(float(np.mean([len(row["expected_route"]) for row in task[split]])))
        cost_penalty = min(0.10, 0.00000016 * bit_cost(library) + 0.0025 * mean_steps)
        evals[split] = {
            "answer_accuracy": acc,
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(max(0.0, acc - cost_penalty)),
            "mean_route_steps": mean_steps,
            "per_step_ram_error": round_float(float(np.mean(ram_error)) if ram_error else 0.0),
            "intermediate_state_drift": round_float(float(np.mean(ram_error)) if ram_error else 0.0),
            "output_calibration_error": round_float(float(np.mean(output_error)) if output_error else 0.0),
            "next_pocket_input_compatibility": round_float(float(np.mean(compatibility)) if compatibility else 0.0),
            "read_context_error": round_float(float(np.mean(read_error)) if read_error else 0.0),
            "write_placement_error": round_float(float(np.mean(write_error)) if write_error else 0.0),
            "step_error_by_index": {str(idx): round_float(float(np.mean(values))) for idx, values in sorted(step_index_errors.items())},
            "bit_budget": bit_cost(library),
            "row_level_samples": samples,
        }
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
        "eval_mean_per_step_ram_error": round_float(float(np.mean([evals[split]["per_step_ram_error"] for split in EVAL_SPLITS]))),
        "eval_mean_intermediate_state_drift": round_float(float(np.mean([evals[split]["intermediate_state_drift"] for split in EVAL_SPLITS]))),
        "eval_mean_output_calibration_error": round_float(float(np.mean([evals[split]["output_calibration_error"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_input_compatibility": round_float(float(np.mean([evals[split]["next_pocket_input_compatibility"] for split in EVAL_SPLITS]))),
        "eval_mean_read_context_error": round_float(float(np.mean([evals[split]["read_context_error"] for split in EVAL_SPLITS]))),
        "eval_mean_write_placement_error": round_float(float(np.mean([evals[split]["write_placement_error"] for split in EVAL_SPLITS]))),
        "parameter_count": sum(e7p.parameter_count(state) for state in library.values()),
        "bit_budget": bit_cost(library),
    }
    if one_real_skill is not None:
        row["one_real_skill"] = one_real_skill
    return row, step_rows


def aggregate_one_real(seed: int, skill_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, list[float]] = {}
    for row in skill_rows:
        for key, value in row.items():
            if key.startswith("eval_mean_") and isinstance(value, (int, float)):
                metrics.setdefault(key, []).append(float(value))
        for key in ("heldout_usefulness", "ood_usefulness", "counterfactual_usefulness", "adversarial_usefulness"):
            metrics.setdefault(key, []).append(float(row[key]))
    aggregate = {
        "seed": seed,
        "system": "one_real_pocket_at_a_time",
        "evals": skill_rows[0]["evals"],
        "skill_results": [{key: value for key, value in row.items() if key not in {"evals"}} for row in skill_rows],
    }
    for key, values in metrics.items():
        aggregate[key] = round_float(float(np.mean(values)))
    aggregate["weakest_skill"] = min(skill_rows, key=lambda row: row["eval_mean_composition_usefulness"])["one_real_skill"]
    aggregate["strongest_skill"] = max(skill_rows, key=lambda row: row["eval_mean_composition_usefulness"])["one_real_skill"]
    return aggregate


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
        "eval_mean_per_step_ram_error",
        "eval_mean_intermediate_state_drift",
        "eval_mean_output_calibration_error",
        "eval_mean_next_pocket_input_compatibility",
        "eval_mean_read_context_error",
        "eval_mean_write_placement_error",
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
    best = max(SYSTEMS, key=lambda system: systems[system]["mean"].get("eval_mean_composition_usefulness", 0.0))
    return {"schema_version": "e7w_aggregate_metrics_v1", "systems": systems, "best_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    mean = {system: aggregate["systems"][system]["mean"].get("eval_mean_composition_usefulness", 0.0) for system in SYSTEMS}
    baseline = mean["baseline_best_current"]
    detail = {
        "baseline": baseline,
        "oracle_route_only": mean["oracle_route_only"],
        "oracle_intermediate": mean["oracle_intermediate_state_after_each_pocket"],
        "one_real": mean["one_real_pocket_at_a_time"],
        "oracle_read": mean["oracle_read_map_real_write"],
        "oracle_write": mean["real_read_map_oracle_write"],
        "calibration": mean["output_calibration_bridge"],
        "residual": mean["residual_delta_integration"],
        "broad": mean["broad_read_tiny_write_reference"],
        "pruned": mean["pruned_read_tiny_write_reference"],
        "best_system": aggregate["best_system"],
    }
    if detail["oracle_write"] >= baseline + 0.10 and detail["oracle_write"] >= detail["oracle_intermediate"] - 0.01:
        decision = "e7w_output_write_contract_bottleneck"
    elif detail["oracle_intermediate"] >= baseline + 0.10:
        decision = "e7w_intermediate_state_drift_bottleneck"
    elif detail["one_real"] <= baseline - 0.08:
        decision = "e7w_specific_pocket_bottleneck_detected"
    elif detail["oracle_read"] >= baseline + 0.05:
        decision = "e7w_read_context_bottleneck"
    elif detail["oracle_write"] >= baseline + 0.05:
        decision = "e7w_output_write_contract_bottleneck"
    elif detail["calibration"] >= baseline + 0.025:
        decision = "e7w_output_calibration_bottleneck"
    elif detail["residual"] >= baseline + 0.025:
        decision = "e7w_flow_integration_bottleneck"
    else:
        decision = "e7w_composition_failure_unlocalized"
    return {"schema_version": "e7w_decision_v1", "decision": decision, "detail": detail, "deterministic_replay_passed": False}


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
    rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        state = e7p.copy_state(trained["state"], "e7w_baseline_standalone")
        baseline_library[skill] = state
        training_rows.append({"seed": seed, "system": "baseline_standalone_pocket", "skill": skill, "state_hash": e7p.state_hash(state), "standalone": trained["standalone"]})

    broad_maps = e7v.broad_read_maps()
    pruned_maps = {skill: e7v.priority_read_map(skill, settings.pruned_read_count) for skill in SKILLS}
    broad_library, broad_contracts, broad_train, _ = e7v.train_read_map_library(seed, "e7w_broad_read_tiny_write_reference", baseline_library, context_tasks, broad_maps, to_e7v_settings(settings), out)
    pruned_library, pruned_contracts, pruned_train, _ = e7v.train_read_map_library(seed, "e7w_pruned_read_tiny_write_reference", baseline_library, context_tasks, pruned_maps, to_e7v_settings(settings), out)
    training_rows.extend(broad_train)
    training_rows.extend(pruned_train)
    calibration, cal_rows = train_output_calibration(pruned_library, pruned_contracts, context_tasks)
    calibration_rows.extend({"seed": seed, **row} for row in cal_rows)

    variants = [
        ("baseline_best_current", pruned_library, pruned_contracts, "hard"),
        ("oracle_route_only", pruned_library, pruned_contracts, "hard"),
        ("oracle_intermediate_state_after_each_pocket", pruned_library, pruned_contracts, "oracle_intermediate"),
        ("oracle_read_map_real_write", pruned_library, pruned_contracts, "oracle_read"),
        ("real_read_map_oracle_write", pruned_library, pruned_contracts, "oracle_write"),
        ("output_calibration_bridge", pruned_library, pruned_contracts, "calibration"),
        ("residual_delta_integration", pruned_library, pruned_contracts, "residual"),
        ("broad_read_tiny_write_reference", broad_library, broad_contracts, "hard"),
        ("pruned_read_tiny_write_reference", pruned_library, pruned_contracts, "hard"),
    ]
    for system, library, contracts, mode in variants:
        row, steps = evaluate_variant(seed, system, library, contracts, composition_task, calibration, mode)
        rows.append(row)
        step_rows.extend(steps)
        if out:
            append_progress(out, "localization_system_evaluated", seed=seed, system=system, usefulness=row["eval_mean_composition_usefulness"])

    one_skill_rows = []
    for skill in SKILLS:
        row, steps = evaluate_variant(seed, f"one_real_{skill}", pruned_library, pruned_contracts, composition_task, calibration, "one_real", one_real_skill=skill)
        one_skill_rows.append(row)
        step_rows.extend(steps)
        attribution_rows.append({"seed": seed, "skill": skill, "usefulness": row["eval_mean_composition_usefulness"], "answer_accuracy": row["eval_mean_answer_accuracy"], "ram_error": row["eval_mean_per_step_ram_error"], "output_error": row["eval_mean_output_calibration_error"]})
    rows.append(aggregate_one_real(seed, one_skill_rows))
    return {"seed": seed, "rows": rows, "training_rows": training_rows, "step_rows": step_rows, "attribution_rows": attribution_rows, "calibration_rows": calibration_rows}


def build_reports(
    rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    lines = [
        "# E7W Numeric Pocket Composition Failure Localization Result",
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
            f"ram_err={mean.get('eval_mean_per_step_ram_error', 0.0):.6f} "
            f"out_err={mean.get('eval_mean_output_calibration_error', 0.0):.6f} "
            f"next={mean.get('eval_mean_next_pocket_input_compatibility', 0.0):.6f}"
        )
    lines.extend([
        "```",
        "",
        "## Boundary",
        "",
        "E7W is a controlled numeric pocket composition localization probe. It does not make raw-language, AGI, consciousness, or model-scale claims.",
        "",
    ])
    localization_rows = []
    for system in SYSTEMS:
        mean = aggregate["systems"][system]["mean"]
        localization_rows.append({"system": system, "usefulness": mean.get("eval_mean_composition_usefulness", 0.0), "answer_accuracy": mean.get("eval_mean_answer_accuracy", 0.0), "ram_error": mean.get("eval_mean_per_step_ram_error", 0.0), "output_calibration_error": mean.get("eval_mean_output_calibration_error", 0.0), "next_pocket_input_compatibility": mean.get("eval_mean_next_pocket_input_compatibility", 0.0)})
    return {
        "pocket_training_report.json": {"schema_version": "e7w_pocket_training_report_v1", "rows": sorted(training_rows, key=lambda row: (row["seed"], row["system"], row["skill"]))},
        "localization_report.json": {"schema_version": "e7w_localization_report_v1", "rows": localization_rows},
        "one_pocket_attribution_report.json": {"schema_version": "e7w_one_pocket_attribution_report_v1", "rows": sorted(attribution_rows, key=lambda row: (row["seed"], row["skill"]))},
        "calibration_report.json": {"schema_version": "e7w_calibration_report_v1", "rows": sorted(calibration_rows, key=lambda row: (row["seed"], row["skill"]))},
        "step_drift_report.json": {"schema_version": "e7w_step_drift_report_v1", "rows": sorted(step_rows, key=lambda row: (row["seed"], row["system"], row["split"], row["row_id"], row["step"], row["skill"]))},
        "system_results.json": {"schema_version": "e7w_system_results_v1", "rows": sorted(rows, key=lambda row: (row["seed"], SYSTEMS.index(row["system"])))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e7w_summary_v1", "decision": decision["decision"], "best_system": aggregate["best_system"], "deterministic_replay_passed": deterministic.get("internal_replay_passed", False), "checker_failure_count": None},
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
        write_json(out / "backend_manifest.json", {"schema_version": "e7w_backend_manifest_v1", "milestone": MILESTONE, "settings": settings_payload(settings), "systems": list(SYSTEMS), "flow_dim": FLOW_DIM, "semantic_lane_labels_as_model_input": False, "diagnostic_interventions": True, "new_architecture": False, "training_performed": True, "device": select_device(settings.device), "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        write_json(out / "task_generation_report.json", {"schema_version": "e7w_task_generation_report_v1", "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()}, "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()}})
        rows: list[dict[str, Any]] = []
        training_rows: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        attribution_rows: list[dict[str, Any]] = []
        calibration_rows: list[dict[str, Any]] = []
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
                step_rows.extend(result["step_rows"])
                attribution_rows.extend(result["attribution_rows"])
                calibration_rows.extend(result["calibration_rows"])
                append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(jobs) - len({row["seed"] for row in rows}))
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_step_rows": len(step_rows), "pending": len(jobs) - len({row["seed"] for row in rows})})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(seed_worker, job): f"seed{job['seed']}" for job in jobs}
                while futures:
                    done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_step_rows": len(step_rows), "pending": len(futures)})
                        continue
                    for future in done:
                        label = futures.pop(future)
                        result = future.result()
                        rows.extend(result["rows"])
                        training_rows.extend(result["training_rows"])
                        step_rows.extend(result["step_rows"])
                        attribution_rows.extend(result["attribution_rows"])
                        calibration_rows.extend(result["calibration_rows"])
                        append_progress(out, "seed_job_complete", label=label, pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(training_rows), "completed_step_rows": len(step_rows), "last_completed": label, "pending": len(futures)})
        aggregate = aggregate_results(rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e7w_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, training_rows, step_rows, attribution_rows, calibration_rows, aggregate, decision, deterministic_placeholder)
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
            deterministic = {"schema_version": "e7w_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, training_rows, step_rows, attribution_rows, calibration_rows, aggregate, decision, deterministic)
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
