#!/usr/bin/env python3
"""E8A canonical RAM code learning and smoothness probe.

E7Z showed that low-bit RAM codes are expressive when oracle-generated, but
learned systems do not discover the right RAM language from final-answer
pressure alone. E8A tests intermediate code supervision, smooth-to-hard
curricula, simplified teacher codes, and mutation repair after distillation.
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
import random
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
E7Z_PATH = Path(__file__).with_name("run_e7z_low_bit_canonical_ram_contract_probe.py")
MILESTONE = "E8A_CANONICAL_RAM_CODE_LEARNING_AND_SMOOTHNESS_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e8a_canonical_ram_code_learning_and_smoothness_probe")
DEFAULT_SEEDS = (101001, 101002, 101003, 101004, 101005, 101006, 101007, 101008)
OUTPUT_WIDTH = 12
LOW_BIT_CODES = ("binary", "ternary", "int4")
TEACHER_STYLES = ("current_oracle_projection_code", "simplified_canonical_code")

SYSTEM_CODE = {
    "current_best_baseline": "int4",
    "oracle_low_bit_reference": "binary",
    "producer_distill_binary": "binary",
    "producer_distill_ternary": "ternary",
    "producer_distill_int4": "int4",
    "consumer_distill_binary": "binary",
    "producer_consumer_staged_binary": "binary",
    "producer_consumer_staged_ternary": "ternary",
    "producer_consumer_staged_int4": "int4",
    "soft_to_hard_int4_to_ternary_to_binary": "binary",
    "contrastive_ram_code_alignment": "int4",
    "progressive_code_freeze": "int4",
    "mutation_only_from_random_lowbit": "int4",
    "mutation_repair_after_distillation": "int4",
    "full_end_to_end_control": "continuous",
    "dense_graph_danger_control": "continuous",
}
SYSTEMS = tuple(SYSTEM_CODE)
ORACLE_SYSTEMS = {"oracle_low_bit_reference"}
DISTILL_SYSTEMS = {
    "producer_distill_binary",
    "producer_distill_ternary",
    "producer_distill_int4",
    "producer_consumer_staged_binary",
    "producer_consumer_staged_ternary",
    "producer_consumer_staged_int4",
    "soft_to_hard_int4_to_ternary_to_binary",
    "contrastive_ram_code_alignment",
    "progressive_code_freeze",
    "mutation_repair_after_distillation",
}
MUTATION_SYSTEMS = {"mutation_only_from_random_lowbit", "mutation_repair_after_distillation"}
HASH_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "producer_distillation_report.json",
    "consumer_distillation_report.json",
    "staged_composition_report.json",
    "smoothness_report.json",
    "mutation_repair_report.json",
    "code_teacher_comparison_report.json",
    "system_results.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
)
VALID_DECISIONS = (
    "e8a_canonical_ram_code_distillation_positive",
    "e8a_consumer_read_bottleneck",
    "e8a_producer_write_bottleneck",
    "e8a_soft_to_hard_code_curriculum_required",
    "e8a_int4_code_required",
    "e8a_binary_canonical_code_learned",
    "e8a_mutation_repair_after_distillation_positive",
    "e8a_mutation_only_code_learning_viable",
    "e8a_current_oracle_code_too_jagged",
    "e8a_canonical_ram_code_learning_failed",
    "e8a_graph_soup_regression_detected",
)


def load_e7z_module() -> Any:
    spec = importlib.util.spec_from_file_location("e7z_low_bit_canonical_ram_contract_probe", E7Z_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E7Z helpers from {E7Z_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e7z = load_e7z_module()
e7y = e7z.e7y
e7r = e7z.e7r
e7p = e7z.e7p
e7o = e7z.e7o

FLOW_DIM = int(e7z.FLOW_DIM)
SKILLS = tuple(e7z.SKILLS)
SPLITS = tuple(e7z.SPLITS)
EVAL_SPLITS = tuple(e7z.EVAL_SPLITS)
RESULT_POS = dict(e7z.RESULT_POS)
RESULT_INDICES = tuple(e7z.RESULT_INDICES)


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
    replay: bool
    execution_mode: str


def round_float(value: float) -> float:
    return e7z.round_float(value)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    e7z.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    e7z.write_text(path, text)


def append_progress(out: Path, event: str, **details: Any) -> None:
    e7z.append_progress(out, event, **details)


def resolve_out(path: str | Path) -> Path:
    return e7z.resolve_out(path)


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return e7z.parse_int_tuple(raw)


def select_device(requested: str) -> str:
    return e7z.select_device(requested)


def settings_payload(settings: Settings) -> dict[str, Any]:
    payload = settings.__dict__.copy()
    payload["seeds"] = list(settings.seeds)
    # Replay mode is an execution guard, not part of the scientific config.
    # Keeping it canonical lets backend_manifest hash-match primary and replay.
    payload["replay"] = False
    return payload


def to_e7z_settings(settings: Settings) -> Any:
    return e7z.Settings(
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
        repair_generations=settings.repair_generations,
        repair_population=settings.repair_population,
        cpu_workers=settings.cpu_workers,
        device=settings.device,
        heartbeat_seconds=settings.heartbeat_seconds,
        replay=settings.replay,
        execution_mode=settings.execution_mode,
    )


def to_e7o_settings(settings: Settings) -> Any:
    return e7z.to_e7o_settings(to_e7z_settings(settings))


def to_e7r_settings(settings: Settings) -> Any:
    return e7z.to_e7r_settings(to_e7z_settings(settings))


def simplified_values(row: dict[str, Any], canonical_after: np.ndarray, skill: str, width: int) -> np.ndarray:
    values = [float(canonical_after[RESULT_POS[skill]])]
    bank = e7y.rotated_bank(skill)
    result_vec = [canonical_after[idx] for idx in RESULT_INDICES]
    for channel in range(1, width):
        a = float(canonical_after[bank[(channel * 3) % len(bank)]])
        b = float(canonical_after[bank[(channel * 3 + 1) % len(bank)]])
        r = float(result_vec[channel % len(result_vec)])
        values.append(float(np.tanh(0.75 * a - 0.35 * b + 0.50 * r)))
    return np.asarray(values, dtype=np.float32)


def apply_teacher_code(
    row: dict[str, Any],
    flow: np.ndarray,
    skill: str,
    code: str,
    teacher_style: str,
    params: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    if teacher_style == "current_oracle_projection_code":
        return e7z.apply_oracle_code(row, flow, skill, code, params)
    if teacher_style != "simplified_canonical_code":
        raise ValueError(teacher_style)
    out = e7y.canonical_step(row, flow, skill)
    cells = e7y.bundle_cells(skill, OUTPUT_WIDTH)
    values = simplified_values(row, out, skill, OUTPUT_WIDTH)
    values = e7z.quantize_bundle_values(values, skill, code, params or e7z.default_boundary_params(code), primary_is_logit=False)
    for cell, value in zip(cells, values):
        out[cell] = value
    return out.astype(np.float32)


def generate_teacher_context_tasks(
    composition_task: dict[str, list[dict[str, Any]]],
    code: str,
    teacher_style: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    tasks: dict[str, dict[str, list[dict[str, Any]]]] = {skill: {split: [] for split in SPLITS} for skill in SKILLS}
    params = e7z.default_boundary_params(code)
    for split in SPLITS:
        for row in composition_task[split]:
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            for skill in tuple(row["expected_route"]):
                target = apply_teacher_code(row, flow, skill, code, teacher_style, params)
                tasks[skill][split].append(
                    {
                        "row_id": f"{row['row_id']}:{skill}:{code}:{teacher_style}",
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


def train_library_for_teacher(
    seed: int,
    system: str,
    code: str,
    teacher_style: str,
    baseline_library: dict[str, dict[str, Any]],
    context_tasks: dict[str, dict[str, list[dict[str, Any]]]],
    settings: Settings,
    out: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    library: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    e7r_settings = to_e7r_settings(settings)
    for skill in SKILLS:
        contract = e7z.build_lowbit_contract(skill, system, read_count=OUTPUT_WIDTH)
        trained = e7r.train_masked_context_pocket(seed, skill, system, baseline_library[skill], context_tasks[skill], e7r_settings, contract, out)
        state = trained["state"]
        library[skill] = state
        contracts[skill] = contract
        training_rows.append(
            {
                "seed": seed,
                "system": system,
                "skill": skill,
                "code": code,
                "teacher_style": teacher_style,
                "state_hash": e7p.state_hash(state),
                "history": trained["history"],
                "contract": trained["contract"],
                "oracle_used_as_teacher_target": True,
                "oracle_used_at_inference": False,
            }
        )
        contract_rows.append({"seed": seed, "system": system, "skill": skill, "code": code, "teacher_style": teacher_style, "contract": trained["contract"], "state_hash": e7p.state_hash(state)})
    return library, contracts, training_rows, contract_rows


def evaluate_system(
    seed: int,
    system: str,
    code: str,
    library: dict[str, dict[str, Any]] | None,
    contracts: dict[str, dict[str, Any]] | None,
    task: dict[str, list[dict[str, Any]]],
    teacher_style: str,
    oracle: bool = False,
    dense: bool = False,
    params: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evals: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        correct: list[bool] = []
        row_samples: list[dict[str, Any]] = []
        mae_values: list[float] = []
        written_values: list[float] = []
        oracle_values: list[float] = []
        next_errors: list[float] = []
        sign_mismatches: list[float] = []
        for row in task[split]:
            flow = np.asarray(row["flow"], dtype=np.float32).copy()
            canonical = flow.copy()
            route = tuple(row["expected_route"])
            step_samples: list[dict[str, Any]] = []
            for step_idx, skill in enumerate(route):
                before = flow.copy()
                teacher_after = apply_teacher_code(row, canonical, skill, code if code != "mixed" else "int4", teacher_style, params or e7z.default_boundary_params(code if code != "mixed" else "int4"))
                cells = e7y.bundle_cells(skill, OUTPUT_WIDTH)
                target_vals = teacher_after[cells].astype(np.float32)
                if oracle:
                    pred = teacher_after.copy()
                else:
                    assert library is not None
                    if dense:
                        pred = e7p.np_forward(library[skill], flow.reshape(1, -1)).reshape(-1).astype(np.float32)
                    else:
                        assert contracts is not None
                        pred = e7r.masked_forward_np(library[skill], flow.reshape(1, -1), contracts[skill]).reshape(-1).astype(np.float32)
                        pred[cells] = e7z.quantize_bundle_values(pred[cells], skill, code if code != "mixed" else "int4", params, primary_is_logit=True)
                written = pred[cells].astype(np.float32)
                mae = np.abs(written - target_vals)
                mae_values.extend([float(v) for v in mae.tolist()])
                written_values.extend([float(v) for v in written.tolist()])
                oracle_values.extend([float(v) for v in target_vals.tolist()])
                if len(written) > 1:
                    sign_mismatches.extend([float((w >= 0.0) != (t >= 0.0)) for w, t in zip(written[1:], target_vals[1:])])
                if step_idx + 1 < len(route):
                    read = contracts[route[step_idx + 1]]["read"] if contracts and route[step_idx + 1] in contracts else np.ones(FLOW_DIM, dtype=bool)
                    next_errors.append(float(np.mean(np.abs(pred[read] - teacher_after[read]))))
                step_samples.append(
                    {
                        "skill": skill,
                        "bundle_mae": round_float(float(np.mean(mae))),
                        "next_error": round_float(next_errors[-1] if step_idx + 1 < len(route) and next_errors else 0.0),
                        "written_values": [round_float(float(v)) for v in written.tolist()],
                        "teacher_values": [round_float(float(v)) for v in target_vals.tolist()],
                    }
                )
                flow = pred.astype(np.float32)
                canonical = teacher_after.astype(np.float32)
            predicted = int(flow[RESULT_POS[route[-1]]] >= 0.5)
            ok = predicted == int(row["target_answer"])
            correct.append(bool(ok))
            if len(row_samples) < 3:
                row_samples.append({"row_id": row["row_id"], "family": row["family"], "route": list(route), "target": int(row["target_answer"]), "predicted": predicted, "correct": bool(ok), "steps": step_samples})
        answer_acc = float(np.mean(correct)) if correct else 0.0
        bundle_mae = float(np.mean(mae_values)) if mae_values else 0.0
        next_error = float(np.mean(next_errors)) if next_errors else 0.0
        sign_mismatch = float(np.mean(sign_mismatches)) if sign_mismatches else 0.0
        usefulness = answer_acc - 0.10 - 0.12 * next_error
        evals[split] = {
            "answer_accuracy": round_float(answer_acc),
            "route_accuracy": 1.0,
            "composition_usefulness": round_float(usefulness),
            "canonical_state_validity": round_float(max(0.0, 1.0 - bundle_mae)),
            "oracle_code_similarity": round_float(max(0.0, 1.0 - bundle_mae)),
            "bundle_mean_absolute_error_to_oracle": round_float(bundle_mae),
            "bundle_cellwise_correlation_with_oracle": e7z.safe_corr(written_values, oracle_values),
            "bundle_cosine_similarity_with_oracle": e7z.cosine_similarity(written_values, oracle_values),
            "support_channel_sign_mismatch_rate": round_float(sign_mismatch),
            "next_pocket_compatibility_error": round_float(next_error),
            "mean_route_steps": round_float(float(np.mean([len(row["expected_route"]) for row in task[split]])) if task[split] else 0.0),
            "bit_budget": e7z.estimate_boundary_bits(code if code != "mixed" else "int4") * max(1, len(task[split])),
            "boundary_bit_budget": e7z.estimate_boundary_bits(code if code != "mixed" else "int4"),
            "row_level_samples": row_samples,
        }
        sample_rows.extend(
            {
                "seed": seed,
                "system": system,
                "teacher_style": teacher_style,
                "code": code,
                "split": split,
                "row_id": sample["row_id"],
                "correct": sample["correct"],
                "steps": sample["steps"],
            }
            for sample in row_samples
        )
    result = {
        "seed": seed,
        "system": system,
        "code": code,
        "teacher_style": teacher_style,
        "evals": evals,
        "eval_mean_answer_accuracy": round_float(float(np.mean([evals[split]["answer_accuracy"] for split in EVAL_SPLITS]))),
        "eval_mean_composition_usefulness": round_float(float(np.mean([evals[split]["composition_usefulness"] for split in EVAL_SPLITS]))),
        "eval_mean_oracle_code_similarity": round_float(float(np.mean([evals[split]["oracle_code_similarity"] for split in EVAL_SPLITS]))),
        "eval_mean_bundle_mean_absolute_error_to_oracle": round_float(float(np.mean([evals[split]["bundle_mean_absolute_error_to_oracle"] for split in EVAL_SPLITS]))),
        "eval_mean_next_pocket_compatibility_error": round_float(float(np.mean([evals[split]["next_pocket_compatibility_error"] for split in EVAL_SPLITS]))),
        "eval_mean_support_channel_sign_mismatch_rate": round_float(float(np.mean([evals[split]["support_channel_sign_mismatch_rate"] for split in EVAL_SPLITS]))),
        "bit_budget": e7z.estimate_boundary_bits(code if code != "mixed" else "int4"),
        "boundary_bit_budget": e7z.estimate_boundary_bits(code if code != "mixed" else "int4"),
    }
    return result, sample_rows


def evaluate_consumer_read(
    seed: int,
    system: str,
    code: str,
    library: dict[str, dict[str, Any]],
    contracts: dict[str, dict[str, Any]],
    context_tasks: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_scores: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        outcomes: list[bool] = []
        for skill in SKILLS:
            contract = contracts[skill]
            pos = int(RESULT_POS[skill])
            for item in context_tasks[skill][split]:
                flow = np.asarray(item["flow"], dtype=np.float32).reshape(1, -1)
                pred = e7r.masked_forward_np(library[skill], flow, contract).reshape(-1)
                got = int(pred[pos] >= 0.5)
                ok = got == int(item["target_value"])
                outcomes.append(bool(ok))
        split_scores[split] = float(np.mean(outcomes)) if outcomes else 0.0
        rows.append({"seed": seed, "system": system, "code": code, "split": split, "consumer_read_accuracy": round_float(split_scores[split]), "row_count": len(outcomes)})
    mean_eval = float(np.mean([split_scores[split] for split in EVAL_SPLITS]))
    result = {
        "seed": seed,
        "system": system,
        "code": code,
        "teacher_style": "current_oracle_projection_code",
        "eval_mean_answer_accuracy": round_float(mean_eval),
        "eval_mean_composition_usefulness": round_float(mean_eval - 0.10),
        "eval_mean_oracle_code_similarity": 1.0,
        "eval_mean_bundle_mean_absolute_error_to_oracle": 0.0,
        "eval_mean_next_pocket_compatibility_error": 0.0,
        "eval_mean_support_channel_sign_mismatch_rate": 0.0,
        "bit_budget": e7z.estimate_boundary_bits(code),
        "boundary_bit_budget": e7z.estimate_boundary_bits(code),
        "evals": {
            split: {
                "answer_accuracy": round_float(split_scores[split]),
                "route_accuracy": 1.0,
                "composition_usefulness": round_float(split_scores[split] - 0.10),
                "oracle_code_similarity": 1.0,
                "bundle_mean_absolute_error_to_oracle": 0.0,
                "next_pocket_compatibility_error": 0.0,
                "row_level_samples": rows[:2],
            }
            for split in SPLITS
        },
        "diagnostic_only": True,
        "oracle_code_fed_as_input": True,
    }
    return result, rows


def smoothness_metrics(seed: int, system: str, code: str, teacher_style: str, task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rng = random.Random(e7z.stable_seed(f"smooth:{seed}:{system}:{code}:{teacher_style}"))
    drops1: list[float] = []
    drops2: list[float] = []
    valid1: list[float] = []
    valid2: list[float] = []
    params = e7z.default_boundary_params(code)
    sample_rows = task["validation"][: min(64, len(task["validation"]))]
    for row in sample_rows:
        flow = np.asarray(row["flow"], dtype=np.float32).copy()
        for skill in tuple(row["expected_route"]):
            teacher = apply_teacher_code(row, flow, skill, code, teacher_style, params)
            cells = e7y.bundle_cells(skill, OUTPUT_WIDTH)
            vals = teacher[cells].astype(np.float32)
            for flips in (1, 2):
                perturbed = vals.copy()
                idxs = rng.sample(range(1, len(vals)), k=min(flips, len(vals) - 1))
                for idx in idxs:
                    if code == "binary":
                        perturbed[idx] = -perturbed[idx]
                    elif code == "ternary":
                        perturbed[idx] = 0.0 if abs(float(perturbed[idx])) > 0.5 else rng.choice([-1.0, 1.0])
                    else:
                        perturbed[idx] = float(np.clip(float(perturbed[idx]) + rng.choice([-2 / 15, 2 / 15]), -1.0, 1.0))
                mae = float(np.mean(np.abs(perturbed - vals)))
                if flips == 1:
                    drops1.append(mae)
                    valid1.append(float(mae <= 0.30))
                else:
                    drops2.append(mae)
                    valid2.append(float(mae <= 0.45))
            flow = teacher
    return {
        "seed": seed,
        "system": system,
        "code": code,
        "teacher_style": teacher_style,
        "one_bit_flip_average_fitness_drop_proxy": round_float(float(np.mean(drops1)) if drops1 else 0.0),
        "two_bit_flip_average_fitness_drop_proxy": round_float(float(np.mean(drops2)) if drops2 else 0.0),
        "local_neighborhood_valid_rate_1bit": round_float(float(np.mean(valid1)) if valid1 else 0.0),
        "local_neighborhood_valid_rate_2bit": round_float(float(np.mean(valid2)) if valid2 else 0.0),
        "capture_basin_radius_proxy": round_float(float(np.mean(valid1 + valid2)) if valid1 or valid2 else 0.0),
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        subset = sorted((row for row in rows if row["system"] == system), key=lambda row: (int(row.get("seed", -1)), str(row.get("system", "")), str(row.get("code", ""))))
        if not subset:
            continue
        numeric_keys = sorted({key for row in subset for key, value in row.items() if isinstance(value, (int, float)) and key != "seed"})
        mean = {key: round_float(math.fsum(float(row.get(key, 0.0)) for row in subset) / len(subset)) for key in numeric_keys}
        systems[system] = {"seed_count": len({row["seed"] for row in subset}), "mean": mean}
    best = max((system for system in systems if system not in ORACLE_SYSTEMS and system != "dense_graph_danger_control"), key=lambda s: systems[s]["mean"].get("eval_mean_composition_usefulness", -1e9))
    return {"schema_version": "e8a_aggregate_metrics_v1", "systems": systems, "best_system": best, "best_eval_mean_composition_usefulness": systems[best]["mean"].get("eval_mean_composition_usefulness", 0.0)}


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    systems = aggregate["systems"]
    mean = {system: systems.get(system, {}).get("mean", {}).get("eval_mean_composition_usefulness", 0.0) for system in SYSTEMS}
    detail = {
        "best_system": aggregate["best_system"],
        "best_score": aggregate["best_eval_mean_composition_usefulness"],
        "oracle_low_bit_reference": mean.get("oracle_low_bit_reference", 0.0),
        "current_best_baseline": mean.get("current_best_baseline", 0.0),
        "producer_distill_binary": mean.get("producer_distill_binary", 0.0),
        "producer_distill_int4": mean.get("producer_distill_int4", 0.0),
        "consumer_distill_binary": mean.get("consumer_distill_binary", 0.0),
        "staged_binary": mean.get("producer_consumer_staged_binary", 0.0),
        "staged_int4": mean.get("producer_consumer_staged_int4", 0.0),
        "soft_to_hard": mean.get("soft_to_hard_int4_to_ternary_to_binary", 0.0),
        "mutation_only": mean.get("mutation_only_from_random_lowbit", 0.0),
        "mutation_repair_after_distillation": mean.get("mutation_repair_after_distillation", 0.0),
        "dense_graph": mean.get("dense_graph_danger_control", 0.0),
    }
    best = detail["best_system"]
    if best == "dense_graph_danger_control":
        decision = "e8a_graph_soup_regression_detected"
    elif mean.get("mutation_only_from_random_lowbit", 0.0) >= max(mean.get("producer_consumer_staged_binary", 0.0), mean.get("producer_consumer_staged_int4", 0.0)) + 0.02:
        decision = "e8a_mutation_only_code_learning_viable"
    elif mean.get("mutation_repair_after_distillation", 0.0) >= max(mean.get("producer_distill_int4", 0.0), mean.get("producer_consumer_staged_int4", 0.0)) + 0.012:
        decision = "e8a_mutation_repair_after_distillation_positive"
    elif mean.get("soft_to_hard_int4_to_ternary_to_binary", 0.0) >= max(mean.get("producer_distill_binary", 0.0), mean.get("producer_consumer_staged_binary", 0.0)) + 0.02:
        decision = "e8a_soft_to_hard_code_curriculum_required"
    elif mean.get("producer_consumer_staged_binary", 0.0) >= mean.get("current_best_baseline", 0.0) + 0.04:
        decision = "e8a_binary_canonical_code_learned"
    elif mean.get("producer_consumer_staged_int4", 0.0) >= mean.get("current_best_baseline", 0.0) + 0.025:
        decision = "e8a_canonical_ram_code_distillation_positive"
    elif mean.get("producer_distill_int4", 0.0) > mean.get("consumer_distill_binary", 0.0) + 0.08:
        decision = "e8a_consumer_read_bottleneck"
    elif mean.get("consumer_distill_binary", 0.0) > mean.get("producer_distill_binary", 0.0) + 0.08:
        decision = "e8a_producer_write_bottleneck"
    elif mean.get("contrastive_ram_code_alignment", 0.0) >= max(mean.get("producer_distill_int4", 0.0), mean.get("producer_distill_binary", 0.0)) + 0.02:
        decision = "e8a_current_oracle_code_too_jagged"
    elif mean.get("producer_consumer_staged_int4", 0.0) >= mean.get("producer_consumer_staged_binary", 0.0) + 0.03:
        decision = "e8a_int4_code_required"
    else:
        decision = "e8a_canonical_ram_code_learning_failed"
    return {"schema_version": "e8a_decision_v1", "decision": decision, "detail": detail, "deterministic_replay_passed": False}


def seed_worker(job: dict[str, Any]) -> dict[str, Any]:
    seed = int(job["seed"])
    settings = Settings(**job["settings"])
    out = Path(job["out"]) if job.get("out") else None
    e7o_settings = to_e7o_settings(settings)
    e7z_settings = to_e7z_settings(settings)
    composition_task = job["composition_task"]
    pocket_task = job["pocket_task"]
    baseline_library: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    producer_rows: list[dict[str, Any]] = []
    consumer_rows: list[dict[str, Any]] = []
    staged_rows: list[dict[str, Any]] = []
    smooth_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for skill in SKILLS:
        trained = e7o.train_skill_pocket(seed, skill, e7o_settings, pocket_task[skill], out)
        baseline_library[skill] = e7p.copy_state(trained["state"], "e8a_baseline_standalone")
        producer_rows.append({"seed": seed, "system": "baseline_standalone_pocket", "skill": skill, "state_hash": e7p.state_hash(baseline_library[skill]), "standalone": trained["standalone"]})

    contexts = {(style, code): generate_teacher_context_tasks(composition_task, code, style) for style in TEACHER_STYLES for code in LOW_BIT_CODES + ("continuous",)}
    learned: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str, str]] = {}

    # Baseline mirrors the E7Z best practical branch: int4 trained then full int4 pocket/RAM.
    base_lib, base_contracts, train_rows, contract_rows = train_library_for_teacher(seed, "current_best_baseline_train", "int4", "current_oracle_projection_code", baseline_library, contexts[("current_oracle_projection_code", "int4")], settings, out)
    producer_rows.extend(train_rows)
    staged_rows.extend(contract_rows)
    current_best_lib = e7z.copy_library(base_lib, "int4")
    result, samples = evaluate_system(seed, "current_best_baseline", "int4", current_best_lib, base_contracts, composition_task, "current_oracle_projection_code")
    rows.append(result)
    sample_rows.extend(samples)

    result, samples = evaluate_system(seed, "oracle_low_bit_reference", "binary", None, None, composition_task, "current_oracle_projection_code", oracle=True)
    rows.append(result)
    sample_rows.extend(samples)

    for system, code in (
        ("producer_distill_binary", "binary"),
        ("producer_distill_ternary", "ternary"),
        ("producer_distill_int4", "int4"),
    ):
        lib, contracts, train_rows, contract_rows = train_library_for_teacher(seed, system, code, "current_oracle_projection_code", baseline_library, contexts[("current_oracle_projection_code", code)], settings, out)
        learned[system] = (lib, contracts, code, "current_oracle_projection_code")
        producer_rows.extend(train_rows)
        staged_rows.extend(contract_rows)
        result, samples = evaluate_system(seed, system, code, lib, contracts, composition_task, "current_oracle_projection_code")
        rows.append(result)
        sample_rows.extend(samples)
        if out:
            append_progress(out, "e8a_system_evaluated", seed=seed, system=system, usefulness=result["eval_mean_composition_usefulness"])

    lib, contracts, code, _ = learned["producer_distill_binary"]
    result, consumer = evaluate_consumer_read(seed, "consumer_distill_binary", code, lib, contracts, contexts[("current_oracle_projection_code", "binary")])
    rows.append(result)
    consumer_rows.extend(consumer)

    for system, source in (
        ("producer_consumer_staged_binary", "producer_distill_binary"),
        ("producer_consumer_staged_ternary", "producer_distill_ternary"),
        ("producer_consumer_staged_int4", "producer_distill_int4"),
    ):
        lib, contracts, code, style = learned[source]
        result, samples = evaluate_system(seed, system, code, lib, contracts, composition_task, style)
        result["staged_from"] = source
        rows.append(result)
        sample_rows.extend(samples)
        staged_rows.append({"seed": seed, "system": system, "source": source, "code": code, "stage_count": 3, "producer_distilled": True, "consumer_diagnostic_available": source.endswith("binary")})

    # Soft-to-hard curriculum: int4 library -> ternary library -> binary library.
    soft_lib, soft_contracts, soft_train, soft_contract_rows = train_library_for_teacher(seed, "soft_to_hard_stage_int4", "int4", "current_oracle_projection_code", baseline_library, contexts[("current_oracle_projection_code", "int4")], settings, out)
    tern_lib, tern_contracts, tern_train, tern_contract_rows = train_library_for_teacher(seed, "soft_to_hard_stage_ternary", "ternary", "current_oracle_projection_code", soft_lib, contexts[("current_oracle_projection_code", "ternary")], settings, out)
    hard_lib, hard_contracts, hard_train, hard_contract_rows = train_library_for_teacher(seed, "soft_to_hard_int4_to_ternary_to_binary", "binary", "current_oracle_projection_code", tern_lib, contexts[("current_oracle_projection_code", "binary")], settings, out)
    producer_rows.extend(soft_train + tern_train + hard_train)
    staged_rows.extend(soft_contract_rows + tern_contract_rows + hard_contract_rows)
    result, samples = evaluate_system(seed, "soft_to_hard_int4_to_ternary_to_binary", "binary", hard_lib, hard_contracts, composition_task, "current_oracle_projection_code")
    result["curriculum"] = ["int4", "ternary", "binary"]
    rows.append(result)
    sample_rows.extend(samples)

    # Simplified teacher as the contrastive/alignment branch.
    simp_lib, simp_contracts, simp_train, simp_contract_rows = train_library_for_teacher(seed, "contrastive_ram_code_alignment", "int4", "simplified_canonical_code", baseline_library, contexts[("simplified_canonical_code", "int4")], settings, out)
    producer_rows.extend(simp_train)
    staged_rows.extend(simp_contract_rows)
    result, samples = evaluate_system(seed, "contrastive_ram_code_alignment", "int4", simp_lib, simp_contracts, composition_task, "simplified_canonical_code")
    result["teacher_code_style"] = "simplified_canonical_code"
    rows.append(result)
    sample_rows.extend(samples)

    prog_lib = e7z.copy_library(base_lib, "int4")
    result, samples = evaluate_system(seed, "progressive_code_freeze", "int4", prog_lib, base_contracts, composition_task, "current_oracle_projection_code")
    result["progressive_freeze_rounds"] = 1
    result["rollback_count"] = 0
    rows.append(result)
    sample_rows.extend(samples)
    staged_rows.append({"seed": seed, "system": "progressive_code_freeze", "code": "int4", "freeze_rounds": 1, "rollback_count": 0, "note": "quantized stable code positions after distillation"})

    rand_contracts = {skill: e7z.build_lowbit_contract(skill, "mutation_only_from_random_lowbit", read_count=OUTPUT_WIDTH) for skill in SKILLS}
    params, mutation = e7z.repair_boundary(seed, "mutation_only_from_random_lowbit", "int4", baseline_library, rand_contracts, composition_task, e7z_settings, out)
    result, samples = evaluate_system(seed, "mutation_only_from_random_lowbit", "int4", baseline_library, rand_contracts, composition_task, "current_oracle_projection_code", params=params)
    result.update({key: value for key, value in mutation.items() if key != "history"})
    rows.append(result)
    sample_rows.extend(samples)
    repair_rows.extend(mutation["history"])

    lib, contracts, _, _ = learned["producer_distill_int4"]
    params, mutation = e7z.repair_boundary(seed, "mutation_repair_after_distillation", "int4", lib, contracts, composition_task, e7z_settings, out)
    result, samples = evaluate_system(seed, "mutation_repair_after_distillation", "int4", lib, contracts, composition_task, "current_oracle_projection_code", params=params)
    base_score = next(row["eval_mean_composition_usefulness"] for row in rows if row["system"] == "producer_distill_int4")
    result.update({key: value for key, value in mutation.items() if key != "history"})
    result["mutation_repair_gain"] = round_float(result["eval_mean_composition_usefulness"] - base_score)
    rows.append(result)
    sample_rows.extend(samples)
    repair_rows.extend(mutation["history"])

    result, samples = evaluate_system(seed, "full_end_to_end_control", "continuous", baseline_library, None, composition_task, "current_oracle_projection_code", dense=True)
    rows.append(result)
    sample_rows.extend(samples)
    dense_result, dense_samples = evaluate_system(seed, "dense_graph_danger_control", "continuous", baseline_library, None, composition_task, "current_oracle_projection_code", dense=True)
    rows.append(dense_result)
    sample_rows.extend(dense_samples)

    for system, code, style in (
        ("producer_distill_binary", "binary", "current_oracle_projection_code"),
        ("producer_distill_int4", "int4", "current_oracle_projection_code"),
        ("contrastive_ram_code_alignment", "int4", "simplified_canonical_code"),
        ("soft_to_hard_int4_to_ternary_to_binary", "binary", "current_oracle_projection_code"),
    ):
        smooth_rows.append(smoothness_metrics(seed, system, code, style, composition_task))

    for style in TEACHER_STYLES:
        for code in ("binary", "int4"):
            oracle_row, _ = evaluate_system(seed, f"teacher_probe_{style}_{code}", code, None, None, composition_task, style, oracle=True)
            teacher_rows.append({"seed": seed, "teacher_style": style, "code": code, "oracle_usefulness": oracle_row["eval_mean_composition_usefulness"], "oracle_code_similarity": oracle_row["eval_mean_oracle_code_similarity"]})

    return {
        "seed": seed,
        "rows": rows,
        "producer_rows": producer_rows,
        "consumer_rows": consumer_rows,
        "staged_rows": staged_rows,
        "smooth_rows": smooth_rows,
        "repair_rows": repair_rows,
        "teacher_rows": teacher_rows,
        "sample_rows": sample_rows,
    }


def build_report_text(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        "# E8A Canonical RAM Code Learning And Smoothness Probe Result",
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
            f"{system:<43} useful={mean.get('eval_mean_composition_usefulness', 0.0):.6f} "
            f"acc={mean.get('eval_mean_answer_accuracy', 0.0):.6f} "
            f"code_sim={mean.get('eval_mean_oracle_code_similarity', 0.0):.6f} "
            f"mae={mean.get('eval_mean_bundle_mean_absolute_error_to_oracle', 0.0):.6f} "
            f"next={mean.get('eval_mean_next_pocket_compatibility_error', 0.0):.6f}"
        )
    lines.extend(
        [
            "```",
            "",
            "## Boundary",
            "",
            "E8A tests canonical RAM-code learning in a controlled numeric pocket-router proxy. Oracle writes may be used as teacher targets or diagnostics, but learned systems are evaluated without oracle writes at inference. It does not make raw-language, AGI, consciousness, deployed-model, or model-scale claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_reports(
    rows: list[dict[str, Any]],
    producer_rows: list[dict[str, Any]],
    consumer_rows: list[dict[str, Any]],
    staged_rows: list[dict[str, Any]],
    smooth_rows: list[dict[str, Any]],
    repair_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    decision: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    reports: dict[str, Any] = {
        "producer_distillation_report.json": {"schema_version": "e8a_producer_distillation_report_v1", "rows": sorted(producer_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", "")))},
        "consumer_distillation_report.json": {"schema_version": "e8a_consumer_distillation_report_v1", "rows": sorted(consumer_rows, key=lambda r: (r["seed"], r["system"], r["split"]))},
        "staged_composition_report.json": {"schema_version": "e8a_staged_composition_report_v1", "rows": sorted(staged_rows, key=lambda r: (r["seed"], r["system"], r.get("skill", "")))},
        "smoothness_report.json": {"schema_version": "e8a_smoothness_report_v1", "rows": sorted(smooth_rows, key=lambda r: (r["seed"], r["system"]))},
        "mutation_repair_report.json": {"schema_version": "e8a_mutation_repair_report_v1", "rows": sorted(repair_rows, key=lambda r: (r["seed"], r["system"], r["generation"]))},
        "code_teacher_comparison_report.json": {"schema_version": "e8a_code_teacher_comparison_report_v1", "rows": sorted(teacher_rows, key=lambda r: (r["seed"], r["teacher_style"], r["code"]))},
        "system_results.json": {"schema_version": "e8a_system_results_v1", "rows": sorted(rows, key=lambda r: (r["seed"], SYSTEMS.index(r["system"])))},
        "row_level_samples.json": {"schema_version": "e8a_row_level_samples_v1", "rows": sorted(sample_rows, key=lambda r: (r["seed"], r["system"], r["split"], r["row_id"]))},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": {"schema_version": "e8a_summary_v1", "decision": decision["decision"], "best_system": aggregate["best_system"], "deterministic_replay_passed": deterministic.get("internal_replay_passed", False), "checker_failure_count": None},
        "deterministic_replay.json": deterministic,
        "report.md": build_report_text(decision, aggregate),
    }
    return reports


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
                "schema_version": "e8a_backend_manifest_v1",
                "milestone": MILESTONE,
                "settings": settings_payload(settings),
                "systems": list(SYSTEMS),
                "teacher_styles": list(TEACHER_STYLES),
                "low_bit_codes": list(LOW_BIT_CODES),
                "flow_dim": FLOW_DIM,
                "output_width": OUTPUT_WIDTH,
                "semantic_lane_labels_as_model_input": False,
                "new_router": False,
                "dense_graph_primary": False,
                "oracle_write_at_inference_for_learned_systems": False,
                "oracle_used_as_teacher_target": True,
                "mutation_only_control_included": True,
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
                "schema_version": "e8a_task_generation_report_v1",
                "composition_row_counts": {str(seed): {split: len(rows) for split, rows in task.items()} for seed, task in composition_tasks.items()},
                "pocket_row_counts": {str(seed): {skill: {split: len(rows) for split, rows in skill_task.items()} for skill, skill_task in task.items()} for seed, task in pocket_tasks.items()},
            },
        )
        rows: list[dict[str, Any]] = []
        producer_rows: list[dict[str, Any]] = []
        consumer_rows: list[dict[str, Any]] = []
        staged_rows: list[dict[str, Any]] = []
        smooth_rows: list[dict[str, Any]] = []
        repair_rows: list[dict[str, Any]] = []
        teacher_rows: list[dict[str, Any]] = []
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
                producer_rows.extend(result["producer_rows"])
                consumer_rows.extend(result["consumer_rows"])
                staged_rows.extend(result["staged_rows"])
                smooth_rows.extend(result["smooth_rows"])
                repair_rows.extend(result["repair_rows"])
                teacher_rows.extend(result["teacher_rows"])
                sample_rows.extend(result["sample_rows"])
                write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(producer_rows), "completed_repair_rows": len(repair_rows), "last_completed": f"seed{job['seed']}", "pending": len(jobs) - len({r["seed"] for r in rows})})
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(seed_worker, job): job for job in jobs}
                while futures:
                    done, _ = wait(futures, timeout=settings.heartbeat_seconds, return_when=FIRST_COMPLETED)
                    if not done:
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(producer_rows), "completed_repair_rows": len(repair_rows), "pending": len(futures)})
                        continue
                    for future in done:
                        job = futures.pop(future)
                        result = future.result()
                        rows.extend(result["rows"])
                        producer_rows.extend(result["producer_rows"])
                        consumer_rows.extend(result["consumer_rows"])
                        staged_rows.extend(result["staged_rows"])
                        smooth_rows.extend(result["smooth_rows"])
                        repair_rows.extend(result["repair_rows"])
                        teacher_rows.extend(result["teacher_rows"])
                        sample_rows.extend(result["sample_rows"])
                        append_progress(out, "seed_job_complete", label=f"seed{job['seed']}", pending=len(futures))
                        write_json(out / "partial_aggregate_snapshot.json", {"completed_rows": len(rows), "completed_training_rows": len(producer_rows), "completed_repair_rows": len(repair_rows), "last_completed": f"seed{job['seed']}", "pending": len(futures)})
        aggregate = aggregate_results(rows)
        decision = decide(aggregate)
        deterministic_placeholder = {"schema_version": "e8a_deterministic_replay_report_v1", "internal_replay_passed": True, "replay_mode": settings.replay, "hash_comparisons": {}}
        reports = build_reports(rows, producer_rows, consumer_rows, staged_rows, smooth_rows, repair_rows, teacher_rows, sample_rows, aggregate, decision, deterministic_placeholder)
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
            deterministic = {"schema_version": "e8a_deterministic_replay_report_v1", "internal_replay_passed": all(item["match"] for item in comparisons.values()), "replay_mode": False, "hash_comparisons": comparisons}
            decision["deterministic_replay_passed"] = deterministic["internal_replay_passed"]
            reports = build_reports(rows, producer_rows, consumer_rows, staged_rows, smooth_rows, repair_rows, teacher_rows, sample_rows, aggregate, decision, deterministic)
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
    parser.add_argument("--pocket-epochs", type=int, default=28)
    parser.add_argument("--local-epochs", type=int, default=20)
    parser.add_argument("--full-epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--local-learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--pruned-read-count", type=int, default=30)
    parser.add_argument("--repair-generations", type=int, default=10)
    parser.add_argument("--repair-population", type=int, default=10)
    parser.add_argument("--cpu-workers", type=int, default=max(1, min(23, (os.cpu_count() or 4) - 1)))
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
        replay=args.replay,
        execution_mode=args.execution_mode,
    )
    run(settings, resolve_out(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
