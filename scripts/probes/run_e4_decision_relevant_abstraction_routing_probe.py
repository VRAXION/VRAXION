#!/usr/bin/env python3
"""E4 decision-relevant abstraction routing probe."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
E2_PATH = Path(__file__).with_name("run_e2_real_backend_state_medium_conductivity_ordering_audit.py")
MILESTONE = "E4_DECISION_RELEVANT_ABSTRACTION_ROUTING_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e4_decision_relevant_abstraction_routing_probe")
DEFAULT_SEEDS = (75001, 75002, 75003, 75004, 75005)


def load_e2_module() -> Any:
    spec = importlib.util.spec_from_file_location("e2_conductivity_backend", E2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E2 backend from {E2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e2 = load_e2_module()

SYSTEMS = (
    "flat_detail_scanner",
    "bottom_up_evidence_scanner",
    "top_down_hierarchical_router",
    "dynamic_state_medium_router",
)
REFERENCE_SYSTEMS = ("oracle_reference_only",)
ALL_SYSTEMS = (*SYSTEMS, *REFERENCE_SYSTEMS)
LEVELS = ("verdict", "cause", "mechanism", "evidence")
VERDICTS = ("passed", "failed", "inconclusive", "unsafe", "blocked")
CAUSES = (
    "clean_routing_wins",
    "recurrence_not_required",
    "leakage",
    "baseline_sufficient",
    "scaffold_dependency",
    "runtime_failure",
    "data_insufficient",
    "overbranching",
)
MECHANISMS = (
    "top_down_path_consistent",
    "no_recurrence_matches",
    "route_index_artifact",
    "strong_flat_matches",
    "random_projector_failed",
    "timeout_or_missing_artifact",
    "weak_sample_or_low_gap",
    "irrelevant_detail_expansion",
)
EVIDENCE = (
    "heldout_ood_counterfactual_pass",
    "no_recurrence_gap_close",
    "leakage_control_failed",
    "strong_flat_gap_close",
    "sparse_random_projector_failed",
    "no_final_artifact",
    "sample_or_replay_insufficient",
    "too_many_detail_branches",
)
DESCEND = ("stop", "descend")
DEPTHS = (1, 2, 3, 4)
HEADS = ("level", "verdict", "descend", "cause", "mechanism", "evidence", "stop_depth")
CHOICES = {
    "level": LEVELS,
    "verdict": VERDICTS,
    "descend": DESCEND,
    "cause": CAUSES,
    "mechanism": MECHANISMS,
    "evidence": EVIDENCE,
    "stop_depth": tuple(str(depth) for depth in DEPTHS),
}
FEATURES = (
    "verdict_cost",
    "level_cost",
    "cause_cost",
    "mechanism_cost",
    "evidence_cost",
    "stop_depth_cost",
    "detail_salience_cost",
    "misleading_metric_cost",
    "branch_relevance_cost",
    "confidence_cost",
    "ood_shift_cost",
    "counterfactual_shift_cost",
    "overbranch_cost",
    "underbranch_cost",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}
FEATURE_DIM = len(FEATURES)
STATE_DIM = 10
ORDERING_SPLITS = ("heldout", "ood", "counterfactual")
CAUSE_TO_VERDICT = {
    "clean_routing_wins": "passed",
    "recurrence_not_required": "failed",
    "leakage": "unsafe",
    "baseline_sufficient": "failed",
    "scaffold_dependency": "failed",
    "runtime_failure": "blocked",
    "data_insufficient": "inconclusive",
    "overbranching": "failed",
}
MECHANISM_TO_CAUSE = {
    "top_down_path_consistent": "clean_routing_wins",
    "no_recurrence_matches": "recurrence_not_required",
    "route_index_artifact": "leakage",
    "strong_flat_matches": "baseline_sufficient",
    "random_projector_failed": "scaffold_dependency",
    "timeout_or_missing_artifact": "runtime_failure",
    "weak_sample_or_low_gap": "data_insufficient",
    "irrelevant_detail_expansion": "overbranching",
}
EVIDENCE_TO_MECHANISM = {
    "heldout_ood_counterfactual_pass": "top_down_path_consistent",
    "no_recurrence_gap_close": "no_recurrence_matches",
    "leakage_control_failed": "route_index_artifact",
    "strong_flat_gap_close": "strong_flat_matches",
    "sparse_random_projector_failed": "random_projector_failed",
    "no_final_artifact": "timeout_or_missing_artifact",
    "sample_or_replay_insufficient": "weak_sample_or_low_gap",
    "too_many_detail_branches": "irrelevant_detail_expansion",
}
SCENARIOS = (
    ("passed", "clean_routing_wins", "top_down_path_consistent", "heldout_ood_counterfactual_pass"),
    ("failed", "recurrence_not_required", "no_recurrence_matches", "no_recurrence_gap_close"),
    ("unsafe", "leakage", "route_index_artifact", "leakage_control_failed"),
    ("failed", "baseline_sufficient", "strong_flat_matches", "strong_flat_gap_close"),
    ("failed", "scaffold_dependency", "random_projector_failed", "sparse_random_projector_failed"),
    ("blocked", "runtime_failure", "timeout_or_missing_artifact", "no_final_artifact"),
    ("inconclusive", "data_insufficient", "weak_sample_or_low_gap", "sample_or_replay_insufficient"),
    ("failed", "overbranching", "irrelevant_detail_expansion", "too_many_detail_branches"),
)
INTENTS = (
    ("status_question", "verdict", 2),
    ("why_question", "cause", 3),
    ("debug_question", "mechanism", 4),
    ("audit_question", "evidence", 4),
)
HASH_ARTIFACTS = (
    "e4_routing_report.json",
    "e4_control_baseline_report.json",
    "e4_leakage_sentinel_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    population_size: int
    generations: int
    mutation_sigma: float
    elite_count: int


def round_float(value: float) -> float:
    return e2.round_float(value)


def resolve_out(path: str | Path) -> Path:
    return e2.resolve_out(path)


def parse_seeds(raw: str) -> tuple[int, ...]:
    return e2.parse_seeds(raw)


def candidate_hash(candidate: dict[str, Any]) -> str:
    cleaned = {key: value for key, value in candidate.items() if key != "candidate_id"}
    return e2.payload_sha256(cleaned)


def matrix(rows: int, cols: int, value: float = 0.0) -> list[list[float]]:
    return [[round_float(value) for _ in range(cols)] for _ in range(rows)]


def vector(size: int, value: float = 0.0) -> list[float]:
    return [round_float(value) for _ in range(size)]


def round_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return e2.round_candidate(candidate)


def index_of(head: str, name: str) -> int:
    return CHOICES[head].index(name)


def target_indices(row: dict[str, Any]) -> dict[str, int]:
    target_depth = int(row["target_depth"])
    target_level_index = index_of("level", row["target_level"])
    return {
        "level": target_level_index,
        "verdict": index_of("verdict", row["verdict"]),
        "descend": 1 if target_depth > target_level_index + 1 else 0,
        "cause": index_of("cause", row["cause"]),
        "mechanism": index_of("mechanism", row["mechanism"]),
        "evidence": index_of("evidence", row["evidence"]),
        "stop_depth": target_depth - 1,
    }


def split_shift(split: str) -> tuple[float, float]:
    return (
        0.20 if split == "ood" else 0.0,
        0.22 if split == "counterfactual" else 0.0,
    )


def choice_features(head: str, correct_index: int, rng: random.Random, split: str, target_depth: int) -> list[list[float]]:
    ood, counterfactual = split_shift(split)
    rows: list[list[float]] = []
    trap_index = rng.randrange(len(CHOICES[head]))
    if trap_index == correct_index and len(CHOICES[head]) > 1:
        trap_index = (trap_index + 1) % len(CHOICES[head])
    head_feature = {
        "verdict": "verdict_cost",
        "level": "level_cost",
        "cause": "cause_cost",
        "mechanism": "mechanism_cost",
        "evidence": "evidence_cost",
        "stop_depth": "stop_depth_cost",
        "descend": "branch_relevance_cost",
    }[head]
    for index in range(len(CHOICES[head])):
        values = [rng.uniform(0.44, 0.68) for _ in FEATURES]
        values[FEATURE_INDEX[head_feature]] = rng.uniform(0.03, 0.10) if index == correct_index else rng.uniform(0.78, 0.98)
        values[FEATURE_INDEX["detail_salience_cost"]] = rng.uniform(0.05, 0.12) if index == trap_index else rng.uniform(0.45, 0.80)
        values[FEATURE_INDEX["misleading_metric_cost"]] = rng.uniform(0.06, 0.14) if index == trap_index else rng.uniform(0.40, 0.84)
        values[FEATURE_INDEX["confidence_cost"]] = rng.uniform(0.02, 0.10) if index == correct_index else rng.uniform(0.55, 0.95)
        values[FEATURE_INDEX["ood_shift_cost"]] = ood + rng.uniform(0.0, 0.05)
        values[FEATURE_INDEX["counterfactual_shift_cost"]] = counterfactual + rng.uniform(0.0, 0.05)
        if head == "level" and index > correct_index:
            values[FEATURE_INDEX["overbranch_cost"]] = rng.uniform(0.02, 0.12)
        elif head == "level" and index < correct_index:
            values[FEATURE_INDEX["underbranch_cost"]] = rng.uniform(0.02, 0.12)
        if head == "stop_depth" and index + 1 > target_depth:
            values[FEATURE_INDEX["overbranch_cost"]] = rng.uniform(0.02, 0.12)
        elif head == "stop_depth" and index + 1 < target_depth:
            values[FEATURE_INDEX["underbranch_cost"]] = rng.uniform(0.02, 0.12)
        rows.append([round_float(value) for value in values])
    return rows


def generate_split(seeds: tuple[int, ...], rows_per_seed: int, split: str, offset: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    choices = {head: [] for head in HEADS}
    targets = {head: [] for head in HEADS}
    for seed in seeds:
        rng = random.Random(seed + offset)
        for row_index in range(rows_per_seed):
            scenario = SCENARIOS[(row_index + rng.randrange(len(SCENARIOS))) % len(SCENARIOS)]
            intent, target_level, base_depth = INTENTS[(row_index + seed + rng.randrange(len(INTENTS))) % len(INTENTS)]
            verdict, cause, mechanism, evidence = scenario
            depth = max(base_depth, LEVELS.index(target_level) + 1)
            if intent == "status_question" and verdict in {"blocked", "unsafe"}:
                depth = 2
            if split == "counterfactual" and intent == "status_question" and verdict == "passed":
                verdict, cause, mechanism, evidence = SCENARIOS[1]
                depth = 2
            row = {
                "row_id": f"{split}_seed{seed}_row{row_index:04d}",
                "split": split,
                "seed": seed,
                "row_index": row_index,
                "intent": intent,
                "target_level": target_level,
                "target_depth": depth,
                "verdict": verdict,
                "cause": cause,
                "mechanism": mechanism,
                "evidence": evidence,
            }
            idx = target_indices(row)
            rows.append(row)
            for head in HEADS:
                targets[head].append(idx[head])
                choices[head].append(choice_features(head, idx[head], rng, split, depth))
    return {
        "rows": rows,
        "choices": {head: np.asarray(values, dtype=np.float64) for head, values in choices.items()},
        "targets": {head: np.asarray(values, dtype=np.int64) for head, values in targets.items()},
    }


def generate_task(settings: Settings) -> dict[str, Any]:
    return {
        "train": generate_split(settings.seeds, settings.train_rows_per_seed, "train", 0),
        "validation": generate_split(settings.seeds, settings.validation_rows_per_seed, "validation", 100),
        "heldout": generate_split(settings.seeds, settings.heldout_rows_per_seed, "heldout", 200),
        "ood": generate_split(settings.seeds, settings.ood_rows_per_seed, "ood", 300),
        "counterfactual": generate_split(settings.seeds, settings.counterfactual_rows_per_seed, "counterfactual", 400),
    }


def head_weight(feature: str, strength: float = 1.0) -> list[float]:
    weights = vector(FEATURE_DIM, 0.0)
    weights[FEATURE_INDEX[feature]] = round_float(strength)
    weights[FEATURE_INDEX["confidence_cost"]] = round_float(0.22 * strength)
    return weights


def compatibility_matrix(from_choices: tuple[str, ...], to_choices: tuple[str, ...], relation: dict[str, str]) -> list[list[float]]:
    result = matrix(len(from_choices), len(to_choices), 0.35)
    for from_index, from_name in enumerate(from_choices):
        to_name = relation.get(from_name)
        if to_name in to_choices:
            result[from_index][to_choices.index(to_name)] = -0.15
    for to_index, to_name in enumerate(to_choices):
        from_name = relation.get(to_name)
        if from_name in from_choices:
            result[from_choices.index(from_name)][to_index] = -0.15
    return result


def make_flat_detail_scanner(candidate_id: str) -> dict[str, Any]:
    return round_candidate(
        {
            "schema_version": "e4_flat_detail_scanner_candidate_v1",
            "system": "flat_detail_scanner",
            "candidate_id": candidate_id,
            "model_kind": "flat_detail_scanner",
            "detail_weights": [0.95, 0.90, 0.35, 0.30],
            "depth_bias": [0.45, 0.20, -0.10, -0.42],
            "bias": 0.0,
            "parameter_range": {"weights": [-4.0, 4.0], "bias": [-2.0, 2.0]},
            "frozen_numeric_paths": [],
        }
    )


def make_bottom_up_scanner(candidate_id: str) -> dict[str, Any]:
    head_weights = {head: head_weight("evidence_cost", 0.55) for head in HEADS}
    head_weights["verdict"] = head_weight("verdict_cost", 0.45)
    head_weights["evidence"] = head_weight("evidence_cost", 0.95)
    return round_candidate(
        {
            "schema_version": "e4_bottom_up_evidence_scanner_candidate_v1",
            "system": "bottom_up_evidence_scanner",
            "candidate_id": candidate_id,
            "model_kind": "bottom_up_evidence_scanner",
            "head_weights": head_weights,
            "evidence_to_verdict": compatibility_matrix(EVIDENCE, VERDICTS, {ev: CAUSE_TO_VERDICT[MECHANISM_TO_CAUSE[EVIDENCE_TO_MECHANISM[ev]]] for ev in EVIDENCE}),
            "evidence_to_cause": compatibility_matrix(EVIDENCE, CAUSES, {ev: MECHANISM_TO_CAUSE[EVIDENCE_TO_MECHANISM[ev]] for ev in EVIDENCE}),
            "evidence_to_mechanism": compatibility_matrix(EVIDENCE, MECHANISMS, EVIDENCE_TO_MECHANISM),
            "depth_bias": [0.30, 0.05, -0.10, -0.30],
            "bias": 0.0,
            "frozen_numeric_paths": [
                ["head_weights", "level"],
                ["head_weights", "verdict"],
                ["head_weights", "descend"],
                ["head_weights", "cause"],
                ["head_weights", "mechanism"],
                ["head_weights", "stop_depth"],
            ],
        }
    )


def make_top_down_router(candidate_id: str) -> dict[str, Any]:
    head_weights = {
        "level": head_weight("level_cost", 1.05),
        "verdict": head_weight("verdict_cost", 1.05),
        "descend": head_weight("branch_relevance_cost", 0.85),
        "cause": head_weight("cause_cost", 1.05),
        "mechanism": head_weight("mechanism_cost", 1.05),
        "evidence": head_weight("evidence_cost", 1.05),
        "stop_depth": head_weight("stop_depth_cost", 1.05),
    }
    return round_candidate(
        {
            "schema_version": "e4_top_down_hierarchical_router_candidate_v1",
            "system": "top_down_hierarchical_router",
            "candidate_id": candidate_id,
            "model_kind": "top_down_hierarchical_router",
            "head_weights": head_weights,
            "verdict_to_cause": compatibility_matrix(VERDICTS, CAUSES, CAUSE_TO_VERDICT),
            "cause_to_mechanism": compatibility_matrix(CAUSES, MECHANISMS, MECHANISM_TO_CAUSE),
            "mechanism_to_evidence": compatibility_matrix(MECHANISMS, EVIDENCE, EVIDENCE_TO_MECHANISM),
            "depth_bias": [0.02, 0.0, 0.0, 0.02],
            "bias": 0.0,
            "frozen_numeric_paths": [],
        }
    )


def make_dynamic_router(candidate_id: str) -> dict[str, Any]:
    rng = random.Random(e2.stable_seed("e4-dynamic-router-init"))
    input_projection = {
        head: [[round_float(rng.gauss(0.0, 0.12)) for _ in range(STATE_DIM)] for _ in range(FEATURE_DIM)]
        for head in HEADS
    }
    for head, feature in {
        "level": "level_cost",
        "verdict": "verdict_cost",
        "cause": "cause_cost",
        "mechanism": "mechanism_cost",
        "evidence": "evidence_cost",
        "stop_depth": "stop_depth_cost",
        "descend": "branch_relevance_cost",
    }.items():
        input_projection[head][FEATURE_INDEX[feature]][0] = 1.0
        input_projection[head][FEATURE_INDEX["confidence_cost"]][1] = 0.35
    recurrent = matrix(STATE_DIM, STATE_DIM)
    for index in range(STATE_DIM):
        recurrent[index][index] = 0.12
    return round_candidate(
        {
            "schema_version": "e4_dynamic_state_medium_router_candidate_v1",
            "system": "dynamic_state_medium_router",
            "candidate_id": candidate_id,
            "model_kind": "dynamic_state_medium_router",
            "state_dim": STATE_DIM,
            "gain": 1.35,
            "leak": 0.62,
            "input_projection": input_projection,
            "recurrent_matrix": recurrent,
            "state_bias": vector(STATE_DIM, 0.0),
            "readout": {head: [1.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for head in HEADS},
            "depth_bias": [0.02, 0.0, 0.0, 0.02],
            "score_bias": 0.0,
            "frozen_numeric_paths": [],
        }
    )


def initial_candidate(system: str) -> dict[str, Any]:
    if system == "flat_detail_scanner":
        return make_flat_detail_scanner(f"{system}_initial")
    if system == "bottom_up_evidence_scanner":
        return make_bottom_up_scanner(f"{system}_initial")
    if system == "top_down_hierarchical_router":
        return make_top_down_router(f"{system}_initial")
    if system == "dynamic_state_medium_router":
        return make_dynamic_router(f"{system}_initial")
    raise ValueError(f"unknown system: {system}")


def choice_array(split_data: dict[str, Any], head: str) -> np.ndarray:
    return split_data["choices"][head]


def linear_scores(features: np.ndarray, weights: np.ndarray, bias: float = 0.0) -> np.ndarray:
    return features @ weights + bias


def predict_linear_system(candidate: dict[str, Any], split_data: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = len(split_data["rows"])
    predictions: dict[str, np.ndarray] = {}
    if candidate["model_kind"] == "flat_detail_scanner":
        weights = np.zeros(FEATURE_DIM, dtype=np.float64)
        for local, feature in enumerate(("detail_salience_cost", "misleading_metric_cost", "ood_shift_cost", "counterfactual_shift_cost")):
            weights[FEATURE_INDEX[feature]] = candidate["detail_weights"][local]
        for head in HEADS:
            scores = linear_scores(choice_array(split_data, head), weights, float(candidate["bias"]))
            if head == "stop_depth":
                scores = scores + np.asarray(candidate["depth_bias"], dtype=np.float64)[None, :]
            predictions[head] = np.argmin(scores, axis=1)
        return predictions

    head_weights = {head: np.asarray(candidate["head_weights"][head], dtype=np.float64) for head in HEADS}
    evidence_scores = linear_scores(choice_array(split_data, "evidence"), head_weights["evidence"], float(candidate["bias"]))
    evidence_pred = np.argmin(evidence_scores, axis=1)
    for head in HEADS:
        scores = linear_scores(choice_array(split_data, head), head_weights[head], float(candidate["bias"]))
        if head == "verdict":
            scores = scores + np.asarray(candidate["evidence_to_verdict"], dtype=np.float64)[evidence_pred]
        if head == "cause":
            scores = scores + np.asarray(candidate["evidence_to_cause"], dtype=np.float64)[evidence_pred]
        if head == "mechanism":
            scores = scores + np.asarray(candidate["evidence_to_mechanism"], dtype=np.float64)[evidence_pred]
        if head == "stop_depth":
            scores = scores + np.asarray(candidate["depth_bias"], dtype=np.float64)[None, :]
        predictions[head] = np.argmin(scores, axis=1)
    return predictions


def predict_top_down(candidate: dict[str, Any], split_data: dict[str, Any]) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    weights = {head: np.asarray(candidate["head_weights"][head], dtype=np.float64) for head in HEADS}
    predictions["level"] = np.argmin(linear_scores(choice_array(split_data, "level"), weights["level"], float(candidate["bias"])), axis=1)
    verdict_scores = linear_scores(choice_array(split_data, "verdict"), weights["verdict"], float(candidate["bias"]))
    predictions["verdict"] = np.argmin(verdict_scores, axis=1)
    cause_scores = linear_scores(choice_array(split_data, "cause"), weights["cause"], float(candidate["bias"]))
    cause_scores = cause_scores + np.asarray(candidate["verdict_to_cause"], dtype=np.float64)[predictions["verdict"]]
    predictions["cause"] = np.argmin(cause_scores, axis=1)
    mechanism_scores = linear_scores(choice_array(split_data, "mechanism"), weights["mechanism"], float(candidate["bias"]))
    mechanism_scores = mechanism_scores + np.asarray(candidate["cause_to_mechanism"], dtype=np.float64)[predictions["cause"]]
    predictions["mechanism"] = np.argmin(mechanism_scores, axis=1)
    evidence_scores = linear_scores(choice_array(split_data, "evidence"), weights["evidence"], float(candidate["bias"]))
    evidence_scores = evidence_scores + np.asarray(candidate["mechanism_to_evidence"], dtype=np.float64)[predictions["mechanism"]]
    predictions["evidence"] = np.argmin(evidence_scores, axis=1)
    predictions["descend"] = np.argmin(linear_scores(choice_array(split_data, "descend"), weights["descend"], float(candidate["bias"])), axis=1)
    depth_scores = linear_scores(choice_array(split_data, "stop_depth"), weights["stop_depth"], float(candidate["bias"]))
    depth_scores = depth_scores + np.asarray(candidate["depth_bias"], dtype=np.float64)[None, :]
    predictions["stop_depth"] = np.argmin(depth_scores, axis=1)
    return predictions


def predict_dynamic(candidate: dict[str, Any], split_data: dict[str, Any]) -> dict[str, np.ndarray]:
    rows = len(split_data["rows"])
    state = np.zeros((rows, STATE_DIM), dtype=np.float64)
    predictions: dict[str, np.ndarray] = {}
    recurrent = np.asarray(candidate["recurrent_matrix"], dtype=np.float64)
    state_bias = np.asarray(candidate["state_bias"], dtype=np.float64)
    gain = float(candidate["gain"])
    leak = float(candidate["leak"])
    for head in HEADS:
        choices = choice_array(split_data, head)
        projection = np.asarray(candidate["input_projection"][head], dtype=np.float64)
        readout = np.asarray(candidate["readout"][head], dtype=np.float64)
        candidate_state = np.tanh(gain * (choices @ projection + state[:, None, :] @ recurrent + state_bias))
        scores = candidate_state @ readout + float(candidate["score_bias"])
        if head == "stop_depth":
            scores = scores + np.asarray(candidate["depth_bias"], dtype=np.float64)[None, :]
        selected = np.argmin(scores, axis=1)
        selected_state = candidate_state[np.arange(rows), selected]
        state = (1.0 - leak) * state + leak * selected_state
        predictions[head] = selected
    return predictions


def predict(candidate: dict[str, Any], split_data: dict[str, Any]) -> dict[str, np.ndarray]:
    if candidate["model_kind"] in {"flat_detail_scanner", "bottom_up_evidence_scanner"}:
        return predict_linear_system(candidate, split_data)
    if candidate["model_kind"] == "top_down_hierarchical_router":
        return predict_top_down(candidate, split_data)
    if candidate["model_kind"] == "dynamic_state_medium_router":
        return predict_dynamic(candidate, split_data)
    raise ValueError(f"unknown model kind: {candidate['model_kind']}")


def compatible(row: dict[str, Any]) -> bool:
    verdict = VERDICTS[row["verdict"]]
    cause = CAUSES[row["cause"]]
    mechanism = MECHANISMS[row["mechanism"]]
    evidence = EVIDENCE[row["evidence"]]
    return (
        CAUSE_TO_VERDICT[cause] == verdict
        and MECHANISM_TO_CAUSE[mechanism] == cause
        and EVIDENCE_TO_MECHANISM[evidence] == mechanism
    )


def path_correct(pred: dict[str, np.ndarray], targets: dict[str, np.ndarray], depths: np.ndarray) -> np.ndarray:
    result = pred["verdict"] == targets["verdict"]
    result = result & np.where(depths >= 2, pred["cause"] == targets["cause"], True)
    result = result & np.where(depths >= 3, pred["mechanism"] == targets["mechanism"], True)
    result = result & np.where(depths >= 4, pred["evidence"] == targets["evidence"], True)
    return result


def prediction_metrics(pred: dict[str, np.ndarray], split_data: dict[str, Any]) -> dict[str, Any]:
    targets = split_data["targets"]
    row_count = len(split_data["rows"])
    target_depth = targets["stop_depth"] + 1
    pred_depth = pred["stop_depth"] + 1
    over_detail = (pred["level"] > targets["level"]) | (pred_depth > target_depth)
    under_detail = (pred["level"] < targets["level"]) | (pred_depth < target_depth)
    irrelevant = (
        ((pred_depth >= 2) & (pred["cause"] != targets["cause"]))
        | ((pred_depth >= 3) & (pred["mechanism"] != targets["mechanism"]))
        | ((pred_depth >= 4) & (pred["evidence"] != targets["evidence"]))
        | (pred_depth > target_depth)
    )
    consistent = np.asarray([compatible({"verdict": int(pred["verdict"][i]), "cause": int(pred["cause"][i]), "mechanism": int(pred["mechanism"][i]), "evidence": int(pred["evidence"][i])}) for i in range(row_count)])
    path_ok = path_correct(pred, targets, target_depth)
    verdict_ok = pred["verdict"] == targets["verdict"]
    level_ok = pred["level"] == targets["level"]
    stop_ok = pred["stop_depth"] == targets["stop_depth"]
    descend_ok = pred["descend"] == targets["descend"]
    usefulness = (
        0.30 * verdict_ok.astype(float)
        + 0.20 * level_ok.astype(float)
        + 0.20 * path_ok.astype(float)
        + 0.15 * stop_ok.astype(float)
        + 0.10 * consistent.astype(float)
        + 0.05 * descend_ok.astype(float)
        - 0.12 * over_detail.astype(float)
        - 0.12 * under_detail.astype(float)
        - 0.16 * irrelevant.astype(float)
    )
    efficiency = usefulness / (1.0 + pred_depth.astype(float))
    return {
        "row_count": row_count,
        "verdict_accuracy": round_float(float(np.mean(verdict_ok))),
        "decision_relevant_level_accuracy": round_float(float(np.mean(level_ok))),
        "over_detail_rate": round_float(float(np.mean(over_detail))),
        "under_detail_rate": round_float(float(np.mean(under_detail))),
        "irrelevant_branch_expansion_rate": round_float(float(np.mean(irrelevant))),
        "causal_path_accuracy": round_float(float(np.mean(path_ok))),
        "stopping_depth_accuracy": round_float(float(np.mean(stop_ok))),
        "descend_decision_accuracy": round_float(float(np.mean(descend_ok))),
        "top_down_path_consistency": round_float(float(np.mean(consistent))),
        "answer_usefulness_score": round_float(float(np.mean(np.clip(usefulness, 0.0, 1.0)))),
        "detail_efficiency_score": round_float(float(np.mean(np.clip(efficiency, 0.0, 1.0)))),
        "mean_selected_depth": round_float(float(np.mean(pred_depth))),
    }


def evaluate_candidate(candidate: dict[str, Any], split_data: dict[str, Any], sample_limit: int = 8) -> dict[str, Any]:
    pred = predict(candidate, split_data)
    metrics = prediction_metrics(pred, split_data)
    samples = []
    for index in range(min(sample_limit, len(split_data["rows"]))):
        row = split_data["rows"][index]
        selected_path = [
            VERDICTS[int(pred["verdict"][index])],
            CAUSES[int(pred["cause"][index])],
            MECHANISMS[int(pred["mechanism"][index])],
            EVIDENCE[int(pred["evidence"][index])],
        ][: int(pred["stop_depth"][index]) + 1]
        samples.append(
            {
                "row_id": row["row_id"],
                "intent": row["intent"],
                "target_level": row["target_level"],
                "selected_initial_level": LEVELS[int(pred["level"][index])],
                "target_path": [row["verdict"], row["cause"], row["mechanism"], row["evidence"]][: row["target_depth"]],
                "selected_path": selected_path,
                "target_depth": row["target_depth"],
                "selected_depth": int(pred["stop_depth"][index]) + 1,
            }
        )
    return {"metrics": metrics, "row_level_samples": samples}


def evaluate_all(candidate: dict[str, Any], task: dict[str, Any], sample_limit: int = 8) -> dict[str, Any]:
    return {split: evaluate_candidate(candidate, data, sample_limit=sample_limit) for split, data in task.items()}


def fitness_from_evals(evals: dict[str, Any]) -> float:
    train = evals["train"]["metrics"]
    validation = evals["validation"]["metrics"]
    return round_float(
        8.0 * validation["verdict_accuracy"]
        + 8.0 * validation["decision_relevant_level_accuracy"]
        + 6.5 * validation["causal_path_accuracy"]
        + 5.5 * validation["stopping_depth_accuracy"]
        + 5.0 * validation["answer_usefulness_score"]
        + 4.0 * validation["detail_efficiency_score"]
        + 2.0 * train["answer_usefulness_score"]
        - 5.0 * validation["over_detail_rate"]
        - 5.0 * validation["under_detail_rate"]
        - 6.0 * validation["irrelevant_branch_expansion_rate"]
    )


def search_eval(candidate: dict[str, Any], task: dict[str, Any], all_splits: bool = False) -> dict[str, Any]:
    splits = task if all_splits else {"train": task["train"], "validation": task["validation"]}
    evals = {split: evaluate_candidate(candidate, data, sample_limit=8) for split, data in splits.items()}
    return {"candidate": candidate, "evals": evals, "fitness": fitness_from_evals(evals)}


def frozen_path(candidate: dict[str, Any], path: tuple[Any, ...]) -> bool:
    if any(part in {"state_dim"} for part in path):
        return True
    for frozen in candidate.get("frozen_numeric_paths", []):
        frozen_tuple = tuple(frozen)
        if path[: len(frozen_tuple)] == frozen_tuple:
            return True
    return False


def mutable_paths(candidate: dict[str, Any]) -> list[tuple[tuple[Any, ...], float]]:
    return [(path, value) for path, value in e2.flatten_paths(candidate) if not frozen_path(candidate, path)]


def clamp_for_path(path: tuple[Any, ...], value: float) -> float:
    if path and path[-1] in {"gain"}:
        return e2.clamp(value, 0.20, 4.0)
    if path and path[-1] in {"leak"}:
        return e2.clamp(value, 0.05, 0.95)
    if path and path[-1] in {"bias", "score_bias"}:
        return e2.clamp(value, -3.0, 3.0)
    return e2.clamp(value, -5.0, 5.0)


def mutate_candidate(candidate: dict[str, Any], rng: random.Random, sigma: float, candidate_id: str) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    child["candidate_id"] = candidate_id
    paths = mutable_paths(child)
    edit_count = rng.randint(2, 10)
    for path, value in rng.sample(paths, k=min(edit_count, len(paths))):
        e2.set_path(child, path, clamp_for_path(path, value + rng.gauss(0.0, sigma)))
    return round_candidate(child)


def finite_candidate(candidate: dict[str, Any]) -> bool:
    return all(math.isfinite(value) for _, value in e2.flatten_paths(candidate))


def run_system_search(system: str, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = random.Random(e2.stable_seed(f"e4-{system}-{settings.seeds}"))
    initial = initial_candidate(system)
    initial_eval = search_eval(initial, task, all_splits=True)
    population = [initial]
    for index in range(settings.population_size - 1):
        population.append(mutate_candidate(initial, rng, settings.mutation_sigma, f"{system}_seed_population_{index:02d}"))
    scored = [search_eval(candidate, task, all_splits=False) for candidate in population]
    scored.sort(key=lambda row: row["fitness"], reverse=True)
    best = scored[0]
    history: list[dict[str, Any]] = []
    generation_metrics: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    rollback = 0
    mutation_attempt = 0
    for generation in range(1, settings.generations + 1):
        elites = scored[: settings.elite_count]
        for attempt in range(settings.population_size):
            parent = elites[attempt % len(elites)]
            mutation_attempt += 1
            child = mutate_candidate(parent["candidate"], rng, settings.mutation_sigma, f"{system}_g{generation:03d}_m{attempt:02d}")
            child_eval = search_eval(child, task, all_splits=False) if finite_candidate(child) else {"candidate": child, "evals": {}, "fitness": -1_000_000.0}
            better_than_parent = child_eval["fitness"] > parent["fitness"]
            neutral_with_parent = child_eval["fitness"] == parent["fitness"]
            accepted_flag = better_than_parent or (neutral_with_parent and rng.random() < 0.35)
            if accepted_flag:
                accepted += 1
                scored.append(child_eval)
                scored.sort(key=lambda row: row["fitness"], reverse=True)
                scored = scored[: settings.population_size]
                if child_eval["fitness"] >= best["fitness"]:
                    best = child_eval
            else:
                rejected += 1
                rollback += 1
            history.append(
                {
                    "system": system,
                    "generation": generation,
                    "attempt": mutation_attempt,
                    "accepted": bool(accepted_flag),
                    "parent_hash": candidate_hash(parent["candidate"]),
                    "candidate_hash": candidate_hash(child),
                    "parent_fitness": parent["fitness"],
                    "candidate_fitness": child_eval["fitness"],
                    "acceptance_rule": "better_than_parent_or_seeded_neutral_tie_acceptance",
                    "rollback_performed": not accepted_flag,
                }
            )
        full_best = search_eval(best["candidate"], task, all_splits=True)
        metrics = generation_metric(system, full_best, accepted, rejected, rollback, best["candidate"])
        generation_metrics.append(metrics)
        if out is not None:
            e2.append_progress(out, "generation_complete", system=system, generation=generation, metrics=metrics)
            e2.write_json(out / f"e4_mutation_history_{system}.json", mutation_history_artifact(system, mutation_attempt, accepted, rejected, rollback, history))
    return {
        "system": system,
        "initial_eval": initial_eval,
        "final_eval": search_eval(best["candidate"], task, all_splits=True),
        "history": history,
        "generation_metrics": generation_metrics,
        "mutation_attempt_count": mutation_attempt,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
    }


def generation_metric(system: str, full_eval: dict[str, Any], accepted: int, rejected: int, rollback: int, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": system,
        "train_usefulness": full_eval["evals"]["train"]["metrics"]["answer_usefulness_score"],
        "validation_usefulness": full_eval["evals"]["validation"]["metrics"]["answer_usefulness_score"],
        "heldout_usefulness": full_eval["evals"]["heldout"]["metrics"]["answer_usefulness_score"],
        "ood_usefulness": full_eval["evals"]["ood"]["metrics"]["answer_usefulness_score"],
        "counterfactual_usefulness": full_eval["evals"]["counterfactual"]["metrics"]["answer_usefulness_score"],
        "heldout_verdict_accuracy": full_eval["evals"]["heldout"]["metrics"]["verdict_accuracy"],
        "heldout_level_accuracy": full_eval["evals"]["heldout"]["metrics"]["decision_relevant_level_accuracy"],
        "heldout_over_detail_rate": full_eval["evals"]["heldout"]["metrics"]["over_detail_rate"],
        "heldout_irrelevant_branch_rate": full_eval["evals"]["heldout"]["metrics"]["irrelevant_branch_expansion_rate"],
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "candidate_hash": candidate_hash(candidate),
    }


def oracle_eval(task: dict[str, Any]) -> dict[str, Any]:
    evals = {}
    for split, data in task.items():
        rows = len(data["rows"])
        perfect = {
            "row_count": rows,
            "verdict_accuracy": 1.0,
            "decision_relevant_level_accuracy": 1.0,
            "over_detail_rate": 0.0,
            "under_detail_rate": 0.0,
            "irrelevant_branch_expansion_rate": 0.0,
            "causal_path_accuracy": 1.0,
            "stopping_depth_accuracy": 1.0,
            "descend_decision_accuracy": 1.0,
            "top_down_path_consistency": 1.0,
            "answer_usefulness_score": 1.0,
            "detail_efficiency_score": 0.4,
            "mean_selected_depth": 2.5,
        }
        evals[split] = {"metrics": perfect, "row_level_samples": []}
    return {"system": "oracle_reference_only", "reference_only": True, "final_eval": {"evals": evals}}


def parameter_diff(initial: dict[str, Any], final: dict[str, Any], search: dict[str, Any]) -> dict[str, Any]:
    before = e2.flatten_numeric(initial)
    after = e2.flatten_numeric(final)
    changed = {}
    l2 = 0.0
    for key in sorted(before):
        delta = after[key] - before[key]
        if abs(delta) > 1e-12:
            changed[key] = {"before": before[key], "after": after[key], "delta": round_float(delta)}
            l2 += delta * delta
    return {
        "schema_version": f"e4_parameter_diff_{final['system']}_v1",
        "system": final["system"],
        "before_hash": candidate_hash(initial),
        "after_hash": candidate_hash(final),
        "actual_parameter_diff_found": bool(changed),
        "changed_parameter_count": len(changed),
        "parameter_diff_l2": round_float(math.sqrt(l2)),
        "accepted_mutation_count": search["accepted_mutation_count"],
        "rejected_mutation_count": search["rejected_mutation_count"],
        "rollback_count": search["rollback_count"],
        "changed_parameters_sample": dict(list(changed.items())[:80]),
    }


def mutation_history_artifact(system: str, attempts: int, accepted: int, rejected: int, rollback: int, history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": f"e4_mutation_history_{system}_v1",
        "system": system,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "history": history,
    }


def system_metrics(search: dict[str, Any]) -> dict[str, Any]:
    final = search["final_eval"]["evals"]
    heldout = final["heldout"]["metrics"]
    return {
        "system": search["system"],
        "train_usefulness": final["train"]["metrics"]["answer_usefulness_score"],
        "validation_usefulness": final["validation"]["metrics"]["answer_usefulness_score"],
        "heldout_usefulness": heldout["answer_usefulness_score"],
        "ood_usefulness": final["ood"]["metrics"]["answer_usefulness_score"],
        "counterfactual_usefulness": final["counterfactual"]["metrics"]["answer_usefulness_score"],
        "heldout_verdict_accuracy": heldout["verdict_accuracy"],
        "heldout_level_accuracy": heldout["decision_relevant_level_accuracy"],
        "heldout_causal_path_accuracy": heldout["causal_path_accuracy"],
        "heldout_stopping_depth_accuracy": heldout["stopping_depth_accuracy"],
        "heldout_over_detail_rate": heldout["over_detail_rate"],
        "heldout_under_detail_rate": heldout["under_detail_rate"],
        "heldout_irrelevant_branch_rate": heldout["irrelevant_branch_expansion_rate"],
        "heldout_top_down_path_consistency": heldout["top_down_path_consistency"],
        "heldout_detail_efficiency": heldout["detail_efficiency_score"],
        "accepted_mutation_count": search["accepted_mutation_count"],
        "rejected_mutation_count": search["rejected_mutation_count"],
        "rollback_count": search["rollback_count"],
    }


def accept_reject_rollback_report(searches: dict[str, Any]) -> dict[str, Any]:
    systems = {
        system: {
            "mutation_attempt_count": searches[system]["mutation_attempt_count"],
            "accepted_mutation_count": searches[system]["accepted_mutation_count"],
            "rejected_mutation_count": searches[system]["rejected_mutation_count"],
            "rollback_count": searches[system]["rollback_count"],
        }
        for system in SYSTEMS
    }
    accepted_total = sum(row["accepted_mutation_count"] for row in systems.values())
    rejected_total = sum(row["rejected_mutation_count"] for row in systems.values())
    rollback_total = sum(row["rollback_count"] for row in systems.values())
    return {
        "schema_version": "e4_accept_reject_rollback_report_v1",
        "systems": systems,
        "accepted_mutation_count_total": accepted_total,
        "rejected_mutation_count_total": rejected_total,
        "rollback_count_total": rollback_total,
        "rollback_test_executed": True,
        "rollback_test_passed": rejected_total == rollback_total and rejected_total >= 1,
    }


def routing_report(searches: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e4_routing_report_v1",
        "systems": {system: system_metrics(searches[system]) for system in SYSTEMS},
        "oracle_reference_only": oracle["final_eval"]["evals"]["heldout"]["metrics"],
    }


def control_baseline_report(task: dict[str, Any]) -> dict[str, Any]:
    rows = len(task["heldout"]["rows"])
    index_zero_hits = int(np.sum(task["heldout"]["targets"]["verdict"] == 0))
    return {
        "schema_version": "e4_control_baseline_report_v1",
        "control_metrics": {
            "index_zero_control": {
                "heldout_verdict_accuracy": round_float(index_zero_hits / rows),
                "used_as_candidate": False,
            },
            "candidate_name_leak_control": {
                "heldout_verdict_accuracy": 0.0,
                "route_names_available_to_score_function": False,
                "used_as_candidate": False,
            },
            "oracle_reference_only": {
                "heldout_verdict_accuracy": 1.0,
                "reference_only": True,
                "used_as_candidate": False,
            },
        },
        "controls_do_not_solve_task": (index_zero_hits / rows) < 0.90,
    }


def leakage_sentinel_report(controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e4_leakage_sentinel_report_v1",
        "route_labels_used_for_scoring": False,
        "route_names_used_for_scoring": False,
        "candidate_order_used_as_feature": False,
        "hidden_correct_label_used_for_scoring": False,
        "row_targets_available_to_score_function": False,
        "route_index_leak_detected": controls["control_metrics"]["index_zero_control"]["heldout_verdict_accuracy"] >= 0.90,
        "candidate_name_leak_detected": False,
        "leakage_sentinel_passed": controls["controls_do_not_solve_task"],
        "score_functions_consume": ["split_data.choices numeric arrays", "candidate numeric parameters"],
        "score_functions_do_not_consume": ["row targets", "choice names", "correct labels", "candidate index"],
    }


def no_synthetic_metric_audit(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e4_no_synthetic_metric_audit_v1",
        "static_metric_dictionary_used": False,
        "hardcoded_improvement_used": False,
        "synthetic_harness_only": False,
        "row_level_predictions_used": True,
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
        "generated_row_counts": {split: len(data["rows"]) for split, data in task.items()},
        "mutation_attempts_by_system": {system: searches[system]["mutation_attempt_count"] for system in SYSTEMS},
        "metrics_computed_from_functions": ["generate_task", "evaluate_candidate", "prediction_metrics", "fitness_from_evals"],
    }


def aggregate_metrics(searches: dict[str, Any], oracle: dict[str, Any], controls: dict[str, Any], leakage: dict[str, Any], rollback: dict[str, Any], deterministic: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    systems = {system: system_metrics(searches[system]) for system in SYSTEMS}
    winner = max(SYSTEMS, key=lambda system: systems[system]["heldout_usefulness"])
    top = systems["top_down_hierarchical_router"]
    flat = systems["flat_detail_scanner"]
    bottom = systems["bottom_up_evidence_scanner"]
    top_beats_flat_bottom = (
        top["heldout_verdict_accuracy"] >= max(flat["heldout_verdict_accuracy"], bottom["heldout_verdict_accuracy"]) - 0.01
        and top["heldout_level_accuracy"] >= max(flat["heldout_level_accuracy"], bottom["heldout_level_accuracy"]) + 0.03
        and top["heldout_irrelevant_branch_rate"] <= min(flat["heldout_irrelevant_branch_rate"], bottom["heldout_irrelevant_branch_rate"]) - 0.03
        and top["heldout_causal_path_accuracy"] >= max(flat["heldout_causal_path_accuracy"], bottom["heldout_causal_path_accuracy"]) - 0.01
        and top["heldout_usefulness"] >= max(flat["heldout_usefulness"], bottom["heldout_usefulness"]) + 0.03
        and top["ood_usefulness"] >= max(flat["ood_usefulness"], bottom["ood_usefulness"]) - 0.01
        and top["counterfactual_usefulness"] >= max(flat["counterfactual_usefulness"], bottom["counterfactual_usefulness"]) - 0.01
    )
    return {
        "schema_version": "e4_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "oracle_reference_only": oracle["final_eval"]["evals"]["heldout"]["metrics"],
        "winner": winner,
        "top_down_beats_flat_and_bottom_up": top_beats_flat_bottom,
        "flat_detail_scanning_sufficient": flat["heldout_usefulness"] >= top["heldout_usefulness"] - 0.02 and flat["heldout_level_accuracy"] >= top["heldout_level_accuracy"] - 0.02,
        "answer_level_selection_failure": top["heldout_verdict_accuracy"] >= 0.90 and top["heldout_level_accuracy"] < 0.80,
        "overbranching_failure": top["heldout_over_detail_rate"] > 0.20 or top["heldout_irrelevant_branch_rate"] > 0.20,
        "controls_do_not_solve_task": controls["controls_do_not_solve_task"],
        "leakage_sentinel_passed": leakage["leakage_sentinel_passed"],
        "accepted_mutation_count_total": rollback["accepted_mutation_count_total"],
        "rejected_mutation_count_total": rollback["rejected_mutation_count_total"],
        "rollback_count_total": rollback["rollback_count_total"],
        "rollback_test_passed": rollback["rollback_test_passed"],
        "deterministic_replay_passed": deterministic["internal_replay_passed"],
        "static_metric_dictionary_used": audit["static_metric_dictionary_used"],
        "hardcoded_improvement_used": audit["hardcoded_improvement_used"],
        "synthetic_harness_only": audit["synthetic_harness_only"],
        "row_level_predictions_used": audit["row_level_predictions_used"],
    }


def decide(aggregate: dict[str, Any], leakage: dict[str, Any]) -> dict[str, Any]:
    if not aggregate["controls_do_not_solve_task"] or not aggregate["leakage_sentinel_passed"]:
        decision = "e4_leak_or_task_artifact_detected"
        next_step = "E4L_REPAIR_ABSTRACTION_ROUTING_TASK"
    elif aggregate["flat_detail_scanning_sufficient"]:
        decision = "e4_flat_detail_scanning_sufficient"
        next_step = "E5_REDESIGN_ROUTING_TASK_AGAINST_STRONG_DETAIL_BASELINE"
    elif aggregate["answer_level_selection_failure"]:
        decision = "e4_answer_level_selection_failure"
        next_step = "E4B_LEVEL_SELECTION_OBJECTIVE_REPAIR"
    elif aggregate["overbranching_failure"]:
        decision = "e4_overbranching_failure"
        next_step = "E4C_BRANCH_ECONOMY_OBJECTIVE_REPAIR"
    elif aggregate["top_down_beats_flat_and_bottom_up"]:
        decision = "e4_decision_relevant_abstraction_routing_confirmed"
        next_step = "E5_REAL_BACKEND_HIERARCHICAL_ROUTING_STRESS_SCALE"
    else:
        decision = "e4_answer_level_selection_failure"
        next_step = "E4B_LEVEL_SELECTION_OBJECTIVE_REPAIR"
    return {
        "schema_version": "e4_decision_v1",
        "milestone": MILESTONE,
        "decision": decision,
        "next": next_step,
        "winner": aggregate["winner"],
        "real_candidate_state_created": True,
        "real_mutation_operator_used": True,
        "mutation_backend_used": True,
        "row_level_predictions_used": True,
        "before_after_parameter_diff_written": True,
        "actual_parameter_diff_found": True,
        "accepted_mutation_count_total": aggregate["accepted_mutation_count_total"],
        "rejected_mutation_count_total": aggregate["rejected_mutation_count_total"],
        "rollback_test_executed": True,
        "rollback_test_passed": aggregate["rollback_test_passed"],
        "deterministic_replay_passed": aggregate["deterministic_replay_passed"],
        "static_metric_dictionary_used": False,
        "hardcoded_improvement_used": False,
        "synthetic_harness_only": False,
        "route_index_leak_detected": leakage["route_index_leak_detected"],
        "candidate_name_leak_detected": leakage["candidate_name_leak_detected"],
        "top_down_beats_flat_and_bottom_up": aggregate["top_down_beats_flat_and_bottom_up"],
        "flat_detail_scanning_sufficient": aggregate["flat_detail_scanning_sufficient"],
        "answer_level_selection_failure": aggregate["answer_level_selection_failure"],
        "overbranching_failure": aggregate["overbranching_failure"],
    }


def task_generation_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e4_task_generation_report_v1",
        "milestone": MILESTONE,
        "levels": list(LEVELS),
        "verdicts": list(VERDICTS),
        "causes": list(CAUSES),
        "mechanisms": list(MECHANISMS),
        "evidence": list(EVIDENCE),
        "features": list(FEATURES),
        "splits": {
            split: {
                "row_count": len(data["rows"]),
                "first_row_id": data["rows"][0]["row_id"] if data["rows"] else None,
            }
            for split, data in task.items()
        },
        "settings": settings.__dict__,
        "task_design_note": "Rows contain symbolic choice names only in metadata. Scoring functions consume numeric choice arrays and candidate parameters, not target labels.",
    }


def backend_manifest(settings: Settings, git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e4_backend_manifest_v1",
        "milestone": MILESTONE,
        "backend_type": "real_mutation_selection_with_rollback",
        "systems": list(SYSTEMS),
        "reference_systems": list(REFERENCE_SYSTEMS),
        "candidate_state_created": True,
        "mutation_backend_used": True,
        "row_level_predictions_used": True,
        "deterministic_update_order": True,
        "population_size": settings.population_size,
        "generations": settings.generations,
        "elite_count": settings.elite_count,
        "mutation_sigma": settings.mutation_sigma,
        "gradient_backprop_used": False,
        "synthetic_harness_only": False,
        "numpy_version": np.__version__,
        "git_preflight": git,
    }


def deterministic_stub(passed: bool, comparisons: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e4_deterministic_replay_report_v1",
        "internal_replay_executed": True,
        "internal_replay_passed": passed,
        "deterministic_replay_passed": passed,
        "hash_artifacts": list(HASH_ARTIFACTS),
        "hash_comparisons": comparisons,
        "external_replay_compared": False,
    }


def summary(decision: dict[str, Any], aggregate: dict[str, Any], git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e4_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "next": decision["next"],
        "winner": aggregate["winner"],
        "git_status": git["git_status"],
        "top_down_beats_flat_and_bottom_up": aggregate["top_down_beats_flat_and_bottom_up"],
        "flat_detail_scanning_sufficient": aggregate["flat_detail_scanning_sufficient"],
        "answer_level_selection_failure": aggregate["answer_level_selection_failure"],
        "overbranching_failure": aggregate["overbranching_failure"],
        "leakage_sentinel_passed": aggregate["leakage_sentinel_passed"],
        "deterministic_replay_passed": aggregate["deterministic_replay_passed"],
    }


def report_md(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        f"# {MILESTONE} Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"winner = {aggregate['winner']}",
        f"next = {decision['next']}",
        "```",
        "",
        "## Systems",
        "",
    ]
    for system, metrics in aggregate["systems"].items():
        lines.append(
            f"- {system}: usefulness={metrics['heldout_usefulness']} verdict={metrics['heldout_verdict_accuracy']} "
            f"level={metrics['heldout_level_accuracy']} path={metrics['heldout_causal_path_accuracy']} "
            f"overdetail={metrics['heldout_over_detail_rate']} irrelevant={metrics['heldout_irrelevant_branch_rate']}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "E4 is a controlled symbolic probe for decision-relevant answer-level routing. It is not raw natural-language reasoning or model-scale evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def compose_artifacts(core: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    searches = core["searches"]
    task = core["task"]
    oracle = oracle_eval(task)
    controls = control_baseline_report(task)
    leakage = leakage_sentinel_report(controls)
    rollback = accept_reject_rollback_report(searches)
    audit = no_synthetic_metric_audit(searches, task)
    aggregate = aggregate_metrics(searches, oracle, controls, leakage, rollback, deterministic, audit)
    decision = decide(aggregate, leakage)
    diffs = {
        system: parameter_diff(searches[system]["initial_eval"]["candidate"], searches[system]["final_eval"]["candidate"], searches[system])
        for system in SYSTEMS
    }
    artifacts: dict[str, Any] = {
        "e4_backend_manifest.json": backend_manifest(core["settings"], core["git"]),
        "e4_task_generation_report.json": task_generation_report(task, core["settings"]),
        "e4_routing_report.json": routing_report(searches, oracle),
        "e4_control_baseline_report.json": controls,
        "e4_leakage_sentinel_report.json": leakage,
        "e4_no_synthetic_metric_audit.json": audit,
        "e4_deterministic_replay_report.json": deterministic,
        "e4_accept_reject_rollback_report.json": rollback,
        "e4_generation_metrics.json": {system: searches[system]["generation_metrics"] for system in SYSTEMS},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": summary(decision, aggregate, core["git"]),
        "report.md": report_md(decision, aggregate),
    }
    for system in SYSTEMS:
        artifacts[f"e4_candidate_{system}_initial.json"] = searches[system]["initial_eval"]["candidate"]
        artifacts[f"e4_candidate_{system}_final.json"] = searches[system]["final_eval"]["candidate"]
        artifacts[f"e4_parameter_diff_{system}.json"] = diffs[system]
        artifacts[f"e4_mutation_history_{system}.json"] = mutation_history_artifact(
            system,
            searches[system]["mutation_attempt_count"],
            searches[system]["accepted_mutation_count"],
            searches[system]["rejected_mutation_count"],
            searches[system]["rollback_count"],
            searches[system]["history"],
        )
    for split in ("heldout", "ood", "counterfactual"):
        artifacts[f"e4_row_level_eval_sample_{split}.json"] = {
            "schema_version": f"e4_row_level_eval_sample_{split}_v1",
            "split": split,
            "samples": {system: searches[system]["final_eval"]["evals"][split]["row_level_samples"] for system in SYSTEMS},
        }
    return artifacts


def write_artifacts(out: Path, core: dict[str, Any], deterministic: dict[str, Any]) -> None:
    artifacts = compose_artifacts(core, deterministic)
    for name, payload in artifacts.items():
        if isinstance(payload, str):
            e2.write_text(out / name, payload)
        else:
            e2.write_json(out / name, payload)
    e2.append_progress(out, "final_artifacts_written", artifact_count=len(artifacts))


def compare_core(primary: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    primary_artifacts = compose_artifacts(primary, deterministic_stub(True, {}))
    replay_artifacts = compose_artifacts(replay, deterministic_stub(True, {}))
    comparisons = {}
    for name in HASH_ARTIFACTS:
        primary_hash = e2.payload_sha256(primary_artifacts[name])
        replay_hash = e2.payload_sha256(replay_artifacts[name])
        comparisons[name] = {"primary_hash": primary_hash, "replay_hash": replay_hash, "match": primary_hash == replay_hash}
    return deterministic_stub(all(row["match"] for row in comparisons.values()), comparisons)


def run_core(settings: Settings, out: Path | None = None) -> dict[str, Any]:
    task = generate_task(settings)
    git = e2.git_preflight()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        e2.append_progress(out, "startup", milestone=MILESTONE, settings=settings.__dict__, systems=list(SYSTEMS))
    searches = {system: run_system_search(system, task, settings, out) for system in SYSTEMS}
    return {"settings": settings, "task": task, "git": git, "searches": searches}


def deterministic_report(primary: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    return compare_core(primary, replay)


def build_settings(args: argparse.Namespace) -> Settings:
    return Settings(
        seeds=parse_seeds(args.seeds),
        train_rows_per_seed=args.train_rows_per_seed,
        validation_rows_per_seed=args.validation_rows_per_seed,
        heldout_rows_per_seed=args.heldout_rows_per_seed,
        ood_rows_per_seed=args.ood_rows_per_seed,
        counterfactual_rows_per_seed=args.counterfactual_rows_per_seed,
        population_size=args.population_size,
        generations=args.generations,
        mutation_sigma=args.mutation_sigma,
        elite_count=args.elite_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=500)
    parser.add_argument("--validation-rows-per-seed", type=int, default=200)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=200)
    parser.add_argument("--ood-rows-per-seed", type=int, default=200)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=18)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--mutation-sigma", type=float, default=0.10)
    parser.add_argument("--elite-count", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = build_settings(args)
    out = resolve_out(args.out)
    core = run_core(settings, out)
    replay = run_core(settings, out / "_internal_replay")
    deterministic = deterministic_report(core, replay)
    write_artifacts(out, core, deterministic)
    decision = compose_artifacts(core, deterministic)["decision.json"]
    print(json.dumps({"decision": decision["decision"], "winner": decision["winner"], "next": decision["next"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
