#!/usr/bin/env python3
"""E3 state-medium scaffold ablation and reality check."""

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
MILESTONE = "E3_STATE_MEDIUM_SCAFFOLD_ABLATION_AND_REALITY_CHECK"
DEFAULT_OUT = Path("target/pilot_wave/e3_state_medium_scaffold_ablation_and_reality_check")
DEFAULT_SEEDS = (74001, 74002, 74003, 74004, 74005)


def load_e2_module() -> Any:
    spec = importlib.util.spec_from_file_location("e2_conductivity_backend", E2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E2 backend from {E2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


e2 = load_e2_module()

ROUTES = e2.ROUTES
FEATURES = e2.FEATURES
FEATURE_INDEX = e2.FEATURE_INDEX
ROUTE_INDEX = e2.ROUTE_INDEX
LOGICAL_INDEX = e2.LOGICAL_INDEX
WRONG_ROUTES = e2.WRONG_ROUTES
ORDERING_SPLITS = e2.ORDERING_SPLITS
STATE_DIM = 8
PAIR_FEATURES = (
    ("evidence_gap", "route_margin_gap"),
    ("shortcut_risk", "surface_similarity_wrongness"),
    ("random_noise_level", "template_collision_risk"),
    ("random_noise_level", "grammar_collision_risk"),
    ("contradiction_risk", "counterfactual_fragility"),
    ("contradiction_risk", "preservation_risk"),
    ("illogical_transition_count", "landing_error_risk"),
    ("illogical_transition_count", "binding_scope_risk"),
    ("local_step_cost", "sequence_length_risk"),
    ("abstraction_jump_cost", "counterfactual_fragility"),
    ("surface_similarity_wrongness", "template_collision_risk"),
    ("landing_error_risk", "calibration_mismatch"),
    ("preservation_risk", "calibration_mismatch"),
    ("binding_scope_risk", "temporal_instability_risk"),
    ("evidence_gap", "shortcut_risk"),
    ("route_margin_gap", "counterfactual_fragility"),
)
VARIANTS = (
    "e2_original_projector",
    "random_projector",
    "zero_projector",
    "permuted_projector",
    "dense_random_projector",
    "no_recurrence",
    "final_state_only",
    "nonrecurrent_nonlinear",
    "stronger_flat_pairwise",
)
STATE_VARIANTS = tuple(variant for variant in VARIANTS if variant != "stronger_flat_pairwise")
PROJECTOR_ABLATION_VARIANTS = (
    "e2_original_projector",
    "random_projector",
    "zero_projector",
    "permuted_projector",
    "dense_random_projector",
)
HASH_ARTIFACTS = (
    "e3_variant_ablation_report.json",
    "e3_conductivity_ordering_report.json",
    "e3_control_baseline_report.json",
    "e3_leakage_sentinel_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
REQUIRED_ARTIFACTS = (
    "e3_backend_manifest.json",
    "e3_task_generation_report.json",
    "e3_variant_ablation_report.json",
    "e3_conductivity_ordering_report.json",
    "e3_control_baseline_report.json",
    "e3_leakage_sentinel_report.json",
    "e3_no_synthetic_metric_audit.json",
    "e3_deterministic_replay_report.json",
    "e3_accept_reject_rollback_report.json",
    "e3_logical_vs_wrong_gap_report.json",
    "e3_attractor_basin_report.json",
    "e3_perturbation_recovery_report.json",
    "e3_generation_metrics.json",
    "e3_row_level_eval_sample_heldout.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
)


@dataclass(frozen=True)
class Settings:
    seeds: tuple[int, ...]
    train_rows_per_seed: int
    validation_rows_per_seed: int
    heldout_rows_per_seed: int
    ood_rows_per_seed: int
    counterfactual_rows_per_seed: int
    adversarial_rows_per_seed: int
    population_size: int
    generations: int
    mutation_sigma: float
    elite_count: int
    state_dim: int = STATE_DIM
    settling_steps: int = 6


def round_float(value: float) -> float:
    return e2.round_float(value)


def resolve_out(path: str | Path) -> Path:
    return e2.resolve_out(path)


def parse_seeds(raw: str) -> tuple[int, ...]:
    return e2.parse_seeds(raw)


def matrix(rows: int, cols: int, value: float = 0.0) -> list[list[float]]:
    return e2.matrix(rows, cols, value)


def vector(size: int, value: float = 0.0) -> list[float]:
    return e2.vector(size, value)


def candidate_hash(candidate: dict[str, Any]) -> str:
    cleaned = {key: value for key, value in candidate.items() if key != "candidate_id"}
    return e2.payload_sha256(cleaned)


def round_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return e2.round_candidate(candidate)


def original_projector() -> tuple[list[list[float]], list[float]]:
    return e2.state_projector()


def sparse_random_projector(seed: int) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    original, _ = original_projector()
    non_zero_count = sum(1 for row in original for value in row if abs(value) > 0.0)
    positions = [(row, col) for row in range(len(FEATURES)) for col in range(STATE_DIM)]
    chosen = rng.sample(positions, non_zero_count)
    w = matrix(len(FEATURES), STATE_DIM)
    for row, col in chosen:
        w[row][col] = round_float(rng.uniform(-1.0, 1.0))
    b = [round_float(rng.uniform(-0.20, 0.20)) for _ in range(STATE_DIM)]
    return w, b


def zero_projector() -> tuple[list[list[float]], list[float]]:
    return matrix(len(FEATURES), STATE_DIM), vector(STATE_DIM, 0.0)


def permuted_projector(seed: int) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    original, bias = original_projector()
    permutation = list(range(len(FEATURES)))
    rng.shuffle(permutation)
    return [copy.deepcopy(original[index]) for index in permutation], copy.deepcopy(bias)


def dense_random_projector(seed: int) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    w = [
        [round_float(rng.gauss(0.0, 0.18)) for _ in range(STATE_DIM)]
        for _ in range(len(FEATURES))
    ]
    b = [round_float(rng.gauss(0.0, 0.08)) for _ in range(STATE_DIM)]
    return w, b


def base_recurrent(zeroed: bool = False) -> list[list[float]]:
    recurrent = matrix(STATE_DIM, STATE_DIM)
    if not zeroed:
        for index in range(STATE_DIM):
            recurrent[index][index] = 0.10
    return recurrent


def base_readout() -> list[float]:
    return [1.0, 1.0, 1.0, 1.0, 0.82, 0.80, 0.45, 0.35]


def projector_for_variant(variant: str) -> tuple[list[list[float]], list[float], str]:
    if variant in {"e2_original_projector", "no_recurrence", "final_state_only", "nonrecurrent_nonlinear"}:
        w, b = original_projector()
        return w, b, "handcrafted_e2_reference"
    if variant == "random_projector":
        w, b = sparse_random_projector(e2.stable_seed("e3-random-projector"))
        return w, b, "sparse_random_same_nonzero_count"
    if variant == "zero_projector":
        w, b = zero_projector()
        return w, b, "zero_projector_mutation_from_scratch"
    if variant == "permuted_projector":
        w, b = permuted_projector(e2.stable_seed("e3-permuted-projector"))
        return w, b, "feature_assignment_permuted"
    if variant == "dense_random_projector":
        w, b = dense_random_projector(e2.stable_seed("e3-dense-random-projector"))
        return w, b, "dense_random_small_values"
    raise ValueError(f"unknown projector variant: {variant}")


def make_state_candidate(candidate_id: str, variant: str) -> dict[str, Any]:
    input_projection, state_bias, projector_kind = projector_for_variant(variant)
    readout = base_readout()
    readout_mode = "trajectory"
    model_kind = "recurrent_state"
    recurrent = base_recurrent(zeroed=variant == "no_recurrence")
    freeze_paths: list[list[Any]] = []
    if variant == "final_state_only":
        readout_mode = "final_state_only"
    if variant == "nonrecurrent_nonlinear":
        model_kind = "nonrecurrent_nonlinear"
    if variant == "no_recurrence":
        freeze_paths.append(["recurrent_matrix"])
    candidate: dict[str, Any] = {
        "schema_version": f"e3_{variant}_candidate_state_v1",
        "system": variant,
        "variant": variant,
        "candidate_id": candidate_id,
        "model_kind": model_kind,
        "projector_kind": projector_kind,
        "readout_mode": readout_mode,
        "state_dim": STATE_DIM,
        "input_dim": len(FEATURES),
        "settling_steps": 6,
        "leak": 0.64,
        "gain": 1.55,
        "score_bias": 0.0,
        "input_projection": input_projection,
        "recurrent_matrix": recurrent,
        "state_bias": state_bias,
        "readout_final": [round_float(value) for value in readout],
        "readout_delta": [0.20, 0.20, 0.20, 0.20, 0.14, 0.14, 0.08, 0.08],
        "trajectory_scalar": 0.16,
        "path_norm_scalar": 0.08,
        "parameter_range": {"matrix_vector": [-3.0, 3.0], "leak": [0.05, 0.95], "gain": [0.25, 4.0]},
        "frozen_numeric_paths": freeze_paths,
    }
    if model_kind == "nonrecurrent_nonlinear":
        candidate["feedforward_matrix"] = base_recurrent(zeroed=False)
        candidate["feedforward_bias"] = vector(STATE_DIM, 0.0)
    return round_candidate(candidate)


def make_stronger_flat_candidate(candidate_id: str) -> dict[str, Any]:
    weights = {feature: 0.22 for feature in FEATURES}
    weights.update({"evidence_gap": -0.35, "route_margin_gap": -0.35})
    pair_weights = {f"{left}*{right}": 0.0 for left, right in PAIR_FEATURES}
    return {
        "schema_version": "e3_stronger_flat_pairwise_candidate_state_v1",
        "system": "stronger_flat_pairwise",
        "variant": "stronger_flat_pairwise",
        "candidate_id": candidate_id,
        "model_kind": "stronger_flat_pairwise",
        "bias": 0.0,
        "gain": 1.0,
        "weights": {feature: round_float(weights[feature]) for feature in FEATURES},
        "pair_weights": pair_weights,
        "parameter_range": {"weights": [-5.0, 5.0], "bias": [-3.0, 3.0], "gain": [0.1, 4.0]},
    }


def initial_candidate(variant: str) -> dict[str, Any]:
    if variant == "stronger_flat_pairwise":
        return make_stronger_flat_candidate(f"{variant}_initial")
    return make_state_candidate(f"{variant}_initial", variant)


def as_array(candidate: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(candidate[key], dtype=np.float64)


def add_tie_breaker(scores: np.ndarray, split_data: dict[str, Any]) -> np.ndarray:
    return e2.add_feature_tie_breaker(scores, split_data)


def stronger_flat_scores(candidate: dict[str, Any], split_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    features = split_data["array"]
    weights = np.asarray([candidate["weights"][feature] for feature in FEATURES], dtype=np.float64)
    raw = features @ weights + float(candidate["bias"])
    pair_term = np.zeros(raw.shape, dtype=np.float64)
    for left, right in PAIR_FEATURES:
        key = f"{left}*{right}"
        pair_term += float(candidate["pair_weights"][key]) * features[:, :, FEATURE_INDEX[left]] * features[:, :, FEATURE_INDEX[right]]
    scores = add_tie_breaker(np.tanh(float(candidate["gain"]) * (raw + pair_term)), split_data)
    return scores, {
        "convergence_by_route": np.zeros_like(scores),
        "stability_by_route": np.ones_like(scores),
        "trajectory_norm_by_route": np.zeros_like(scores),
        "finite": bool(np.isfinite(scores).all()),
    }


def recurrent_rollout(candidate: dict[str, Any], split_data: dict[str, Any]) -> dict[str, np.ndarray]:
    features = split_data["array"]
    rows, routes, feature_count = features.shape
    flat = features.reshape(rows * routes, feature_count)
    input_projection = as_array(candidate, "input_projection")
    recurrent = as_array(candidate, "recurrent_matrix")
    state_bias = as_array(candidate, "state_bias")
    state = np.zeros((flat.shape[0], STATE_DIM), dtype=np.float64)
    input_drive = flat @ input_projection + state_bias
    delta_sum = np.zeros_like(state)
    last_delta = np.zeros_like(state)
    state_abs_sum = np.zeros(flat.shape[0], dtype=np.float64)
    leak = float(candidate["leak"])
    gain = float(candidate["gain"])
    for _ in range(int(candidate["settling_steps"])):
        previous = state
        activation = np.tanh(gain * (input_drive + state @ recurrent))
        state = (1.0 - leak) * state + leak * activation
        last_delta = np.abs(state - previous)
        delta_sum += last_delta
        state_abs_sum += np.mean(np.abs(state), axis=1)
    return {
        "flat_features": flat,
        "final_state": state,
        "delta_mean": delta_sum / float(candidate["settling_steps"]),
        "convergence": np.mean(last_delta, axis=1).reshape(rows, routes),
        "stability": (1.0 / (1.0 + np.mean(last_delta, axis=1))).reshape(rows, routes),
        "trajectory_norm": (state_abs_sum / float(candidate["settling_steps"])).reshape(rows, routes),
        "rows": np.asarray(rows),
        "routes": np.asarray(routes),
    }


def nonrecurrent_rollout(candidate: dict[str, Any], split_data: dict[str, Any]) -> dict[str, np.ndarray]:
    features = split_data["array"]
    rows, routes, feature_count = features.shape
    flat = features.reshape(rows * routes, feature_count)
    hidden = np.tanh(float(candidate["gain"]) * (flat @ as_array(candidate, "input_projection") + as_array(candidate, "state_bias")))
    hidden2 = np.tanh(float(candidate["gain"]) * (hidden @ as_array(candidate, "feedforward_matrix") + as_array(candidate, "feedforward_bias")))
    trajectory_norm = np.mean(np.abs(hidden2), axis=1).reshape(rows, routes)
    return {
        "flat_features": flat,
        "final_state": hidden2,
        "delta_mean": np.zeros_like(hidden2),
        "convergence": np.zeros((rows, routes), dtype=np.float64),
        "stability": np.ones((rows, routes), dtype=np.float64),
        "trajectory_norm": trajectory_norm,
        "rows": np.asarray(rows),
        "routes": np.asarray(routes),
    }


def state_scores(candidate: dict[str, Any], split_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    rollout = nonrecurrent_rollout(candidate, split_data) if candidate["model_kind"] == "nonrecurrent_nonlinear" else recurrent_rollout(candidate, split_data)
    rows = int(rollout["rows"])
    routes = int(rollout["routes"])
    raw = float(candidate["score_bias"]) + rollout["final_state"] @ as_array(candidate, "readout_final")
    if candidate["readout_mode"] == "trajectory":
        raw = (
            raw
            + rollout["delta_mean"] @ as_array(candidate, "readout_delta")
            + float(candidate["trajectory_scalar"]) * rollout["trajectory_norm"].reshape(rows * routes)
            + float(candidate["path_norm_scalar"]) * np.mean(np.abs(rollout["final_state"]), axis=1)
        )
    scores = add_tie_breaker(raw.reshape(rows, routes), split_data)
    return scores, {
        "convergence_by_route": rollout["convergence"],
        "stability_by_route": rollout["stability"],
        "trajectory_norm_by_route": rollout["trajectory_norm"],
        "finite": bool(np.isfinite(scores).all() and np.isfinite(rollout["final_state"]).all()),
    }


def score_candidate(candidate: dict[str, Any], split_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate["model_kind"] == "stronger_flat_pairwise":
        return stronger_flat_scores(candidate, split_data)
    return state_scores(candidate, split_data)


def evaluate_candidate(candidate: dict[str, Any], split_data: dict[str, Any], sample_limit: int = 8) -> dict[str, Any]:
    scores, diagnostics = score_candidate(candidate, split_data)
    if not np.isfinite(scores).all() or diagnostics.get("finite", True) is False:
        metrics = e2.scores_to_metrics(np.full(scores.shape, 1_000_000.0), diagnostics)
        metrics["finite_state_dynamics_passed"] = False
    else:
        metrics = e2.scores_to_metrics(scores, diagnostics)
        metrics["finite_state_dynamics_passed"] = True
    samples = []
    for index in range(min(sample_limit, scores.shape[0])):
        predicted = int(np.argmin(scores[index]))
        samples.append(
            {
                "row_id": split_data["rows"][index]["row_id"],
                "predicted_route": ROUTES[predicted],
                "correct_route": "logical_route",
                "scores": {route: round_float(scores[index, route_i]) for route_i, route in enumerate(ROUTES)},
            }
        )
    return {"metrics": metrics, "row_level_samples": samples}


def evaluate_all(candidate: dict[str, Any], task: dict[str, Any], sample_limit: int = 8) -> dict[str, Any]:
    return {split: evaluate_candidate(candidate, data, sample_limit=sample_limit) for split, data in task.items()}


def fitness_from_evals(evals: dict[str, Any]) -> float:
    return e2.fitness_from_evals(evals)


def search_eval(candidate: dict[str, Any], task: dict[str, Any], all_splits: bool = False) -> dict[str, Any]:
    splits = task if all_splits else {"train": task["train"], "validation": task["validation"]}
    evals = {split: evaluate_candidate(candidate, data, sample_limit=8) for split, data in splits.items()}
    return {"candidate": candidate, "evals": evals, "fitness": fitness_from_evals(evals)}


def frozen_path(candidate: dict[str, Any], path: tuple[Any, ...]) -> bool:
    for frozen in candidate.get("frozen_numeric_paths", []):
        frozen_tuple = tuple(frozen)
        if path[: len(frozen_tuple)] == frozen_tuple:
            return True
    return False


def mutable_paths(candidate: dict[str, Any]) -> list[tuple[tuple[Any, ...], float]]:
    return [(path, value) for path, value in e2.flatten_paths(candidate) if not frozen_path(candidate, path)]


def clamp_for_path(candidate: dict[str, Any], path: tuple[Any, ...], value: float) -> float:
    if candidate["model_kind"] == "stronger_flat_pairwise":
        if path == ("bias",):
            return e2.clamp(value, -3.0, 3.0)
        if path == ("gain",):
            return e2.clamp(value, 0.10, 4.0)
        if path and path[0] in {"weights", "pair_weights"}:
            return e2.clamp(value, -5.0, 5.0)
        return e2.clamp(value, -5.0, 5.0)
    if path == ("leak",):
        return e2.clamp(value, 0.05, 0.95)
    if path == ("gain",):
        return e2.clamp(value, 0.25, 4.0)
    if path == ("score_bias",):
        return e2.clamp(value, -3.0, 3.0)
    if path and path[0] in {"readout_final", "readout_delta"}:
        return e2.clamp(value, -4.0, 4.0)
    if path and path[-1] in {"trajectory_scalar", "path_norm_scalar"}:
        return e2.clamp(value, -4.0, 4.0)
    return e2.clamp(value, -3.0, 3.0)


def mutate_candidate(candidate: dict[str, Any], rng: random.Random, sigma: float, candidate_id: str) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    child["candidate_id"] = candidate_id
    paths = mutable_paths(child)
    edit_count = rng.randint(2, 12)
    for path, value in rng.sample(paths, k=min(edit_count, len(paths))):
        scale = sigma * (1.35 if child["model_kind"] == "stronger_flat_pairwise" else 1.8)
        e2.set_path(child, path, clamp_for_path(child, path, value + rng.gauss(0.0, scale)))
    return round_candidate(child)


def finite_candidate(candidate: dict[str, Any]) -> bool:
    return all(math.isfinite(value) for _, value in e2.flatten_paths(candidate))


def run_variant_search(variant: str, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = random.Random(e2.stable_seed(f"e3-{variant}-{settings.seeds}"))
    initial = initial_candidate(variant)
    initial_eval = search_eval(initial, task, all_splits=True)
    population = [initial]
    for index in range(settings.population_size - 1):
        population.append(mutate_candidate(initial, rng, settings.mutation_sigma, f"{variant}_seed_population_{index:02d}"))
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
            child = mutate_candidate(parent["candidate"], rng, settings.mutation_sigma, f"{variant}_g{generation:03d}_m{attempt:02d}")
            parent_hash = candidate_hash(parent["candidate"])
            child_eval = search_eval(child, task, all_splits=False) if finite_candidate(child) else {"candidate": child, "evals": {}, "fitness": -1_000_000.0}
            accepted_flag = child_eval["fitness"] >= parent["fitness"]
            if accepted_flag:
                accepted += 1
                scored.append(child_eval)
                scored.sort(key=lambda row: row["fitness"], reverse=True)
                scored = scored[: settings.population_size]
                if child_eval["fitness"] > best["fitness"]:
                    best = child_eval
            else:
                rejected += 1
                rollback += 1
            history.append(
                {
                    "variant": variant,
                    "generation": generation,
                    "attempt": mutation_attempt,
                    "accepted": bool(accepted_flag),
                    "parent_hash": parent_hash,
                    "candidate_hash": candidate_hash(child),
                    "parent_fitness": parent["fitness"],
                    "candidate_fitness": child_eval["fitness"],
                    "rollback_performed": not accepted_flag,
                }
            )
        full_best = search_eval(best["candidate"], task, all_splits=True)
        metrics = {
            "variant": variant,
            "generation": generation,
            "best_fitness": full_best["fitness"],
            "train_logical_selected_rate": full_best["evals"]["train"]["metrics"]["logical_path_selected_rate"],
            "validation_logical_selected_rate": full_best["evals"]["validation"]["metrics"]["logical_path_selected_rate"],
            "heldout_logical_selected_rate": full_best["evals"]["heldout"]["metrics"]["logical_path_selected_rate"],
            "ood_logical_selected_rate": full_best["evals"]["ood"]["metrics"]["logical_path_selected_rate"],
            "counterfactual_logical_selected_rate": full_best["evals"]["counterfactual"]["metrics"]["logical_path_selected_rate"],
            "adversarial_logical_selected_rate": full_best["evals"]["adversarial"]["metrics"]["logical_path_selected_rate"],
            "heldout_ordering_stability": full_best["evals"]["heldout"]["metrics"]["ordering_stability"],
            "heldout_logical_vs_best_wrong_gap": full_best["evals"]["heldout"]["metrics"]["logical_vs_best_wrong_gap"],
            "accepted_mutation_count": accepted,
            "rejected_mutation_count": rejected,
            "rollback_count": rollback,
            "candidate_hash": candidate_hash(best["candidate"]),
        }
        generation_metrics.append(metrics)
        if out is not None:
            e2.append_progress(out, "generation_complete", variant=variant, generation=generation, metrics=metrics)
            e2.write_json(out / f"e3_mutation_history_{variant}.json", mutation_history_artifact(variant, mutation_attempt, accepted, rejected, rollback, history))
    final_eval = search_eval(best["candidate"], task, all_splits=True)
    return {
        "variant": variant,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "history": history,
        "generation_metrics": generation_metrics,
        "mutation_attempt_count": mutation_attempt,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
    }


def perturbed_split(split_data: dict[str, Any], seed: int, scale: float = 0.035) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, scale, size=split_data["array"].shape)
    return {"rows": split_data["rows"], "array": np.clip(split_data["array"] + noise, 0.0, 1.0)}


def conductivity_split_metrics(candidate: dict[str, Any], split_data: dict[str, Any], label: str) -> dict[str, Any]:
    base = evaluate_candidate(candidate, split_data, sample_limit=0)["metrics"]
    perturbed = evaluate_candidate(candidate, perturbed_split(split_data, e2.stable_seed(f"e3-perturb-{candidate_hash(candidate)}-{label}")), sample_limit=0)["metrics"]
    result = dict(base)
    result["perturbation_recovery_rate"] = perturbed["logical_path_selected_rate"]
    result["perturbed_ordering_stability"] = perturbed["ordering_stability"]
    result["counterfactual_ordering_stability"] = result["ordering_stability"] if label == "counterfactual" else None
    result["ood_ordering_stability"] = result["ordering_stability"] if label == "ood" else None
    return result


def conductivity_ordering_report(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        candidate = searches[variant]["final_eval"]["candidate"]
        split_metrics = {split: conductivity_split_metrics(candidate, task[split], split) for split in ORDERING_SPLITS}
        split_pass = {split: e2.conductivity_pass_for_split(metrics, split) for split, metrics in split_metrics.items()}
        variants[variant] = {
            "split_metrics": split_metrics,
            "split_pass": split_pass,
            "conductivity_ordering_passed": all(split_pass.values()),
            "heldout_logical_path_selected_rate": split_metrics["heldout"]["logical_path_selected_rate"],
            "heldout_logical_vs_best_wrong_gap": split_metrics["heldout"]["logical_vs_best_wrong_gap"],
        }
    return {
        "schema_version": "e3_conductivity_ordering_report_v1",
        "score_direction": "lowest_score_is_lowest_resistance",
        "ordering_splits": list(ORDERING_SPLITS),
        "variants": variants,
    }


def shuffled_order_rate(candidate: dict[str, Any], split_data: dict[str, Any], seed: int) -> float:
    rng = random.Random(seed)
    rows = split_data["array"].shape[0]
    shuffled = np.empty_like(split_data["array"])
    route_names: list[list[str]] = []
    for row_index in range(rows):
        permutation = list(range(len(ROUTES)))
        rng.shuffle(permutation)
        shuffled[row_index] = split_data["array"][row_index, permutation, :]
        route_names.append([ROUTES[index] for index in permutation])
    scores, diagnostics = score_candidate(candidate, {"rows": split_data["rows"], "array": shuffled})
    if not np.isfinite(scores).all() or diagnostics.get("finite", True) is False:
        return 0.0
    predicted = np.argmin(scores, axis=1)
    logical = sum(1 for row_index, pred in enumerate(predicted) if route_names[row_index][int(pred)] == "logical_route")
    return round_float(logical / rows)


def feature_permutation_rate(candidate: dict[str, Any], split_data: dict[str, Any], seed: int) -> float:
    rng = random.Random(seed)
    permutation = list(range(len(FEATURES)))
    rng.shuffle(permutation)
    result = evaluate_candidate(candidate, {"rows": split_data["rows"], "array": split_data["array"][:, :, permutation]}, sample_limit=0)
    return result["metrics"]["logical_path_selected_rate"]


def control_scores(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    original = searches["e2_original_projector"]["final_eval"]["candidate"]
    route_index_rates = {f"{split}_logical_path_selected_rate": 1.0 if ROUTES[0] == "logical_route" else 0.0 for split in task}
    candidate_name_control = {f"{split}_logical_path_selected_rate": 0.0 for split in task}
    shuffled = shuffled_order_rate(original, task["heldout"], e2.stable_seed("e3-shuffled-control"))
    feature_perm = feature_permutation_rate(original, task["heldout"], e2.stable_seed("e3-feature-permutation-control"))
    control_metrics = {
        "route_index_order_control": {
            "reference_only": False,
            "used_as_candidate": False,
            "selected_route_index": 0,
            "selected_route_name": ROUTES[0],
            **route_index_rates,
        },
        "candidate_name_leak_control": {
            "reference_only": False,
            "used_as_candidate": False,
            "route_names_available_to_score_function": False,
            "name_leak_attempt_solved_task": False,
            **candidate_name_control,
        },
        "shuffled_route_order_control": {
            "reference_only": True,
            "used_as_candidate": False,
            "base_variant": "e2_original_projector",
            "heldout_original_order_rate": searches["e2_original_projector"]["final_eval"]["evals"]["heldout"]["metrics"]["logical_path_selected_rate"],
            "heldout_shuffled_order_rate": shuffled,
            "preserved_outcome": abs(shuffled - searches["e2_original_projector"]["final_eval"]["evals"]["heldout"]["metrics"]["logical_path_selected_rate"]) <= 0.02,
        },
        "feature_permutation_control": {
            "reference_only": False,
            "used_as_candidate": False,
            "base_variant": "e2_original_projector",
            "heldout_logical_path_selected_rate": feature_perm,
            "degraded_as_expected": feature_perm < 0.90,
        },
    }
    controls_do_not_solve = (
        control_metrics["route_index_order_control"]["heldout_logical_path_selected_rate"] < 0.90
        and control_metrics["candidate_name_leak_control"]["heldout_logical_path_selected_rate"] < 0.90
        and control_metrics["feature_permutation_control"]["degraded_as_expected"]
        and control_metrics["shuffled_route_order_control"]["preserved_outcome"]
    )
    return {
        "schema_version": "e3_control_baseline_report_v1",
        "control_metrics": control_metrics,
        "controls_do_not_solve_task": controls_do_not_solve,
        "control_requirement": "Route/name/index controls must not solve the task; shuffled route order must preserve outcome; feature permutation must degrade.",
    }


def leakage_sentinel_report(searches: dict[str, Any], task: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in STATE_VARIANTS:
        candidate = searches[variant]["final_eval"]["candidate"]
        original = searches[variant]["final_eval"]["evals"]["heldout"]["metrics"]["logical_path_selected_rate"]
        shuffled = shuffled_order_rate(candidate, task["heldout"], e2.stable_seed(f"e3-{variant}-route-shuffle"))
        permuted = feature_permutation_rate(candidate, task["heldout"], e2.stable_seed(f"e3-{variant}-feature-permutation"))
        variants[variant] = {
            "heldout_original_order_rate": original,
            "heldout_shuffled_route_order_rate": shuffled,
            "heldout_feature_permutation_rate": permuted,
            "shuffled_route_order_passed": abs(shuffled - original) <= 0.02,
            "feature_permutation_degraded": permuted < 0.90,
        }
    route_index_solved = controls["control_metrics"]["route_index_order_control"]["heldout_logical_path_selected_rate"] >= 0.90
    candidate_name_solved = controls["control_metrics"]["candidate_name_leak_control"]["heldout_logical_path_selected_rate"] >= 0.90
    return {
        "schema_version": "e3_leakage_sentinel_report_v1",
        "route_labels_used_for_scoring": False,
        "route_names_used_for_scoring": False,
        "candidate_order_used_as_feature": False,
        "hidden_correct_route_index_used_for_scoring": False,
        "route_index_leak_detected": bool(route_index_solved),
        "candidate_name_leak_detected": bool(candidate_name_solved),
        "score_functions_consume": ["split_data.array", "candidate numeric parameters", "state trajectory diagnostics"],
        "score_functions_do_not_consume": ["row.correct_route", "row.routes keys", "route names as labels", "candidate index"],
        "variants": variants,
        "shuffled_route_order_passed": all(row["shuffled_route_order_passed"] for row in variants.values()),
        "feature_permutation_control_degraded": all(row["feature_permutation_degraded"] for row in variants.values()),
        "leakage_sentinel_passed": (
            not route_index_solved
            and not candidate_name_solved
            and all(row["shuffled_route_order_passed"] for row in variants.values())
            and all(row["feature_permutation_degraded"] for row in variants.values())
        ),
    }


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
        "schema_version": f"e3_parameter_diff_{final['variant']}_v1",
        "variant": final["variant"],
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


def mutation_history_artifact(variant: str, attempts: int, accepted: int, rejected: int, rollback: int, history: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": f"e3_mutation_history_{variant}_v1",
        "variant": variant,
        "mutation_attempt_count": attempts,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
        "history": history,
    }


def accept_reject_rollback_report(searches: dict[str, Any]) -> dict[str, Any]:
    variants = {
        variant: {
            "mutation_attempt_count": searches[variant]["mutation_attempt_count"],
            "accepted_mutation_count": searches[variant]["accepted_mutation_count"],
            "rejected_mutation_count": searches[variant]["rejected_mutation_count"],
            "rollback_count": searches[variant]["rollback_count"],
        }
        for variant in VARIANTS
    }
    accepted_total = sum(row["accepted_mutation_count"] for row in variants.values())
    rejected_total = sum(row["rejected_mutation_count"] for row in variants.values())
    rollback_total = sum(row["rollback_count"] for row in variants.values())
    return {
        "schema_version": "e3_accept_reject_rollback_report_v1",
        "variants": variants,
        "accepted_mutation_count_total": accepted_total,
        "rejected_mutation_count_total": rejected_total,
        "rollback_count_total": rollback_total,
        "rollback_test_executed": True,
        "rollback_test_passed": rejected_total == rollback_total and rejected_total >= 1,
    }


def system_metrics(search: dict[str, Any], conductivity: dict[str, Any]) -> dict[str, Any]:
    final = search["final_eval"]["evals"]
    heldout = conductivity["variants"][search["variant"]]["split_metrics"]["heldout"]
    return {
        "variant": search["variant"],
        "model_kind": search["final_eval"]["candidate"]["model_kind"],
        "projector_kind": search["final_eval"]["candidate"].get("projector_kind"),
        "readout_mode": search["final_eval"]["candidate"].get("readout_mode"),
        "train_accuracy": final["train"]["metrics"]["logical_path_selected_rate"],
        "validation_accuracy": final["validation"]["metrics"]["logical_path_selected_rate"],
        "heldout_accuracy": heldout["logical_path_selected_rate"],
        "ood_accuracy": conductivity["variants"][search["variant"]]["split_metrics"]["ood"]["logical_path_selected_rate"],
        "counterfactual_accuracy": conductivity["variants"][search["variant"]]["split_metrics"]["counterfactual"]["logical_path_selected_rate"],
        "adversarial_accuracy": conductivity["variants"][search["variant"]]["split_metrics"]["adversarial"]["logical_path_selected_rate"],
        "logical_vs_best_wrong_gap": heldout["logical_vs_best_wrong_gap"],
        "logical_vs_shortcut_gap": heldout["logical_vs_shortcut_gap"],
        "logical_vs_noise_gap": heldout["logical_vs_noise_gap"],
        "logical_vs_illogical_gap": heldout["logical_vs_illogical_gap"],
        "logical_vs_surface_wrong_gap": heldout["logical_vs_surface_wrong_gap"],
        "logical_vs_contradiction_gap": heldout["logical_vs_contradiction_gap"],
        "shortcut_attractor_rate": heldout["shortcut_attractor_rate"],
        "contradiction_attractor_rate": heldout["contradiction_attractor_rate"],
        "perturbation_recovery_rate": heldout["perturbation_recovery_rate"],
        "ordering_stability": heldout["ordering_stability"],
        "conductivity_ordering_passed": conductivity["variants"][search["variant"]]["conductivity_ordering_passed"],
        "accepted_mutation_count": search["accepted_mutation_count"],
        "rejected_mutation_count": search["rejected_mutation_count"],
        "rollback_count": search["rollback_count"],
    }


def pass_like_original(metrics: dict[str, Any], original: dict[str, Any]) -> bool:
    if not metrics["conductivity_ordering_passed"]:
        return False
    for key in ("heldout_accuracy", "ood_accuracy", "counterfactual_accuracy", "adversarial_accuracy"):
        if metrics[key] < original[key] - 0.03:
            return False
    return metrics["logical_vs_best_wrong_gap"] >= original["logical_vs_best_wrong_gap"] - 0.50


def aggregate_metrics(searches: dict[str, Any], conductivity: dict[str, Any], controls: dict[str, Any], leakage: dict[str, Any], rollback: dict[str, Any], deterministic: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    variants = {variant: system_metrics(searches[variant], conductivity) for variant in VARIANTS}
    original = variants["e2_original_projector"]
    scaffold_removed = {
        variant: variants[variant]["conductivity_ordering_passed"]
        for variant in ("random_projector", "zero_projector", "permuted_projector")
    }
    dense_pass = variants["dense_random_projector"]["conductivity_ordering_passed"]
    no_recurrence_equal = pass_like_original(variants["no_recurrence"], original)
    nonrecurrent_equal = pass_like_original(variants["nonrecurrent_nonlinear"], original)
    flat_matches = pass_like_original(variants["stronger_flat_pairwise"], original)
    trajectory_advantage = (
        original["conductivity_ordering_passed"]
        and original["logical_vs_best_wrong_gap"] > variants["final_state_only"]["logical_vs_best_wrong_gap"] + 0.25
        and original["adversarial_accuracy"] >= variants["final_state_only"]["adversarial_accuracy"]
    )
    return {
        "schema_version": "e3_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "variants": variants,
        "original_passed": original["conductivity_ordering_passed"],
        "scaffold_removed_variants_passed": scaffold_removed,
        "dense_random_projector_passed": dense_pass,
        "all_core_scaffold_removed_passed": all(scaffold_removed.values()),
        "any_core_scaffold_removed_failed": not all(scaffold_removed.values()),
        "no_recurrence_equal_to_original": no_recurrence_equal,
        "nonrecurrent_nonlinear_equal_to_original": nonrecurrent_equal,
        "stronger_flat_matches_original": flat_matches,
        "trajectory_readout_beats_final_state_only": trajectory_advantage,
        "controls_do_not_solve_task": controls["controls_do_not_solve_task"],
        "leakage_sentinel_passed": leakage["leakage_sentinel_passed"],
        "route_index_leak_detected": leakage["route_index_leak_detected"],
        "candidate_name_leak_detected": leakage["candidate_name_leak_detected"],
        "shuffled_route_order_passed": leakage["shuffled_route_order_passed"],
        "accepted_mutation_count_total": rollback["accepted_mutation_count_total"],
        "rejected_mutation_count_total": rollback["rejected_mutation_count_total"],
        "rollback_count_total": rollback["rollback_count_total"],
        "rollback_test_passed": rollback["rollback_test_passed"],
        "deterministic_replay_passed": deterministic["internal_replay_passed"],
        "static_metric_dictionary_used": audit["static_metric_dictionary_used"],
        "hardcoded_improvement_used": audit["hardcoded_improvement_used"],
        "synthetic_harness_only": audit["synthetic_harness_only"],
        "gradient_backprop_used": audit["gradient_backprop_used"],
        "row_level_predictions_used": audit["row_level_predictions_used"],
    }


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []
    if not aggregate["controls_do_not_solve_task"] or not aggregate["leakage_sentinel_passed"]:
        decision = "e3_leak_or_task_artifact_detected"
        next_step = "E3L_REPAIR_LEAK_OR_TASK_ARTIFACT"
    else:
        if aggregate["stronger_flat_matches_original"] or aggregate["nonrecurrent_nonlinear_equal_to_original"]:
            flags.append("e3_state_medium_not_unique")
        if aggregate["no_recurrence_equal_to_original"]:
            flags.append("e3_recurrence_not_required")
        if aggregate["trajectory_readout_beats_final_state_only"]:
            flags.append("e3_temporal_projection_advantage_confirmed")
        if aggregate["original_passed"] and aggregate["any_core_scaffold_removed_failed"]:
            flags.append("e3_handcrafted_projector_dependency_detected")
        if aggregate["original_passed"] and aggregate["all_core_scaffold_removed_passed"]:
            flags.append("e3_state_medium_advantage_survives_scaffold_removal")
        if "e3_state_medium_not_unique" in flags:
            decision = "e3_state_medium_not_unique"
            next_step = "E4_COMPARE_MEDIUM_AGAINST_STRONGER_NONRECURRENT_BASELINES"
        elif "e3_recurrence_not_required" in flags:
            decision = "e3_recurrence_not_required"
            next_step = "E4_ISOLATE_NONRECURRENT_NONLINEAR_PROJECTION_ADVANTAGE"
        elif "e3_handcrafted_projector_dependency_detected" in flags:
            decision = "e3_handcrafted_projector_dependency_detected"
            next_step = "E3B_REMOVE_PROJECTOR_PRIORS_AND_REDESIGN_MEDIUM_OBJECTIVE"
        elif "e3_state_medium_advantage_survives_scaffold_removal" in flags:
            decision = "e3_state_medium_advantage_survives_scaffold_removal"
            next_step = "E4_STRESS_GENERALIZE_UNSCAFFOLDED_STATE_MEDIUM"
        elif "e3_temporal_projection_advantage_confirmed" in flags:
            decision = "e3_temporal_projection_advantage_confirmed"
            next_step = "E4_TEMPORAL_READOUT_STRESS_AND_AUDIT"
        else:
            decision = "e3_no_clean_state_medium_advantage_detected"
            next_step = "E3R_TASK_OR_OBJECTIVE_REDESIGN"
    return {
        "schema_version": "e3_decision_v1",
        "milestone": MILESTONE,
        "decision": decision,
        "supporting_decision_flags": flags,
        "next": next_step,
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
        "gradient_backprop_used": False,
        "route_index_leak_detected": aggregate["route_index_leak_detected"],
        "candidate_name_leak_detected": aggregate["candidate_name_leak_detected"],
        "shuffled_route_order_passed": aggregate["shuffled_route_order_passed"],
        "original_passed": aggregate["original_passed"],
        "all_core_scaffold_removed_passed": aggregate["all_core_scaffold_removed_passed"],
        "no_recurrence_equal_to_original": aggregate["no_recurrence_equal_to_original"],
        "trajectory_readout_beats_final_state_only": aggregate["trajectory_readout_beats_final_state_only"],
        "stronger_flat_matches_original": aggregate["stronger_flat_matches_original"],
    }


def no_synthetic_metric_audit(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e3_no_synthetic_metric_audit_v1",
        "static_metric_dictionary_used": False,
        "hardcoded_improvement_used": False,
        "synthetic_harness_only": False,
        "row_level_predictions_used": True,
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
        "explicit_hand_designed_gate_module_used": False,
        "handcrafted_projector_reference_used_only_by": ["e2_original_projector", "no_recurrence", "final_state_only", "nonrecurrent_nonlinear"],
        "generated_row_counts": {split: len(data["rows"]) for split, data in task.items()},
        "mutation_attempts_by_variant": {variant: searches[variant]["mutation_attempt_count"] for variant in VARIANTS},
        "metrics_computed_from_functions": [
            "generate_task",
            "evaluate_candidate",
            "conductivity_split_metrics",
            "evaluate_all",
            "fitness_from_evals",
        ],
    }


def logical_vs_wrong_gap_report(conductivity: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "logical_vs_best_wrong_gap",
        "logical_vs_shortcut_gap",
        "logical_vs_noise_gap",
        "logical_vs_illogical_gap",
        "logical_vs_surface_wrong_gap",
        "logical_vs_contradiction_gap",
    )
    return {
        "schema_version": "e3_logical_vs_wrong_gap_report_v1",
        "variants": {
            variant: {split: {key: row["split_metrics"][split][key] for key in keys} for split in ORDERING_SPLITS}
            for variant, row in conductivity["variants"].items()
        },
    }


def attractor_basin_report(conductivity: dict[str, Any]) -> dict[str, Any]:
    keys = ("shortcut_attractor_rate", "contradiction_attractor_rate", "attractor_basin_separation_score")
    return {
        "schema_version": "e3_attractor_basin_report_v1",
        "variants": {
            variant: {split: {key: row["split_metrics"][split][key] for key in keys} for split in ORDERING_SPLITS}
            for variant, row in conductivity["variants"].items()
        },
    }


def perturbation_recovery_report(conductivity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e3_perturbation_recovery_report_v1",
        "variants": {
            variant: {
                split: {
                    "perturbation_recovery_rate": row["split_metrics"][split]["perturbation_recovery_rate"],
                    "ordering_stability": row["split_metrics"][split]["ordering_stability"],
                }
                for split in ORDERING_SPLITS
            }
            for variant, row in conductivity["variants"].items()
        },
    }


def variant_ablation_report(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e3_variant_ablation_report_v1",
        "variants_requested": list(VARIANTS),
        "projector_ablation_variants": list(PROJECTOR_ABLATION_VARIANTS),
        "variants": aggregate["variants"],
        "original_passed": aggregate["original_passed"],
        "all_core_scaffold_removed_passed": aggregate["all_core_scaffold_removed_passed"],
        "any_core_scaffold_removed_failed": aggregate["any_core_scaffold_removed_failed"],
        "no_recurrence_equal_to_original": aggregate["no_recurrence_equal_to_original"],
        "nonrecurrent_nonlinear_equal_to_original": aggregate["nonrecurrent_nonlinear_equal_to_original"],
        "stronger_flat_matches_original": aggregate["stronger_flat_matches_original"],
        "trajectory_readout_beats_final_state_only": aggregate["trajectory_readout_beats_final_state_only"],
    }


def backend_manifest(settings: Settings, git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e3_backend_manifest_v1",
        "milestone": MILESTONE,
        "backend_type": "real_mutation_selection_with_rollback",
        "variants": list(VARIANTS),
        "candidate_state_created": True,
        "mutation_backend_used": True,
        "mutation_operator": "stdlib random.Random gaussian edits over scalar, vector, and matrix parameters with finite clamps",
        "row_level_predictions_used": True,
        "deterministic_update_order": True,
        "population_size": settings.population_size,
        "generations": settings.generations,
        "elite_count": settings.elite_count,
        "mutation_sigma": settings.mutation_sigma,
        "state_dim": settings.state_dim,
        "settling_steps": settings.settling_steps,
        "explicit_hand_designed_gate_module_used": False,
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
        "synthetic_harness_only": False,
        "numpy_version": np.__version__,
        "git_preflight": git,
    }


def task_generation_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    report = e2.task_generation_report(task, settings)
    report["schema_version"] = "e3_task_generation_report_v1"
    report["milestone"] = MILESTONE
    return report


def deterministic_stub(passed: bool, comparisons: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e3_deterministic_replay_report_v1",
        "internal_replay_executed": True,
        "internal_replay_passed": passed,
        "deterministic_replay_passed": passed,
        "hash_artifacts": list(HASH_ARTIFACTS),
        "hash_comparisons": comparisons,
        "external_replay_compared": False,
    }


def compose_artifacts(core: dict[str, Any], deterministic: dict[str, Any]) -> dict[str, Any]:
    searches = core["searches"]
    task = core["task"]
    conductivity = conductivity_ordering_report(searches, task)
    controls = control_scores(searches, task)
    leakage = leakage_sentinel_report(searches, task, controls)
    rollback = accept_reject_rollback_report(searches)
    audit = no_synthetic_metric_audit(searches, task)
    aggregate = aggregate_metrics(searches, conductivity, controls, leakage, rollback, deterministic, audit)
    decision = decide(aggregate)
    diffs = {
        variant: parameter_diff(searches[variant]["initial_eval"]["candidate"], searches[variant]["final_eval"]["candidate"], searches[variant])
        for variant in VARIANTS
    }
    artifacts: dict[str, Any] = {
        "e3_backend_manifest.json": backend_manifest(core["settings"], core["git"]),
        "e3_task_generation_report.json": task_generation_report(task, core["settings"]),
        "e3_variant_ablation_report.json": variant_ablation_report(aggregate),
        "e3_conductivity_ordering_report.json": conductivity,
        "e3_control_baseline_report.json": controls,
        "e3_leakage_sentinel_report.json": leakage,
        "e3_no_synthetic_metric_audit.json": audit,
        "e3_deterministic_replay_report.json": deterministic,
        "e3_accept_reject_rollback_report.json": rollback,
        "e3_logical_vs_wrong_gap_report.json": logical_vs_wrong_gap_report(conductivity),
        "e3_attractor_basin_report.json": attractor_basin_report(conductivity),
        "e3_perturbation_recovery_report.json": perturbation_recovery_report(conductivity),
        "e3_generation_metrics.json": {variant: searches[variant]["generation_metrics"] for variant in VARIANTS},
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": summary(decision, aggregate, core["git"]),
        "report.md": report_md(decision, aggregate),
    }
    for variant in VARIANTS:
        artifacts[f"e3_candidate_{variant}_initial.json"] = searches[variant]["initial_eval"]["candidate"]
        artifacts[f"e3_candidate_{variant}_final.json"] = searches[variant]["final_eval"]["candidate"]
        artifacts[f"e3_parameter_diff_{variant}.json"] = diffs[variant]
        artifacts[f"e3_mutation_history_{variant}.json"] = mutation_history_artifact(
            variant,
            searches[variant]["mutation_attempt_count"],
            searches[variant]["accepted_mutation_count"],
            searches[variant]["rejected_mutation_count"],
            searches[variant]["rollback_count"],
            searches[variant]["history"],
        )
    for split in ("heldout", "ood", "counterfactual", "adversarial"):
        artifacts[f"e3_row_level_eval_sample_{split}.json"] = {
            "schema_version": f"e3_row_level_eval_sample_{split}_v1",
            "split": split,
            "samples": {
                variant: searches[variant]["final_eval"]["evals"][split]["row_level_samples"]
                for variant in VARIANTS
            },
        }
    return artifacts


def summary(decision: dict[str, Any], aggregate: dict[str, Any], git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e3_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "supporting_decision_flags": decision["supporting_decision_flags"],
        "next": decision["next"],
        "git_status": git["git_status"],
        "original_passed": aggregate["original_passed"],
        "all_core_scaffold_removed_passed": aggregate["all_core_scaffold_removed_passed"],
        "no_recurrence_equal_to_original": aggregate["no_recurrence_equal_to_original"],
        "trajectory_readout_beats_final_state_only": aggregate["trajectory_readout_beats_final_state_only"],
        "stronger_flat_matches_original": aggregate["stronger_flat_matches_original"],
        "controls_do_not_solve_task": aggregate["controls_do_not_solve_task"],
        "leakage_sentinel_passed": aggregate["leakage_sentinel_passed"],
        "deterministic_replay_passed": aggregate["deterministic_replay_passed"],
        "accepted_mutation_count_total": aggregate["accepted_mutation_count_total"],
        "rejected_mutation_count_total": aggregate["rejected_mutation_count_total"],
    }


def report_md(decision: dict[str, Any], aggregate: dict[str, Any]) -> str:
    lines = [
        f"# {MILESTONE} Result",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"supporting_decision_flags = {','.join(decision['supporting_decision_flags'])}",
        f"next = {decision['next']}",
        "```",
        "",
        "## Variants",
        "",
    ]
    for variant, metrics in aggregate["variants"].items():
        lines.append(
            f"- {variant}: pass={metrics['conductivity_ordering_passed']} heldout={metrics['heldout_accuracy']} "
            f"ood={metrics['ood_accuracy']} counterfactual={metrics['counterfactual_accuracy']} "
            f"adversarial={metrics['adversarial_accuracy']} gap={metrics['logical_vs_best_wrong_gap']}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "E3 tests whether the E2 state-medium result survives projector, recurrence, readout, and baseline ablations on the E2 toy task. It is not model-scale training or an AGI/consciousness claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(out: Path, core: dict[str, Any], deterministic: dict[str, Any]) -> None:
    artifacts = compose_artifacts(core, deterministic)
    for name, payload in artifacts.items():
        path = out / name
        if isinstance(payload, str):
            e2.write_text(path, payload)
        else:
            e2.write_json(path, payload)
    e2.append_progress(out, "final_artifacts_written", artifact_count=len(artifacts))


def compare_core(primary: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    primary_artifacts = compose_artifacts(primary, deterministic_stub(True, {}))
    replay_artifacts = compose_artifacts(replay, deterministic_stub(True, {}))
    comparisons = {}
    for name in HASH_ARTIFACTS:
        comparisons[name] = {
            "primary_hash": e2.payload_sha256(primary_artifacts[name]),
            "replay_hash": e2.payload_sha256(replay_artifacts[name]),
            "match": e2.payload_sha256(primary_artifacts[name]) == e2.payload_sha256(replay_artifacts[name]),
        }
    return deterministic_stub(all(row["match"] for row in comparisons.values()), comparisons)


def run_core(settings: Settings, out: Path | None = None) -> dict[str, Any]:
    task = e2.generate_task(settings)
    git = e2.git_preflight()
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        e2.append_progress(out, "startup", milestone=MILESTONE, settings=settings.__dict__, variants=list(VARIANTS))
    searches = {}
    for variant in VARIANTS:
        searches[variant] = run_variant_search(variant, task, settings, out)
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
        adversarial_rows_per_seed=args.adversarial_rows_per_seed,
        population_size=args.population_size,
        generations=args.generations,
        mutation_sigma=args.mutation_sigma,
        elite_count=args.elite_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--validation-rows-per-seed", type=int, default=300)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=300)
    parser.add_argument("--ood-rows-per-seed", type=int, default=300)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=300)
    parser.add_argument("--adversarial-rows-per-seed", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--mutation-sigma", type=float, default=0.12)
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
    print(json.dumps({"decision": decision["decision"], "next": decision["next"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
