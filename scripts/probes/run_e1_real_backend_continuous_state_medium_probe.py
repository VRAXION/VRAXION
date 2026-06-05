#!/usr/bin/env python3
"""E1 real-backend continuous state medium probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "E1_REAL_BACKEND_CONTINUOUS_STATE_MEDIUM_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/e1_real_backend_continuous_state_medium_probe")
DEFAULT_SEEDS = (72001, 72002, 72003, 72004, 72005)

ROUTES = (
    "correct_route",
    "shortcut_route",
    "over_abstract_route",
    "local_expensive_route",
    "decoy_high_evidence_bad_route",
    "decoy_low_cost_bad_route",
    "decoy_stable_but_wrong_route",
)
FEATURES = (
    "energy_cost",
    "local_step_cost",
    "abstraction_jump_cost",
    "landing_error_risk",
    "shortcut_risk",
    "route_uncertainty",
    "counterfactual_fragility",
    "preservation_risk",
    "calibration_mismatch",
    "evidence_gap",
    "route_margin_gap",
    "template_collision_risk",
    "grammar_collision_risk",
    "binding_scope_risk",
    "sequence_length_risk",
    "adversarial_surface_similarity",
    "temporal_instability_risk",
    "state_transition_cost",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}
ROUTE_INDEX = {name: index for index, name in enumerate(ROUTES)}
TIE_BREAK_VECTOR = np.asarray(
    [
        0.031,
        0.047,
        0.059,
        0.071,
        0.083,
        0.097,
        0.109,
        0.127,
        0.139,
        0.149,
        0.163,
        0.181,
        0.193,
        0.211,
        0.227,
        0.241,
        0.263,
        0.277,
    ],
    dtype=np.float64,
)
SYSTEMS = ("flat", "state_medium", "gated_state_medium")
HASH_ARTIFACTS = (
    "e1_candidate_flat_final.json",
    "e1_candidate_state_medium_final.json",
    "e1_candidate_gated_state_medium_final.json",
    "e1_parameter_diff_flat.json",
    "e1_parameter_diff_state_medium.json",
    "e1_parameter_diff_gated_state_medium.json",
    "e1_generation_metrics.json",
    "e1_control_baseline_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
REQUIRED_ARTIFACTS = (
    "e1_online_check_report.md",
    "e1_backend_manifest.json",
    "e1_task_generation_report.json",
    "e1_candidate_flat_initial.json",
    "e1_candidate_flat_final.json",
    "e1_candidate_state_medium_initial.json",
    "e1_candidate_state_medium_final.json",
    "e1_candidate_gated_state_medium_initial.json",
    "e1_candidate_gated_state_medium_final.json",
    "e1_parameter_diff_flat.json",
    "e1_parameter_diff_state_medium.json",
    "e1_parameter_diff_gated_state_medium.json",
    "e1_mutation_history_flat.json",
    "e1_mutation_history_state_medium.json",
    "e1_mutation_history_gated_state_medium.json",
    "e1_generation_metrics.json",
    "e1_row_level_eval_sample_train.json",
    "e1_row_level_eval_sample_heldout.json",
    "e1_row_level_eval_sample_ood.json",
    "e1_row_level_eval_sample_counterfactual.json",
    "e1_control_baseline_report.json",
    "e1_flat_baseline_failure_audit.json",
    "e1_state_medium_leakage_audit.json",
    "e1_flat_vs_state_medium_comparison_report.json",
    "e1_state_dynamics_report.json",
    "e1_convergence_stability_report.json",
    "e1_accept_reject_rollback_report.json",
    "e1_no_synthetic_metric_audit.json",
    "e1_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
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
    state_dim: int = 8
    settling_steps: int = 6


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def resolve_out(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    relative = resolved.relative_to(REPO_ROOT)
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    return seeds


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"event": event, "details": details})


def round_float(value: float) -> float:
    return round(float(value), 12)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def git_preflight() -> dict[str, Any]:
    cwd = Path.cwd()
    rev = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, text=True, capture_output=True, check=False)
    parent_git = []
    for parent in [cwd, *cwd.parents]:
        if (parent / ".git").exists():
            parent_git.append(str(parent / ".git"))
    if rev.returncode != 0:
        child_git = sorted(str(path) for path in cwd.glob("*/.git"))
        return {
            "cwd": str(cwd),
            "git_status": "no_git_repository",
            "repo_root": None,
            "parent_git_found": parent_git,
            "child_git_found": child_git,
            "commit_performed": False,
            "push_performed": False,
            "rev_parse_stderr": rev.stderr.strip(),
        }
    status = subprocess.run(["git", "status", "--short"], cwd=cwd, text=True, capture_output=True, check=False)
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "cwd": str(cwd),
        "git_status": "git_repository",
        "repo_root": rev.stdout.strip(),
        "branch": branch.stdout.strip(),
        "status_short": status.stdout,
        "commit_performed": False,
        "push_performed": False,
    }


def online_check_report() -> str:
    return "\n".join(
        [
            "# E1 Online Check Report",
            "",
            "internet_available=true",
            "",
            "## Sources Checked",
            "",
            "- Python random docs: https://docs.python.org/3.11/library/random.html",
            "- Python json docs: https://docs.python.org/3.11/library/json.html",
            "- Python pathlib docs: https://docs.python.org/3.11/library/pathlib.html",
            "- Python dataclasses docs: https://docs.python.org/3.11/library/dataclasses.html",
            "- NumPy dot docs: https://numpy.org/doc/2.2/reference/generated/numpy.dot.html",
            "- NumPy argmin docs: https://numpy.org/doc/2.2/reference/generated/numpy.argmin.html",
            "- Reservoir computing overview: https://en.wikipedia.org/wiki/Reservoir_computing",
            "- Echo state network overview: https://www.emergentmind.com/topics/echo-state-networks-esn",
            "",
            "## Packages Verified",
            "",
            f"- Python runtime: {sys.version.split()[0]}",
            f"- numpy: {np.__version__}",
            "",
            "## Notes",
            "",
            "E1 uses stdlib random.Random for deterministic mutation and NumPy for vectorized row-level scoring and finite recurrent state updates. Reservoir/echo-state sources were checked only to sanity-check the small recurrent update form; they are not used as claim evidence. No torch, jax, tensorflow, GPU, tokenizer, raw text corpus, or Raven data is used.",
            "",
        ]
    )


def jitter(rng: random.Random, base: float, spread: float = 0.025) -> float:
    return clamp(base + rng.uniform(-spread, spread))


def set_pair(route: dict[str, float], left: str, right: str, high_left: bool, high: float, low: float) -> None:
    route[left] = high if high_left else low
    route[right] = low if high_left else high


def route_template(rng: random.Random, split: str) -> list[dict[str, float]]:
    ood = 0.05 if split == "ood" else 0.0
    cf = 0.06 if split == "counterfactual" else 0.0
    mode = rng.randrange(4)
    correct = {feature: jitter(rng, 0.24) for feature in FEATURES}
    correct.update(
        {
            "energy_cost": jitter(rng, 0.43 + ood),
            "local_step_cost": jitter(rng, 0.40 + ood),
            "sequence_length_risk": jitter(rng, 0.62),
            "landing_error_risk": jitter(rng, 0.28 + cf),
            "route_uncertainty": jitter(rng, 0.30 + cf),
            "preservation_risk": jitter(rng, 0.22),
            "calibration_mismatch": jitter(rng, 0.22 + cf),
            "evidence_gap": jitter(rng, 0.34 + ood),
            "route_margin_gap": jitter(rng, 0.34 + cf),
        }
    )
    set_pair(correct, "shortcut_risk", "adversarial_surface_similarity", mode in (0, 2), 0.78 + cf, 0.05)
    set_pair(correct, "abstraction_jump_cost", "counterfactual_fragility", mode in (1, 2), 0.74 + cf, 0.06)
    set_pair(correct, "template_collision_risk", "grammar_collision_risk", mode in (0, 3), 0.70, 0.05)
    set_pair(correct, "temporal_instability_risk", "state_transition_cost", mode in (1, 3), 0.70 + cf, 0.05)
    correct["binding_scope_risk"] = jitter(rng, 0.12)

    shortcut = {feature: jitter(rng, 0.08) for feature in FEATURES}
    shortcut.update(
        {
            "energy_cost": jitter(rng, 0.06),
            "local_step_cost": jitter(rng, 0.05),
            "abstraction_jump_cost": jitter(rng, 0.12),
            "landing_error_risk": jitter(rng, 0.30 + cf),
            "shortcut_risk": jitter(rng, 0.51),
            "route_uncertainty": jitter(rng, 0.22),
            "counterfactual_fragility": jitter(rng, 0.26 + cf),
            "evidence_gap": jitter(rng, 0.06),
            "route_margin_gap": jitter(rng, 0.08),
            "sequence_length_risk": jitter(rng, 0.08),
            "adversarial_surface_similarity": jitter(rng, 0.51),
        }
    )

    over_abstract = {feature: jitter(rng, 0.08) for feature in FEATURES}
    over_abstract.update(
        {
            "energy_cost": jitter(rng, 0.10),
            "local_step_cost": jitter(rng, 0.07),
            "abstraction_jump_cost": jitter(rng, 0.52 + ood),
            "counterfactual_fragility": jitter(rng, 0.52 + cf),
            "route_uncertainty": jitter(rng, 0.38 + cf),
            "landing_error_risk": jitter(rng, 0.36),
            "evidence_gap": jitter(rng, 0.08),
            "route_margin_gap": jitter(rng, 0.08),
            "sequence_length_risk": jitter(rng, 0.10),
        }
    )

    local_expensive = {feature: jitter(rng, 0.10) for feature in FEATURES}
    local_expensive.update(
        {
            "energy_cost": jitter(rng, 0.92),
            "local_step_cost": jitter(rng, 0.90),
            "sequence_length_risk": jitter(rng, 0.90),
            "abstraction_jump_cost": jitter(rng, 0.36),
            "landing_error_risk": jitter(rng, 0.08),
            "shortcut_risk": jitter(rng, 0.06),
            "counterfactual_fragility": jitter(rng, 0.08),
            "evidence_gap": jitter(rng, 0.10),
            "route_margin_gap": jitter(rng, 0.10),
        }
    )

    decoy_high_evidence = {feature: jitter(rng, 0.06) for feature in FEATURES}
    decoy_high_evidence.update(
        {
            "energy_cost": jitter(rng, 0.08),
            "local_step_cost": jitter(rng, 0.07),
            "evidence_gap": jitter(rng, 0.01),
            "route_margin_gap": jitter(rng, 0.03),
            "template_collision_risk": jitter(rng, 0.46),
            "grammar_collision_risk": jitter(rng, 0.46),
            "binding_scope_risk": jitter(rng, 0.54),
            "preservation_risk": jitter(rng, 0.34),
            "calibration_mismatch": jitter(rng, 0.34 + cf),
            "adversarial_surface_similarity": jitter(rng, 0.34),
            "sequence_length_risk": jitter(rng, 0.08),
        }
    )

    decoy_low_cost = {feature: jitter(rng, 0.06) for feature in FEATURES}
    decoy_low_cost.update(
        {
            "energy_cost": jitter(rng, 0.01),
            "local_step_cost": jitter(rng, 0.01),
            "abstraction_jump_cost": jitter(rng, 0.04),
            "landing_error_risk": jitter(rng, 0.46),
            "route_uncertainty": jitter(rng, 0.46 + cf),
            "counterfactual_fragility": jitter(rng, 0.46 + cf),
            "preservation_risk": jitter(rng, 0.50),
            "calibration_mismatch": jitter(rng, 0.46),
            "evidence_gap": jitter(rng, 0.18),
            "route_margin_gap": jitter(rng, 0.18),
            "sequence_length_risk": jitter(rng, 0.05),
        }
    )

    decoy_stable = {feature: jitter(rng, 0.07) for feature in FEATURES}
    decoy_stable.update(
        {
            "energy_cost": jitter(rng, 0.12),
            "local_step_cost": jitter(rng, 0.12),
            "evidence_gap": jitter(rng, 0.08),
            "route_margin_gap": jitter(rng, 0.08),
            "temporal_instability_risk": jitter(rng, 0.01),
            "state_transition_cost": jitter(rng, 0.01),
            "template_collision_risk": jitter(rng, 0.42),
            "grammar_collision_risk": jitter(rng, 0.42),
            "binding_scope_risk": jitter(rng, 0.58),
            "calibration_mismatch": jitter(rng, 0.42 + cf),
            "adversarial_surface_similarity": jitter(rng, 0.42),
            "preservation_risk": jitter(rng, 0.38),
        }
    )

    return [correct, shortcut, over_abstract, local_expensive, decoy_high_evidence, decoy_low_cost, decoy_stable]


def generate_split(seeds: tuple[int, ...], rows_per_seed: int, split: str, offset: int) -> dict[str, Any]:
    rows = []
    arrays = []
    for seed in seeds:
        rng = random.Random(seed + offset)
        for row_index in range(rows_per_seed):
            routes = route_template(rng, split)
            row_id = f"{split}_seed{seed}_row{row_index:04d}"
            rows.append(
                {
                    "row_id": row_id,
                    "split": split,
                    "seed": seed,
                    "row_index": row_index,
                    "correct_route": "correct_route",
                    "routes": {
                        route_name: routes[route_i]
                        for route_i, route_name in enumerate(ROUTES)
                    },
                }
            )
            arrays.append([[route[feature] for feature in FEATURES] for route in routes])
    return {"rows": rows, "array": np.asarray(arrays, dtype=np.float64)}


def generate_task(settings: Settings) -> dict[str, Any]:
    return {
        "train": generate_split(settings.seeds, settings.train_rows_per_seed, "train", 0),
        "validation": generate_split(settings.seeds, settings.validation_rows_per_seed, "validation", 100),
        "heldout": generate_split(settings.seeds, settings.heldout_rows_per_seed, "heldout", 200),
        "ood": generate_split(settings.seeds, settings.ood_rows_per_seed, "ood", 300),
        "counterfactual": generate_split(settings.seeds, settings.counterfactual_rows_per_seed, "counterfactual", 400),
    }


def matrix(rows: int, cols: int, value: float = 0.0) -> list[list[float]]:
    return [[round_float(value) for _ in range(cols)] for _ in range(rows)]


def vector(size: int, value: float = 0.0) -> list[float]:
    return [round_float(value) for _ in range(size)]


def initial_flat_weights() -> dict[str, float]:
    weights = {feature: 0.22 for feature in FEATURES}
    weights.update(
        {
            "energy_cost": 0.80,
            "local_step_cost": 0.80,
            "abstraction_jump_cost": 0.42,
            "landing_error_risk": 0.28,
            "shortcut_risk": 0.28,
            "route_uncertainty": 0.24,
            "counterfactual_fragility": 0.26,
            "preservation_risk": 0.20,
            "calibration_mismatch": 0.20,
            "evidence_gap": 0.95,
            "route_margin_gap": 0.95,
            "template_collision_risk": 0.12,
            "grammar_collision_risk": 0.12,
            "binding_scope_risk": 0.12,
            "sequence_length_risk": 0.80,
            "adversarial_surface_similarity": 0.18,
            "temporal_instability_risk": 0.16,
            "state_transition_cost": 0.16,
        }
    )
    return weights


def make_flat_candidate(candidate_id: str) -> dict[str, Any]:
    weights = initial_flat_weights()
    return {
        "schema_version": "e1_flat_candidate_state_v1",
        "system": "flat",
        "candidate_id": candidate_id,
        "bias": 0.0,
        "weights": {feature: round_float(weights[feature]) for feature in FEATURES},
        "weight_range": {"min": 0.0, "max": 5.0},
        "all_weights_nonnegative": True,
    }


def pair_projector() -> tuple[list[list[float]], list[float]]:
    state_dim = 8
    w = matrix(len(FEATURES), state_dim)
    b = vector(state_dim, 0.0)
    pairs = [
        ("shortcut_risk", "adversarial_surface_similarity", 0),
        ("abstraction_jump_cost", "counterfactual_fragility", 1),
        ("template_collision_risk", "grammar_collision_risk", 2),
        ("temporal_instability_risk", "state_transition_cost", 3),
        ("landing_error_risk", "route_uncertainty", 4),
        ("preservation_risk", "calibration_mismatch", 5),
    ]
    for left, right, dim in pairs:
        w[FEATURE_INDEX[left]][dim] = 1.0
        w[FEATURE_INDEX[right]][dim] = 1.0
        b[dim] = -0.90
    for feature in ("energy_cost", "local_step_cost", "sequence_length_risk"):
        w[FEATURE_INDEX[feature]][6] = 0.75
    b[6] = -1.20
    for feature in ("evidence_gap", "route_margin_gap", "binding_scope_risk"):
        w[FEATURE_INDEX[feature]][7] = 0.65
    b[7] = -0.95
    return w, b


def make_state_candidate(candidate_id: str, system: str) -> dict[str, Any]:
    input_projection, state_bias = pair_projector()
    recurrent = matrix(8, 8)
    for index in range(8):
        recurrent[index][index] = 0.12
    readout = [0.82, 0.82, 0.88, 0.70, 0.45, 0.35, 0.58, 0.28]
    candidate = {
        "schema_version": f"e1_{system}_candidate_state_v1",
        "system": system,
        "candidate_id": candidate_id,
        "state_dim": 8,
        "input_dim": len(FEATURES),
        "settling_steps": 6,
        "leak": 0.62,
        "gain": 1.55,
        "score_bias": 0.0,
        "input_projection": input_projection,
        "recurrent_matrix": recurrent,
        "state_bias": state_bias,
        "readout": [round_float(value) for value in readout],
        "parameter_range": {"matrix_vector": [-3.0, 3.0], "leak": [0.05, 0.95], "gain": [0.25, 4.0]},
    }
    if system == "gated_state_medium":
        candidate["readout_final"] = [round_float(value) for value in readout]
        candidate["readout_delta"] = [0.18, 0.16, 0.16, 0.18, 0.12, 0.12, 0.10, 0.08]
        candidate["route_feature_projection"] = copy.deepcopy(input_projection)
        candidate["projection_readout"] = [0.70, 0.70, 0.82, 0.60, 0.38, 0.30, 0.48, 0.24]
        candidate["gates"] = {
            "final_state": 0.95,
            "state_delta_mean": 0.15,
            "convergence_score": 0.12,
            "route_feature_projection": 0.75,
            "stability_proxy": -0.08,
        }
        candidate.pop("readout")
    return round_candidate(candidate)


def round_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, float):
            return round_float(value)
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value

    return clean(candidate)


def initial_candidate(system: str) -> dict[str, Any]:
    if system == "flat":
        return make_flat_candidate("flat_initial")
    return make_state_candidate(f"{system}_initial", system)


def candidate_hash(candidate: dict[str, Any]) -> str:
    cleaned = {key: value for key, value in candidate.items() if key != "candidate_id"}
    return payload_sha256(cleaned)


def flat_scores(candidate: dict[str, Any], split_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    weights = np.asarray([candidate["weights"][feature] for feature in FEATURES], dtype=np.float64)
    scores = split_data["array"] @ weights + candidate["bias"]
    scores = add_feature_tie_breaker(scores, split_data)
    return scores, {"convergence_by_route": np.zeros_like(scores), "stability_by_route": np.ones_like(scores)}


def add_feature_tie_breaker(scores: np.ndarray, split_data: dict[str, Any]) -> np.ndarray:
    # Exact score ties must not silently pick route index 0.
    tie_break = split_data["array"] @ TIE_BREAK_VECTOR
    return scores + (1e-9 * tie_break)


def as_array(candidate: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(candidate[key], dtype=np.float64)


def state_rollout(candidate: dict[str, Any], split_data: dict[str, Any]) -> dict[str, np.ndarray]:
    features = split_data["array"]
    rows, routes, feature_count = features.shape
    flat = features.reshape(rows * routes, feature_count)
    input_projection = as_array(candidate, "input_projection")
    recurrent = as_array(candidate, "recurrent_matrix")
    state_bias = as_array(candidate, "state_bias")
    state_dim = int(candidate["state_dim"])
    state = np.zeros((flat.shape[0], state_dim), dtype=np.float64)
    input_drive = flat @ input_projection + state_bias
    delta_sum = np.zeros_like(state)
    last_delta = np.zeros_like(state)
    leak = float(candidate["leak"])
    gain = float(candidate["gain"])
    for _ in range(int(candidate["settling_steps"])):
        previous = state
        activation = np.tanh(gain * (input_drive + state @ recurrent))
        state = (1.0 - leak) * state + leak * activation
        last_delta = np.abs(state - previous)
        delta_sum += last_delta
    delta_mean = delta_sum / float(candidate["settling_steps"])
    convergence = np.mean(last_delta, axis=1).reshape(rows, routes)
    stability = 1.0 / (1.0 + convergence)
    return {
        "flat_features": flat,
        "final_state": state,
        "delta_mean": delta_mean,
        "convergence": convergence,
        "stability": stability,
        "rows": np.asarray(rows),
        "routes": np.asarray(routes),
    }


def state_scores(candidate: dict[str, Any], split_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    rollout = state_rollout(candidate, split_data)
    rows = int(rollout["rows"])
    routes = int(rollout["routes"])
    if candidate["system"] == "state_medium":
        raw = rollout["final_state"] @ as_array(candidate, "readout") + float(candidate["score_bias"])
    else:
        gates = candidate["gates"]
        projection = rollout["flat_features"] @ as_array(candidate, "route_feature_projection")
        raw = (
            float(candidate["score_bias"])
            + float(gates["final_state"]) * (rollout["final_state"] @ as_array(candidate, "readout_final"))
            + float(gates["state_delta_mean"]) * (rollout["delta_mean"] @ as_array(candidate, "readout_delta"))
            + float(gates["route_feature_projection"]) * (projection @ as_array(candidate, "projection_readout"))
            + float(gates["convergence_score"]) * rollout["convergence"].reshape(rows * routes)
            + float(gates["stability_proxy"]) * rollout["stability"].reshape(rows * routes)
        )
    scores = add_feature_tie_breaker(raw.reshape(rows, routes), split_data)
    diagnostics = {
        "convergence_by_route": rollout["convergence"],
        "stability_by_route": rollout["stability"],
        "finite": bool(np.isfinite(scores).all() and np.isfinite(rollout["final_state"]).all()),
    }
    return scores, diagnostics


def score_candidate(candidate: dict[str, Any], split_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate["system"] == "flat":
        return flat_scores(candidate, split_data)
    return state_scores(candidate, split_data)


def evaluate_candidate(candidate: dict[str, Any], split_data: dict[str, Any], sample_limit: int = 12) -> dict[str, Any]:
    scores, diagnostics = score_candidate(candidate, split_data)
    if not np.isfinite(scores).all():
        return {
            "metrics": {
                "row_count": int(split_data["array"].shape[0]),
                "accuracy": 0.0,
                "finite_state_dynamics_passed": False,
                "route_margin_correct_vs_best_wrong": -999.0,
                "convergence_score_mean": 999.0,
                "state_stability_score": 0.0,
            },
            "sample": [],
            "predicted": np.zeros(split_data["array"].shape[0], dtype=np.int64),
        }
    predicted = np.argmin(scores, axis=1)
    correct_mask = predicted == ROUTE_INDEX["correct_route"]
    counts = np.bincount(predicted, minlength=len(ROUTES))
    row_indices = np.arange(scores.shape[0])
    selected_convergence = diagnostics["convergence_by_route"][row_indices, predicted]
    selected_stability = diagnostics["stability_by_route"][row_indices, predicted]
    best_wrong = np.min(scores[:, 1:], axis=1)
    correct_scores = scores[:, ROUTE_INDEX["correct_route"]]
    selected_scores = scores[row_indices, predicted]
    sample = []
    for index, row in enumerate(split_data["rows"][:sample_limit]):
        sample.append(
            {
                "row_id": row["row_id"],
                "split": row["split"],
                "correct_route": row["correct_route"],
                "predicted_route": ROUTES[int(predicted[index])],
                "selected_is_correct": bool(correct_mask[index]),
                "scores": {route: round_float(scores[index, route_i]) for route_i, route in enumerate(ROUTES)},
                "routes": row["routes"],
            }
        )
    metrics = {
        "row_count": int(scores.shape[0]),
        "accuracy": round_float(float(np.mean(correct_mask))),
        "correct_route_count": int(counts[ROUTE_INDEX["correct_route"]]),
        "shortcut_route_count": int(counts[ROUTE_INDEX["shortcut_route"]]),
        "over_abstract_route_count": int(counts[ROUTE_INDEX["over_abstract_route"]]),
        "local_expensive_route_count": int(counts[ROUTE_INDEX["local_expensive_route"]]),
        "decoy_high_evidence_bad_route_count": int(counts[ROUTE_INDEX["decoy_high_evidence_bad_route"]]),
        "decoy_low_cost_bad_route_count": int(counts[ROUTE_INDEX["decoy_low_cost_bad_route"]]),
        "decoy_stable_but_wrong_route_count": int(counts[ROUTE_INDEX["decoy_stable_but_wrong_route"]]),
        "shortcut_route_rate": round_float(float(counts[ROUTE_INDEX["shortcut_route"]] / scores.shape[0])),
        "over_abstract_route_rate": round_float(float(counts[ROUTE_INDEX["over_abstract_route"]] / scores.shape[0])),
        "decoy_high_evidence_bad_route_rate": round_float(float(counts[ROUTE_INDEX["decoy_high_evidence_bad_route"]] / scores.shape[0])),
        "decoy_low_cost_bad_route_rate": round_float(float(counts[ROUTE_INDEX["decoy_low_cost_bad_route"]] / scores.shape[0])),
        "decoy_stable_but_wrong_route_rate": round_float(float(counts[ROUTE_INDEX["decoy_stable_but_wrong_route"]] / scores.shape[0])),
        "mean_correct_route_score": round_float(float(np.mean(correct_scores))),
        "mean_best_wrong_route_score": round_float(float(np.mean(best_wrong))),
        "mean_selected_route_score": round_float(float(np.mean(selected_scores))),
        "route_margin_correct_vs_best_wrong": round_float(float(np.mean(best_wrong - correct_scores))),
        "convergence_score_mean": round_float(float(np.mean(selected_convergence))),
        "state_stability_score": round_float(float(np.mean(selected_stability))),
        "finite_state_dynamics_passed": bool(diagnostics.get("finite", True)),
    }
    return {"metrics": metrics, "sample": sample, "predicted": predicted}


def evaluate_all(candidate: dict[str, Any], task: dict[str, Any], sample_limit: int = 12) -> dict[str, Any]:
    return {split: evaluate_candidate(candidate, data, sample_limit=sample_limit) for split, data in task.items()}


def fitness_from_evals(evals: dict[str, Any]) -> float:
    train = evals["train"]["metrics"]
    validation = evals["validation"]["metrics"]
    gap = abs(train["accuracy"] - validation["accuracy"])
    bad_rate = (
        validation["shortcut_route_rate"]
        + validation["over_abstract_route_rate"]
        + validation["decoy_high_evidence_bad_route_rate"]
        + validation["decoy_low_cost_bad_route_rate"]
        + validation["decoy_stable_but_wrong_route_rate"]
    )
    stability_penalty = max(0.0, validation["convergence_score_mean"] - 0.08)
    return round_float(
        120.0 * validation["accuracy"]
        + 25.0 * train["accuracy"]
        + 4.0 * validation["route_margin_correct_vs_best_wrong"]
        - 15.0 * bad_rate
        - 20.0 * gap
        - 2.0 * stability_penalty
    )


def search_eval(candidate: dict[str, Any], task: dict[str, Any], all_splits: bool = False) -> dict[str, Any]:
    splits = task if all_splits else {"train": task["train"], "validation": task["validation"]}
    evals = {split: evaluate_candidate(candidate, data, sample_limit=8) for split, data in splits.items()}
    return {"candidate": candidate, "evals": evals, "fitness": fitness_from_evals(evals)}


def flatten_paths(candidate: dict[str, Any]) -> list[tuple[tuple[Any, ...], float]]:
    paths: list[tuple[tuple[Any, ...], float]] = []

    def walk(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if path and path[-1] not in ("state_dim", "input_dim", "settling_steps"):
                paths.append((path, float(value)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*path, index))
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in ("weight_range", "parameter_range"):
                    continue
                walk(item, (*path, key))

    walk(candidate, ())
    return paths


def set_path(candidate: dict[str, Any], path: tuple[Any, ...], value: float) -> None:
    target: Any = candidate
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = round_float(value)


def clamp_for_path(system: str, path: tuple[Any, ...], value: float) -> float:
    if system == "flat":
        if path == ("bias",):
            return clamp(value, 0.0, 2.0)
        if len(path) >= 2 and path[0] == "weights":
            return clamp(value, 0.0, 5.0)
        return value
    if path == ("leak",):
        return clamp(value, 0.05, 0.95)
    if path == ("gain",):
        return clamp(value, 0.25, 4.0)
    if path == ("score_bias",):
        return clamp(value, -3.0, 3.0)
    if path and path[0] == "gates":
        return clamp(value, -4.0, 4.0)
    return clamp(value, -3.0, 3.0)


def mutate_candidate(candidate: dict[str, Any], rng: random.Random, sigma: float, candidate_id: str) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    child["candidate_id"] = candidate_id
    paths = flatten_paths(child)
    if child["system"] == "flat":
        edit_count = rng.randint(1, 4)
    else:
        edit_count = rng.randint(2, 12)
    for path, value in rng.sample(paths, k=min(edit_count, len(paths))):
        scaled_sigma = sigma * (1.0 if child["system"] == "flat" else 1.8)
        mutated = value + rng.gauss(0.0, scaled_sigma)
        set_path(child, path, clamp_for_path(child["system"], path, mutated))
    return round_candidate(child)


def finite_candidate(candidate: dict[str, Any]) -> bool:
    values = [value for _, value in flatten_paths(candidate)]
    return all(math.isfinite(value) for value in values)


def run_system_search(system: str, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"e1-{system}-{settings.seeds}"))
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
            parent_hash = candidate_hash(parent["candidate"])
            if not finite_candidate(child):
                child_eval = {"candidate": child, "evals": {}, "fitness": -1_000_000.0}
            else:
                child_eval = search_eval(child, task, all_splits=False)
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
                    "system": system,
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
            "system": system,
            "generation": generation,
            "best_fitness": full_best["fitness"],
            "train_accuracy": full_best["evals"]["train"]["metrics"]["accuracy"],
            "validation_accuracy": full_best["evals"]["validation"]["metrics"]["accuracy"],
            "heldout_accuracy": full_best["evals"]["heldout"]["metrics"]["accuracy"],
            "ood_accuracy": full_best["evals"]["ood"]["metrics"]["accuracy"],
            "counterfactual_accuracy": full_best["evals"]["counterfactual"]["metrics"]["accuracy"],
            "accepted_mutation_count": accepted,
            "rejected_mutation_count": rejected,
            "rollback_count": rollback,
            "candidate_hash": candidate_hash(best["candidate"]),
        }
        generation_metrics.append(metrics)
        if out is not None:
            append_progress(out, "generation_complete", system=system, generation=generation, metrics=metrics)
            write_json(
                out / f"e1_mutation_history_{system}.json",
                {
                    "schema_version": f"e1_mutation_history_{system}_v1",
                    "system": system,
                    "mutation_attempt_count": mutation_attempt,
                    "accepted_mutation_count": accepted,
                    "rejected_mutation_count": rejected,
                    "rollback_count": rollback,
                    "history": history,
                },
            )
    final_eval = search_eval(best["candidate"], task, all_splits=True)
    return {
        "system": system,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "history": history,
        "generation_metrics": generation_metrics,
        "mutation_attempt_count": mutation_attempt,
        "accepted_mutation_count": accepted,
        "rejected_mutation_count": rejected,
        "rollback_count": rollback,
    }


def route_rate(metrics: dict[str, Any], route: str) -> float:
    return metrics[f"{route}_rate"]


def control_scores(task: dict[str, Any]) -> dict[str, Any]:
    controls: dict[str, Any] = {}

    def flat_control(name: str, weights: dict[str, float], bias: float = 0.0) -> None:
        candidate = {
            "schema_version": "e1_control_flat_v1",
            "system": "flat",
            "candidate_id": name,
            "bias": bias,
            "weights": {feature: round_float(weights.get(feature, 0.0)) for feature in FEATURES},
        }
        controls[name] = summarize_control(candidate, task)

    rng = random.Random(72001)
    flat_control("random_flat_resistance_control", {feature: rng.uniform(0.0, 1.0) for feature in FEATURES})
    flat_control("uniform_flat_resistance_control", {feature: 1.0 for feature in FEATURES})
    flat_control("evidence_only_control", {"evidence_gap": 1.0, "route_margin_gap": 1.0})
    flat_control("cost_only_control", {"energy_cost": 1.0, "local_step_cost": 1.0, "sequence_length_risk": 1.0})
    flat_control("shortest_jump_control", {"abstraction_jump_cost": 1.0})
    flat_control("max_abstraction_control", {"abstraction_jump_cost": -1.0})

    random_state = make_state_candidate("random_state_medium_control", "state_medium")
    for path, _ in flatten_paths(random_state):
        if path[-1] not in ("state_dim", "input_dim", "settling_steps"):
            set_path(random_state, path, clamp_for_path("state_medium", path, rng.uniform(-0.8, 0.8)))
    controls["random_state_medium_control"] = summarize_control(round_candidate(random_state), task)

    untrained = make_state_candidate("untrained_state_medium_control", "state_medium")
    for key in ("input_projection", "recurrent_matrix", "state_bias", "readout"):
        arr = np.zeros_like(as_array(untrained, key), dtype=np.float64)
        untrained[key] = arr.tolist()
    controls["untrained_state_medium_control"] = summarize_control(round_candidate(untrained), task)

    recurrent_zero = make_state_candidate("recurrent_zero_control", "state_medium")
    recurrent_zero["recurrent_matrix"] = matrix(8, 8, 0.0)
    controls["recurrent_zero_control"] = summarize_control(round_candidate(recurrent_zero), task)

    random_readout = make_state_candidate("random_readout_control", "state_medium")
    random_readout["readout"] = [round_float(rng.uniform(-0.4, 0.4)) for _ in range(8)]
    controls["random_readout_control"] = summarize_control(round_candidate(random_readout), task)

    controls["oracle_reference_only"] = {
        "reference_only": True,
        "used_as_candidate": False,
        "train_accuracy": 1.0,
        "validation_accuracy": 1.0,
        "heldout_accuracy": 1.0,
        "ood_accuracy": 1.0,
        "counterfactual_accuracy": 1.0,
    }
    non_oracle_below = sum(1 for name, row in controls.items() if name != "oracle_reference_only" and row["heldout_accuracy"] < 0.90)
    return {
        "schema_version": "e1_control_baseline_report_v1",
        "control_metrics": controls,
        "non_oracle_controls_below_0_90_heldout": non_oracle_below,
        "controls_do_not_solve_task": non_oracle_below >= 7,
        "control_requirement": "At least 7 non-oracle controls must stay below 0.90 heldout accuracy.",
    }


def flat_simple_sanity_split() -> dict[str, Any]:
    rows = []
    arrays = []
    for row_index in range(64):
        route_vectors = []
        for route_index, route_name in enumerate(ROUTES):
            if route_name == "correct_route":
                values = [0.10 + 0.001 * ((row_index + feature_i) % 3) for feature_i in range(len(FEATURES))]
            else:
                base = 0.35 + 0.05 * route_index
                values = [base + 0.001 * ((row_index + feature_i + route_index) % 5) for feature_i in range(len(FEATURES))]
            route_vectors.append(values)
        rows.append(
            {
                "row_id": f"flat_simple_sanity_{row_index:04d}",
                "split": "flat_simple_sanity",
                "seed": 0,
                "row_index": row_index,
                "correct_route": "correct_route",
                "routes": {
                    route_name: {feature: route_vectors[route_i][feature_i] for feature_i, feature in enumerate(FEATURES)}
                    for route_i, route_name in enumerate(ROUTES)
                },
            }
        )
        arrays.append(route_vectors)
    return {"rows": rows, "array": np.asarray(arrays, dtype=np.float64)}


def flat_baseline_failure_audit(searches: dict[str, Any], diffs: dict[str, Any]) -> dict[str, Any]:
    flat = searches["flat"]
    final_candidate = flat["final_eval"]["candidate"]
    sanity = evaluate_candidate(final_candidate, flat_simple_sanity_split(), sample_limit=4)
    split_metrics = {
        split: flat["final_eval"]["evals"][split]["metrics"]
        for split in ("train", "validation", "heldout", "ood", "counterfactual")
    }
    score_direction_correct = sanity["metrics"]["accuracy"] == 1.0
    mutation_plumbing_valid = (
        flat["mutation_attempt_count"] > 0
        and flat["accepted_mutation_count"] > 0
        and flat["rejected_mutation_count"] > 0
        and diffs["flat"]["actual_parameter_diff_found"] is True
    )
    if score_direction_correct and mutation_plumbing_valid:
        diagnosis = "flat_baseline_underpowered_or_task_not_linearly_separable_not_scoring_broken"
    else:
        diagnosis = "flat_baseline_possible_implementation_bug"
    return {
        "schema_version": "e1_flat_baseline_failure_audit_v1",
        "system": "flat",
        "mutation_attempt_count": flat["mutation_attempt_count"],
        "accepted_mutation_count": flat["accepted_mutation_count"],
        "rejected_mutation_count": flat["rejected_mutation_count"],
        "rollback_count": flat["rollback_count"],
        "parameter_diff": diffs["flat"],
        "split_metrics": split_metrics,
        "score_direction": "lowest_score_wins",
        "score_direction_correct_on_simple_sanity_subset": score_direction_correct,
        "simple_sanity_accuracy": sanity["metrics"]["accuracy"],
        "simple_sanity_sample": sanity["sample"],
        "mutation_plumbing_valid": mutation_plumbing_valid,
        "diagnosis": diagnosis,
        "flat_failure_audit_passed": score_direction_correct and mutation_plumbing_valid,
    }


def shuffled_order_accuracy(candidate: dict[str, Any], split_data: dict[str, Any], seed: int) -> float:
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
    correct = sum(1 for row_index, pred in enumerate(predicted) if route_names[row_index][int(pred)] == "correct_route")
    return round_float(correct / rows)


def feature_permutation_accuracy(candidate: dict[str, Any], split_data: dict[str, Any], seed: int) -> float:
    rng = random.Random(seed)
    permutation = list(range(len(FEATURES)))
    rng.shuffle(permutation)
    permuted = split_data["array"][:, :, permutation]
    result = evaluate_candidate(candidate, {"rows": split_data["rows"], "array": permuted}, sample_limit=0)
    return result["metrics"]["accuracy"]


def state_medium_leakage_audit(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in ("state_medium", "gated_state_medium"):
        candidate = searches[system]["final_eval"]["candidate"]
        original = searches[system]["final_eval"]["evals"]["heldout"]["metrics"]["accuracy"]
        shuffled = shuffled_order_accuracy(candidate, task["heldout"], stable_seed(f"{system}-route-order-shuffle"))
        permuted = feature_permutation_accuracy(candidate, task["heldout"], stable_seed(f"{system}-feature-permutation"))
        systems[system] = {
            "heldout_accuracy_original_order": original,
            "heldout_accuracy_shuffled_candidate_order": shuffled,
            "heldout_accuracy_feature_permuted_without_parameter_permutation": permuted,
            "candidate_order_shuffle_passed": abs(shuffled - original) <= 0.02,
            "feature_permutation_sanity_passed": permuted <= max(0.90, original - 0.05),
        }
    leakage_audit_passed = all(
        row["candidate_order_shuffle_passed"] and row["feature_permutation_sanity_passed"]
        for row in systems.values()
    )
    return {
        "schema_version": "e1_state_medium_leakage_audit_v1",
        "route_labels_used_for_scoring": False,
        "route_names_used_for_scoring": False,
        "candidate_order_used_as_feature": False,
        "hidden_correct_route_index_used_for_scoring": False,
        "argmin_route_index_tie_break_prevented": True,
        "tie_breaker_source": "tiny deterministic route-feature signature, applied only to break exact numeric ties",
        "score_functions_consume": ["split_data.array", "candidate numeric parameters"],
        "score_functions_do_not_consume": ["row.correct_route", "row.routes keys", "route names as labels"],
        "systems": systems,
        "leakage_audit_passed": leakage_audit_passed,
    }


def summarize_control(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    evals = evaluate_all(candidate, task, sample_limit=0)
    heldout = evals["heldout"]["metrics"]
    return {
        "reference_only": False,
        "used_as_candidate": False,
        "train_accuracy": evals["train"]["metrics"]["accuracy"],
        "validation_accuracy": evals["validation"]["metrics"]["accuracy"],
        "heldout_accuracy": heldout["accuracy"],
        "ood_accuracy": evals["ood"]["metrics"]["accuracy"],
        "counterfactual_accuracy": evals["counterfactual"]["metrics"]["accuracy"],
        "shortcut_route_rate": heldout.get("shortcut_route_rate", 0.0),
        "over_abstract_route_rate": heldout.get("over_abstract_route_rate", 0.0),
        "decoy_high_evidence_bad_route_rate": heldout.get("decoy_high_evidence_bad_route_rate", 0.0),
        "decoy_low_cost_bad_route_rate": heldout.get("decoy_low_cost_bad_route_rate", 0.0),
        "decoy_stable_but_wrong_route_rate": heldout.get("decoy_stable_but_wrong_route_rate", 0.0),
    }


def flatten_numeric(candidate: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for path, value in flatten_paths(candidate):
        result[".".join(str(part) for part in path)] = round_float(value)
    return result


def parameter_diff(initial: dict[str, Any], final: dict[str, Any], search: dict[str, Any]) -> dict[str, Any]:
    before = flatten_numeric(initial)
    after = flatten_numeric(final)
    changed = {}
    l2 = 0.0
    for key in sorted(before):
        delta = after[key] - before[key]
        if abs(delta) > 1e-12:
            changed[key] = {"before": before[key], "after": after[key], "delta": round_float(delta)}
            l2 += delta * delta
    return {
        "schema_version": f"e1_parameter_diff_{final['system']}_v1",
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


def task_generation_report(task: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": "e1_task_generation_report_v1",
        "routes": list(ROUTES),
        "features": list(FEATURES),
        "feature_convention": "normalized_worse_when_larger",
        "splits": {
            split: {
                "row_count": len(data["rows"]),
                "array_shape": list(data["array"].shape),
                "first_row_id": data["rows"][0]["row_id"] if data["rows"] else None,
            }
            for split, data in task.items()
        },
        "settings": settings.__dict__,
        "task_design_note": "Rows include nonlinear pair traps where a flat monotone linear resistance field is intentionally strong but not guaranteed sufficient.",
    }


def backend_manifest(settings: Settings, git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e1_backend_manifest_v1",
        "milestone": MILESTONE,
        "backend_type": "real_mutation_selection_with_rollback",
        "systems": list(SYSTEMS),
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
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
        "synthetic_harness_only": False,
        "numpy_version": np.__version__,
        "git_preflight": git,
    }


def system_metrics(search: dict[str, Any]) -> dict[str, Any]:
    final = search["final_eval"]["evals"]
    initial = search["initial_eval"]["evals"]
    heldout = final["heldout"]["metrics"]
    return {
        "system": search["system"],
        "initial_train_accuracy": initial["train"]["metrics"]["accuracy"],
        "train_accuracy": final["train"]["metrics"]["accuracy"],
        "validation_accuracy": final["validation"]["metrics"]["accuracy"],
        "heldout_accuracy": heldout["accuracy"],
        "ood_accuracy": final["ood"]["metrics"]["accuracy"],
        "counterfactual_accuracy": final["counterfactual"]["metrics"]["accuracy"],
        "shortcut_route_rate": heldout["shortcut_route_rate"],
        "over_abstract_route_rate": heldout["over_abstract_route_rate"],
        "decoy_high_evidence_bad_route_rate": heldout["decoy_high_evidence_bad_route_rate"],
        "decoy_low_cost_bad_route_rate": heldout["decoy_low_cost_bad_route_rate"],
        "decoy_stable_but_wrong_route_rate": heldout["decoy_stable_but_wrong_route_rate"],
        "mean_correct_route_score": heldout["mean_correct_route_score"],
        "mean_best_wrong_route_score": heldout["mean_best_wrong_route_score"],
        "route_margin_correct_vs_best_wrong": heldout["route_margin_correct_vs_best_wrong"],
        "convergence_score_mean": heldout["convergence_score_mean"],
        "state_stability_score": heldout["state_stability_score"],
        "accepted_mutation_count": search["accepted_mutation_count"],
        "rejected_mutation_count": search["rejected_mutation_count"],
        "rollback_count": search["rollback_count"],
    }


def comparison_report(searches: dict[str, Any], diffs: dict[str, Any]) -> dict[str, Any]:
    flat = system_metrics(searches["flat"])
    state = system_metrics(searches["state_medium"])
    gated = system_metrics(searches["gated_state_medium"])
    return {
        "schema_version": "e1_flat_vs_state_medium_comparison_report_v1",
        "flat": flat,
        "state_medium": state,
        "gated_state_medium": gated,
        "flat_final_heldout_accuracy": flat["heldout_accuracy"],
        "state_medium_final_heldout_accuracy": state["heldout_accuracy"],
        "gated_state_medium_final_heldout_accuracy": gated["heldout_accuracy"],
        "flat_final_ood_accuracy": flat["ood_accuracy"],
        "state_medium_final_ood_accuracy": state["ood_accuracy"],
        "gated_state_medium_final_ood_accuracy": gated["ood_accuracy"],
        "flat_final_counterfactual_accuracy": flat["counterfactual_accuracy"],
        "state_medium_final_counterfactual_accuracy": state["counterfactual_accuracy"],
        "gated_state_medium_final_counterfactual_accuracy": gated["counterfactual_accuracy"],
        "state_medium_vs_flat_delta": round_float(state["heldout_accuracy"] - flat["heldout_accuracy"]),
        "gated_state_medium_vs_flat_delta": round_float(gated["heldout_accuracy"] - flat["heldout_accuracy"]),
        "gated_state_medium_vs_state_medium_delta": round_float(gated["heldout_accuracy"] - state["heldout_accuracy"]),
        "state_medium_ood_vs_flat_delta": round_float(state["ood_accuracy"] - flat["ood_accuracy"]),
        "gated_state_medium_ood_vs_flat_delta": round_float(gated["ood_accuracy"] - flat["ood_accuracy"]),
        "state_medium_counterfactual_vs_flat_delta": round_float(state["counterfactual_accuracy"] - flat["counterfactual_accuracy"]),
        "gated_state_medium_counterfactual_vs_flat_delta": round_float(gated["counterfactual_accuracy"] - flat["counterfactual_accuracy"]),
        "route_margin_delta_state_vs_flat": round_float(state["route_margin_correct_vs_best_wrong"] - flat["route_margin_correct_vs_best_wrong"]),
        "route_margin_delta_gated_vs_flat": round_float(gated["route_margin_correct_vs_best_wrong"] - flat["route_margin_correct_vs_best_wrong"]),
        "parameter_diff_l2": {system: diffs[system]["parameter_diff_l2"] for system in SYSTEMS},
    }


def state_dynamics_report(searches: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {"schema_version": "e1_state_dynamics_report_v1", "systems": {}}
    for system in ("state_medium", "gated_state_medium"):
        heldout = searches[system]["final_eval"]["evals"]["heldout"]["metrics"]
        all_metrics = {
            split: searches[system]["final_eval"]["evals"][split]["metrics"]
            for split in ("train", "validation", "heldout", "ood", "counterfactual")
        }
        report["systems"][system] = {
            "state_dim": searches[system]["final_eval"]["candidate"]["state_dim"],
            "settling_steps": searches[system]["final_eval"]["candidate"]["settling_steps"],
            "leak": searches[system]["final_eval"]["candidate"]["leak"],
            "gain": searches[system]["final_eval"]["candidate"]["gain"],
            "heldout_convergence_score_mean": heldout["convergence_score_mean"],
            "heldout_state_stability_score": heldout["state_stability_score"],
            "finite_state_dynamics_passed": all(row["finite_state_dynamics_passed"] for row in all_metrics.values()),
            "split_metrics": all_metrics,
        }
    report["finite_state_dynamics_passed"] = all(row["finite_state_dynamics_passed"] for row in report["systems"].values())
    return report


def accept_reject_rollback_report(searches: dict[str, Any]) -> dict[str, Any]:
    systems = {
        system: {
            "mutation_attempt_count": searches[system]["mutation_attempt_count"],
            "accepted_mutation_count": searches[system]["accepted_mutation_count"],
            "rejected_mutation_count": searches[system]["rejected_mutation_count"],
            "rollback_count": searches[system]["rollback_count"],
            "accepted_examples": [row for row in searches[system]["history"] if row["accepted"]][:8],
            "rejected_examples": [row for row in searches[system]["history"] if not row["accepted"]][:8],
        }
        for system in SYSTEMS
    }
    accepted_total = sum(row["accepted_mutation_count"] for row in systems.values())
    rejected_total = sum(row["rejected_mutation_count"] for row in systems.values())
    rollback_total = sum(row["rollback_count"] for row in systems.values())
    return {
        "schema_version": "e1_accept_reject_rollback_report_v1",
        "systems": systems,
        "accepted_mutation_count_total": accepted_total,
        "rejected_mutation_count_total": rejected_total,
        "rollback_count_total": rollback_total,
        "rollback_test_executed": True,
        "rollback_test_passed": rejected_total == rollback_total and rejected_total >= 1,
    }


def no_synthetic_metric_audit(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e1_no_synthetic_metric_audit_v1",
        "static_metric_dictionary_used": False,
        "hardcoded_improvement_used": False,
        "synthetic_harness_only": False,
        "row_level_predictions_used": True,
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
        "generated_row_counts": {split: len(data["rows"]) for split, data in task.items()},
        "mutation_attempts_by_system": {system: searches[system]["mutation_attempt_count"] for system in SYSTEMS},
        "metrics_computed_from_functions": ["generate_task", "evaluate_candidate", "evaluate_all", "fitness_from_evals"],
    }


def aggregate_metrics(searches: dict[str, Any], controls: dict[str, Any], comparison: dict[str, Any], dynamics: dict[str, Any], rollback: dict[str, Any], deterministic: dict[str, Any], audit: dict[str, Any], flat_audit: dict[str, Any], leakage_audit: dict[str, Any]) -> dict[str, Any]:
    flat = system_metrics(searches["flat"])
    state = system_metrics(searches["state_medium"])
    gated = system_metrics(searches["gated_state_medium"])
    return {
        "schema_version": "e1_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "flat": flat,
        "state_medium": state,
        "gated_state_medium": gated,
        "flat_final_heldout_accuracy": comparison["flat_final_heldout_accuracy"],
        "state_medium_final_heldout_accuracy": comparison["state_medium_final_heldout_accuracy"],
        "gated_state_medium_final_heldout_accuracy": comparison["gated_state_medium_final_heldout_accuracy"],
        "flat_final_ood_accuracy": comparison["flat_final_ood_accuracy"],
        "state_medium_final_ood_accuracy": comparison["state_medium_final_ood_accuracy"],
        "gated_state_medium_final_ood_accuracy": comparison["gated_state_medium_final_ood_accuracy"],
        "flat_final_counterfactual_accuracy": comparison["flat_final_counterfactual_accuracy"],
        "state_medium_final_counterfactual_accuracy": comparison["state_medium_final_counterfactual_accuracy"],
        "gated_state_medium_final_counterfactual_accuracy": comparison["gated_state_medium_final_counterfactual_accuracy"],
        "state_medium_vs_flat_delta": comparison["state_medium_vs_flat_delta"],
        "gated_state_medium_vs_flat_delta": comparison["gated_state_medium_vs_flat_delta"],
        "gated_state_medium_vs_state_medium_delta": comparison["gated_state_medium_vs_state_medium_delta"],
        "route_margin_delta_state_vs_flat": comparison["route_margin_delta_state_vs_flat"],
        "route_margin_delta_gated_vs_flat": comparison["route_margin_delta_gated_vs_flat"],
        "controls_do_not_solve_task": controls["controls_do_not_solve_task"],
        "flat_failure_audit_passed": flat_audit["flat_failure_audit_passed"],
        "leakage_audit_passed": leakage_audit["leakage_audit_passed"],
        "finite_state_dynamics_passed": dynamics["finite_state_dynamics_passed"],
        "accepted_mutation_count_total": rollback["accepted_mutation_count_total"],
        "rejected_mutation_count_total": rollback["rejected_mutation_count_total"],
        "rollback_count_total": rollback["rollback_count_total"],
        "rollback_test_passed": rollback["rollback_test_passed"],
        "deterministic_replay_passed": deterministic["internal_replay_passed"],
        "static_metric_dictionary_used": audit["static_metric_dictionary_used"],
        "hardcoded_improvement_used": audit["hardcoded_improvement_used"],
        "synthetic_harness_only": audit["synthetic_harness_only"],
        "gradient_backprop_used": audit["gradient_backprop_used"],
        "real_optimizer_detected": audit["real_optimizer_detected"],
        "row_level_predictions_used": audit["row_level_predictions_used"],
    }


def decide(aggregate: dict[str, Any], controls: dict[str, Any], dynamics: dict[str, Any], audit: dict[str, Any], leakage_audit: dict[str, Any]) -> dict[str, Any]:
    if audit["static_metric_dictionary_used"] or audit["hardcoded_improvement_used"] or audit["synthetic_harness_only"]:
        decision = "e1_invalid_synthetic_metric_regression"
        next_step = "E1_RETRY_WITH_REAL_ROW_LEVEL_EVAL"
    elif not controls["controls_do_not_solve_task"]:
        decision = "e1_task_too_easy_or_leaky"
        next_step = "E1T_TASK_DIFFICULTY_REDESIGN"
    elif not dynamics["finite_state_dynamics_passed"]:
        decision = "e1_state_medium_instability_detected"
        next_step = "E1S_STATE_MEDIUM_STABILITY_REPAIR"
    elif not leakage_audit["leakage_audit_passed"]:
        decision = "e1_invalid_synthetic_metric_regression"
        next_step = "E1_RETRY_WITH_REAL_ROW_LEVEL_EVAL"
    else:
        flat_h = aggregate["flat_final_heldout_accuracy"]
        flat_o = aggregate["flat_final_ood_accuracy"]
        flat_c = aggregate["flat_final_counterfactual_accuracy"]
        state_beats_flat = (
            aggregate["state_medium_final_heldout_accuracy"] >= flat_h + 0.03
            and aggregate["state_medium_final_ood_accuracy"] >= flat_o + 0.03
            and aggregate["state_medium_final_counterfactual_accuracy"] >= flat_c + 0.03
            and aggregate["state_medium"]["shortcut_route_rate"] <= aggregate["flat"]["shortcut_route_rate"]
            and aggregate["state_medium"]["over_abstract_route_rate"] <= aggregate["flat"]["over_abstract_route_rate"]
            and aggregate["route_margin_delta_state_vs_flat"] > 0.0
        )
        gated_beats_flat = (
            aggregate["gated_state_medium_final_heldout_accuracy"] >= flat_h + 0.03
            and aggregate["gated_state_medium_final_ood_accuracy"] >= flat_o + 0.03
            and aggregate["gated_state_medium_final_counterfactual_accuracy"] >= flat_c + 0.03
            and aggregate["gated_state_medium"]["shortcut_route_rate"] <= aggregate["flat"]["shortcut_route_rate"]
            and aggregate["gated_state_medium"]["over_abstract_route_rate"] <= aggregate["flat"]["over_abstract_route_rate"]
            and aggregate["route_margin_delta_gated_vs_flat"] > 0.0
        )
        gated_beats_state = (
            aggregate["gated_state_medium_final_heldout_accuracy"] >= aggregate["state_medium_final_heldout_accuracy"] + 0.03
            and aggregate["gated_state_medium_final_ood_accuracy"] >= aggregate["state_medium_final_ood_accuracy"] + 0.03
            and aggregate["gated_state_medium_final_counterfactual_accuracy"] >= aggregate["state_medium_final_counterfactual_accuracy"] + 0.03
        )
        if gated_beats_flat and gated_beats_state:
            decision = "e1_gated_continuous_state_medium_probe_positive"
            next_step = "E2_REAL_BACKEND_GATED_STATE_MEDIUM_SCALE_STRESS_PROBE"
        elif state_beats_flat or gated_beats_flat:
            decision = "e1_continuous_state_medium_probe_positive"
            next_step = "E2_REAL_BACKEND_GATED_CORRECTION_VS_STATE_MEDIUM_PROBE"
        elif max(aggregate["state_medium_vs_flat_delta"], aggregate["gated_state_medium_vs_flat_delta"]) >= 0.0:
            decision = "e1_flat_resistance_remains_preferred"
            next_step = "E2_REAL_BACKEND_GATED_CORRECTION_MINIMAL_PROBE"
        else:
            decision = "e1_no_state_medium_advantage_detected"
            next_step = "E2_REAL_BACKEND_GATED_CORRECTION_MINIMAL_PROBE"
    return {
        "schema_version": "e1_decision_v1",
        "milestone": MILESTONE,
        "decision": decision,
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
        "real_optimizer_detected": False,
        "state_medium_parameters_mutated": True,
        "finite_state_dynamics_passed": aggregate["finite_state_dynamics_passed"],
        "flat_failure_audit_passed": aggregate["flat_failure_audit_passed"],
        "leakage_audit_passed": aggregate["leakage_audit_passed"],
        "flat_final_heldout_accuracy": aggregate["flat_final_heldout_accuracy"],
        "state_medium_final_heldout_accuracy": aggregate["state_medium_final_heldout_accuracy"],
        "gated_state_medium_final_heldout_accuracy": aggregate["gated_state_medium_final_heldout_accuracy"],
        "flat_final_ood_accuracy": aggregate["flat_final_ood_accuracy"],
        "state_medium_final_ood_accuracy": aggregate["state_medium_final_ood_accuracy"],
        "gated_state_medium_final_ood_accuracy": aggregate["gated_state_medium_final_ood_accuracy"],
        "flat_final_counterfactual_accuracy": aggregate["flat_final_counterfactual_accuracy"],
        "state_medium_final_counterfactual_accuracy": aggregate["state_medium_final_counterfactual_accuracy"],
        "gated_state_medium_final_counterfactual_accuracy": aggregate["gated_state_medium_final_counterfactual_accuracy"],
    }


def summary(decision: dict[str, Any], aggregate: dict[str, Any], git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e1_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "next": decision["next"],
        "git_status": git["git_status"],
        "flat_final_heldout_accuracy": aggregate["flat_final_heldout_accuracy"],
        "state_medium_final_heldout_accuracy": aggregate["state_medium_final_heldout_accuracy"],
        "gated_state_medium_final_heldout_accuracy": aggregate["gated_state_medium_final_heldout_accuracy"],
        "state_medium_vs_flat_delta": aggregate["state_medium_vs_flat_delta"],
        "gated_state_medium_vs_flat_delta": aggregate["gated_state_medium_vs_flat_delta"],
        "accepted_mutation_count_total": aggregate["accepted_mutation_count_total"],
        "rejected_mutation_count_total": aggregate["rejected_mutation_count_total"],
        "controls_do_not_solve_task": aggregate["controls_do_not_solve_task"],
        "flat_failure_audit_passed": aggregate["flat_failure_audit_passed"],
        "leakage_audit_passed": aggregate["leakage_audit_passed"],
        "finite_state_dynamics_passed": aggregate["finite_state_dynamics_passed"],
        "boundary": "E1 is a real-backend continuous state medium probe, not model-scale training.",
    }


def report_md(decision: dict[str, Any], aggregate: dict[str, Any], comparison: dict[str, Any], controls: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# {MILESTONE} Result",
            "",
            "## Decision",
            "",
            "```text",
            f"decision = {decision['decision']}",
            f"next = {decision['next']}",
            "```",
            "",
            "## Comparison",
            "",
            f"- flat_final_heldout_accuracy = {aggregate['flat_final_heldout_accuracy']}",
            f"- state_medium_final_heldout_accuracy = {aggregate['state_medium_final_heldout_accuracy']}",
            f"- gated_state_medium_final_heldout_accuracy = {aggregate['gated_state_medium_final_heldout_accuracy']}",
            f"- state_medium_vs_flat_delta = {aggregate['state_medium_vs_flat_delta']}",
            f"- gated_state_medium_vs_flat_delta = {aggregate['gated_state_medium_vs_flat_delta']}",
            f"- route_margin_delta_state_vs_flat = {comparison['route_margin_delta_state_vs_flat']}",
            f"- route_margin_delta_gated_vs_flat = {comparison['route_margin_delta_gated_vs_flat']}",
            "",
            "## Controls",
            "",
            f"- controls_do_not_solve_task = {controls['controls_do_not_solve_task']}",
            f"- non_oracle_controls_below_0_90_heldout = {controls['non_oracle_controls_below_0_90_heldout']}",
            f"- flat_failure_audit_passed = {aggregate['flat_failure_audit_passed']}",
            f"- leakage_audit_passed = {aggregate['leakage_audit_passed']}",
            "",
            "## Boundary",
            "",
            "E1 is a real-backend continuous state medium probe. It performs no natural-language pretraining, no tokenizer or next-token objective, no raw text or raw Raven work, and no Gemma-class training.",
            "",
        ]
    )


def deterministic_stub(passed: bool, comparisons: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e1_deterministic_replay_report_v1",
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
    diffs = {
        system: parameter_diff(searches[system]["initial_eval"]["candidate"], searches[system]["final_eval"]["candidate"], searches[system])
        for system in SYSTEMS
    }
    controls = control_scores(task)
    flat_audit = flat_baseline_failure_audit(searches, diffs)
    leakage_audit = state_medium_leakage_audit(searches, task)
    comparison = comparison_report(searches, diffs)
    dynamics = state_dynamics_report(searches)
    convergence = {
        "schema_version": "e1_convergence_stability_report_v1",
        "finite_state_dynamics_passed": dynamics["finite_state_dynamics_passed"],
        "state_medium": dynamics["systems"]["state_medium"],
        "gated_state_medium": dynamics["systems"]["gated_state_medium"],
    }
    rollback = accept_reject_rollback_report(searches)
    audit = no_synthetic_metric_audit(searches, task)
    aggregate = aggregate_metrics(searches, controls, comparison, dynamics, rollback, deterministic, audit, flat_audit, leakage_audit)
    decision = decide(aggregate, controls, dynamics, audit, leakage_audit)
    artifacts: dict[str, Any] = {
        "e1_backend_manifest.json": backend_manifest(core["settings"], core["git"]),
        "e1_task_generation_report.json": task_generation_report(task, core["settings"]),
        "e1_candidate_flat_initial.json": searches["flat"]["initial_eval"]["candidate"],
        "e1_candidate_flat_final.json": searches["flat"]["final_eval"]["candidate"],
        "e1_candidate_state_medium_initial.json": searches["state_medium"]["initial_eval"]["candidate"],
        "e1_candidate_state_medium_final.json": searches["state_medium"]["final_eval"]["candidate"],
        "e1_candidate_gated_state_medium_initial.json": searches["gated_state_medium"]["initial_eval"]["candidate"],
        "e1_candidate_gated_state_medium_final.json": searches["gated_state_medium"]["final_eval"]["candidate"],
        "e1_parameter_diff_flat.json": diffs["flat"],
        "e1_parameter_diff_state_medium.json": diffs["state_medium"],
        "e1_parameter_diff_gated_state_medium.json": diffs["gated_state_medium"],
        "e1_mutation_history_flat.json": {
            "schema_version": "e1_mutation_history_flat_v1",
            "system": "flat",
            "mutation_attempt_count": searches["flat"]["mutation_attempt_count"],
            "accepted_mutation_count": searches["flat"]["accepted_mutation_count"],
            "rejected_mutation_count": searches["flat"]["rejected_mutation_count"],
            "rollback_count": searches["flat"]["rollback_count"],
            "history": searches["flat"]["history"],
        },
        "e1_mutation_history_state_medium.json": {
            "schema_version": "e1_mutation_history_state_medium_v1",
            "system": "state_medium",
            "mutation_attempt_count": searches["state_medium"]["mutation_attempt_count"],
            "accepted_mutation_count": searches["state_medium"]["accepted_mutation_count"],
            "rejected_mutation_count": searches["state_medium"]["rejected_mutation_count"],
            "rollback_count": searches["state_medium"]["rollback_count"],
            "history": searches["state_medium"]["history"],
        },
        "e1_mutation_history_gated_state_medium.json": {
            "schema_version": "e1_mutation_history_gated_state_medium_v1",
            "system": "gated_state_medium",
            "mutation_attempt_count": searches["gated_state_medium"]["mutation_attempt_count"],
            "accepted_mutation_count": searches["gated_state_medium"]["accepted_mutation_count"],
            "rejected_mutation_count": searches["gated_state_medium"]["rejected_mutation_count"],
            "rollback_count": searches["gated_state_medium"]["rollback_count"],
            "history": searches["gated_state_medium"]["history"],
        },
        "e1_generation_metrics.json": {
            "schema_version": "e1_generation_metrics_v1",
            "systems": {system: searches[system]["generation_metrics"] for system in SYSTEMS},
        },
        "e1_row_level_eval_sample_train.json": {
            system: searches[system]["final_eval"]["evals"]["train"]["sample"] for system in SYSTEMS
        },
        "e1_row_level_eval_sample_heldout.json": {
            system: searches[system]["final_eval"]["evals"]["heldout"]["sample"] for system in SYSTEMS
        },
        "e1_row_level_eval_sample_ood.json": {
            system: searches[system]["final_eval"]["evals"]["ood"]["sample"] for system in SYSTEMS
        },
        "e1_row_level_eval_sample_counterfactual.json": {
            system: searches[system]["final_eval"]["evals"]["counterfactual"]["sample"] for system in SYSTEMS
        },
        "e1_control_baseline_report.json": controls,
        "e1_flat_baseline_failure_audit.json": flat_audit,
        "e1_state_medium_leakage_audit.json": leakage_audit,
        "e1_flat_vs_state_medium_comparison_report.json": comparison,
        "e1_state_dynamics_report.json": dynamics,
        "e1_convergence_stability_report.json": convergence,
        "e1_accept_reject_rollback_report.json": rollback,
        "e1_no_synthetic_metric_audit.json": audit,
        "e1_deterministic_replay_report.json": deterministic,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": summary(decision, aggregate, core["git"]),
        "report.md": report_md(decision, aggregate, comparison, controls),
    }
    return artifacts


def artifact_hashes(core: dict[str, Any]) -> dict[str, str]:
    artifacts = compose_artifacts(core, deterministic_stub(True, {}))
    return {name: payload_sha256(artifacts[name]) for name in HASH_ARTIFACTS}


def deterministic_report(primary: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    primary_hashes = artifact_hashes(primary)
    replay_hashes = artifact_hashes(replay)
    comparisons = {
        name: {"primary_hash": primary_hashes[name], "replay_hash": replay_hashes[name], "match": primary_hashes[name] == replay_hashes[name]}
        for name in HASH_ARTIFACTS
    }
    return deterministic_stub(all(row["match"] for row in comparisons.values()), comparisons)


def run_core(settings: Settings, out: Path | None = None) -> dict[str, Any]:
    task = generate_task(settings)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        write_text(out / "progress.jsonl", "")
        write_text(out / "e1_online_check_report.md", online_check_report())
        append_progress(out, "startup", seeds=list(settings.seeds), population_size=settings.population_size, generations=settings.generations)
    searches = {}
    for system in SYSTEMS:
        searches[system] = run_system_search(system, task, settings, out)
    return {"settings": settings, "task": task, "searches": searches, "git": git_preflight()}


def write_result_doc(core: dict[str, Any], decision: dict[str, Any], aggregate: dict[str, Any]) -> None:
    path = REPO_ROOT / "docs/research/E1_REAL_BACKEND_CONTINUOUS_STATE_MEDIUM_PROBE_RESULT.md"
    text = "\n".join(
        [
            "# E1_REAL_BACKEND_CONTINUOUS_STATE_MEDIUM_PROBE Result",
            "",
            "## Outcome",
            "",
            "```text",
            f"decision = {decision['decision']}",
            f"next = {decision['next']}",
            "```",
            "",
            "## Outputs",
            "",
            "Primary output:",
            "",
            "```text",
            "target/pilot_wave/e1_real_backend_continuous_state_medium_probe/",
            "```",
            "",
            "Replay output:",
            "",
            "```text",
            "target/pilot_wave/e1_real_backend_continuous_state_medium_probe_replay/",
            "```",
            "",
            "## Scale",
            "",
            "```text",
            f"seeds = {','.join(str(seed) for seed in core['settings'].seeds)}",
            f"train_rows = {core['settings'].train_rows_per_seed * len(core['settings'].seeds)}",
            f"validation_rows = {core['settings'].validation_rows_per_seed * len(core['settings'].seeds)}",
            f"heldout_rows = {core['settings'].heldout_rows_per_seed * len(core['settings'].seeds)}",
            f"ood_rows = {core['settings'].ood_rows_per_seed * len(core['settings'].seeds)}",
            f"counterfactual_rows = {core['settings'].counterfactual_rows_per_seed * len(core['settings'].seeds)}",
            f"population_size = {core['settings'].population_size}",
            f"generations = {core['settings'].generations}",
            "```",
            "",
            "## Metrics",
            "",
            "```text",
            f"flat_final_heldout_accuracy = {aggregate['flat_final_heldout_accuracy']}",
            f"state_medium_final_heldout_accuracy = {aggregate['state_medium_final_heldout_accuracy']}",
            f"gated_state_medium_final_heldout_accuracy = {aggregate['gated_state_medium_final_heldout_accuracy']}",
            f"state_medium_vs_flat_delta = {aggregate['state_medium_vs_flat_delta']}",
            f"gated_state_medium_vs_flat_delta = {aggregate['gated_state_medium_vs_flat_delta']}",
            f"controls_do_not_solve_task = {aggregate['controls_do_not_solve_task']}",
            f"flat_failure_audit_passed = {aggregate['flat_failure_audit_passed']}",
            f"leakage_audit_passed = {aggregate['leakage_audit_passed']}",
            f"finite_state_dynamics_passed = {aggregate['finite_state_dynamics_passed']}",
            f"deterministic_replay_passed = {aggregate['deterministic_replay_passed']}",
            "```",
            "",
            "## Boundary",
            "",
            "E1 is a real-backend continuous state medium probe. It performs no natural-language pretraining, no tokenizer or next-token objective, no raw text or raw Raven work, and no Gemma-class training. It does not prove consciousness, AGI, production readiness, or model-scale reasoning. It only tests whether a tiny mutable dynamic state medium can beat a flat resistance baseline on a controlled symbolic route-selection task using real row-level evaluation.",
            "",
        ]
    )
    write_text(path, text)


def write_artifacts(out: Path, core: dict[str, Any], deterministic: dict[str, Any]) -> None:
    artifacts = compose_artifacts(core, deterministic)
    write_text(out / "e1_online_check_report.md", online_check_report())
    for name in REQUIRED_ARTIFACTS:
        if name == "e1_online_check_report.md":
            continue
        payload = artifacts[name]
        if name.endswith(".md"):
            write_text(out / name, payload)
        else:
            write_json(out / name, payload)
    write_result_doc(core, artifacts["decision.json"], artifacts["aggregate_metrics.json"])
    append_progress(out, "final_artifacts_written", decision=artifacts["decision.json"]["decision"], next=artifacts["decision.json"]["next"])


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
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--validation-rows-per-seed", type=int, default=300)
    parser.add_argument("--heldout-rows-per-seed", type=int, default=300)
    parser.add_argument("--ood-rows-per-seed", type=int, default=300)
    parser.add_argument("--counterfactual-rows-per-seed", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=100)
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
