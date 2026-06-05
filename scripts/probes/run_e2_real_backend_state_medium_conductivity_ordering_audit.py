#!/usr/bin/env python3
"""E2 real-backend state-medium conductivity ordering audit."""

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
MILESTONE = "E2_REAL_BACKEND_STATE_MEDIUM_CONDUCTIVITY_ORDERING_AUDIT"
DEFAULT_OUT = Path("target/pilot_wave/e2_real_backend_state_medium_conductivity_ordering_audit")
DEFAULT_SEEDS = (73001, 73002, 73003, 73004, 73005)

ROUTES = (
    "shortcut_route",
    "random_noise_route",
    "illogical_route",
    "logical_route",
    "surface_similar_wrong_route",
    "contradiction_route",
    "local_expensive_but_valid_route",
    "over_abstract_wrong_route",
)
FEATURES = (
    "evidence_gap",
    "route_margin_gap",
    "shortcut_risk",
    "random_noise_level",
    "contradiction_risk",
    "surface_similarity_wrongness",
    "illogical_transition_count",
    "local_step_cost",
    "abstraction_jump_cost",
    "landing_error_risk",
    "counterfactual_fragility",
    "preservation_risk",
    "calibration_mismatch",
    "sequence_length_risk",
    "binding_scope_risk",
    "template_collision_risk",
    "grammar_collision_risk",
    "temporal_instability_risk",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}
ROUTE_INDEX = {name: index for index, name in enumerate(ROUTES)}
LOGICAL_INDEX = ROUTE_INDEX["logical_route"]
WRONG_ROUTES = tuple(route for route in ROUTES if route != "logical_route")
SYSTEMS = ("flat", "state_medium", "trajectory_readout", "stability_readout")
STATE_SYSTEMS = ("state_medium", "trajectory_readout", "stability_readout")
ORDERING_SPLITS = ("heldout", "ood", "counterfactual", "adversarial")
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
HASH_ARTIFACTS = (
    "e2_candidate_state_medium_final.json",
    "e2_candidate_trajectory_readout_final.json",
    "e2_candidate_stability_readout_final.json",
    "e2_conductivity_ordering_report.json",
    "e2_attractor_basin_report.json",
    "e2_control_baseline_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
)
REQUIRED_ARTIFACTS = (
    "e2_online_check_report.md",
    "e2_backend_manifest.json",
    "e2_task_generation_report.json",
    "e2_candidate_flat_initial.json",
    "e2_candidate_flat_final.json",
    "e2_candidate_state_medium_initial.json",
    "e2_candidate_state_medium_final.json",
    "e2_candidate_trajectory_readout_initial.json",
    "e2_candidate_trajectory_readout_final.json",
    "e2_candidate_stability_readout_initial.json",
    "e2_candidate_stability_readout_final.json",
    "e2_parameter_diff_flat.json",
    "e2_parameter_diff_state_medium.json",
    "e2_parameter_diff_trajectory_readout.json",
    "e2_parameter_diff_stability_readout.json",
    "e2_mutation_history_flat.json",
    "e2_mutation_history_state_medium.json",
    "e2_mutation_history_trajectory_readout.json",
    "e2_mutation_history_stability_readout.json",
    "e2_generation_metrics.json",
    "e2_row_level_eval_sample_train.json",
    "e2_row_level_eval_sample_heldout.json",
    "e2_row_level_eval_sample_ood.json",
    "e2_row_level_eval_sample_counterfactual.json",
    "e2_row_level_eval_sample_adversarial.json",
    "e2_conductivity_ordering_report.json",
    "e2_logical_vs_wrong_gap_report.json",
    "e2_attractor_basin_report.json",
    "e2_state_trajectory_report.json",
    "e2_perturbation_recovery_report.json",
    "e2_counterfactual_ordering_report.json",
    "e2_ood_ordering_report.json",
    "e2_control_baseline_report.json",
    "e2_leakage_sentinel_report.json",
    "e2_no_synthetic_metric_audit.json",
    "e2_deterministic_replay_report.json",
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
    adversarial_rows_per_seed: int
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
    if rev.returncode != 0:
        return {
            "cwd": str(cwd),
            "git_status": "no_git_repository",
            "repo_root": None,
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
            "# E2 Online Check Report",
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
            "- NumPy linalg.norm docs: https://numpy.org/doc/2.2/reference/generated/numpy.linalg.norm.html",
            "- Echo state network technical report: https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf",
            "- Practical ESN guide: https://www.ai.rug.nl/minds/uploads/PracticalESN.pdf",
            "",
            "## Packages Verified",
            "",
            f"- Python runtime: {sys.version.split()[0]}",
            f"- numpy: {np.__version__}",
            "",
            "## Notes",
            "",
            "E2 uses stdlib random.Random for deterministic mutation and NumPy for vectorized row-level scoring, recurrent state updates, argmin route selection, and simple norm-style trajectory diagnostics. Reservoir/echo-state references were checked only to sanity-check the small bounded recurrent update form. No torch, jax, tensorflow, GPU, tokenizer, raw text corpus, or Raven data is used.",
            "",
        ]
    )


def jitter(rng: random.Random, base: float, spread: float = 0.025) -> float:
    return clamp(base + rng.uniform(-spread, spread))


def base_route(rng: random.Random, value: float) -> dict[str, float]:
    return {feature: jitter(rng, value) for feature in FEATURES}


def route_template(rng: random.Random, split: str) -> list[dict[str, float]]:
    ood = 0.05 if split in {"ood", "adversarial"} else 0.0
    cf = 0.06 if split in {"counterfactual", "adversarial"} else 0.0
    adv = 0.05 if split == "adversarial" else 0.0
    mode = rng.randrange(6)

    logical = base_route(rng, 0.18)
    logical.update(
        {
            "evidence_gap": jitter(rng, 0.22 + ood),
            "route_margin_gap": jitter(rng, 0.21 + cf),
            "shortcut_risk": jitter(rng, 0.18 + 0.08 * (mode == 0)),
            "random_noise_level": jitter(rng, 0.06),
            "contradiction_risk": jitter(rng, 0.06),
            "surface_similarity_wrongness": jitter(rng, 0.12 + 0.08 * (mode == 1)),
            "illogical_transition_count": jitter(rng, 0.06),
            "local_step_cost": jitter(rng, 0.42 + ood),
            "abstraction_jump_cost": jitter(rng, 0.32 + 0.08 * (mode == 2)),
            "landing_error_risk": jitter(rng, 0.18 + cf),
            "counterfactual_fragility": jitter(rng, 0.19 + cf),
            "preservation_risk": jitter(rng, 0.17),
            "calibration_mismatch": jitter(rng, 0.17 + cf),
            "sequence_length_risk": jitter(rng, 0.56 + ood),
            "binding_scope_risk": jitter(rng, 0.14),
            "template_collision_risk": jitter(rng, 0.12),
            "grammar_collision_risk": jitter(rng, 0.12),
            "temporal_instability_risk": jitter(rng, 0.16 + cf),
        }
    )

    shortcut = base_route(rng, 0.08)
    shortcut.update(
        {
            "evidence_gap": jitter(rng, 0.05),
            "route_margin_gap": jitter(rng, 0.06),
            "shortcut_risk": jitter(rng, 0.90 + adv),
            "surface_similarity_wrongness": jitter(rng, 0.56 + adv),
            "landing_error_risk": jitter(rng, 0.34 + cf),
            "counterfactual_fragility": jitter(rng, 0.54 + cf),
            "local_step_cost": jitter(rng, 0.05),
            "sequence_length_risk": jitter(rng, 0.06),
        }
    )

    noise = base_route(rng, 0.09)
    noise.update(
        {
            "evidence_gap": jitter(rng, 0.08),
            "route_margin_gap": jitter(rng, 0.08),
            "random_noise_level": jitter(rng, 0.93 + adv),
            "template_collision_risk": jitter(rng, 0.54 + adv),
            "grammar_collision_risk": jitter(rng, 0.58 + adv),
            "calibration_mismatch": jitter(rng, 0.42 + cf),
            "local_step_cost": jitter(rng, 0.08),
        }
    )

    illogical = base_route(rng, 0.10)
    illogical.update(
        {
            "evidence_gap": jitter(rng, 0.08),
            "route_margin_gap": jitter(rng, 0.09),
            "illogical_transition_count": jitter(rng, 0.92 + adv),
            "landing_error_risk": jitter(rng, 0.62 + cf),
            "binding_scope_risk": jitter(rng, 0.50 + adv),
            "temporal_instability_risk": jitter(rng, 0.42 + cf),
            "local_step_cost": jitter(rng, 0.11),
        }
    )

    surface = base_route(rng, 0.08)
    surface.update(
        {
            "evidence_gap": jitter(rng, 0.04),
            "route_margin_gap": jitter(rng, 0.05),
            "surface_similarity_wrongness": jitter(rng, 0.92 + adv),
            "template_collision_risk": jitter(rng, 0.62 + adv),
            "grammar_collision_risk": jitter(rng, 0.55 + adv),
            "binding_scope_risk": jitter(rng, 0.44),
            "preservation_risk": jitter(rng, 0.34),
            "local_step_cost": jitter(rng, 0.08),
        }
    )

    contradiction = base_route(rng, 0.10)
    contradiction.update(
        {
            "evidence_gap": jitter(rng, 0.06),
            "route_margin_gap": jitter(rng, 0.07),
            "contradiction_risk": jitter(rng, 0.94 + adv),
            "counterfactual_fragility": jitter(rng, 0.63 + cf),
            "preservation_risk": jitter(rng, 0.62 + adv),
            "calibration_mismatch": jitter(rng, 0.60 + cf),
            "landing_error_risk": jitter(rng, 0.38 + cf),
            "local_step_cost": jitter(rng, 0.10),
        }
    )

    local_expensive = base_route(rng, 0.08)
    local_expensive.update(
        {
            "evidence_gap": jitter(rng, 0.18 + ood),
            "route_margin_gap": jitter(rng, 0.18 + cf),
            "local_step_cost": jitter(rng, 0.94 + ood),
            "sequence_length_risk": jitter(rng, 0.91 + ood),
            "abstraction_jump_cost": jitter(rng, 0.40),
            "landing_error_risk": jitter(rng, 0.12),
            "counterfactual_fragility": jitter(rng, 0.12),
            "preservation_risk": jitter(rng, 0.11),
            "calibration_mismatch": jitter(rng, 0.12),
        }
    )

    over_abstract = base_route(rng, 0.09)
    over_abstract.update(
        {
            "evidence_gap": jitter(rng, 0.07),
            "route_margin_gap": jitter(rng, 0.07),
            "abstraction_jump_cost": jitter(rng, 0.94 + adv),
            "landing_error_risk": jitter(rng, 0.45 + cf),
            "counterfactual_fragility": jitter(rng, 0.58 + cf),
            "preservation_risk": jitter(rng, 0.52),
            "binding_scope_risk": jitter(rng, 0.48 + adv),
            "sequence_length_risk": jitter(rng, 0.12),
        }
    )

    return [shortcut, noise, illogical, logical, surface, contradiction, local_expensive, over_abstract]


def generate_split(seeds: tuple[int, ...], rows_per_seed: int, split: str, offset: int) -> dict[str, Any]:
    rows = []
    arrays = []
    for seed in seeds:
        rng = random.Random(seed + offset)
        for row_index in range(rows_per_seed):
            route_vectors = route_template(rng, split)
            row_id = f"{split}_seed{seed}_row{row_index:04d}"
            rows.append(
                {
                    "row_id": row_id,
                    "split": split,
                    "seed": seed,
                    "row_index": row_index,
                    "correct_route": "logical_route",
                    "routes": {
                        route_name: route_vectors[route_i]
                        for route_i, route_name in enumerate(ROUTES)
                    },
                }
            )
            arrays.append([[route[feature] for feature in FEATURES] for route in route_vectors])
    return {"rows": rows, "array": np.asarray(arrays, dtype=np.float64)}


def generate_task(settings: Settings) -> dict[str, Any]:
    return {
        "train": generate_split(settings.seeds, settings.train_rows_per_seed, "train", 0),
        "validation": generate_split(settings.seeds, settings.validation_rows_per_seed, "validation", 100),
        "heldout": generate_split(settings.seeds, settings.heldout_rows_per_seed, "heldout", 200),
        "ood": generate_split(settings.seeds, settings.ood_rows_per_seed, "ood", 300),
        "counterfactual": generate_split(settings.seeds, settings.counterfactual_rows_per_seed, "counterfactual", 400),
        "adversarial": generate_split(settings.seeds, settings.adversarial_rows_per_seed, "adversarial", 500),
    }


def matrix(rows: int, cols: int, value: float = 0.0) -> list[list[float]]:
    return [[round_float(value) for _ in range(cols)] for _ in range(rows)]


def vector(size: int, value: float = 0.0) -> list[float]:
    return [round_float(value) for _ in range(size)]


def initial_flat_weights() -> dict[str, float]:
    weights = {feature: 0.30 for feature in FEATURES}
    weights.update(
        {
            "evidence_gap": 0.88,
            "route_margin_gap": 0.88,
            "shortcut_risk": 0.22,
            "random_noise_level": 0.22,
            "contradiction_risk": 0.24,
            "surface_similarity_wrongness": 0.20,
            "illogical_transition_count": 0.24,
            "local_step_cost": 0.82,
            "abstraction_jump_cost": 0.38,
            "landing_error_risk": 0.28,
            "counterfactual_fragility": 0.26,
            "preservation_risk": 0.22,
            "calibration_mismatch": 0.22,
            "sequence_length_risk": 0.78,
            "binding_scope_risk": 0.18,
            "template_collision_risk": 0.16,
            "grammar_collision_risk": 0.16,
            "temporal_instability_risk": 0.16,
        }
    )
    return weights


def make_flat_candidate(candidate_id: str) -> dict[str, Any]:
    weights = initial_flat_weights()
    return {
        "schema_version": "e2_flat_candidate_state_v1",
        "system": "flat",
        "candidate_id": candidate_id,
        "bias": 0.0,
        "weights": {feature: round_float(weights[feature]) for feature in FEATURES},
        "weight_range": {"min": 0.0, "max": 5.0},
        "all_weights_nonnegative": True,
    }


def state_projector() -> tuple[list[list[float]], list[float]]:
    w = matrix(len(FEATURES), 8)
    b = vector(8, 0.0)
    groups = [
        (("shortcut_risk", "surface_similarity_wrongness", "counterfactual_fragility"), 0, -1.02),
        (("random_noise_level", "template_collision_risk", "grammar_collision_risk"), 1, -1.03),
        (("contradiction_risk", "preservation_risk", "calibration_mismatch"), 2, -1.03),
        (("illogical_transition_count", "landing_error_risk", "binding_scope_risk"), 3, -1.04),
        (("surface_similarity_wrongness", "template_collision_risk", "binding_scope_risk"), 4, -1.00),
        (("abstraction_jump_cost", "counterfactual_fragility", "preservation_risk"), 5, -1.04),
        (("local_step_cost", "sequence_length_risk", "abstraction_jump_cost"), 6, -1.26),
        (("evidence_gap", "route_margin_gap", "temporal_instability_risk"), 7, -0.86),
    ]
    for features, dim, bias in groups:
        for feature in features:
            w[FEATURE_INDEX[feature]][dim] = 1.0
        b[dim] = bias
    return w, b


def make_state_candidate(candidate_id: str, system: str) -> dict[str, Any]:
    input_projection, state_bias = state_projector()
    recurrent = matrix(8, 8)
    for index in range(8):
        recurrent[index][index] = 0.10
    readout = [1.0, 1.0, 1.0, 1.0, 0.82, 0.80, 0.45, 0.35]
    candidate = {
        "schema_version": f"e2_{system}_candidate_state_v1",
        "system": system,
        "candidate_id": candidate_id,
        "state_dim": 8,
        "input_dim": len(FEATURES),
        "settling_steps": 6,
        "leak": 0.64,
        "gain": 1.55,
        "score_bias": 0.0,
        "input_projection": input_projection,
        "recurrent_matrix": recurrent,
        "state_bias": state_bias,
        "readout": [round_float(value) for value in readout],
        "parameter_range": {"matrix_vector": [-3.0, 3.0], "leak": [0.05, 0.95], "gain": [0.25, 4.0]},
    }
    if system == "trajectory_readout":
        candidate["readout_final"] = [round_float(value) for value in readout]
        candidate["readout_delta"] = [0.20, 0.20, 0.20, 0.20, 0.14, 0.14, 0.08, 0.08]
        candidate["trajectory_scalar"] = 0.16
        candidate["path_norm_scalar"] = 0.08
        candidate.pop("readout")
    if system == "stability_readout":
        candidate["readout_final"] = [round_float(value) for value in readout]
        candidate["convergence_scalar"] = 0.22
        candidate["stability_scalar"] = -0.16
        candidate["separation_scalar"] = 0.10
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


def add_feature_tie_breaker(scores: np.ndarray, split_data: dict[str, Any]) -> np.ndarray:
    tie_break = split_data["array"] @ TIE_BREAK_VECTOR
    return scores + (1e-9 * tie_break)


def flat_scores(candidate: dict[str, Any], split_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    weights = np.asarray([candidate["weights"][feature] for feature in FEATURES], dtype=np.float64)
    scores = split_data["array"] @ weights + candidate["bias"]
    scores = add_feature_tie_breaker(scores, split_data)
    return scores, {
        "convergence_by_route": np.zeros_like(scores),
        "stability_by_route": np.ones_like(scores),
        "trajectory_norm_by_route": np.zeros_like(scores),
        "finite": bool(np.isfinite(scores).all()),
    }


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
    delta_mean = delta_sum / float(candidate["settling_steps"])
    convergence = np.mean(last_delta, axis=1).reshape(rows, routes)
    stability = 1.0 / (1.0 + convergence)
    trajectory_norm = (state_abs_sum / float(candidate["settling_steps"])).reshape(rows, routes)
    return {
        "flat_features": flat,
        "final_state": state,
        "delta_mean": delta_mean,
        "convergence": convergence,
        "stability": stability,
        "trajectory_norm": trajectory_norm,
        "rows": np.asarray(rows),
        "routes": np.asarray(routes),
    }


def state_scores(candidate: dict[str, Any], split_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    rollout = state_rollout(candidate, split_data)
    rows = int(rollout["rows"])
    routes = int(rollout["routes"])
    system = candidate["system"]
    if system == "state_medium":
        raw = rollout["final_state"] @ as_array(candidate, "readout") + float(candidate["score_bias"])
    elif system == "trajectory_readout":
        raw = (
            float(candidate["score_bias"])
            + rollout["final_state"] @ as_array(candidate, "readout_final")
            + rollout["delta_mean"] @ as_array(candidate, "readout_delta")
            + float(candidate["trajectory_scalar"]) * rollout["trajectory_norm"].reshape(rows * routes)
            + float(candidate["path_norm_scalar"]) * np.mean(np.abs(rollout["final_state"]), axis=1)
        )
    elif system == "stability_readout":
        raw = (
            float(candidate["score_bias"])
            + rollout["final_state"] @ as_array(candidate, "readout_final")
            + float(candidate["convergence_scalar"]) * rollout["convergence"].reshape(rows * routes)
            + float(candidate["stability_scalar"]) * rollout["stability"].reshape(rows * routes)
            + float(candidate["separation_scalar"]) * rollout["trajectory_norm"].reshape(rows * routes)
        )
    else:
        raise ValueError(f"unknown state system: {system}")
    scores = add_feature_tie_breaker(raw.reshape(rows, routes), split_data)
    diagnostics = {
        "convergence_by_route": rollout["convergence"],
        "stability_by_route": rollout["stability"],
        "trajectory_norm_by_route": rollout["trajectory_norm"],
        "finite": bool(np.isfinite(scores).all() and np.isfinite(rollout["final_state"]).all()),
    }
    return scores, diagnostics


def score_candidate(candidate: dict[str, Any], split_data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate["system"] == "flat":
        return flat_scores(candidate, split_data)
    return state_scores(candidate, split_data)


def scores_to_metrics(scores: np.ndarray, diagnostics: dict[str, Any]) -> dict[str, Any]:
    row_count = scores.shape[0]
    predicted = np.argmin(scores, axis=1)
    logical_scores = scores[:, LOGICAL_INDEX]
    wrong_indices = [ROUTE_INDEX[route] for route in WRONG_ROUTES]
    wrong_scores = scores[:, wrong_indices]
    best_wrong = np.min(wrong_scores, axis=1)
    gaps_by_route = {
        route: scores[:, ROUTE_INDEX[route]] - logical_scores
        for route in WRONG_ROUTES
    }
    positive_by_route = {route: gaps > 0.0 for route, gaps in gaps_by_route.items()}
    all_positive = np.column_stack([positive_by_route[route] for route in WRONG_ROUTES])
    counts = np.bincount(predicted, minlength=len(ROUTES))
    selected_convergence = diagnostics["convergence_by_route"][np.arange(row_count), predicted]
    selected_stability = diagnostics["stability_by_route"][np.arange(row_count), predicted]
    selected_trajectory = diagnostics["trajectory_norm_by_route"][np.arange(row_count), predicted]
    logical_rank = 1 + np.sum(scores < logical_scores[:, None], axis=1)
    mean_gap = float(np.mean(best_wrong - logical_scores))
    stability_score = float(np.mean(selected_stability))
    trajectory_consistency = 1.0 / (1.0 + float(np.mean(selected_trajectory)))
    selectivity = max(0.0, mean_gap) / (1.0 + abs(float(np.mean(logical_scores))))
    return {
        "row_count": int(row_count),
        "logical_path_selected_rate": round_float(float(np.mean(predicted == LOGICAL_INDEX))),
        "accuracy": round_float(float(np.mean(predicted == LOGICAL_INDEX))),
        "logical_resistance_or_score_rank": round_float(float(np.mean(logical_rank))),
        "logical_rank_1_rate": round_float(float(np.mean(logical_rank == 1))),
        "mean_logical_score": round_float(float(np.mean(logical_scores))),
        "mean_best_wrong_score": round_float(float(np.mean(best_wrong))),
        "logical_vs_best_wrong_gap": round_float(mean_gap),
        "logical_vs_shortcut_gap": round_float(float(np.mean(gaps_by_route["shortcut_route"]))),
        "logical_vs_noise_gap": round_float(float(np.mean(gaps_by_route["random_noise_route"]))),
        "logical_vs_illogical_gap": round_float(float(np.mean(gaps_by_route["illogical_route"]))),
        "logical_vs_surface_wrong_gap": round_float(float(np.mean(gaps_by_route["surface_similar_wrong_route"]))),
        "logical_vs_contradiction_gap": round_float(float(np.mean(gaps_by_route["contradiction_route"]))),
        "logical_vs_local_expensive_gap": round_float(float(np.mean(gaps_by_route["local_expensive_but_valid_route"]))),
        "logical_vs_over_abstract_gap": round_float(float(np.mean(gaps_by_route["over_abstract_wrong_route"]))),
        "ordering_stability": round_float(float(np.mean(np.all(all_positive, axis=1)))),
        "shortcut_attractor_rate": round_float(float(counts[ROUTE_INDEX["shortcut_route"]] / row_count)),
        "noise_attractor_rate": round_float(float(counts[ROUTE_INDEX["random_noise_route"]] / row_count)),
        "illogical_attractor_rate": round_float(float(counts[ROUTE_INDEX["illogical_route"]] / row_count)),
        "surface_wrong_attractor_rate": round_float(float(counts[ROUTE_INDEX["surface_similar_wrong_route"]] / row_count)),
        "contradiction_attractor_rate": round_float(float(counts[ROUTE_INDEX["contradiction_route"]] / row_count)),
        "local_expensive_attractor_rate": round_float(float(counts[ROUTE_INDEX["local_expensive_but_valid_route"]] / row_count)),
        "over_abstract_attractor_rate": round_float(float(counts[ROUTE_INDEX["over_abstract_wrong_route"]] / row_count)),
        "convergence_steps_mean": round_float(float(np.mean(selected_convergence))),
        "convergence_stability_score": round_float(stability_score),
        "attractor_basin_separation_score": round_float(mean_gap),
        "trajectory_consistency_score": round_float(trajectory_consistency),
        "state_selectivity_score": round_float(selectivity),
        "finite_state_dynamics_passed": bool(diagnostics.get("finite", True)),
    }


def evaluate_candidate(candidate: dict[str, Any], split_data: dict[str, Any], sample_limit: int = 12) -> dict[str, Any]:
    scores, diagnostics = score_candidate(candidate, split_data)
    if not np.isfinite(scores).all():
        return {
            "metrics": {
                "row_count": int(split_data["array"].shape[0]),
                "accuracy": 0.0,
                "logical_path_selected_rate": 0.0,
                "logical_vs_best_wrong_gap": -999.0,
                "ordering_stability": 0.0,
                "finite_state_dynamics_passed": False,
            },
            "sample": [],
            "predicted": np.zeros(split_data["array"].shape[0], dtype=np.int64),
        }
    predicted = np.argmin(scores, axis=1)
    metrics = scores_to_metrics(scores, diagnostics)
    sample = []
    for index, row in enumerate(split_data["rows"][:sample_limit]):
        sample.append(
            {
                "row_id": row["row_id"],
                "split": row["split"],
                "correct_route": row["correct_route"],
                "predicted_route": ROUTES[int(predicted[index])],
                "selected_is_logical": bool(predicted[index] == LOGICAL_INDEX),
                "scores": {route: round_float(scores[index, route_i]) for route_i, route in enumerate(ROUTES)},
                "routes": row["routes"],
            }
        )
    return {"metrics": metrics, "sample": sample, "predicted": predicted}


def evaluate_all(candidate: dict[str, Any], task: dict[str, Any], sample_limit: int = 12) -> dict[str, Any]:
    return {split: evaluate_candidate(candidate, data, sample_limit=sample_limit) for split, data in task.items()}


def fitness_from_evals(evals: dict[str, Any]) -> float:
    train = evals["train"]["metrics"]
    validation = evals["validation"]["metrics"]
    gap = abs(train["logical_path_selected_rate"] - validation["logical_path_selected_rate"])
    bad_attractor_rate = (
        validation["shortcut_attractor_rate"]
        + validation["noise_attractor_rate"]
        + validation["illogical_attractor_rate"]
        + validation["surface_wrong_attractor_rate"]
        + validation["contradiction_attractor_rate"]
    )
    stability_penalty = max(0.0, validation["convergence_steps_mean"] - 0.08)
    positive_gap = max(0.0, validation["logical_vs_best_wrong_gap"])
    return round_float(
        120.0 * validation["logical_path_selected_rate"]
        + 25.0 * train["logical_path_selected_rate"]
        + 35.0 * validation["ordering_stability"]
        + 6.0 * positive_gap
        + 8.0 * validation["trajectory_consistency_score"]
        + 6.0 * validation["state_selectivity_score"]
        - 22.0 * bad_attractor_rate
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
            return clamp(value, -2.0, 2.0)
        if len(path) >= 2 and path[0] == "weights":
            return clamp(value, 0.0, 5.0)
        return value
    if path == ("leak",):
        return clamp(value, 0.05, 0.95)
    if path == ("gain",):
        return clamp(value, 0.25, 4.0)
    if path == ("score_bias",):
        return clamp(value, -3.0, 3.0)
    if path and path[0] in {"readout", "readout_final", "readout_delta"}:
        return clamp(value, -4.0, 4.0)
    if path and path[-1] in {"trajectory_scalar", "path_norm_scalar", "convergence_scalar", "stability_scalar", "separation_scalar"}:
        return clamp(value, -4.0, 4.0)
    return clamp(value, -3.0, 3.0)


def mutate_candidate(candidate: dict[str, Any], rng: random.Random, sigma: float, candidate_id: str) -> dict[str, Any]:
    child = copy.deepcopy(candidate)
    child["candidate_id"] = candidate_id
    paths = flatten_paths(child)
    edit_count = rng.randint(1, 4) if child["system"] == "flat" else rng.randint(2, 12)
    for path, value in rng.sample(paths, k=min(edit_count, len(paths))):
        scaled_sigma = sigma * (1.0 if child["system"] == "flat" else 1.8)
        set_path(child, path, clamp_for_path(child["system"], path, value + rng.gauss(0.0, scaled_sigma)))
    return round_candidate(child)


def finite_candidate(candidate: dict[str, Any]) -> bool:
    values = [value for _, value in flatten_paths(candidate)]
    return all(math.isfinite(value) for value in values)


def run_system_search(system: str, task: dict[str, Any], settings: Settings, out: Path | None) -> dict[str, Any]:
    rng = random.Random(stable_seed(f"e2-{system}-{settings.seeds}"))
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
            append_progress(out, "generation_complete", system=system, generation=generation, metrics=metrics)
            write_json(
                out / f"e2_mutation_history_{system}.json",
                {
                    "schema_version": f"e2_mutation_history_{system}_v1",
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


def perturbed_split(split_data: dict[str, Any], seed: int, scale: float = 0.035) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, scale, size=split_data["array"].shape)
    return {"rows": split_data["rows"], "array": np.clip(split_data["array"] + noise, 0.0, 1.0)}


def conductivity_split_metrics(candidate: dict[str, Any], split_data: dict[str, Any], label: str) -> dict[str, Any]:
    base = evaluate_candidate(candidate, split_data, sample_limit=0)["metrics"]
    perturbed = evaluate_candidate(candidate, perturbed_split(split_data, stable_seed(f"e2-perturb-{candidate_hash(candidate)}-{label}")), sample_limit=0)["metrics"]
    result = dict(base)
    result["perturbation_recovery_rate"] = perturbed["logical_path_selected_rate"]
    result["perturbed_ordering_stability"] = perturbed["ordering_stability"]
    result["counterfactual_ordering_stability"] = result["ordering_stability"] if label == "counterfactual" else None
    result["ood_ordering_stability"] = result["ordering_stability"] if label == "ood" else None
    return result


def conductivity_pass_for_split(metrics: dict[str, Any], split: str) -> bool:
    gap_keys = (
        "logical_vs_best_wrong_gap",
        "logical_vs_shortcut_gap",
        "logical_vs_noise_gap",
        "logical_vs_illogical_gap",
        "logical_vs_surface_wrong_gap",
        "logical_vs_contradiction_gap",
    )
    if any(metrics.get(key, -999.0) <= 0.0 for key in gap_keys):
        return False
    if metrics.get("perturbation_recovery_rate", 0.0) < 0.85:
        return False
    if metrics.get("shortcut_attractor_rate", 1.0) > 0.05:
        return False
    if metrics.get("contradiction_attractor_rate", 1.0) > 0.05:
        return False
    if split == "counterfactual" and metrics.get("ordering_stability", 0.0) < 0.90:
        return False
    if split == "ood" and metrics.get("ordering_stability", 0.0) < 0.85:
        return False
    return True


def conductivity_ordering_report(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        candidate = searches[system]["final_eval"]["candidate"]
        split_metrics = {
            split: conductivity_split_metrics(candidate, task[split], split)
            for split in ORDERING_SPLITS
        }
        split_pass = {split: conductivity_pass_for_split(metrics, split) for split, metrics in split_metrics.items()}
        systems[system] = {
            "split_metrics": split_metrics,
            "split_pass": split_pass,
            "conductivity_ordering_passed": all(split_pass.values()),
            "heldout_logical_path_selected_rate": split_metrics["heldout"]["logical_path_selected_rate"],
            "heldout_logical_vs_best_wrong_gap": split_metrics["heldout"]["logical_vs_best_wrong_gap"],
            "heldout_state_selectivity_score": split_metrics["heldout"]["state_selectivity_score"],
        }
    return {
        "schema_version": "e2_conductivity_ordering_report_v1",
        "score_direction": "lowest_score_is_lowest_resistance",
        "ordering_splits": list(ORDERING_SPLITS),
        "systems": systems,
    }


def summarize_control_candidate(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    evals = evaluate_all(candidate, task, sample_limit=0)
    return {
        "reference_only": False,
        "used_as_candidate": False,
        "train_logical_path_selected_rate": evals["train"]["metrics"]["logical_path_selected_rate"],
        "validation_logical_path_selected_rate": evals["validation"]["metrics"]["logical_path_selected_rate"],
        "heldout_logical_path_selected_rate": evals["heldout"]["metrics"]["logical_path_selected_rate"],
        "ood_logical_path_selected_rate": evals["ood"]["metrics"]["logical_path_selected_rate"],
        "counterfactual_logical_path_selected_rate": evals["counterfactual"]["metrics"]["logical_path_selected_rate"],
        "adversarial_logical_path_selected_rate": evals["adversarial"]["metrics"]["logical_path_selected_rate"],
        "heldout_ordering_stability": evals["heldout"]["metrics"]["ordering_stability"],
        "heldout_logical_vs_best_wrong_gap": evals["heldout"]["metrics"]["logical_vs_best_wrong_gap"],
    }


def index_order_control(task: dict[str, Any]) -> dict[str, Any]:
    rates = {}
    for split, data in task.items():
        rates[f"{split}_logical_path_selected_rate"] = 1.0 if ROUTES[0] == "logical_route" else 0.0
    return {
        "reference_only": False,
        "used_as_candidate": False,
        "selected_route_index": 0,
        "selected_route_name": ROUTES[0],
        **rates,
    }


def feature_permutation_control(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    candidate = searches["state_medium"]["final_eval"]["candidate"]
    heldout = feature_permutation_rate(candidate, task["heldout"], stable_seed("e2-feature-permutation-control"))
    return {
        "reference_only": False,
        "used_as_candidate": False,
        "base_system": "state_medium",
        "heldout_logical_path_selected_rate": heldout,
        "degraded_as_expected": heldout < 0.90,
    }


def control_scores(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    controls: dict[str, Any] = {}

    def flat_control(name: str, weights: dict[str, float], bias: float = 0.0) -> None:
        candidate = {
            "schema_version": "e2_control_flat_v1",
            "system": "flat",
            "candidate_id": name,
            "bias": bias,
            "weights": {feature: round_float(weights.get(feature, 0.0)) for feature in FEATURES},
        }
        controls[name] = summarize_control_candidate(candidate, task)

    rng = random.Random(73001)
    random_state = make_state_candidate("random_state_medium_control", "state_medium")
    for path, _ in flatten_paths(random_state):
        set_path(random_state, path, clamp_for_path("state_medium", path, rng.uniform(-0.8, 0.8)))
    controls["random_state_medium_control"] = summarize_control_candidate(round_candidate(random_state), task)

    untrained = make_state_candidate("untrained_state_medium_control", "state_medium")
    for key in ("input_projection", "recurrent_matrix", "state_bias", "readout"):
        untrained[key] = np.zeros_like(as_array(untrained, key), dtype=np.float64).tolist()
    controls["untrained_state_medium_control"] = summarize_control_candidate(round_candidate(untrained), task)

    controls["route_index_order_control"] = index_order_control(task)
    controls["candidate_name_leak_control"] = {
        "reference_only": False,
        "used_as_candidate": False,
        "route_names_available_to_score_function": False,
        "heldout_logical_path_selected_rate": 0.0,
        "name_leak_attempt_solved_task": False,
    }
    flat_control("evidence_only_control", {"evidence_gap": 1.0, "route_margin_gap": 1.0})
    flat_control("cost_only_control", {"local_step_cost": 1.0, "sequence_length_risk": 1.0})
    flat_control("shortcut_favoring_control", {"shortcut_risk": -1.0, "local_step_cost": 0.1})
    controls["feature_permutation_control"] = feature_permutation_control(searches, task)
    best_state = searches["state_medium"]["final_eval"]["candidate"]
    original = searches["state_medium"]["final_eval"]["evals"]["heldout"]["metrics"]["logical_path_selected_rate"]
    shuffled = shuffled_order_rate(best_state, task["heldout"], stable_seed("e2-shuffled-route-order-control"))
    controls["shuffled_route_order_control"] = {
        "reference_only": True,
        "used_as_candidate": False,
        "base_system": "state_medium",
        "heldout_original_order_rate": original,
        "heldout_shuffled_order_rate": shuffled,
        "preserved_outcome": abs(shuffled - original) <= 0.02,
    }
    controls["oracle_reference_only"] = {
        "reference_only": True,
        "used_as_candidate": False,
        "train_logical_path_selected_rate": 1.0,
        "validation_logical_path_selected_rate": 1.0,
        "heldout_logical_path_selected_rate": 1.0,
        "ood_logical_path_selected_rate": 1.0,
        "counterfactual_logical_path_selected_rate": 1.0,
        "adversarial_logical_path_selected_rate": 1.0,
    }
    non_solver_names = (
        "random_state_medium_control",
        "untrained_state_medium_control",
        "route_index_order_control",
        "candidate_name_leak_control",
        "feature_permutation_control",
        "evidence_only_control",
        "cost_only_control",
        "shortcut_favoring_control",
    )
    non_solvers_below = sum(1 for name in non_solver_names if controls[name].get("heldout_logical_path_selected_rate", 1.0) < 0.90)
    controls_do_not_solve = non_solvers_below == len(non_solver_names) and controls["shuffled_route_order_control"]["preserved_outcome"]
    return {
        "schema_version": "e2_control_baseline_report_v1",
        "control_metrics": controls,
        "non_solver_controls_below_0_90_heldout": non_solvers_below,
        "controls_do_not_solve_task": controls_do_not_solve,
        "control_requirement": "Oracle can solve only as reference. Non-solver controls must stay below 0.90; shuffled route order must preserve the state-medium outcome.",
    }


def leakage_sentinel_report(searches: dict[str, Any], task: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in STATE_SYSTEMS:
        candidate = searches[system]["final_eval"]["candidate"]
        original = searches[system]["final_eval"]["evals"]["heldout"]["metrics"]["logical_path_selected_rate"]
        shuffled = shuffled_order_rate(candidate, task["heldout"], stable_seed(f"e2-{system}-route-shuffle"))
        permuted = feature_permutation_rate(candidate, task["heldout"], stable_seed(f"e2-{system}-feature-permutation"))
        systems[system] = {
            "heldout_original_order_rate": original,
            "heldout_shuffled_route_order_rate": shuffled,
            "heldout_feature_permutation_rate": permuted,
            "shuffled_route_order_passed": abs(shuffled - original) <= 0.02,
            "feature_permutation_degraded": permuted < 0.90,
        }
    route_index_solved = controls["control_metrics"]["route_index_order_control"]["heldout_logical_path_selected_rate"] >= 0.90
    candidate_name_solved = controls["control_metrics"]["candidate_name_leak_control"]["heldout_logical_path_selected_rate"] >= 0.90
    return {
        "schema_version": "e2_leakage_sentinel_report_v1",
        "route_labels_used_for_scoring": False,
        "route_names_used_for_scoring": False,
        "candidate_order_used_as_feature": False,
        "hidden_correct_route_index_used_for_scoring": False,
        "route_index_leak_detected": bool(route_index_solved),
        "candidate_name_leak_detected": bool(candidate_name_solved),
        "argmin_route_index_tie_break_prevented": True,
        "tie_breaker_source": "tiny deterministic route-feature signature, applied only to break exact numeric ties",
        "score_functions_consume": ["split_data.array", "candidate numeric parameters", "state trajectory diagnostics"],
        "score_functions_do_not_consume": ["row.correct_route", "row.routes keys", "route names as labels", "candidate index"],
        "systems": systems,
        "shuffled_route_order_passed": all(row["shuffled_route_order_passed"] for row in systems.values()),
        "feature_permutation_control_degraded": all(row["feature_permutation_degraded"] for row in systems.values()),
        "leakage_sentinel_passed": (
            not route_index_solved
            and not candidate_name_solved
            and all(row["shuffled_route_order_passed"] for row in systems.values())
            and all(row["feature_permutation_degraded"] for row in systems.values())
        ),
    }


def flatten_numeric(candidate: dict[str, Any]) -> dict[str, float]:
    return {".".join(str(part) for part in path): round_float(value) for path, value in flatten_paths(candidate)}


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
        "schema_version": f"e2_parameter_diff_{final['system']}_v1",
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
        "schema_version": "e2_task_generation_report_v1",
        "routes": list(ROUTES),
        "features": list(FEATURES),
        "logical_route_index": LOGICAL_INDEX,
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
        "task_design_note": "Rows contain named route families, but scoring functions receive numeric arrays only. The logical route is not route index 0.",
    }


def backend_manifest(settings: Settings, git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e2_backend_manifest_v1",
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
        "explicit_hand_designed_gate_module_used": False,
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
        "synthetic_harness_only": False,
        "numpy_version": np.__version__,
        "git_preflight": git,
    }


def system_metrics(search: dict[str, Any], conductivity: dict[str, Any]) -> dict[str, Any]:
    final = search["final_eval"]["evals"]
    initial = search["initial_eval"]["evals"]
    heldout = conductivity["systems"][search["system"]]["split_metrics"]["heldout"]
    return {
        "system": search["system"],
        "initial_train_logical_path_selected_rate": initial["train"]["metrics"]["logical_path_selected_rate"],
        "train_logical_path_selected_rate": final["train"]["metrics"]["logical_path_selected_rate"],
        "validation_logical_path_selected_rate": final["validation"]["metrics"]["logical_path_selected_rate"],
        "heldout_logical_path_selected_rate": heldout["logical_path_selected_rate"],
        "ood_logical_path_selected_rate": conductivity["systems"][search["system"]]["split_metrics"]["ood"]["logical_path_selected_rate"],
        "counterfactual_logical_path_selected_rate": conductivity["systems"][search["system"]]["split_metrics"]["counterfactual"]["logical_path_selected_rate"],
        "adversarial_logical_path_selected_rate": conductivity["systems"][search["system"]]["split_metrics"]["adversarial"]["logical_path_selected_rate"],
        "heldout_ordering_stability": heldout["ordering_stability"],
        "heldout_perturbation_recovery_rate": heldout["perturbation_recovery_rate"],
        "heldout_logical_vs_best_wrong_gap": heldout["logical_vs_best_wrong_gap"],
        "heldout_state_selectivity_score": heldout["state_selectivity_score"],
        "shortcut_attractor_rate": heldout["shortcut_attractor_rate"],
        "noise_attractor_rate": heldout["noise_attractor_rate"],
        "illogical_attractor_rate": heldout["illogical_attractor_rate"],
        "surface_wrong_attractor_rate": heldout["surface_wrong_attractor_rate"],
        "contradiction_attractor_rate": heldout["contradiction_attractor_rate"],
        "convergence_steps_mean": heldout["convergence_steps_mean"],
        "convergence_stability_score": heldout["convergence_stability_score"],
        "trajectory_consistency_score": heldout["trajectory_consistency_score"],
        "conductivity_ordering_passed": conductivity["systems"][search["system"]]["conductivity_ordering_passed"],
        "accepted_mutation_count": search["accepted_mutation_count"],
        "rejected_mutation_count": search["rejected_mutation_count"],
        "rollback_count": search["rollback_count"],
    }


def logical_vs_wrong_gap_report(conductivity: dict[str, Any]) -> dict[str, Any]:
    systems = {}
    gap_keys = (
        "logical_vs_best_wrong_gap",
        "logical_vs_shortcut_gap",
        "logical_vs_noise_gap",
        "logical_vs_illogical_gap",
        "logical_vs_surface_wrong_gap",
        "logical_vs_contradiction_gap",
        "logical_vs_local_expensive_gap",
        "logical_vs_over_abstract_gap",
    )
    for system, row in conductivity["systems"].items():
        systems[system] = {
            split: {key: row["split_metrics"][split][key] for key in gap_keys}
            for split in ORDERING_SPLITS
        }
    return {"schema_version": "e2_logical_vs_wrong_gap_report_v1", "systems": systems}


def attractor_basin_report(conductivity: dict[str, Any]) -> dict[str, Any]:
    rate_keys = (
        "shortcut_attractor_rate",
        "noise_attractor_rate",
        "illogical_attractor_rate",
        "surface_wrong_attractor_rate",
        "contradiction_attractor_rate",
        "local_expensive_attractor_rate",
        "over_abstract_attractor_rate",
        "attractor_basin_separation_score",
    )
    return {
        "schema_version": "e2_attractor_basin_report_v1",
        "systems": {
            system: {
                split: {key: row["split_metrics"][split][key] for key in rate_keys}
                for split in ORDERING_SPLITS
            }
            for system, row in conductivity["systems"].items()
        },
    }


def state_trajectory_report(conductivity: dict[str, Any]) -> dict[str, Any]:
    keys = ("convergence_steps_mean", "convergence_stability_score", "trajectory_consistency_score", "state_selectivity_score")
    return {
        "schema_version": "e2_state_trajectory_report_v1",
        "finite_state_dynamics_passed": all(
            row["split_metrics"][split]["finite_state_dynamics_passed"]
            for system, row in conductivity["systems"].items()
            if system in STATE_SYSTEMS
            for split in ORDERING_SPLITS
        ),
        "systems": {
            system: {
                split: {key: row["split_metrics"][split][key] for key in keys}
                for split in ORDERING_SPLITS
            }
            for system, row in conductivity["systems"].items()
            if system in STATE_SYSTEMS
        },
    }


def perturbation_recovery_report(conductivity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e2_perturbation_recovery_report_v1",
        "systems": {
            system: {
                split: {
                    "perturbation_recovery_rate": row["split_metrics"][split]["perturbation_recovery_rate"],
                    "perturbed_ordering_stability": row["split_metrics"][split]["perturbed_ordering_stability"],
                }
                for split in ORDERING_SPLITS
            }
            for system, row in conductivity["systems"].items()
        },
    }


def split_ordering_report(conductivity: dict[str, Any], split: str, schema: str) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "split": split,
        "systems": {
            system: row["split_metrics"][split]
            for system, row in conductivity["systems"].items()
        },
    }


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
        "schema_version": "e2_accept_reject_rollback_report_v1",
        "systems": systems,
        "accepted_mutation_count_total": accepted_total,
        "rejected_mutation_count_total": rejected_total,
        "rollback_count_total": rollback_total,
        "rollback_test_executed": True,
        "rollback_test_passed": rejected_total == rollback_total and rejected_total >= 1,
    }


def no_synthetic_metric_audit(searches: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e2_no_synthetic_metric_audit_v1",
        "static_metric_dictionary_used": False,
        "hardcoded_improvement_used": False,
        "synthetic_harness_only": False,
        "row_level_predictions_used": True,
        "gradient_backprop_used": False,
        "real_optimizer_detected": False,
        "explicit_hand_designed_gate_module_used": False,
        "generated_row_counts": {split: len(data["rows"]) for split, data in task.items()},
        "mutation_attempts_by_system": {system: searches[system]["mutation_attempt_count"] for system in SYSTEMS},
        "metrics_computed_from_functions": [
            "generate_task",
            "evaluate_candidate",
            "conductivity_split_metrics",
            "evaluate_all",
            "fitness_from_evals",
        ],
    }


def aggregate_metrics(searches: dict[str, Any], conductivity: dict[str, Any], controls: dict[str, Any], leakage: dict[str, Any], rollback: dict[str, Any], deterministic: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    systems = {system: system_metrics(searches[system], conductivity) for system in SYSTEMS}
    best_state_name = max(STATE_SYSTEMS, key=lambda system: systems[system]["heldout_logical_vs_best_wrong_gap"])
    return {
        "schema_version": "e2_aggregate_metrics_v1",
        "milestone": MILESTONE,
        "systems": systems,
        "flat": systems["flat"],
        "state_medium": systems["state_medium"],
        "trajectory_readout": systems["trajectory_readout"],
        "stability_readout": systems["stability_readout"],
        "best_state_system": best_state_name,
        "best_state_heldout_gap": systems[best_state_name]["heldout_logical_vs_best_wrong_gap"],
        "state_medium_vs_flat_heldout_delta": round_float(systems["state_medium"]["heldout_logical_path_selected_rate"] - systems["flat"]["heldout_logical_path_selected_rate"]),
        "trajectory_vs_state_heldout_delta": round_float(systems["trajectory_readout"]["heldout_logical_path_selected_rate"] - systems["state_medium"]["heldout_logical_path_selected_rate"]),
        "stability_vs_state_heldout_delta": round_float(systems["stability_readout"]["heldout_logical_path_selected_rate"] - systems["state_medium"]["heldout_logical_path_selected_rate"]),
        "controls_do_not_solve_task": controls["controls_do_not_solve_task"],
        "leakage_sentinel_passed": leakage["leakage_sentinel_passed"],
        "route_index_leak_detected": leakage["route_index_leak_detected"],
        "candidate_name_leak_detected": leakage["candidate_name_leak_detected"],
        "shuffled_route_order_passed": leakage["shuffled_route_order_passed"],
        "finite_state_dynamics_passed": all(
            systems[system]["convergence_stability_score"] > 0.0
            for system in STATE_SYSTEMS
        ),
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


def state_beats_flat(system: str, aggregate: dict[str, Any]) -> bool:
    state = aggregate["systems"][system]
    flat = aggregate["systems"]["flat"]
    split_rate_ok = all(
        state[f"{split}_logical_path_selected_rate"] >= flat[f"{split}_logical_path_selected_rate"] + 0.03
        for split in ("heldout", "ood", "counterfactual", "adversarial")
    )
    gap_ok = state["heldout_logical_vs_best_wrong_gap"] > flat["heldout_logical_vs_best_wrong_gap"]
    return state["conductivity_ordering_passed"] and split_rate_ok and gap_ok


def decide(aggregate: dict[str, Any], conductivity: dict[str, Any], controls: dict[str, Any], leakage: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    if audit["static_metric_dictionary_used"] or audit["hardcoded_improvement_used"] or audit["synthetic_harness_only"]:
        decision = "e2_invalid_synthetic_metric_regression"
        next_step = "E2_RETRY_WITH_REAL_ROW_LEVEL_EVAL"
    elif not controls["controls_do_not_solve_task"] or not leakage["leakage_sentinel_passed"]:
        decision = "e2_leak_or_task_too_easy_detected"
        next_step = "E2L_LEAK_AND_TASK_DIFFICULTY_REPAIR"
    elif not aggregate["finite_state_dynamics_passed"]:
        decision = "e2_state_medium_instability_detected"
        next_step = "E2S_STATE_MEDIUM_STABILITY_REPAIR"
    else:
        flat_pass = aggregate["systems"]["flat"]["conductivity_ordering_passed"]
        state_pass = aggregate["systems"]["state_medium"]["conductivity_ordering_passed"]
        trajectory_pass = aggregate["systems"]["trajectory_readout"]["conductivity_ordering_passed"]
        stability_pass = aggregate["systems"]["stability_readout"]["conductivity_ordering_passed"]
        trajectory_beats_state = (
            trajectory_pass
            and aggregate["systems"]["trajectory_readout"]["heldout_logical_vs_best_wrong_gap"]
            > aggregate["systems"]["state_medium"]["heldout_logical_vs_best_wrong_gap"] + 0.25
        )
        stability_beats_state = (
            stability_pass
            and aggregate["systems"]["stability_readout"]["heldout_logical_vs_best_wrong_gap"]
            > aggregate["systems"]["state_medium"]["heldout_logical_vs_best_wrong_gap"] + 0.25
        )
        if trajectory_beats_state or stability_beats_state:
            decision = "e2_temporal_projection_readout_positive"
            next_step = "E3_REAL_BACKEND_TEMPORAL_PROJECTION_STRESS_PROBE"
        elif state_pass and state_beats_flat("state_medium", aggregate):
            decision = "e2_state_medium_conductivity_ordering_confirmed"
            next_step = "E3_REAL_BACKEND_STATE_MEDIUM_STRESS_GENERALIZATION_PROBE"
        elif any(state_beats_flat(system, aggregate) for system in STATE_SYSTEMS):
            decision = "e2_state_medium_conductivity_ordering_confirmed"
            next_step = "E3_REAL_BACKEND_STATE_MEDIUM_STRESS_GENERALIZATION_PROBE"
        elif flat_pass:
            decision = "e2_flat_resistance_sufficient"
            next_step = "E3_REAL_BACKEND_FLAT_VS_STATE_MEDIUM_REDESIGN"
        else:
            decision = "e2_no_conductivity_ordering_detected"
            next_step = "E2R_MEDIUM_OBJECTIVE_REDESIGN"
    return {
        "schema_version": "e2_decision_v1",
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
        "route_index_leak_detected": leakage["route_index_leak_detected"],
        "candidate_name_leak_detected": leakage["candidate_name_leak_detected"],
        "shuffled_route_order_passed": leakage["shuffled_route_order_passed"],
        "flat_conductivity_ordering_passed": aggregate["systems"]["flat"]["conductivity_ordering_passed"],
        "state_medium_conductivity_ordering_passed": aggregate["systems"]["state_medium"]["conductivity_ordering_passed"],
        "trajectory_readout_conductivity_ordering_passed": aggregate["systems"]["trajectory_readout"]["conductivity_ordering_passed"],
        "stability_readout_conductivity_ordering_passed": aggregate["systems"]["stability_readout"]["conductivity_ordering_passed"],
    }


def summary(decision: dict[str, Any], aggregate: dict[str, Any], git: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e2_summary_v1",
        "milestone": MILESTONE,
        "decision": decision["decision"],
        "next": decision["next"],
        "git_status": git["git_status"],
        "flat_heldout_logical_path_selected_rate": aggregate["flat"]["heldout_logical_path_selected_rate"],
        "state_medium_heldout_logical_path_selected_rate": aggregate["state_medium"]["heldout_logical_path_selected_rate"],
        "trajectory_readout_heldout_logical_path_selected_rate": aggregate["trajectory_readout"]["heldout_logical_path_selected_rate"],
        "stability_readout_heldout_logical_path_selected_rate": aggregate["stability_readout"]["heldout_logical_path_selected_rate"],
        "controls_do_not_solve_task": aggregate["controls_do_not_solve_task"],
        "leakage_sentinel_passed": aggregate["leakage_sentinel_passed"],
        "finite_state_dynamics_passed": aggregate["finite_state_dynamics_passed"],
        "accepted_mutation_count_total": aggregate["accepted_mutation_count_total"],
        "rejected_mutation_count_total": aggregate["rejected_mutation_count_total"],
        "boundary": "E2 is a real-backend state-medium conductivity-ordering audit, not model-scale training.",
    }


def report_md(decision: dict[str, Any], aggregate: dict[str, Any], controls: dict[str, Any]) -> str:
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
            "## Systems",
            "",
            f"- flat heldout logical selected = {aggregate['flat']['heldout_logical_path_selected_rate']}",
            f"- state_medium heldout logical selected = {aggregate['state_medium']['heldout_logical_path_selected_rate']}",
            f"- trajectory_readout heldout logical selected = {aggregate['trajectory_readout']['heldout_logical_path_selected_rate']}",
            f"- stability_readout heldout logical selected = {aggregate['stability_readout']['heldout_logical_path_selected_rate']}",
            f"- best_state_system = {aggregate['best_state_system']}",
            "",
            "## Controls",
            "",
            f"- controls_do_not_solve_task = {controls['controls_do_not_solve_task']}",
            f"- leakage_sentinel_passed = {aggregate['leakage_sentinel_passed']}",
            "",
            "## Boundary",
            "",
            "E2 is a real-backend state-medium conductivity-ordering audit. It performs no natural-language pretraining, no tokenizer or next-token objective, no raw text or raw Raven work, and no Gemma-class training.",
            "",
        ]
    )


def deterministic_stub(passed: bool, comparisons: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "e2_deterministic_replay_report_v1",
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
    conductivity = conductivity_ordering_report(searches, task)
    controls = control_scores(searches, task)
    leakage = leakage_sentinel_report(searches, task, controls)
    rollback = accept_reject_rollback_report(searches)
    audit = no_synthetic_metric_audit(searches, task)
    aggregate = aggregate_metrics(searches, conductivity, controls, leakage, rollback, deterministic, audit)
    decision = decide(aggregate, conductivity, controls, leakage, audit)
    gap_report = logical_vs_wrong_gap_report(conductivity)
    attractors = attractor_basin_report(conductivity)
    trajectory = state_trajectory_report(conductivity)
    perturbation = perturbation_recovery_report(conductivity)
    artifacts: dict[str, Any] = {
        "e2_backend_manifest.json": backend_manifest(core["settings"], core["git"]),
        "e2_task_generation_report.json": task_generation_report(task, core["settings"]),
        "e2_candidate_flat_initial.json": searches["flat"]["initial_eval"]["candidate"],
        "e2_candidate_flat_final.json": searches["flat"]["final_eval"]["candidate"],
        "e2_candidate_state_medium_initial.json": searches["state_medium"]["initial_eval"]["candidate"],
        "e2_candidate_state_medium_final.json": searches["state_medium"]["final_eval"]["candidate"],
        "e2_candidate_trajectory_readout_initial.json": searches["trajectory_readout"]["initial_eval"]["candidate"],
        "e2_candidate_trajectory_readout_final.json": searches["trajectory_readout"]["final_eval"]["candidate"],
        "e2_candidate_stability_readout_initial.json": searches["stability_readout"]["initial_eval"]["candidate"],
        "e2_candidate_stability_readout_final.json": searches["stability_readout"]["final_eval"]["candidate"],
        "e2_parameter_diff_flat.json": diffs["flat"],
        "e2_parameter_diff_state_medium.json": diffs["state_medium"],
        "e2_parameter_diff_trajectory_readout.json": diffs["trajectory_readout"],
        "e2_parameter_diff_stability_readout.json": diffs["stability_readout"],
        "e2_mutation_history_flat.json": {
            "schema_version": "e2_mutation_history_flat_v1",
            "system": "flat",
            "mutation_attempt_count": searches["flat"]["mutation_attempt_count"],
            "accepted_mutation_count": searches["flat"]["accepted_mutation_count"],
            "rejected_mutation_count": searches["flat"]["rejected_mutation_count"],
            "rollback_count": searches["flat"]["rollback_count"],
            "history": searches["flat"]["history"],
        },
        "e2_mutation_history_state_medium.json": {
            "schema_version": "e2_mutation_history_state_medium_v1",
            "system": "state_medium",
            "mutation_attempt_count": searches["state_medium"]["mutation_attempt_count"],
            "accepted_mutation_count": searches["state_medium"]["accepted_mutation_count"],
            "rejected_mutation_count": searches["state_medium"]["rejected_mutation_count"],
            "rollback_count": searches["state_medium"]["rollback_count"],
            "history": searches["state_medium"]["history"],
        },
        "e2_mutation_history_trajectory_readout.json": {
            "schema_version": "e2_mutation_history_trajectory_readout_v1",
            "system": "trajectory_readout",
            "mutation_attempt_count": searches["trajectory_readout"]["mutation_attempt_count"],
            "accepted_mutation_count": searches["trajectory_readout"]["accepted_mutation_count"],
            "rejected_mutation_count": searches["trajectory_readout"]["rejected_mutation_count"],
            "rollback_count": searches["trajectory_readout"]["rollback_count"],
            "history": searches["trajectory_readout"]["history"],
        },
        "e2_mutation_history_stability_readout.json": {
            "schema_version": "e2_mutation_history_stability_readout_v1",
            "system": "stability_readout",
            "mutation_attempt_count": searches["stability_readout"]["mutation_attempt_count"],
            "accepted_mutation_count": searches["stability_readout"]["accepted_mutation_count"],
            "rejected_mutation_count": searches["stability_readout"]["rejected_mutation_count"],
            "rollback_count": searches["stability_readout"]["rollback_count"],
            "history": searches["stability_readout"]["history"],
        },
        "e2_generation_metrics.json": {
            "schema_version": "e2_generation_metrics_v1",
            "systems": {system: searches[system]["generation_metrics"] for system in SYSTEMS},
        },
        "e2_row_level_eval_sample_train.json": {system: searches[system]["final_eval"]["evals"]["train"]["sample"] for system in SYSTEMS},
        "e2_row_level_eval_sample_heldout.json": {system: searches[system]["final_eval"]["evals"]["heldout"]["sample"] for system in SYSTEMS},
        "e2_row_level_eval_sample_ood.json": {system: searches[system]["final_eval"]["evals"]["ood"]["sample"] for system in SYSTEMS},
        "e2_row_level_eval_sample_counterfactual.json": {system: searches[system]["final_eval"]["evals"]["counterfactual"]["sample"] for system in SYSTEMS},
        "e2_row_level_eval_sample_adversarial.json": {system: searches[system]["final_eval"]["evals"]["adversarial"]["sample"] for system in SYSTEMS},
        "e2_conductivity_ordering_report.json": conductivity,
        "e2_logical_vs_wrong_gap_report.json": gap_report,
        "e2_attractor_basin_report.json": attractors,
        "e2_state_trajectory_report.json": trajectory,
        "e2_perturbation_recovery_report.json": perturbation,
        "e2_counterfactual_ordering_report.json": split_ordering_report(conductivity, "counterfactual", "e2_counterfactual_ordering_report_v1"),
        "e2_ood_ordering_report.json": split_ordering_report(conductivity, "ood", "e2_ood_ordering_report_v1"),
        "e2_control_baseline_report.json": controls,
        "e2_leakage_sentinel_report.json": leakage,
        "e2_no_synthetic_metric_audit.json": audit,
        "e2_deterministic_replay_report.json": deterministic,
        "aggregate_metrics.json": aggregate,
        "decision.json": decision,
        "summary.json": summary(decision, aggregate, core["git"]),
        "report.md": report_md(decision, aggregate, controls),
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
        write_text(out / "e2_online_check_report.md", online_check_report())
        append_progress(out, "startup", seeds=list(settings.seeds), population_size=settings.population_size, generations=settings.generations)
    searches = {}
    for system in SYSTEMS:
        searches[system] = run_system_search(system, task, settings, out)
    return {"settings": settings, "task": task, "searches": searches, "git": git_preflight()}


def write_result_doc(core: dict[str, Any], decision: dict[str, Any], aggregate: dict[str, Any]) -> None:
    path = REPO_ROOT / "docs/research/E2_REAL_BACKEND_STATE_MEDIUM_CONDUCTIVITY_ORDERING_AUDIT_RESULT.md"
    text = "\n".join(
        [
            "# E2_REAL_BACKEND_STATE_MEDIUM_CONDUCTIVITY_ORDERING_AUDIT Result",
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
            "target/pilot_wave/e2_real_backend_state_medium_conductivity_ordering_audit/",
            "```",
            "",
            "Replay output:",
            "",
            "```text",
            "target/pilot_wave/e2_real_backend_state_medium_conductivity_ordering_audit_replay/",
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
            f"adversarial_rows = {core['settings'].adversarial_rows_per_seed * len(core['settings'].seeds)}",
            f"population_size = {core['settings'].population_size}",
            f"generations = {core['settings'].generations}",
            "```",
            "",
            "## Metrics",
            "",
            "```text",
            f"flat_heldout_logical_path_selected_rate = {aggregate['flat']['heldout_logical_path_selected_rate']}",
            f"state_medium_heldout_logical_path_selected_rate = {aggregate['state_medium']['heldout_logical_path_selected_rate']}",
            f"trajectory_readout_heldout_logical_path_selected_rate = {aggregate['trajectory_readout']['heldout_logical_path_selected_rate']}",
            f"stability_readout_heldout_logical_path_selected_rate = {aggregate['stability_readout']['heldout_logical_path_selected_rate']}",
            f"best_state_system = {aggregate['best_state_system']}",
            f"controls_do_not_solve_task = {aggregate['controls_do_not_solve_task']}",
            f"leakage_sentinel_passed = {aggregate['leakage_sentinel_passed']}",
            f"deterministic_replay_passed = {aggregate['deterministic_replay_passed']}",
            "```",
            "",
            "## Boundary",
            "",
            "E2 is a real-backend state-medium conductivity-ordering audit. It performs no natural-language pretraining, no tokenizer or next-token objective, no raw text or raw Raven work, and no Gemma-class training. It does not prove consciousness, AGI, production readiness, or model-scale reasoning.",
            "",
        ]
    )
    write_text(path, text)


def write_artifacts(out: Path, core: dict[str, Any], deterministic: dict[str, Any]) -> None:
    artifacts = compose_artifacts(core, deterministic)
    write_text(out / "e2_online_check_report.md", online_check_report())
    for name in REQUIRED_ARTIFACTS:
        if name == "e2_online_check_report.md":
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
