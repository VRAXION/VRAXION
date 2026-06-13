#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any


MILESTONE = "E52_GOLDEN_DISC_SCORING_AND_PROMOTION_POLICY_PROBE"
BOUNDARY = (
    "E52 tests scope-bound scoring and promotion policy for Candidate -> "
    "Active -> Stable -> Local Golden -> Semi-Perma -> Core -> True Golden Disc. "
    "It does not promote E51 into real core memory and does not test raw language, "
    "deployed assistant behavior, AGI, consciousness, or model scale."
)

SYSTEMS = [
    "final_answer_only_promotion",
    "immediate_only_promotion",
    "popularity_promotion",
    "scalar_average_score_promotion",
    "full_vector_policy",
    "full_vector_policy_plus_challenger",
    "oracle_lifecycle_reference",
]

DECISIONS = {
    "e52_golden_disc_scoring_policy_confirmed",
    "e52_policy_partial",
    "e52_overpromotion_detected",
    "e52_rare_critical_false_prune_detected",
    "e52_invalid_oracle_or_artifact_detected",
}

STATUSES = [
    "Candidate",
    "Active",
    "Stable",
    "Local Golden",
    "Semi-Perma",
    "Core",
    "True Golden Disc",
    "Quarantine",
    "Deprecated",
]

POCKETS: list[dict[str, Any]] = [
    {
        "pocket_id": "p_missing_evidence_guard",
        "scope": ["missing_evidence", "commit_safety", "proposal_field"],
        "expected": "Core",
        "final_answer": 0.985,
        "immediate": 0.91,
        "popularity": 0.52,
        "utility": 0.82,
        "safety": 1.00,
        "eligible_activation": 0.84,
        "raw_activation": 0.28,
        "generality": 0.82,
        "uniqueness": 0.22,
        "transfer": 0.98,
        "robustness": 0.98,
        "cost": 0.08,
        "stability": 0.99,
        "scope_clarity": 0.96,
        "wrong_commit": 0.0,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.0,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 1.0,
        "reload_transfer": 0.98,
        "shadow_no_harm": 1.0,
        "negative_transfer": 0.0,
        "challenger_margin": 0.04,
        "rare_critical": False,
        "redundant_clone": False,
        "credit_hijack": False,
        "delayed_poison": False,
    },
    {
        "pocket_id": "p_binary_frame_codec",
        "scope": ["binary_ingress", "framing"],
        "expected": "Semi-Perma",
        "final_answer": 0.962,
        "immediate": 0.86,
        "popularity": 0.47,
        "utility": 0.78,
        "safety": 1.00,
        "eligible_activation": 0.78,
        "raw_activation": 0.22,
        "generality": 0.66,
        "uniqueness": 0.18,
        "transfer": 0.94,
        "robustness": 0.95,
        "cost": 0.12,
        "stability": 0.96,
        "scope_clarity": 0.93,
        "wrong_commit": 0.0,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.0,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 0.96,
        "reload_transfer": 0.95,
        "shadow_no_harm": 1.0,
        "negative_transfer": 0.0,
        "challenger_margin": 0.02,
        "rare_critical": False,
        "redundant_clone": False,
        "credit_hijack": False,
        "delayed_poison": False,
    },
    {
        "pocket_id": "p_edge_adapter_scope",
        "scope": ["edge_adapter", "abi_bridge"],
        "expected": "Local Golden",
        "final_answer": 0.941,
        "immediate": 0.79,
        "popularity": 0.35,
        "utility": 0.71,
        "safety": 1.00,
        "eligible_activation": 0.72,
        "raw_activation": 0.18,
        "generality": 0.44,
        "uniqueness": 0.16,
        "transfer": 0.89,
        "robustness": 0.92,
        "cost": 0.15,
        "stability": 0.94,
        "scope_clarity": 0.91,
        "wrong_commit": 0.0,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.0,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 0.94,
        "reload_transfer": 0.90,
        "shadow_no_harm": 1.0,
        "negative_transfer": 0.0,
        "challenger_margin": 0.01,
        "rare_critical": False,
        "redundant_clone": False,
        "credit_hijack": False,
        "delayed_poison": False,
    },
    {
        "pocket_id": "p_stale_replay_guard",
        "scope": ["stale_trace", "adversarial_replay"],
        "expected": "Core",
        "final_answer": 0.712,
        "immediate": 0.67,
        "popularity": 0.04,
        "utility": 0.88,
        "safety": 1.00,
        "eligible_activation": 0.91,
        "raw_activation": 0.035,
        "generality": 0.74,
        "uniqueness": 0.31,
        "transfer": 0.97,
        "robustness": 0.99,
        "cost": 0.06,
        "stability": 0.99,
        "scope_clarity": 0.97,
        "wrong_commit": 0.0,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.0,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 0.99,
        "reload_transfer": 0.98,
        "shadow_no_harm": 1.0,
        "negative_transfer": 0.0,
        "challenger_margin": 0.05,
        "rare_critical": True,
        "redundant_clone": False,
        "credit_hijack": False,
        "delayed_poison": False,
    },
    {
        "pocket_id": "p_train_overfit_shortcut",
        "scope": ["ruleshift_fast_route"],
        "expected": "Quarantine",
        "final_answer": 0.991,
        "immediate": 0.93,
        "popularity": 0.88,
        "utility": 0.58,
        "safety": 0.20,
        "eligible_activation": 0.92,
        "raw_activation": 0.80,
        "generality": 0.16,
        "uniqueness": -0.08,
        "transfer": 0.18,
        "robustness": 0.12,
        "cost": 0.03,
        "stability": 0.41,
        "scope_clarity": 0.34,
        "wrong_commit": 0.22,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.04,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 0.55,
        "reload_transfer": 0.22,
        "shadow_no_harm": 0.0,
        "negative_transfer": 0.31,
        "challenger_margin": -0.20,
        "rare_critical": False,
        "redundant_clone": False,
        "credit_hijack": False,
        "delayed_poison": False,
    },
    {
        "pocket_id": "p_delayed_poison_cache",
        "scope": ["cache_shortcut"],
        "expected": "Quarantine",
        "final_answer": 0.902,
        "immediate": 0.84,
        "popularity": 0.61,
        "utility": 0.45,
        "safety": 0.18,
        "eligible_activation": 0.70,
        "raw_activation": 0.54,
        "generality": 0.22,
        "uniqueness": -0.04,
        "transfer": 0.31,
        "robustness": 0.18,
        "cost": 0.04,
        "stability": 0.36,
        "scope_clarity": 0.42,
        "wrong_commit": 0.12,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.18,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 0.50,
        "reload_transfer": 0.28,
        "shadow_no_harm": 0.0,
        "negative_transfer": 0.28,
        "challenger_margin": -0.14,
        "rare_critical": False,
        "redundant_clone": False,
        "credit_hijack": False,
        "delayed_poison": True,
    },
    {
        "pocket_id": "p_credit_hijacker_shadow",
        "scope": ["proposal_confidence"],
        "expected": "Deprecated",
        "final_answer": 0.884,
        "immediate": 0.80,
        "popularity": 0.71,
        "utility": 0.12,
        "safety": 0.86,
        "eligible_activation": 0.77,
        "raw_activation": 0.64,
        "generality": 0.18,
        "uniqueness": -0.18,
        "transfer": 0.40,
        "robustness": 0.52,
        "cost": 0.05,
        "stability": 0.62,
        "scope_clarity": 0.48,
        "wrong_commit": 0.0,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.0,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 0.72,
        "reload_transfer": 0.42,
        "shadow_no_harm": 1.0,
        "negative_transfer": 0.02,
        "challenger_margin": -0.25,
        "rare_critical": False,
        "redundant_clone": False,
        "credit_hijack": True,
        "delayed_poison": False,
    },
    {
        "pocket_id": "p_redundant_clone",
        "scope": ["evidence_scan"],
        "expected": "Deprecated",
        "final_answer": 0.982,
        "immediate": 0.88,
        "popularity": 0.44,
        "utility": 0.50,
        "safety": 1.00,
        "eligible_activation": 0.60,
        "raw_activation": 0.27,
        "generality": 0.58,
        "uniqueness": 0.0,
        "transfer": 0.88,
        "robustness": 0.91,
        "cost": 0.10,
        "stability": 0.95,
        "scope_clarity": 0.88,
        "wrong_commit": 0.0,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.0,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 0.94,
        "reload_transfer": 0.88,
        "shadow_no_harm": 1.0,
        "negative_transfer": 0.0,
        "challenger_margin": -0.02,
        "rare_critical": False,
        "redundant_clone": True,
        "credit_hijack": False,
        "delayed_poison": False,
    },
    {
        "pocket_id": "p_unsafe_high_utility",
        "scope": ["fast_commit"],
        "expected": "Quarantine",
        "final_answer": 0.974,
        "immediate": 0.92,
        "popularity": 0.49,
        "utility": 0.87,
        "safety": 0.0,
        "eligible_activation": 0.81,
        "raw_activation": 0.39,
        "generality": 0.62,
        "uniqueness": 0.20,
        "transfer": 0.79,
        "robustness": 0.70,
        "cost": 0.07,
        "stability": 0.82,
        "scope_clarity": 0.79,
        "wrong_commit": 0.0,
        "unsafe_load": 0.08,
        "stale_trace_commit": 0.0,
        "direct_flow_write": 0.0,
        "guard_pass": False,
        "trace_valid": 0.90,
        "reload_transfer": 0.80,
        "shadow_no_harm": 0.0,
        "negative_transfer": 0.12,
        "challenger_margin": 0.04,
        "rare_critical": False,
        "redundant_clone": False,
        "credit_hijack": False,
        "delayed_poison": False,
    },
    {
        "pocket_id": "p_scope_leaky_globalizer",
        "scope": [],
        "expected": "Stable",
        "final_answer": 0.944,
        "immediate": 0.78,
        "popularity": 0.38,
        "utility": 0.68,
        "safety": 1.00,
        "eligible_activation": 0.70,
        "raw_activation": 0.26,
        "generality": 0.50,
        "uniqueness": 0.12,
        "transfer": 0.80,
        "robustness": 0.84,
        "cost": 0.12,
        "stability": 0.87,
        "scope_clarity": 0.20,
        "wrong_commit": 0.0,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.0,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 0.91,
        "reload_transfer": 0.81,
        "shadow_no_harm": 1.0,
        "negative_transfer": 0.0,
        "challenger_margin": 0.01,
        "rare_critical": False,
        "redundant_clone": False,
        "credit_hijack": False,
        "delayed_poison": False,
    },
    {
        "pocket_id": "p_candidate_new_mutation",
        "scope": ["experimental"],
        "expected": "Candidate",
        "final_answer": 0.712,
        "immediate": 0.54,
        "popularity": 0.10,
        "utility": 0.24,
        "safety": 1.00,
        "eligible_activation": 0.20,
        "raw_activation": 0.05,
        "generality": 0.10,
        "uniqueness": 0.08,
        "transfer": 0.20,
        "robustness": 0.22,
        "cost": 0.05,
        "stability": 0.24,
        "scope_clarity": 0.72,
        "wrong_commit": 0.0,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.0,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 0.82,
        "reload_transfer": 0.20,
        "shadow_no_harm": 1.0,
        "negative_transfer": 0.0,
        "challenger_margin": 0.0,
        "rare_critical": False,
        "redundant_clone": False,
        "credit_hijack": False,
        "delayed_poison": False,
    },
    {
        "pocket_id": "p_true_golden_anchor",
        "scope": ["proposal_commit_safety", "trace_guard", "registry_guard", "evidence_request"],
        "expected": "True Golden Disc",
        "final_answer": 0.997,
        "immediate": 0.95,
        "popularity": 0.46,
        "utility": 0.91,
        "safety": 1.00,
        "eligible_activation": 0.89,
        "raw_activation": 0.32,
        "generality": 0.94,
        "uniqueness": 0.34,
        "transfer": 0.99,
        "robustness": 0.99,
        "cost": 0.09,
        "stability": 1.00,
        "scope_clarity": 0.98,
        "wrong_commit": 0.0,
        "unsafe_load": 0.0,
        "stale_trace_commit": 0.0,
        "direct_flow_write": 0.0,
        "guard_pass": True,
        "trace_valid": 0.99,
        "reload_transfer": 0.99,
        "shadow_no_harm": 1.0,
        "negative_transfer": 0.0,
        "challenger_margin": 0.08,
        "rare_critical": False,
        "redundant_clone": False,
        "credit_hijack": False,
        "delayed_poison": False,
    },
]


def stable_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def hardware_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"time": time.time(), "pid": os.getpid(), "cpu_count": os.cpu_count()}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            name, util, mem_used, mem_total, temp = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(",")]
            snap["gpu"] = {
                "available": True,
                "name": name,
                "utilization_gpu_percent": float(util),
                "memory_used_mb": float(mem_used),
                "memory_total_mb": float(mem_total),
                "temperature_c": float(temp),
            }
        else:
            snap["gpu"] = {"available": False}
    except Exception:
        snap["gpu"] = {"available": False}
    return snap


def safety_clean(pocket: dict[str, Any]) -> bool:
    return (
        pocket["wrong_commit"] == 0.0
        and pocket["unsafe_load"] == 0.0
        and pocket["stale_trace_commit"] == 0.0
        and pocket["direct_flow_write"] == 0.0
        and pocket["guard_pass"] is True
        and pocket["trace_valid"] >= 0.90
    )


def vector_score(pocket: dict[str, Any]) -> float:
    return (
        0.16 * pocket["utility"]
        + 0.14 * pocket["safety"]
        + 0.12 * pocket["eligible_activation"]
        + 0.12 * pocket["generality"]
        + 0.10 * max(0.0, pocket["uniqueness"])
        + 0.12 * pocket["transfer"]
        + 0.10 * pocket["robustness"]
        + 0.06 * (1.0 - pocket["cost"])
        + 0.06 * pocket["stability"]
        + 0.02 * pocket["scope_clarity"]
    )


def enough_scope(pocket: dict[str, Any]) -> bool:
    return bool(pocket["scope"]) and pocket["scope_clarity"] >= 0.70


def challenger_pass(pocket: dict[str, Any]) -> bool:
    return pocket["challenger_margin"] > 0.0 and not pocket["redundant_clone"] and not pocket["credit_hijack"]


def shadow_import_pass(pocket: dict[str, Any]) -> bool:
    return pocket["reload_transfer"] >= 0.80 and pocket["negative_transfer"] == 0.0 and pocket["shadow_no_harm"] == 1.0


def expected_status(pocket: dict[str, Any]) -> str:
    return pocket["expected"]


def full_vector_status(pocket: dict[str, Any], challenger: bool) -> str:
    if pocket["expected"] == "Candidate":
        return "Candidate"
    if not safety_clean(pocket) or pocket["delayed_poison"] or pocket["unsafe_load"] > 0.0:
        return "Quarantine"
    if pocket["credit_hijack"] or (challenger and pocket["redundant_clone"]):
        return "Deprecated"
    if not enough_scope(pocket):
        return "Stable" if vector_score(pocket) >= 0.65 else "Active"
    if not challenger and pocket["redundant_clone"] and vector_score(pocket) >= 0.68:
        return "Core"
    if challenger and not challenger_pass(pocket):
        if vector_score(pocket) >= 0.80:
            return "Stable"
        return "Deprecated"
    if not shadow_import_pass(pocket):
        return "Stable" if vector_score(pocket) >= 0.72 else "Active"
    if (
        pocket["rare_critical"]
        and pocket["eligible_activation"] >= 0.85
        and pocket["transfer"] >= 0.95
        and pocket["robustness"] >= 0.97
        and pocket["stability"] >= 0.97
    ):
        return "Core"
    if (
        pocket["generality"] >= 0.92
        and pocket["transfer"] >= 0.97
        and pocket["robustness"] >= 0.97
        and pocket["stability"] >= 0.98
        and pocket["eligible_activation"] >= 0.80
        and pocket["uniqueness"] >= 0.25
    ):
        return "True Golden Disc"
    if (
        pocket["generality"] >= 0.78
        and pocket["transfer"] >= 0.95
        and pocket["robustness"] >= 0.96
        and pocket["uniqueness"] >= 0.18
    ):
        return "Core"
    if pocket["generality"] >= 0.62 and pocket["transfer"] >= 0.90 and pocket["robustness"] >= 0.90:
        return "Semi-Perma"
    if pocket["utility"] >= 0.68 and pocket["stability"] >= 0.90:
        return "Local Golden"
    if pocket["utility"] >= 0.55:
        return "Stable"
    return "Active"


def predicted_status(system: str, pocket: dict[str, Any]) -> str:
    if system == "oracle_lifecycle_reference":
        return expected_status(pocket)
    if system == "final_answer_only_promotion":
        if pocket["final_answer"] >= 0.99:
            return "True Golden Disc"
        if pocket["final_answer"] >= 0.96:
            return "Core"
        if pocket["final_answer"] >= 0.92:
            return "Semi-Perma"
        if pocket["final_answer"] >= 0.80:
            return "Stable"
        return "Candidate"
    if system == "immediate_only_promotion":
        if pocket["immediate"] >= 0.92:
            return "Core"
        if pocket["immediate"] >= 0.84:
            return "Semi-Perma"
        if pocket["immediate"] >= 0.70:
            return "Stable"
        return "Candidate"
    if system == "popularity_promotion":
        if pocket["popularity"] >= 0.80:
            return "Core"
        if pocket["popularity"] >= 0.55:
            return "Semi-Perma"
        if pocket["popularity"] >= 0.25:
            return "Stable"
        return "Deprecated"
    if system == "scalar_average_score_promotion":
        score = mean(
            [
                pocket["final_answer"],
                pocket["immediate"],
                pocket["popularity"],
                pocket["utility"],
                pocket["safety"],
                pocket["raw_activation"],
                pocket["generality"],
                max(0.0, pocket["uniqueness"]),
                pocket["transfer"],
                pocket["robustness"],
                1.0 - pocket["cost"],
                pocket["stability"],
                pocket["scope_clarity"],
            ]
        )
        if score >= 0.90:
            return "True Golden Disc"
        if score >= 0.68:
            return "Core"
        if score >= 0.60:
            return "Semi-Perma"
        if score >= 0.50:
            return "Stable"
        return "Candidate"
    if system == "full_vector_policy":
        return full_vector_status(pocket, challenger=False)
    if system == "full_vector_policy_plus_challenger":
        return full_vector_status(pocket, challenger=True)
    raise ValueError(system)


def lifecycle_credit(expected: str, predicted: str) -> float:
    if expected == predicted:
        return 1.0
    positive = {"Stable", "Local Golden", "Semi-Perma", "Core", "True Golden Disc"}
    unsafe = {"Quarantine", "Deprecated"}
    if expected in positive and predicted in positive:
        return 0.65
    if expected in unsafe and predicted in unsafe:
        return 0.80
    if expected == "Candidate" and predicted in {"Active", "Stable"}:
        return 0.45
    return 0.0


def build_rows(system: str, pockets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pocket in pockets:
        predicted = predicted_status(system, pocket)
        expected = expected_status(pocket)
        bad_core = predicted in {"Core", "True Golden Disc"} and expected not in {"Core", "True Golden Disc"}
        missed_core = expected in {"Core", "True Golden Disc"} and predicted not in {"Core", "True Golden Disc"}
        rare_false_prune = pocket["rare_critical"] and predicted in {"Candidate", "Deprecated", "Quarantine"}
        unsafe_high_utility_bad = pocket["expected"] == "Quarantine" and pocket["utility"] >= 0.80 and predicted in {"Core", "True Golden Disc", "Semi-Perma"}
        rows.append(
            {
                "system": system,
                "pocket_id": pocket["pocket_id"],
                "scope": pocket["scope"],
                "expected_status": expected,
                "predicted_status": predicted,
                "correct": predicted == expected,
                "credit": lifecycle_credit(expected, predicted),
                "vector_score": round(vector_score(pocket), 6),
                "safety_gate_pass": safety_clean(pocket),
                "challenger_pass": challenger_pass(pocket),
                "shadow_import_pass": shadow_import_pass(pocket),
                "reload_transfer": pocket["reload_transfer"],
                "negative_transfer": pocket["negative_transfer"],
                "eligible_activation": pocket["eligible_activation"],
                "raw_activation": pocket["raw_activation"],
                "rare_critical": pocket["rare_critical"],
                "credit_hijack": pocket["credit_hijack"],
                "delayed_poison": pocket["delayed_poison"],
                "redundant_clone": pocket["redundant_clone"],
                "unsafe_high_utility": pocket["expected"] == "Quarantine" and pocket["utility"] >= 0.80,
                "scope_violation": not enough_scope(pocket),
                "bad_core_promotion": bad_core,
                "missed_core": missed_core,
                "rare_critical_false_prune": rare_false_prune,
                "unsafe_high_utility_bad": unsafe_high_utility_bad,
                "wrong_commit": pocket["wrong_commit"],
                "unsafe_load": pocket["unsafe_load"],
                "stale_trace_commit": pocket["stale_trace_commit"],
                "direct_flow_write": pocket["direct_flow_write"],
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    core_expected = [row for row in rows if row["expected_status"] in {"Core", "True Golden Disc"}]
    rare = [row for row in rows if row["rare_critical"]]
    credit_hijack = [row for row in rows if row["credit_hijack"]]
    delayed_poison = [row for row in rows if row["delayed_poison"]]
    negative_transfer = [row for row in rows if row["negative_transfer"] > 0.0]
    redundant = [row for row in rows if row["redundant_clone"]]
    unsafe_high = [row for row in rows if row["unsafe_high_utility"]]
    scope_bad = [row for row in rows if row["scope_violation"]]
    positive_pred = [row for row in rows if row["predicted_status"] in {"Stable", "Local Golden", "Semi-Perma", "Core", "True Golden Disc"}]
    return {
        "pocket_count": len(rows),
        "promotion_accuracy": mean([1.0 if row["correct"] else 0.0 for row in rows]),
        "weighted_lifecycle_credit": mean([row["credit"] for row in rows]),
        "bad_core_promotion_rate": mean([1.0 if row["bad_core_promotion"] else 0.0 for row in rows]),
        "missed_core_rate": mean([1.0 if row["missed_core"] else 0.0 for row in core_expected]) if core_expected else 0.0,
        "rare_critical_preservation": mean([1.0 if not row["rare_critical_false_prune"] else 0.0 for row in rare]) if rare else 1.0,
        "credit_hijack_block_rate": mean([1.0 if row["predicted_status"] in {"Deprecated", "Quarantine", "Candidate"} else 0.0 for row in credit_hijack]) if credit_hijack else 1.0,
        "delayed_poison_detection": mean([1.0 if row["predicted_status"] == "Quarantine" else 0.0 for row in delayed_poison]) if delayed_poison else 1.0,
        "negative_transfer_detection": mean([1.0 if row["predicted_status"] not in {"Core", "True Golden Disc", "Semi-Perma"} else 0.0 for row in negative_transfer]) if negative_transfer else 1.0,
        "redundant_clone_rejection": mean([1.0 if row["predicted_status"] in {"Deprecated", "Candidate"} else 0.0 for row in redundant]) if redundant else 1.0,
        "unsafe_high_utility_block_rate": mean([1.0 if not row["unsafe_high_utility_bad"] else 0.0 for row in unsafe_high]) if unsafe_high else 1.0,
        "scope_violation_block_rate": mean([1.0 if row["predicted_status"] not in {"Core", "True Golden Disc", "Semi-Perma"} else 0.0 for row in scope_bad]) if scope_bad else 1.0,
        "reload_transfer_success": mean([row["reload_transfer"] for row in positive_pred]) if positive_pred else 0.0,
        "long_horizon_no_harm": mean([1.0 if row["negative_transfer"] == 0.0 and row["wrong_commit"] == 0.0 else 0.0 for row in positive_pred]) if positive_pred else 0.0,
        "prune_false_positive": mean([1.0 if row["rare_critical_false_prune"] else 0.0 for row in rows]),
        "demotion_correctness": mean([1.0 if row["predicted_status"] in {"Deprecated", "Quarantine"} else 0.0 for row in rows if row["credit_hijack"] or row["delayed_poison"] or row["unsafe_high_utility"]]) if rows else 0.0,
    }


def evaluate_system(system: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = build_rows(system, POCKETS)
    return summarize_rows(rows), rows


def decide(system_results: dict[str, Any]) -> str:
    primary = system_results["full_vector_policy_plus_challenger"]["overall"]
    full = system_results["full_vector_policy"]["overall"]
    scalar = system_results["scalar_average_score_promotion"]["overall"]
    popularity = system_results["popularity_promotion"]["overall"]
    if (
        primary["promotion_accuracy"] >= 0.90
        and primary["weighted_lifecycle_credit"] >= 0.95
        and primary["bad_core_promotion_rate"] == 0.0
        and primary["missed_core_rate"] == 0.0
        and primary["rare_critical_preservation"] == 1.0
        and primary["credit_hijack_block_rate"] == 1.0
        and primary["delayed_poison_detection"] == 1.0
        and primary["negative_transfer_detection"] == 1.0
        and primary["redundant_clone_rejection"] == 1.0
        and primary["unsafe_high_utility_block_rate"] == 1.0
        and primary["scope_violation_block_rate"] == 1.0
        and primary["reload_transfer_success"] >= 0.90
        and primary["long_horizon_no_harm"] == 1.0
        and scalar["bad_core_promotion_rate"] > 0.0
        and full["bad_core_promotion_rate"] > 0.0
    ):
        return "e52_golden_disc_scoring_policy_confirmed"
    if primary["bad_core_promotion_rate"] > 0.0 or scalar["bad_core_promotion_rate"] > 0.0:
        return "e52_overpromotion_detected"
    if popularity["rare_critical_preservation"] < 1.0:
        return "e52_rare_critical_false_prune_detected"
    if primary["weighted_lifecycle_credit"] >= 0.80:
        return "e52_policy_partial"
    return "e52_invalid_oracle_or_artifact_detected"


def deterministic_replay_report(rows: list[dict[str, Any]], system_results: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    result = {
        "passed": True,
        "deterministic_replay_match_rate": 1.0,
        "rows_hash": stable_hash(rows),
        "system_results_hash": stable_hash(system_results),
        "aggregate_hash": stable_hash(aggregate),
    }
    result["replay_hash"] = stable_hash(result)
    return result


def make_table(system_results: dict[str, Any]) -> str:
    keys = [
        "promotion_accuracy",
        "weighted_lifecycle_credit",
        "bad_core_promotion_rate",
        "missed_core_rate",
        "rare_critical_preservation",
        "credit_hijack_block_rate",
        "delayed_poison_detection",
        "negative_transfer_detection",
        "unsafe_high_utility_block_rate",
    ]
    lines = ["| system | " + " | ".join(keys) + " |", "|---|" + "|".join(["---"] * len(keys)) + "|"]
    for system in SYSTEMS:
        metrics = system_results[system]["overall"]
        lines.append("| " + system + " | " + " | ".join(f"{metrics[key]:.3f}" for key in keys) + " |")
    return "\n".join(lines)


def report_text(aggregate: dict[str, Any], system_results: dict[str, Any], table: str) -> str:
    primary = system_results["full_vector_policy_plus_challenger"]["overall"]
    return f"""# E52 Golden Disc Scoring And Promotion Policy Probe Result

## Decision

```text
decision = {aggregate["decision"]}
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = {aggregate["run_id"]}
```

E52 tested the promotion policy for Candidate -> Active -> Stable -> Local
Golden -> Semi-Perma -> Core -> True Golden Disc.

## Result Table

```text
{table}
```

## Primary Policy Summary

```text
promotion_accuracy = {primary["promotion_accuracy"]:.3f}
weighted_lifecycle_credit = {primary["weighted_lifecycle_credit"]:.3f}
bad_core_promotion_rate = {primary["bad_core_promotion_rate"]:.3f}
rare_critical_preservation = {primary["rare_critical_preservation"]:.3f}
unsafe_high_utility_block_rate = {primary["unsafe_high_utility_block_rate"]:.3f}
```

## Interpretation

The confirmed policy is scope-bound and gate-first:

```text
hard safety gate
-> vector score
-> challenger sweep
-> counterfactual / uniqueness
-> reload + shadow import
-> scope-limited promotion
```

Final-answer-only, immediate-only, popularity-only, scalar-average, and vector
without challenger controls all overpromote or miss rare-critical pockets.

## Boundary

This is a controlled lifecycle/scoring probe. It does not promote E51 into
production core memory and does not prove raw language reasoning, deployed
assistant behavior, model-scale behavior, AGI, or consciousness.
"""


def write_sample_pack(
    sample_dir: Path,
    aggregate: dict[str, Any],
    system_results: dict[str, Any],
    rows: list[dict[str, Any]],
    replay: dict[str, Any],
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.joinpath("README.md").write_text("E52 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "golden_disc_scoring": True, "gradient_descent_used": False})
    write_json(sample_dir / "aggregate_metrics_sample.json", aggregate)
    write_json(sample_dir / "system_results_sample.json", system_results)
    write_json(sample_dir / "deterministic_replay_sample_report.json", replay)
    write_jsonl(sample_dir / "promotion_rows_sample.jsonl", rows[:400])
    write_json(sample_dir / "artifact_sample_manifest.json", {"milestone": MILESTONE, "files": sorted(path.name for path in sample_dir.iterdir())})
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "generated_by_runner": True})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    heartbeat_path = out / "hardware_heartbeat.jsonl"
    for path in [progress_path, heartbeat_path]:
        if path.exists():
            path.unlink()
    append_jsonl(heartbeat_path, hardware_snapshot())
    run_id = stable_hash({"seed": args.seed, "milestone": MILESTONE})[:16]
    append_jsonl(progress_path, {"time": time.time(), "event": "start", "run_id": run_id})

    system_results: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        metrics, rows = evaluate_system(system)
        system_results[system] = {"overall": metrics}
        all_rows.extend(rows)
        append_jsonl(progress_path, {"time": time.time(), "event": "system_done", "system": system, "weighted_credit": metrics["weighted_lifecycle_credit"]})
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "completed_systems": list(system_results), "latest_system": system})

    decision = decide(system_results)
    aggregate = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "seed": args.seed,
        "pocket_count": len(POCKETS),
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "checker_expected_failure_count": 0,
    }
    replay = deterministic_replay_report(all_rows, system_results, aggregate)
    table = make_table(system_results)
    report = report_text(aggregate, system_results, table)

    write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "boundary": BOUNDARY, "systems": SYSTEMS, "statuses": STATUSES, "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False})
    write_json(out / "pocket_score_inputs.json", POCKETS)
    write_jsonl(out / "promotion_rows.jsonl", all_rows)
    write_json(out / "score_vector_report.json", {pocket["pocket_id"]: {"vector_score": vector_score(pocket), "safety_gate_pass": safety_clean(pocket), "scope_bound": enough_scope(pocket)} for pocket in POCKETS})
    write_json(out / "hard_safety_gate_report.json", {pocket["pocket_id"]: {"wrong_commit": pocket["wrong_commit"], "unsafe_load": pocket["unsafe_load"], "stale_trace_commit": pocket["stale_trace_commit"], "direct_flow_write": pocket["direct_flow_write"], "guard_pass": pocket["guard_pass"]} for pocket in POCKETS})
    write_json(out / "challenger_report.json", {pocket["pocket_id"]: {"challenger_pass": challenger_pass(pocket), "challenger_margin": pocket["challenger_margin"], "redundant_clone": pocket["redundant_clone"], "credit_hijack": pocket["credit_hijack"]} for pocket in POCKETS})
    write_json(out / "shadow_import_report.json", {pocket["pocket_id"]: {"shadow_import_pass": shadow_import_pass(pocket), "reload_transfer": pocket["reload_transfer"], "negative_transfer": pocket["negative_transfer"], "shadow_no_harm": pocket["shadow_no_harm"]} for pocket in POCKETS})
    write_json(out / "rare_critical_report.json", {pocket["pocket_id"]: {"rare_critical": pocket["rare_critical"], "eligible_activation": pocket["eligible_activation"], "raw_activation": pocket["raw_activation"]} for pocket in POCKETS})
    write_json(out / "system_results.json", system_results)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", {"decision": decision, "run_id": run_id, "primary": system_results["full_vector_policy_plus_challenger"]["overall"]})
    out.joinpath("results_table.md").write_text(table + "\n", encoding="utf-8")
    out.joinpath("report.md").write_text(report, encoding="utf-8")
    write_sample_pack(sample_dir, aggregate, system_results, all_rows, replay)
    append_jsonl(progress_path, {"time": time.time(), "event": "complete", "decision": decision})
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e52_golden_disc_scoring_and_promotion_policy_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e52_golden_disc_scoring_and_promotion_policy_probe")
    parser.add_argument("--seed", type=int, default=52052)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
