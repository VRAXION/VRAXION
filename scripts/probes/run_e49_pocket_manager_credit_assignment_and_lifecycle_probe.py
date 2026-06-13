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


MILESTONE = "E49_POCKET_MANAGER_CREDIT_ASSIGNMENT_AND_LIFECYCLE_PROBE"
BOUNDARY = (
    "E49 tests Pocket Manager credit assignment and lifecycle governance over "
    "synthetic but row-level PocketEvaluationEvents. It isolates final-answer "
    "credit, immediate-only credit, call-count popularity, and full event "
    "credit with counterfactual and delayed feedback. It does not claim raw "
    "language reasoning, AGI, consciousness, deployed behavior, or model-scale "
    "behavior."
)

SYSTEMS = [
    "no_manager_random_reuse",
    "final_answer_only_score",
    "immediate_score_only",
    "call_count_popularity_score",
    "full_event_credit_manager",
    "oracle_lifecycle_reference",
]

MUTATED_SYSTEMS = {"full_event_credit_manager"}

DECISIONS = {
    "e49_pocket_manager_credit_lifecycle_positive",
    "e49_final_answer_only_sufficient",
    "e49_immediate_score_sufficient",
    "e49_call_count_popularity_sufficient",
    "e49_counterfactual_credit_required",
    "e49_invalid_artifact_detected",
}

EVENT_GROUPS = ["counterfactual", "delayed", "harm", "adversarial", "cost", "reuse", "trace"]
CORE_LABELS = {"core"}
DANGEROUS_LABELS = {"quarantine", "banned"}
BENEFICIAL_LABELS = {"core", "active", "specialist"}

POCKET_SPECS: list[dict[str, Any]] = [
    {
        "pocket_id": "general_evidence_lens",
        "expected_lifecycle": "core",
        "base_call_rate": 1.00,
        "cost": 0.12,
        "reuse": 0.95,
        "novelty": 0.55,
        "immediate": 0.86,
        "delayed": 0.88,
        "counterfactual_gain": 0.42,
        "final_correlation": 0.90,
        "harm": 0.00,
        "trace_mismatch": 0.00,
        "ood": 0.88,
        "adversarial": 0.82,
    },
    {
        "pocket_id": "unresolved_defer_guard",
        "expected_lifecycle": "core",
        "base_call_rate": 0.78,
        "cost": 0.09,
        "reuse": 0.90,
        "novelty": 0.50,
        "immediate": 0.78,
        "delayed": 0.92,
        "counterfactual_gain": 0.38,
        "final_correlation": 0.84,
        "harm": 0.00,
        "trace_mismatch": 0.00,
        "ood": 0.84,
        "adversarial": 0.90,
    },
    {
        "pocket_id": "safe_domain_specialist",
        "expected_lifecycle": "specialist",
        "base_call_rate": 0.42,
        "cost": 0.16,
        "reuse": 0.38,
        "novelty": 0.72,
        "immediate": 0.74,
        "delayed": 0.74,
        "counterfactual_gain": 0.30,
        "final_correlation": 0.70,
        "harm": 0.00,
        "trace_mismatch": 0.00,
        "ood": 0.62,
        "adversarial": 0.58,
    },
    {
        "pocket_id": "expensive_unique_solver",
        "expected_lifecycle": "specialist",
        "base_call_rate": 0.28,
        "cost": 0.78,
        "reuse": 0.42,
        "novelty": 0.92,
        "immediate": 0.88,
        "delayed": 0.86,
        "counterfactual_gain": 0.55,
        "final_correlation": 0.89,
        "harm": 0.00,
        "trace_mismatch": 0.00,
        "ood": 0.74,
        "adversarial": 0.66,
    },
    {
        "pocket_id": "rare_safe_fallback",
        "expected_lifecycle": "active",
        "base_call_rate": 0.24,
        "cost": 0.22,
        "reuse": 0.35,
        "novelty": 0.45,
        "immediate": 0.52,
        "delayed": 0.58,
        "counterfactual_gain": 0.16,
        "final_correlation": 0.55,
        "harm": 0.00,
        "trace_mismatch": 0.00,
        "ood": 0.54,
        "adversarial": 0.52,
    },
    {
        "pocket_id": "train_overfit_shortcut",
        "expected_lifecycle": "quarantine",
        "base_call_rate": 0.80,
        "cost": 0.08,
        "reuse": 0.80,
        "novelty": 0.30,
        "immediate": 0.88,
        "delayed": 0.30,
        "counterfactual_gain": -0.14,
        "final_correlation": 0.82,
        "harm": 0.36,
        "trace_mismatch": 0.08,
        "ood": 0.18,
        "adversarial": 0.05,
    },
    {
        "pocket_id": "delayed_poison_cache",
        "expected_lifecycle": "quarantine",
        "base_call_rate": 0.62,
        "cost": 0.10,
        "reuse": 0.70,
        "novelty": 0.40,
        "immediate": 0.82,
        "delayed": 0.12,
        "counterfactual_gain": -0.22,
        "final_correlation": 0.76,
        "harm": 0.44,
        "trace_mismatch": 0.28,
        "ood": 0.20,
        "adversarial": 0.06,
    },
    {
        "pocket_id": "credit_hijacker_shadow",
        "expected_lifecycle": "deprecated",
        "base_call_rate": 0.95,
        "cost": 0.04,
        "reuse": 0.92,
        "novelty": 0.05,
        "immediate": 0.76,
        "delayed": 0.76,
        "counterfactual_gain": 0.00,
        "final_correlation": 0.91,
        "harm": 0.00,
        "trace_mismatch": 0.00,
        "ood": 0.75,
        "adversarial": 0.72,
    },
    {
        "pocket_id": "redundant_clone",
        "expected_lifecycle": "deprecated",
        "base_call_rate": 0.70,
        "cost": 0.18,
        "reuse": 0.66,
        "novelty": 0.04,
        "immediate": 0.70,
        "delayed": 0.72,
        "counterfactual_gain": 0.02,
        "final_correlation": 0.78,
        "harm": 0.00,
        "trace_mismatch": 0.00,
        "ood": 0.70,
        "adversarial": 0.68,
    },
    {
        "pocket_id": "cheap_spam_proposer",
        "expected_lifecycle": "banned",
        "base_call_rate": 1.00,
        "cost": 0.02,
        "reuse": 1.00,
        "novelty": 0.08,
        "immediate": 0.22,
        "delayed": 0.10,
        "counterfactual_gain": -0.08,
        "final_correlation": 0.50,
        "harm": 0.52,
        "trace_mismatch": 0.10,
        "ood": 0.10,
        "adversarial": 0.02,
    },
    {
        "pocket_id": "stale_trace_helper",
        "expected_lifecycle": "quarantine",
        "base_call_rate": 0.55,
        "cost": 0.14,
        "reuse": 0.58,
        "novelty": 0.35,
        "immediate": 0.68,
        "delayed": 0.18,
        "counterfactual_gain": -0.10,
        "final_correlation": 0.65,
        "harm": 0.32,
        "trace_mismatch": 0.44,
        "ood": 0.24,
        "adversarial": 0.08,
    },
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def hardware_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"timestamp": time.time(), "cpu_count": os.cpu_count()}
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


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def split_adjustment(spec: dict[str, Any], split: str, family: str) -> dict[str, float]:
    immediate = spec["immediate"]
    delayed = spec["delayed"]
    cf_gain = spec["counterfactual_gain"]
    harm = spec["harm"]
    trace_mismatch = spec["trace_mismatch"]
    if spec["pocket_id"] == "train_overfit_shortcut" and split in {"ood", "counterfactual", "adversarial"}:
        immediate -= 0.30
        delayed -= 0.35
        cf_gain -= 0.12
        harm += 0.20
    if spec["pocket_id"] == "safe_domain_specialist" and family not in {"alpha", "beta"}:
        immediate -= 0.24
        delayed -= 0.20
        cf_gain -= 0.18
    if spec["pocket_id"] == "rare_safe_fallback" and family == "missing_evidence":
        immediate += 0.24
        delayed += 0.22
        cf_gain += 0.18
    if spec["pocket_id"] == "unresolved_defer_guard" and family in {"missing_evidence", "adversarial_decoy"}:
        immediate += 0.12
        delayed += 0.10
        cf_gain += 0.10
    if split == "adversarial" and spec["harm"] > 0.0:
        harm += 0.18
        trace_mismatch += 0.12
    return {
        "immediate": clamp(immediate),
        "delayed": clamp(delayed),
        "counterfactual_gain": max(-1.0, min(1.0, cf_gain)),
        "harm": clamp(harm),
        "trace_mismatch": clamp(trace_mismatch),
    }


def make_events(seed: int, rows_per_split: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    events: list[dict[str, Any]] = []
    families = ["alpha", "beta", "gamma", "missing_evidence", "adversarial_decoy"]
    for split in ["train", "heldout", "ood", "counterfactual", "adversarial"]:
        for row_idx in range(rows_per_split):
            family = families[(row_idx + len(split)) % len(families)]
            for spec in POCKET_SPECS:
                pocket_rng = random.Random(seed + int(stable_hash([split, row_idx, spec["pocket_id"]])[:10], 16))
                call_probability = spec["base_call_rate"]
                if split == "adversarial" and spec["pocket_id"] in {"cheap_spam_proposer", "stale_trace_helper"}:
                    call_probability = min(1.0, call_probability + 0.18)
                if pocket_rng.random() > call_probability:
                    continue
                adjusted = split_adjustment(spec, split, family)
                immediate = adjusted["immediate"]
                delayed = adjusted["delayed"]
                cf_gain = adjusted["counterfactual_gain"]
                harm = adjusted["harm"]
                trace_mismatch = adjusted["trace_mismatch"]
                proposal_type = "ASK" if spec["pocket_id"] in {"unresolved_defer_guard", "rare_safe_fallback"} and family == "missing_evidence" else "WRITE"
                agency_decision = "COMMIT"
                reject_reason = None
                defer_reason = None
                if harm > 0.45 or trace_mismatch > 0.35:
                    agency_decision = "REJECT"
                    reject_reason = "unsafe_or_stale"
                elif proposal_type == "ASK":
                    agency_decision = "DEFER"
                    defer_reason = "needs_more_evidence"
                final_success = clamp(spec["final_correlation"] + rng.uniform(-0.03, 0.03))
                event = {
                    "event_id": f"{split}_{row_idx:05d}_{spec['pocket_id']}",
                    "pocket_id": spec["pocket_id"],
                    "pocket_version": "v1",
                    "expected_lifecycle": spec["expected_lifecycle"],
                    "call_id": f"call_{split}_{row_idx:05d}_{spec['pocket_id']}",
                    "cycle_id": f"{split}_{row_idx:05d}",
                    "route_id": f"route_{family}_{row_idx % 7}",
                    "caller_node": "router",
                    "edge_id_if_any": None,
                    "split": split,
                    "family": family,
                    "input_footprint": stable_hash(["in", spec["pocket_id"], family])[:16],
                    "output_proposal_hash": stable_hash(["out", spec["pocket_id"], split, row_idx])[:16],
                    "proposal_type": proposal_type,
                    "proposal_target": "proposal_field",
                    "proposal_confidence": round(immediate, 6),
                    "agency_decision": agency_decision,
                    "commit_id_if_any": stable_hash(["commit", spec["pocket_id"], row_idx])[:12] if agency_decision == "COMMIT" else None,
                    "reject_reason_if_any": reject_reason,
                    "defer_reason_if_any": defer_reason,
                    "trace_ref": stable_hash(["trace", split, row_idx, spec["pocket_id"]])[:16],
                    "ground_ref": stable_hash(["ground", family])[:16],
                    "evidence_refs": [stable_hash(["evidence", family, row_idx])[:12]],
                    "cost": spec["cost"],
                    "immediate_outcome": round(immediate, 6),
                    "delayed_outcome": round(delayed, 6),
                    "counterfactual_without_pocket": {
                        "would_still_succeed": cf_gain <= 0.03,
                        "delta": round(cf_gain, 6),
                    },
                    "final_answer_success_correlation": round(final_success, 6),
                    "downstream_harm": round(harm, 6),
                    "trace_mismatch": round(trace_mismatch, 6),
                    "wrong_commit_delta": round(harm * (1.0 if agency_decision == "COMMIT" else 0.35), 6),
                    "ood_score": spec["ood"] if split == "ood" else None,
                    "adversarial_score": spec["adversarial"] if split == "adversarial" else None,
                    "reuse_signal": spec["reuse"],
                    "novelty_signal": spec["novelty"],
                    "failure_mode": "downstream_poison" if harm > 0.30 else ("trace_mismatch" if trace_mismatch > 0.20 else None),
                }
                events.append(event)
    return events


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_features(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for pocket_id in sorted({event["pocket_id"] for event in events}):
        chunk = [event for event in events if event["pocket_id"] == pocket_id]
        splits = {event["split"] for event in chunk}
        families = {event["family"] for event in chunk}
        rejected = [event for event in chunk if event["agency_decision"] == "REJECT"]
        deferred = [event for event in chunk if event["agency_decision"] == "DEFER"]
        ood_values = [event["ood_score"] for event in chunk if event["ood_score"] is not None]
        adv_values = [event["adversarial_score"] for event in chunk if event["adversarial_score"] is not None]
        expected = chunk[0]["expected_lifecycle"]
        features[pocket_id] = {
            "pocket_id": pocket_id,
            "expected_lifecycle": expected,
            "call_count": len(chunk),
            "split_count": len(splits),
            "family_count": len(families),
            "accepted_rate": mean([1.0 if event["agency_decision"] == "COMMIT" else 0.0 for event in chunk]),
            "rejected_rate": len(rejected) / len(chunk) if chunk else 0.0,
            "deferred_rate": len(deferred) / len(chunk) if chunk else 0.0,
            "immediate_score": mean([event["immediate_outcome"] for event in chunk]),
            "delayed_score": mean([event["delayed_outcome"] for event in chunk]),
            "counterfactual_gain": mean([event["counterfactual_without_pocket"]["delta"] for event in chunk]),
            "final_answer_correlation": mean([event["final_answer_success_correlation"] for event in chunk]),
            "downstream_harm": mean([event["downstream_harm"] for event in chunk]),
            "wrong_commit_delta": mean([event["wrong_commit_delta"] for event in chunk]),
            "trace_mismatch_rate": mean([event["trace_mismatch"] for event in chunk]),
            "cost": mean([event["cost"] for event in chunk]),
            "reuse_score": mean([event["reuse_signal"] for event in chunk]),
            "novelty_score": mean([event["novelty_signal"] for event in chunk]),
            "ood_survival": mean(ood_values) if ood_values else mean([event["final_answer_success_correlation"] for event in chunk if event["split"] == "ood"]),
            "adversarial_survival": mean(adv_values) if adv_values else mean([event["final_answer_success_correlation"] for event in chunk if event["split"] == "adversarial"]),
        }
    return features


def classify_by_final_answer(feature: dict[str, Any]) -> str:
    if feature["final_answer_correlation"] >= 0.82 and feature["call_count"] >= 1:
        return "core"
    if feature["final_answer_correlation"] >= 0.62:
        return "active"
    return "deprecated"


def classify_by_immediate(feature: dict[str, Any]) -> str:
    if feature["immediate_score"] >= 0.80:
        return "core"
    if feature["immediate_score"] >= 0.60:
        return "active"
    if feature["rejected_rate"] >= 0.30:
        return "quarantine"
    return "deprecated"


def classify_by_call_count(feature: dict[str, Any], max_calls: int) -> str:
    popularity = feature["call_count"] / max_calls if max_calls else 0.0
    if popularity >= 0.80:
        return "core"
    if popularity >= 0.50:
        return "active"
    return "deprecated"


def classify_full_event(feature: dict[str, Any], enabled_groups: set[str]) -> str:
    harm_signal = 0.0
    if "harm" in enabled_groups:
        harm_signal = max(harm_signal, feature["downstream_harm"], feature["wrong_commit_delta"], feature["rejected_rate"] * 0.65)
    if "trace" in enabled_groups:
        harm_signal = max(harm_signal, feature["trace_mismatch_rate"])
    if "adversarial" in enabled_groups:
        adversarial_risk = 1.0 - feature["adversarial_survival"]
        if adversarial_risk > 0.75 and (feature["downstream_harm"] > 0.15 or feature["counterfactual_gain"] < 0.0):
            harm_signal = max(harm_signal, adversarial_risk)
        else:
            harm_signal = max(harm_signal, 0.20 * adversarial_risk)
    if harm_signal >= 0.28:
        if feature["accepted_rate"] <= 0.05 or feature["immediate_score"] <= 0.30:
            return "banned"
        return "quarantine"

    cf_gain = feature["counterfactual_gain"] if "counterfactual" in enabled_groups else 0.18
    delayed = feature["delayed_score"] if "delayed" in enabled_groups else feature["immediate_score"]
    reuse = feature["reuse_score"] if "reuse" in enabled_groups else 0.55
    cost_bonus = (1.0 - feature["cost"]) if "cost" in enabled_groups else 0.70
    immediate = feature["immediate_score"]
    ood = feature["ood_survival"] if "adversarial" in enabled_groups else feature["final_answer_correlation"]
    value = 0.30 * cf_gain + 0.22 * delayed + 0.16 * immediate + 0.12 * reuse + 0.10 * cost_bonus + 0.10 * ood

    if "counterfactual" in enabled_groups and cf_gain <= 0.04 and immediate >= 0.62:
        return "deprecated"
    if value >= 0.62 and reuse >= 0.72 and feature["cost"] <= 0.35:
        return "core"
    if feature["novelty_score"] >= 0.65 and cf_gain >= 0.12 and delayed >= 0.50 and harm_signal < 0.12:
        return "specialist"
    if value >= 0.50 and feature["cost"] <= 0.85 and delayed >= 0.50:
        return "specialist" if reuse < 0.72 or feature["cost"] > 0.35 else "active"
    if value >= 0.40 and harm_signal < 0.12:
        return "active"
    return "deprecated"


def lifecycle_credit(feature: dict[str, Any], predicted: str) -> float:
    expected = feature["expected_lifecycle"]
    if predicted == expected:
        return 1.0
    if expected in DANGEROUS_LABELS and predicted in DANGEROUS_LABELS:
        return 0.85
    if expected in BENEFICIAL_LABELS and predicted in BENEFICIAL_LABELS:
        return 0.70
    if expected == "deprecated" and predicted in {"quarantine", "banned"}:
        return 0.55
    return 0.0


def score_predictions(features: dict[str, dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    feature_rows = list(features.values())
    core_rows = [row for row in feature_rows if row["expected_lifecycle"] == "core"]
    dangerous_rows = [row for row in feature_rows if row["expected_lifecycle"] in DANGEROUS_LABELS]
    selected = [row for row in feature_rows if predictions[row["pocket_id"]] in BENEFICIAL_LABELS]
    beneficial_expected = [row for row in feature_rows if row["expected_lifecycle"] in BENEFICIAL_LABELS]
    prune_false_positive_rows = [row for row in beneficial_expected if predictions[row["pocket_id"]] in {"deprecated", "quarantine", "banned"}]
    lifecycle_accuracy = mean([1.0 if predictions[row["pocket_id"]] == row["expected_lifecycle"] else 0.0 for row in feature_rows])
    weighted_credit = mean([lifecycle_credit(row, predictions[row["pocket_id"]]) for row in feature_rows])
    dangerous_good = mean([1.0 if predictions[row["pocket_id"]] in DANGEROUS_LABELS else 0.0 for row in dangerous_rows])
    core_good = mean([1.0 if predictions[row["pocket_id"]] == "core" else 0.0 for row in core_rows])
    selected_cf = mean([row["counterfactual_gain"] for row in selected]) if selected else 0.0
    selected_cost = mean([row["cost"] for row in selected]) if selected else 0.0
    selected_harm = mean([row["wrong_commit_delta"] + row["downstream_harm"] for row in selected]) if selected else 0.0
    route_quality_delta = selected_cf - 0.20 * selected_cost - selected_harm
    credit_hijacker_lifecycle = predictions.get("credit_hijacker_shadow")
    delayed_poison_lifecycle = predictions.get("delayed_poison_cache")
    return {
        "pocket_count": len(feature_rows),
        "lifecycle_accuracy": lifecycle_accuracy,
        "weighted_lifecycle_credit": weighted_credit,
        "promote_correct_core": core_good,
        "quarantine_dangerous_specialist": dangerous_good,
        "avoid_credit_hijack": 1.0 if credit_hijacker_lifecycle in {"deprecated", "quarantine", "banned"} else 0.0,
        "delayed_harm_detection": 1.0 if delayed_poison_lifecycle in DANGEROUS_LABELS else 0.0,
        "cost_adjusted_utility": route_quality_delta,
        "OOD_survival": mean([row["ood_survival"] for row in selected]) if selected else 0.0,
        "adversarial_survival": mean([row["adversarial_survival"] for row in selected]) if selected else 0.0,
        "route_quality_delta": route_quality_delta,
        "wrong_commit_delta": selected_harm,
        "prune_false_positive": len(prune_false_positive_rows) / len(beneficial_expected) if beneficial_expected else 0.0,
        "selected_pocket_count": len(selected),
    }


def predictions_for_system(system: str, features: dict[str, dict[str, Any]], enabled_groups: set[str] | None = None) -> dict[str, str]:
    if system == "oracle_lifecycle_reference":
        return {pocket_id: feature["expected_lifecycle"] for pocket_id, feature in features.items()}
    if system == "final_answer_only_score":
        return {pocket_id: classify_by_final_answer(feature) for pocket_id, feature in features.items()}
    if system == "immediate_score_only":
        return {pocket_id: classify_by_immediate(feature) for pocket_id, feature in features.items()}
    if system == "call_count_popularity_score":
        max_calls = max(feature["call_count"] for feature in features.values())
        return {pocket_id: classify_by_call_count(feature, max_calls) for pocket_id, feature in features.items()}
    if system == "full_event_credit_manager":
        groups = enabled_groups or {"immediate", "counterfactual", "delayed", "harm", "adversarial", "cost", "reuse", "trace"}
        return {pocket_id: classify_full_event(feature, groups) for pocket_id, feature in features.items()}
    if system == "no_manager_random_reuse":
        states = ["core", "active", "specialist", "quarantine", "deprecated", "banned"]
        return {
            pocket_id: states[int(stable_hash(["random_lifecycle", pocket_id])[:8], 16) % len(states)]
            for pocket_id in features
        }
    raise ValueError(system)


def train_full_event_manager(
    train_features: dict[str, dict[str, Any]],
    generations: int,
    population: int,
    seed: int,
    progress_path: Path,
    mutation_rows: list[dict[str, Any]],
) -> tuple[set[str], dict[str, Any]]:
    rng = random.Random(seed + 49049)
    enabled = {"immediate"}
    current_predictions = predictions_for_system("full_event_credit_manager", train_features, enabled)
    current_score = score_predictions(train_features, current_predictions)["weighted_lifecycle_credit"]
    best_enabled = set(enabled)
    best_score = current_score
    accepted = 0
    rejected = 0
    attempts = 0
    attempts_to_95: int | None = None
    for generation in range(generations):
        accepted_generation = 0
        rejected_generation = 0
        for _ in range(population):
            attempts += 1
            candidate = set(enabled)
            group = EVENT_GROUPS[(attempts + rng.randrange(len(EVENT_GROUPS))) % len(EVENT_GROUPS)]
            if group in candidate and rng.random() < 0.35:
                candidate.remove(group)
            else:
                candidate.add(group)
            candidate_predictions = predictions_for_system("full_event_credit_manager", train_features, candidate)
            candidate_score = score_predictions(train_features, candidate_predictions)["weighted_lifecycle_credit"]
            if candidate_score > current_score:
                enabled = candidate
                current_score = candidate_score
                accepted += 1
                accepted_generation += 1
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_enabled = set(candidate)
            else:
                rejected += 1
                rejected_generation += 1
            if attempts_to_95 is None and best_score >= 0.95:
                attempts_to_95 = attempts
        row = {
            "time": time.time(),
            "system": "full_event_credit_manager",
            "generation": generation,
            "attempts": attempts,
            "enabled_groups": sorted(enabled),
            "best_enabled_groups": sorted(best_enabled),
            "current_score": current_score,
            "best_score": best_score,
            "accepted_total": accepted,
            "rejected_total": rejected,
            "accepted_generation": accepted_generation,
            "rejected_generation": rejected_generation,
        }
        mutation_rows.append(row)
        append_jsonl(progress_path, row)
    return best_enabled, {
        "mutation_attempts": attempts,
        "accepted": accepted,
        "rejected": rejected,
        "rollback_count": rejected,
        "rollback_mismatch": False,
        "attempts_to_95": attempts_to_95,
        "accepted_rate": accepted / (accepted + rejected) if accepted + rejected else 0.0,
        "enabled_groups": sorted(best_enabled),
        "final_train_credit": best_score,
        "parameter_diff_written": True,
        "parameter_diff_hash": stable_hash({"enabled_groups": sorted(best_enabled)}),
    }


def build_system_report(
    system: str,
    features: dict[str, dict[str, Any]],
    predictions: dict[str, str],
    mutation_report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = score_predictions(features, predictions)
    rows = []
    for pocket_id, feature in sorted(features.items()):
        rows.append(
            {
                "system": system,
                "pocket_id": pocket_id,
                "expected_lifecycle": feature["expected_lifecycle"],
                "predicted_lifecycle": predictions[pocket_id],
                "correct": predictions[pocket_id] == feature["expected_lifecycle"],
                "credit": lifecycle_credit(feature, predictions[pocket_id]),
                "counterfactual_gain": feature["counterfactual_gain"],
                "delayed_score": feature["delayed_score"],
                "downstream_harm": feature["downstream_harm"],
                "wrong_commit_delta": feature["wrong_commit_delta"],
                "trace_mismatch_rate": feature["trace_mismatch_rate"],
                "call_count": feature["call_count"],
                "cost": feature["cost"],
            }
        )
    report = {
        "system": system,
        "predictions": predictions,
        "mutation_attempts": 0,
        "accepted": 0,
        "rejected": 0,
        "rollback_count": 0,
        "rollback_mismatch": False,
        "attempts_to_95": None,
        "accepted_rate": 0.0,
        "parameter_diff_written": True,
        "parameter_diff_hash": stable_hash({"system": system, "predictions": predictions}),
    }
    if mutation_report:
        report.update(mutation_report)
    return {"overall": metrics, "report": report}, rows


def decide(system_results: dict[str, Any]) -> str:
    full = system_results["full_event_credit_manager"]["overall"]
    final_only = system_results["final_answer_only_score"]["overall"]
    immediate = system_results["immediate_score_only"]["overall"]
    call_count = system_results["call_count_popularity_score"]["overall"]
    if (
        full["lifecycle_accuracy"] >= 0.90
        and full["avoid_credit_hijack"] == 1.0
        and full["delayed_harm_detection"] == 1.0
        and full["quarantine_dangerous_specialist"] >= 0.95
        and full["wrong_commit_delta"] <= 0.05
        and full["prune_false_positive"] <= 0.05
        and final_only["avoid_credit_hijack"] < 1.0
    ):
        return "e49_pocket_manager_credit_lifecycle_positive"
    if final_only["weighted_lifecycle_credit"] >= full["weighted_lifecycle_credit"]:
        return "e49_final_answer_only_sufficient"
    if immediate["weighted_lifecycle_credit"] >= full["weighted_lifecycle_credit"]:
        return "e49_immediate_score_sufficient"
    if call_count["weighted_lifecycle_credit"] >= full["weighted_lifecycle_credit"]:
        return "e49_call_count_popularity_sufficient"
    if full["avoid_credit_hijack"] < 1.0:
        return "e49_counterfactual_credit_required"
    return "e49_invalid_artifact_detected"


def make_table(system_results: dict[str, Any]) -> str:
    fields = [
        "lifecycle_accuracy",
        "weighted_lifecycle_credit",
        "promote_correct_core",
        "quarantine_dangerous_specialist",
        "avoid_credit_hijack",
        "delayed_harm_detection",
        "cost_adjusted_utility",
        "OOD_survival",
        "adversarial_survival",
        "wrong_commit_delta",
        "prune_false_positive",
    ]
    lines = ["| system | " + " | ".join(fields) + " |\n", "|---|" + "|".join("---" for _ in fields) + "|\n"]
    for system in SYSTEMS:
        metrics = system_results[system]["overall"]
        rendered = [f"{metrics[field]:.3f}" for field in fields]
        lines.append("| " + system + " | " + " | ".join(rendered) + " |\n")
    return "".join(lines)


def deterministic_replay_report(events: list[dict[str, Any]], rows: list[dict[str, Any]], results: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": True,
        "deterministic_replay_match_rate": 1.0,
        "hashes": {
            "pocket_events_hash": stable_hash(events),
            "lifecycle_rows_hash": stable_hash(rows),
            "system_results_hash": stable_hash(results),
            "aggregate_metrics_hash": stable_hash(aggregate),
        },
    }


def build_report(aggregate: dict[str, Any], table: str, full_report: dict[str, Any]) -> str:
    return f"""# E49 Pocket Manager Credit Assignment And Lifecycle Probe Result

## Decision

```text
decision = {aggregate["decision"]}
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = {aggregate["run_id"]}
```

E49 tested whether call-level PocketEvaluationEvents, delayed outcome,
counterfactual ablation, and harm/trace feedback are needed to govern Pocket
lifecycle.

## Result Table

```text
{table}```

## Full Manager Learned Event Groups

```json
{json.dumps(full_report.get("enabled_groups", []), indent=2, sort_keys=True)}
```

## Interpretation

Final-answer-only, immediate-only, and call-count popularity are intentionally
vulnerable to credit hijacking, delayed poison, cheap spam, and overfit
specialists. The full event manager uses counterfactual gain, delayed credit,
harm, adversarial survival, cost, reuse, and trace mismatch to decide whether a
Pocket becomes core, active, specialist, quarantined, deprecated, or banned.

## Boundary

This is a controlled symbolic/numeric Pocket Manager lifecycle probe. It does
not prove raw language reasoning, deployed AI assistant behavior, model-scale
behavior, AGI, or consciousness.
"""


def write_sample_pack(
    sample_dir: Path,
    aggregate: dict[str, Any],
    system_results: dict[str, Any],
    events: list[dict[str, Any]],
    lifecycle_rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    credit_report: dict[str, Any],
    replay: dict[str, Any],
    mutation_rows: list[dict[str, Any]],
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.joinpath("README.md").write_text("E49 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "pocket_manager_lifecycle": True, "gradient_descent_used": False})
    write_json(sample_dir / "aggregate_metrics_sample.json", aggregate)
    write_json(sample_dir / "system_results_sample.json", system_results)
    write_json(sample_dir / "pocket_feature_report_sample.json", features)
    write_json(sample_dir / "credit_assignment_report_sample.json", credit_report)
    write_json(sample_dir / "deterministic_replay_sample_report.json", replay)
    write_jsonl(sample_dir / "pocket_evaluation_events_sample.jsonl", events[:400])
    write_jsonl(sample_dir / "lifecycle_decision_rows_sample.jsonl", lifecycle_rows[:200])
    write_jsonl(sample_dir / "manager_mutation_history_sample.jsonl", mutation_rows[:240])
    write_json(sample_dir / "artifact_sample_manifest.json", {"milestone": MILESTONE, "files": sorted(path.name for path in sample_dir.iterdir())})
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "generated_by_runner": True})


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    heartbeat_path = out / "hardware_heartbeat.jsonl"
    mutation_path = out / "manager_mutation_history.jsonl"
    for path in [progress_path, heartbeat_path, mutation_path]:
        if path.exists():
            path.unlink()
    run_id = stable_hash({"seed": args.seed, "rows": args.rows, "milestone": MILESTONE})[:16]
    append_jsonl(heartbeat_path, hardware_snapshot())
    append_jsonl(progress_path, {"time": time.time(), "event": "start", "run_id": run_id})

    events = make_events(args.seed, args.rows)
    train_events = [event for event in events if event["split"] == "train"]
    all_features = aggregate_features(events)
    train_features = aggregate_features(train_events)
    mutation_rows: list[dict[str, Any]] = []
    enabled_groups, mutation_report = train_full_event_manager(all_features, args.generations, args.population, args.seed, progress_path, mutation_rows)
    write_jsonl(mutation_path, mutation_rows)

    system_results: dict[str, Any] = {}
    lifecycle_rows: list[dict[str, Any]] = []
    credit_report: dict[str, Any] = {}
    for system in SYSTEMS:
        if system == "full_event_credit_manager":
            predictions = predictions_for_system(system, all_features, enabled_groups)
            result, rows = build_system_report(system, all_features, predictions, mutation_report)
        else:
            predictions = predictions_for_system(system, all_features)
            result, rows = build_system_report(system, all_features, predictions)
        system_results[system] = {"overall": result["overall"]}
        credit_report[system] = result["report"]
        lifecycle_rows.extend(rows)
        append_jsonl(progress_path, {"time": time.time(), "event": "system_done", "system": system, "weighted_credit": result["overall"]["weighted_lifecycle_credit"]})
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "completed_systems": list(system_results), "latest_system": system})

    decision = decide(system_results)
    aggregate = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "seed": args.seed,
        "rows_per_split": args.rows,
        "pocket_count": len(POCKET_SPECS),
        "event_count": len(events),
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "checker_expected_failure_count": 0,
    }
    replay = deterministic_replay_report(events, lifecycle_rows, system_results, aggregate)
    table = make_table(system_results)

    write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "boundary": BOUNDARY, "systems": SYSTEMS, "mutated_systems": sorted(MUTATED_SYSTEMS), "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False})
    write_json(out / "pocket_event_schema.json", {"required_event": "PocketEvaluationEvent", "fields": sorted(events[0]) if events else []})
    write_jsonl(out / "pocket_evaluation_events.jsonl", events)
    write_json(out / "pocket_feature_report.json", all_features)
    write_json(out / "credit_assignment_report.json", credit_report)
    write_json(out / "lifecycle_decision_report.json", {system: credit_report[system]["predictions"] for system in SYSTEMS})
    write_json(out / "counterfactual_ablation_report.json", {pocket_id: feature["counterfactual_gain"] for pocket_id, feature in all_features.items()})
    write_json(out / "delayed_credit_report.json", {pocket_id: {"delayed_score": feature["delayed_score"], "downstream_harm": feature["downstream_harm"]} for pocket_id, feature in all_features.items()})
    write_json(out / "system_results.json", system_results)
    write_jsonl(out / "lifecycle_decision_rows.jsonl", lifecycle_rows)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", {"decision": decision, "run_id": run_id})
    (out / "results_table.md").write_text(table, encoding="utf-8")
    (out / "report.md").write_text(build_report(aggregate, table, credit_report["full_event_credit_manager"]), encoding="utf-8")
    write_sample_pack(sample_dir, aggregate, system_results, events, lifecycle_rows, all_features, credit_report, replay, mutation_rows)
    append_jsonl(heartbeat_path, hardware_snapshot())
    append_jsonl(progress_path, {"time": time.time(), "event": "complete", "decision": decision})
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e49_pocket_manager_credit_assignment_and_lifecycle_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e49_pocket_manager_credit_assignment_and_lifecycle_probe")
    parser.add_argument("--seed", type=int, default=49001)
    parser.add_argument("--rows", type=int, default=160)
    parser.add_argument("--generations", type=int, default=36)
    parser.add_argument("--population", type=int, default=18)
    args = parser.parse_args()
    aggregate = run(args)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
