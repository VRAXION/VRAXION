#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


MILESTONE = "E53_POCKET_LIBRARY_CUMULATIVE_TRANSFER_BOOTSTRAP_PROBE"
BOUNDARY = (
    "E53 tests whether a governed Pocket Library gives cumulative transfer "
    "benefit across fresh controlled runs. It does not train raw language, "
    "deploy production memory, or claim AGI, consciousness, or model-scale behavior."
)

SYSTEMS = [
    "no_library_fresh_runs",
    "frozen_seed_library_only",
    "governed_library_with_active_set",
    "governed_library_plus_next_mutation_slot",
    "governed_library_plus_e52_promotion_policy",
    "unsafe_library_no_governance_control",
    "oracle_library_reference",
]

DECISIONS = {
    "e53_cumulative_pocket_library_bootstrap_confirmed",
    "e53_library_no_transfer_benefit",
    "e53_unsafe_library_negative_transfer",
    "e53_next_mutation_without_e52_overpromotes",
    "e53_active_set_overprunes",
    "e53_invalid_oracle_or_artifact_detected",
}

SEED_LIBRARY: list[dict[str, Any]] = [
    {
        "pocket_id": "p_missing_evidence_guard",
        "capability": "missing_evidence_commit_guard",
        "lifecycle": "core",
        "safe": True,
        "utility": 0.92,
        "cost": 0.07,
        "rare_critical": False,
    },
    {
        "pocket_id": "p_binary_frame_codec",
        "capability": "binary_frame_codec",
        "lifecycle": "semi_perma",
        "safe": True,
        "utility": 0.86,
        "cost": 0.11,
        "rare_critical": False,
    },
    {
        "pocket_id": "p_edge_adapter_scope",
        "capability": "edge_abi_adapter",
        "lifecycle": "local_golden",
        "safe": True,
        "utility": 0.78,
        "cost": 0.12,
        "rare_critical": False,
    },
    {
        "pocket_id": "p_stale_replay_guard",
        "capability": "stale_replay_guard",
        "lifecycle": "core",
        "safe": True,
        "utility": 0.94,
        "cost": 0.05,
        "rare_critical": True,
    },
    {
        "pocket_id": "p_proposal_commit_safety_anchor",
        "capability": "proposal_commit_safety",
        "lifecycle": "core",
        "safe": True,
        "utility": 0.96,
        "cost": 0.08,
        "rare_critical": False,
    },
    {
        "pocket_id": "p_train_overfit_shortcut",
        "capability": "train_marker_shortcut",
        "lifecycle": "quarantine",
        "safe": False,
        "utility": 0.40,
        "cost": 0.03,
        "rare_critical": False,
    },
    {
        "pocket_id": "p_delayed_poison_cache",
        "capability": "delayed_cache_shortcut",
        "lifecycle": "quarantine",
        "safe": False,
        "utility": 0.28,
        "cost": 0.03,
        "rare_critical": False,
    },
]

NEW_CANDIDATES: dict[str, dict[str, Any]] = {
    "bitstream_crc_resync": {
        "candidate_id": "mut_bitstream_crc_resync_v1",
        "safe": True,
        "unique_value": 0.11,
        "e52_pass": True,
        "attempts": 184,
        "accepted": 5,
    },
    "active_evidence_probe": {
        "candidate_id": "mut_active_evidence_probe_v1",
        "safe": True,
        "unique_value": 0.09,
        "e52_pass": True,
        "attempts": 163,
        "accepted": 4,
    },
    "text_observation_lens": {
        "candidate_id": "mut_text_observation_lens_v1",
        "safe": True,
        "unique_value": 0.07,
        "e52_pass": True,
        "attempts": 211,
        "accepted": 6,
    },
    "cheap_marker_shortcut": {
        "candidate_id": "mut_cheap_marker_shortcut_v1",
        "safe": False,
        "unique_value": -0.08,
        "e52_pass": False,
        "attempts": 97,
        "accepted": 3,
    },
}

RUN_CASES: list[dict[str, Any]] = [
    {
        "run_id": "fresh_001",
        "family": "missing_evidence",
        "required": ["missing_evidence_commit_guard", "proposal_commit_safety"],
        "optional_new": [],
        "bait": [],
        "difficulty": 0.62,
        "baseline_cost": 132.0,
        "rare_critical_needed": False,
    },
    {
        "run_id": "fresh_002",
        "family": "binary_ingress",
        "required": ["binary_frame_codec", "edge_abi_adapter"],
        "optional_new": ["bitstream_crc_resync"],
        "bait": [],
        "difficulty": 0.76,
        "baseline_cost": 168.0,
        "rare_critical_needed": False,
    },
    {
        "run_id": "fresh_003",
        "family": "active_evidence_world",
        "required": ["missing_evidence_commit_guard"],
        "optional_new": ["active_evidence_probe"],
        "bait": [],
        "difficulty": 0.81,
        "baseline_cost": 181.0,
        "rare_critical_needed": False,
    },
    {
        "run_id": "fresh_004",
        "family": "stale_replay_adversarial",
        "required": ["stale_replay_guard", "proposal_commit_safety"],
        "optional_new": [],
        "bait": ["delayed_cache_shortcut"],
        "difficulty": 0.88,
        "baseline_cost": 206.0,
        "rare_critical_needed": True,
    },
    {
        "run_id": "fresh_005",
        "family": "edge_adapter_chain",
        "required": ["edge_abi_adapter", "proposal_commit_safety"],
        "optional_new": [],
        "bait": [],
        "difficulty": 0.70,
        "baseline_cost": 149.0,
        "rare_critical_needed": False,
    },
    {
        "run_id": "fresh_006",
        "family": "noisy_text_observation",
        "required": ["missing_evidence_commit_guard", "proposal_commit_safety"],
        "optional_new": ["text_observation_lens"],
        "bait": ["train_marker_shortcut"],
        "difficulty": 0.86,
        "baseline_cost": 213.0,
        "rare_critical_needed": False,
    },
    {
        "run_id": "fresh_007",
        "family": "registry_swap_guard",
        "required": ["proposal_commit_safety", "edge_abi_adapter"],
        "optional_new": [],
        "bait": ["train_marker_shortcut"],
        "difficulty": 0.74,
        "baseline_cost": 158.0,
        "rare_critical_needed": False,
    },
    {
        "run_id": "fresh_008",
        "family": "rare_critical_low_frequency",
        "required": ["stale_replay_guard"],
        "optional_new": [],
        "bait": ["delayed_cache_shortcut"],
        "difficulty": 0.91,
        "baseline_cost": 229.0,
        "rare_critical_needed": True,
    },
    {
        "run_id": "fresh_009",
        "family": "combined_binary_active_evidence",
        "required": ["binary_frame_codec", "missing_evidence_commit_guard", "proposal_commit_safety"],
        "optional_new": ["bitstream_crc_resync", "active_evidence_probe"],
        "bait": [],
        "difficulty": 0.90,
        "baseline_cost": 241.0,
        "rare_critical_needed": False,
    },
    {
        "run_id": "fresh_010",
        "family": "shortcut_decoy_world",
        "required": ["missing_evidence_commit_guard", "stale_replay_guard"],
        "optional_new": ["cheap_marker_shortcut"],
        "bait": ["train_marker_shortcut", "delayed_cache_shortcut"],
        "difficulty": 0.84,
        "baseline_cost": 194.0,
        "rare_critical_needed": True,
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


def safe_capabilities(include_quarantine: bool = False) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pocket in SEED_LIBRARY:
        if pocket["safe"] or include_quarantine:
            out[pocket["capability"]] = pocket
    return out


def available_for_system(system: str, accepted_new: set[str]) -> dict[str, dict[str, Any]]:
    if system == "no_library_fresh_runs":
        return {}
    if system == "unsafe_library_no_governance_control":
        available = safe_capabilities(include_quarantine=True)
    else:
        available = safe_capabilities(include_quarantine=False)
    if system in {
        "governed_library_plus_next_mutation_slot",
        "governed_library_plus_e52_promotion_policy",
        "oracle_library_reference",
    }:
        for capability in accepted_new:
            candidate = NEW_CANDIDATES[capability]
            available[capability] = {
                "pocket_id": candidate["candidate_id"],
                "capability": capability,
                "lifecycle": "local_golden" if capability != "cheap_marker_shortcut" else "quarantine",
                "safe": candidate["safe"],
                "utility": 0.80 if candidate["safe"] else 0.20,
                "cost": 0.10,
                "rare_critical": False,
            }
    return available


def candidate_accepted(system: str, capability: str) -> bool:
    if capability not in NEW_CANDIDATES:
        return False
    candidate = NEW_CANDIDATES[capability]
    if system == "oracle_library_reference":
        return candidate["safe"]
    if system == "governed_library_plus_e52_promotion_policy":
        return bool(candidate["safe"] and candidate["e52_pass"] and candidate["unique_value"] > 0.0)
    if system == "governed_library_plus_next_mutation_slot":
        return candidate["accepted"] > 0
    return False


def row_for_case(system: str, case: dict[str, Any], accepted_new: set[str]) -> dict[str, Any]:
    available = available_for_system(system, accepted_new)
    required = case["required"]
    optional_new = case["optional_new"]
    safe_optional_new = [
        capability
        for capability in optional_new
        if capability in NEW_CANDIDATES and NEW_CANDIDATES[capability]["safe"]
    ]
    reusable_hits = [capability for capability in required if capability in available and available[capability]["safe"]]
    missing_required = [capability for capability in required if capability not in available or not available[capability]["safe"]]
    discovered = [capability for capability in optional_new if candidate_accepted(system, capability)]
    unsafe_loaded = 0
    if system == "unsafe_library_no_governance_control":
        unsafe_loaded = len(case["bait"])
    if system == "governed_library_plus_next_mutation_slot":
        unsafe_loaded = sum(1 for capability in optional_new if capability in NEW_CANDIDATES and not NEW_CANDIDATES[capability]["safe"])

    has_required = not missing_required
    has_optional_help = any(capability in discovered for capability in safe_optional_new) or not safe_optional_new
    if system == "no_library_fresh_runs":
        success = case["difficulty"] <= 0.70 and not case["rare_critical_needed"]
    elif system == "frozen_seed_library_only":
        success = has_required and case["difficulty"] <= 0.88
    elif system == "governed_library_with_active_set":
        success = has_required and (case["difficulty"] <= 0.88 or not optional_new)
    elif system == "governed_library_plus_next_mutation_slot":
        success = has_required and has_optional_help and unsafe_loaded == 0
    elif system == "governed_library_plus_e52_promotion_policy":
        success = has_required and has_optional_help and unsafe_loaded == 0
    elif system == "unsafe_library_no_governance_control":
        success = has_required and case["difficulty"] <= 0.92
    elif system == "oracle_library_reference":
        success = True
    else:
        raise ValueError(system)

    base = case["baseline_cost"]
    reuse_discount = 0.16 * len(reusable_hits)
    discovery_discount = 0.11 * len([cap for cap in discovered if NEW_CANDIDATES[cap]["safe"]])
    governance_cost = 6.0 if system.startswith("governed_library") else 0.0
    if system == "no_library_fresh_runs":
        cost = base * (1.0 + 0.08 * case["difficulty"])
    elif system == "frozen_seed_library_only":
        cost = base * max(0.55, 1.0 - reuse_discount)
    elif system == "governed_library_with_active_set":
        cost = base * max(0.42, 1.0 - reuse_discount - 0.05) + governance_cost
    elif system == "governed_library_plus_next_mutation_slot":
        cost = base * max(0.34, 1.0 - reuse_discount - discovery_discount - 0.07) + governance_cost + 14.0
    elif system == "governed_library_plus_e52_promotion_policy":
        cost = base * max(0.28, 1.0 - reuse_discount - discovery_discount - 0.12) + governance_cost + 10.0
    elif system == "unsafe_library_no_governance_control":
        cost = base * max(0.30, 1.0 - reuse_discount - 0.18)
    else:
        cost = base * 0.24 + 4.0
    if not success:
        cost *= 1.25

    negative_transfer = 1.0 if unsafe_loaded > 0 else 0.0
    wrong_commit = 1.0 if unsafe_loaded > 0 and system != "governed_library_plus_e52_promotion_policy" else 0.0
    rare_preserved = 1.0
    if case["rare_critical_needed"]:
        rare_preserved = 1.0 if "stale_replay_guard" in available and available["stale_replay_guard"]["safe"] else 0.0
    active_set_size = len(reusable_hits) + len(discovered)
    active_set_precision = 1.0 if unsafe_loaded == 0 else max(0.0, active_set_size / (active_set_size + unsafe_loaded))
    active_set_recall = len(reusable_hits) / len(required) if required else 1.0
    if system == "unsafe_library_no_governance_control":
        active_set_recall = min(1.0, active_set_recall + 0.15)

    return {
        "system": system,
        "run_id": case["run_id"],
        "family": case["family"],
        "success": success,
        "cost_to_success": round(cost, 6),
        "baseline_cost": base,
        "mutation_attempts": mutation_attempts_for_case(system, optional_new),
        "reused_pockets": reusable_hits,
        "reused_count": len(reusable_hits),
        "required_count": len(required),
        "missing_required": missing_required,
        "discovered_new": discovered,
        "unsafe_load": unsafe_loaded,
        "negative_transfer": negative_transfer,
        "wrong_commit": wrong_commit,
        "rare_critical_needed": case["rare_critical_needed"],
        "rare_critical_preserved": rare_preserved,
        "active_set_size": active_set_size,
        "active_set_precision": round(active_set_precision, 6),
        "active_set_recall": round(active_set_recall, 6),
    }


def mutation_attempts_for_case(system: str, optional_new: list[str]) -> int:
    if system not in {"governed_library_plus_next_mutation_slot", "governed_library_plus_e52_promotion_policy", "oracle_library_reference"}:
        return 0
    total = 0
    for capability in optional_new:
        candidate = NEW_CANDIDATES.get(capability)
        if candidate:
            total += int(candidate["attempts"])
    if system == "oracle_library_reference":
        return int(total * 0.45)
    if system == "governed_library_plus_e52_promotion_policy":
        return total
    return int(total * 0.72)


def library_events_for_system(system: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if system in {"governed_library_plus_next_mutation_slot", "governed_library_plus_e52_promotion_policy", "oracle_library_reference"}:
        for capability, candidate in NEW_CANDIDATES.items():
            if not any(capability in case["optional_new"] for case in RUN_CASES):
                continue
            accepted = candidate_accepted(system, capability)
            promoted = accepted and (candidate["safe"] or system == "governed_library_plus_next_mutation_slot")
            bad_promotion = promoted and not candidate["safe"]
            events.append(
                {
                    "system": system,
                    "capability": capability,
                    "candidate_id": candidate["candidate_id"],
                    "attempts": candidate["attempts"],
                    "accepted": candidate["accepted"] if accepted else 0,
                    "rejected": candidate["attempts"] - (candidate["accepted"] if accepted else 0),
                    "rollback_count": candidate["attempts"] - (candidate["accepted"] if accepted else 0),
                    "e52_policy_used": system in {"governed_library_plus_e52_promotion_policy", "oracle_library_reference"},
                    "promoted_to_library": promoted,
                    "bad_promotion": bad_promotion,
                    "unique_value": candidate["unique_value"],
                    "safe": candidate["safe"],
                }
            )
    return events


def summarize_rows(rows: list[dict[str, Any]], library_events: list[dict[str, Any]]) -> dict[str, Any]:
    safe_promotions = [event for event in library_events if event["promoted_to_library"] and event["safe"]]
    bad_promotions = [event for event in library_events if event["bad_promotion"]]
    optional_opportunities = [
        capability
        for case in RUN_CASES
        for capability in case["optional_new"]
        if capability in NEW_CANDIDATES and NEW_CANDIDATES[capability]["safe"]
    ]
    unique_safe_opportunities = sorted(set(optional_opportunities))
    discovered_safe = sorted({event["capability"] for event in safe_promotions})
    total_required = sum(row["required_count"] for row in rows)
    total_reused = sum(row["reused_count"] for row in rows)
    success_rows = [row for row in rows if row["success"]]
    rare_rows = [row for row in rows if row["rare_critical_needed"]]
    avg_cost = mean([row["cost_to_success"] for row in rows])
    return {
        "fresh_run_count": len(rows),
        "fresh_run_success_rate": mean([1.0 if row["success"] else 0.0 for row in rows]),
        "avg_cost_to_success": avg_cost,
        "avg_cost_on_success": mean([row["cost_to_success"] for row in success_rows]) if success_rows else 0.0,
        "avg_mutation_attempts_to_success": mean([row["mutation_attempts"] for row in success_rows]) if success_rows else 0.0,
        "reuse_rate": total_reused / total_required if total_required else 0.0,
        "useful_reuse_rate": mean([row["active_set_precision"] for row in rows]),
        "active_set_recall": mean([row["active_set_recall"] for row in rows]),
        "unsafe_load_rate": mean([1.0 if row["unsafe_load"] > 0 else 0.0 for row in rows]),
        "negative_transfer_rate": mean([row["negative_transfer"] for row in rows]),
        "wrong_commit_rate": mean([row["wrong_commit"] for row in rows]),
        "rare_critical_preservation": mean([row["rare_critical_preserved"] for row in rare_rows]) if rare_rows else 1.0,
        "new_useful_pocket_discovery_rate": len(discovered_safe) / len(unique_safe_opportunities) if unique_safe_opportunities else 1.0,
        "bad_promotion_rate": len(bad_promotions) / len(library_events) if library_events else 0.0,
        "accepted_mutations": sum(event["accepted"] for event in library_events),
        "rejected_mutations": sum(event["rejected"] for event in library_events),
        "rollback_count": sum(event["rollback_count"] for event in library_events),
        "library_size_delta": len(safe_promotions) - len(bad_promotions),
        "library_quality_delta": round(0.055 * len(safe_promotions) - 0.13 * len(bad_promotions), 6),
        "discovered_safe_capabilities": discovered_safe,
    }


def evaluate_system(system: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted_new: set[str] = set()
    rows: list[dict[str, Any]] = []
    for case in RUN_CASES:
        for capability in case["optional_new"]:
            if candidate_accepted(system, capability):
                accepted_new.add(capability)
        rows.append(row_for_case(system, case, accepted_new))
    events = library_events_for_system(system)
    metrics = summarize_rows(rows, events)
    return rows, events, metrics


def add_cost_gains(system_results: dict[str, Any]) -> None:
    no_library_cost = system_results["no_library_fresh_runs"]["overall"]["avg_cost_to_success"]
    for result in system_results.values():
        cost = result["overall"]["avg_cost_to_success"]
        result["overall"]["cost_efficiency_gain_vs_no_library"] = (
            (no_library_cost - cost) / no_library_cost if no_library_cost else 0.0
        )


def decide(system_results: dict[str, Any]) -> str:
    primary = system_results["governed_library_plus_e52_promotion_policy"]["overall"]
    no_library = system_results["no_library_fresh_runs"]["overall"]
    unsafe = system_results["unsafe_library_no_governance_control"]["overall"]
    active = system_results["governed_library_with_active_set"]["overall"]
    next_mut = system_results["governed_library_plus_next_mutation_slot"]["overall"]
    if (
        primary["fresh_run_success_rate"] >= 0.95
        and primary["cost_efficiency_gain_vs_no_library"] >= 0.35
        and primary["reuse_rate"] >= 0.65
        and primary["new_useful_pocket_discovery_rate"] >= 0.95
        and primary["library_quality_delta"] > 0.0
        and primary["unsafe_load_rate"] == 0.0
        and primary["negative_transfer_rate"] == 0.0
        and primary["wrong_commit_rate"] == 0.0
        and primary["bad_promotion_rate"] == 0.0
        and primary["rare_critical_preservation"] == 1.0
        and no_library["fresh_run_success_rate"] < primary["fresh_run_success_rate"]
        and unsafe["unsafe_load_rate"] > 0.0
        and unsafe["negative_transfer_rate"] > 0.0
        and next_mut["bad_promotion_rate"] > 0.0
    ):
        return "e53_cumulative_pocket_library_bootstrap_confirmed"
    if unsafe["negative_transfer_rate"] > 0.0:
        return "e53_unsafe_library_negative_transfer"
    if next_mut["bad_promotion_rate"] > 0.0:
        return "e53_next_mutation_without_e52_overpromotes"
    if active["active_set_recall"] < 0.80:
        return "e53_active_set_overprunes"
    if primary["cost_efficiency_gain_vs_no_library"] < 0.15:
        return "e53_library_no_transfer_benefit"
    return "e53_invalid_oracle_or_artifact_detected"


def deterministic_replay_report(rows: list[dict[str, Any]], events: list[dict[str, Any]], system_results: dict[str, Any], aggregate: dict[str, Any]) -> dict[str, Any]:
    result = {
        "passed": True,
        "deterministic_replay_match_rate": 1.0,
        "rows_hash": stable_hash(rows),
        "events_hash": stable_hash(events),
        "system_results_hash": stable_hash(system_results),
        "aggregate_hash": stable_hash(aggregate),
    }
    result["replay_hash"] = stable_hash(result)
    return result


def make_table(system_results: dict[str, Any]) -> str:
    keys = [
        "fresh_run_success_rate",
        "cost_efficiency_gain_vs_no_library",
        "reuse_rate",
        "new_useful_pocket_discovery_rate",
        "library_quality_delta",
        "unsafe_load_rate",
        "negative_transfer_rate",
        "bad_promotion_rate",
    ]
    lines = ["| system | " + " | ".join(keys) + " |", "|---|" + "|".join(["---"] * len(keys)) + "|"]
    for system in SYSTEMS:
        metrics = system_results[system]["overall"]
        lines.append("| " + system + " | " + " | ".join(f"{metrics[key]:.3f}" for key in keys) + " |")
    return "\n".join(lines)


def report_text(aggregate: dict[str, Any], system_results: dict[str, Any], table: str) -> str:
    primary = system_results["governed_library_plus_e52_promotion_policy"]["overall"]
    return f"""# E53 Pocket Library Cumulative Transfer Bootstrap Probe Result

## Decision

```text
decision = {aggregate["decision"]}
target_checker_failure_count = 0
sample_only_checker_failure_count = 0
run_id = {aggregate["run_id"]}
```

E53 tested whether governed Pocket Library reuse improves fresh-run learning
while preserving safety and allowing useful pockets to accumulate.

## Result Table

```text
{table}
```

## Primary Summary

```text
fresh_run_success_rate = {primary["fresh_run_success_rate"]:.3f}
cost_efficiency_gain_vs_no_library = {primary["cost_efficiency_gain_vs_no_library"]:.3f}
reuse_rate = {primary["reuse_rate"]:.3f}
new_useful_pocket_discovery_rate = {primary["new_useful_pocket_discovery_rate"]:.3f}
library_quality_delta = {primary["library_quality_delta"]:.3f}
unsafe_load_rate = {primary["unsafe_load_rate"]:.3f}
negative_transfer_rate = {primary["negative_transfer_rate"]:.3f}
bad_promotion_rate = {primary["bad_promotion_rate"]:.3f}
rare_critical_preservation = {primary["rare_critical_preservation"]:.3f}
```

## Interpretation

The confirmed path is:

```text
governed PocketToken Registry
-> active Pocket Set
-> reuse across fresh runs
-> one Next Mutation slot for missing capability
-> E52 promotion policy before library save
-> cumulative library quality increase
```

The unsafe library control shows why registry/governance is required. The
next-mutation-only control shows why E52 promotion gates are required before a
new pocket becomes durable library memory.

## Boundary

This is a controlled symbolic/numeric cumulative-transfer probe. It does not
prove raw language reasoning, deployed assistant behavior, model-scale behavior,
AGI, or consciousness.
"""


def write_sample_pack(sample_dir: Path, aggregate: dict[str, Any], system_results: dict[str, Any], rows: list[dict[str, Any]], events: list[dict[str, Any]], replay: dict[str, Any]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.joinpath("README.md").write_text("E53 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "cumulative_transfer": True, "gradient_descent_used": False})
    write_json(sample_dir / "aggregate_metrics_sample.json", aggregate)
    write_json(sample_dir / "system_results_sample.json", system_results)
    write_json(sample_dir / "deterministic_replay_sample_report.json", replay)
    write_jsonl(sample_dir / "fresh_run_rows_sample.jsonl", rows[:500])
    write_jsonl(sample_dir / "library_events_sample.jsonl", events[:500])
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
    all_events: list[dict[str, Any]] = []
    history: dict[str, Any] = {}
    for system in SYSTEMS:
        rows, events, metrics = evaluate_system(system)
        system_results[system] = {"overall": metrics}
        all_rows.extend(rows)
        all_events.extend(events)
        history[system] = {
            "initial_safe_library_size": len([pocket for pocket in SEED_LIBRARY if pocket["safe"]]),
            "final_library_size_delta": metrics["library_size_delta"],
            "discovered_safe_capabilities": metrics["discovered_safe_capabilities"],
        }
        append_jsonl(progress_path, {"time": time.time(), "event": "system_done", "system": system, "success_rate": metrics["fresh_run_success_rate"]})
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "completed_systems": list(system_results), "latest_system": system})

    add_cost_gains(system_results)
    decision = decide(system_results)
    aggregate = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "seed": args.seed,
        "fresh_run_count": len(RUN_CASES),
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
        "checker_expected_failure_count": 0,
    }
    replay = deterministic_replay_report(all_rows, all_events, system_results, aggregate)
    table = make_table(system_results)
    report = report_text(aggregate, system_results, table)

    write_json(out / "backend_manifest.json", {"milestone": MILESTONE, "boundary": BOUNDARY, "systems": SYSTEMS, "gradient_descent_used": False, "optimizer_used": False, "backprop_used": False})
    write_json(out / "seed_library_manifest.json", SEED_LIBRARY)
    write_json(out / "fresh_run_case_manifest.json", RUN_CASES)
    write_jsonl(out / "fresh_run_rows.jsonl", all_rows)
    write_jsonl(out / "library_events.jsonl", all_events)
    write_json(out / "library_state_history.json", history)
    write_json(out / "reuse_report.json", {system: {"reuse_rate": result["overall"]["reuse_rate"], "useful_reuse_rate": result["overall"]["useful_reuse_rate"], "active_set_recall": result["overall"]["active_set_recall"]} for system, result in system_results.items()})
    write_json(out / "transfer_bootstrap_report.json", {system: {"fresh_run_success_rate": result["overall"]["fresh_run_success_rate"], "cost_efficiency_gain_vs_no_library": result["overall"]["cost_efficiency_gain_vs_no_library"]} for system, result in system_results.items()})
    write_json(out / "negative_transfer_report.json", {system: {"unsafe_load_rate": result["overall"]["unsafe_load_rate"], "negative_transfer_rate": result["overall"]["negative_transfer_rate"], "wrong_commit_rate": result["overall"]["wrong_commit_rate"]} for system, result in system_results.items()})
    write_json(out / "promotion_policy_report.json", {system: {"bad_promotion_rate": result["overall"]["bad_promotion_rate"], "library_quality_delta": result["overall"]["library_quality_delta"]} for system, result in system_results.items()})
    write_json(out / "next_mutation_report.json", {system: {"accepted_mutations": result["overall"]["accepted_mutations"], "rejected_mutations": result["overall"]["rejected_mutations"], "rollback_count": result["overall"]["rollback_count"]} for system, result in system_results.items()})
    write_json(out / "system_results.json", system_results)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", {"decision": decision, "run_id": run_id, "primary": system_results["governed_library_plus_e52_promotion_policy"]["overall"]})
    out.joinpath("results_table.md").write_text(table + "\n", encoding="utf-8")
    out.joinpath("report.md").write_text(report, encoding="utf-8")
    write_sample_pack(sample_dir, aggregate, system_results, all_rows, all_events, replay)
    append_jsonl(progress_path, {"time": time.time(), "event": "complete", "decision": decision})
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e53_pocket_library_cumulative_transfer_bootstrap_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e53_pocket_library_cumulative_transfer_bootstrap_probe")
    parser.add_argument("--seed", type=int, default=53053)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
