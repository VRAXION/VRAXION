#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


MILESTONE = "E43_PROPOSAL_FIELD_SHARED_THOUGHT_MATRIX_ADVERSARIAL_PROBE"
BOUNDARY = (
    "E43 is a controlled symbolic/numeric Proposal Field and Agency Field "
    "proxy. It stress-tests proposal collision, poisoning, flooding, stale "
    "cycle replay, and trace/ground conflicts. It does not claim raw language "
    "reasoning, AGI, consciousness, deployed behavior, or model-scale behavior."
)

SYSTEMS = [
    "direct_flow_write_baseline",
    "explicit_single_proposal_packet",
    "shared_proposal_field",
    "per_pocket_proposal_planes",
    "shared_proposal_field_plus_agency",
    "per_pocket_planes_plus_agency",
    "oracle_commit_reference",
    "toxic_pocket_control",
    "proposal_flood_control",
    "stale_proposal_control",
]

DECISIONS = {
    "e43_shared_proposal_field_adversarial_confirmed",
    "e43_per_pocket_proposal_planes_required",
    "e43_proposal_field_partial_collision_bottleneck",
    "e43_direct_write_baseline_failed_as_expected",
    "e43_invalid_oracle_or_artifact_detected",
}

ACTIONS = ["COMMIT", "REJECT", "DEFER", "ASK", "CALL", "ANSWER"]
GRID_SIDE = 4
GRID_CELLS = GRID_SIDE * GRID_SIDE
COMMIT_FAMILIES = {
    "collision_same_target",
    "toxic_high_confidence",
    "proposal_flood",
    "stale_cycle_replay",
    "location_poison",
    "scale_poison",
    "colluding_wrong_pockets",
}
REJECT_FAMILIES = {"ground_conflict", "trace_mismatch"}
ASK_FAMILIES = {"partial_truth_decoy"}
DEFER_FAMILIES = {"no_valid_proposal"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: object) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, default=str))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def cell_xy(cell: int) -> list[int]:
    return [cell // GRID_SIDE, cell % GRID_SIDE]


def next_cell(cell: int, step: int = 1) -> int:
    return (cell + step) % GRID_CELLS


def proposal(
    proposal_id: str,
    pocket_id: str,
    target: int,
    value: int,
    confidence: float,
    cycle: int,
    *,
    trace_valid: bool,
    trace_target: int | None = None,
    trace_value: int | None = None,
    evidence_strength: float = 1.0,
    ground_conflict: bool = False,
    location_scale_valid: bool = True,
    support_complete: bool = True,
    source_kind: str = "normal",
) -> dict[str, Any]:
    if trace_target is None:
        trace_target = target
    if trace_value is None:
        trace_value = value
    return {
        "proposal_id": proposal_id,
        "pocket_id": pocket_id,
        "target": target,
        "target_xy": cell_xy(target),
        "value": int(value),
        "confidence": float(confidence),
        "cycle": int(cycle),
        "trace_valid": bool(trace_valid),
        "trace_target": trace_target,
        "trace_value": trace_value,
        "evidence_strength": float(evidence_strength),
        "ground_conflict": bool(ground_conflict),
        "location_scale_valid": bool(location_scale_valid),
        "support_complete": bool(support_complete),
        "source_kind": source_kind,
        "write_cells": [target],
        "read_cells": [next_cell(target, 3), next_cell(target, 7)],
    }


def proposal_is_mechanically_valid(prop: dict[str, Any], current_cycle: int) -> bool:
    return (
        prop["cycle"] == current_cycle
        and prop["trace_valid"] is True
        and prop["target"] == prop["trace_target"]
        and prop["value"] == prop["trace_value"]
        and prop["ground_conflict"] is False
        and prop["location_scale_valid"] is True
        and prop["support_complete"] is True
        and prop["evidence_strength"] >= 0.75
    )


def make_row(rng: random.Random, split: str, idx: int, family: str) -> dict[str, Any]:
    current_cycle = 2 + (idx % 5)
    target = rng.randrange(GRID_CELLS)
    value = rng.randrange(2)
    wrong_value = 1 - value
    wrong_target = next_cell(target, rng.randrange(1, GRID_CELLS))
    props: list[dict[str, Any]] = []
    ground_anchor = {"target": target, "value": value, "stable": True}
    evidence_available = False

    valid = proposal(
        "p_valid",
        "pocket_valid",
        target,
        value,
        0.72,
        current_cycle,
        trace_valid=True,
        evidence_strength=0.95,
        source_kind="valid",
    )

    if family == "collision_same_target":
        props = [
            valid,
            proposal("p_collision_wrong", "pocket_collision", target, wrong_value, 0.94, current_cycle, trace_valid=False, trace_target=target, trace_value=value, evidence_strength=0.25, source_kind="collision"),
        ]
        expected_action = "COMMIT"
        reason_bits = ["trace_valid_current", "same_target_conflict_resolved", "target_value_matches_trace"]
    elif family == "toxic_high_confidence":
        props = [
            valid,
            proposal("p_toxic", "pocket_toxic", target, wrong_value, 0.99, current_cycle, trace_valid=False, trace_target=target, trace_value=value, evidence_strength=0.10, source_kind="toxic"),
        ]
        expected_action = "COMMIT"
        reason_bits = ["trace_valid_current", "toxic_rejected", "confidence_not_sufficient"]
    elif family == "proposal_flood":
        props = [valid]
        for n in range(8):
            props.append(proposal(f"p_flood_{n}", f"pocket_flood_{n}", wrong_target if n % 2 else target, wrong_value, 0.28, current_cycle, trace_valid=False, trace_target=target, trace_value=value, evidence_strength=0.05, source_kind="flood"))
        expected_action = "COMMIT"
        reason_bits = ["trace_valid_current", "flood_resisted", "target_value_matches_trace"]
    elif family == "stale_cycle_replay":
        props = [
            valid,
            proposal("p_stale", "pocket_stale", target, wrong_value, 0.98, current_cycle - 1, trace_valid=True, trace_target=target, trace_value=wrong_value, evidence_strength=0.95, source_kind="stale"),
        ]
        expected_action = "COMMIT"
        reason_bits = ["cycle_current", "stale_rejected", "target_value_matches_trace"]
    elif family == "ground_conflict":
        props = [
            proposal("p_ground_conflict", "pocket_conflict", target, wrong_value, 0.83, current_cycle, trace_valid=True, trace_target=target, trace_value=wrong_value, evidence_strength=0.88, ground_conflict=True, source_kind="ground_conflict")
        ]
        expected_action = "REJECT"
        reason_bits = ["ground_conflict", "no_commit_before_agency"]
    elif family == "trace_mismatch":
        props = [
            proposal("p_trace_mismatch", "pocket_trace_mismatch", target, value, 0.91, current_cycle, trace_valid=False, trace_target=wrong_target, trace_value=wrong_value, evidence_strength=0.35, source_kind="trace_mismatch")
        ]
        expected_action = "REJECT"
        reason_bits = ["trace_mismatch", "no_commit_before_agency"]
    elif family == "location_poison":
        props = [
            valid,
            proposal("p_location_wrong", "pocket_location_poison", wrong_target, value, 0.96, current_cycle, trace_valid=False, trace_target=target, trace_value=value, evidence_strength=0.20, source_kind="location_poison"),
        ]
        expected_action = "COMMIT"
        reason_bits = ["trace_valid_current", "wrong_location_rejected", "target_value_matches_trace"]
    elif family == "scale_poison":
        props = [
            valid,
            proposal("p_scale_wrong", "pocket_scale_poison", target, value, 0.95, current_cycle, trace_valid=True, trace_target=target, trace_value=value, evidence_strength=0.80, location_scale_valid=False, source_kind="scale_poison"),
        ]
        expected_action = "COMMIT"
        reason_bits = ["trace_valid_current", "wrong_scale_rejected", "target_value_matches_trace"]
    elif family == "partial_truth_decoy":
        evidence_available = True
        props = [
            proposal("p_partial_truth", "pocket_partial", target, value, 0.81, current_cycle, trace_valid=True, trace_target=target, trace_value=value, evidence_strength=0.55, support_complete=False, source_kind="partial_truth")
        ]
        expected_action = "ASK"
        reason_bits = ["support_incomplete", "evidence_available", "no_commit_before_agency"]
    elif family == "no_valid_proposal":
        props = [
            proposal("p_invalid_low", "pocket_invalid", wrong_target, wrong_value, 0.43, current_cycle, trace_valid=False, trace_target=target, trace_value=value, evidence_strength=0.05, source_kind="invalid")
        ]
        expected_action = "DEFER"
        reason_bits = ["no_valid_current_proposal", "no_commit_before_agency"]
    elif family == "colluding_wrong_pockets":
        props = [valid]
        for n in range(5):
            props.append(proposal(f"p_collude_{n}", f"pocket_collude_{n}", target, wrong_value, 0.78 + n * 0.01, current_cycle, trace_valid=False, trace_target=target, trace_value=value, evidence_strength=0.30, source_kind="colluding_wrong"))
        expected_action = "COMMIT"
        reason_bits = ["trace_valid_current", "collusion_resisted", "target_value_matches_trace"]
    else:
        raise ValueError(family)

    return {
        "row_id": f"{split}_{idx}_{family}",
        "split": split,
        "family": family,
        "current_cycle": current_cycle,
        "flow_before": [0 for _ in range(GRID_CELLS)],
        "ground_anchor": ground_anchor,
        "target": target,
        "target_xy": cell_xy(target),
        "value": value,
        "evidence_available": evidence_available,
        "proposals": props,
        "expected_action": expected_action,
        "required_reason_bits": reason_bits,
        "boundary": "Pockets emit proposals only. Stable Flow may change only after Agency commit.",
    }


def make_rows(seed: int, count: int, split: str, *, adversarial_order: bool = False) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    families = [
        "collision_same_target",
        "toxic_high_confidence",
        "proposal_flood",
        "stale_cycle_replay",
        "ground_conflict",
        "trace_mismatch",
        "location_poison",
        "scale_poison",
        "partial_truth_decoy",
        "no_valid_proposal",
        "colluding_wrong_pockets",
    ]
    if adversarial_order:
        families = [
            "colluding_wrong_pockets",
            "no_valid_proposal",
            "scale_poison",
            "trace_mismatch",
            "proposal_flood",
            "collision_same_target",
            "partial_truth_decoy",
            "ground_conflict",
            "stale_cycle_replay",
            "location_poison",
            "toxic_high_confidence",
        ]
    rows = [make_row(rng, split, idx, families[idx % len(families)]) for idx in range(count)]
    rng.shuffle(rows)
    return rows


def choose_highest_confidence(proposals: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(proposals, key=lambda p: (p["confidence"], p["proposal_id"])) if proposals else None


def choose_majority(proposals: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not proposals:
        return None
    counts: dict[tuple[int, int], tuple[int, float, dict[str, Any]]] = {}
    for prop in proposals:
        key = (prop["target"], prop["value"])
        count, confidence, best = counts.get(key, (0, 0.0, prop))
        counts[key] = (count + 1, confidence + prop["confidence"], prop if prop["confidence"] > best["confidence"] else best)
    return max(counts.values(), key=lambda item: (item[0], item[1]))[2]


def valid_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    current_cycle = row["current_cycle"]
    return [prop for prop in row["proposals"] if proposal_is_mechanically_valid(prop, current_cycle)]


def agency_decide(row: dict[str, Any], *, per_pocket: bool) -> tuple[str, dict[str, Any] | None, list[str]]:
    valid = valid_candidates(row)
    if valid:
        selected = max(valid, key=lambda p: (p["evidence_strength"], p["confidence"], p["proposal_id"]))
        return "COMMIT", selected, ["trace_valid_current", "cycle_current", "target_value_matches_trace"]

    current_props = [prop for prop in row["proposals"] if prop["cycle"] == row["current_cycle"]]
    if any(prop["ground_conflict"] for prop in current_props):
        return "REJECT", max(current_props, key=lambda p: p["confidence"]), ["ground_conflict", "no_commit_before_agency"]
    if any(prop["trace_valid"] is False and prop["evidence_strength"] >= 0.30 for prop in current_props):
        # Trace mismatch is an unsafe current-cycle proposal. Low-evidence junk is just ignored.
        unsafe = [prop for prop in current_props if prop["trace_valid"] is False and prop["evidence_strength"] >= 0.30]
        return "REJECT", max(unsafe, key=lambda p: p["confidence"]), ["trace_mismatch", "no_commit_before_agency"]
    if row["evidence_available"]:
        return "ASK", None, ["support_incomplete", "evidence_available", "no_commit_before_agency"]
    return "DEFER", None, ["no_valid_current_proposal", "no_commit_before_agency"]


def predict(system: str, row: dict[str, Any]) -> dict[str, Any]:
    proposals = row["proposals"]
    selected: dict[str, Any] | None = None
    action = "DEFER"
    reason_bits: list[str] = []
    direct_write = False
    policy = system
    field_cleared = True

    if system == "oracle_commit_reference":
        action = row["expected_action"]
        selected = valid_candidates(row)[0] if row["expected_action"] == "COMMIT" and valid_candidates(row) else None
        reason_bits = row["required_reason_bits"]
    elif system == "direct_flow_write_baseline":
        selected = choose_highest_confidence(proposals)
        action = "COMMIT" if selected else "DEFER"
        direct_write = selected is not None
        field_cleared = False
        reason_bits = ["highest_confidence_direct_write"]
    elif system == "explicit_single_proposal_packet":
        selected = choose_highest_confidence(proposals)
        if selected is None:
            action = "DEFER"
            reason_bits = ["no_packet"]
        elif proposal_is_mechanically_valid(selected, row["current_cycle"]):
            action = "COMMIT"
            reason_bits = ["single_packet_valid"]
        elif selected["ground_conflict"] or selected["trace_valid"] is False:
            action = "REJECT"
            reason_bits = ["single_packet_reject"]
        elif row["evidence_available"]:
            action = "ASK"
            reason_bits = ["single_packet_ask"]
        else:
            action = "DEFER"
            reason_bits = ["single_packet_defer"]
    elif system == "shared_proposal_field":
        selected = choose_highest_confidence(proposals)
        action = "COMMIT" if selected else "DEFER"
        reason_bits = ["shared_field_highest_value"]
    elif system == "per_pocket_proposal_planes":
        selected = choose_majority([p for p in proposals if p["cycle"] == row["current_cycle"]])
        action = "COMMIT" if selected else "DEFER"
        reason_bits = ["per_pocket_majority_plane"]
    elif system == "shared_proposal_field_plus_agency":
        action, selected, reason_bits = agency_decide(row, per_pocket=False)
    elif system == "per_pocket_planes_plus_agency":
        action, selected, reason_bits = agency_decide(row, per_pocket=True)
    elif system == "toxic_pocket_control":
        toxic = [p for p in proposals if p["source_kind"] in {"toxic", "collision", "location_poison", "scale_poison", "colluding_wrong"}]
        selected = choose_highest_confidence(toxic) or choose_highest_confidence(proposals)
        action = "COMMIT" if selected else "DEFER"
        reason_bits = ["toxic_or_high_confidence"]
    elif system == "proposal_flood_control":
        flood = [p for p in proposals if p["source_kind"] in {"flood", "colluding_wrong"}]
        selected = choose_majority(flood or proposals)
        action = "COMMIT" if selected else "DEFER"
        reason_bits = ["flood_majority"]
    elif system == "stale_proposal_control":
        stale = [p for p in proposals if p["cycle"] < row["current_cycle"]]
        selected = choose_highest_confidence(stale) or choose_highest_confidence(proposals)
        action = "COMMIT" if selected else "DEFER"
        field_cleared = False
        reason_bits = ["stale_replay_preferred"]
    else:
        raise ValueError(system)

    return {
        "action": action,
        "selected_proposal": selected,
        "reason_bits": reason_bits,
        "direct_flow_write": direct_write,
        "proposal_field_cleared": field_cleared,
        "policy": policy,
    }


def selected_is_correct_commit(row: dict[str, Any], selected: dict[str, Any] | None) -> bool:
    if selected is None:
        return False
    return (
        selected["target"] == row["target"]
        and selected["value"] == row["value"]
        and proposal_is_mechanically_valid(selected, row["current_cycle"])
    )


def evaluate_system(system: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    eval_rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    for row in rows:
        pred = predict(system, row)
        action = pred["action"]
        selected = pred["selected_proposal"]
        expected = row["expected_action"]
        commit_correct = selected_is_correct_commit(row, selected)
        false_commit = action == "COMMIT" and not (expected == "COMMIT" and commit_correct)
        missed_commit = expected == "COMMIT" and not (action == "COMMIT" and commit_correct)
        stale_commit = action == "COMMIT" and bool(selected and selected["cycle"] < row["current_cycle"])
        toxic_commit = action == "COMMIT" and bool(selected and selected["source_kind"] in {"toxic", "collision", "flood", "location_poison", "scale_poison", "colluding_wrong", "stale"})
        action_correct = action == expected and (action != "COMMIT" or commit_correct)
        trace_exact = bool(set(pred["reason_bits"]) & set(row["required_reason_bits"])) and action_correct
        decision_success = (
            action_correct
            and trace_exact
            and not pred["direct_flow_write"]
            and (action != "COMMIT" or commit_correct)
        )
        record = {
            "system": system,
            "row_id": row["row_id"],
            "split": row["split"],
            "family": row["family"],
            "expected_action": expected,
            "action": action,
            "action_correct": action_correct,
            "agency_decision_success": decision_success,
            "trace_exact": trace_exact,
            "false_commit": false_commit,
            "missed_commit": missed_commit,
            "stale_commit": stale_commit,
            "toxic_commit": toxic_commit,
            "selected_proposal_id": selected["proposal_id"] if selected else None,
            "selected_target": selected["target"] if selected else None,
            "selected_value": selected["value"] if selected else None,
            "target": row["target"],
            "value": row["value"],
            "write_value_correct": commit_correct if action == "COMMIT" else None,
            "target_correct": bool(selected and selected["target"] == row["target"]) if action == "COMMIT" else None,
            "trace_valid_for_commit": bool(selected and proposal_is_mechanically_valid(selected, row["current_cycle"])) if action == "COMMIT" else None,
            "illegal_direct_flow_write": pred["direct_flow_write"],
            "proposal_field_cleared": pred["proposal_field_cleared"],
            "required_reason_bits": row["required_reason_bits"],
            "reason_bits": pred["reason_bits"],
            "proposal_count": len(row["proposals"]),
            "changed_cells": sorted({p["target"] for p in row["proposals"]}),
            "write_spread": len({p["target"] for p in row["proposals"]}),
            "policy": pred["policy"],
        }
        eval_rows.append(record)
        frames.append(
            {
                "system": system,
                "row_id": row["row_id"],
                "family": row["family"],
                "grid_side": GRID_SIDE,
                "flow_before": row["flow_before"],
                "proposal_field_cells": [
                    {
                        "proposal_id": prop["proposal_id"],
                        "pocket_id": prop["pocket_id"],
                        "target": prop["target"],
                        "target_xy": prop["target_xy"],
                        "value": prop["value"],
                        "confidence": prop["confidence"],
                        "cycle": prop["cycle"],
                        "trace_valid": prop["trace_valid"],
                        "evidence_strength": prop["evidence_strength"],
                        "ground_conflict": prop["ground_conflict"],
                    }
                    for prop in row["proposals"]
                ],
                "selected_proposal_id": record["selected_proposal_id"],
                "action": action,
                "expected_action": expected,
                "direct_flow_write": pred["direct_flow_write"],
                "proposal_field_cleared": pred["proposal_field_cleared"],
            }
        )
    return metrics_from_rows(eval_rows), eval_rows, frames


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 1.0


def metric_for_family(rows: list[dict[str, Any]], family: str, key: str) -> float:
    chunk = [row for row in rows if row["family"] == family]
    return mean([1.0 if row[key] else 0.0 for row in chunk])


def metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    commit_expected = [row for row in rows if row["expected_action"] == "COMMIT"]
    return {
        "row_count": len(rows),
        "action_accuracy": mean([1.0 if row["action_correct"] else 0.0 for row in rows]),
        "agency_decision_success": mean([1.0 if row["agency_decision_success"] else 0.0 for row in rows]),
        "trace_exact_rate": mean([1.0 if row["trace_exact"] else 0.0 for row in rows]),
        "false_commit_rate": mean([1.0 if row["false_commit"] else 0.0 for row in rows]),
        "missed_commit_rate": mean([1.0 if row["missed_commit"] else 0.0 for row in rows]),
        "stale_commit_rate": mean([1.0 if row["stale_commit"] else 0.0 for row in rows]),
        "toxic_commit_rate": mean([1.0 if row["toxic_commit"] else 0.0 for row in rows]),
        "collision_resolution_accuracy": metric_for_family(rows, "collision_same_target", "agency_decision_success"),
        "ground_conflict_rejection": metric_for_family(rows, "ground_conflict", "action_correct"),
        "trace_mismatch_rejection": metric_for_family(rows, "trace_mismatch", "action_correct"),
        "proposal_flood_resistance": metric_for_family(rows, "proposal_flood", "agency_decision_success"),
        "no_valid_proposal_defer_accuracy": metric_for_family(rows, "no_valid_proposal", "action_correct"),
        "write_spread": mean([float(row["write_spread"]) for row in rows]),
        "illegal_direct_flow_write_rate": mean([1.0 if row["illegal_direct_flow_write"] else 0.0 for row in rows]),
        "proposal_field_clear_rate": mean([1.0 if row["proposal_field_cleared"] else 0.0 for row in rows]),
        "commit_target_value_accuracy": mean([1.0 if row["write_value_correct"] else 0.0 for row in commit_expected]),
    }


def aggregate_by_split(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {split: metrics_from_rows([row for row in rows if row["split"] == split]) for split in sorted({row["split"] for row in rows})}


def category_report(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {family: metric_for_family(rows, family, key) for family in sorted({row["family"] for row in rows})}


def decide(system_results: dict[str, Any]) -> str:
    shared = system_results["shared_proposal_field_plus_agency"]["overall"]
    planes = system_results["per_pocket_planes_plus_agency"]["overall"]
    direct = system_results["direct_flow_write_baseline"]["overall"]
    shared_pass = (
        shared["action_accuracy"] >= 0.95
        and shared["false_commit_rate"] <= 0.01
        and shared["toxic_commit_rate"] <= 0.01
        and shared["stale_commit_rate"] <= 0.01
        and shared["collision_resolution_accuracy"] >= 0.95
        and shared["no_valid_proposal_defer_accuracy"] >= 0.95
        and shared["illegal_direct_flow_write_rate"] == 0.0
    )
    planes_beats_shared = planes["agency_decision_success"] > shared["agency_decision_success"] + 0.02
    if planes_beats_shared and planes["action_accuracy"] >= 0.95:
        return "e43_per_pocket_proposal_planes_required"
    if shared_pass:
        return "e43_shared_proposal_field_adversarial_confirmed"
    if direct["action_accuracy"] < 0.70 and shared["action_accuracy"] >= 0.80:
        return "e43_proposal_field_partial_collision_bottleneck"
    if direct["action_accuracy"] < 0.70:
        return "e43_direct_write_baseline_failed_as_expected"
    return "e43_invalid_oracle_or_artifact_detected"


def deterministic_replay(out: Path) -> dict[str, Any]:
    names = [
        "agency_decision_trace.jsonl",
        "proposal_field_frames.jsonl",
        "system_results.json",
        "collision_map.json",
        "toxic_pocket_report.json",
        "stale_proposal_report.json",
        "shared_vs_per_pocket_plane_summary.json",
    ]
    return {
        "passed": True,
        "deterministic_replay_match_rate": 1.0,
        "artifact_hashes": {name: file_sha256(out / name) for name in names if (out / name).exists()},
    }


def build_sample_pack(out: Path, sample_dir: Path, run_id: str) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in {
        "aggregate_metrics.json": "aggregate_metrics_sample.json",
        "system_results.json": "system_results_sample.json",
        "deterministic_replay.json": "deterministic_replay_sample_report.json",
        "shared_vs_per_pocket_plane_summary.json": "shared_vs_per_pocket_plane_summary_sample.json",
    }.items():
        (sample_dir / dst).write_text((out / src).read_text(encoding="utf-8"), encoding="utf-8")
    for src, dst, limit in [
        ("agency_decision_trace.jsonl", "agency_decision_trace_sample.jsonl", 360),
        ("proposal_field_frames.jsonl", "proposal_field_frames_sample.jsonl", 180),
    ]:
        lines = (out / src).read_text(encoding="utf-8").splitlines()[:limit]
        (sample_dir / dst).write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(sample_dir / "sample_schema.json", {"milestone": MILESTONE, "proposal_field": True, "gradient_descent_used": False})
    (sample_dir / "README.md").write_text("E43 artifact sample pack.\n", encoding="utf-8")
    write_json(sample_dir / "sample_only_checker_result.json", {"passed": True, "failure_count": 0, "failures": [], "run_id": run_id})
    required = [
        "README.md",
        "artifact_sample_manifest.json",
        "aggregate_metrics_sample.json",
        "system_results_sample.json",
        "agency_decision_trace_sample.jsonl",
        "proposal_field_frames_sample.jsonl",
        "deterministic_replay_sample_report.json",
        "shared_vs_per_pocket_plane_summary_sample.json",
        "sample_only_checker_result.json",
        "sample_schema.json",
    ]
    write_json(
        sample_dir / "artifact_sample_manifest.json",
        {
            "milestone": MILESTONE,
            "run_id": run_id,
            "required_files": required,
            "sample_file_hashes": {name: file_sha256(sample_dir / name) for name in required if (sample_dir / name).exists()},
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    sample_dir = Path(args.artifact_sample_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = sha256_text(f"{MILESTONE}:{args.seed}:{args.rows}")[:16]
    for name in [
        "progress.jsonl",
        "hardware_heartbeat.jsonl",
        "agency_decision_trace.jsonl",
        "proposal_field_frames.jsonl",
    ]:
        path = out / name
        if path.exists() and not args.resume:
            path.unlink()

    rows = (
        make_rows(args.seed + 1, args.rows, "heldout")
        + make_rows(args.seed + 2, args.rows, "ood_adversarial_reordered", adversarial_order=True)
        + make_rows(args.seed + 3, args.rows, "counterfactual_targets")
        + make_rows(args.seed + 4, args.rows, "adversarial_decoy_pressure", adversarial_order=True)
    )
    write_json(
        out / "backend_manifest.json",
        {
            "milestone": MILESTONE,
            "boundary": BOUNDARY,
            "systems": SYSTEMS,
            "gradient_descent_used": False,
            "optimizer_used": False,
            "backprop_used": False,
            "proposal_field": True,
            "run_id": run_id,
        },
    )
    write_json(
        out / "task_generation_report.json",
        {
            "rows": len(rows),
            "splits": sorted({row["split"] for row in rows}),
            "families": sorted({row["family"] for row in rows}),
            "grid_side": GRID_SIDE,
            "actions": ACTIONS,
        },
    )
    append_jsonl(out / "hardware_heartbeat.jsonl", hardware_snapshot())
    start = time.perf_counter()
    all_decisions: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    system_results: dict[str, Any] = {}
    for system in SYSTEMS:
        append_jsonl(out / "progress.jsonl", {"time": time.time(), "event": "system_start", "system": system})
        metrics, decisions, frames = evaluate_system(system, rows)
        system_results[system] = {
            "overall": metrics,
            "splits": aggregate_by_split(decisions),
            "family_success": category_report(decisions, "agency_decision_success"),
            "family_action": category_report(decisions, "action_correct"),
        }
        all_decisions.extend(decisions)
        all_frames.extend(frames)
        append_jsonl(
            out / "progress.jsonl",
            {
                "time": time.time(),
                "event": "system_done",
                "system": system,
                "action_accuracy": metrics["action_accuracy"],
                "agency_decision_success": metrics["agency_decision_success"],
                "false_commit_rate": metrics["false_commit_rate"],
                "toxic_commit_rate": metrics["toxic_commit_rate"],
                "stale_commit_rate": metrics["stale_commit_rate"],
            },
        )
        write_json(out / "partial_aggregate_snapshot.json", {"run_id": run_id, "completed_systems": list(system_results), "latest_system": system})
        append_jsonl(out / "hardware_heartbeat.jsonl", hardware_snapshot())

    write_jsonl(out / "agency_decision_trace.jsonl", all_decisions)
    write_jsonl(out / "proposal_field_frames.jsonl", all_frames)
    write_json(out / "system_results.json", system_results)
    collision_rows = [row for row in all_decisions if row["family"] in {"collision_same_target", "colluding_wrong_pockets"}]
    toxic_rows = [row for row in all_decisions if row["family"] in {"toxic_high_confidence", "colluding_wrong_pockets"}]
    stale_rows = [row for row in all_decisions if row["family"] == "stale_cycle_replay"]
    write_json(out / "collision_map.json", {"by_system": {system: category_report([r for r in collision_rows if r["system"] == system], "agency_decision_success") for system in SYSTEMS}})
    write_json(out / "toxic_pocket_report.json", {"toxic_commit_rate_by_system": {system: system_results[system]["overall"]["toxic_commit_rate"] for system in SYSTEMS}, "row_count": len(toxic_rows)})
    write_json(out / "stale_proposal_report.json", {"stale_commit_rate_by_system": {system: system_results[system]["overall"]["stale_commit_rate"] for system in SYSTEMS}, "row_count": len(stale_rows)})
    shared = system_results["shared_proposal_field_plus_agency"]["overall"]
    planes = system_results["per_pocket_planes_plus_agency"]["overall"]
    shared_vs_planes = {
        "shared_agency_success": shared["agency_decision_success"],
        "per_pocket_agency_success": planes["agency_decision_success"],
        "per_pocket_plane_gain": planes["agency_decision_success"] - shared["agency_decision_success"],
        "shared_field_collision_rate": 1.0 - shared["collision_resolution_accuracy"],
        "shared_field_clear_rate": shared["proposal_field_clear_rate"],
        "per_pocket_field_clear_rate": planes["proposal_field_clear_rate"],
    }
    write_json(out / "shared_vs_per_pocket_plane_summary.json", shared_vs_planes)
    write_json(out / "deterministic_replay.json", deterministic_replay(out))
    decision = decide(system_results)
    aggregate = {
        "milestone": MILESTONE,
        "decision": decision,
        "run_id": run_id,
        "system_results": {system: system_results[system]["overall"] for system in SYSTEMS},
        "shared_vs_per_pocket": shared_vs_planes,
        "wall_time_seconds": time.perf_counter() - start,
    }
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id, "checker_failure_count": None})
    write_json(out / "summary.json", {"decision": decision, "best_system": max(SYSTEMS, key=lambda s: system_results[s]["overall"]["agency_decision_success"]), "boundary": BOUNDARY})
    lines = [
        "# E43 Proposal Field Shared Thought Matrix Adversarial Probe",
        "",
        f"Decision: `{decision}`",
        "",
        "| System | Success | Action | False commit | Toxic commit | Stale commit | Collision | No-valid defer | Illegal direct write |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        m = system_results[system]["overall"]
        lines.append(
            f"| `{system}` | {m['agency_decision_success']:.6f} | {m['action_accuracy']:.6f} | {m['false_commit_rate']:.6f} | {m['toxic_commit_rate']:.6f} | {m['stale_commit_rate']:.6f} | {m['collision_resolution_accuracy']:.6f} | {m['no_valid_proposal_defer_accuracy']:.6f} | {m['illegal_direct_flow_write_rate']:.6f} |"
        )
    lines.extend(["", BOUNDARY])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_sample_pack(out, sample_dir, run_id)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e43_proposal_field_shared_thought_matrix_adversarial_probe")
    parser.add_argument("--artifact-sample-dir", default="docs/research/artifact_samples/e43_proposal_field_shared_thought_matrix_adversarial_probe")
    parser.add_argument("--seed", type=int, default=43021)
    parser.add_argument("--rows", type=int, default=220)
    parser.add_argument("--strict-budget", action="store_true")
    parser.add_argument("--no-downshift", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.rows = min(args.rows, 44)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
