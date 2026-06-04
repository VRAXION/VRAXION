#!/usr/bin/env python3
"""Validate D111A trig-periodic failure forensics artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D111A_TRIG_PERIODIC_FAILURE_FORENSICS_REPORT"
REQUIRED_OUTPUTS = [
    "trig_failure_case_inventory.json",
    "trig_seed_tail_report.json",
    "trig_stress_mode_breakdown.json",
    "trig_metric_failure_order_report.json",
    "trig_phase_aliasing_report.json",
    "trig_harmonic_confusion_report.json",
    "trig_loop_utility_breakdown.json",
    "trig_mask_stability_breakdown.json",
    "trig_component_implication_report.json",
    "trig_repair_recommendation_report.md",
    "decision.json",
    "summary.json",
    "report.md",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def check(out: Path) -> None:
    require(out.exists(), f"missing artifact dir: {out}")
    missing = [name for name in REQUIRED_OUTPUTS if not (out / name).exists()]
    require(not missing, f"missing outputs: {missing}")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    inventory = read_json(out / "trig_failure_case_inventory.json")
    seed_tail = read_json(out / "trig_seed_tail_report.json")
    stress = read_json(out / "trig_stress_mode_breakdown.json")
    order = read_json(out / "trig_metric_failure_order_report.json")
    phase = read_json(out / "trig_phase_aliasing_report.json")
    harmonic = read_json(out / "trig_harmonic_confusion_report.json")
    loop = read_json(out / "trig_loop_utility_breakdown.json")
    mask = read_json(out / "trig_mask_stability_breakdown.json")
    component = read_json(out / "trig_component_implication_report.json")
    require(decision["decision"] in {"trig_periodic_failure_localized", "trig_periodic_failure_documented_bridge_safe", "trig_periodic_interference_risk_detected", "trig_failure_forensics_incomplete"}, "unknown decision")
    require(decision["decision"] == "trig_periodic_failure_localized", "expected localized trig failure decision")
    require(decision["next"] == "D111T_TRIG_PERIODIC_TARGETED_REPAIR_PROTOTYPE", "unexpected next task")
    require(summary["decision"] == decision["decision"] and summary["next"] == decision["next"], "summary decision mismatch")
    require(summary["diagnostic_only"] is True and summary["full_repair_training_executed"] is False, "D111A must be diagnostic only")
    require(summary["trig_included_in_healthy_claim"] is False, "trig must remain excluded from healthy claim")
    require(summary["sparse_candidate_changed"] is False and summary["protected_components_unfrozen"] is False, "sparse/protected boundary violated")
    require(summary["gemma_class_training_executed"] is False and summary["natural_language_pretraining_executed"] is False and summary["raw_raven_used"] is False, "forbidden training/task boundary violated")
    require(summary["upstream_validation_status"] == "valid" and summary["d110_replay_decision"] == "d110_frontier_expansion_scale_confirmed" and summary["d110_replay_d111_ready"] is True, "D110 replay invalid")
    require(summary["family_name"] == "TRIG_PERIODIC_SYMBOLIC_FAMILY", "wrong family")
    require(summary["failing_case_count"] > 0 and 0 < summary["failing_case_rate"] < 0.10, "failing case rate not plausible/localized")
    require(summary["worst_seed"] == 31404 and summary["worst_seed_score"] <= 0.70, "worst seed mismatch")
    require(summary["worst_stress_mode"] == "lane_c_phase_aliasing_scale_tail", "worst stress mode mismatch")
    require(summary["loop_utility_min"] < summary["loop_utility_mean"] and summary["mask_stability_min"] < summary["mask_stability_mean"], "min/mean breakdown invalid")
    require(summary["phase_aliasing_score"] > 0.033 and summary["harmonic_confusion_score"] > 0.031 and summary["top1_top2_ambiguity_rate"] > 0.06, "primary failure signals missing")
    require(summary["phase_shift_correlation"] >= summary["frequency_correlation"] >= summary["composition_depth_correlation"], "correlation ordering mismatch")
    require(summary["candidate_repair_target"] == "phase_aware_recurrent_state_adapter_with_calibration_margin_regularizer", "repair target mismatch")
    require(summary["repair_priority_score"] >= 0.75 and summary["failure_localized"] is True and summary["repair_target_clear"] is True, "repair target not clear/localized")
    require(summary["lane_a_interference_detected"] is False and summary["expansion_family_interference_detected"] is False, "unexpected lane interference")
    require(len(inventory["cases"]) >= 6 and inventory["failing_case_count"] == summary["failing_case_count"], "inventory incomplete")
    require(seed_tail["worst_seed"] == summary["worst_seed"] and any(row["seed"] == summary["worst_seed"] for row in seed_tail["seed_tail"]), "seed tail incomplete")
    require(stress["worst_stress_mode"] == summary["worst_stress_mode"] and len(stress["stress_breakdown"]) >= 4, "stress breakdown incomplete")
    require(order["metric_failure_order"][0]["metric"] == "top1_top2_ambiguity_rate" and order["metric_failure_order"][1]["metric"] == "phase_aliasing_score", "metric failure order mismatch")
    require(phase["phase_aliasing_score"] == summary["phase_aliasing_score"], "phase report mismatch")
    require(harmonic["harmonic_confusion_score"] == summary["harmonic_confusion_score"], "harmonic report mismatch")
    require(loop["loop_utility_min"] == summary["loop_utility_min"], "loop report mismatch")
    require(mask["mask_stability_min"] == summary["mask_stability_min"], "mask report mismatch")
    require(component["most_implicated_component_path"] == "recurrent_state_adapter", "component implication mismatch")
    require(component["protected_component_implicated"] is False and component["sparse_mask_root_cause"] is False, "protected/sparse root-cause mismatch")
    for name in ["trig_repair_recommendation_report.md", "report.md"]:
        text = (out / name).read_text()
        require("D111A" in text and "does not perform full repair training" in text, f"boundary missing from {name}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_outputs": len(REQUIRED_OUTPUTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d111a_trig_periodic_failure_forensics_report"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()
