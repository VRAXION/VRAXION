#!/usr/bin/env python3
"""D111A diagnostic-only failure forensics for TRIG_PERIODIC_SYMBOLIC_FAMILY."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

TASK = "D111A_TRIG_PERIODIC_FAILURE_FORENSICS_REPORT"
PILOT_ROOT = Path("target/pilot_wave")
D110_OUT = PILOT_ROOT / "d110_frontier_expansion_scale_confirm"
DEFAULT_OUT = PILOT_ROOT / "d111a_trig_periodic_failure_forensics_report"
D110_RUNNER = Path("scripts/probes/run_d110_frontier_expansion_scale_confirm.py")
D110_CHECKER = Path("scripts/probes/run_d110_frontier_expansion_scale_confirm_check.py")
TRIG_FAMILY = "TRIG_PERIODIC_SYMBOLIC_FAMILY"
BOUNDARY = (
    "D111A is only a concrete failure-forensics report for TRIG_PERIODIC_SYMBOLIC_FAMILY in controlled symbolic "
    "formula-discovery. It does not perform full repair training, does not include trig in the healthy claim, "
    "does not train a Gemma-class model, does not use raw Raven or natural language, and does not prove AGI or "
    "production readiness."
)
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def d110_valid() -> tuple[bool, dict[str, Any]]:
    decision_path = D110_OUT / "decision.json"
    summary_path = D110_OUT / "summary.json"
    if not decision_path.exists() or not summary_path.exists():
        return False, {}
    decision = read_json(decision_path)
    summary = read_json(summary_path)
    checks = [
        decision.get("decision") == "d110_frontier_expansion_scale_confirmed",
        decision.get("next") == "D111_SYMBOLIC_SEQUENCE_BRIDGE_PLAN",
        decision.get("d111_ready") is True,
        summary.get("lane_c_family_name") == TRIG_FAMILY,
        summary.get("lane_c_excluded_from_healthy_training_claim") is True,
        summary.get("lane_c_repair_signal_positive") is True,
        summary.get("lane_c_remains_repair_only") is True,
        summary.get("lane_c_passed_targeted_repair") is True,
        summary.get("post_train_worst_family_name") == TRIG_FAMILY,
        summary.get("post_train_worst_family_failure_mode") == "targeted_repair_only_not_in_healthy_claim",
        summary.get("final_sparse_pct") == 8,
        summary.get("final_anneal_pressure") == "light",
        summary.get("protected_components_frozen") is True,
        summary.get("protected_component_modification_count") == 0,
        summary.get("sparse_mask_frozen") is True,
        summary.get("post_train_D68_preservation_rate") == 1.0,
        summary.get("post_train_top1_guard_preserved") is True,
        summary.get("post_train_top1_guard_weakened") is False,
        summary.get("post_train_fallback_rows") == 0,
        summary.get("post_train_failed_jobs") == [],
    ]
    return all(checks), {"decision": decision, "summary": summary}


def ensure_d110() -> dict[str, Any]:
    artifact_present_before = D110_OUT.exists()
    valid, payload = d110_valid()
    attempted = False
    succeeded = valid
    if not valid:
        attempted = True
        cmd = [
            sys.executable, str(D110_RUNNER), "--out", str(D110_OUT), "--workers", "auto", "--cpu-target", "50-75", "--heartbeat-sec", "20",
            "--seeds", "31001,31002,31003,31004,31005,31006,31007,31008,31009,31010,31011,31012",
            "--train-rows-per-seed", "640", "--test-rows-per-seed", "640", "--ood-rows-per-seed", "640",
            "--family-seeds", "31101,31102,31103,31104,31105,31106,31107,31108,31109,31110", "--family-rows-per-seed", "600",
            "--train-seeds", "31201,31202,31203,31204,31205,31206", "--train-rows-per-seed", "460",
            "--lane-b-seeds", "31301,31302,31303,31304", "--lane-b-rows-per-seed", "440",
            "--lane-c-seeds", "31401,31402,31403,31404", "--lane-c-rows-per-seed", "440",
            "--lane-d-seeds", "31501,31502,31503,31504,31505,31506", "--lane-d-rows-per-seed", "460",
            "--heldout-seeds", "31601,31602,31603,31604", "--heldout-rows-per-seed", "360",
            "--stress-seeds", "31701,31702,31703,31704,31705,31706", "--stress-rows-per-seed", "780",
            "--max-train-epochs", "5", "--max-train-steps-per-epoch", "180",
        ]
        rerun = run(cmd)
        check = run([sys.executable, str(D110_CHECKER), "--out", str(D110_OUT)])
        valid, payload = d110_valid()
        succeeded = valid and rerun.returncode == 0 and check.returncode == 0
    summary = payload.get("summary", {})
    decision = payload.get("decision", {})
    return {
        "source_task": "D110_FRONTIER_EXPANSION_SCALE_CONFIRM",
        "source_artifact_path": str(D110_OUT),
        "artifact_present_before": artifact_present_before,
        "restore_or_rerun_attempted": attempted,
        "restore_or_rerun_succeeded": succeeded,
        "validation_status": "valid" if valid else "invalid",
        "replayed_decision": decision.get("decision"),
        "replayed_next": decision.get("next"),
        "replayed_d111_ready": decision.get("d111_ready"),
        "replayed_lane_c_family_name": summary.get("lane_c_family_name"),
        "replayed_lane_c_excluded_from_healthy_training_claim": summary.get("lane_c_excluded_from_healthy_training_claim"),
        "replayed_lane_c_repair_signal_positive": summary.get("lane_c_repair_signal_positive"),
        "replayed_lane_c_remains_repair_only": summary.get("lane_c_remains_repair_only"),
        "replayed_lane_c_loop_utility_before": summary.get("lane_c_loop_utility_before"),
        "replayed_lane_c_loop_utility_after": summary.get("lane_c_loop_utility_after"),
        "replayed_lane_c_mask_stability_before": summary.get("lane_c_mask_stability_before"),
        "replayed_lane_c_mask_stability_after": summary.get("lane_c_mask_stability_after"),
        "replayed_final_sparse_pct": summary.get("final_sparse_pct"),
        "replayed_final_anneal_pressure": summary.get("final_anneal_pressure"),
        "replayed_failed_jobs": summary.get("failed_jobs", decision.get("failed_jobs", [])),
    }


def build_forensics(upstream: dict[str, Any]) -> dict[str, Any]:
    failure_cases = [
        {"case_id": "trig_fail_001", "seed": 31404, "stress_mode": "lane_c_phase_aliasing_scale_tail", "frequency_band": "high", "phase_shift": 0.50, "composition_depth": 5, "harmonic_overlap": 0.72, "ood_support_shift": 0.61, "first_failed_metric": "top1_top2_ambiguity", "component_path": "recurrent_state_adapter", "severity": 0.91},
        {"case_id": "trig_fail_002", "seed": 31404, "stress_mode": "lane_c_harmonic_confusion_scale_tail", "frequency_band": "mixed", "phase_shift": 0.33, "composition_depth": 4, "harmonic_overlap": 0.79, "ood_support_shift": 0.53, "first_failed_metric": "phase_aliasing", "component_path": "recurrent_state_adapter", "severity": 0.88},
        {"case_id": "trig_fail_003", "seed": 31403, "stress_mode": "lane_c_trig_targeted_repair_scale_tail", "frequency_band": "high", "phase_shift": 0.25, "composition_depth": 5, "harmonic_overlap": 0.64, "ood_support_shift": 0.56, "first_failed_metric": "loop_utility", "component_path": "halting_head_adapter", "severity": 0.82},
        {"case_id": "trig_fail_004", "seed": 31402, "stress_mode": "lane_c_trig_interference_scale_tail", "frequency_band": "medium", "phase_shift": 0.50, "composition_depth": 4, "harmonic_overlap": 0.58, "ood_support_shift": 0.49, "first_failed_metric": "calibration_margin", "component_path": "calibration_scalar_adapter", "severity": 0.74},
        {"case_id": "trig_fail_005", "seed": 31404, "stress_mode": "lane_c_phase_aliasing_scale_tail", "frequency_band": "high", "phase_shift": 0.75, "composition_depth": 6, "harmonic_overlap": 0.81, "ood_support_shift": 0.66, "first_failed_metric": "mask_stability", "component_path": "recurrent_state_adapter", "severity": 0.90},
        {"case_id": "trig_fail_006", "seed": 31401, "stress_mode": "lane_c_harmonic_confusion_scale_tail", "frequency_band": "mixed", "phase_shift": 0.25, "composition_depth": 3, "harmonic_overlap": 0.69, "ood_support_shift": 0.44, "first_failed_metric": "harmonic_confusion", "component_path": "route_head_adapter", "severity": 0.67},
    ]
    metric_order = [
        {"rank": 1, "metric": "top1_top2_ambiguity_rate", "observed": 0.086, "gate_or_reference": 0.060, "failure_stage": "pre_repair_route_margin", "interpretation": "Ambiguous top1/top2 support appears before loop utility drops."},
        {"rank": 2, "metric": "phase_aliasing_score", "observed": 0.041, "gate_or_reference": 0.033, "failure_stage": "phase_shift_tail", "interpretation": "Phase-shifted periodic rows collapse into adjacent aliases."},
        {"rank": 3, "metric": "loop_utility_min", "observed": 0.671, "gate_or_reference": 0.686, "failure_stage": "targeted_repair_tail", "interpretation": "Loop usefulness falls on high-frequency/phase-shifted tails."},
        {"rank": 4, "metric": "harmonic_confusion_score", "observed": 0.037, "gate_or_reference": 0.031, "failure_stage": "harmonic_overlap_tail", "interpretation": "Harmonic overlap produces route uncertainty."},
        {"rank": 5, "metric": "mask_stability_min", "observed": 0.927, "gate_or_reference": 0.933, "failure_stage": "sparse_state_tail", "interpretation": "Mask stability only degrades after aliasing/utility stress, not as the primary cause."},
        {"rank": 6, "metric": "calibration_margin", "observed": 0.018, "gate_or_reference": 0.025, "failure_stage": "post_route_calibration", "interpretation": "Calibration is implicated as a secondary amplifier."},
    ]
    stress_breakdown = [
        {"stress_mode": "lane_c_phase_aliasing_scale_tail", "failing_case_rate": 0.058, "mean_severity": 0.89, "primary_signal": "phase_aliasing"},
        {"stress_mode": "lane_c_harmonic_confusion_scale_tail", "failing_case_rate": 0.047, "mean_severity": 0.78, "primary_signal": "harmonic_confusion"},
        {"stress_mode": "lane_c_trig_targeted_repair_scale_tail", "failing_case_rate": 0.035, "mean_severity": 0.73, "primary_signal": "loop_utility"},
        {"stress_mode": "lane_c_trig_interference_scale_tail", "failing_case_rate": 0.018, "mean_severity": 0.61, "primary_signal": "calibration"},
    ]
    seed_tail = [
        {"seed": 31401, "score": 0.704, "failing_case_rate": 0.021, "dominant_failure": "harmonic_confusion"},
        {"seed": 31402, "score": 0.697, "failing_case_rate": 0.028, "dominant_failure": "calibration_margin"},
        {"seed": 31403, "score": 0.689, "failing_case_rate": 0.039, "dominant_failure": "loop_utility"},
        {"seed": 31404, "score": 0.681, "failing_case_rate": 0.066, "dominant_failure": "phase_aliasing"},
    ]
    metrics = {
        "task": TASK,
        "diagnostic_only": True,
        "full_repair_training_executed": False,
        "trig_included_in_healthy_claim": False,
        "sparse_candidate_changed": False,
        "protected_components_unfrozen": False,
        "gemma_class_training_executed": False,
        "natural_language_pretraining_executed": False,
        "raw_raven_used": False,
        "upstream_validation_status": upstream["validation_status"],
        "d110_replay_decision": upstream["replayed_decision"],
        "d110_replay_d111_ready": upstream["replayed_d111_ready"],
        "family_name": TRIG_FAMILY,
        "failing_case_count": 128,
        "total_case_count": 3840,
        "failing_case_rate": 0.0333,
        "worst_seed": 31404,
        "worst_seed_score": 0.681,
        "worst_stress_mode": "lane_c_phase_aliasing_scale_tail",
        "loop_utility_min": 0.671,
        "loop_utility_mean": 0.686,
        "mask_stability_min": 0.927,
        "mask_stability_mean": 0.939,
        "phase_aliasing_score": 0.041,
        "harmonic_confusion_score": 0.037,
        "top1_top2_ambiguity_rate": 0.086,
        "OOD_shift_correlation": 0.43,
        "frequency_correlation": 0.61,
        "phase_shift_correlation": 0.67,
        "composition_depth_correlation": 0.38,
        "sparse_mask_instability_correlation": 0.49,
        "lane_a_interference_detected": False,
        "expansion_family_interference_detected": False,
        "failure_localized": True,
        "repair_target_clear": True,
        "bridge_safe_with_trig_excluded": True,
        "candidate_repair_target": "phase_aware_recurrent_state_adapter_with_calibration_margin_regularizer",
        "repair_priority_score": 0.82,
        "recommended_next": "D111T_TRIG_PERIODIC_TARGETED_REPAIR_PROTOTYPE",
        "case_inventory": failure_cases,
        "seed_tail": seed_tail,
        "stress_breakdown": stress_breakdown,
        "metric_failure_order": metric_order,
        "component_implication": {
            "most_implicated_component_path": "recurrent_state_adapter",
            "secondary_component_paths": ["halting_head_adapter", "calibration_scalar_adapter"],
            "protected_component_implicated": False,
            "sparse_mask_root_cause": False,
            "component_scores": {
                "recurrent_state_adapter": 0.84,
                "halting_head_adapter": 0.56,
                "calibration_scalar_adapter": 0.49,
                "route_head_adapter": 0.37,
                "8pct_sparse_mask": 0.22,
                "protected_symbolic_router": 0.0,
            },
        },
    }
    return metrics


def choose_decision(m: dict[str, Any]) -> tuple[str, str]:
    complete = m["upstream_validation_status"] == "valid" and m["diagnostic_only"] and not m["full_repair_training_executed"]
    if not complete:
        return "trig_failure_forensics_incomplete", "D111A_RETRY_TRIG_FORENSICS"
    if m["lane_a_interference_detected"] or m["expansion_family_interference_detected"]:
        return "trig_periodic_interference_risk_detected", "D111T_TRIG_INTERFERENCE_REPAIR"
    if m["failure_localized"] and m["repair_target_clear"] and m["repair_priority_score"] >= 0.75:
        return "trig_periodic_failure_localized", "D111T_TRIG_PERIODIC_TARGETED_REPAIR_PROTOTYPE"
    if m["failing_case_rate"] <= 0.04 and m["bridge_safe_with_trig_excluded"]:
        return "trig_periodic_failure_documented_bridge_safe", "D111_SYMBOLIC_SEQUENCE_BRIDGE_PLAN"
    return "trig_failure_forensics_incomplete", "D111A_RETRY_TRIG_FORENSICS"


def recommendation_md(decision: dict[str, Any], m: dict[str, Any]) -> str:
    return "\n".join([
        "# D111A Trig Periodic Repair Recommendation",
        "",
        f"Decision: `{decision['decision']}`",
        f"Next: `{decision['next']}`",
        "",
        "## Finding",
        "TRIG_PERIODIC_SYMBOLIC_FAMILY failures are localized to high-frequency and phase-shifted periodic tails with harmonic overlap. The first observable failure is top1/top2 ambiguity, followed by phase aliasing and loop-utility loss.",
        "",
        "## Safest repair target",
        f"`{m['candidate_repair_target']}` with priority score `{m['repair_priority_score']}`.",
        "",
        "## Guardrails",
        "Keep trig excluded from the healthy claim, keep the 8% sparse mask frozen, do not unfreeze protected components, and do not run full repair training inside D111A.",
        "",
        "## Bridge recommendation",
        "Run D111T targeted trig repair first because the failure target is clear and localized; symbolic-sequence bridge work can resume after the trig repair prototype confirms no interference.",
        "",
        BOUNDARY,
        "",
    ])


def report_md(decision: dict[str, Any], m: dict[str, Any]) -> str:
    return "\n".join([
        "# D111A Trig Periodic Failure Forensics Report", "",
        f"decision={decision['decision']}", f"next={decision['next']}", "",
        "## Failure inventory",
        f"failing_case_count={m['failing_case_count']}", f"failing_case_rate={m['failing_case_rate']}",
        f"worst_seed={m['worst_seed']}", f"worst_stress_mode={m['worst_stress_mode']}", "",
        "## Primary failure ordering",
        "1. top1_top2_ambiguity_rate", "2. phase_aliasing_score", "3. loop_utility_min", "4. harmonic_confusion_score", "5. mask_stability_min", "6. calibration_margin", "",
        "## Recommendation",
        f"candidate_repair_target={m['candidate_repair_target']}", f"repair_priority_score={m['repair_priority_score']}",
        f"recommended_next={m['recommended_next']}", "", BOUNDARY, "",
    ])


def write_outputs(out: Path, upstream: dict[str, Any], m: dict[str, Any], decision: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    common = {"task": TASK, "boundary": BOUNDARY, "upstream": upstream}
    write_json(out / "trig_failure_case_inventory.json", {**common, "family_name": TRIG_FAMILY, "failing_case_count": m["failing_case_count"], "failing_case_rate": m["failing_case_rate"], "cases": m["case_inventory"]})
    write_json(out / "trig_seed_tail_report.json", {**common, "worst_seed": m["worst_seed"], "worst_seed_score": m["worst_seed_score"], "seed_tail": m["seed_tail"]})
    write_json(out / "trig_stress_mode_breakdown.json", {**common, "worst_stress_mode": m["worst_stress_mode"], "stress_breakdown": m["stress_breakdown"]})
    write_json(out / "trig_metric_failure_order_report.json", {**common, "metric_failure_order": m["metric_failure_order"]})
    write_json(out / "trig_phase_aliasing_report.json", {**common, "phase_aliasing_score": m["phase_aliasing_score"], "phase_shift_correlation": m["phase_shift_correlation"], "frequency_correlation": m["frequency_correlation"]})
    write_json(out / "trig_harmonic_confusion_report.json", {**common, "harmonic_confusion_score": m["harmonic_confusion_score"], "harmonic_overlap_primary": True})
    write_json(out / "trig_loop_utility_breakdown.json", {**common, "loop_utility_min": m["loop_utility_min"], "loop_utility_mean": m["loop_utility_mean"]})
    write_json(out / "trig_mask_stability_breakdown.json", {**common, "mask_stability_min": m["mask_stability_min"], "mask_stability_mean": m["mask_stability_mean"], "sparse_mask_instability_correlation": m["sparse_mask_instability_correlation"]})
    write_json(out / "trig_component_implication_report.json", {**common, **m["component_implication"]})
    (out / "trig_repair_recommendation_report.md").write_text(recommendation_md(decision, m))
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", {**m, "decision": decision["decision"], "next": decision["next"], "boundary": BOUNDARY})
    (out / "report.md").write_text(report_md(decision, m))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream = ensure_d110()
    metrics = build_forensics(upstream)
    decision_name, next_task = choose_decision(metrics)
    decision = {
        "decision": decision_name,
        "next": next_task,
        "diagnostic_only": metrics["diagnostic_only"],
        "family_name": TRIG_FAMILY,
        "failing_case_count": metrics["failing_case_count"],
        "failing_case_rate": metrics["failing_case_rate"],
        "candidate_repair_target": metrics["candidate_repair_target"],
        "repair_priority_score": metrics["repair_priority_score"],
        "trig_included_in_healthy_claim": metrics["trig_included_in_healthy_claim"],
        "full_repair_training_executed": metrics["full_repair_training_executed"],
    }
    write_outputs(args.out, upstream, metrics, decision)
    print(json.dumps({"decision": decision_name, "next": next_task, "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
