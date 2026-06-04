#!/usr/bin/env python3
"""Validate D111T trig-periodic targeted repair prototype artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D111T_TRIG_PERIODIC_TARGETED_REPAIR_PROTOTYPE"
REPAIR_TARGET = "phase_aware_recurrent_state_adapter_with_calibration_margin_regularizer"
REPORTS = """d111a_upstream_manifest.json d111t_scale_report.json d111t_trig_baseline_replay_report.json d111t_phase_aliasing_repair_report.json d111t_harmonic_confusion_repair_report.json d111t_top1_top2_margin_repair_report.json d111t_loop_utility_repair_report.json d111t_mask_stability_repair_report.json d111t_calibration_margin_repair_report.json d111t_component_update_report.json d111t_checkpoint_rollback_report.json d111t_lane_a_preservation_report.json d111t_lane_b_preservation_report.json d111t_lane_d_preservation_report.json d111t_integrated_repair_eval_report.json d111t_trig_promotion_gate_report.json d111t_rust_invocation_report.json d111t_label_shuffle_sentinel_report.json d111t_regime_label_leak_sentinel_report.json d111t_family_label_leak_sentinel_report.json d111t_failure_label_shortcut_sentinel_report.json d111t_phase_bin_shortcut_sentinel_report.json d111t_frequency_bin_shortcut_sentinel_report.json d111t_harmonic_overlap_shortcut_sentinel_report.json d111t_row_id_lookup_sentinel_report.json d111t_python_hash_lookup_sentinel_report.json d111t_file_order_artifact_sentinel_report.json d111t_seed_id_shortcut_sentinel_report.json d111t_hidden_state_label_leak_sentinel_report.json d111t_hidden_state_row_lookup_sentinel_report.json d111t_halt_step_shortcut_sentinel_report.json d111t_step_count_shortcut_sentinel_report.json d111t_mask_id_shortcut_sentinel_report.json d111t_sparsity_pattern_shortcut_sentinel_report.json d111t_checkpoint_id_shortcut_sentinel_report.json d111t_component_id_shortcut_sentinel_report.json d111t_adapter_step_id_shortcut_sentinel_report.json d111t_gradient_bucket_id_shortcut_sentinel_report.json d111t_split_integrity_report.json d111t_overfit_memorization_report.json d111t_negative_controls_report.json d111t_truth_leak_oracle_isolation_report.json d111t_report_schema_metric_crosscheck_report.json d111t_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
EXEMPT = {"aggregate_metrics.json", "decision.json", "summary.json", "report.md"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def check(out: Path) -> None:
    require(out.exists(), f"missing artifact dir: {out}")
    missing = [report for report in REPORTS if not (out / report).exists()]
    require(not missing, f"missing reports: {missing}")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    manifest = read_json(out / "d111a_upstream_manifest.json")
    scale = aggregate["scale"]
    m = aggregate["metrics"]
    gates = aggregate["positive_gates"]
    require(decision["decision"] == "d111t_trig_periodic_targeted_repair_prototype_confirmed", "unexpected decision")
    require(decision["next"] == "D112_TRIG_PERIODIC_REPAIR_SCALE_CONFIRM", "unexpected next")
    require(decision["d112_ready"] is True and m["d112_ready"] is True, "D112 readiness missing")
    require(all(gates.values()), f"failed positive gates: {[k for k, v in gates.items() if not v]}")
    require(manifest["validation_status"] == "valid" and manifest["replayed_decision"] == "trig_periodic_failure_localized", "D111A replay invalid")
    require(manifest["replayed_candidate_repair_target"] == REPAIR_TARGET and manifest["replayed_failed_jobs"] == [], "D111A repair target/failures invalid")
    require(m["d111a_replay_validation_passed"] is True and m["d111a_replay_decision"] == "trig_periodic_failure_localized", "D111A metric replay invalid")
    require(scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False, "scale reduced")
    require(scale["requested_total_rows"] == 114000 and scale["stress_mode_count"] == 22 and scale["all_required_stress_modes_executed"] is True, "scale/stress mismatch")
    require(scale["failed_jobs"] == [] and scale["fallback_rows"] == 0, "scale fallback/failure")
    require(m["sparse_candidate_identity_preserved"] is True and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light", "sparse identity mismatch")
    require(m["protected_components_frozen"] is True and m["protected_component_modification_count"] == 0 and m["sparse_mask_frozen"] is True and m["sparse_mask_drift_rate"] <= 0.002, "freeze/sparse drift mismatch")
    require(set(m["trainable_adapter_names"]) == {"recurrent_state_adapter_phase_aware_repair_delta", "calibration_scalar_adapter_margin_delta"}, "trainable adapters mismatch")
    require(m["trig_repair_training_executed"] is True and m["training_updates_executed"] is True and m["total_repair_steps_executed"] > 0 and 1 <= m["epochs_executed"] <= 3, "training execution mismatch")
    require(m["checkpoint_count"] >= 6 and m["failed_checkpoint_count"] == 0 and m["rollback_triggered"] is False and m["final_candidate_selected"] is True, "checkpoint/rollback mismatch")
    require(m["trig_failing_case_rate_after"] < m["trig_failing_case_rate_before"] and m["trig_failure_reduction"] >= 0.20, "trig failure reduction gate failed")
    require(m["trig_loop_utility_delta"] > 0 and m["trig_mask_stability_delta"] > 0 and m["calibration_margin_delta"] > 0, "loop/mask/calibration repair gate failed")
    require(m["phase_aliasing_reduction"] >= 0.15 and m["harmonic_confusion_reduction"] >= 0.10 and m["top1_top2_ambiguity_reduction"] >= 0.10, "phase/harmonic/top1 repair gate failed")
    require(m["worst_seed_before"] == 31404 and m["worst_seed_after"] == 31404 and m["worst_seed_improvement"] > 0, "worst seed repair mismatch")
    require(m["trig_repair_signal_positive"] is True and m["trig_promotion_gate_passed"] is False and m["trig_remains_repair_only"] is True, "trig promotion policy mismatch")
    require(m["lane_a_interference"] <= 0.01 and m["lane_b_interference"] <= 0.01 and m["lane_d_interference"] <= 0.012, "lane preservation interference failed")
    require(m["lane_a_D68_preservation_rate"] == 1.0 and m["lane_a_top1_guard_preserved"] is True and m["lane_a_routing_failure_rows"] == 0, "Lane A guard/D68 failed")
    require(m["lane_b_status_preserved"] is True and m["lane_d_expansion_preserved"] is True, "Lane B/D preservation failed")
    require(m["post_repair_false_confidence_rate"] <= 0.0049 and m["post_repair_rust_path_invoked"] is True and m["post_repair_fallback_rows"] == 0 and m["post_repair_failed_jobs"] == [], "post repair Rust/fallback gate failed")
    require(m["forbidden_feature_detected"] is False and m["route_distillation_label_leak_risk"] is False, "truth leak detected")
    for key in ["failure_label_shortcut_detected", "phase_bin_shortcut_detected", "frequency_bin_shortcut_detected", "harmonic_overlap_shortcut_detected"]:
        require(m[key] is False, f"shortcut detected: {key}")
    for key, value in m.items():
        if key.endswith("_sentinel_accuracy"):
            require(value <= 0.56, f"sentinel too high: {key}={value}")
    require(m["sentinel_collapse_passed"] is True and m["split_integrity_passed"] is True and m["train_test_ood_contamination_detected"] is False and m["memorization_risk_score"] <= 0.10, "sentinel/split/memorization gate failed")
    require(m["deterministic_replay_passed"] is True and m["report_schema_consistency_passed"] is True and m["metric_crosscheck_passed"] is True and m["rust_path_invoked"] is True and m["fallback_rows"] == 0 and m["failed_jobs"] == [], "infrastructure gate failed")
    require(summary["decision"] == decision["decision"] and summary["next"] == decision["next"], "summary decision mismatch")
    require("does not perform natural-language pretraining" in (out / "report.md").read_text(), "boundary missing from report")
    for report in REPORTS:
        if report in EXEMPT:
            continue
        payload = read_json(out / report)
        require(payload.get("task") == TASK, f"wrong task in {report}")
        require(payload.get("passed") is True, f"report did not pass: {report}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "checked_reports": len(REPORTS)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d111t_trig_periodic_targeted_repair_prototype"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()
