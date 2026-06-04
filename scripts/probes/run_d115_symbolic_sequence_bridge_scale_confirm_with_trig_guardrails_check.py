#!/usr/bin/env python3
"""Validate D115 symbolic sequence bridge scale-confirm artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TASK = "D115_SYMBOLIC_SEQUENCE_BRIDGE_SCALE_CONFIRM_WITH_TRIG_GUARDRAILS"
REPORTS = """d114_upstream_manifest.json d115_scale_report.json d115_sparse_candidate_identity_report.json d115_bridge_scale_baseline_report.json d115_symbolic_sequence_routing_scale_report.json d115_ordered_rule_chain_scale_report.json d115_interleaved_bridge_scale_report.json d115_language_like_symbolic_command_guarded_scale_report.json d115_symbolic_command_composition_guarded_scale_report.json d115_variable_binding_sequence_guarded_scale_report.json d115_multi_step_instruction_reference_scale_report.json d115_trig_guardrail_scale_report.json d115_lane_a_preservation_scale_report.json d115_lane_b_preservation_scale_report.json d115_lane_d_preservation_scale_report.json d115_integrated_bridge_scale_eval_report.json d115_checkpoint_rollback_report.json d115_adapter_update_report.json d115_rust_invocation_report.json d115_label_shuffle_sentinel_report.json d115_regime_label_leak_sentinel_report.json d115_family_label_leak_sentinel_report.json d115_bridge_task_id_shortcut_sentinel_report.json d115_command_template_id_shortcut_sentinel_report.json d115_grammar_rule_id_shortcut_sentinel_report.json d115_sequence_position_label_shortcut_sentinel_report.json d115_multi_step_instruction_label_shortcut_sentinel_report.json d115_row_id_lookup_sentinel_report.json d115_python_hash_lookup_sentinel_report.json d115_file_order_artifact_sentinel_report.json d115_seed_id_shortcut_sentinel_report.json d115_scale_run_id_shortcut_sentinel_report.json d115_hidden_state_label_leak_sentinel_report.json d115_hidden_state_row_lookup_sentinel_report.json d115_halt_step_shortcut_sentinel_report.json d115_step_count_shortcut_sentinel_report.json d115_mask_id_shortcut_sentinel_report.json d115_sparsity_pattern_shortcut_sentinel_report.json d115_checkpoint_id_shortcut_sentinel_report.json d115_component_id_shortcut_sentinel_report.json d115_adapter_step_id_shortcut_sentinel_report.json d115_gradient_bucket_id_shortcut_sentinel_report.json d115_split_integrity_report.json d115_overfit_memorization_report.json d115_negative_controls_report.json d115_truth_leak_oracle_isolation_report.json d115_report_schema_metric_crosscheck_report.json d115_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()
EXEMPT = {"aggregate_metrics.json", "summary.json", "decision.json", "report.md"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def check(out: Path) -> None:
    missing = [r for r in REPORTS if not (out / r).exists()]
    require(not missing, f"missing reports: {missing}")
    aggregate = read_json(out / "aggregate_metrics.json")
    summary = read_json(out / "summary.json")
    decision = read_json(out / "decision.json")
    manifest = read_json(out / "d114_upstream_manifest.json")
    scale, m, gates = aggregate["scale"], aggregate["metrics"], aggregate["positive_gates"]
    require(decision["decision"] == "d115_symbolic_sequence_bridge_scale_confirmed", "unexpected decision")
    require(decision["next"] == "D116_MULTI_STEP_INSTRUCTION_BRIDGE_PLAN_WITH_SEQUENCE_GUARDRAILS", "unexpected next")
    require(decision["d116_ready"] is True and m["d116_ready"] is True, "D116 readiness missing")
    require(all(gates.values()), f"failed positive gates: {[k for k, v in gates.items() if not v]}")
    require(manifest["validation_status"] == "valid" and manifest["replayed_decision"] == "d114_symbolic_sequence_bridge_prototype_confirmed", "D114 replay invalid")
    require(manifest["replayed_next"] == "D115_SYMBOLIC_SEQUENCE_BRIDGE_SCALE_CONFIRM_WITH_TRIG_GUARDRAILS" and manifest["replayed_d115_ready"] is True, "D114 next/readiness invalid")
    require(manifest["replayed_bridge_passed_all_gates"] is True and manifest["replayed_trig_remains_repair_only"] is True and manifest["replayed_multi_step_reference_only"] is True, "D114 replay metrics invalid")
    require(manifest["replayed_language_like_symbolic_command_is_natural_language"] is False and manifest["replayed_failed_jobs"] == [], "D114 boundary replay invalid")
    require(m["d114_replay_validation_passed"] is True and m["d114_replay_decision"] == "d114_symbolic_sequence_bridge_prototype_confirmed", "D114 metric replay invalid")
    require(scale["requested_total_rows"] == scale["actual_total_rows"] and scale["scale_reduced"] is False, "scale reduced")
    require(scale["requested_total_rows"] == 277980 and scale["stress_mode_count"] == 36 and scale["all_required_stress_modes_executed"] is True, "scale/stress mismatch")
    require(scale["failed_jobs"] == [] and scale["fallback_rows"] == 0, "scale fallback/failure")
    for key in ["natural_language_pretraining_executed", "gemma_class_training_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "raw_raven_used", "language_like_symbolic_command_is_natural_language"]:
        require(m[key] is False, f"boundary violation: {key}")
    require(m["sparse_candidate_identity_preserved"] is True and m["final_sparse_pct"] == 8 and m["final_anneal_pressure"] == "light", "sparse identity mismatch")
    require(m["protected_components_frozen"] is True and m["protected_component_modification_count"] == 0 and m["sparse_mask_frozen"] is True and m["sparse_mask_drift_rate"] <= 0.002, "freeze/sparse drift mismatch")
    require(set(m["trainable_adapter_names"]) == {"route_head_adapter_bridge_scale_delta", "halting_head_adapter_bridge_scale_delta", "recurrent_state_adapter_bridge_scale_delta", "calibration_scalar_adapter_bridge_scale_delta"}, "trainable adapters mismatch")
    require(m["bridge_scale_training_executed"] is True and m["training_updates_executed"] is True and m["total_bridge_steps_executed"] > 0 and 1 <= m["epochs_executed"] <= 4, "training execution mismatch")
    require(m["checkpoint_count"] >= 9 and m["failed_checkpoint_count"] == 0 and m["rollback_triggered"] is False and m["final_candidate_selected"] is True, "checkpoint/rollback mismatch")
    require(m["bridge_family_count"] == 6 and m["bridge_trainable_family_count"] == 2 and m["bridge_guarded_family_count"] == 3 and m["bridge_reference_only_family_count"] == 1, "bridge family counts mismatch")
    for key in ["symbolic_sequence_routing_passed", "ordered_rule_chain_passed", "language_like_symbolic_command_guarded_passed", "symbolic_command_composition_guarded_passed", "variable_binding_sequence_guarded_passed", "multi_step_instruction_reference_only"]:
        require(m[key] is True, f"bridge family gate failed: {key}")
    require(m["bridge_test_accuracy"] >= 0.9915 and m["bridge_ood_accuracy"] >= 0.9895 and m["bridge_stress_accuracy"] >= 0.9890 and m["bridge_loop_utility"] >= 0.675, "bridge accuracy/utility gate failed")
    require(m["bridge_halting_risk"] <= 0.05 and m["bridge_guard_risk"] <= 0.04 and m["bridge_D68_risk"] <= 0.02 and m["bridge_shortcut_risk"] <= 0.10 and m["bridge_routing_failure_rows"] == 0 and m["bridge_passed_all_gates"] is True, "bridge risk gate failed")
    for family in m["bridge_family_metrics"]:
        if family["bridge_family_status"] == "reference_only_hold":
            continue
        require(family["family_test_accuracy"] >= 0.9910 and family["family_ood_accuracy"] >= 0.9890 and family["family_stress_accuracy"] >= 0.9885, f"family accuracy gate failed: {family['bridge_family_name']}")
        require(family["family_loop_utility"] >= 0.672 and family["family_halting_risk"] <= 0.052 and family["family_guard_risk"] <= 0.042 and family["family_D68_risk"] <= 0.022 and family["family_shortcut_risk"] <= 0.10 and family["family_routing_failure_rows"] == 0 and family["family_passed_gate"] is True, f"family risk gate failed: {family['bridge_family_name']}")
    require(m["multi_step_instruction_reference_only"] is True and m["multi_step_instruction_in_healthy_claim"] is False and m["multi_step_instruction_long_sequence_halting_risk"] is not None and m["multi_step_instruction_shortcut_risk"] is not None and m["multi_step_instruction_recommended_next"], "multi-step reference gate failed")
    require(m["trig_remains_repair_only"] is True and m["trig_included_in_healthy_claim"] is False and m["trig_guardrail_risk"] <= 0.04, "trig guardrail gate failed")
    require(m["lane_a_interference"] <= 0.01 and m["lane_b_interference"] <= 0.01 and m["lane_d_interference"] <= 0.012, "lane interference gate failed")
    require(m["lane_a_D68_preservation_rate"] == 1.0 and m["lane_a_top1_guard_preserved"] is True and m["lane_a_routing_failure_rows"] == 0 and m["lane_b_status_preserved"] is True and m["lane_d_expansion_preserved"] is True, "lane preservation failed")
    require(m["post_bridge_false_confidence_rate"] <= 0.0049 and m["post_bridge_rust_path_invoked"] is True and m["post_bridge_fallback_rows"] == 0 and m["post_bridge_failed_jobs"] == [], "post bridge Rust/fallback gate failed")
    require(m["forbidden_feature_detected"] is False and m["route_distillation_label_leak_risk"] is False, "truth leak detected")
    for key in ["bridge_task_id_shortcut_detected", "command_template_id_shortcut_detected", "grammar_rule_id_shortcut_detected", "sequence_position_label_shortcut_detected", "multi_step_instruction_label_shortcut_detected", "scale_run_id_shortcut_detected"]:
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
    parser.add_argument("--out", type=Path, default=Path("target/pilot_wave/d115_symbolic_sequence_bridge_scale_confirm_with_trig_guardrails"))
    check(parser.parse_args().out)


if __name__ == "__main__":
    main()
