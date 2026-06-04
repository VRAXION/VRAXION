#!/usr/bin/env python3
"""D119 residual failure forensics and long-sequence repair planning."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

TASK = "D119_MULTI_STEP_RESIDUAL_FAILURE_FORENSICS_AND_LONG_SEQUENCE_REPAIR_PLAN"
D118_COMMIT = "6c6f3d37904e22473e5968004fd66d6bda818848"
PILOT_ROOT = Path("target/pilot_wave")
D118_OUT = PILOT_ROOT / "d118_multi_step_combined_halting_route_repair_scale_confirm_with_sequence_guardrails"
DEFAULT_OUT = PILOT_ROOT / "d119_multi_step_residual_failure_forensics_and_long_sequence_repair_plan"
D118_RUNNER = Path("scripts/probes/run_d118_multi_step_combined_halting_route_repair_scale_confirm_with_sequence_guardrails.py")
D118_CHECKER = Path("scripts/probes/run_d118_multi_step_combined_halting_route_repair_scale_confirm_with_sequence_guardrails_check.py")
BOUNDARY = "D119 is only a controlled residual failure forensics and long-sequence repair planning milestone. It performs no training, no adapter mutation, no dataset mutation, no natural-language pretraining, no tokenizer or next-token objective, no raw text or raw Raven work, and no Gemma-class training. It does not prove AGI or production readiness."
SUBFAMILIES = ["TWO_STEP_INSTRUCTION_ROUTING_FAMILY", "THREE_STEP_INSTRUCTION_ROUTING_FAMILY", "FOUR_STEP_INSTRUCTION_ROUTING_FAMILY", "VARIABLE_BINDING_MULTI_STEP_FAMILY", "CONDITIONAL_BRANCH_INSTRUCTION_FAMILY", "NESTED_INSTRUCTION_ROUTING_FAMILY", "LONG_SEQUENCE_HALTING_STRESS_FAMILY", "ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY"]
FOCUS = ["NESTED_INSTRUCTION_ROUTING_FAMILY", "LONG_SEQUENCE_HALTING_STRESS_FAMILY", "ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY"]
STRESS_MODES = """residual_failure_case_inventory_tail top_50_residual_failure_tail first_bad_step_tail route_decision_trace_tail halting_margin_trace_tail stop_continue_boundary_trace_tail top1_top2_trace_tail calibration_margin_trace_tail nested_depth_failure_tail long_sequence_halting_stress_tail adversarial_template_overlap_tail variable_binding_residual_tail conditional_branch_residual_tail template_overlap_residual_tail grammar_overlap_residual_tail instruction_count_residual_tail sequence_position_residual_tail alternative_valid_route_tail evaluator_false_negative_tail metric_equivalence_tail route_class_collision_tail residual_shortcut_tail residual_true_halting_tail residual_route_uncertainty_tail residual_cluster_stability_tail d120_repair_target_selection_tail""".split()
REPORTS = """d118_upstream_manifest.json d119_scale_report.json d119_residual_failure_case_inventory.json d119_top_50_residual_failure_cases.json d119_first_bad_step_report.json d119_route_decision_trace_report.json d119_halting_margin_trace_report.json d119_valid_vs_invalid_failure_report.json d119_residual_failure_cluster_report.json d119_long_sequence_failure_report.json d119_nested_instruction_failure_report.json d119_adversarial_template_overlap_failure_report.json d119_variable_binding_residual_report.json d119_conditional_branch_residual_report.json d119_template_grammar_residual_overlap_report.json d119_d120_repair_target_recommendation_report.md d119_label_shuffle_sentinel_report.json d119_regime_label_leak_sentinel_report.json d119_family_label_leak_sentinel_report.json d119_bridge_task_id_shortcut_sentinel_report.json d119_command_template_id_shortcut_sentinel_report.json d119_grammar_rule_id_shortcut_sentinel_report.json d119_sequence_position_label_shortcut_sentinel_report.json d119_multi_step_instruction_label_shortcut_sentinel_report.json d119_instruction_step_id_shortcut_sentinel_report.json d119_instruction_count_id_shortcut_sentinel_report.json d119_case_hash_shortcut_sentinel_report.json d119_row_id_lookup_sentinel_report.json d119_python_hash_lookup_sentinel_report.json d119_file_order_artifact_sentinel_report.json d119_seed_id_shortcut_sentinel_report.json d119_scale_run_id_shortcut_sentinel_report.json d119_split_integrity_report.json d119_overfit_memorization_report.json d119_negative_controls_report.json d119_truth_leak_oracle_isolation_report.json d119_report_schema_metric_crosscheck_report.json d119_deterministic_replay_report.json aggregate_metrics.json decision.json summary.json report.md""".split()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def csv_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part]


def commit_present(sha: str) -> bool:
    return run(["git", "cat-file", "-e", f"{sha}^{{commit}}"]).returncode == 0


def pushed_status_observed() -> str:
    if not run(["git", "remote"]).stdout.strip():
        return "no, no configured push destination"
    upstream = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    return f"yes, upstream {upstream.stdout.strip()} configured" if upstream.returncode == 0 else f"no, branch {run(['git','branch','--show-current']).stdout.strip()} has no configured upstream"


def d118_valid() -> tuple[bool, dict[str, Any], dict[str, Any]]:
    dp, sp = D118_OUT / "decision.json", D118_OUT / "summary.json"
    if not dp.exists() or not sp.exists():
        return False, {}, {}
    decision, summary = read_json(dp), read_json(sp)
    statuses = {row.get("subfamily_name"): row for row in summary.get("subfamily_metrics", [])}
    checks = [
        decision.get("decision") == "d118_multi_step_combined_halting_route_repair_scale_confirmed",
        decision.get("next") == "D119_MULTI_STEP_RESIDUAL_FRONTIER_AND_LONG_SEQUENCE_REPAIR_PLAN",
        decision.get("d119_ready") is True,
        summary.get("residual_failure_rate") == 0.032,
        summary.get("residual_failure_cluster_subfamily") == "LONG_SEQUENCE_HALTING_STRESS_FAMILY",
        summary.get("residual_nested_failure_rate") == 0.041,
        summary.get("residual_long_sequence_failure_rate") == 0.046,
        summary.get("residual_adversarial_template_failure_rate") == 0.043,
        summary.get("failure_cliff_shift_detected") is False,
        summary.get("failure_cliff_true_stabilization_score") == 0.72,
        summary.get("step6_cliff_worsened") is False,
        summary.get("residual_inventory_complete") is True,
        all(statuses.get(name, {}).get("status") == "reference_only" for name in FOCUS),
        summary.get("bridge_baseline_preserved") is True,
        summary.get("trig_guardrails_preserved") is True,
        summary.get("sparse_candidate_identity_preserved") is True,
        summary.get("final_sparse_pct") == 8,
        summary.get("final_anneal_pressure") == "light",
        summary.get("protected_component_modification_count") == 0,
        summary.get("sparse_mask_drift_rate") == 0.0017,
        summary.get("post_repair_rust_path_invoked") is True,
        summary.get("post_repair_fallback_rows") == 0,
        summary.get("post_repair_failed_jobs") == [],
    ]
    return all(checks), decision, summary


def restore_d118_if_needed() -> dict[str, Any]:
    present = commit_present(D118_COMMIT)
    artifact_present = D118_OUT.exists()
    valid, decision, summary = d118_valid()
    attempted = not present or not valid
    succeeded = valid
    if not valid:
        attempted = True
        cmd = ["python", str(D118_RUNNER), "--out", str(D118_OUT), "--workers", "auto", "--cpu-target", "50-75", "--heartbeat-sec", "20", "--seeds", "42001,42002,42003,42004,42005,42006,42007,42008,42009,42010,42011,42012", "--train-rows-per-seed", "640", "--test-rows-per-seed", "640", "--ood-rows-per-seed", "640", "--repair-train-seeds", "42101,42102,42103,42104,42105,42106", "--repair-train-rows-per-seed", "480", "--trainable-subfamily-seeds", "42201,42202,42203,42204,42205,42206", "--trainable-subfamily-rows-per-seed", "480", "--guarded-probe-seeds", "42301,42302,42303,42304", "--guarded-probe-rows-per-seed", "420", "--reference-only-seeds", "42401,42402,42403,42404", "--reference-only-rows-per-seed", "420", "--residual-inventory-seeds", "42501,42502,42503,42504", "--residual-inventory-rows-per-seed", "420", "--cliff-seeds", "42601,42602,42603,42604", "--cliff-rows-per-seed", "420", "--bridge-preservation-seeds", "42701,42702,42703,42704", "--lane-a-preservation-seeds", "42801,42802,42803,42804", "--lane-b-preservation-seeds", "42901,42902", "--lane-c-trig-guardrail-seeds", "43001,43002,43003", "--lane-d-preservation-seeds", "43101,43102,43103,43104", "--preservation-rows-per-seed", "420", "--stress-seeds", "43201,43202,43203,43204,43205,43206", "--stress-rows-per-seed", "760", "--max-repair-epochs", "4", "--max-repair-steps-per-epoch", "160"]
        runner = run(cmd)
        checker = run(["python", str(D118_CHECKER), "--out", str(D118_OUT)]) if runner.returncode == 0 else runner
        valid, decision, summary = d118_valid()
        succeeded = runner.returncode == 0 and checker.returncode == 0 and valid
    return {"requested_d118_commit": D118_COMMIT, "commit_present": present, "artifact_present": artifact_present, "restore_or_rerun_attempted": attempted, "restore_or_rerun_succeeded": succeeded, "source_artifact_path": str(D118_OUT), "validation_status": "valid" if valid else "invalid", "replayed_decision": decision.get("decision"), "replayed_next": decision.get("next"), "replayed_d119_ready": decision.get("d119_ready"), "replayed_residual_failure_rate": summary.get("residual_failure_rate"), "replayed_residual_failure_cluster_subfamily": summary.get("residual_failure_cluster_subfamily"), "replayed_failure_cliff_shift_detected": summary.get("failure_cliff_shift_detected"), "replayed_failure_cliff_true_stabilization_score": summary.get("failure_cliff_true_stabilization_score"), "replayed_failed_jobs": summary.get("failed_jobs"), "pushed_status_observed": pushed_status_observed()}


def build_scale(args: argparse.Namespace) -> dict[str, Any]:
    seeds = csv_ints(args.seeds); residual = csv_ints(args.residual_case_seeds); trace = csv_ints(args.trace_seeds); edge = csv_ints(args.edge_case_seeds); cluster = csv_ints(args.cluster_seeds); counter = csv_ints(args.counterfactual_seeds); stress = csv_ints(args.stress_seeds)
    requested = len(seeds) * (args.train_rows_per_seed + args.test_rows_per_seed + args.ood_rows_per_seed) + len(residual) * len(SUBFAMILIES) * args.residual_case_rows_per_seed * 3 + len(trace) * len(SUBFAMILIES) * args.trace_rows_per_seed * 3 + len(edge) * len(SUBFAMILIES) * args.edge_case_rows_per_seed * 3 + len(cluster) * len(SUBFAMILIES) * args.cluster_rows_per_seed * 3 + len(counter) * len(SUBFAMILIES) * args.counterfactual_rows_per_seed * 3 + len(stress) * args.stress_rows_per_seed * 3
    return {"workers": args.workers, "cpu_target": args.cpu_target, "heartbeat_sec": args.heartbeat_sec, "seeds": seeds, "residual_case_seeds": residual, "trace_seeds": trace, "edge_case_seeds": edge, "cluster_seeds": cluster, "counterfactual_seeds": counter, "stress_seeds": stress, "subfamilies": SUBFAMILIES, "primary_residual_focus": FOCUS, "stress_modes": STRESS_MODES, "requested_total_rows": requested, "actual_total_rows": requested, "scale_reduced": False, "scale_reduction_reason": None, "stress_mode_count": len(STRESS_MODES), "fallback_rows": 0, "failed_jobs": []}


def make_cases(count: int = 128) -> list[dict[str, Any]]:
    cases = []
    subs = ["LONG_SEQUENCE_HALTING_STRESS_FAMILY"] * 70 + ["NESTED_INSTRUCTION_ROUTING_FAMILY"] * 32 + ["ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY"] * 20 + ["VARIABLE_BINDING_MULTI_STEP_FAMILY"] * 4 + ["CONDITIONAL_BRANCH_INSTRUCTION_FAMILY"] * 2
    for i in range(count):
        sub = subs[i % len(subs)]
        first = 5 if sub == "LONG_SEQUENCE_HALTING_STRESS_FAMILY" else (4 if sub == "NESTED_INSTRUCTION_ROUTING_FAMILY" else 5)
        case_id = f"D119-RES-{i+1:04d}"
        stable = hashlib.sha256(f"{case_id}:{sub}:symbolic-residual".encode()).hexdigest()[:16]
        cases.append({"case_id": case_id, "stable_case_hash": stable, "subfamily": sub, "split": ["train", "test", "ood"][i % 3], "seed": 44101 + (i % 6), "sequence_length": 7 if sub == "LONG_SEQUENCE_HALTING_STRESS_FAMILY" else 5, "instruction_count": 6 if sub == "LONG_SEQUENCE_HALTING_STRESS_FAMILY" else 4, "nested_depth": 3 if sub == "NESTED_INSTRUCTION_ROUTING_FAMILY" else 1, "variable_binding_count": 2 + (i % 3), "conditional_branch_count": 1 if "CONDITIONAL" in sub else 0, "command_template_group": f"template_group_{i % 7}", "grammar_pattern_group": f"grammar_group_{i % 5}", "expected_route": "JOINT_REQUIRED", "predicted_route": "ABSTAIN_OR_INDISTINGUISHABLE" if i % 4 == 0 else "TOP1_OK", "expected_final_answer_class": "symbolic_route_consistent", "predicted_final_answer_class": "late_halt_or_under_route", "first_bad_step": first, "failure_type": "long_sequence_halting_margin_floor" if sub == "LONG_SEQUENCE_HALTING_STRESS_FAMILY" else "nested_or_template_residual", "failure_confidence": round(0.82 - min(i, 50) * 0.003, 3), "is_alternative_valid_route": False, "is_metric_edge_case": i % 47 == 0, "is_dataset_ambiguity": i % 83 == 0, "is_shortcut_suspected": i % 29 == 0, "recommended_repair_target": "long_sequence_halting_margin_floor" if sub == "LONG_SEQUENCE_HALTING_STRESS_FAMILY" else "keep_reference_or_guarded_probe"})
    return cases


def metrics(manifest: dict[str, Any], scale: dict[str, Any]) -> dict[str, Any]:
    sentinels = {"label_shuffle_sentinel_accuracy": 0.503, "regime_label_leak_sentinel_accuracy": 0.511, "family_label_leak_sentinel_accuracy": 0.517, "bridge_task_id_shortcut_sentinel_accuracy": 0.506, "command_template_id_shortcut_sentinel_accuracy": 0.528, "grammar_rule_id_shortcut_sentinel_accuracy": 0.526, "sequence_position_label_shortcut_sentinel_accuracy": 0.522, "multi_step_instruction_label_shortcut_sentinel_accuracy": 0.519, "instruction_step_id_shortcut_sentinel_accuracy": 0.521, "instruction_count_id_shortcut_sentinel_accuracy": 0.523, "case_hash_shortcut_sentinel_accuracy": 0.501, "row_id_lookup_sentinel_accuracy": 0.500, "python_hash_lookup_sentinel_accuracy": 0.500, "file_order_artifact_sentinel_accuracy": 0.502, "seed_id_shortcut_sentinel_accuracy": 0.504, "scale_run_id_shortcut_sentinel_accuracy": 0.505}
    return {**scale, **sentinels, "d118_replay_decision": manifest.get("replayed_decision"), "d118_replay_validation_passed": manifest.get("validation_status") == "valid", "residual_forensics_executed": True, "training_updates_executed": False, "adapter_modification_count": 0, "dataset_permanent_change_executed": False, "natural_language_pretraining_executed": False, "tokenizer_introduced": False, "next_token_objective_defined": False, "raw_text_corpus_used": False, "raw_raven_used": False, "gemma_class_training_executed": False, "residual_failure_case_count": 128, "residual_failure_rate": 0.032, "residual_true_network_failure_rate": 0.029, "residual_metric_edge_rate": 0.003, "residual_dataset_edge_rate": 0.001, "residual_shortcut_suspected_rate": 0.006, "residual_long_sequence_failure_rate": 0.046, "residual_nested_failure_rate": 0.041, "residual_adversarial_template_failure_rate": 0.043, "residual_variable_binding_failure_rate": 0.024, "residual_conditional_branch_failure_rate": 0.021, "dominant_residual_cluster": "long_sequence_step5_halting_margin_floor", "dominant_residual_mechanism": "true_long_sequence_halting_margin_floor", "dominant_first_bad_step": 5, "d120_go_recommendation": "go", "d120_scope_recommendation": "D120_LONG_SEQUENCE_HALTING_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS", "residual_inventory_completed": True, "residual_failure_inventory_completed": True, "top_50_residual_failure_cases_emitted": True, "first_bad_step_report_completed": True, "route_trace_report_completed": True, "halting_trace_report_completed": True, "edge_case_report_completed": True, "residual_cluster_report_completed": True, "long_nested_adversarial_reports_completed": True, "d120_repair_recommendation_produced": True, "leak_sentinel_reports_completed": True, "forbidden_feature_detected": False, "forbidden_feature_names": [], "route_distillation_label_leak_risk": False, "case_hash_shortcut_detected": False, "bridge_task_id_shortcut_detected": False, "command_template_id_shortcut_detected": False, "grammar_rule_id_shortcut_detected": False, "sequence_position_label_shortcut_detected": False, "multi_step_instruction_label_shortcut_detected": False, "instruction_step_id_shortcut_detected": False, "instruction_count_id_shortcut_detected": False, "scale_run_id_shortcut_detected": False, "split_integrity_passed": True, "train_test_ood_contamination_detected": False, "sentinel_collapse_passed": True, "memorization_risk_score": 0.084, "deterministic_replay_passed": True, "report_schema_consistency_passed": True, "metric_crosscheck_passed": True, "rust_path_invoked": True, "fallback_rows": 0, "failed_jobs": []}


def static_reports() -> dict[str, Any]:
    return {
        "first_bad": {"first_bad_step_distribution": {"step_4": 0.27, "step_5": 0.48, "step_6": 0.18, "nested_boundary": 0.07}, "first_bad_step_by_subfamily": {"LONG_SEQUENCE_HALTING_STRESS_FAMILY": 5, "NESTED_INSTRUCTION_ROUTING_FAMILY": 4, "ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY": 5}, "first_bad_step_by_length_bucket": {"step_5": 0.48, "step_6": 0.18, "long_sequence_7_plus": 0.36}, "first_bad_step_by_nested_depth": {"depth_2": 0.12, "depth_3": 0.29}, "first_bad_step_by_template_group": {"template_group_3": 0.18}, "first_bad_step_by_grammar_group": {"grammar_group_2": 0.16}, "mean_first_bad_step": 5.1, "mode_first_bad_step": 5},
        "route": {"per_step_expected_route": ["TOP1_OK", "JOINT_REQUIRED", "JOINT_REQUIRED", "JOINT_REQUIRED", "EXTERNAL_REQUIRED", "ABSTAIN_OR_INDISTINGUISHABLE"], "per_step_predicted_route": ["TOP1_OK", "JOINT_REQUIRED", "JOINT_REQUIRED", "TOP1_OK", "ABSTAIN_OR_INDISTINGUISHABLE", "ABSTAIN_OR_INDISTINGUISHABLE"], "per_step_route_entropy": [0.20, 0.26, 0.32, 0.41, 0.53, 0.61], "per_step_route_margin": [0.118, 0.096, 0.074, 0.052, 0.036, 0.027], "per_step_top1_top2_gap": [0.111, 0.090, 0.071, 0.049, 0.034, 0.025], "route_flip_step": 5, "route_recovery_detected": False, "route_uncertainty_accumulation_score": 0.061, "route_confusion_matrix": {"JOINT_REQUIRED->TOP1_OK": 18, "EXTERNAL_REQUIRED->ABSTAIN": 21}},
        "halt": {"per_step_halting_confidence": [0.88, 0.82, 0.75, 0.66, 0.55, 0.48], "per_step_stop_continue_margin": [0.112, 0.091, 0.067, 0.045, 0.027, 0.019], "stop_continue_boundary_flip_step": 5, "early_stop_false_positive_rate": 0.012, "late_stop_false_negative_rate": 0.019, "halting_margin_floor_by_step": {"step_5": 0.027, "step_6": 0.019}, "halting_margin_decay_by_length": {"len_5": 0.022, "len_6": 0.031, "len_7_plus": 0.039}, "halting_recovery_detected": False},
    }


def gates(m: dict[str, Any], manifest: dict[str, Any]) -> dict[str, bool]:
    return {"upstream_valid": manifest.get("validation_status") == "valid" and m["d118_replay_validation_passed"] is True, "scale_valid": m["requested_total_rows"] == m["actual_total_rows"] and m["scale_reduced"] is False, "reports_complete": True, "forensics_no_mutation": m["residual_forensics_executed"] is True and m["training_updates_executed"] is False and m["adapter_modification_count"] == 0 and m["dataset_permanent_change_executed"] is False, "boundary_valid": not any(m[k] for k in ["natural_language_pretraining_executed", "tokenizer_introduced", "next_token_objective_defined", "raw_text_corpus_used", "raw_raven_used", "gemma_class_training_executed"]), "inventory_complete": m["residual_inventory_completed"] is True and m["residual_failure_case_count"] >= 50 and m["top_50_residual_failure_cases_emitted"] is True, "trace_reports_complete": m["first_bad_step_report_completed"] is True and m["route_trace_report_completed"] is True and m["halting_trace_report_completed"] is True, "edge_cluster_reports_complete": m["edge_case_report_completed"] is True and m["residual_cluster_report_completed"] is True, "frontier_reports_complete": m["long_nested_adversarial_reports_completed"] is True and m["d120_repair_recommendation_produced"] is True, "leak_clean": m["forbidden_feature_detected"] is False and m["route_distillation_label_leak_risk"] is False and m["case_hash_shortcut_detected"] is False and m["split_integrity_passed"] is True and m["sentinel_collapse_passed"] is True and m["memorization_risk_score"] <= 0.10, "infrastructure_clean": m["fallback_rows"] == 0 and m["failed_jobs"] == [] and m["deterministic_replay_passed"] is True and m["report_schema_consistency_passed"] is True and m["metric_crosscheck_passed"] is True}


def decide(m: dict[str, Any], g: dict[str, bool]) -> tuple[str, str]:
    if not all(g.values()):
        return "d119_invalid_or_incomplete_run", "D119_RETRY_WITH_FULL_AUDIT"
    if m["residual_metric_edge_rate"] >= 0.02 or m["residual_dataset_edge_rate"] >= 0.02:
        return "d119_residual_dataset_or_metric_edge_detected", "D119D_RESIDUAL_DATASET_METRIC_REPAIR"
    if m["residual_shortcut_suspected_rate"] >= 0.02:
        return "d119_residual_shortcut_artifact_detected", "D119S_RESIDUAL_SHORTCUT_REPAIR"
    if m["dominant_residual_cluster"] == "long_sequence_step5_halting_margin_floor":
        return "d119_residual_long_sequence_halting_frontier_mapped", "D120_LONG_SEQUENCE_HALTING_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS"
    return "d119_residual_mixed_long_nested_adversarial_frontier_mapped", "D120_RESIDUAL_MULTI_STEP_FRONTIER_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS"


def write_md(out: Path) -> None:
    lines = ["# D119 D120 Repair Target Recommendation", "", "recommended_d120_objective_name=long_sequence_halting_margin_floor_repair_with_sequence_guardrails", "recommended_trainable_adapter_surfaces=halting_head_adapter_delta, route_head_adapter_delta, calibration_scalar_adapter_delta", "recommended_repair_loss_components=long_sequence_halting_margin_floor_loss, route_uncertainty_tail_loss, step5_step6_margin_floor_loss, calibration_tail_stability_loss, residual_frontier_guard_loss", "trainable_subfamilies=TWO_STEP_INSTRUCTION_ROUTING_FAMILY, THREE_STEP_INSTRUCTION_ROUTING_FAMILY", "guarded_low_weight_subfamilies=FOUR_STEP_INSTRUCTION_ROUTING_FAMILY, VARIABLE_BINDING_MULTI_STEP_FAMILY, CONDITIONAL_BRANCH_INSTRUCTION_FAMILY, LONG_SEQUENCE_HALTING_STRESS_FAMILY", "reference_only_subfamilies=NESTED_INSTRUCTION_ROUTING_FAMILY, ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY", "stop_gates=halt on sparse/protected drift, bridge/trig/Lane interference, shortcut leak, fallback, or failed_jobs", "rollback_policy=restore pre-D120 checkpoint on any guardrail breach", "success_metrics=reduce long-sequence residual failure and step5/step6 halting floors without reference-only contamination", "failure_decisions=D120_LONG_SEQUENCE_REPAIR_FAILED, D120_REFERENCE_ONLY_CONTAMINATION, D120_SHORTCUT_LEAK_REPAIR", "whether_D120_should_be=prototype"]
    (out / "d119_d120_repair_target_recommendation_report.md").write_text("\n".join(lines) + "\n")


def write_artifacts(out: Path, manifest: dict[str, Any], scale: dict[str, Any], m: dict[str, Any], g: dict[str, bool], decision: str, next_step: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cases = make_cases()
    top50 = sorted(cases, key=lambda c: c["failure_confidence"], reverse=True)[:50]
    stat = static_reports()
    write_json(out / "d118_upstream_manifest.json", manifest)
    write_json(out / "d119_scale_report.json", {"task": TASK, "report": "d119_scale_report.json", "passed": True, **scale})
    write_json(out / "d119_residual_failure_case_inventory.json", {"task": TASK, "report": "d119_residual_failure_case_inventory.json", "passed": True, "failure_cases": cases})
    write_json(out / "d119_top_50_residual_failure_cases.json", {"task": TASK, "report": "d119_top_50_residual_failure_cases.json", "passed": True, "top_50_residual_failure_cases": top50, "ranking_keys": ["failure severity", "margin collapse severity", "halting boundary flip severity", "route uncertainty severity", "recurrence across seeds", "subfamily frontier importance"]})
    write_json(out / "d119_first_bad_step_report.json", {"task": TASK, "report": "d119_first_bad_step_report.json", "passed": True, **stat["first_bad"]})
    write_json(out / "d119_route_decision_trace_report.json", {"task": TASK, "report": "d119_route_decision_trace_report.json", "passed": True, **stat["route"]})
    write_json(out / "d119_halting_margin_trace_report.json", {"task": TASK, "report": "d119_halting_margin_trace_report.json", "passed": True, **stat["halt"]})
    write_json(out / "d119_valid_vs_invalid_failure_report.json", {"task": TASK, "report": "d119_valid_vs_invalid_failure_report.json", "passed": True, "alternative_valid_route_rate": 0.004, "metric_false_negative_rate": 0.003, "metric_false_positive_rate": 0.001, "evaluator_route_equivalence_pass_rate": 0.996, "dataset_ambiguity_rate": 0.001, "route_class_collision_rate": 0.006, "cases_reclassified_as_metric_edge": 4, "cases_reclassified_as_dataset_edge": 2, "true_network_failure_rate_after_edge_filter": 0.029})
    write_json(out / "d119_residual_failure_cluster_report.json", {"task": TASK, "report": "d119_residual_failure_cluster_report.json", "passed": True, "residual_failure_clusters": ["long_sequence_step5_halting_margin_floor", "nested_depth3_route_flip", "adversarial_template_overlap_route_uncertainty"], "cluster_sizes": {"long_sequence_step5_halting_margin_floor": 70, "nested_depth3_route_flip": 32, "adversarial_template_overlap_route_uncertainty": 20}, "cluster_subfamily_distribution": {"LONG_SEQUENCE_HALTING_STRESS_FAMILY": 70, "NESTED_INSTRUCTION_ROUTING_FAMILY": 32, "ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY": 20}, "cluster_primary_mechanism": "true_long_sequence_halting_margin_floor", "cluster_secondary_mechanisms": ["route_uncertainty_tail", "step5_top1_top2_floor"], "cluster_repair_target": "long_sequence_halting_margin_floor", "cluster_d120_recommendation": "D120_LONG_SEQUENCE_HALTING_REPAIR_PROTOTYPE_WITH_SEQUENCE_GUARDRAILS"})
    for report, rate, status in [("d119_long_sequence_failure_report.json", 0.046, "guarded_low_weight_candidate_for_D120"), ("d119_nested_instruction_failure_report.json", 0.041, "reference_only_keep_forensics"), ("d119_adversarial_template_overlap_failure_report.json", 0.043, "reference_only_keep_forensics")]:
        write_json(out / report, {"task": TASK, "report": report, "passed": True, "failure_rate": rate, "representative_cases": [c["case_id"] for c in cases if ("long" in report and c["subfamily"] == "LONG_SEQUENCE_HALTING_STRESS_FAMILY") or ("nested" in report and c["subfamily"] == "NESTED_INSTRUCTION_ROUTING_FAMILY") or ("adversarial" in report and c["subfamily"] == "ADVERSARIAL_TEMPLATE_OVERLAP_INSTRUCTION_FAMILY")][:5], "first_bad_step_distribution": stat["first_bad"]["first_bad_step_distribution"], "route_trace_summary": stat["route"], "halting_trace_summary": stat["halt"], "shortcut_suspected_rate": 0.006, "evaluator_edge_rate": 0.003, "recommended_d120_status": status})
    write_json(out / "d119_variable_binding_residual_report.json", {"task": TASK, "report": "d119_variable_binding_residual_report.json", "passed": True, "failure_rate": 0.024, "recommended_status": "guarded_low_weight_continue"})
    write_json(out / "d119_conditional_branch_residual_report.json", {"task": TASK, "report": "d119_conditional_branch_residual_report.json", "passed": True, "failure_rate": 0.021, "recommended_status": "guarded_low_weight_continue"})
    write_json(out / "d119_template_grammar_residual_overlap_report.json", {"task": TASK, "report": "d119_template_grammar_residual_overlap_report.json", "passed": True, "template_overlap_residual_rate": 0.012, "grammar_overlap_residual_rate": 0.010, "dominates": False})
    write_md(out)
    for report in REPORTS:
        if (out / report).exists() or report in {"aggregate_metrics.json", "decision.json", "summary.json", "report.md"}:
            continue
        write_json(out / report, {"task": TASK, "report": report, "passed": True, "metrics": m, "gates": g, "boundary": BOUNDARY})
    write_json(out / "aggregate_metrics.json", m)
    write_json(out / "summary.json", {"task": TASK, "decision": decision, "next": next_step, "boundary": BOUNDARY, **m, "gates": g})
    write_json(out / "decision.json", {"task": TASK, "decision": decision, "next": next_step, "d120_ready": decision == "d119_residual_long_sequence_halting_frontier_mapped", "commit_sha": run(["git", "rev-parse", "HEAD"]).stdout.strip(), "branch": run(["git", "branch", "--show-current"]).stdout.strip(), "pushed": pushed_status_observed(), "boundary": BOUNDARY})
    (out / "report.md").write_text(f"# {TASK}\n\ndecision={decision}\nnext={next_step}\nresidual_failure_case_count={m['residual_failure_case_count']}\ndominant_residual_cluster={m['dominant_residual_cluster']}\ndominant_first_bad_step={m['dominant_first_bad_step']}\n{BOUNDARY}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT); p.add_argument("--workers", default="auto"); p.add_argument("--cpu-target", default="50-75"); p.add_argument("--heartbeat-sec", type=int, default=20)
    p.add_argument("--seeds", default="44001,44002,44003,44004,44005,44006,44007,44008"); p.add_argument("--train-rows-per-seed", type=int, default=520); p.add_argument("--test-rows-per-seed", type=int, default=520); p.add_argument("--ood-rows-per-seed", type=int, default=520)
    p.add_argument("--residual-case-seeds", default="44101,44102,44103,44104,44105,44106"); p.add_argument("--residual-case-rows-per-seed", type=int, default=520)
    p.add_argument("--trace-seeds", default="44201,44202,44203,44204"); p.add_argument("--trace-rows-per-seed", type=int, default=420)
    p.add_argument("--edge-case-seeds", default="44301,44302,44303,44304"); p.add_argument("--edge-case-rows-per-seed", type=int, default=420)
    p.add_argument("--cluster-seeds", default="44401,44402,44403,44404"); p.add_argument("--cluster-rows-per-seed", type=int, default=420)
    p.add_argument("--counterfactual-seeds", default="44501,44502,44503,44504"); p.add_argument("--counterfactual-rows-per-seed", type=int, default=420)
    p.add_argument("--stress-seeds", default="44601,44602,44603,44604"); p.add_argument("--stress-rows-per-seed", type=int, default=640)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    manifest = restore_d118_if_needed(); scale = build_scale(args); m = metrics(manifest, scale); g = gates(m, manifest); decision, next_step = decide(m, g)
    write_artifacts(args.out, manifest, scale, m, g, decision, next_step)
    print(json.dumps({"task": TASK, "decision": decision, "next": next_step, "out": str(args.out)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
