#!/usr/bin/env python3
"""Checker for D77 joint-recall component integration plan artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_REPORTS = [
    "d76_upstream_manifest.json",
    "integration_target_selection_report.json",
    "component_interface_report.json",
    "required_input_feature_report.json",
    "action_influence_report.json",
    "D68_regression_prevention_report.json",
    "Rust_sparse_integration_surface_report.json",
    "D78_proof_gate_report.json",
    "risk_register.json",
    "truth_leak_audit_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

ALLOWED_DECISIONS = {
    "joint_recall_integration_plan_selected",
    "joint_recall_requires_sparse_diagnostic_integration",
    "joint_recall_external_combo_plan_selected",
    "joint_recall_integration_plan_not_ready",
}

EXPECTED_D78_GATES = [
    "average_total_support_used",
    "distance_to_concrete_oracle_support",
    "gap_reduction_vs_D73_bound",
    "exact_joint_accuracy",
    "correlated_echo_accuracy",
    "adversarial_distractor_accuracy",
    "external_test_required_accuracy",
    "joint_counter_recall_on_joint_required_rows",
    "external_recall_on_external_required_rows",
    "wrong_concrete_counter_rate",
    "weak_top1_top2_path_failure_rate",
    "top1_top2_sufficient_false_joint_rate",
    "false_confidence_rate",
    "indistinguishable_abstain_rate",
    "D68_loss_repair_preservation_rate",
    "routing_failure_rows",
    "rust_path_invoked",
    "fallback_rows",
    "failed_jobs",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise SystemExit(f"D77 check failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/d77_joint_recall_component_integration_plan")
    args = parser.parse_args()
    out = Path(args.out)

    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    if missing:
        fail(f"missing required reports: {missing}")

    manifest = load_json(out / "d76_upstream_manifest.json")
    selection = load_json(out / "integration_target_selection_report.json")
    interface = load_json(out / "component_interface_report.json")
    features = load_json(out / "required_input_feature_report.json")
    actions = load_json(out / "action_influence_report.json")
    d68 = load_json(out / "D68_regression_prevention_report.json")
    rust = load_json(out / "Rust_sparse_integration_surface_report.json")
    d78 = load_json(out / "D78_proof_gate_report.json")
    risk = load_json(out / "risk_register.json")
    truth = load_json(out / "truth_leak_audit_report.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")

    if decision.get("decision") not in ALLOWED_DECISIONS:
        fail(f"unexpected decision: {decision.get('decision')}")
    if aggregate.get("failed_jobs") or decision.get("failed_jobs") or risk.get("failed_jobs"):
        fail(f"failed jobs visible and non-empty: {aggregate.get('failed_jobs') or decision.get('failed_jobs') or risk.get('failed_jobs')}")

    d76 = manifest.get("d76_artifacts", {})
    if d76.get("decision") != "joint_recall_component_scale_confirmed":
        fail(f"D76 upstream decision mismatch: {d76.get('decision')}")
    if d76.get("next") != "D77_JOINT_RECALL_COMPONENT_INTEGRATION_PLAN":
        fail(f"D76 upstream next mismatch: {d76.get('next')}")
    if d76.get("scaled_arm") != "D75_JOINT_RECALL_COST_AWARE_REPLAY":
        fail(f"D76 scaled arm mismatch: {d76.get('scaled_arm')}")
    if not manifest.get("d76_commit_present", {}).get("present") and not manifest.get("rerun", {}).get("rerun_attempted"):
        fail("D76 commit missing but rerun/restore status was not explicit")
    if manifest.get("rerun", {}).get("rerun_attempted") and not manifest.get("rerun", {}).get("rerun_succeeded"):
        fail("D76 rerun was attempted but did not succeed")

    if decision.get("decision") == "joint_recall_integration_plan_selected":
        if decision.get("selected_integration_target") != "COUNTER_ACTION_ROUTER_JOINT_RECALL_MODULE":
            fail(f"unexpected selected target: {decision.get('selected_integration_target')}")
        if decision.get("next") != "D78_JOINT_RECALL_INTEGRATED_CONTROLLER_PROTOTYPE":
            fail(f"unexpected next milestone: {decision.get('next')}")
        if not selection.get("single_best_clear") or not selection.get("passed"):
            fail("selection report did not mark a single clear best target")

    for name, report in [
        ("component interface", interface),
        ("required input feature", features),
        ("action influence", actions),
        ("D68 prevention", d68),
        ("Rust sparse surface", rust),
        ("D78 proof gate", d78),
        ("risk register", risk),
        ("truth leak", truth),
    ]:
        if not report.get("passed"):
            fail(f"{name} report did not pass")

    if "top1_top2_sufficient_flag" not in interface.get("input_contract", {}):
        fail("component interface lacks explicit D68 top1/top2 sufficiency input")
    if "joint_required_signal" not in interface.get("input_contract", {}):
        fail("component interface lacks joint-required signal")
    if "external_required_signal" not in interface.get("input_contract", {}):
        fail("component interface lacks separate external-required signal")
    if not interface.get("location_in_stack", "").startswith("Rust sparse ECF/IPF controller counter-action router"):
        fail("component is not located in the Rust sparse counter-action router stack")

    feature_names = {item.get("feature") for item in features.get("features", [])}
    required_features = {"top1_top2_margin", "top1_top2_sufficient_flag", "joint_required_signal", "counter_candidate_supports", "external_required_signal", "indistinguishable_signal"}
    if not required_features.issubset(feature_names):
        fail(f"missing required input features: {sorted(required_features - feature_names)}")
    if not features.get("truth_leak_guarded"):
        fail("input feature report does not declare truth leak guarding")

    rules = "\n".join(d68.get("rules", []))
    for needle in ["Never bypass", "top1/top2", "selected-counter", "D68 loss"]:
        if needle not in rules:
            fail(f"D68 prevention rules missing {needle!r}")
    guard_metrics = d68.get("D78_guard_metrics", {})
    for key in ["weak_top1_top2_path_failure_rate", "top1_top2_sufficient_false_joint_rate", "D68_loss_repair_preservation_rate", "routing_failure_rows"]:
        if key not in guard_metrics:
            fail(f"D68 guard metric missing: {key}")

    measurable = d78.get("measurable_gates", {})
    absent_gates = [key for key in EXPECTED_D78_GATES if key not in measurable]
    if absent_gates:
        fail(f"D78 measurable gates missing: {absent_gates}")
    if d78.get("next_milestone") != "D78_JOINT_RECALL_INTEGRATED_CONTROLLER_PROTOTYPE":
        fail(f"D78 next milestone mismatch: {d78.get('next_milestone')}")

    if truth.get("fair_arms_using_truth_label"):
        fail("fair truth-label leak detected")
    if truth.get("fair_arms_using_support_regime_label"):
        fail("fair support-regime leak detected")
    if truth.get("label_echo_fair_oracle_used"):
        fail("label echo fair oracle detected")
    if truth.get("row_id_lookup_used") or truth.get("python_hash_used"):
        fail("row-id lookup or Python hash hard gate failed")
    if not truth.get("truth_hidden_from_fair_arms"):
        fail("truth hidden from fair arms not declared")

    if not aggregate.get("planning_metrics_are_estimates_not_empirical_D77_results"):
        fail("planning metrics are not explicitly labeled as estimates")
    if "full VRAXION brain" in json.dumps(aggregate) and "does not prove full VRAXION brain" not in aggregate.get("boundary", ""):
        fail("boundary language missing around prohibited claims")

    print(
        json.dumps(
            {
                "check": "passed",
                "out": str(out),
                "decision": decision,
                "selected_integration_target": decision.get("selected_integration_target"),
                "component": interface.get("component_name"),
                "d76_rerun_attempted": manifest.get("rerun", {}).get("rerun_attempted"),
                "d76_rerun_succeeded": manifest.get("rerun", {}).get("rerun_succeeded"),
                "failed_jobs": aggregate.get("failed_jobs"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
