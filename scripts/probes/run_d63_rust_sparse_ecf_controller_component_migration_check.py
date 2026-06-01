#!/usr/bin/env python3
"""Checker for D63 Rust sparse ECF controller component migration."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d62_upstream_manifest.json",
    "partial_test_metrics_by_track.json",
    "partial_ood_metrics_by_track.json",
    "track_metrics_test_SATURATED_STABILITY.json",
    "track_metrics_test_HARD_CAP8_LEARNING.json",
    "track_metrics_test_MIXED_EVAL.json",
    "track_metrics_test_OOD_CONTEXT_SHIFT.json",
    "track_metrics_test_ADVERSARIAL_GATE_CONFUSION.json",
    "track_metrics_test_EXTERNAL_TEST_REQUIRED.json",
    "track_metrics_test_INDISTINGUISHABLE_SUPPORT.json",
    "track_metrics_test_NOISY_CONTEXT.json",
    "track_metrics_test_HIDDEN_BUDGET_CONTEXT.json",
    "track_metrics_ood_SATURATED_STABILITY.json",
    "track_metrics_ood_HARD_CAP8_LEARNING.json",
    "track_metrics_ood_MIXED_EVAL.json",
    "track_metrics_ood_OOD_CONTEXT_SHIFT.json",
    "track_metrics_ood_ADVERSARIAL_GATE_CONFUSION.json",
    "track_metrics_ood_EXTERNAL_TEST_REQUIRED.json",
    "track_metrics_ood_INDISTINGUISHABLE_SUPPORT.json",
    "track_metrics_ood_NOISY_CONTEXT.json",
    "track_metrics_ood_HIDDEN_BUDGET_CONTEXT.json",
    "diagnostic_feature_definition_report.json",
    "rust_estimator_mapping_report.json",
    "estimator_accuracy_report.json",
    "gate_with_rust_diagnostics_report.json",
    "component_ablation_report.json",
    "truth_leak_audit_report.json",
    "rust_invocation_report.json",
    "support_cost_frontier_report.json",
    "false_confidence_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
    "trained_policy_manifest.json",
]

TRACKS = {
    "SATURATED_STABILITY",
    "HARD_CAP8_LEARNING",
    "MIXED_EVAL",
    "OOD_CONTEXT_SHIFT",
    "ADVERSARIAL_GATE_CONFUSION",
    "EXTERNAL_TEST_REQUIRED",
    "INDISTINGUISHABLE_SUPPORT",
    "NOISY_CONTEXT",
    "HIDDEN_BUDGET_CONTEXT",
}

REGIMES = {
    "CLEAN_INDEPENDENT_SUPPORT",
    "CORRELATED_ECHO_SUPPORT",
    "ADVERSARIAL_DISTRACTOR_SUPPORT",
    "MIXED_CLEAN_AND_CORRELATED",
    "MIXED_CLEAN_AND_ADVERSARIAL",
    "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
    "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
    "EXTERNAL_TEST_REQUIRED_SUPPORT",
}

DIAGNOSTICS = {
    "support_budget_pressure",
    "counterfactual_pressure",
    "adversarial_pressure",
    "internal_unresolvable",
    "external_channel",
}

ARMS = [
    "D62_SYMBOLIC_FEATURE_GATE_REFERENCE",
    "RUST_SPARSE_BUDGET_PRESSURE_ESTIMATOR",
    "RUST_SPARSE_COUNTERFACTUAL_PRESSURE_ESTIMATOR",
    "RUST_SPARSE_ADVERSARIAL_PRESSURE_ESTIMATOR",
    "RUST_SPARSE_UNRESOLVABLE_ESTIMATOR",
    "RUST_SPARSE_EXTERNAL_NEED_ESTIMATOR",
    "RUST_SPARSE_ALL_DIAGNOSTICS_GATE",
    "HYBRID_SYMBOLIC_RUST_DIAGNOSTICS_GATE",
    "SHUFFLED_DIAGNOSTIC_CONTROL",
    "RANDOM_DIAGNOSTIC_CONTROL",
    "DIAGNOSTIC_ABLATION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]

FAIR_ARMS = {
    "D62_SYMBOLIC_FEATURE_GATE_REFERENCE",
    "RUST_SPARSE_BUDGET_PRESSURE_ESTIMATOR",
    "RUST_SPARSE_COUNTERFACTUAL_PRESSURE_ESTIMATOR",
    "RUST_SPARSE_ADVERSARIAL_PRESSURE_ESTIMATOR",
    "RUST_SPARSE_UNRESOLVABLE_ESTIMATOR",
    "RUST_SPARSE_EXTERNAL_NEED_ESTIMATOR",
    "RUST_SPARSE_ALL_DIAGNOSTICS_GATE",
    "HYBRID_SYMBOLIC_RUST_DIAGNOSTICS_GATE",
}

RUST_DIAGNOSTIC_ARMS = FAIR_ARMS - {"D62_SYMBOLIC_FEATURE_GATE_REFERENCE"}
CONTROL_ARMS = {"SHUFFLED_DIAGNOSTIC_CONTROL", "RANDOM_DIAGNOSTIC_CONTROL", "DIAGNOSTIC_ABLATION_CONTROL"}
REFERENCE_ONLY_ARMS = {"TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}

ALLOWED_DECISIONS = {
    "rust_sparse_ecf_diagnostic_component_migration_confirmed",
    "hybrid_diagnostic_migration_positive",
    "diagnostic_component_migration_not_confirmed",
    "invalid_diagnostic_truth_leak",
}

ROW_FIELDS = [
    "row_id",
    "seed",
    "split",
    "track",
    "mixed_source_track",
    "difficulty_variant",
    "runtime_support_budget_cap",
    "arm",
    "selected_action",
    "support_regime",
    "truth_joint",
    "pred_joint",
    "truth_pair_equivalence",
    "pred_pair_equivalence",
    "truth_operator",
    "pred_operator",
    "exact_joint_correct",
    "cell_pair_equivalence_correct",
    "operator_exact_correct",
    "correct",
    "original_support_used",
    "total_support_used",
    "counter_support_used",
    "external_test_used",
    "abstained",
    "false_confidence",
    "confidence",
    "error_type",
    "rust_network_path_invoked",
    "rust_propagate_sparse_called",
    "python_fallback_used",
    "gate_selected_policy",
    "gate_basis",
    "gate_features",
    "diagnostic_estimates",
    "diagnostic_targets",
    "diagnostic_correct",
    "diagnostic_used_forbidden_feature",
    "reference_only_arm",
    "fair_arm",
    "control_arm",
    "rust_diagnostic_estimator_used",
    "rust_diagnostic_estimator_invoked",
    "rust_diagnostic_python_fallback_used",
]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_guardrails(repo_root):
    source = (repo_root / "scripts/probes/run_d63_rust_sparse_ecf_controller_component_migration.py").read_text(encoding="utf-8")
    compact = source.replace(" ", "").replace("\n", "").lower()
    for needle, message in [
        ("hash(", "Python hash forbidden"),
        ("random.random()<", "fake random threshold sampling forbidden"),
        ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
        ("expected_selected", "expected_selected feature forbidden"),
        ("openai", "external API reference forbidden"),
        ("modal", "Modal reference forbidden"),
    ]:
        require(needle not in compact, f"{message} in D63 runner")
    require("run_rust_estimator_bridge" in source, "D63 runner missing Rust diagnostic estimator bridge")
    require("Network::propagate_sparse" in source or "ensure_rust_multi_harness" in source, "D63 runner missing canonical Rust sparse path")
    require("TRUTH_LEAK_SENTINEL_REFERENCE_ONLY" in source, "D63 runner missing truth leak sentinel")
    require("FORBIDDEN_DIAGNOSTIC_FEATURES" in source, "D63 runner missing forbidden diagnostic feature list")


def inspect_jsonl(path):
    rows = 0
    arms = set()
    tracks = set()
    regimes = set()
    fair_rows = {arm: 0 for arm in FAIR_ARMS}
    rust_diag_rows = {arm: 0 for arm in RUST_DIAGNOSTIC_ARMS | CONTROL_ARMS}
    rust_diag_fallback = {arm: 0 for arm in RUST_DIAGNOSTIC_ARMS | CONTROL_ARMS}
    ref_forbidden_rows = 0
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing fields {missing}")
            require(row["total_support_used"] >= row["original_support_used"], f"{path} row {idx} support accounting invalid")
            require(isinstance(row["gate_features"], dict), f"{path} row {idx} gate_features not dict")
            require(set(row["diagnostic_estimates"]) == DIAGNOSTICS, f"{path} row {idx} diagnostic estimates mismatch")
            require(set(row["diagnostic_targets"]) == DIAGNOSTICS, f"{path} row {idx} diagnostic targets mismatch")
            require(set(row["diagnostic_correct"]) == DIAGNOSTICS, f"{path} row {idx} diagnostic correct mismatch")
            arms.add(row["arm"])
            tracks.add(row["track"])
            regimes.add(row["support_regime"])
            if row["arm"] in FAIR_ARMS:
                fair_rows[row["arm"]] += 1
                require(row["fair_arm"] is True, f"{path} row {idx} fair arm flag missing")
                require(row["diagnostic_used_forbidden_feature"] is False, f"{path} row {idx} fair arm used forbidden feature")
                require(row["reference_only_arm"] is False, f"{path} row {idx} fair arm marked reference-only")
            if row["arm"] in RUST_DIAGNOSTIC_ARMS | CONTROL_ARMS:
                if row["rust_diagnostic_estimator_invoked"]:
                    rust_diag_rows[row["arm"]] += 1
                if row["rust_diagnostic_python_fallback_used"]:
                    rust_diag_fallback[row["arm"]] += 1
            if row["arm"] in REFERENCE_ONLY_ARMS and row["diagnostic_used_forbidden_feature"]:
                ref_forbidden_rows += 1
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(TRACKS.issubset(tracks), f"{path} missing tracks {TRACKS - tracks}")
    require(REGIMES.issubset(regimes), f"{path} missing regimes {REGIMES - regimes}")
    for arm in FAIR_ARMS:
        require(fair_rows[arm] > 0, f"{path} missing fair rows for {arm}")
    for arm in RUST_DIAGNOSTIC_ARMS | CONTROL_ARMS:
        require(rust_diag_rows[arm] > 0, f"{path} missing Rust diagnostic estimator rows for {arm}")
        require(rust_diag_fallback[arm] == 0, f"{path} diagnostic fallback used by {arm}")
    require(ref_forbidden_rows > 0, f"{path} reference-only truth/track sentinel was not exercised")


def require_track(metrics, track):
    require(track in metrics, f"missing track metrics {track}")
    item = metrics[track]
    require(set(ARMS).issubset(item["by_arm_core"].keys()), f"{track} missing arms in core metrics")
    for arm in ARMS:
        require(REGIMES.issubset(item["by_arm_and_regime"][arm].keys()), f"{track} {arm} missing regimes")
        require(DIAGNOSTICS.issubset(item["diagnostic_accuracy"][arm].keys()), f"{track} {arm} missing diagnostic accuracy")
    return item


def exact(track_metrics, arm):
    return float(track_metrics["by_arm_core"][arm]["exact_joint_accuracy"])


def support(track_metrics, arm):
    return float(track_metrics["by_arm_core"][arm]["average_total_support_used"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    require(not missing, f"missing artifacts: {missing}")

    repo_root = Path(__file__).resolve().parents[2]
    source_guardrails(repo_root)
    inspect_jsonl(out / "row_outputs_test.jsonl")
    inspect_jsonl(out / "row_outputs_ood.jsonl")

    dataset = load(out / "dataset_manifest.json")
    upstream = load(out / "d62_upstream_manifest.json")
    definitions = load(out / "diagnostic_feature_definition_report.json")
    estimator_map = load(out / "rust_estimator_mapping_report.json")
    truth_audit = load(out / "truth_leak_audit_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    report = (out / "report.md").read_text(encoding="utf-8")

    require(dataset["truth_hidden_from_controller_inputs"] is True, "truth not hidden from controller inputs")
    require(dataset["truth_hidden_from_diagnostic_estimators"] is True, "truth not hidden from diagnostic estimators")
    require(dataset["controller_only_not_formula_solver"] is True, "controller-only flag missing")
    require(dataset["formula_solver_learning_used"] is False, "formula solver learning flag wrong")
    require(set(dataset["tracks"]) == TRACKS, "dataset tracks mismatch")
    require(set(dataset["diagnostic_features"]) >= {"support_budget_pressure_norm", "external_channel_available"}, "diagnostic feature list incomplete")
    require(upstream["expected_decision"] == "policy_ensemble_learned_gate_confirmed", "bad D62 expected decision")
    require(upstream["expected_next"] == "D63_RUST_SPARSE_ECF_CONTROLLER_COMPONENT_MIGRATION", "bad D62 next")
    require(upstream["decision_present"] and upstream["summary_present"], "D62 upstream artifacts missing")
    require(upstream["learned_gate_present"], "D62 learned gate missing")
    require(definitions["truth_labels_used"] is False, "diagnostic definitions use truth labels")
    require(definitions["support_regime_labels_used"] is False, "diagnostic definitions use support regime labels")
    require(estimator_map["rust_path"].startswith("canonical Network::propagate_sparse"), "bad Rust estimator mapping path")
    require(truth_audit["fair_arms_with_truth_leak"] == [], "fair arms with truth leak")
    require(truth_audit["fair_arms_using_forbidden_features"] == [], "fair arms using forbidden features")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D63 only tests migration" in decision["boundary"], "decision boundary missing")
    require("D63 only tests migration" in report, "report boundary missing")
    require(aggregate["rust_path_invoked"] is True, "Rust path not invoked")
    require(aggregate["fallback_rows"] == 0, "Python fallback rows nonzero")
    require(aggregate["failed_jobs"] == [], "failed jobs present")
    require(summary["decision"] == decision["decision"], "summary/decision mismatch")

    test_tracks = aggregate["test_metrics_by_track"]
    for track in TRACKS:
        require_track(test_tracks, track)

    sat = test_tracks["SATURATED_STABILITY"]
    hard = test_tracks["HARD_CAP8_LEARNING"]
    mixed = test_tracks["MIXED_EVAL"]
    ood = test_tracks["OOD_CONTEXT_SHIFT"]
    gate_conf = test_tracks["ADVERSARIAL_GATE_CONFUSION"]
    external = test_tracks["EXTERNAL_TEST_REQUIRED"]
    indist = test_tracks["INDISTINGUISHABLE_SUPPORT"]
    hidden = test_tracks["HIDDEN_BUDGET_CONTEXT"]
    best = decision["best_arm"]
    require(best in FAIR_ARMS | REFERENCE_ONLY_ARMS, "best arm is not a gate arm")
    if decision["decision"] != "invalid_diagnostic_truth_leak":
        require(best in FAIR_ARMS, "non-leak decision chose reference-only arm")

    arm_summary = aggregate["arm_summaries"][best]
    best_metrics = {
        "sat": exact(sat, best),
        "hard": exact(hard, best),
        "mixed": exact(mixed, best),
        "ood": exact(ood, best),
        "gate_confusion": exact(gate_conf, best),
        "hidden": exact(hidden, best),
        "external": float(arm_summary["EXTERNAL_TEST_REQUIRED"]["external"]),
        "indistinguishable_abstain": float(arm_summary["INDISTINGUISHABLE_SUPPORT"]["abstain"]),
        "max_false_confidence": max(float(arm_summary[track]["false_confidence"]) for track in TRACKS),
        "support_mixed": support(mixed, best),
    }
    all_diag = "RUST_SPARSE_ALL_DIAGNOSTICS_GATE"
    all_diag_metrics = {
        "sat": exact(sat, all_diag),
        "hard": exact(hard, all_diag),
        "mixed": exact(mixed, all_diag),
        "ood": exact(ood, all_diag),
        "gate_confusion": exact(gate_conf, all_diag),
        "hidden": exact(hidden, all_diag),
        "external": float(aggregate["arm_summaries"][all_diag]["EXTERNAL_TEST_REQUIRED"]["external"]),
        "indistinguishable_abstain": float(aggregate["arm_summaries"][all_diag]["INDISTINGUISHABLE_SUPPORT"]["abstain"]),
    }
    estimator_accuracy = test_tracks["MIXED_EVAL"]["diagnostic_accuracy"][all_diag]
    for name in DIAGNOSTICS:
        require(estimator_accuracy[name]["accuracy"] >= 0.99, f"{name} estimator accuracy below 0.99")

    if decision["decision"] == "rust_sparse_ecf_diagnostic_component_migration_confirmed":
        require(best == all_diag, "all-diagnostics confirmation did not choose all-diagnostics arm")
        require(best_metrics["sat"] >= 0.9948, "positive decision without saturated accuracy")
        require(best_metrics["hard"] >= 0.9906, "positive decision without hard accuracy")
        require(best_metrics["mixed"] >= 0.99245, "positive decision without mixed accuracy")
        require(best_metrics["ood"] >= 0.9906, "positive decision without OOD accuracy")
        require(best_metrics["gate_confusion"] >= 0.99245, "positive decision without gate-confusion accuracy")
        require(best_metrics["hidden"] >= 0.99, "positive decision without hidden-budget accuracy")
        require(best_metrics["external"] >= 0.99, "positive decision without external accuracy")
        require(best_metrics["indistinguishable_abstain"] >= 0.99, "positive decision without indistinguishable abstain")
        require(best_metrics["max_false_confidence"] <= 0.01, "positive decision with false confidence")
        require(best_metrics["mixed"] > exact(mixed, "SHUFFLED_DIAGNOSTIC_CONTROL"), "shuffled diagnostic control not worse")
        require(best_metrics["mixed"] > exact(mixed, "RANDOM_DIAGNOSTIC_CONTROL"), "random diagnostic control not worse")
        require(best_metrics["mixed"] > exact(mixed, "DIAGNOSTIC_ABLATION_CONTROL"), "diagnostic ablation control not worse")
    elif decision["decision"] == "hybrid_diagnostic_migration_positive":
        require(best == "HYBRID_SYMBOLIC_RUST_DIAGNOSTICS_GATE", "hybrid decision did not choose hybrid")
        require(best_metrics["mixed"] >= 0.99245, "hybrid positive without mixed accuracy")
    elif decision["decision"] == "invalid_diagnostic_truth_leak":
        require(truth_audit["fair_arms_with_truth_leak"], "truth leak decision without fair leak evidence")
    elif decision["decision"] == "diagnostic_component_migration_not_confirmed":
        require(
            all_diag_metrics["mixed"] < 0.99245
            or all_diag_metrics["hidden"] < 0.99
            or all_diag_metrics["external"] < 0.99
            or all_diag_metrics["indistinguishable_abstain"] < 0.99,
            "not-confirmed decision lacks failure evidence",
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "best_arm": best,
                "best_metrics": best_metrics,
                "all_diagnostics_metrics": all_diag_metrics,
                "mixed_estimator_accuracy": estimator_accuracy,
                "fallback_rows": aggregate["fallback_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
