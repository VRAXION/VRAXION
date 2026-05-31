#!/usr/bin/env python3
"""Checker for D64 Rust sparse IPF diagnostic layer prototype."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d63_upstream_manifest.json",
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
    "ipf_diagnostic_definition_report.json",
    "score_vector_input_report.json",
    "proxy_leakage_audit_report.json",
    "diagnostic_estimator_accuracy_report.json",
    "calibration_report.json",
    "noisy_perturbation_report.json",
    "rust_estimator_mapping_report.json",
    "estimator_accuracy_report.json",
    "gate_with_ipf_diagnostics_report.json",
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
    "entropy_high",
    "margin_low",
    "collision_pressure",
    "support_independence_low",
    "support_effort_pressure",
    "counterfactual_pressure",
    "adversarial_pressure",
    "internal_unresolvable",
    "external_test_need",
}

FORBIDDEN_RUST_INPUT_FEATURES = {
    "support_budget_pressure_norm",
    "counterfactual_pressure_norm",
    "adversarial_pressure_norm",
    "internal_unresolvable_indicator",
    "external_channel_available",
    "support_regime",
    "track",
    "mixed_source_track",
    "row_id",
    "seed",
}

SCORE_NATIVE_DIAGNOSTICS = {
    "entropy_high",
    "margin_low",
    "collision_pressure",
    "support_independence_low",
}

ARMS = [
    "D63_SYMBOLIC_DIAGNOSTIC_REFERENCE",
    "RUST_SPARSE_ENTROPY_MARGIN_LAYER",
    "RUST_SPARSE_COLLISION_LAYER",
    "RUST_SPARSE_SUPPORT_INDEPENDENCE_LAYER",
    "RUST_SPARSE_COUNTERFACTUAL_PRESSURE_LAYER",
    "RUST_SPARSE_ADVERSARIAL_PRESSURE_LAYER",
    "RUST_SPARSE_UNRESOLVABLE_ESTIMATOR",
    "RUST_SPARSE_EXTERNAL_TEST_NEED_ESTIMATOR",
    "RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER",
    "HYBRID_SYMBOLIC_RUST_IPF_LAYER",
    "SHUFFLED_SCORE_VECTOR_CONTROL",
    "RANDOM_DIAGNOSTIC_CONTROL",
    "DIAGNOSTIC_ABLATION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]

FAIR_ARMS = {
    "D63_SYMBOLIC_DIAGNOSTIC_REFERENCE",
    "RUST_SPARSE_ENTROPY_MARGIN_LAYER",
    "RUST_SPARSE_COLLISION_LAYER",
    "RUST_SPARSE_SUPPORT_INDEPENDENCE_LAYER",
    "RUST_SPARSE_COUNTERFACTUAL_PRESSURE_LAYER",
    "RUST_SPARSE_ADVERSARIAL_PRESSURE_LAYER",
    "RUST_SPARSE_UNRESOLVABLE_ESTIMATOR",
    "RUST_SPARSE_EXTERNAL_TEST_NEED_ESTIMATOR",
    "RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER",
    "HYBRID_SYMBOLIC_RUST_IPF_LAYER",
}

RUST_DIAGNOSTIC_ARMS = FAIR_ARMS - {"D63_SYMBOLIC_DIAGNOSTIC_REFERENCE"}
CONTROL_ARMS = {"SHUFFLED_SCORE_VECTOR_CONTROL", "RANDOM_DIAGNOSTIC_CONTROL", "DIAGNOSTIC_ABLATION_CONTROL"}
REFERENCE_ONLY_ARMS = {"TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}

ALLOWED_DECISIONS = {
    "rust_sparse_ipf_diagnostic_layer_confirmed",
    "hybrid_ipf_diagnostic_layer_positive",
    "rust_sparse_ipf_diagnostic_layer_not_confirmed",
    "d64_instrumentation_too_proxy_like",
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
    "score_vector_inputs",
    "diagnostic_estimates",
    "diagnostic_targets",
    "diagnostic_correct",
    "diagnostic_used_forbidden_feature",
    "proxy_input_violation",
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
    source = (repo_root / "scripts/probes/run_d64_rust_sparse_ipf_diagnostic_layer_prototype.py").read_text(encoding="utf-8")
    compact = source.replace(" ", "").replace("\n", "").lower()
    for needle, message in [
        ("hash(", "Python hash forbidden"),
        ("random.random()<", "fake random threshold sampling forbidden"),
        ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
        ("expected_selected", "expected_selected feature forbidden"),
        ("openai", "external API reference forbidden"),
        ("modal", "Modal reference forbidden"),
    ]:
        require(needle not in compact, f"{message} in D64 runner")
    require("run_rust_estimator_bridge" in source, "D64 runner missing Rust diagnostic estimator bridge")
    require("Network::propagate_sparse" in source or "ensure_rust_multi_harness" in source, "D64 runner missing canonical Rust sparse path")
    require("TRUTH_LEAK_SENTINEL_REFERENCE_ONLY" in source, "D64 runner missing truth leak sentinel")
    require("FORBIDDEN_DIAGNOSTIC_FEATURES" in source, "D64 runner missing forbidden diagnostic feature list")
    require("FORBIDDEN_RUST_INPUT_FEATURES" in source, "D64 runner missing forbidden Rust input feature list")
    require("score_vector_inputs" in source, "D64 runner missing score-vector input logging")


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
                require(row["proxy_input_violation"] is False, f"{path} row {idx} fair arm used clean proxy input")
                require(not (set(row["score_vector_inputs"]) & FORBIDDEN_RUST_INPUT_FEATURES), f"{path} row {idx} score-vector inputs contain forbidden proxy")
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
    d63_upstream = load(out / "d63_upstream_manifest.json")
    definitions = load(out / "ipf_diagnostic_definition_report.json")
    score_inputs = load(out / "score_vector_input_report.json")
    proxy_audit = load(out / "proxy_leakage_audit_report.json")
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
    require(set(dataset["diagnostic_features"]) & FORBIDDEN_RUST_INPUT_FEATURES == set(), "diagnostic inputs contain clean proxy features")
    require(dataset["score_vector_summary_inputs_used"] is True, "score-vector summary input flag missing")
    require(dataset["clean_d63_proxy_inputs_used_by_rust_estimators"] is False, "clean D63 proxy input flag wrong")
    require(upstream["expected_decision"] == "policy_ensemble_learned_gate_confirmed", "bad D62 expected decision")
    require(upstream["expected_next"] == "D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_PROTOTYPE", "bad D62 next")
    require(upstream["decision_present"] and upstream["summary_present"], "D62 upstream artifacts missing")
    require(upstream["learned_gate_present"], "D62 learned gate missing")
    require(d63_upstream["expected_decision"] == "rust_sparse_ecf_diagnostic_component_migration_confirmed", "bad D63 expected decision")
    require(d63_upstream["expected_next"] == "D64_RUST_SPARSE_IPF_DIAGNOSTIC_LAYER_PROTOTYPE", "bad D63 next")
    require(d63_upstream["decision_present"] and d63_upstream["summary_present"], "D63 upstream artifacts missing")
    require(definitions["truth_labels_used"] is False, "diagnostic definitions use truth labels")
    require(definitions["support_regime_labels_used"] is False, "diagnostic definitions use support regime labels")
    require(definitions["track_labels_used_by_fair_estimators"] is False, "diagnostic definitions use track labels")
    require(definitions["clean_d63_proxy_features_in_rust_input"] is False, "diagnostic definitions use clean D63 proxy input")
    require(set(score_inputs["summary_features"]) & FORBIDDEN_RUST_INPUT_FEATURES == set(), "score input report contains forbidden proxy")
    require(proxy_audit["violating_input_features"] == [], "proxy leakage audit found violating inputs")
    require(proxy_audit["uses_clean_d63_proxy_flags_as_rust_inputs"] is False, "proxy leakage audit says clean proxy flags are Rust inputs")
    require(estimator_map["rust_path"].startswith("canonical Network::propagate_sparse"), "bad Rust estimator mapping path")
    require(truth_audit["fair_arms_with_truth_leak"] == [], "fair arms with truth leak")
    require(truth_audit["fair_arms_using_forbidden_features"] == [], "fair arms using forbidden features")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D64 only tests a Rust sparse IPF diagnostic layer" in decision["boundary"], "decision boundary missing")
    require("D64 only tests a Rust sparse IPF diagnostic layer" in report, "report boundary missing")
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
    full = "RUST_SPARSE_FULL_IPF_DIAGNOSTIC_LAYER"
    hybrid = "HYBRID_SYMBOLIC_RUST_IPF_LAYER"
    full_metrics = {
        "sat": exact(sat, full),
        "hard": exact(hard, full),
        "mixed": exact(mixed, full),
        "ood": exact(ood, full),
        "gate_confusion": exact(gate_conf, full),
        "hidden": exact(hidden, full),
        "external": float(aggregate["arm_summaries"][full]["EXTERNAL_TEST_REQUIRED"]["external"]),
        "indistinguishable_abstain": float(aggregate["arm_summaries"][full]["INDISTINGUISHABLE_SUPPORT"]["abstain"]),
    }
    estimator_accuracy = test_tracks["MIXED_EVAL"]["diagnostic_accuracy"][full]
    score_native_rows = min(estimator_accuracy[name]["rows"] for name in SCORE_NATIVE_DIAGNOSTICS)
    score_native_mean = sum(float(estimator_accuracy[name]["accuracy"]) for name in SCORE_NATIVE_DIAGNOSTICS) / len(SCORE_NATIVE_DIAGNOSTICS)
    if score_native_rows >= 500:
        require(score_native_mean >= 0.60, "score-native diagnostic accuracy mean below 0.60")

    d62_best = aggregate["d62_best_summary"]

    def positive_like(metrics):
        return (
            metrics["sat"] >= float(d62_best["SATURATED_STABILITY"]["exact"]) - 0.007
            and metrics["hard"] >= float(d62_best["HARD_CAP8_LEARNING"]["exact"]) - 0.007
            and metrics["mixed"] >= float(d62_best["MIXED_EVAL"]["exact"]) - 0.007
            and metrics["ood"] >= float(d62_best["OOD_CONTEXT_SHIFT"]["exact"]) - 0.007
            and metrics["gate_confusion"] >= float(d62_best["ADVERSARIAL_GATE_CONFUSION"]["exact"]) - 0.007
            and metrics["hidden"] >= 0.985
            and metrics["external"] >= 0.985
            and metrics["indistinguishable_abstain"] >= 0.99
        )

    controls_worse_than_best = (
        best_metrics["mixed"] > exact(mixed, "SHUFFLED_SCORE_VECTOR_CONTROL")
        and best_metrics["mixed"] > exact(mixed, "RANDOM_DIAGNOSTIC_CONTROL")
        and best_metrics["mixed"] > exact(mixed, "DIAGNOSTIC_ABLATION_CONTROL")
    )

    if decision["decision"] == "rust_sparse_ipf_diagnostic_layer_confirmed":
        require(best == full, "full IPF confirmation did not choose full Rust IPF diagnostic arm")
        require(positive_like(best_metrics), "positive decision without D64 gate metrics")
        require(best_metrics["max_false_confidence"] <= 0.01, "positive decision with false confidence")
        require(controls_worse_than_best, "controls not worse than best D64 arm")
    elif decision["decision"] == "hybrid_ipf_diagnostic_layer_positive":
        require(best == hybrid, "hybrid decision did not choose hybrid IPF arm")
        require(positive_like(best_metrics), "hybrid positive without D64 gate metrics")
        require(best_metrics["max_false_confidence"] <= 0.01, "hybrid positive with false confidence")
        require(controls_worse_than_best, "controls not worse than hybrid D64 arm")
    elif decision["decision"] == "invalid_diagnostic_truth_leak":
        require(truth_audit["fair_arms_with_truth_leak"], "truth leak decision without fair leak evidence")
    elif decision["decision"] == "d64_instrumentation_too_proxy_like":
        require(proxy_audit["uses_clean_d63_proxy_flags_as_rust_inputs"] is True or proxy_audit["violating_input_features"], "proxy-like decision without proxy audit evidence")
    elif decision["decision"] == "rust_sparse_ipf_diagnostic_layer_not_confirmed":
        require(
            not positive_like(full_metrics)
            or full_metrics["mixed"] <= exact(mixed, "SHUFFLED_SCORE_VECTOR_CONTROL")
            or full_metrics["mixed"] <= exact(mixed, "RANDOM_DIAGNOSTIC_CONTROL")
            or full_metrics["mixed"] <= exact(mixed, "DIAGNOSTIC_ABLATION_CONTROL"),
            "not-confirmed decision lacks failure evidence",
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "best_arm": best,
                "best_metrics": best_metrics,
                "full_ipf_metrics": full_metrics,
                "mixed_estimator_accuracy": estimator_accuracy,
                "fallback_rows": aggregate["fallback_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
