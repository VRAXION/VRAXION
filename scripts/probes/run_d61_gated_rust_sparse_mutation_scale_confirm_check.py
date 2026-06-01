#!/usr/bin/env python3
"""Checker for D61 gated Rust sparse mutation scale confirm."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d60s_upstream_manifest.json",
    "gate_training_report.json",
    "partial_test_metrics_by_track.json",
    "partial_ood_metrics_by_track.json",
    "track_metrics_test_SATURATED_STABILITY.json",
    "track_metrics_test_HARD_CAP8_LEARNING.json",
    "track_metrics_test_MIXED_EVAL.json",
    "track_metrics_test_OOD_CONTEXT_SHIFT.json",
    "track_metrics_test_ADVERSARIAL_GATE_CONFUSION.json",
    "track_metrics_ood_SATURATED_STABILITY.json",
    "track_metrics_ood_HARD_CAP8_LEARNING.json",
    "track_metrics_ood_MIXED_EVAL.json",
    "track_metrics_ood_OOD_CONTEXT_SHIFT.json",
    "track_metrics_ood_ADVERSARIAL_GATE_CONFUSION.json",
    "gate_feature_audit_report.json",
    "gate_routing_accuracy_report.json",
    "truth_leak_audit_report.json",
    "multi_track_scale_report.json",
    "ood_context_shift_report.json",
    "adversarial_gate_confusion_report.json",
    "gate_ablation_report.json",
    "policy_comparison_report.json",
    "support_cost_frontier_report.json",
    "false_confidence_report.json",
    "rust_invocation_report.json",
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

RUST_CONTROLLER_ARMS = [
    "D59_REFERENCE",
    "D60_HARD_POLICY_REPLAY",
    "SINGLE_POLICY_MULTI_ENV_CONTROL",
    "THRESHOLD_ABLATION",
    "REWIRE_ABLATION",
]

FAIR_GATED_ARMS = [
    "DUAL_POLICY_GATED_CONTROLLER",
    "CONTEXT_GATED_POLICY_ENSEMBLE",
    "LEARNED_GATE_MUTATION_CONTROLLER",
]

REFERENCE_ONLY_ARMS = [
    "D60S_DUAL_POLICY_GATED_REPLAY",
    "ORACLE_TRACK_GATE_REFERENCE_ONLY",
    "TRUTH_LEAK_SENTINEL_CONTROL",
]

GATE_CONTROL_ARMS = [
    "RANDOM_GATE_CONTROL",
    "WRONG_GATE_CONTROL",
    "GATE_ABLATION",
]

POLICY_CONTROL_ARMS = [
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "SPIKE_SHUFFLE_CONTROL",
]

ARMS = RUST_CONTROLLER_ARMS + FAIR_GATED_ARMS + REFERENCE_ONLY_ARMS + GATE_CONTROL_ARMS + POLICY_CONTROL_ARMS
RUST_REQUIRED_ARMS = RUST_CONTROLLER_ARMS + FAIR_GATED_ARMS + REFERENCE_ONLY_ARMS + GATE_CONTROL_ARMS

ALLOWED_DECISIONS = {
    "gated_rust_sparse_mutation_scale_confirmed",
    "handcoded_gate_required",
    "invalid_gate_truth_leak_detected",
    "gated_rust_sparse_mutation_scale_not_confirmed",
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
    "gate_features",
    "gate_expected_policy",
    "gate_selected_policy",
    "gate_policy_correct",
    "gate_basis",
    "gate_used_truth_label",
    "gate_used_forbidden_feature",
    "fair_gate_arm",
    "reference_only_arm",
    "ood_context_shift",
    "adversarial_gate_confusion",
]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_guardrails(repo_root):
    source = (repo_root / "scripts/probes/run_d61_gated_rust_sparse_mutation_scale_confirm.py").read_text(encoding="utf-8")
    compact = source.replace(" ", "").replace("\n", "").lower()
    for needle, message in [
        ("hash(", "Python hash forbidden"),
        ("random.random()<", "fake random threshold sampling forbidden"),
        ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
        ("expected_selected", "expected_selected feature forbidden"),
        ("openai", "external API reference forbidden"),
        ("modal", "Modal reference forbidden"),
    ]:
        require(needle not in compact, f"{message} in D61 runner")
    require("run_rust_multi_bridge" in source, "D61 runner does not invoke canonical Rust bridge")
    require("ALLOWED_GATE_FEATURES" in source and "FORBIDDEN_GATE_FEATURES" in source, "D61 runner missing gate feature audit lists")
    require("TRUTH_LEAK_SENTINEL_CONTROL" in source, "D61 runner missing truth leak sentinel")
    require("adversarial_gate_decoy_norm" not in source.split("candidate_features =", 1)[1].split("]", 1)[0], "learned gate candidate features include decoy")


def inspect_jsonl(path):
    rows = 0
    arms = set()
    tracks = set()
    regimes = set()
    rust_rows = {arm: 0 for arm in RUST_REQUIRED_ARMS}
    fallback_rows = {arm: 0 for arm in RUST_REQUIRED_ARMS}
    fair_gate_rows = {arm: 0 for arm in FAIR_GATED_ARMS}
    fair_gate_policies = set()
    ref_forbidden_rows = 0
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing fields {missing}")
            require(row["total_support_used"] >= row["original_support_used"], f"{path} row {idx} support accounting invalid")
            require(row["total_support_used"] >= row["counter_support_used"], f"{path} row {idx} counter support accounting invalid")
            require(isinstance(row["gate_features"], dict), f"{path} row {idx} gate_features not dict")
            arms.add(row["arm"])
            tracks.add(row["track"])
            regimes.add(row["support_regime"])
            if row["arm"] in RUST_REQUIRED_ARMS:
                if row["rust_network_path_invoked"] and row["rust_propagate_sparse_called"]:
                    rust_rows[row["arm"]] += 1
                if row["python_fallback_used"]:
                    fallback_rows[row["arm"]] += 1
            if row["arm"] in FAIR_GATED_ARMS:
                require(row["fair_gate_arm"] is True, f"{path} row {idx} fair gate flag missing")
                require(row["reference_only_arm"] is False, f"{path} row {idx} fair gate marked reference-only")
                require(row["gate_selected_policy"] in {"D59_REFERENCE", "D60_HARD_POLICY_REPLAY"}, f"{path} row {idx} bad selected policy")
                require(row["gate_used_truth_label"] is False, f"{path} row {idx} fair gate used truth label")
                require(row["gate_used_forbidden_feature"] is False, f"{path} row {idx} fair gate used forbidden feature")
                require("adversarial_gate_decoy_norm" not in row["gate_features"], f"{path} row {idx} decoy exposed to fair gate")
                fair_gate_rows[row["arm"]] += 1
                fair_gate_policies.add(row["gate_selected_policy"])
            if row["arm"] in REFERENCE_ONLY_ARMS and (row["gate_used_truth_label"] or row["gate_used_forbidden_feature"]):
                ref_forbidden_rows += 1
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(TRACKS.issubset(tracks), f"{path} missing tracks {TRACKS - tracks}")
    require(REGIMES.issubset(regimes), f"{path} missing regimes {REGIMES - regimes}")
    for arm in RUST_REQUIRED_ARMS:
        require(rust_rows[arm] > 0, f"{path} {arm} did not invoke Rust")
        require(fallback_rows[arm] == 0, f"{path} {arm} used Python fallback")
    for arm in FAIR_GATED_ARMS:
        require(fair_gate_rows[arm] > 0, f"{path} missing fair gate rows for {arm}")
    require({"D59_REFERENCE", "D60_HARD_POLICY_REPLAY"}.issubset(fair_gate_policies), f"{path} fair gates did not exercise both policies")
    require(ref_forbidden_rows > 0, f"{path} reference-only forbidden/truth sentinel was not exercised")


def require_track(metrics, track):
    require(track in metrics, f"missing track metrics {track}")
    item = metrics[track]
    require(set(ARMS).issubset(item["by_arm_core"].keys()), f"{track} missing arms in core metrics")
    for arm in ARMS:
        require(REGIMES.issubset(item["by_arm_and_regime"][arm].keys()), f"{track} {arm} missing regimes")
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
    upstream = load(out / "d60s_upstream_manifest.json")
    gate_training = load(out / "gate_training_report.json")
    feature_audit = load(out / "gate_feature_audit_report.json")
    truth_audit = load(out / "truth_leak_audit_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    report = (out / "report.md").read_text(encoding="utf-8")

    require(dataset["truth_hidden_from_controller_inputs"] is True, "truth not hidden from controller inputs")
    require(dataset["truth_hidden_from_fair_gate_inputs"] is True, "truth not hidden from fair gate inputs")
    require(dataset["controller_only_not_formula_solver"] is True, "controller-only flag missing")
    require(dataset["formula_solver_learning_used"] is False, "formula solver learning flag wrong")
    require(set(dataset["tracks"]) == TRACKS, "dataset tracks mismatch")
    require("adversarial_gate_decoy_norm" not in dataset["allowed_gate_features"], "decoy listed as allowed gate feature")
    require(upstream["expected_decision"] == "gated_policy_required_for_no_forgetting", "bad D60S expected decision")
    require(upstream["expected_next"] == "D61_GATED_RUST_SPARSE_MUTATION_SCALE_CONFIRM", "bad D60S next")
    require(upstream["decision_present"] and upstream["summary_present"], "D60S upstream artifacts missing")
    require(upstream["d59_reference_loaded"] and upstream["d60_hard_policy_loaded"], "D60S controllers missing")
    require(gate_training["allowed_features_only"] is True, "learned gate used non-allowed feature")
    require(gate_training["learned_gate"]["feature"] in dataset["allowed_gate_features"], "learned gate feature not allowed")
    require(feature_audit["track_label_used_by_fair_gates"] is False, "fair gate uses track label")
    require(feature_audit["support_regime_used_by_fair_gates"] is False, "fair gate uses support regime")
    require(feature_audit["truth_fields_used_by_fair_gates"] is False, "fair gate uses truth fields")
    require(truth_audit["fair_arms_with_truth_leak"] == [], "fair arms with truth leak")
    require(truth_audit["fair_arms_using_forbidden_features"] == [], "fair arms using forbidden features")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D61 only scale-confirms" in decision["boundary"], "decision boundary missing")
    require("D61 only scale-confirms" in report, "report boundary missing")
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
    confusion = test_tracks["ADVERSARIAL_GATE_CONFUSION"]
    best = decision["best_arm"]
    require(best in FAIR_GATED_ARMS + REFERENCE_ONLY_ARMS, "best arm is not a gate/controller arm")
    if decision["decision"] != "invalid_gate_truth_leak_detected":
        require(best in FAIR_GATED_ARMS, "non-leak decision chose reference-only arm")

    saturated_floor = float(aggregate["saturated_floor"])
    arm_summary = aggregate["arm_summaries"][best]
    max_false_conf = max(float(arm_summary[track]["false_confidence"]) for track in TRACKS)
    min_abstain = min(float(arm_summary[track]["abstain"]) for track in TRACKS)
    best_metrics = {
        "sat": exact(sat, best),
        "hard": exact(hard, best),
        "mixed": exact(mixed, best),
        "ood": exact(ood, best),
        "gate_confusion": exact(confusion, best),
        "hard_gain_vs_D58": float(arm_summary["hard_gain_vs_D58"]),
        "sat_regression_vs_D59": float(arm_summary["saturated_regression_vs_D59"]),
        "max_false_confidence": max_false_conf,
        "min_abstain": min_abstain,
        "support_mixed": support(mixed, best),
    }

    if decision["decision"] in {"gated_rust_sparse_mutation_scale_confirmed", "handcoded_gate_required"}:
        require(best_metrics["sat"] >= saturated_floor, "positive decision without saturated stability")
        require(best_metrics["hard"] >= 0.99, "positive decision without hard accuracy")
        require(best_metrics["mixed"] >= 0.995, "positive decision without mixed accuracy")
        require(best_metrics["ood"] >= 0.99, "positive decision without OOD accuracy")
        require(best_metrics["gate_confusion"] >= 0.99, "positive decision without gate-confusion accuracy")
        require(best_metrics["hard_gain_vs_D58"] >= 0.30, "positive decision without hard gain")
        require(best_metrics["sat_regression_vs_D59"] >= -0.002, "positive decision with saturated regression")
        require(max_false_conf <= 0.01, "positive decision with false confidence")
        require(min_abstain >= 0.99, "positive decision without indistinguishable abstain")
        require(best_metrics["mixed"] > exact(mixed, "RANDOM_GATE_CONTROL"), "random gate control not worse")
        require(best_metrics["mixed"] > exact(mixed, "WRONG_GATE_CONTROL"), "wrong gate control not worse")
        require(best_metrics["mixed"] > exact(mixed, "GATE_ABLATION"), "gate ablation not worse")
        require(best_metrics["mixed"] > exact(mixed, "RANDOM_POLICY_CONTROL"), "random policy control not worse")
        require(best_metrics["mixed"] > exact(mixed, "SPIKE_SHUFFLE_CONTROL"), "spike shuffle control not worse")
        if decision["decision"] == "gated_rust_sparse_mutation_scale_confirmed":
            require(best == "LEARNED_GATE_MUTATION_CONTROLLER", "scale confirmed decision did not choose learned gate")
        else:
            require(best in {"DUAL_POLICY_GATED_CONTROLLER", "CONTEXT_GATED_POLICY_ENSEMBLE"}, "handcoded gate decision did not choose handcoded fair gate")
    elif decision["decision"] == "invalid_gate_truth_leak_detected":
        require(truth_audit["fair_arms_with_truth_leak"], "truth leak decision without fair leak evidence")
    elif decision["decision"] == "gated_rust_sparse_mutation_scale_not_confirmed":
        require(
            best_metrics["hard"] < 0.99
            or best_metrics["mixed"] < 0.995
            or best_metrics["ood"] < 0.99
            or max_false_conf > 0.01
            or min_abstain < 0.99,
            "not-confirmed decision lacks failure evidence",
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "best_arm": best,
                "best_metrics": best_metrics,
                "learned_gate": gate_training["learned_gate"],
                "fallback_rows": aggregate["fallback_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
