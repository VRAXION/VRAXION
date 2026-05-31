#!/usr/bin/env python3
"""Checker for D64S score-vector structure repair."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d64b_upstream_manifest.json",
    "d63_upstream_manifest.json",
    "d62_upstream_manifest.json",
    "partial_test_metrics_by_track.json",
    "partial_ood_metrics_by_track.json",
    "score_structure_dependency_report.json",
    "shuffle_control_matrix_report.json",
    "structure_gap_report.json",
    "candidate_vs_shape_report.json",
    "support_coherence_report.json",
    "counterfactual_delta_report.json",
    "cluster_structure_report.json",
    "truth_leak_audit_report.json",
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

ARMS = [
    "D64B_CALIBRATED_REPLAY",
    "CANDIDATE_IDENTITY_LAYER",
    "SCORE_SHAPE_ONLY_LAYER",
    "SUPPORT_COHERENCE_LAYER",
    "COUNTERFACTUAL_DELTA_LAYER",
    "CLUSTER_STRUCTURE_LAYER",
    "FULL_STRUCTURE_AWARE_LAYER",
    "CANDIDATE_ID_SHUFFLE",
    "TOPK_VALUE_SHUFFLE",
    "MARGIN_PRESERVING_SHUFFLE",
    "ENTROPY_PRESERVING_SHUFFLE",
    "SUPPORT_ORDER_SHUFFLE",
    "SUPPORT_COHERENCE_BREAK",
    "COUNTERFACTUAL_DELTA_SHUFFLE",
    "CLUSTER_STRUCTURE_SHUFFLE",
    "FULL_SCORE_NOISE_CONTROL",
    "RANDOM_DIAGNOSTIC_CONTROL",
    "DIAGNOSTIC_ABLATION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]

FAIR_ARMS = {
    "D64B_CALIBRATED_REPLAY",
    "CANDIDATE_IDENTITY_LAYER",
    "SCORE_SHAPE_ONLY_LAYER",
    "SUPPORT_COHERENCE_LAYER",
    "COUNTERFACTUAL_DELTA_LAYER",
    "CLUSTER_STRUCTURE_LAYER",
    "FULL_STRUCTURE_AWARE_LAYER",
}

CONTROL_ARMS = {
    "CANDIDATE_ID_SHUFFLE",
    "TOPK_VALUE_SHUFFLE",
    "MARGIN_PRESERVING_SHUFFLE",
    "ENTROPY_PRESERVING_SHUFFLE",
    "SUPPORT_ORDER_SHUFFLE",
    "SUPPORT_COHERENCE_BREAK",
    "COUNTERFACTUAL_DELTA_SHUFFLE",
    "CLUSTER_STRUCTURE_SHUFFLE",
    "FULL_SCORE_NOISE_CONTROL",
    "RANDOM_DIAGNOSTIC_CONTROL",
    "DIAGNOSTIC_ABLATION_CONTROL",
}

REFERENCE_ONLY_ARMS = {"TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"}

ALLOWED_DECISIONS = {
    "score_vector_structure_repair_confirmed",
    "score_shape_dependency_confirmed_candidate_identity_not_required",
    "score_vector_structure_dependency_not_confirmed",
    "invalid_score_structure_truth_leak",
}

ROW_FIELDS = [
    "row_id",
    "seed",
    "split",
    "track",
    "mixed_source_track",
    "arm",
    "selected_action",
    "support_regime",
    "truth_joint",
    "pred_joint",
    "exact_joint_correct",
    "total_support_used",
    "counter_support_used",
    "external_test_used",
    "abstained",
    "false_confidence",
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
    "rust_diagnostic_estimator_invoked",
    "rust_diagnostic_python_fallback_used",
]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_guardrails(repo_root):
    source = (repo_root / "scripts/probes/run_d64s_score_vector_structure_repair.py").read_text(encoding="utf-8")
    compact = source.replace(" ", "").replace("\n", "").lower()
    for needle, message in [
        ("hash(", "Python hash forbidden"),
        ("random.random()<", "fake random threshold sampling forbidden"),
        ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
        ("expected_selected", "expected_selected feature forbidden"),
        ("openai", "external API reference forbidden"),
        ("modal", "Modal reference forbidden"),
    ]:
        require(needle not in compact, f"{message} in D64S runner")
    require("run_rust_estimator_bridge" in source, "D64S runner missing Rust diagnostic estimator bridge")
    require("FULL_STRUCTURE_AWARE_LAYER" in source, "D64S runner missing full structure arm")
    require("SUPPORT_COHERENCE_BREAK" in source, "D64S runner missing support coherence break")
    require("TRUTH_LEAK_SENTINEL_REFERENCE_ONLY" in source, "D64S runner missing truth leak sentinel")


def inspect_jsonl(path, strict):
    rows = 0
    arms = set()
    tracks = set()
    regimes = set()
    ref_forbidden_rows = 0
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing fields {missing}")
            require(set(row["diagnostic_estimates"]) == DIAGNOSTICS, f"{path} row {idx} diagnostic estimates mismatch")
            require(set(row["diagnostic_targets"]) == DIAGNOSTICS, f"{path} row {idx} diagnostic targets mismatch")
            require(set(row["diagnostic_correct"]) == DIAGNOSTICS, f"{path} row {idx} diagnostic correct mismatch")
            arms.add(row["arm"])
            tracks.add(row["track"])
            regimes.add(row["support_regime"])
            if row["arm"] in FAIR_ARMS:
                require(row["fair_arm"] is True, f"{path} row {idx} fair arm flag missing")
                require(row["diagnostic_used_forbidden_feature"] is False, f"{path} row {idx} fair arm used forbidden feature")
                require(row["proxy_input_violation"] is False, f"{path} row {idx} fair arm used proxy input")
                require(not (set(row["score_vector_inputs"]) & FORBIDDEN_RUST_INPUT_FEATURES), f"{path} row {idx} forbidden score-vector input")
                require(row["reference_only_arm"] is False, f"{path} row {idx} fair arm marked reference-only")
            if row["arm"] in FAIR_ARMS | CONTROL_ARMS:
                require(row["rust_diagnostic_estimator_invoked"] is True, f"{path} row {idx} Rust estimator not invoked")
                require(row["rust_diagnostic_python_fallback_used"] is False, f"{path} row {idx} diagnostic fallback used")
            if row["arm"] in REFERENCE_ONLY_ARMS and row["diagnostic_used_forbidden_feature"]:
                ref_forbidden_rows += 1
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    if strict:
        require(TRACKS.issubset(tracks), f"{path} missing tracks {TRACKS - tracks}")
        require(REGIMES.issubset(regimes), f"{path} missing regimes {REGIMES - regimes}")
    require(ref_forbidden_rows > 0, f"{path} reference-only sentinel was not exercised")


def exact(track_metrics, arm):
    return float(track_metrics["by_arm_core"][arm]["exact_joint_accuracy"])


def positive_like(summary, d64b_best):
    return (
        summary["SATURATED_STABILITY"]["exact"] >= float(d64b_best["SATURATED_STABILITY"]["exact"]) - 0.003
        and summary["HARD_CAP8_LEARNING"]["exact"] >= float(d64b_best["HARD_CAP8_LEARNING"]["exact"]) - 0.003
        and summary["MIXED_EVAL"]["exact"] >= float(d64b_best["MIXED_EVAL"]["exact"]) - 0.003
        and summary["OOD_CONTEXT_SHIFT"]["exact"] >= float(d64b_best["OOD_CONTEXT_SHIFT"]["exact"]) - 0.003
        and summary["ADVERSARIAL_GATE_CONFUSION"]["exact"] >= float(d64b_best["ADVERSARIAL_GATE_CONFUSION"]["exact"]) - 0.003
        and summary["MIXED_EVAL"]["corr"] >= float(d64b_best["MIXED_EVAL"]["corr"]) - 0.005
        and summary["MIXED_EVAL"]["adv"] >= float(d64b_best["MIXED_EVAL"]["adv"]) - 0.005
        and summary["EXTERNAL_TEST_REQUIRED"]["external"] >= 0.985
        and summary["INDISTINGUISHABLE_SUPPORT"]["abstain"] >= 0.99
        and max(float(summary[track]["false_confidence"]) for track in TRACKS) <= 0.01
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    require(not missing, f"missing artifacts: {missing}")

    queue = load(out / "queue.json")
    strict_row_coverage = queue.get("args", {}).get("scale_mode") not in {"micro", "sanity"}
    repo_root = Path(__file__).resolve().parents[2]
    source_guardrails(repo_root)
    inspect_jsonl(out / "row_outputs_test.jsonl", strict_row_coverage)
    inspect_jsonl(out / "row_outputs_ood.jsonl", strict_row_coverage)

    dataset = load(out / "dataset_manifest.json")
    d64b_upstream = load(out / "d64b_upstream_manifest.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    truth_audit = load(out / "truth_leak_audit_report.json")
    structure_gap = load(out / "structure_gap_report.json")
    report = (out / "report.md").read_text(encoding="utf-8")

    require(dataset["truth_hidden_from_controller_inputs"] is True, "truth not hidden from controller inputs")
    require(dataset["truth_hidden_from_diagnostic_estimators"] is True, "truth not hidden from diagnostic estimators")
    require(dataset["controller_only_not_formula_solver"] is True, "controller-only flag missing")
    require(dataset["score_vector_summary_inputs_used"] is True, "score-vector input flag missing")
    require("structure_groups" in dataset, "structure groups missing")
    require(d64b_upstream["expected_decision"] == "score_vector_shuffle_gap_insufficient", "bad D64B expected decision")
    require(d64b_upstream["expected_next"] == "D64S_SCORE_VECTOR_STRUCTURE_REPAIR", "bad D64B expected next")
    require(d64b_upstream["decision_present"] and d64b_upstream["summary_present"], "D64B upstream artifacts missing")
    require(truth_audit["fair_arms_with_truth_leak"] == [], "fair arms with truth leak")
    require(truth_audit["fair_arms_using_forbidden_features"] == [], "fair arms using forbidden features")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D64S only tests score-vector structure dependency" in decision["boundary"], "decision boundary missing")
    require("D64S only tests score-vector structure dependency" in report, "report boundary missing")
    require(aggregate["rust_path_invoked"] is True, "Rust path not invoked")
    require(aggregate["fallback_rows"] == 0, "Python fallback rows nonzero")
    require(aggregate["failed_jobs"] == [], "failed jobs present")
    require(summary["decision"] == decision["decision"], "summary/decision mismatch")

    test_tracks = aggregate["test_metrics_by_track"]
    for track, metrics in test_tracks.items():
        require(track in TRACKS, f"unexpected track {track}")
        require(set(ARMS).issubset(metrics["by_arm_core"].keys()), f"{track} missing arms")
        for arm in ARMS:
            require(REGIMES.issubset(metrics["by_arm_and_regime"][arm].keys()), f"{track} {arm} missing regimes")

    best = decision["best_arm"]
    require(best in FAIR_ARMS | REFERENCE_ONLY_ARMS, "best arm is not a D64S arm")
    if decision["decision"] != "invalid_score_structure_truth_leak":
        require(best in FAIR_ARMS, "non-leak decision chose reference-only arm")

    d64b_best = aggregate["d64b_best_summary"]
    full = aggregate["arm_summaries"]["FULL_STRUCTURE_AWARE_LAYER"]
    full_positive = positive_like(full, d64b_best)
    mixed = test_tracks["MIXED_EVAL"]
    full_mixed = exact(mixed, "FULL_STRUCTURE_AWARE_LAYER")
    max_gap = max(float(item["gap_vs_full_structure"]) for item in structure_gap.values())
    random_ablation_worse = (
        full_mixed > exact(mixed, "RANDOM_DIAGNOSTIC_CONTROL")
        and full_mixed > exact(mixed, "DIAGNOSTIC_ABLATION_CONTROL")
    )
    candidate_gap = float(structure_gap["CANDIDATE_ID_SHUFFLE"]["gap_vs_full_structure"])

    if decision["decision"] in {
        "score_vector_structure_repair_confirmed",
        "score_shape_dependency_confirmed_candidate_identity_not_required",
    }:
        require(best == "FULL_STRUCTURE_AWARE_LAYER", "positive D64S decision did not choose full structure arm")
        require(full_positive, "positive D64S decision without D64B-level metrics")
        require(max_gap >= 0.03, "positive D64S decision without structure gap")
        require(random_ablation_worse, "positive D64S decision without random/ablation controls worse")
    elif decision["decision"] == "score_vector_structure_dependency_not_confirmed":
        require((not full_positive) or max_gap < 0.03 or not random_ablation_worse, "not-confirmed decision lacks failure evidence")
    elif decision["decision"] == "invalid_score_structure_truth_leak":
        require(truth_audit["fair_arms_with_truth_leak"], "truth leak decision without fair leak evidence")

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "best_arm": best,
                "full_mixed_exact": full_mixed,
                "max_structure_gap": max_gap,
                "candidate_id_gap": candidate_gap,
                "fallback_rows": aggregate["fallback_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
