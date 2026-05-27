#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "d40_upstream_manifest.json",
    "dataset_manifest.json",
    "dataset_invariant_report.json",
    "support_evidence_audit.json",
    "ood_rule_invariance_audit.json",
    "random_baseline_report.json",
    "query_only_monolithic_baseline_report.json",
    "support_evidence_oracle_rule_selector_report.json",
    "true_family_oracle_upper_bound_report.json",
    "mutable_learned_rule_family_inference_report.json",
    "mutable_learned_rule_inference_plus_router_report.json",
    "shuffled_support_evidence_control_report.json",
    "no_support_evidence_control_report.json",
    "wrong_support_control_report.json",
    "same_query_different_support_counterfactual_report.json",
    "support_count_generalization_report.json",
    "support_margin_strata_report.json",
    "noisy_support_report.json",
    "rule_inference_matrix_report.json",
    "rule_identity_alignment_report.json",
    "rule_matrix_stability_report.json",
    "mutation_acceptance_report.json",
    "convergence_report.json",
    "seed_variance_report.json",
    "per_seed_report.json",
    "per_family_report.json",
    "pocket_confusion_matrix.json",
    "family_confusion_matrix.json",
    "score_margin_report.json",
    "arm_comparison_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
ARMS = [
    "RANDOM_BASELINE",
    "QUERY_ONLY_MONOLITHIC_BASELINE",
    "SUPPORT_EVIDENCE_ORACLE_RULE_SELECTOR",
    "TRUE_FAMILY_ORACLE_UPPER_BOUND",
    "MUTABLE_LEARNED_RULE_FAMILY_INFERENCE",
    "MUTABLE_LEARNED_RULE_INFERENCE_PLUS_LEARNED_ROUTER",
    "SHUFFLED_SUPPORT_EVIDENCE_CONTROL",
    "NO_SUPPORT_EVIDENCE_CONTROL",
    "WRONG_SUPPORT_CONTROL",
    "SAME_QUERY_DIFFERENT_SUPPORT_COUNTERFACTUAL",
    "SUPPORT_COUNT_GENERALIZATION_REPORT_ONLY",
    "SUPPORT_MARGIN_STRATA_REPORT_ONLY",
]
ALLOWED_DECISIONS = {
    "d41_dataset_invariant_failure",
    "d41_support_evidence_not_inferable",
    "d41_ood_rule_invariance_failure",
    "rule_family_inference_router_scale_confirmed",
    "rule_family_inference_mean_positive_support_strata_gap",
    "rule_inference_positive_but_counterfactual_control_failed",
    "rule_family_inference_scale_partial_signal",
    "rule_family_inference_scale_not_confirmed",
}


def load_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if "target/pilot_wave/d41_rule_family_inference_router_scale_confirm/" not in out.as_posix():
        raise SystemExit("--out must stay under target/pilot_wave/d41_rule_family_inference_router_scale_confirm/")
    missing = [name for name in REQUIRED if not (out / name).exists()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    upstream = load_json(out / "d40_upstream_manifest.json")
    assert upstream["d40_decision"] == "rule_family_inference_router_prototype_positive"
    assert upstream["d40_verdict"] == "D40_RULE_FAMILY_INFERENCE_ROUTER_PROTOTYPE_POSITIVE"
    assert upstream["d40_next"] == "D41_RULE_FAMILY_INFERENCE_ROUTER_SCALE_CONFIRM"
    assert upstream["d40_learned_selected_pocket_test_accuracy"] == 1.0
    assert upstream["d40_learned_selected_pocket_ood_accuracy"] == 1.0
    assert upstream["d40_rule_family_test_accuracy"] == 1.0
    assert upstream["d40_rule_family_ood_accuracy"] == 1.0
    assert upstream["d40_wrong_support_follow_rate"] == 1.0
    assert upstream["d40_same_query_different_support_accuracy"] == 1.0

    inv = load_json(out / "dataset_invariant_report.json")
    assert inv["duplicate_target_pocket_rate"] == 0.0
    assert inv["missing_target_pocket_rate"] == 0.0
    assert inv["expected_selected_points_to_target_rate"] == 1.0
    assert inv["ambiguous_support_rate"] == 0.0
    assert inv["multi_family_support_tie_rate"] == 0.0
    assert inv["intended_family_unique_evidence_rate"] == 1.0
    assert inv["counterfactual_target_collision_rate"] == 0.0
    assert inv["wrong_support_query_mismatch_rate"] == 1.0

    support = load_json(out / "support_evidence_audit.json")
    assert support["support_rule_oracle_test_accuracy"] == 1.0
    assert support["support_rule_oracle_ood_accuracy"] == 1.0
    assert support["support_selected_pocket_oracle_test_accuracy"] == 1.0
    assert support["support_selected_pocket_oracle_ood_accuracy"] == 1.0

    audit = load_json(out / "ood_rule_invariance_audit.json")
    assert audit["support_rule_oracle_test_accuracy"] == 1.0
    assert audit["support_rule_oracle_ood_accuracy"] == 1.0
    assert audit["support_selected_pocket_oracle_test_accuracy"] == 1.0
    assert audit["support_selected_pocket_oracle_ood_accuracy"] == 1.0
    assert audit["known_rule_oracle_test_accuracy"] == 1.0
    assert audit["known_rule_oracle_ood_accuracy"] == 1.0
    assert audit["ood_label_rule_changed"] is False

    src = Path("scripts/probes/run_d41_rule_family_inference_router_scale_confirm.py").read_text()
    compact = src.replace(" ", "")
    for pattern in ["hit=random.random()<p", "random.random()<p", "base={", "fixed_base_accuracy"]:
        assert pattern not in compact
    assert "hash(" not in src

    aggregate = load_json(out / "aggregate_metrics.json")
    for arm in ARMS:
        assert arm in aggregate["arms"]
        arm_report = aggregate["arms"][arm]
        for metric in [
            "train_accuracy",
            "test_accuracy",
            "ood_accuracy",
            "min_seed_test_accuracy",
            "min_seed_ood_accuracy",
            "per_support_count_accuracy",
            "per_margin_strata_accuracy",
            "per_family_accuracy",
            "failed_seed_count",
        ]:
            assert metric in arm_report
        assert set(arm_report["per_support_count_accuracy"]) == {"support_count_1", "support_count_2", "support_count_3", "support_count_5"}
        assert set(arm_report["per_margin_strata_accuracy"]) == {"margin_low", "margin_high", "clean_unanimous", "noisy_majority"}
        assert set(arm_report["per_family_accuracy"]) == {"row", "col", "pair", "mirror", "diag"}

    learned = load_json(out / "mutable_learned_rule_family_inference_report.json")
    for metric in [
        "learned_rule_family_train_accuracy",
        "learned_rule_family_test_accuracy",
        "learned_rule_family_ood_accuracy",
        "learned_selected_pocket_train_accuracy",
        "learned_selected_pocket_test_accuracy",
        "learned_selected_pocket_ood_accuracy",
        "min_seed_learned_test_accuracy",
        "min_seed_learned_ood_accuracy",
        "min_support_count_accuracy",
        "min_margin_strata_accuracy",
        "rule_identity_alignment_score_mean",
        "rule_identity_alignment_score_min",
        "rule_diagonal_mass_mean",
        "rule_off_diagonal_mass_mean",
        "rule_argmax_mapping_by_seed",
        "rule_entropy_mean",
        "accepted_mutations_by_type",
        "rejected_mutations_by_type",
        "mutation_acceptance_rate",
        "convergence_generation_median",
        "seed_variance",
    ]:
        assert metric in learned

    matrix = load_json(out / "rule_inference_matrix_report.json")
    for metric in [
        "rule_argmax_mapping_by_seed",
        "rule_identity_alignment_score_mean",
        "rule_identity_alignment_score_min",
        "rule_diagonal_mass_mean",
        "rule_off_diagonal_mass_mean",
        "rule_entropy_mean",
        "effective_rule_matrix_by_seed",
    ]:
        assert metric in matrix

    comparison = load_json(out / "arm_comparison_report.json")
    for metric in [
        "learned_vs_query_only_test_delta",
        "learned_vs_shuffled_support_test_delta",
        "learned_vs_no_support_test_delta",
    ]:
        assert metric in comparison["deltas"]
    assert "wrong_support_behavior" in comparison
    assert "same_query_different_support_accuracy" in comparison

    per_seed = load_json(out / "per_seed_report.json")
    attempted = per_seed["attempted_jobs"]
    completed = per_seed["completed_jobs"]
    failed = per_seed["failed_jobs"]
    assert len(attempted) == len(completed) + len(failed)
    attempted_pairs = {(job["seed"], job["arm"]) for job in attempted}
    seen_pairs = {(job["seed"], job["arm"]) for job in completed} | {(job["seed"], job["arm"]) for job in failed}
    assert attempted_pairs == seen_pairs
    for job in attempted:
        job_dir = out / f"arm_{job['arm']}" / f"seed_{job['seed']}"
        assert (job_dir / "progress.jsonl").exists()
        assert (job_dir / "train_metrics.jsonl").exists()
        assert (job_dir / "best_individual.json").exists()
        if (job_dir / "metrics.json").exists():
            assert (job_dir / "row_outputs_test.jsonl").exists()
            assert (job_dir / "row_outputs_ood.jsonl").exists()
        else:
            assert (job_dir / "error.json").exists()

    decision = load_json(out / "decision.json")
    assert decision["decision"] in ALLOWED_DECISIONS
    if decision["decision"] == "rule_family_inference_router_scale_confirmed":
        learned_arm = aggregate["arms"]["MUTABLE_LEARNED_RULE_FAMILY_INFERENCE"]
        assert learned_arm["test_accuracy"] >= 0.98
        assert learned_arm["ood_accuracy"] >= 0.98
        assert learned_arm["rule_family_test_accuracy"] >= 0.98
        assert learned_arm["rule_family_ood_accuracy"] >= 0.98
        assert learned_arm["min_seed_test_accuracy"] >= 0.95
        assert learned_arm["min_seed_ood_accuracy"] >= 0.95
        assert learned["min_support_count_accuracy"] >= 0.95
        assert learned["min_margin_strata_accuracy"] >= 0.95
        assert comparison["deltas"]["learned_vs_query_only_test_delta"] >= 0.60
        assert comparison["deltas"]["learned_vs_shuffled_support_test_delta"] >= 0.80
        assert comparison["deltas"]["learned_vs_no_support_test_delta"] >= 0.60
        assert comparison["same_query_different_support_accuracy"] >= 0.98
        assert comparison["wrong_support_behavior"]["wrong_support_follow_rate_test"] >= 0.98
        assert comparison["wrong_support_behavior"]["wrong_support_selected_pocket_test_accuracy"] <= 0.05

    text = (
        Path("docs/research/D41_RULE_FAMILY_INFERENCE_ROUTER_SCALE_CONFIRM_CONTRACT.md").read_text()
        + "\n"
        + Path("docs/research/D41_RULE_FAMILY_INFERENCE_ROUTER_SCALE_CONFIRM_RESULT.md").read_text()
        + "\n"
        + (out / "report.md").read_text()
    ).lower()
    for phrase in [
        "does not prove raw visual raven reasoning",
        "full hidden-rule raven solving",
        "natural-language reasoning",
        "architecture superiority",
        "consciousness",
        "general intelligence",
    ]:
        assert phrase in text

    print(json.dumps({"status": "ok", "decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
