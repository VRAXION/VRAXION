#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "d41_upstream_manifest.json",
    "dataset_manifest.json",
    "dataset_invariant_report.json",
    "support_evidence_oracle_audit.json",
    "ood_rule_invariance_audit.json",
    "random_baseline_report.json",
    "query_only_baseline_report.json",
    "precomputed_support_evidence_upper_bound_report.json",
    "oracle_raw_support_evidence_extractor_report.json",
    "mutable_learned_raw_support_evidence_extractor_report.json",
    "raw_support_plus_learned_router_composition_report.json",
    "shuffled_center_control_report.json",
    "shuffled_formula_candidate_control_report.json",
    "no_center_control_report.json",
    "no_formula_candidate_control_report.json",
    "wrong_support_control_report.json",
    "same_query_different_raw_support_counterfactual_report.json",
    "support_count_generalization_report.json",
    "support_margin_strata_report.json",
    "raw_evidence_extractor_matrix_report.json",
    "equality_kernel_report.json",
    "channel_gate_report.json",
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
    "QUERY_ONLY_BASELINE",
    "PRECOMPUTED_SUPPORT_EVIDENCE_UPPER_BOUND",
    "ORACLE_RAW_SUPPORT_EVIDENCE_EXTRACTOR",
    "MUTABLE_LEARNED_RAW_SUPPORT_EVIDENCE_EXTRACTOR",
    "SHUFFLED_CENTER_CONTROL",
    "SHUFFLED_FORMULA_CANDIDATE_CONTROL",
    "NO_CENTER_CONTROL",
    "NO_FORMULA_CANDIDATE_CONTROL",
    "WRONG_SUPPORT_CONTROL",
    "SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL",
    "RAW_SUPPORT_PLUS_LEARNED_ROUTER_COMPOSITION",
]
ALLOWED_DECISIONS = {
    "d42_dataset_invariant_failure",
    "d42_support_evidence_oracle_failure",
    "d42_ood_rule_invariance_failure",
    "oracle_raw_support_evidence_confirmed_learned_extractor_failed",
    "raw_support_evidence_extraction_prototype_positive",
    "raw_support_extraction_mean_positive_support_strata_gap",
    "raw_support_evidence_extraction_partial_signal",
    "raw_support_evidence_extraction_not_confirmed",
}


def load_json(path):
    return json.loads(path.read_text())


def function_body(src, name):
    marker = f"def {name}"
    start = src.index(marker)
    tail = src[start + len(marker):]
    next_def = tail.find("\ndef ")
    if next_def == -1:
        return src[start:]
    return src[start : start + len(marker) + next_def]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if "target/pilot_wave/d42_raw_support_evidence_extraction_prototype/" not in out.as_posix():
        raise SystemExit("--out must stay under target/pilot_wave/d42_raw_support_evidence_extraction_prototype/")
    missing = [name for name in REQUIRED if not (out / name).exists()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    upstream = load_json(out / "d41_upstream_manifest.json")
    assert upstream["d41_decision"] == "rule_family_inference_router_scale_confirmed"
    assert upstream["d41_verdict"] == "D41_RULE_FAMILY_INFERENCE_ROUTER_SCALE_CONFIRMED"
    assert upstream["d41_next"] == "D42_RAW_SUPPORT_EVIDENCE_EXTRACTION_PROTOTYPE"
    assert upstream["d41_support_count_strata_pass"] is True
    assert upstream["d41_margin_strata_pass"] is True
    assert upstream["d41_wrong_support_control_pass"] is True
    assert upstream["d41_same_query_counterfactual_control_pass"] is True

    inv = load_json(out / "dataset_invariant_report.json")
    assert inv["duplicate_target_pocket_rate"] == 0.0
    assert inv["missing_target_pocket_rate"] == 0.0
    assert inv["expected_selected_points_to_target_rate"] == 1.0
    assert inv["ambiguous_support_rate"] == 0.0
    assert inv["multi_family_support_tie_rate"] == 0.0
    assert inv["intended_family_unique_evidence_rate"] == 1.0
    assert inv["counterfactual_target_collision_rate"] == 0.0
    assert inv["wrong_support_query_mismatch_rate"] == 1.0

    support = load_json(out / "support_evidence_oracle_audit.json")
    assert support["support_evidence_oracle_test_accuracy"] == 1.0
    assert support["support_evidence_oracle_ood_accuracy"] == 1.0
    assert support["support_selected_pocket_oracle_test_accuracy"] == 1.0
    assert support["support_selected_pocket_oracle_ood_accuracy"] == 1.0

    audit = load_json(out / "ood_rule_invariance_audit.json")
    assert audit["support_evidence_oracle_test_accuracy"] == 1.0
    assert audit["support_evidence_oracle_ood_accuracy"] == 1.0
    assert audit["support_selected_pocket_oracle_test_accuracy"] == 1.0
    assert audit["support_selected_pocket_oracle_ood_accuracy"] == 1.0
    assert audit["known_rule_oracle_test_accuracy"] == 1.0
    assert audit["known_rule_oracle_ood_accuracy"] == 1.0
    assert audit["ood_label_rule_changed"] is False

    manifest = load_json(out / "dataset_manifest.json")
    assert manifest["tier_a_fixed_formula_candidate_primitives"] is True
    assert manifest["formula_primitive_discovery_unavailable"] is True

    src_path = Path("scripts/probes/run_d42_raw_support_evidence_extraction_prototype.py")
    src = src_path.read_text()
    compact = src.replace(" ", "")
    for pattern in ["hit=random.random()<p", "random.random()<p", "base={", "fixed_base_accuracy"]:
        assert pattern not in compact
    assert "hash(" not in src
    learned_path = function_body(src, "raw_evidence_scores") + "\n" + function_body(src, "predict_learned_raw")
    assert "support_evidence_vector" not in learned_path
    assert "wrong_support_evidence_vector" not in learned_path
    assert "expected_selected" not in function_body(src, "raw_evidence_scores")

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

    learned = load_json(out / "mutable_learned_raw_support_evidence_extractor_report.json")
    for metric in [
        "learned_raw_extractor_rule_family_train_accuracy",
        "learned_raw_extractor_rule_family_test_accuracy",
        "learned_raw_extractor_rule_family_ood_accuracy",
        "learned_raw_extractor_selected_pocket_train_accuracy",
        "learned_raw_extractor_selected_pocket_test_accuracy",
        "learned_raw_extractor_selected_pocket_ood_accuracy",
        "min_seed_learned_raw_extractor_test_accuracy",
        "min_seed_learned_raw_extractor_ood_accuracy",
        "min_support_count_accuracy",
        "min_margin_strata_accuracy",
        "equality_kernel_diagonal_mass",
        "equality_kernel_off_diagonal_mass",
        "equality_kernel_argmax_mapping",
        "equality_kernel_entropy",
        "channel_gate_identity_alignment_score",
        "channel_gate_diagonal_mass",
        "channel_gate_off_diagonal_mass",
        "channel_gate_argmax_mapping_by_seed",
        "accepted_mutations_by_type",
        "rejected_mutations_by_type",
        "mutation_acceptance_rate",
        "convergence_generation_median",
        "seed_variance",
    ]:
        assert metric in learned

    equality = load_json(out / "equality_kernel_report.json")
    for metric in [
        "equality_kernel_diagonal_mass",
        "equality_kernel_off_diagonal_mass",
        "equality_kernel_argmax_mapping",
        "equality_kernel_entropy",
        "equality_kernel_matrix_by_seed",
    ]:
        assert metric in equality
    channel = load_json(out / "channel_gate_report.json")
    for metric in [
        "channel_gate_identity_alignment_score",
        "channel_gate_diagonal_mass",
        "channel_gate_off_diagonal_mass",
        "channel_gate_argmax_mapping_by_seed",
        "effective_channel_gate_by_seed",
    ]:
        assert metric in channel

    comparison = load_json(out / "arm_comparison_report.json")
    for metric in [
        "learned_vs_query_only_test_delta",
        "learned_vs_shuffled_center_test_delta",
        "learned_vs_no_center_test_delta",
    ]:
        assert metric in comparison["deltas"]
    assert "wrong_support_behavior" in comparison
    assert "same_query_different_raw_support_accuracy" in comparison

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
    if decision["decision"] == "raw_support_evidence_extraction_prototype_positive":
        learned_arm = aggregate["arms"]["MUTABLE_LEARNED_RAW_SUPPORT_EVIDENCE_EXTRACTOR"]
        assert learned_arm["test_accuracy"] >= 0.95
        assert learned_arm["ood_accuracy"] >= 0.95
        assert learned_arm["rule_family_test_accuracy"] >= 0.95
        assert learned_arm["rule_family_ood_accuracy"] >= 0.95
        assert learned_arm["min_seed_test_accuracy"] >= 0.90
        assert learned_arm["min_seed_ood_accuracy"] >= 0.90
        assert learned["min_support_count_accuracy"] >= 0.90
        assert learned["min_margin_strata_accuracy"] >= 0.90
        assert aggregate["arms"]["PRECOMPUTED_SUPPORT_EVIDENCE_UPPER_BOUND"]["test_accuracy"] >= 0.99
        assert aggregate["arms"]["ORACLE_RAW_SUPPORT_EVIDENCE_EXTRACTOR"]["test_accuracy"] >= 0.99
        assert comparison["deltas"]["learned_vs_query_only_test_delta"] >= 0.60
        assert comparison["deltas"]["learned_vs_shuffled_center_test_delta"] >= 0.70
        assert comparison["deltas"]["learned_vs_no_center_test_delta"] >= 0.60
        assert comparison["same_query_different_raw_support_accuracy"] >= 0.95
        assert comparison["wrong_support_behavior"]["wrong_support_follow_rate_test"] >= 0.95
        assert comparison["wrong_support_behavior"]["wrong_support_selected_pocket_test_accuracy"] <= 0.05

    text = (
        Path("docs/research/D42_RAW_SUPPORT_EVIDENCE_EXTRACTION_PROTOTYPE_CONTRACT.md").read_text()
        + "\n"
        + Path("docs/research/D42_RAW_SUPPORT_EVIDENCE_EXTRACTION_PROTOTYPE_RESULT.md").read_text()
        + "\n"
        + (out / "report.md").read_text()
    ).lower()
    for phrase in [
        "does not prove raw visual raven reasoning",
        "formula primitive discovery",
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
