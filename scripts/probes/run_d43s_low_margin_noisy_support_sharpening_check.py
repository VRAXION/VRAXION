#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "d43_upstream_manifest.json",
    "d43_failure_tail_manifest.json",
    "dataset_manifest.json",
    "dataset_invariant_report.json",
    "ood_rule_invariance_audit.json",
    "baseline_replay_report.json",
    "margin_preserving_objective_report.json",
    "temperature_sharpened_evidence_report.json",
    "family_balanced_edge_oversampling_report.json",
    "combined_sharpened_model_report.json",
    "hard_vote_oracle_upper_bound_report.json",
    "shuffled_center_control_report.json",
    "shuffled_formula_candidate_control_report.json",
    "no_center_control_report.json",
    "same_query_different_raw_support_counterfactual_report.json",
    "wrong_support_control_report.json",
    "low_margin_tail_error_report.json",
    "family_confusion_tail_report.json",
    "support_count_tail_report.json",
    "evidence_gap_preservation_report.json",
    "equality_kernel_report.json",
    "channel_gate_report.json",
    "mutation_acceptance_report.json",
    "convergence_report.json",
    "seed_variance_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
ARMS = [
    "D43_BASELINE_REPLAY",
    "MARGIN_PRESERVING_OBJECTIVE",
    "TEMPERATURE_SHARPENED_EVIDENCE",
    "FAMILY_BALANCED_EDGE_OVERSAMPLING",
    "COMBINED_SHARPENED_MODEL",
    "HARD_VOTE_ORACLE_UPPER_BOUND",
    "SHUFFLED_CENTER_CONTROL",
    "SHUFFLED_FORMULA_CANDIDATE_CONTROL",
    "NO_CENTER_CONTROL",
    "SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL",
    "WRONG_SUPPORT_CONTROL",
]
STRATA = [
    "noisy_majority_margin_low",
    "noisy_majority_margin_high",
    "clean_unanimous",
    "pair_low_margin",
    "diag_low_margin",
    "row_low_margin",
    "support_count_3_low_margin",
    "support_count_5_low_margin",
]
ALLOWED_DECISIONS = {
    "d43s_dataset_or_ood_failure",
    "d43s_oracle_tail_failure",
    "low_margin_noisy_support_tail_hardened",
    "d43_tail_not_reproduced_baseline_already_solved",
    "low_margin_noisy_support_tail_not_hardened",
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
    if "target/pilot_wave/d43s_low_margin_noisy_support_sharpening/" not in out.as_posix():
        raise SystemExit("--out must stay under target/pilot_wave/d43s_low_margin_noisy_support_sharpening/")
    missing = [name for name in REQUIRED if not (out / name).exists()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    upstream = load_json(out / "d43_upstream_manifest.json")
    assert upstream["d43_decision"] == "raw_support_evidence_extraction_scale_confirmed"
    assert upstream["d43_verdict"] == "D43_RAW_SUPPORT_EVIDENCE_EXTRACTION_SCALE_CONFIRMED"
    assert upstream["d43_next"] == "D44_FORMULA_PRIMITIVE_DISCOVERY_PLAN"

    manifest = load_json(out / "dataset_manifest.json")
    assert manifest["support_counts"] == [3, 5]
    assert manifest["noisy_majority_margin_low_rate"] >= 0.70
    assert manifest["tier_a_fixed_formula_candidate_primitives"] is True
    assert manifest["formula_primitive_discovery_unavailable"] is True
    assert manifest["learned_arms_receive_precomputed_support_evidence"] is False
    assert manifest["learned_arms_receive_boolean_equality"] is False

    inv = load_json(out / "dataset_invariant_report.json")
    assert inv["duplicate_target_pocket_rate"] == 0.0
    assert inv["missing_target_pocket_rate"] == 0.0
    assert inv["expected_selected_points_to_target_rate"] == 1.0
    assert inv["ambiguous_support_rate"] == 0.0
    assert inv["multi_family_support_tie_rate"] == 0.0
    assert inv["intended_family_unique_evidence_rate"] == 1.0
    assert inv["counterfactual_target_collision_rate"] == 0.0
    assert inv["wrong_support_query_mismatch_rate"] == 1.0

    audit = load_json(out / "ood_rule_invariance_audit.json")
    assert audit["support_evidence_oracle_test_accuracy"] == 1.0
    assert audit["support_evidence_oracle_ood_accuracy"] == 1.0
    assert audit["support_selected_pocket_oracle_test_accuracy"] == 1.0
    assert audit["support_selected_pocket_oracle_ood_accuracy"] == 1.0
    assert audit["known_rule_oracle_test_accuracy"] == 1.0
    assert audit["known_rule_oracle_ood_accuracy"] == 1.0
    assert audit["ood_label_rule_changed"] is False

    src = Path("scripts/probes/run_d43s_low_margin_noisy_support_sharpening.py").read_text()
    compact = src.replace(" ", "")
    for pattern in ["hit=random.random()<p", "random.random()<p", "fixed_base_accuracy"]:
        assert pattern not in compact
    assert "hash(" not in src
    learned_path = function_body(src, "raw_evidence_scores") + "\n" + function_body(src, "predict_learned_raw")
    assert "support_evidence_vector" not in learned_path
    assert "wrong_support_evidence_vector" not in learned_path
    assert "expected_selected" not in function_body(src, "raw_evidence_scores")
    assert "center == family_target" not in learned_path

    aggregate = load_json(out / "aggregate_metrics.json")
    for arm in ARMS:
        assert arm in aggregate["arms"]
        report = load_json(out / {
            "D43_BASELINE_REPLAY": "baseline_replay_report.json",
            "MARGIN_PRESERVING_OBJECTIVE": "margin_preserving_objective_report.json",
            "TEMPERATURE_SHARPENED_EVIDENCE": "temperature_sharpened_evidence_report.json",
            "FAMILY_BALANCED_EDGE_OVERSAMPLING": "family_balanced_edge_oversampling_report.json",
            "COMBINED_SHARPENED_MODEL": "combined_sharpened_model_report.json",
            "HARD_VOTE_ORACLE_UPPER_BOUND": "hard_vote_oracle_upper_bound_report.json",
            "SHUFFLED_CENTER_CONTROL": "shuffled_center_control_report.json",
            "SHUFFLED_FORMULA_CANDIDATE_CONTROL": "shuffled_formula_candidate_control_report.json",
            "NO_CENTER_CONTROL": "no_center_control_report.json",
            "SAME_QUERY_DIFFERENT_RAW_SUPPORT_COUNTERFACTUAL": "same_query_different_raw_support_counterfactual_report.json",
            "WRONG_SUPPORT_CONTROL": "wrong_support_control_report.json",
        }[arm])
        for metric in ["train_accuracy", "test_accuracy", "ood_accuracy", "min_seed_test_accuracy", "min_seed_ood_accuracy", "error_count", "family_confusion_matrix", "median_evidence_gap", "low_margin_error_rate"]:
            assert metric in report
        for stratum in STRATA:
            assert f"{stratum}_accuracy" in report

    hard_vote = load_json(out / "hard_vote_oracle_upper_bound_report.json")
    assert hard_vote["test_accuracy"] == 1.0
    assert hard_vote["ood_accuracy"] == 1.0

    comparison = aggregate["comparison"]
    assert "baseline_noisy_majority_margin_low_accuracy" in comparison
    assert "combined_noisy_majority_margin_low_accuracy" in comparison
    assert "noisy_low_delta" in comparison
    assert "easy_case_regression" in comparison
    assert "controls" in comparison

    tail = load_json(out / "low_margin_tail_error_report.json")
    assert "baseline_replay" in tail
    assert "combined_sharpened" in tail
    assert "comparison" in tail

    equality = load_json(out / "equality_kernel_report.json")
    assert "equality_kernel_argmax_mapping" in equality
    assert "equality_kernel_diagonal_mass" in equality
    channel = load_json(out / "channel_gate_report.json")
    assert "channel_gate_argmax_mapping_by_seed" in channel
    assert "channel_gate_identity_alignment_score" in channel

    per_seed = load_json(out / "per_seed_report.json")
    attempted = per_seed["attempted_jobs"]
    completed = per_seed["completed_jobs"]
    failed = per_seed["failed_jobs"]
    assert len(attempted) == len(completed) + len(failed)
    assert not failed

    decision = load_json(out / "decision.json")
    assert decision["decision"] in ALLOWED_DECISIONS
    if decision["decision"] == "low_margin_noisy_support_tail_hardened":
        combined = load_json(out / "combined_sharpened_model_report.json")
        assert comparison["noisy_low_delta"] >= 0.001
        assert combined["noisy_majority_margin_low_accuracy"] >= 0.999
        assert combined["test_accuracy"] >= 0.999
        assert combined["ood_accuracy"] >= 0.999
        assert comparison["easy_case_regression"] <= 0.0005
        assert comparison["controls"]["shuffled_center_test_accuracy"] <= 0.25
        assert comparison["controls"]["shuffled_formula_candidate_test_accuracy"] <= 0.25
        assert comparison["controls"]["no_center_test_accuracy"] <= 0.35
        assert comparison["controls"]["same_query_different_raw_support_accuracy"] >= 0.98
        assert comparison["controls"]["wrong_support_follow_rate_test"] >= 0.98
        assert comparison["controls"]["wrong_support_selected_pocket_test_accuracy"] <= 0.05

    text = (
        Path("docs/research/D43S_LOW_MARGIN_NOISY_SUPPORT_SHARPENING_CONTRACT.md").read_text()
        + "\n"
        + Path("docs/research/D43S_LOW_MARGIN_NOISY_SUPPORT_SHARPENING_RESULT.md").read_text()
        + "\n"
        + (out / "report.md").read_text()
    ).lower()
    for phrase in [
        "does not prove formula primitive discovery",
        "raw visual raven reasoning",
        "raven solved",
        "architecture superiority",
        "consciousness",
        "general intelligence",
    ]:
        assert phrase in text
    print(json.dumps({"status": "ok", "decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
