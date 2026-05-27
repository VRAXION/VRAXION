#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "dataset_manifest.json",
    "dataset_invariant_report.json",
    "ood_rule_invariance_audit.json",
    "d38_upstream_manifest.json",
    "machine_utilization_report.json",
    "monolithic_formula_baseline_report.json",
    "oracle_gated_rule_formula_report.json",
    "mutable_learned_router_gate_report.json",
    "shuffled_gate_control_report.json",
    "no_family_input_control_report.json",
    "explicit_target_state_upper_bound_report.json",
    "gate_matrix_report.json",
    "gate_identity_alignment_report.json",
    "gate_stability_report.json",
    "seed_variance_report.json",
    "convergence_report.json",
    "mutation_acceptance_report.json",
    "per_seed_report.json",
    "per_family_report.json",
    "pocket_confusion_matrix.json",
    "score_margin_report.json",
    "arm_comparison_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]
ARMS = [
    "MONOLITHIC_FORMULA_BASELINE",
    "ORACLE_GATED_RULE_FORMULA_UPPER_BOUND",
    "MUTABLE_LEARNED_ROUTER_GATE",
    "SHUFFLED_GATE_CONTROL",
    "NO_FAMILY_INPUT_CONTROL",
    "EXPLICIT_TARGET_STATE_UPPER_BOUND",
]
ALLOWED_DECISIONS = {
    "d39_dataset_invariant_failure",
    "d39_ood_rule_invariance_failure",
    "learned_conditioning_router_field_scale_confirmed",
    "learned_router_mean_positive_seed_variance_edge",
    "learned_router_scale_partial_signal",
    "learned_router_scale_not_confirmed",
}


def load_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if "target/pilot_wave/d39_learned_router_layer_scale_confirm/" not in out.as_posix():
        raise SystemExit("--out must stay under target/pilot_wave/d39_learned_router_layer_scale_confirm/")
    missing = [name for name in REQUIRED if not (out / name).exists()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    inv = load_json(out / "dataset_invariant_report.json")
    assert inv["duplicate_target_pocket_rate"] == 0.0
    assert inv["missing_target_pocket_rate"] == 0.0
    assert inv["expected_selected_points_to_target_rate"] == 1.0

    audit = load_json(out / "ood_rule_invariance_audit.json")
    assert audit["known_rule_oracle_test_accuracy"] == 1.0
    assert audit["known_rule_oracle_ood_accuracy"] == 1.0
    assert audit["ood_label_rule_changed"] is False

    src = Path("scripts/probes/run_d39_learned_router_layer_scale_confirm.py").read_text()
    compact = src.replace(" ", "")
    for pattern in ["hit=random.random()<p", "random.random()<p", "base={", "fixed_base_accuracy"]:
        assert pattern not in compact
    assert "hash(" not in src
    assert "expected_selected as feature" not in src

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
            "per_family_accuracy",
            "pocket_confusion_matrix",
            "median_score_margin",
            "low_margin_error_rate",
            "failed_seed_count",
        ]:
            assert metric in arm_report
        assert set(arm_report["per_family_accuracy"]) == {"row", "col", "pair", "mirror", "diag"}

    deltas = aggregate["deltas"]
    for metric in [
        "monolithic_vs_learned_test_delta",
        "learned_vs_shuffled_test_delta",
        "learned_vs_no_family_test_delta",
    ]:
        assert metric in deltas

    learned = load_json(out / "mutable_learned_router_gate_report.json")
    for metric in [
        "learned_gate_train_accuracy",
        "learned_gate_test_accuracy",
        "learned_gate_ood_accuracy",
        "min_seed_learned_gate_test_accuracy",
        "min_seed_learned_gate_ood_accuracy",
        "gate_identity_alignment_score_mean",
        "gate_identity_alignment_score_min",
        "diagonal_gate_mass_mean",
        "off_diagonal_gate_mass_mean",
        "gate_argmax_mapping_by_seed",
        "gate_entropy_mean",
        "accepted_mutations_by_type",
        "rejected_mutations_by_type",
        "mutation_acceptance_rate",
        "convergence_generation_median",
        "seed_variance",
    ]:
        assert metric in learned

    gate_report = load_json(out / "gate_matrix_report.json")
    for metric in [
        "gate_argmax_mapping_by_seed",
        "gate_identity_alignment_score_mean",
        "gate_identity_alignment_score_min",
        "diagonal_gate_mass_mean",
        "off_diagonal_gate_mass_mean",
        "gate_entropy_mean",
        "effective_gate_matrix_by_seed",
    ]:
        assert metric in gate_report

    per_seed = load_json(out / "per_seed_report.json")
    attempted = per_seed["attempted_jobs"]
    completed = per_seed["completed_jobs"]
    failed = per_seed["failed_jobs"]
    assert len(attempted) == len(completed) + len(failed)
    attempted_pairs = {(job["seed"], job["arm"]) for job in attempted}
    seen_pairs = {(job["seed"], job["arm"]) for job in completed} | {(job["seed"], job["arm"]) for job in failed}
    assert attempted_pairs == seen_pairs

    decision = load_json(out / "decision.json")
    assert decision["decision"] in ALLOWED_DECISIONS
    if decision["decision"] == "learned_conditioning_router_field_scale_confirmed":
        assert aggregate["arms"]["MUTABLE_LEARNED_ROUTER_GATE"]["test_accuracy"] >= 0.95
        assert aggregate["arms"]["MUTABLE_LEARNED_ROUTER_GATE"]["ood_accuracy"] >= 0.95
        assert aggregate["arms"]["MUTABLE_LEARNED_ROUTER_GATE"]["min_seed_test_accuracy"] >= 0.90
        assert aggregate["arms"]["MUTABLE_LEARNED_ROUTER_GATE"]["min_seed_ood_accuracy"] >= 0.90
        assert aggregate["arms"]["ORACLE_GATED_RULE_FORMULA_UPPER_BOUND"]["test_accuracy"] >= 0.99
        assert aggregate["arms"]["EXPLICIT_TARGET_STATE_UPPER_BOUND"]["test_accuracy"] >= 0.99
        assert deltas["monolithic_vs_learned_test_delta"] >= 0.45
        assert deltas["learned_vs_shuffled_test_delta"] >= 0.70
        assert deltas["learned_vs_no_family_test_delta"] >= 0.45
        assert aggregate["arms"]["SHUFFLED_GATE_CONTROL"]["test_accuracy"] <= 0.25
        assert aggregate["arms"]["NO_FAMILY_INPUT_CONTROL"]["test_accuracy"] <= 0.50

    text = (
        Path("docs/research/D39_LEARNED_ROUTER_LAYER_SCALE_CONFIRM_CONTRACT.md").read_text()
        + "\n"
        + Path("docs/research/D39_LEARNED_ROUTER_LAYER_SCALE_CONFIRM_RESULT.md").read_text()
        + "\n"
        + (out / "report.md").read_text()
    ).lower()
    for phrase in [
        "does not prove hidden-rule raven reasoning",
        "does not prove hidden-rule raven reasoning, natural-language reasoning",
        "architecture superiority",
        "general intelligence",
    ]:
        assert phrase in text

    print(json.dumps({"status": "ok", "decision": decision["decision"], "verdict": decision["verdict"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
