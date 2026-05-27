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
    "machine_utilization_report.json",
    "monolithic_formula_baseline_report.json",
    "oracle_gated_rule_formula_report.json",
    "mutable_learned_router_gate_report.json",
    "shuffled_gate_control_report.json",
    "no_family_input_control_report.json",
    "explicit_target_state_upper_bound_report.json",
    "gate_matrix_report.json",
    "mutation_acceptance_report.json",
    "per_seed_report.json",
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
    "d38_dataset_invariant_failure",
    "d38_ood_rule_invariance_failure",
    "learned_conditioning_router_field_confirmed",
    "learned_conditioning_router_field_not_confirmed",
}


def load_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
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

    src = Path("scripts/probes/run_d38_learned_conditioning_router_field_proof.py").read_text()
    compact = src.replace(" ", "")
    forbidden = ["hit=random.random()<p", "random.random()<p", "base={", "fixed_base_accuracy"]
    for pattern in forbidden:
        assert pattern not in compact
    assert "hash(" not in src

    aggregate = load_json(out / "aggregate_metrics.json")
    for arm in ARMS:
        assert arm in aggregate["arms"]
        arm_report = aggregate["arms"][arm]
        for metric in ["train_accuracy", "test_accuracy", "ood_accuracy", "min_seed_test_accuracy", "min_seed_ood_accuracy"]:
            assert metric in arm_report

    gate_report = load_json(out / "gate_matrix_report.json")
    for metric in [
        "gate_argmax_mapping_by_seed",
        "gate_identity_alignment_score_mean",
        "gate_identity_alignment_score_min",
        "diagonal_gate_mass_mean",
        "off_diagonal_gate_mass_mean",
        "gate_entropy_mean",
    ]:
        assert metric in gate_report

    decision = load_json(out / "decision.json")
    assert decision["decision"] in ALLOWED_DECISIONS

    text = (
        Path("docs/research/D38_LEARNED_CONDITIONING_ROUTER_FIELD_PROOF_CONTRACT.md").read_text()
        + "\n"
        + Path("docs/research/D38_LEARNED_CONDITIONING_ROUTER_FIELD_PROOF_RESULT.md").read_text()
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
