#!/usr/bin/env python3
"""Checker for D53 mutable ECF integration with VRAXION mutation architecture."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d52_upstream_manifest.json",
    "canonical_vraxion_smoke_report.json",
    "trained_policy_manifest.json",
    "representation_report.json",
    "mutation_acceptance_report.json",
    "fitness_landscape_report.json",
    "policy_action_distribution_report.json",
    "support_cost_frontier_report.json",
    "false_confidence_report.json",
    "regime_breakdown_report.json",
    "controller_comparison_report.json",
    "vraxion_integration_boundary_report.json",
    "best_policy_report.json",
    "min_seed_gate_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]

ARMS = [
    "D52_BEST_RULE_TABLE_REPLAY",
    "HANDCODED_D50_FULL_REFERENCE",
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "COST_ONLY_MUTATION_CONTROL",
    "VRAXION_MUTABLE_RULE_TABLE",
    "VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER",
    "VRAXION_MUTABLE_POCKET_STATE_CONTROLLER",
    "VRAXION_MUTABLE_HYBRID_CONTROLLER",
]

VRAXION_ARMS = [
    "VRAXION_MUTABLE_RULE_TABLE",
    "VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER",
    "VRAXION_MUTABLE_POCKET_STATE_CONTROLLER",
    "VRAXION_MUTABLE_HYBRID_CONTROLLER",
]

REGIMES = [
    "CLEAN_INDEPENDENT_SUPPORT",
    "CORRELATED_ECHO_SUPPORT",
    "ADVERSARIAL_DISTRACTOR_SUPPORT",
    "MIXED_CLEAN_AND_CORRELATED",
    "MIXED_CLEAN_AND_ADVERSARIAL",
    "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
    "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
    "EXTERNAL_TEST_REQUIRED_SUPPORT",
]

ROW_FIELDS = [
    "row_id",
    "seed",
    "split",
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
]

ALLOWED_DECISIONS = {
    "vraxion_mutable_ecf_controller_integration_positive",
    "vraxion_mutable_ecf_controller_positive_high_cost",
    "vraxion_integration_partial_rule_table_only",
    "vraxion_mutation_controller_not_confirmed",
    "vraxion_mutation_controller_failed_jobs_present",
}


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_jsonl(path):
    rows = 0
    arms = set()
    regimes = set()
    actions = set()
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing {missing}")
            require(0.0 <= row["confidence"] <= 1.0, f"{path} row {idx} invalid confidence")
            expected_total = row["original_support_used"] + row["counter_support_used"] + row["external_test_used"]
            require(row["total_support_used"] == expected_total, f"{path} row {idx} support accounting mismatch")
            arms.add(row["arm"])
            regimes.add(row["support_regime"])
            actions.add(row["selected_action"])
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    require(actions, f"{path} no controller actions recorded")


def source_guardrails(repo_root):
    runner = (repo_root / "scripts/probes/run_d53_mutable_ecf_integration_with_vraxion_mutation_architecture.py").read_text(encoding="utf-8")
    compact = runner.replace(" ", "").lower()
    forbidden = [
        ("hash(", "Python hash forbidden"),
        ("random.random()<", "fake random threshold sampling forbidden"),
        ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
        ("expected_selected", "expected_selected feature forbidden"),
        ("openai", "external API reference forbidden"),
        ("modal", "Modal reference forbidden"),
    ]
    for needle, message in forbidden:
        require(needle not in compact, message)
    feature_block = runner[runner.find("FEATURE_NAMES"): runner.find("BOUNDARY")]
    for leaked in ["truth_joint", "truth_pair", "true_operator", "false_joint"]:
        require(leaked not in feature_block, f"truth leakage in controller feature list: {leaked}")


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
    upstream = load(out / "d52_upstream_manifest.json")
    canonical = load(out / "canonical_vraxion_smoke_report.json")
    representation = load(out / "representation_report.json")
    boundary_report = load(out / "vraxion_integration_boundary_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    report = (out / "report.md").read_text(encoding="utf-8")
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    best = aggregate["best_vraxion_arm"]

    require(dataset["truth_hidden_from_controller_inputs"], "truth hidden flag missing")
    require(dataset["controller_only_not_formula_solver"], "controller-only flag missing")
    require(upstream["expected_decision"] == "mutable_ecf_controller_scale_confirmed", "bad upstream expected decision")
    require(upstream["expected_next"] == "D53_MUTABLE_ECF_INTEGRATION_WITH_VRAXION_MUTATION_ARCHITECTURE", "bad upstream expected next")
    require(canonical["source_smoke_passed"] is True, "canonical source smoke failed")
    require(canonical["action_output_encoding_smoke_passed"] is True, "action output smoke failed")
    require(canonical["sparse_firing_used_in_d53"] is False, "D53 should not claim sparse firing use")
    require(representation["full_sparse_firing_brain_used"] is False, "representation overclaims sparse brain")
    require(boundary_report["controller_integration_only"] is True, "boundary report missing controller-only flag")
    require(boundary_report["no_formula_solver_learning"] is True, "formula solver learning overclaim")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require(best in VRAXION_ARMS, "best arm is not VRAXION mutable")
    require("D53 only tests VRAXION-style mutation integration" in decision["boundary"], "boundary missing in decision")
    require("D53 only tests VRAXION-style mutation integration" in report, "boundary missing in report")
    require(set(ARMS).issubset(core.keys()), "missing core arm metrics")
    require(summary["best_vraxion_arm"] == best, "summary best mismatch")

    mutation = load(out / "mutation_acceptance_report.json")
    for arm in VRAXION_ARMS:
        require(arm in mutation, f"missing mutation report for {arm}")
        require("mutation_counts" in mutation[arm], f"missing mutation counts for {arm}")
        require(mutation[arm]["mutation_counts"], f"empty mutation counts for {arm}")

    best_row = core[best]
    corr = regimes[best]["CORRELATED_ECHO_SUPPORT"]["accuracy"]
    adv = regimes[best]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"]
    external = regimes[best]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"]
    indist = regimes[best]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    min_seed = load(out / "min_seed_gate_report.json")[best]
    require(core["RANDOM_POLICY_CONTROL"]["accuracy"] < best_row["accuracy"], "random control not worse")
    require(core["GREEDY_DECIDE_CONTROL"]["accuracy"] < best_row["accuracy"], "greedy decide control not worse")
    require(core["COST_ONLY_MUTATION_CONTROL"]["accuracy"] < best_row["accuracy"], "cost-only control not worse")
    require(core["ALWAYS_COUNTER_CONTROL"]["average_total_support_used"] > best_row["average_total_support_used"], "always-counter not higher cost")
    require(indist["false_confidence_rate"] <= 0.01, "indistinguishable false confidence above gate")

    if decision["decision"] == "vraxion_mutable_ecf_controller_integration_positive":
        require(best_row["exact_joint_accuracy"] >= 0.995, "best exact below gate")
        require(corr >= 0.995, "best correlated below gate")
        require(adv >= 0.995, "best adversarial below gate")
        require(external >= 0.99, "best external-test below gate")
        require(indist["abstain_rate"] >= 0.99, "best indistinguishable abstain below gate")
        require(min_seed["min_seed_exact_joint"] >= 0.99, "min seed exact below gate")
        require(min_seed["min_seed_correlated_echo"] >= 0.99, "min seed correlated below gate")
        require(min_seed["min_seed_adversarial_distractor"] >= 0.99, "min seed adversarial below gate")
        require(best_row["average_total_support_used"] <= core["HANDCODED_D50_FULL_REFERENCE"]["average_total_support_used"], "best support above handcoded full")

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "verdict": decision.get("verdict"),
                "next": decision.get("next"),
                "best_vraxion_arm": best,
                "scale_mode": aggregate["scale_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
