#!/usr/bin/env python3
"""Checker for D51 mutable ECF controller prototype."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d50_upstream_manifest.json",
    "trained_policy_manifest.json",
    "controller_input_feature_report.json",
    "mutation_acceptance_report.json",
    "policy_action_distribution_report.json",
    "support_cost_frontier_report.json",
    "false_confidence_report.json",
    "regime_breakdown_report.json",
    "controller_comparison_report.json",
    "best_policy_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]

ARMS = [
    "HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE",
    "HANDCODED_CAP_7_REFERENCE",
    "HANDCODED_CAP_9_REFERENCE",
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_SUPPORT_CONTROL",
    "MUTABLE_LINEAR_CONTROLLER",
    "MUTABLE_RULE_TABLE_CONTROLLER",
    "MUTABLE_SMALL_TREE_CONTROLLER",
    "MUTABLE_HYBRID_CONTROLLER",
]

MUTABLE_ARMS = [
    "MUTABLE_LINEAR_CONTROLLER",
    "MUTABLE_RULE_TABLE_CONTROLLER",
    "MUTABLE_SMALL_TREE_CONTROLLER",
    "MUTABLE_HYBRID_CONTROLLER",
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
    "total_support_used",
    "counter_support_used",
    "external_test_used",
    "abstained",
    "false_confidence",
    "confidence",
    "error_type",
]

ALLOWED_DECISIONS = {
    "mutable_ecf_controller_prototype_positive",
    "mutable_controller_positive_high_cost",
    "mutable_controller_not_confirmed",
    "mutable_controller_false_confidence_failure",
    "mutable_controller_failed_jobs_present",
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
            arms.add(row["arm"])
            regimes.add(row["support_regime"])
            actions.add(row["selected_action"])
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    require(actions, f"{path} no controller actions recorded")


def source_guardrails(repo_root):
    runner = (repo_root / "scripts/probes/run_d51_mutable_ecf_controller_prototype.py").read_text(encoding="utf-8")
    compact = runner.replace(" ", "")
    forbidden = [
        ("hash(", "Python hash forbidden"),
        ("random.random()<", "fake random threshold sampling forbidden"),
        ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
        ("expected_selected", "expected_selected feature forbidden"),
        ("openai", "external API reference forbidden"),
        ("modal", "Modal reference forbidden"),
    ]
    for needle, message in forbidden:
        require(needle not in compact.lower(), message)
    feature_block = runner[runner.find("FEATURE_NAMES"): runner.find("GLOBAL_BUNDLE")]
    for leaked in ["truth_joint", "truth_pair", "true_operator", "false_joint"]:
        require(leaked not in feature_block, f"truth leakage in controller feature list: {leaked}")


def cost_adjusted(row):
    return row["exact_joint_accuracy"] - 0.0010 * row["average_total_support_used"] - 0.0015 * row["average_counter_support_used"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    require(not missing, f"missing artifacts: {missing}")
    source_guardrails(Path(__file__).resolve().parents[2])
    inspect_jsonl(out / "row_outputs_test.jsonl")
    inspect_jsonl(out / "row_outputs_ood.jsonl")

    dataset = load(out / "dataset_manifest.json")
    upstream = load(out / "d50_upstream_manifest.json")
    features = load(out / "controller_input_feature_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    report = (out / "report.md").read_text(encoding="utf-8")
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    best = aggregate["best_mutable_arm"]

    require(dataset["truth_hidden_from_controller_inputs"], "truth hidden flag missing")
    require(dataset["controller_only_not_formula_solver"], "controller-only flag missing")
    require(features["truth_labels_as_controller_inputs"] is False, "feature report truth leakage")
    require(upstream["expected_decision"] == "joint_formula_discovery_scale_confirmed", "bad upstream expected decision")
    require(upstream["expected_next"] == "D51_MUTABLE_ECF_CONTROLLER_PROTOTYPE", "bad upstream expected next")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require(best in MUTABLE_ARMS, "best mutable arm is not mutable")
    require("D51 only tests mutable control policy" in decision["boundary"], "boundary missing in decision")
    require("D51 only tests mutable control policy" in report, "boundary missing in report")
    require(set(ARMS).issubset(core.keys()), "missing core arm metrics")
    require(summary["best_mutable_arm"] == best, "summary best mismatch")

    best_row = core[best]
    cap9 = core["HANDCODED_CAP_9_REFERENCE"]
    full = core["HANDCODED_D50_FULL_REPAIRED_ECF_REFERENCE"]
    corr = regimes[best]["CORRELATED_ECHO_SUPPORT"]["accuracy"]
    adv = regimes[best]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"]
    indist = regimes[best]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    require(core["RANDOM_POLICY_CONTROL"]["accuracy"] < best_row["accuracy"], "random control not worse")
    require(core["GREEDY_DECIDE_CONTROL"]["accuracy"] < best_row["accuracy"], "greedy decide control not worse")
    require(indist["false_confidence_rate"] <= 0.01, "indistinguishable false confidence above gate")

    if decision["decision"] == "mutable_ecf_controller_prototype_positive":
        require(best_row["exact_joint_accuracy"] >= 0.995, "best exact below gate")
        require(corr >= 0.995, "best correlated below gate")
        require(adv >= 0.995, "best adversarial below gate")
        require(best_row["average_total_support_used"] <= full["average_total_support_used"], "best support above full reference")
        require(cost_adjusted(best_row) >= cost_adjusted(cap9), "best does not beat/equal cap9 cost frontier")

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "verdict": decision.get("verdict"),
                "next": decision.get("next"),
                "best_mutable_arm": best,
                "scale_mode": aggregate["scale_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
