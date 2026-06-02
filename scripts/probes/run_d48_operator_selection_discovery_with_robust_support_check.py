#!/usr/bin/env python3
"""Checker for D48 operator-selection discovery with robust support."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "train_manifest.json",
    "policy_comparison_report.json",
    "regime_by_policy_report.json",
    "operator_diagnostic_report.json",
    "counterfactual_effect_report.json",
    "support_cost_frontier_report.json",
    "control_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]

ARMS = [
    "CURRENT_OP_ORACLE_REFERENCE_ONLY",
    "ALL_OPERATOR_ENUMERATION_SOFT_BASELINE",
    "OPERATOR_FAMILY_FACTORISED_FIELD",
    "OPERATOR_EQUIVALENCE_GROUPING",
    "COUNTERFACTUAL_TOP1_TOP2_REPAIR",
    "FULL_ROBUST_ECF_CONTROLLER",
    "FULL_ROBUST_ECF_CONTROLLER_CAP_5",
    "FULL_ROBUST_ECF_CONTROLLER_CAP_7",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_SIGNAL_CONTROL",
    "SHUFFLED_OPERATOR_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
]

REGIMES = [
    "CLEAN_INDEPENDENT_SUPPORT",
    "CORRELATED_ECHO_SUPPORT",
    "ADVERSARIAL_DISTRACTOR_SUPPORT",
    "MIXED_CLEAN_AND_CORRELATED",
    "MIXED_CLEAN_AND_ADVERSARIAL",
]

ROW_FIELDS = [
    "row_id",
    "seed",
    "split",
    "arm",
    "primitive_space",
    "support_regime",
    "pair_family",
    "pair",
    "true_operator",
    "pred_operator",
    "false_operator",
    "true_operator_family",
    "pred_operator_family",
    "true_equivalence",
    "pred_equivalence",
    "exact_operator_correct",
    "operator_family_correct",
    "operator_equivalence_correct",
    "correct",
    "reference_arm",
    "support_budget_cap",
    "original_support_used",
    "counter_support_used",
    "total_support_used",
    "support_cluster_count",
    "dominant_cluster_fraction",
    "collision_count",
    "correlated_echo_detected",
    "counter_support_requested",
    "counter_support_resolved",
    "top1_top2_margin",
    "entropy",
    "confidence",
    "baseline_exact_correct",
    "error_type",
]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_jsonl(path):
    rows = 0
    arms = set()
    regimes = set()
    budgets = set()
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing {missing}")
            require(0.0 <= row["confidence"] <= 1.0, f"{path} row {idx} invalid confidence")
            require(row["total_support_used"] >= row["original_support_used"], f"{path} row {idx} bad support count")
            arms.add(row["arm"])
            regimes.add(row["support_regime"])
            budgets.add(row["support_budget_cap"])
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    require(5 in budgets, f"{path} missing budget 5")


def source_guardrails(repo_root):
    runner = (repo_root / "scripts/probes/run_d48_operator_selection_discovery_with_robust_support.py").read_text(
        encoding="utf-8"
    )
    compact = runner.replace(" ", "")
    require("hash(" not in runner, "Python hash forbidden")
    require("random.random()<" not in compact, "fake random threshold sampling forbidden")
    require("fixed_synthetic_accuracy" not in runner, "fixed synthetic accuracy dict forbidden")
    require("expected_selected" not in runner, "expected_selected feature forbidden")
    require("openai" not in runner.lower(), "external API reference forbidden")
    require("modal" not in runner.lower(), "Modal reference forbidden")


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
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    frontier = load(out / "support_cost_frontier_report.json")
    report = (out / "report.md").read_text(encoding="utf-8")
    metrics = aggregate["test_metrics"]["by_arm"]
    regime = aggregate["test_metrics"]["by_arm_and_regime"]
    require(dataset["true_operator_hidden_from_fair_arms"], "truth-hidden flag missing")
    require(dataset["label_echo_reference_only_not_fair"], "label echo reference flag missing")
    require(dataset["candidate_family_equivalence_operator_metrics_separated"], "metric separation flag missing")
    require(set(ARMS).issubset(set(dataset["arms"])), "dataset manifest missing arms")
    require(set(map(str, [1, 2, 3, 4, 5])).issubset(set(frontier.keys())), "support frontier missing budgets")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require("D48 only tests controlled symbolic operator-selection discovery" in decision["boundary"], "boundary missing in decision")
    require("D48 only tests controlled symbolic operator-selection discovery" in report, "boundary missing in report")
    allowed = {
        "operator_selection_discovery_with_robust_support_positive",
        "robust_support_operator_transfer_failed",
        "operator_selection_positive_high_support_cost",
    }
    require(decision["decision"] in allowed, f"unexpected decision {decision['decision']}")
    if decision["decision"] == "operator_selection_discovery_with_robust_support_positive":
        full = metrics["FULL_ROBUST_ECF_CONTROLLER"]
        baseline = metrics["ALL_OPERATOR_ENUMERATION_SOFT_BASELINE"]
        require(regime["FULL_ROBUST_ECF_CONTROLLER"]["CLEAN_INDEPENDENT_SUPPORT"]["accuracy"] >= 0.995, "clean below gate")
        require(regime["FULL_ROBUST_ECF_CONTROLLER"]["CORRELATED_ECHO_SUPPORT"]["accuracy"] >= 0.95, "correlated below gate")
        require(regime["FULL_ROBUST_ECF_CONTROLLER"]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"] >= 0.95, "adversarial below gate")
        require(regime["FULL_ROBUST_ECF_CONTROLLER"]["MIXED_CLEAN_AND_CORRELATED"]["accuracy"] >= 0.95, "mixed correlated below gate")
        require(regime["FULL_ROBUST_ECF_CONTROLLER"]["MIXED_CLEAN_AND_ADVERSARIAL"]["accuracy"] >= 0.95, "mixed adversarial below gate")
        require(full["exact_operator_accuracy"] >= 0.95, "exact operator below gate")
        require(full["operator_equivalence_accuracy"] >= 0.95, "operator equivalence below gate")
        require(baseline["accuracy"] - full["accuracy"] <= 0.005, "clean regression too high")
        require(aggregate["controls_worse"], "controls_worse false")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
