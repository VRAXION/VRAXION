#!/usr/bin/env python3
"""Checker for D49 joint cell+operator discovery with robust support."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "train_manifest.json",
    "indistinguishability_certificate_report.json",
    "policy_comparison_report.json",
    "regime_by_policy_report.json",
    "support_cost_frontier_report.json",
    "counterfactual_effect_report.json",
    "exact_equivalence_audit_report.json",
    "indistinguishable_abstain_report.json",
    "false_confidence_report.json",
    "control_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]

ARMS = [
    "LABEL_ECHO_REFERENCE_ONLY",
    "JOINT_ENUMERATION_SOFT_BASELINE",
    "CELL_THEN_OPERATOR_PIPELINE",
    "OPERATOR_THEN_CELL_PIPELINE",
    "JOINT_SOFT_FIELD_FACTORISED",
    "COUNTERFACTUAL_TOP1_TOP2_REPAIR",
    "FULL_ROBUST_ECF_CONTROLLER",
    "FULL_ROBUST_ECF_CAP_5",
    "FULL_ROBUST_ECF_CAP_7",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_SIGNAL_CONTROL",
    "SHUFFLED_CELL_CONTROL",
    "SHUFFLED_OPERATOR_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
    "ABSTAIN_ON_INDISTINGUISHABLE_CASES",
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
    "primitive_space",
    "support_regime",
    "truth_joint",
    "pred_joint",
    "false_joint",
    "truth_pair",
    "pred_pair",
    "false_pair",
    "truth_pair_equivalence",
    "pred_pair_equivalence",
    "truth_operator",
    "pred_operator",
    "false_operator",
    "truth_operator_equivalence",
    "pred_operator_equivalence",
    "truth_group",
    "pred_group",
    "exact_joint_correct",
    "cell_pair_equivalence_correct",
    "cell_hit_top2",
    "cell_hit_top2_correct",
    "operator_exact_correct",
    "operator_equivalence_correct",
    "family_group_correct",
    "correct",
    "reference_arm",
    "support_budget_cap",
    "original_support_used",
    "counter_support_used",
    "external_test_used",
    "total_support_used",
    "support_cluster_count",
    "dominant_cluster_fraction",
    "collision_count",
    "correlated_echo_detected",
    "counter_support_requested",
    "counter_support_resolved",
    "oracle_distinguishable",
    "external_test_available",
    "abstained",
    "false_confidence",
    "confidence",
    "top1_top2_margin",
    "entropy",
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
            require(0.0 <= row["cell_hit_top2"] <= 1.0, f"{path} row {idx} invalid cell_hit_top2")
            require(row["total_support_used"] >= row["original_support_used"], f"{path} row {idx} bad support count")
            arms.add(row["arm"])
            regimes.add(row["support_regime"])
            budgets.add(row["support_budget_cap"])
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    require(5 in budgets, f"{path} missing budget 5")


def source_guardrails(repo_root):
    runner = (repo_root / "scripts/probes/run_d49_joint_cell_operator_discovery_with_robust_support.py").read_text(
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
    cert = load(out / "indistinguishability_certificate_report.json")
    audit = load(out / "exact_equivalence_audit_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    frontier = load(out / "support_cost_frontier_report.json")
    report = (out / "report.md").read_text(encoding="utf-8")
    metrics = aggregate["test_metrics"]
    primary = metrics["by_arm_core"]
    regime = metrics["by_arm_and_regime"]
    require(dataset["truth_hidden_from_fair_arms"], "truth-hidden flag missing")
    require(dataset["label_echo_reference_only_not_fair"], "label echo reference flag missing")
    require(dataset["candidate_family_equivalence_cell_operator_metrics_separated"], "metric separation flag missing")
    require(dataset["row_outputs_are_sampled_but_metrics_use_full_rows"], "row sampling/full metrics flag missing")
    require(set(ARMS).issubset(set(dataset["arms"])), "dataset manifest missing arms")
    require(set(REGIMES).issubset(set(dataset["support_regimes"])), "dataset manifest missing regimes")
    require(set(map(str, [1, 2, 3, 4, 5])).issubset(set(frontier.keys())), "support frontier missing budgets")
    require("INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" in cert["summary"], "missing indistinguishable cert summary")
    require(cert["summary"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["internal_delta_max"] == 0.0, "indistinguishable internal delta not zero")
    require(cert["summary"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["internal_delta_max"] == 0.0, "external-required internal delta not zero")
    require(cert["summary"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["external_delta_max"] > 0.0, "external test did not separate")
    require(not audit["operator_equivalence_below_exact"], "operator equivalence below exact")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require("D49 only tests controlled symbolic joint cell+operator discovery" in decision["boundary"], "boundary missing in decision")
    require("D49 only tests controlled symbolic joint cell+operator discovery" in report, "boundary missing in report")
    allowed = {
        "joint_cell_operator_discovery_with_robust_support_positive",
        "joint_discovery_positive_high_support_cost",
        "joint_binding_bottleneck",
        "false_confidence_under_joint_indistinguishability",
    }
    require(decision["decision"] in allowed, f"unexpected decision {decision['decision']}")
    if decision["decision"] == "joint_cell_operator_discovery_with_robust_support_positive":
        full = primary["FULL_ROBUST_ECF_CONTROLLER"]
        require(regime["FULL_ROBUST_ECF_CONTROLLER"]["CLEAN_INDEPENDENT_SUPPORT"]["accuracy"] >= 0.995, "clean below gate")
        require(regime["FULL_ROBUST_ECF_CONTROLLER"]["CORRELATED_ECHO_SUPPORT"]["accuracy"] >= 0.95, "correlated below gate")
        require(regime["FULL_ROBUST_ECF_CONTROLLER"]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"] >= 0.95, "adversarial below gate")
        require(regime["FULL_ROBUST_ECF_CONTROLLER"]["MIXED_CLEAN_AND_CORRELATED"]["accuracy"] >= 0.95, "mixed correlated below gate")
        require(regime["FULL_ROBUST_ECF_CONTROLLER"]["MIXED_CLEAN_AND_ADVERSARIAL"]["accuracy"] >= 0.95, "mixed adversarial below gate")
        for field in [
            "exact_joint_accuracy",
            "cell_pair_equivalence_accuracy",
            "cell_hit_top2_accuracy",
            "operator_exact_accuracy",
            "operator_equivalence_accuracy",
        ]:
            require(full[field] >= 0.95, f"{field} below gate")
        require(
            regime["ABSTAIN_ON_INDISTINGUISHABLE_CASES"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"]
            <= 0.01,
            "indistinguishable false confidence too high",
        )
        require(
            regime["ABSTAIN_ON_INDISTINGUISHABLE_CASES"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"]
            >= 0.95,
            "indistinguishable abstain below gate",
        )
        require(aggregate["controls_worse"], "controls_worse false")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
