#!/usr/bin/env python3
"""Artifact checker for D47 cell-reference discovery with robust support."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "policy_comparison_report.json",
    "regime_by_policy_report.json",
    "primitive_space_by_policy_report.json",
    "per_family_accuracy_report.json",
    "support_cost_frontier_report.json",
    "cell_reference_diagnostic_report.json",
    "counterfactual_effect_report.json",
    "control_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]

ARMS = [
    "CURRENT5_ORACLE_REFERENCE_ONLY",
    "ALL28_PAIR_ENUMERATION_SOFT_BASELINE",
    "CELL_REFERENCE_FACTORISED_FIELD",
    "CELL_REFERENCE_EQUIVALENCE_GROUPING",
    "COUNTERFACTUAL_TOP1_TOP2_REPAIR",
    "FULL_ROBUST_ECF_CONTROLLER",
    "FULL_ROBUST_ECF_CONTROLLER_CAP_5",
    "FULL_ROBUST_ECF_CONTROLLER_CAP_7",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_SIGNAL_CONTROL",
    "SHUFFLED_CELL_REFERENCE_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
]

ROW_FIELDS = [
    "row_id",
    "seed",
    "split",
    "policy",
    "primitive_space",
    "support_regime",
    "truth_family",
    "pred_family",
    "truth_pair",
    "pred_pair",
    "truth_equivalence",
    "pred_equivalence",
    "target_candidate",
    "pred_candidate",
    "ordered_candidate_correct",
    "unordered_pair_correct",
    "equivalence_correct",
    "family_group_correct",
    "cell_hit_top2",
    "cell_hit_top2_correct",
    "correct",
    "reference_arm",
    "support_budget_cap",
    "original_support_used",
    "counter_support_used",
    "total_support_used",
    "support_cluster_count",
    "dominant_cluster_fraction",
    "duplicate_support_score",
    "collision_count",
    "leave_one_out_flip_count",
    "entropy",
    "top1_top2_margin",
    "counter_support_requested",
    "counter_support_target",
    "counter_support_resolved",
    "correlated_echo_detected",
    "adversarial_support_detected",
    "view_signals",
    "baseline_unordered_correct",
    "error_type",
]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load(path):
    return json.loads(Path(path).read_text())


def inspect_jsonl(path):
    policies = set()
    regimes = set()
    budgets = set()
    spaces = set()
    rows = 0
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing {missing}")
            require(isinstance(row["view_signals"], dict), f"{path} row {idx} view_signals not dict")
            require(0.0 <= row["cell_hit_top2"] <= 1.0, f"{path} row {idx} invalid cell_hit_top2")
            policies.add(row["policy"])
            regimes.add(row["support_regime"])
            budgets.add(row["support_budget_cap"])
            spaces.add(row["primitive_space"])
    require(rows > 0, f"{path} empty")
    require("FULL_ROBUST_ECF_CONTROLLER" in policies, f"{path} missing full robust rows")
    require("COUNTERFACTUAL_TOP1_TOP2_REPAIR" in policies, f"{path} missing counterfactual rows")
    require("SHUFFLED_COUNTER_SUPPORT_CONTROL" in policies, f"{path} missing shuffled counter-support rows")
    require("CORRELATED_NOISE_SUPPORT" in regimes, f"{path} missing correlated regime")
    require("ADVERSARIAL_DISTRACTOR_SUPPORT" in regimes, f"{path} missing adversarial regime")
    require(5 in budgets, f"{path} missing support budget 5")
    require("ALL28_UNORDERED" in spaces, f"{path} missing primary space")


def source_guardrails(repo_root):
    runner = (repo_root / "scripts/probes/run_d47_cell_reference_discovery_with_robust_support.py").read_text()
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
    diagnostics = load(out / "cell_reference_diagnostic_report.json")
    support = load(out / "support_cost_frontier_report.json")
    report = (out / "report.md").read_text()
    require(dataset["candidate_family_equivalence_cell_metrics_separated"], "candidate/family/equivalence/cell metrics not separated")
    require(dataset["true_family_and_cells_hidden_from_fair_arms"], "truth hidden flag missing")
    require(set(ARMS).issubset(set(dataset["arms"])), "dataset manifest missing arms")
    require(set(map(str, [1, 2, 3, 4, 5])).issubset(set(support["FULL_ROBUST_ECF_CONTROLLER"].keys())), "support budget frontier incomplete")
    primary = aggregate["primary_policy_metrics"]
    for arm in ARMS:
        require(arm in primary, f"primary metrics missing {arm}")
        for field in [
            "ordered_candidate_accuracy",
            "unordered_pair_accuracy",
            "equivalence_accuracy",
            "family_group_accuracy",
            "cell_hit_top2_accuracy",
            "average_total_support_used",
        ]:
            require(field in primary[arm], f"{arm} missing {field}")
    full = primary["FULL_ROBUST_ECF_CONTROLLER"]
    baseline = primary["ALL28_PAIR_ENUMERATION_SOFT_BASELINE"]
    random_control = primary["RANDOM_EXTRA_SUPPORT_CONTROL"]
    bad = primary["BAD_SIGNAL_CONTROL"]
    shuffled_cell = primary["SHUFFLED_CELL_REFERENCE_CONTROL"]
    shuffled_counter = primary["SHUFFLED_COUNTER_SUPPORT_CONTROL"]
    no_counter = primary["NO_COUNTERFACTUAL_CONTROL"]
    require("echo_detection_quality" in diagnostics, "echo diagnostic missing")
    require("adversarial_detection_quality" in diagnostics, "adversarial diagnostic missing")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require("no raw visual Raven" in decision["boundary"], "boundary missing in decision")
    require("D47 only tests controlled symbolic cell-reference discovery" in report, "boundary missing in report")
    allowed = {
        "cell_reference_discovery_with_robust_support_positive",
        "cell_reference_factorisation_not_confirmed",
        "robust_support_transfer_failed",
        "cell_reference_positive_high_support_cost",
        "d47_instrumentation_incomplete",
    }
    require(decision["decision"] in allowed, f"unexpected decision {decision['decision']}")
    if decision["decision"] == "cell_reference_discovery_with_robust_support_positive":
        require(full["clean_accuracy"] >= 0.995, "full clean below gate")
        require(full["correlated_accuracy"] >= 0.95, "full correlated below gate")
        require(full["adversarial_accuracy"] >= 0.95, "full adversarial below gate")
        require(full["mixed_accuracy"] >= 0.95, "full mixed below gate")
        require(full["equivalence_accuracy"] >= 0.95, "full equivalence below gate")
        require(full["cell_hit_top2_accuracy"] >= 0.95, "full cell hit below gate")
        require(baseline["clean_accuracy"] - full["clean_accuracy"] <= 0.005, "clean regression too high")
        for control in [random_control, bad, shuffled_cell, shuffled_counter, no_counter]:
            require(full["correlated_accuracy"] > control["correlated_accuracy"], "a control is not worse on correlated support")
        require(aggregate["controls_worse"], "controls_worse false")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
