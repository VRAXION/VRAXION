#!/usr/bin/env python3
"""Artifact checker for D45 robust support policy prototype."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "d44g_upstream_manifest.json",
    "home_repo_catchup_report.json",
    "dataset_manifest.json",
    "support_regime_report.json",
    "naive_ipf_baseline_report.json",
    "staged_support_baseline_report.json",
    "duplicate_support_downweighting_report.json",
    "source_diversity_weighting_report.json",
    "leave_one_support_out_stability_report.json",
    "robust_median_field_aggregation_report.json",
    "counter_support_query_policy_report.json",
    "adversarial_distractor_detector_report.json",
    "robust_combined_policy_report.json",
    "random_extra_support_control_report.json",
    "bad_robustness_signal_control_report.json",
    "correlated_noise_repair_report.json",
    "adversarial_distractor_repair_report.json",
    "support_independence_report.json",
    "counter_support_effectiveness_report.json",
    "policy_comparison_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]

POLICIES = [
    "NAIVE_IPF_BASELINE",
    "STAGED_SUPPORT_BASELINE",
    "DUPLICATE_SUPPORT_DOWNWEIGHTING",
    "SOURCE_DIVERSITY_WEIGHTING",
    "LEAVE_ONE_SUPPORT_OUT_STABILITY",
    "ROBUST_MEDIAN_FIELD_AGGREGATION",
    "COUNTER_SUPPORT_QUERY_POLICY",
    "ADVERSARIAL_DISTRACTOR_DETECTOR",
    "ROBUST_COMBINED_POLICY",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_ROBUSTNESS_SIGNAL_CONTROL",
]

ROW_FIELDS = [
    "row_id",
    "split",
    "support_regime",
    "primitive_space",
    "policy",
    "truth_family",
    "pred_family",
    "truth_equivalence",
    "pred_equivalence",
    "support_used",
    "support_cluster_count",
    "dominant_cluster_fraction",
    "top1_top2_margin",
    "entropy",
    "collision_count",
    "correlated_support_detected",
    "adversarial_support_detected",
    "counter_support_requested",
    "counter_support_resolved",
    "correct",
    "error_type",
]


def load_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def inspect_rows(path):
    seen_policies = set()
    seen_regimes = set()
    seen_spaces = set()
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing fields {missing}")
            seen_policies.add(row["policy"])
            seen_regimes.add(row["support_regime"])
            seen_spaces.add(row["primitive_space"])
    require(seen_policies == set(POLICIES), f"{path} missing policy coverage: {set(POLICIES) - seen_policies}")
    require("CORRELATED_NOISE_SUPPORT" in seen_regimes, f"{path} missing correlated regime")
    require("ADVERSARIAL_DISTRACTOR_SUPPORT" in seen_regimes, f"{path} missing adversarial regime")
    require("ALL28_UNORDERED" in seen_spaces, f"{path} missing ALL28 space")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    require(not missing, f"missing artifacts: {missing}")
    inspect_rows(out / "row_outputs_test.jsonl")
    inspect_rows(out / "row_outputs_ood.jsonl")
    aggregate = load_json(out / "aggregate_metrics.json")
    decision = load_json(out / "decision.json")
    require(aggregate["candidate_family_equivalence_metrics_separated"], "candidate/family/equivalence metrics not separated")
    require(aggregate["clean_correlated_adversarial_regimes_separated"], "regimes not separated")
    require(aggregate["no_label_echo_as_fair_oracle"], "label echo oracle forbidden")
    require(aggregate["no_python_hash"], "Python hash forbidden")
    require(aggregate["no_fake_sampling"], "fake sampling forbidden")
    require(aggregate["no_fixed_synthetic_accuracy_dict"], "fixed synthetic accuracy dict forbidden")
    require(aggregate["true_family_hidden_from_fair_arms"], "true family must be hidden from fair arms")
    robust = aggregate["primary_robust_combined"]
    naive = aggregate["naive_baseline"]
    random_control = aggregate["random_extra_support_control"]
    bad_control = aggregate["bad_robustness_signal_control"]
    require(robust["clean_test_accuracy"] >= 0.995, "clean accuracy below D45 target")
    require(robust["correlated_support_test_accuracy"] >= 0.90, "correlated support repair below D45 target")
    require(robust["adversarial_support_test_accuracy"] >= 0.95, "adversarial support repair below D45 target")
    require(robust["correlated_support_test_accuracy"] - naive["correlated_support_test_accuracy"] >= 0.50, "correlated gain too small")
    require(robust["adversarial_support_test_accuracy"] - naive["adversarial_support_test_accuracy"] >= 0.05, "adversarial gain too small")
    require(robust["correlated_support_test_accuracy"] > random_control["correlated_support_test_accuracy"], "random extra support control not worse")
    require(robust["correlated_support_test_accuracy"] > bad_control["correlated_support_test_accuracy"], "bad robustness signal control not worse")
    require(naive["clean_test_accuracy"] - robust["clean_test_accuracy"] <= 0.005, "clean regression too large")
    require(decision["decision"] == "robust_support_policy_prototype_positive", f"unexpected decision {decision['decision']}")
    boundary = decision.get("boundary", "")
    for forbidden_claim in ["raw visual Raven", "Raven solved", "DNA/genome success", "consciousness", "AGI", "architecture superiority"]:
        require(forbidden_claim in boundary, f"boundary missing {forbidden_claim}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
