#!/usr/bin/env python3
"""Artifact checker for D45B robust support policy metric/component audit."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "d45_upstream_manifest.json",
    "d45_source_audit_report.json",
    "d45_metric_semantics_audit.json",
    "dataset_manifest.json",
    "component_ablation_report.json",
    "support_cost_frontier_report.json",
    "counter_support_effectiveness_report.json",
    "detection_confusion_matrix_report.json",
    "correlated_noise_repair_report.json",
    "adversarial_distractor_repair_report.json",
    "clean_regression_report.json",
    "regime_by_policy_report.json",
    "primitive_space_by_policy_report.json",
    "support_cluster_report.json",
    "leave_one_out_stability_report.json",
    "shuffled_counter_support_control_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]
ROW_FIELDS = [
    "row_id",
    "split",
    "policy",
    "primitive_space",
    "support_regime",
    "truth_family",
    "pred_family",
    "truth_equivalence",
    "pred_equivalence",
    "clean_or_robust_case",
    "original_support_used",
    "counter_support_used",
    "total_support_used",
    "support_cluster_count",
    "dominant_cluster_fraction",
    "duplicate_support_score",
    "leave_one_out_flip_count",
    "counter_support_requested",
    "counter_support_target",
    "counter_support_resolved",
    "correlated_support_detected",
    "adversarial_support_detected",
    "correct",
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
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing {missing}")
            policies.add(row["policy"])
            regimes.add(row["support_regime"])
    require("FULL_ROBUST_COMBINED_REPLAY" in policies, f"{path} missing full robust policy")
    require("SHUFFLED_COUNTER_SUPPORT_CONTROL" in policies, f"{path} missing shuffled control")
    require("CORRELATED_NOISE_SUPPORT" in regimes, f"{path} missing correlated regime")
    require("ADVERSARIAL_DISTRACTOR_SUPPORT" in regimes, f"{path} missing adversarial regime")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    require(not missing, f"missing artifacts: {missing}")
    inspect_jsonl(out / "row_outputs_test.jsonl")
    inspect_jsonl(out / "row_outputs_ood.jsonl")
    source = load(out / "d45_source_audit_report.json")
    semantics = load(out / "d45_metric_semantics_audit.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    primary = aggregate["primary_policy_metrics"]
    full = primary["FULL_ROBUST_COMBINED_REPLAY"]
    naive = primary["NAIVE_IPF_BASELINE"]
    random_control = primary["RANDOM_EXTRA_SUPPORT_CONTROL"]
    bad = primary["BAD_ROBUSTNESS_SIGNAL_CONTROL"]
    shuffled = primary["SHUFFLED_COUNTER_SUPPORT_CONTROL"]
    require(not source["uses_python_hash"], "Python hash forbidden")
    require(not source["uses_random_threshold_fake_sampling"], "fake random threshold sampling forbidden")
    require("total support" in semantics["average_support_used"], "support metric semantics not clarified")
    require(full["correlated_accuracy"] >= 0.95, "robust correlated below target")
    require(full["adversarial_accuracy"] >= 0.95, "robust adversarial below target")
    require(naive["clean_accuracy"] - full["clean_accuracy"] <= 0.005, "clean regression too high")
    require(full["correlated_accuracy"] > random_control["correlated_accuracy"], "random control not worse")
    require(full["correlated_accuracy"] > bad["correlated_accuracy"], "bad signal control not worse")
    require(full["correlated_accuracy"] > shuffled["correlated_accuracy"], "shuffled counter-support control not worse")
    require(aggregate["component_attribution_clear"], "component attribution not clear")
    require(aggregate["metric_semantics_clear"], "metric semantics not clear")
    require(aggregate["support_cost_frontier"], "support cost frontier missing")
    require(decision["decision"] in {"robust_policy_components_identified", "robust_policy_effective_but_support_cost_high"}, f"unexpected decision {decision['decision']}")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
