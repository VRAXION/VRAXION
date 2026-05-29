#!/usr/bin/env python3
"""Checker for D47B indistinguishable correlated support sanity check."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "identifiability_report.json",
    "indistinguishability_certificate_report.json",
    "counter_support_availability_report.json",
    "abstain_behavior_report.json",
    "false_confidence_report.json",
    "regime_metrics_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]

ARMS = [
    "NAIVE_IPF",
    "ROBUST_ECF_COUNTER_SUPPORT",
    "ROBUST_ECF_WITH_ABSTAIN",
    "ROBUST_ECF_WITH_EXTERNAL_TEST",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "LABEL_ECHO_REFERENCE_ONLY",
]

REGIMES = [
    "INDEPENDENT_TRUE_SUPPORT",
    "CORRELATED_TRUE_SUPPORT",
    "DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
    "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT",
    "EXTERNAL_TEST_REQUIRED_SUPPORT",
]

ROW_FIELDS = [
    "row_id",
    "seed",
    "split",
    "arm",
    "support_regime",
    "primitive_space",
    "truth_family",
    "pred_family",
    "truth_pair",
    "false_pair",
    "truth_candidate",
    "false_candidate",
    "pred_candidate",
    "truth_equivalence",
    "pred_equivalence",
    "original_support_used",
    "counter_support_used",
    "external_test_used",
    "total_support_used",
    "support_cluster_count",
    "dominant_cluster_fraction",
    "collision_count",
    "correlated_support_detected",
    "counter_support_requested",
    "counter_support_resolved",
    "oracle_distinguishable",
    "external_test_available",
    "identifiability_upper_bound",
    "abstained",
    "correct",
    "accuracy_credit",
    "effective_accuracy_counting_abstain_wrong",
    "confidence",
    "false_confidence",
    "top1_top2_margin",
    "entropy",
    "reference_arm",
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
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")


def source_guardrails(repo_root):
    runner = (repo_root / "scripts/probes/run_d47b_indistinguishable_correlated_support_sanity_check.py").read_text(
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
    ident = load(out / "identifiability_report.json")
    certs = load(out / "indistinguishability_certificate_report.json")
    counter = load(out / "counter_support_availability_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    report = (out / "report.md").read_text(encoding="utf-8")
    metrics = aggregate["test_metrics"]["by_arm_and_regime"]
    require(dataset["true_family_hidden_from_fair_arms"], "truth-hidden flag missing")
    require(dataset["label_echo_reference_only_not_fair"], "label echo reference flag missing")
    require(dataset["identifiability_upper_bound_reported"], "upper bound flag missing")
    require(set(ARMS).issubset(set(dataset["arms"])), "dataset manifest missing arms")
    require("INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" in certs["summary"], "missing indistinguishable cert summary")
    require(
        certs["summary"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["internal_counter_max_delta"] == 0.0,
        "indistinguishable internal delta is not zero",
    )
    require(
        certs["summary"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["internal_counter_max_delta"] == 0.0,
        "external-required internal delta is not zero",
    )
    require(
        certs["summary"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["external_counter_max_delta"] > 0.0,
        "external-required external delta did not separate",
    )
    require(counter["distinguishable_false_counter_support_available"], "counter support availability false")
    require(ident["identifiability_limit_confirmed"], "identifiability limit not confirmed")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require("D47B only tests identifiability limits" in decision["boundary"], "boundary missing in decision")
    require("D47B only tests identifiability limits" in report, "boundary missing in report")
    allowed = {
        "indistinguishability_boundary_confirmed",
        "false_confidence_under_indistinguishable_support",
        "counter_support_repair_not_confirmed",
    }
    require(decision["decision"] in allowed, f"unexpected decision {decision['decision']}")
    if decision["decision"] == "indistinguishability_boundary_confirmed":
        require(metrics["ROBUST_ECF_WITH_ABSTAIN"]["INDEPENDENT_TRUE_SUPPORT"]["accuracy"] >= 0.95, "independent true failed")
        require(metrics["ROBUST_ECF_WITH_ABSTAIN"]["CORRELATED_TRUE_SUPPORT"]["accuracy"] >= 0.95, "correlated true failed")
        require(
            metrics["ROBUST_ECF_COUNTER_SUPPORT"]["DISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["accuracy"] >= 0.95,
            "distinguishable false not repaired",
        )
        require(
            metrics["ROBUST_ECF_WITH_ABSTAIN"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"] >= 0.95,
            "indistinguishable false was not abstained",
        )
        require(
            metrics["ROBUST_ECF_WITH_ABSTAIN"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"][
                "false_confidence_rate"
            ]
            <= 0.01,
            "abstain arm is falsely confident",
        )
        require(
            metrics["ROBUST_ECF_WITH_EXTERNAL_TEST"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"] >= 0.95,
            "external test arm failed",
        )
        require(
            metrics["ROBUST_ECF_COUNTER_SUPPORT"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"] < 0.95,
            "internal-only arm solved external-required case",
        )
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
