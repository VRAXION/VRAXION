#!/usr/bin/env python3
"""Checker for D49B joint binding repair."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "train_manifest.json",
    "d49_upstream_manifest.json",
    "joint_error_taxonomy_report.json",
    "cell_vs_operator_error_report.json",
    "binding_consistency_report.json",
    "counterfactual_stage_report.json",
    "external_test_required_report.json",
    "support_cost_frontier_report.json",
    "regime_breakdown_report.json",
    "control_report.json",
    "indistinguishable_false_confidence_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]

ARMS = [
    "D49_BASELINE_REPLAY",
    "FACTORISED_CELL_OPERATOR_SCORE",
    "CELL_FIRST_OPERATOR_SECOND_PIPELINE",
    "OPERATOR_FIRST_CELL_SECOND_PIPELINE",
    "JOINT_BINDING_MATRIX",
    "CELL_ONLY_COUNTERFACTUAL",
    "OPERATOR_ONLY_COUNTERFACTUAL",
    "JOINT_INTERACTION_COUNTERFACTUAL",
    "MULTI_STAGE_COUNTERFACTUAL_REPAIR",
    "FULL_REPAIRED_ECF_CONTROLLER",
    "FULL_REPAIRED_ECF_CAP_7",
    "FULL_REPAIRED_ECF_CAP_9",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "SHUFFLED_COUNTER_SUPPORT_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
    "ABSTAIN_ON_INDISTINGUISHABLE",
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
    "truth_pair",
    "pred_pair",
    "truth_pair_equivalence",
    "pred_pair_equivalence",
    "truth_operator",
    "pred_operator",
    "truth_operator_equivalence",
    "pred_operator_equivalence",
    "exact_joint_correct",
    "cell_pair_equivalence_correct",
    "cell_hit_top2",
    "operator_exact_correct",
    "operator_equivalence_correct",
    "joint_binding_consistency",
    "correct",
    "original_support_used",
    "cell_counter_support_used",
    "operator_counter_support_used",
    "joint_counter_support_used",
    "counter_support_used",
    "external_test_used",
    "total_support_used",
    "counter_support_requested",
    "counter_support_resolved",
    "oracle_distinguishable",
    "external_test_available",
    "abstained",
    "false_confidence",
    "confidence",
    "top1_top2_margin",
    "entropy",
    "error_type",
]

ALLOWED_DECISIONS = {
    "joint_binding_repair_positive",
    "joint_binding_repair_positive_high_cost",
    "joint_interaction_binding_bottleneck",
    "external_test_required_joint_gap",
    "false_confidence_under_joint_indistinguishability",
    "d49b_failed_jobs_present",
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
    errors = set()
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
            errors.add(row["error_type"])
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    require("ok" in errors, f"{path} missing ok rows")


def source_guardrails(repo_root):
    runner_path = repo_root / "scripts/probes/run_d49b_joint_binding_repair.py"
    runner = runner_path.read_text(encoding="utf-8")
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
    upstream = load(out / "d49_upstream_manifest.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    taxonomy = load(out / "joint_error_taxonomy_report.json")
    stages = load(out / "counterfactual_stage_report.json")
    external = load(out / "external_test_required_report.json")
    frontier = load(out / "support_cost_frontier_report.json")
    report = (out / "report.md").read_text(encoding="utf-8")
    metrics = aggregate["test_metrics"]
    core = metrics["by_arm_core"]
    regime = metrics["by_arm_and_regime"]

    require(dataset["truth_hidden_from_fair_arms"], "truth hidden flag missing")
    require(dataset["label_echo_reference_only_not_fair"], "label echo reference flag missing")
    require(dataset["candidate_family_equivalence_cell_operator_metrics_separated"], "metric separation flag missing")
    require(dataset["indistinguishable_false_confidence_reported"], "indistinguishable report flag missing")
    require(set(ARMS).issubset(set(dataset["arms"])), "dataset manifest missing arms")
    require(set(REGIMES).issubset(set(dataset["support_regimes"])), "dataset manifest missing regimes")
    require(upstream["expected_decision"] == "joint_binding_bottleneck", "bad upstream expected decision")
    require(upstream["expected_next"] == "D49B_JOINT_BINDING_REPAIR", "bad upstream next")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D49B only tests controlled symbolic joint cell+operator binding repair" in decision["boundary"], "boundary missing in decision")
    require("D49B only tests controlled symbolic joint cell+operator binding repair" in report, "boundary missing in report")
    require("FULL_REPAIRED_ECF_CONTROLLER" in taxonomy, "taxonomy missing full controller")
    require("stage_gains" in stages, "stage gains missing")
    require("full_cap7" in frontier and "full_cap9" in frontier and "full_controller" in frontier, "support frontier incomplete")
    require("FULL_REPAIRED_ECF_CONTROLLER" in external, "external report missing full controller")

    full = core["FULL_REPAIRED_ECF_CONTROLLER"]
    controls = ["RANDOM_EXTRA_SUPPORT_CONTROL", "SHUFFLED_COUNTER_SUPPORT_CONTROL", "NO_COUNTERFACTUAL_CONTROL"]
    require(all(full["accuracy"] > core[arm]["accuracy"] for arm in controls), "controls not worse than full controller")
    require(
        regime["FULL_REPAIRED_ECF_CONTROLLER"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"] <= 0.01,
        "indistinguishable false-confidence above gate",
    )

    if decision["decision"] in {"joint_binding_repair_positive", "joint_binding_repair_positive_high_cost"}:
        require(regime["FULL_REPAIRED_ECF_CONTROLLER"]["CLEAN_INDEPENDENT_SUPPORT"]["accuracy"] >= 0.995, "clean below gate")
        require(regime["FULL_REPAIRED_ECF_CONTROLLER"]["CORRELATED_ECHO_SUPPORT"]["accuracy"] >= 0.95, "correlated below gate")
        require(regime["FULL_REPAIRED_ECF_CONTROLLER"]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"] >= 0.95, "adversarial below gate")
        require(regime["FULL_REPAIRED_ECF_CONTROLLER"]["MIXED_CLEAN_AND_CORRELATED"]["accuracy"] >= 0.95, "mixed correlated below gate")
        require(regime["FULL_REPAIRED_ECF_CONTROLLER"]["MIXED_CLEAN_AND_ADVERSARIAL"]["accuracy"] >= 0.95, "mixed adversarial below gate")
        require(full["exact_joint_accuracy"] >= 0.97, "exact joint below gate")
        require(full["cell_pair_equivalence_accuracy"] >= 0.97, "cell pair below gate")
        require(full["operator_exact_accuracy"] >= 0.97, "operator exact below gate")

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "verdict": decision.get("verdict"),
                "next": decision.get("next"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
