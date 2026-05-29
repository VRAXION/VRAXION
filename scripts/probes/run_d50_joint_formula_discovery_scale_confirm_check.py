#!/usr/bin/env python3
"""Checker for D50 joint formula discovery scale confirm."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "train_manifest.json",
    "d49b_upstream_manifest.json",
    "scale_summary.json",
    "component_ablation_report.json",
    "support_cost_frontier_report.json",
    "regime_breakdown_report.json",
    "indistinguishability_report.json",
    "external_test_required_report.json",
    "error_taxonomy_report.json",
    "control_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]

ARMS = [
    "D49B_BASELINE_REPLAY",
    "JOINT_INTERACTION_COUNTERFACTUAL",
    "MULTI_STAGE_COUNTERFACTUAL_REPAIR",
    "FULL_REPAIRED_ECF_CONTROLLER",
    "FULL_REPAIRED_ECF_CAP_7",
    "FULL_REPAIRED_ECF_CAP_9",
    "NO_CELL_COUNTERFACTUAL",
    "NO_OPERATOR_COUNTERFACTUAL",
    "NO_JOINT_INTERACTION_COUNTERFACTUAL",
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
    "joint_formula_discovery_scale_confirmed",
    "joint_formula_discovery_scale_confirmed_high_cost",
    "joint_interaction_counterfactual_required_confirmed",
    "joint_formula_discovery_scale_not_confirmed",
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
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing {missing}")
            require(0.0 <= row["confidence"] <= 1.0, f"{path} row {idx} invalid confidence")
            require(0.0 <= row["cell_hit_top2"] <= 1.0, f"{path} row {idx} invalid cell_hit_top2")
            arms.add(row["arm"])
            regimes.add(row["support_regime"])
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")


def source_guardrails(repo_root):
    runner = (repo_root / "scripts/probes/run_d50_joint_formula_discovery_scale_confirm.py").read_text(encoding="utf-8")
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
    upstream = load(out / "d49b_upstream_manifest.json")
    scale = load(out / "scale_summary.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    report = (out / "report.md").read_text(encoding="utf-8")
    metrics = aggregate["test_metrics"]
    core = metrics["by_arm_core"]
    regime = metrics["by_arm_and_regime"]
    full = core["FULL_REPAIRED_ECF_CONTROLLER"]

    require(dataset["truth_hidden_from_fair_arms"], "truth hidden flag missing")
    require(dataset["candidate_cell_operator_equivalence_metrics_separated"], "metric separation flag missing")
    require(dataset["indistinguishable_abstain_false_confidence_included"], "indistinguishable flag missing")
    require(set(ARMS).issubset(set(dataset["arms"])), "dataset manifest missing arms")
    require(set(REGIMES).issubset(set(dataset["support_regimes"])), "dataset manifest missing regimes")
    require(scale["scale_mode"] in {"full", "scale_lite"}, "bad scale mode")
    require(upstream["expected_decision"] == "joint_binding_repair_positive", "bad upstream decision")
    require(upstream["expected_next"] == "D50_JOINT_FORMULA_DISCOVERY_SCALE_CONFIRM", "bad upstream next")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D50 only scale-confirms controlled symbolic joint formula discovery" in decision["boundary"], "boundary missing in decision")
    require("D50 only scale-confirms controlled symbolic joint formula discovery" in report, "boundary missing in report")
    require(all(full["accuracy"] > core[arm]["accuracy"] for arm in ["RANDOM_EXTRA_SUPPORT_CONTROL", "SHUFFLED_COUNTER_SUPPORT_CONTROL", "NO_COUNTERFACTUAL_CONTROL"]), "controls not worse")
    require(regime["FULL_REPAIRED_ECF_CONTROLLER"]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"] <= 0.01, "false confidence above gate")

    if decision["decision"] in {
        "joint_formula_discovery_scale_confirmed",
        "joint_formula_discovery_scale_confirmed_high_cost",
        "joint_interaction_counterfactual_required_confirmed",
    }:
        require(full["exact_joint_accuracy"] >= 0.995, "exact joint below gate")
        require(regime["FULL_REPAIRED_ECF_CONTROLLER"]["CORRELATED_ECHO_SUPPORT"]["accuracy"] >= 0.995, "correlated below gate")
        require(regime["FULL_REPAIRED_ECF_CONTROLLER"]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"] >= 0.995, "adversarial below gate")
        require(regime["FULL_REPAIRED_ECF_CONTROLLER"]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"] >= 0.99, "external below gate")
        require(aggregate["min_seed_metrics"]["min_seed_exact_joint"] >= 0.99, "min seed exact below gate")

    print(
        json.dumps(
            {
                "status": "ok",
                "scale_mode": scale["scale_mode"],
                "decision": decision["decision"],
                "verdict": decision.get("verdict"),
                "next": decision.get("next"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
