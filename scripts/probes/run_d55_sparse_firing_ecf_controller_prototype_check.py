#!/usr/bin/env python3
"""Checker for D55 sparse firing ECF controller prototype."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d54_upstream_manifest.json",
    "canonical_sparse_firing_audit_report.json",
    "sparse_firing_usage_report.json",
    "network_topology_report.json",
    "firing_dynamics_report.json",
    "mutation_acceptance_report.json",
    "action_readout_report.json",
    "threshold_ablation_report.json",
    "spike_shuffle_control_report.json",
    "support_cost_frontier_report.json",
    "false_confidence_report.json",
    "regime_breakdown_report.json",
    "controller_comparison_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
    "trained_policy_manifest.json",
]

REFERENCE_ARMS = [
    "D54_BEST_HYBRID_REPLAY",
    "D54_SPARSE_GATE_REPLAY",
    "HANDCODED_D50_FULL_REFERENCE",
]

CONTROL_ARMS = [
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "MUTABLE_RULE_TABLE_REFERENCE",
]

SPARSE_ARMS = [
    "REAL_SPARSE_FIRING_CONTROLLER_SMALL",
    "REAL_SPARSE_FIRING_CONTROLLER_MEDIUM",
    "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION",
    "REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION",
]

ABLATION_ARMS = [
    "SPIKE_SHUFFLE_CONTROL",
    "FIRING_THRESHOLD_ABLATION",
    "CONNECTION_REWIRE_ABLATION",
]

ARMS = REFERENCE_ARMS + CONTROL_ARMS + SPARSE_ARMS + ABLATION_ARMS

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
    "sparse_firing_used",
    "spike_update_executed",
    "spike_update_count",
    "fired_gate_count",
]

ALLOWED_DECISIONS = {
    "sparse_firing_ecf_controller_prototype_strong_positive",
    "sparse_firing_ecf_controller_prototype_positive",
    "sparse_firing_path_not_exercised",
    "sparse_firing_ecf_controller_not_confirmed",
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
    sparse_rows = 0
    spike_updates = 0
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
            if row["sparse_firing_used"]:
                sparse_rows += 1
                require(row["spike_update_executed"] is True, f"{path} row {idx} sparse row without spike update")
                require(row["spike_update_count"] > 0, f"{path} row {idx} sparse row has no spike updates")
                spike_updates += row["spike_update_count"]
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    require(sparse_rows > 0, f"{path} has no sparse rows")
    require(spike_updates > 0, f"{path} has no sparse updates")


def source_guardrails(repo_root):
    runner = (repo_root / "scripts/probes/run_d55_sparse_firing_ecf_controller_prototype.py").read_text(encoding="utf-8")
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
    dataset_block = runner[runner.find("FEATURE_NAMES"): runner.find("BOUNDARY")]
    for leaked in ["truth_joint", "truth_pair", "true_operator", "false_joint"]:
        require(leaked not in dataset_block, f"truth leakage in controller feature list: {leaked}")


def regime_accuracy(metrics, arm, regime):
    return metrics["by_arm_and_regime"][arm][regime]["accuracy"]


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
    upstream = load(out / "d54_upstream_manifest.json")
    canonical = load(out / "canonical_sparse_firing_audit_report.json")
    sparse_usage = load(out / "sparse_firing_usage_report.json")
    topology = load(out / "network_topology_report.json")
    firing = load(out / "firing_dynamics_report.json")
    mutation = load(out / "mutation_acceptance_report.json")
    comparison = load(out / "controller_comparison_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    report = (out / "report.md").read_text(encoding="utf-8")

    require(dataset["truth_hidden_from_controller_inputs"] is True, "truth hidden flag missing")
    require(dataset["controller_only_not_formula_solver"] is True, "controller-only flag missing")
    require(upstream["expected_decision"] == "vraxion_sparse_gate_controller_path_confirmed", "bad D54 upstream expected decision")
    require(upstream["expected_next"] == "D55_SPARSE_FIRING_ECF_CONTROLLER_PROTOTYPE", "bad D54 upstream expected next")
    require(upstream.get("required_d54_policies_loaded", {}).get("VRAXION_MUTABLE_SPARSE_GATE_CONTROLLER") is True, "D54 sparse gate policy missing")
    require(canonical["canonical_rust_network_audited"] is True, "canonical sparse network surface missing")
    require(sparse_usage["sparse_firing_used"] is True, "sparse firing not used")
    require(sparse_usage["actual_spike_update_executed"] is True, "spike update not executed")
    require(sparse_usage["spike_update_executed_count"] > 0, "spike update count zero")
    require(sparse_usage["full_sparse_firing_brain_trained"] is False, "overclaims full sparse brain")
    require(sparse_usage["controller_only_not_formula_solver"] is True, "sparse usage must be controller-only")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D55 only tests a controller-local sparse firing ECF action policy" in decision["boundary"], "boundary missing in decision")
    require("D55 only tests a controller-local sparse firing ECF action policy" in report, "boundary missing in report")
    require(summary["sparse_firing_used"] is True, "summary sparse flag missing")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")

    for arm in SPARSE_ARMS + ["FIRING_THRESHOLD_ABLATION", "CONNECTION_REWIRE_ABLATION"]:
        require(arm in topology, f"missing topology for {arm}")
        require(topology[arm]["total_neurons"] > 0, f"bad topology for {arm}")
    require("REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION" in mutation, "missing sparse mutation report")
    require("mutation_counts" in mutation["REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"], "missing mutation counts")
    require("test" in firing and "ood" in firing, "firing dynamics missing split")
    require("REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION" in firing["test"], "firing stats missing main sparse arm")

    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    arm = "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION"
    require(set(ARMS).issubset(core.keys()), "missing core arm metrics")
    require(set(ARMS).issubset(comparison.keys()), "missing comparison arms")
    row = core[arm]
    corr = regime_accuracy(test, arm, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(test, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(test, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]

    if decision["decision"] != "sparse_firing_ecf_controller_not_confirmed":
        require(row["exact_joint_accuracy"] >= 0.98, "sparse exact below gate")
        require(corr >= 0.95, "sparse correlated below gate")
        require(adv >= 0.95, "sparse adversarial below gate")
        require(external >= 0.95, "sparse external below gate")
        require(indist["abstain_rate"] >= 0.99, "sparse indistinguishable abstain below gate")
        require(indist["false_confidence_rate"] <= 0.01, "sparse false confidence above gate")
        require(row["average_total_support_used"] <= core["ALWAYS_COUNTER_CONTROL"]["average_total_support_used"], "sparse support above always counter")
        require(row["accuracy"] > core["RANDOM_POLICY_CONTROL"]["accuracy"], "random control not worse")
        require(row["accuracy"] > core["GREEDY_DECIDE_CONTROL"]["accuracy"], "greedy control not worse")
        require(row["accuracy"] > core["SPIKE_SHUFFLE_CONTROL"]["accuracy"], "spike shuffle control not worse")
        require(core["FIRING_THRESHOLD_ABLATION"]["accuracy"] < row["accuracy"], "threshold ablation not worse")
        require(core["CONNECTION_REWIRE_ABLATION"]["accuracy"] < row["accuracy"], "rewire ablation not worse")

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "verdict": decision.get("verdict"),
                "next": decision.get("next"),
                "scale_mode": aggregate["scale_mode"],
                "sparse_firing_used": sparse_usage["sparse_firing_used"],
                "spike_update_executed_count": sparse_usage["spike_update_executed_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
