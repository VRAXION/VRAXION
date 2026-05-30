#!/usr/bin/env python3
"""Checker for D56 sparse firing ECF controller scale confirm."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d55_upstream_manifest.json",
    "sparse_firing_usage_report.json",
    "canonical_sparse_path_report.json",
    "python_local_vs_rust_path_report.json",
    "mutation_causality_report.json",
    "network_topology_report.json",
    "firing_dynamics_report.json",
    "mutation_acceptance_report.json",
    "action_readout_report.json",
    "support_cost_frontier_report.json",
    "false_confidence_report.json",
    "regime_breakdown_report.json",
    "ablation_report.json",
    "controller_comparison_report.json",
    "min_seed_gate_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
    "trained_policy_manifest.json",
]

ARMS = [
    "D55_BEST_SPARSE_REPLAY",
    "D54_BEST_HYBRID_REPLAY",
    "D50_HANDCODED_FULL_REFERENCE",
    "REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION",
    "REAL_SPARSE_FIRING_CONTROLLER_NO_MUTATION",
    "SMALL_SPARSE_CONTROLLER",
    "MEDIUM_SPARSE_CONTROLLER",
    "SPIKE_SHUFFLE_CONTROL",
    "THRESHOLD_ABLATION",
    "CONNECTION_REWIRE_ABLATION",
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
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
    "sparse_firing_ecf_controller_scale_confirmed",
    "sparse_firing_controller_scale_confirmed_python_path_only",
    "sparse_firing_ecf_controller_scale_not_confirmed",
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
    spike_updates = 0
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing {missing}")
            expected_total = row["original_support_used"] + row["counter_support_used"] + row["external_test_used"]
            require(row["total_support_used"] == expected_total, f"{path} row {idx} support accounting mismatch")
            arms.add(row["arm"])
            regimes.add(row["support_regime"])
            if row["sparse_firing_used"]:
                require(row["spike_update_executed"] is True, f"{path} row {idx} sparse row without update")
                require(row["spike_update_count"] > 0, f"{path} row {idx} sparse row zero update")
                spike_updates += row["spike_update_count"]
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    require(spike_updates > 0, f"{path} has no spike updates")


def source_guardrails(repo_root):
    for filename in [
        "scripts/probes/run_d56_sparse_firing_ecf_controller_scale_confirm.py",
        "scripts/probes/run_d55_sparse_firing_ecf_controller_prototype.py",
    ]:
        source = (repo_root / filename).read_text(encoding="utf-8")
        compact = source.replace(" ", "").lower()
        for needle, message in [
            ("hash(", "Python hash forbidden"),
            ("random.random()<", "fake random threshold sampling forbidden"),
            ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
            ("expected_selected", "expected_selected feature forbidden"),
            ("openai", "external API reference forbidden"),
            ("modal", "Modal reference forbidden"),
        ]:
            require(needle not in compact, f"{message} in {filename}")


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
    upstream = load(out / "d55_upstream_manifest.json")
    usage = load(out / "sparse_firing_usage_report.json")
    canonical = load(out / "canonical_sparse_path_report.json")
    bridge = load(out / "python_local_vs_rust_path_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    report = (out / "report.md").read_text(encoding="utf-8")

    require(dataset["truth_hidden_from_controller_inputs"] is True, "truth hidden flag missing")
    require(dataset["controller_only_not_formula_solver"] is True, "controller-only flag missing")
    require(upstream["expected_decision"] == "sparse_firing_ecf_controller_prototype_strong_positive", "bad D55 expected decision")
    require(upstream["expected_next"] == "D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRM", "bad D55 expected next")
    require(upstream.get("d55_best_sparse_loaded") is True, "D55 best sparse controller not loaded")
    require(usage["sparse_firing_used"] is True, "sparse firing not used")
    require(usage["actual_spike_update_executed"] is True, "spike update not executed")
    require(usage["spike_update_executed_count"] > 0, "spike update count zero")
    require(usage["full_sparse_firing_brain_trained"] is False, "overclaims full sparse brain")
    require(canonical["canonical_rust_network_audited"] is True, "canonical Rust sparse surface not audited")
    require("canonical_rust_network_path_invoked" in canonical, "canonical Rust invocation status missing")
    require(bridge["python_controller_local_sparse_path_used"] is True, "python sparse path report missing")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D56 only scale-confirms a sparse-firing ECF controller" in decision["boundary"], "boundary missing in decision")
    require("D56 only scale-confirms a sparse-firing ECF controller" in report, "boundary missing in report")
    require(summary["sparse_firing_used"] is True, "summary sparse flag missing")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")

    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    require(set(ARMS).issubset(core.keys()), "missing core arms")
    best = decision.get("best_sparse_arm", summary["best_sparse_arm"])
    require(best in {"REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION", "D55_BEST_SPARSE_REPLAY"}, "bad best sparse arm")
    row = core[best]
    corr = regime_accuracy(test, best, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(test, best, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(test, best, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = regimes[best]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    min_seed = load(out / "min_seed_gate_report.json")[best]

    if decision["decision"] != "sparse_firing_ecf_controller_scale_not_confirmed":
        require(row["exact_joint_accuracy"] >= 0.995, "best sparse exact below gate")
        require(corr >= 0.995, "best sparse correlated below gate")
        require(adv >= 0.995, "best sparse adversarial below gate")
        require(external >= 0.99, "best sparse external below gate")
        require(indist["abstain_rate"] >= 0.99, "best sparse indistinguishable abstain below gate")
        require(indist["false_confidence_rate"] <= 0.01, "best sparse false confidence above gate")
        require(row["average_total_support_used"] <= core["ALWAYS_COUNTER_CONTROL"]["average_total_support_used"], "best support above always-counter")
        require(row["accuracy"] > core["RANDOM_POLICY_CONTROL"]["accuracy"], "random control not worse")
        require(row["accuracy"] > core["GREEDY_DECIDE_CONTROL"]["accuracy"], "greedy control not worse")
        require(row["accuracy"] > core["SPIKE_SHUFFLE_CONTROL"]["accuracy"], "spike shuffle control not worse")
        require(row["accuracy"] > core["THRESHOLD_ABLATION"]["accuracy"], "threshold ablation not worse")
        require(row["accuracy"] > core["CONNECTION_REWIRE_ABLATION"]["accuracy"], "rewire ablation not worse")
        require(min_seed["min_seed_exact_joint"] >= 0.99, "min seed exact below gate")
        require(min_seed["min_seed_correlated_echo"] >= 0.99, "min seed correlated below gate")
        require(min_seed["min_seed_adversarial_distractor"] >= 0.99, "min seed adversarial below gate")
        if canonical["canonical_rust_network_path_invoked"] is False:
            require(
                decision["decision"] == "sparse_firing_controller_scale_confirmed_python_path_only",
                "python-only pass must route to bridge",
            )

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "verdict": decision.get("verdict"),
                "next": decision.get("next"),
                "best_sparse_arm": best,
                "scale_mode": aggregate["scale_mode"],
                "spike_update_executed_count": usage["spike_update_executed_count"],
                "rust_network_path_invoked": canonical["canonical_rust_network_path_invoked"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
