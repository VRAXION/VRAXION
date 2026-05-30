#!/usr/bin/env python3
"""Checker for D57 canonical Rust sparse path bridge."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d56_upstream_manifest.json",
    "rust_bridge_build_report.json",
    "rust_bridge_invocation_report.json",
    "rust_path_usage_report.json",
    "python_fallback_audit.json",
    "python_vs_rust_action_parity_report.json",
    "canonical_sparse_path_report.json",
    "action_readout_report.json",
    "firing_dynamics_report.json",
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
    "D56_PYTHON_SPARSE_REPLAY",
    "PYTHON_FALLBACK_REFERENCE",
    "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE",
    "RUST_SPIKE_SHUFFLE_CONTROL",
    "RUST_THRESHOLD_ABLATION",
    "RUST_REWIRE_ABLATION",
    "RUST_PATH_DISABLED_CONTROL",
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
    "rust_network_path_invoked",
    "rust_propagate_sparse_called",
    "python_fallback_used",
]

ALLOWED_DECISIONS = {
    "canonical_rust_sparse_path_bridge_positive",
    "rust_path_invoked_but_behavior_mismatch",
    "rust_path_not_actually_invoked",
    "canonical_rust_sparse_path_bridge_not_confirmed",
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
    rust_rows = 0
    canonical_rust_rows = 0
    canonical_python_fallback_rows = 0
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
            if row["rust_network_path_invoked"]:
                rust_rows += 1
            if row["arm"] == "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE":
                if row["rust_network_path_invoked"]:
                    canonical_rust_rows += 1
                if row["python_fallback_used"]:
                    canonical_python_fallback_rows += 1
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    require(rust_rows > 0, f"{path} has no Rust rows")
    require(canonical_rust_rows > 0, f"{path} canonical arm did not invoke Rust")
    require(canonical_python_fallback_rows == 0, f"{path} canonical arm used Python fallback")


def source_guardrails(repo_root):
    source = (repo_root / "scripts/probes/run_d57_canonical_rust_sparse_path_bridge.py").read_text(encoding="utf-8")
    compact = source.replace(" ", "").lower()
    for needle, message in [
        ("hash(", "Python hash forbidden"),
        ("random.random()<", "fake random threshold sampling forbidden"),
        ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
        ("expected_selected", "expected_selected feature forbidden"),
        ("openai", "external API reference forbidden"),
        ("modal", "Modal reference forbidden"),
    ]:
        require(needle not in compact, f"{message} in D57 runner")
    require("Network::propagate_sparse" in source, "D57 runner does not generate a Rust propagate_sparse bridge")


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
    upstream = load(out / "d56_upstream_manifest.json")
    rust_usage = load(out / "rust_path_usage_report.json")
    fallback = load(out / "python_fallback_audit.json")
    parity = load(out / "python_vs_rust_action_parity_report.json")
    canonical = load(out / "canonical_sparse_path_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    report = (out / "report.md").read_text(encoding="utf-8")

    require(dataset["truth_hidden_from_controller_inputs"] is True, "truth hidden flag missing")
    require(dataset["controller_only_not_formula_solver"] is True, "controller-only flag missing")
    require(upstream["expected_decision"] == "sparse_firing_controller_scale_confirmed_python_path_only", "bad D56 expected decision")
    require(upstream["expected_next"] == "D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE", "bad D56 expected next")
    require(upstream.get("d56_best_sparse_loaded") is True, "D56 best sparse controller not loaded")
    require(rust_usage["canonical_rust_network_path_invoked"] is True, "canonical Rust path not invoked")
    require(rust_usage["rust_propagate_sparse_called"] is True, "Rust propagate_sparse not called")
    require(rust_usage["canonical_arm_python_fallback_rows"] == 0, "canonical arm used Python fallback")
    require(fallback["canonical_arm_python_fallback_used"] is False, "fallback audit reports canonical fallback")
    require(canonical["canonical_rust_network_audited"] is True, "canonical Rust sparse surface not audited")
    require(canonical["canonical_rust_network_path_invoked"] is True, "canonical report says Rust path not invoked")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D57 only tests a canonical Rust sparse-path bridge" in decision["boundary"], "boundary missing in decision")
    require("D57 only tests a canonical Rust sparse-path bridge" in report, "boundary missing in report")
    require(summary["rust_network_path_invoked"] is True, "summary Rust flag missing")
    require(summary["python_fallback_used_for_canonical_arm"] is False, "summary fallback flag bad")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")

    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    require(set(ARMS).issubset(core.keys()), "missing core arms")
    arm = "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE"
    row = core[arm]
    corr = regime_accuracy(test, arm, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(test, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(test, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = test["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    if decision["decision"] == "canonical_rust_sparse_path_bridge_positive":
        require(row["exact_joint_accuracy"] >= 0.995, "canonical exact below gate")
        require(corr >= 0.995, "canonical correlated below gate")
        require(adv >= 0.995, "canonical adversarial below gate")
        require(external >= 0.99, "canonical external below gate")
        require(indist["abstain_rate"] >= 0.99, "canonical indistinguishable abstain below gate")
        require(indist["false_confidence_rate"] <= 0.01, "canonical false confidence above gate")
        require(row["average_total_support_used"] <= core["ALWAYS_COUNTER_CONTROL"]["average_total_support_used"], "canonical support above always-counter")
        for control in ["RANDOM_POLICY_CONTROL", "GREEDY_DECIDE_CONTROL", "RUST_SPIKE_SHUFFLE_CONTROL", "RUST_THRESHOLD_ABLATION", "RUST_REWIRE_ABLATION"]:
            require(row["accuracy"] > core[control]["accuracy"], f"{control} not worse")
        require(parity["parity_rate"] >= 0.995, "Python/Rust action parity below gate")
    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "verdict": decision.get("verdict"),
                "next": decision.get("next"),
                "rust_network_path_invoked": canonical["canonical_rust_network_path_invoked"],
                "python_vs_rust_action_parity": parity["parity_rate"],
                "canonical_exact": row["exact_joint_accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
