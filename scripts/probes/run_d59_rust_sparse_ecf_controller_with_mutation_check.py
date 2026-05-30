#!/usr/bin/env python3
"""Checker for D59 Rust sparse ECF controller with mutation."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d58_upstream_manifest.json",
    "rust_mutation_representation_report.json",
    "rust_invocation_report.json",
    "rust_path_usage_report.json",
    "python_fallback_audit.json",
    "mutation_acceptance_report.json",
    "fitness_landscape_report.json",
    "before_after_mutation_report.json",
    "support_cost_frontier_report.json",
    "false_confidence_report.json",
    "regime_breakdown_report.json",
    "ablation_report.json",
    "controller_comparison_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
    "trained_policy_manifest.json",
]

ARMS = [
    "D58_RUST_REPLAY_REFERENCE",
    "RUST_SPARSE_MUTATION_CONTROLLER",
    "RUST_SPARSE_MUTATION_CONTROLLER_COST_WEIGHTED",
    "RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED",
    "MUTATION_DISABLED_CONTROL",
    "RANDOM_MUTATION_CONTROL",
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "RUST_SPIKE_SHUFFLE_CONTROL",
    "RUST_THRESHOLD_ABLATION",
    "RUST_REWIRE_ABLATION",
]

RUST_ARMS = [
    "D58_RUST_REPLAY_REFERENCE",
    "RUST_SPARSE_MUTATION_CONTROLLER",
    "RUST_SPARSE_MUTATION_CONTROLLER_COST_WEIGHTED",
    "RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED",
    "MUTATION_DISABLED_CONTROL",
    "RANDOM_MUTATION_CONTROL",
    "RUST_THRESHOLD_ABLATION",
    "RUST_REWIRE_ABLATION",
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
    "rust_sparse_mutation_controller_positive",
    "rust_sparse_mutation_path_confirmed_no_gain",
    "rust_sparse_mutation_path_not_exercised",
    "rust_sparse_mutation_controller_not_confirmed",
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
    rust_rows = {arm: 0 for arm in RUST_ARMS}
    fallback_rows = {arm: 0 for arm in RUST_ARMS}
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
            if row["arm"] in rust_rows:
                if row["rust_network_path_invoked"]:
                    rust_rows[row["arm"]] += 1
                if row["python_fallback_used"]:
                    fallback_rows[row["arm"]] += 1
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    for arm in RUST_ARMS:
        require(rust_rows[arm] > 0, f"{path} {arm} did not invoke Rust")
        require(fallback_rows[arm] == 0, f"{path} {arm} used Python fallback")


def source_guardrails(repo_root):
    source = (repo_root / "scripts/probes/run_d59_rust_sparse_ecf_controller_with_mutation.py").read_text(encoding="utf-8")
    compact = source.replace(" ", "").lower()
    for needle, message in [
        ("hash(", "Python hash forbidden"),
        ("random.random()<", "fake random threshold sampling forbidden"),
        ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
        ("expected_selected", "expected_selected feature forbidden"),
        ("openai", "external API reference forbidden"),
        ("modal", "Modal reference forbidden"),
    ]:
        require(needle not in compact, f"{message} in D59 runner")
    require("Network::propagate_sparse" in source, "D59 runner does not generate Rust propagate_sparse candidate eval")
    require("mutate_controller" in source, "D59 runner missing mutation function")


def regime_accuracy(metrics, arm, regime):
    return metrics["by_arm_and_regime"][arm][regime]["accuracy"]


def mutation_exercised(report):
    return any(value.get("accepted_total", 0) > 0 for value in report.values())


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
    upstream = load(out / "d58_upstream_manifest.json")
    representation = load(out / "rust_mutation_representation_report.json")
    rust_usage = load(out / "rust_path_usage_report.json")
    fallback = load(out / "python_fallback_audit.json")
    mutation = load(out / "mutation_acceptance_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    report = (out / "report.md").read_text(encoding="utf-8")

    require(dataset["truth_hidden_from_controller_inputs"] is True, "truth hidden flag missing")
    require(dataset["controller_only_not_formula_solver"] is True, "controller-only flag missing")
    require(dataset["train_validation_test_ood_separated"] is True, "train/validation/test/ood separation missing")
    require(upstream["expected_decision"] == "canonical_rust_sparse_controller_scale_confirmed", "bad D58 expected decision")
    require(upstream["expected_next"] == "D59_RUST_SPARSE_ECF_CONTROLLER_WITH_MUTATION", "bad D58 expected next")
    require(upstream.get("d58_canonical_controller_loaded") is True, "D58 canonical controller not loaded")
    require(representation["candidate_eval_path"].endswith("Network::propagate_sparse"), "candidate eval path not Rust sparse")
    require(fallback["rust_arms_have_zero_fallback"] is True, "Rust arms used Python fallback")
    require(mutation_exercised(mutation), "no accepted mutations reported")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D59 only tests mutation and selection" in decision["boundary"], "boundary missing in decision")
    require("D59 only tests mutation and selection" in report, "boundary missing in report")
    require(summary["rust_path_invoked"] is True, "summary Rust path false")
    require(summary["fallback_rows"] == 0, "summary fallback nonzero")
    require(summary["mutation_path_exercised"] is True, "summary mutation path not exercised")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")

    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    require(set(ARMS).issubset(core.keys()), "missing core arms")
    best = decision.get("best_arm") or summary["best_arm"]
    row = core[best]
    corr = regime_accuracy(test, best, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(test, best, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(test, best, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = test["by_arm_and_regime"][best]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    if decision["decision"] in {"rust_sparse_mutation_controller_positive", "rust_sparse_mutation_path_confirmed_no_gain"}:
        require(best in {"RUST_SPARSE_MUTATION_CONTROLLER", "RUST_SPARSE_MUTATION_CONTROLLER_COST_WEIGHTED", "RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED"}, "best is not mutated arm")
        require(rust_usage["by_arm"][best]["rust_rows"] > 0, "best arm no Rust rows")
        require(rust_usage["by_arm"][best].get("python_fallback_rows", 0) == 0, "best arm fallback rows nonzero")
        require(row["exact_joint_accuracy"] >= 0.995, "best exact below gate")
        require(corr >= 0.995, "best correlated below gate")
        require(adv >= 0.995, "best adversarial below gate")
        require(external >= 0.99, "best external below gate")
        require(indist["abstain_rate"] >= 0.99, "best abstain below gate")
        require(indist["false_confidence_rate"] <= 0.01, "best false confidence above gate")
        for control in ["RANDOM_POLICY_CONTROL", "GREEDY_DECIDE_CONTROL", "RUST_SPIKE_SHUFFLE_CONTROL", "RUST_THRESHOLD_ABLATION", "RUST_REWIRE_ABLATION"]:
            require(row["accuracy"] > core[control]["accuracy"], f"{control} not worse")
        require(core["ALWAYS_COUNTER_CONTROL"]["average_total_support_used"] > row["average_total_support_used"], "always-counter not higher cost")
    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "verdict": decision.get("verdict"),
                "next": decision.get("next"),
                "best_arm": best,
                "best_exact": row["exact_joint_accuracy"],
                "mutation_path_exercised": mutation_exercised(mutation),
                "fallback_rows": summary["fallback_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
