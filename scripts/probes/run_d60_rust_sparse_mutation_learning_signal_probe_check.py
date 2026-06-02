#!/usr/bin/env python3
"""Checker for D60 Rust sparse mutation learning-signal probe."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d59_upstream_manifest.json",
    "task_difficulty_report.json",
    "oracle_upper_bound_report.json",
    "saturated_track_report.json",
    "hard_learning_track_report.json",
    "mutation_causality_report.json",
    "accepted_mutation_delta_report.json",
    "pareto_frontier_report.json",
    "support_cost_frontier_report.json",
    "safety_constraint_report.json",
    "fitness_definition_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
    "trained_policy_manifest.json",
]

ARMS = [
    "D58_REPLAY_REFERENCE",
    "D59_BEST_REPLAY",
    "MUTATION_DISABLED_CONTROL",
    "RANDOM_MUTATION_CONTROL",
    "COST_ONLY_MUTATION_CONTROL",
    "ACCURACY_ONLY_MUTATION",
    "SUPPORT_COST_TARGETED_MUTATION",
    "HARD_STRESS_MUTATION",
    "MULTI_OBJECTIVE_PARETO_MUTATION",
    "LARGE_STEP_MUTATION",
    "STRUCTURED_GATE_MUTATION",
    "NOVELTY_DIVERSITY_MUTATION",
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "SPIKE_SHUFFLE_CONTROL",
    "THRESHOLD_ABLATION",
    "REWIRE_ABLATION",
]

RUST_ARMS = [
    "D58_REPLAY_REFERENCE",
    "D59_BEST_REPLAY",
    "MUTATION_DISABLED_CONTROL",
    "RANDOM_MUTATION_CONTROL",
    "COST_ONLY_MUTATION_CONTROL",
    "ACCURACY_ONLY_MUTATION",
    "SUPPORT_COST_TARGETED_MUTATION",
    "HARD_STRESS_MUTATION",
    "MULTI_OBJECTIVE_PARETO_MUTATION",
    "LARGE_STEP_MUTATION",
    "STRUCTURED_GATE_MUTATION",
    "NOVELTY_DIVERSITY_MUTATION",
    "THRESHOLD_ABLATION",
    "REWIRE_ABLATION",
]

TRACKS = {"SATURATED_STABILITY", "HARD_NON_SATURATED_LEARNING"}

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
    "track",
    "difficulty_variant",
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
    "hard_budget_violation",
]

ALLOWED_DECISIONS = {
    "rust_sparse_mutation_learning_signal_confirmed",
    "rust_sparse_mutation_path_confirmed_no_learning_signal",
    "d60_hard_task_invalid",
    "rust_sparse_mutation_safety_failure",
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
    tracks = set()
    rust_rows = {arm: 0 for arm in RUST_ARMS}
    fallback_rows = {arm: 0 for arm in RUST_ARMS}
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing {missing}")
            require(row["total_support_used"] >= row["original_support_used"], f"{path} row {idx} invalid support accounting")
            require(row["total_support_used"] >= row["counter_support_used"], f"{path} row {idx} invalid counter support accounting")
            arms.add(row["arm"])
            regimes.add(row["support_regime"])
            tracks.add(row["track"])
            if row["arm"] in rust_rows:
                if row["rust_network_path_invoked"]:
                    rust_rows[row["arm"]] += 1
                if row["python_fallback_used"]:
                    fallback_rows[row["arm"]] += 1
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    require(TRACKS.issubset(tracks), f"{path} missing tracks {TRACKS - tracks}")
    for arm in RUST_ARMS:
        require(rust_rows[arm] > 0, f"{path} {arm} did not invoke Rust")
        require(fallback_rows[arm] == 0, f"{path} {arm} used Python fallback")


def source_guardrails(repo_root):
    source = (repo_root / "scripts/probes/run_d60_rust_sparse_mutation_learning_signal_probe.py").read_text(encoding="utf-8")
    compact = source.replace(" ", "").lower()
    for needle, message in [
        ("hash(", "Python hash forbidden"),
        ("random.random()<", "fake random threshold sampling forbidden"),
        ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
        ("expected_selected", "expected_selected feature forbidden"),
        ("openai", "external API reference forbidden"),
        ("modal", "Modal reference forbidden"),
    ]:
        require(needle not in compact, f"{message} in D60 runner")
    require("Network::propagate_sparse" in source or "run_rust_multi_bridge" in source, "D60 runner does not use Rust sparse bridge")
    require("oracle_upper_bound_for_packs" in source, "D60 runner missing oracle upper-bound logic")
    require("hard_budget_violation" in source, "D60 runner missing hard task budget instrumentation")


def require_track(metrics, track):
    require(track in metrics, f"missing track metrics {track}")
    track_metrics = metrics[track]
    require(set(ARMS).issubset(track_metrics["by_arm_core"].keys()), f"{track} missing core arms")
    for arm in ARMS:
        require(set(REGIMES).issubset(track_metrics["by_arm_and_regime"][arm].keys()), f"{track} {arm} missing regime metrics")
    return track_metrics


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
    upstream = load(out / "d59_upstream_manifest.json")
    difficulty = load(out / "task_difficulty_report.json")
    oracle = load(out / "oracle_upper_bound_report.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    saturated = load(out / "saturated_track_report.json")
    hard = load(out / "hard_learning_track_report.json")
    causality = load(out / "mutation_causality_report.json")
    safety = load(out / "safety_constraint_report.json")
    report = (out / "report.md").read_text(encoding="utf-8")

    require(dataset["truth_hidden_from_controller_inputs"] is True, "truth hidden flag missing")
    require(dataset["controller_only_not_formula_solver"] is True, "controller-only flag missing")
    require(dataset["formula_solver_learning_used"] is False, "formula solver learning flag wrong")
    require(set(dataset["tracks"]) == TRACKS, "dataset tracks mismatch")
    require(upstream["expected_decision"] == "rust_sparse_mutation_path_confirmed_no_gain", "bad D59 expected decision")
    require(upstream["reinterpreted_next"] == "D60_RUST_SPARSE_MUTATION_LEARNING_SIGNAL_PROBE", "bad D60 reinterpretation")
    require(upstream.get("d59_best_controller_loaded") is True, "D59 best controller not loaded")
    require("support_budget_cap_5" in difficulty and "support_budget_cap_6" in difficulty, "requested hard cap variants missing")
    require(oracle.get("selected_variant_name"), "missing selected hard variant")
    require("test_oracle" in oracle and "ood_oracle" in oracle, "missing oracle split reports")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D60 only tests learning signal" in decision["boundary"], "decision boundary missing")
    require("D60 only tests learning signal" in report, "report boundary missing")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require(aggregate["rust_path_invoked"] is True, "Rust path not invoked")
    require(aggregate["fallback_rows"] == 0, "fallback rows nonzero")

    sat_metrics = require_track(aggregate["test_metrics_by_track"], "SATURATED_STABILITY")
    hard_metrics = require_track(aggregate["test_metrics_by_track"], "HARD_NON_SATURATED_LEARNING")
    require(saturated["rust_usage_ok"] is True, "saturated rust usage not clean")
    require(causality["mutation_path_exercised"] is True, "mutation path not exercised")
    require(safety["saturated_stable"] == saturated["stable"], "safety/stability mismatch")
    best = decision.get("best_arm") or hard["best_arm"]
    require(best in hard_metrics["by_arm_core"], "best arm missing hard metrics")
    require(best in safety["hard_false_confidence_by_arm"], "best arm missing safety metrics")

    if decision["decision"] == "rust_sparse_mutation_learning_signal_confirmed":
        require(hard["learning_success"] is True, "learning decision without learning success")
        require(hard["exact_gain"] >= 0.03 or hard["cost_adjusted_gain"] >= 0.02 or hard["support_delta"] <= -0.25, "learning gate not met")
        require(saturated["stable"] is True, "learning decision without saturated stability")
    elif decision["decision"] == "rust_sparse_mutation_path_confirmed_no_learning_signal":
        require(saturated["stable"] is True, "no-learning decision without saturated stability")
        require(hard["learning_success"] is False, "no-learning decision but hard learning success true")
    elif decision["decision"] == "d60_hard_task_invalid":
        require(not oracle.get("selected_variant", {}).get("valid_oracle", False), "hard-task-invalid but selected oracle valid")
    elif decision["decision"] == "rust_sparse_mutation_safety_failure":
        require(saturated["stable"] is False or safety["hard_false_confidence_by_arm"][best] > 0.01 or aggregate["failed_jobs"], "safety failure lacks evidence")

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "verdict": decision.get("verdict"),
                "next": decision.get("next"),
                "best_arm": best,
                "hard_variant": oracle.get("selected_variant_name"),
                "saturated_stable": saturated["stable"],
                "learning_success": hard["learning_success"],
                "exact_gain": hard["exact_gain"],
                "cost_adjusted_gain": hard["cost_adjusted_gain"],
                "support_delta": hard["support_delta"],
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
