#!/usr/bin/env python3
"""Checker for D60S Rust sparse mutation safety/no-forgetting repair."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "d60_upstream_manifest.json",
    "fitness_definition_report.json",
    "multi_environment_eval_report.json",
    "saturated_stability_report.json",
    "hard_learning_report.json",
    "mixed_eval_report.json",
    "policy_gate_report.json",
    "no_forgetting_report.json",
    "pareto_frontier_report.json",
    "mutation_causality_report.json",
    "support_cost_frontier_report.json",
    "safety_constraint_report.json",
    "rust_invocation_report.json",
    "ablation_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
    "trained_policy_manifest.json",
]

TRACKS = {
    "SATURATED_STABILITY",
    "HARD_CAP8_LEARNING",
    "MIXED_EVAL",
}

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

RUST_ARMS = [
    "D59_REFERENCE",
    "D60_HARD_BEST_REPLAY",
    "SINGLE_POLICY_MULTI_ENV_FITNESS",
    "LEXICOGRAPHIC_SAFETY_FIRST_FITNESS",
    "PARETO_MULTI_ENV_MUTATION",
    "STABILITY_REGULARIZED_MUTATION",
    "COST_ONLY_MUTATION_CONTROL",
    "ACCURACY_ONLY_MUTATION_CONTROL",
    "RANDOM_MUTATION_CONTROL",
    "MUTATION_DISABLED_CONTROL",
    "THRESHOLD_ABLATION",
    "REWIRE_ABLATION",
]

GATED_ARMS = [
    "DUAL_POLICY_GATED_CONTROLLER",
    "CONTEXT_GATED_POLICY_ENSEMBLE",
]

CONTROL_ARMS = [
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "SPIKE_SHUFFLE_CONTROL",
]

TRAINED_ARMS = [
    "SINGLE_POLICY_MULTI_ENV_FITNESS",
    "LEXICOGRAPHIC_SAFETY_FIRST_FITNESS",
    "PARETO_MULTI_ENV_MUTATION",
    "STABILITY_REGULARIZED_MUTATION",
]

ARMS = RUST_ARMS + GATED_ARMS + CONTROL_ARMS

ROW_FIELDS = [
    "row_id",
    "seed",
    "split",
    "track",
    "mixed_source_track",
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
    "context_gate_used",
    "context_gate_basis",
    "gate_selected_policy",
]

ALLOWED_DECISIONS = {
    "rust_sparse_mutation_safety_fitness_repaired",
    "gated_policy_required_for_no_forgetting",
    "safety_fitness_repair_not_confirmed",
    "learning_signal_lost_under_safety_fitness",
}


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_guardrails(repo_root):
    source = (repo_root / "scripts/probes/run_d60s_rust_sparse_mutation_safety_fitness_repair.py").read_text(encoding="utf-8")
    compact = source.replace(" ", "").replace("\n", "").lower()
    for needle, message in [
        ("hash(", "Python hash forbidden"),
        ("random.random()<", "fake random threshold sampling forbidden"),
        ("fixed_synthetic_accuracy", "fixed synthetic accuracy dict forbidden"),
        ("expected_selected", "expected_selected feature forbidden"),
        ("openai", "external API reference forbidden"),
        ("modal", "Modal reference forbidden"),
    ]:
        require(needle not in compact, f"{message} in D60S runner")
    require("run_rust_multi_bridge" in source, "D60S runner does not invoke canonical Rust bridge")
    require("d58_hard_replay_exact" in source, "D60S runner missing D58 hard baseline accounting")
    require("context_gate_used" in source, "D60S runner missing context-gate instrumentation")
    require("partial_mutation_" in source, "D60S runner missing partial mutation writeouts")


def inspect_jsonl(path):
    rows = 0
    arms = set()
    tracks = set()
    regimes = set()
    rust_rows = {arm: 0 for arm in RUST_ARMS + GATED_ARMS}
    fallback_rows = {arm: 0 for arm in RUST_ARMS + GATED_ARMS}
    gated_rows = {arm: 0 for arm in GATED_ARMS}
    gated_policies = set()
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing {missing}")
            require(row["total_support_used"] >= row["original_support_used"], f"{path} row {idx} invalid support accounting")
            require(row["total_support_used"] >= row["counter_support_used"], f"{path} row {idx} invalid counter-support accounting")
            arms.add(row["arm"])
            tracks.add(row["track"])
            regimes.add(row["support_regime"])
            if row["arm"] in rust_rows:
                if row["rust_network_path_invoked"]:
                    rust_rows[row["arm"]] += 1
                if row["python_fallback_used"]:
                    fallback_rows[row["arm"]] += 1
            if row["arm"] in GATED_ARMS:
                require(row["context_gate_used"] is True, f"{path} gated row {idx} did not mark gate usage")
                require(row["gate_selected_policy"] in {"D59_REFERENCE", "D60_HARD_BEST_REPLAY"}, f"{path} gated row {idx} bad policy")
                gated_rows[row["arm"]] += 1
                gated_policies.add(row["gate_selected_policy"])
    require(rows > 0, f"{path} empty")
    require(set(ARMS).issubset(arms), f"{path} missing arms {set(ARMS) - arms}")
    require(TRACKS.issubset(tracks), f"{path} missing tracks {TRACKS - tracks}")
    require(set(REGIMES).issubset(regimes), f"{path} missing regimes {set(REGIMES) - regimes}")
    for arm in RUST_ARMS + GATED_ARMS:
        require(rust_rows[arm] > 0, f"{path} {arm} did not invoke Rust")
        require(fallback_rows[arm] == 0, f"{path} {arm} used Python fallback")
    for arm in GATED_ARMS:
        require(gated_rows[arm] > 0, f"{path} {arm} has no gated rows")
    require({"D59_REFERENCE", "D60_HARD_BEST_REPLAY"}.issubset(gated_policies), f"{path} gated policy did not exercise both policies")


def require_track(metrics, track):
    require(track in metrics, f"missing track metrics {track}")
    item = metrics[track]
    require(set(ARMS).issubset(item["by_arm_core"].keys()), f"{track} missing core arms")
    for arm in ARMS:
        require(set(REGIMES).issubset(item["by_arm_and_regime"][arm].keys()), f"{track} {arm} missing regime metrics")
    return item


def arm_exact(track_metrics, arm):
    return float(track_metrics["by_arm_core"][arm]["exact_joint_accuracy"])


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
    upstream = load(out / "d60_upstream_manifest.json")
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    summary = load(out / "summary.json")
    gate = load(out / "policy_gate_report.json")
    safety = load(out / "safety_constraint_report.json")
    causality = load(out / "mutation_causality_report.json")
    support_frontier = load(out / "support_cost_frontier_report.json")
    report = (out / "report.md").read_text(encoding="utf-8")

    require(dataset["truth_hidden_from_controller_inputs"] is True, "truth hidden flag missing")
    require(dataset["controller_only_not_formula_solver"] is True, "controller-only flag missing")
    require(dataset["formula_solver_learning_used"] is False, "formula solver learning flag wrong")
    require(set(dataset["tracks"]) == TRACKS, "dataset tracks mismatch")
    require(dataset["hard_variant"]["name"] == "support_budget_cap_8", "hard variant mismatch")
    require(upstream["expected_decision"] == "rust_sparse_mutation_safety_failure", "bad D60 expected decision")
    require(upstream["expected_next"] == "D60S_SAFETY_FITNESS_REPAIR", "bad D60 expected next")
    require(upstream["decision_present"] is True and upstream["summary_present"] is True, "D60 artifacts missing")
    require(decision["decision"] in ALLOWED_DECISIONS, f"unexpected decision {decision['decision']}")
    require("D60S only tests safety/no-forgetting" in decision["boundary"], "decision boundary missing")
    require("D60S only tests safety/no-forgetting" in report, "report boundary missing")
    require(aggregate["rust_path_invoked"] is True, "Rust path not invoked")
    require(aggregate["fallback_rows"] == 0, "fallback rows nonzero")
    require(aggregate["d60_summary"]["d58_hard_replay_exact"] < aggregate["d60_summary"]["d60_hard_best_exact"], "bad D58 hard baseline")

    test_tracks = aggregate["test_metrics_by_track"]
    sat = require_track(test_tracks, "SATURATED_STABILITY")
    hard = require_track(test_tracks, "HARD_CAP8_LEARNING")
    mixed = require_track(test_tracks, "MIXED_EVAL")

    for arm in TRAINED_ARMS:
        require(arm in aggregate["mutation_reports"], f"missing mutation report for {arm}")
        require(aggregate["mutation_reports"][arm]["generations"] >= 1, f"{arm} did not run generations")
    require(causality["mutation_path_exercised"] is True, "mutation path not exercised")
    require(set(gate["gated_arms"]) == set(GATED_ARMS), "gated arm list mismatch")
    require("D59_REFERENCE" in str(gate) and "D60_HARD_BEST_REPLAY" in str(gate), "gate report missing selected policies")
    require(TRACKS.issubset(set(support_frontier.keys())), "support frontier missing tracks")
    for track in TRACKS:
        require(set(ARMS).issubset(set(support_frontier[track].keys())), f"support frontier {track} missing arms")

    best = decision.get("best_arm")
    require(best in TRAINED_ARMS + GATED_ARMS, "best arm is not a repair candidate")
    saturated_floor = float(aggregate["saturated_floor"])
    best_sat = arm_exact(sat, best)
    best_hard = arm_exact(hard, best)
    best_mixed = arm_exact(mixed, best)
    hard_gain = float(aggregate["arm_summaries"][best]["hard_gain_vs_D58"])
    false_conf = max(
        float(aggregate["arm_summaries"][best]["SATURATED_STABILITY"]["false_confidence"]),
        float(aggregate["arm_summaries"][best]["HARD_CAP8_LEARNING"]["false_confidence"]),
        float(aggregate["arm_summaries"][best]["MIXED_EVAL"]["false_confidence"]),
    )
    abstain_floor = min(
        float(aggregate["arm_summaries"][best]["SATURATED_STABILITY"]["abstain"]),
        float(aggregate["arm_summaries"][best]["HARD_CAP8_LEARNING"]["abstain"]),
        float(aggregate["arm_summaries"][best]["MIXED_EVAL"]["abstain"]),
    )
    if decision["decision"] in {"rust_sparse_mutation_safety_fitness_repaired", "gated_policy_required_for_no_forgetting"}:
        require(best_sat >= saturated_floor, "positive decision without saturated stability")
        require(best_hard >= 0.99 and best_mixed >= 0.99, "positive decision without hard/mixed accuracy")
        require(hard_gain >= 0.30, "positive decision without hard gain over D58")
        require(false_conf <= 0.01, "positive decision with false confidence")
        require(abstain_floor >= 0.99, "positive decision without indistinguishable abstain")
        require(best_mixed > arm_exact(mixed, "RANDOM_POLICY_CONTROL"), "random control not worse")
        require(best_mixed > arm_exact(mixed, "GREEDY_DECIDE_CONTROL"), "greedy control not worse")
        require(best_mixed > arm_exact(mixed, "SPIKE_SHUFFLE_CONTROL"), "shuffle control not worse")
        if decision["decision"] == "gated_policy_required_for_no_forgetting":
            require(best in GATED_ARMS, "gated decision best arm not gated")
        else:
            require(best in TRAINED_ARMS, "single-policy decision best arm not trained")
    elif decision["decision"] == "safety_fitness_repair_not_confirmed":
        require(best_hard >= 0.99 or best_sat < saturated_floor or false_conf > 0.01, "repair-not-confirmed lacks evidence")
    elif decision["decision"] == "learning_signal_lost_under_safety_fitness":
        require(hard_gain < 0.30, "learning-lost decision but hard gain present")

    print(
        json.dumps(
            {
                "status": "ok",
                "decision": decision["decision"],
                "best_arm": best,
                "best_saturated_exact": best_sat,
                "best_hard_exact": best_hard,
                "best_mixed_exact": best_mixed,
                "hard_gain_vs_D58": hard_gain,
                "false_confidence": false_conf,
                "abstain_floor": abstain_floor,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
