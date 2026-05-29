#!/usr/bin/env python3
"""Artifact checker for D46X multi-view ECF control barrage."""

import argparse
import json
from pathlib import Path

REQUIRED = [
    "queue.json",
    "progress.jsonl",
    "compute_probe.json",
    "dataset_manifest.json",
    "policy_comparison_report.json",
    "regime_by_policy_report.json",
    "primitive_space_by_policy_report.json",
    "support_cost_frontier_report.json",
    "view_diagnostics_report.json",
    "per_view_ablation_delta_report.json",
    "view_redundancy_correlation_matrix.json",
    "view_synergy_report.json",
    "stress_test_report.json",
    "control_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]

ARMS = [
    "SCALAR_ARGMAX_ONLY",
    "VECTOR_FIELD_ONLY",
    "ENTROPY_SUPPORT_POLICY",
    "MARGIN_SUPPORT_POLICY",
    "COLLISION_SUPPORT_POLICY",
    "SUPPORT_INDEPENDENCE_DEDUP_POLICY",
    "COUNTERFACTUAL_TOP1_TOP2_POLICY",
    "SCALAR+VECTOR",
    "VECTOR+ENTROPY+MARGIN",
    "VECTOR+COLLISION+INDEPENDENCE",
    "FULL_MULTI_VIEW_ECF_POLICY",
    "RANDOM_EXTRA_SUPPORT_CONTROL",
    "BAD_VIEW_CONTROL",
    "SHUFFLED_VECTOR_FIELD_CONTROL",
    "NO_COUNTERFACTUAL_CONTROL",
]

ROW_FIELDS = [
    "row_id",
    "seed",
    "split",
    "policy",
    "primitive_space",
    "support_regime",
    "truth_family",
    "pred_family",
    "truth_equivalence",
    "pred_equivalence",
    "pred_candidate",
    "candidate_correct",
    "family_correct",
    "equivalence_correct",
    "correct",
    "support_budget_cap",
    "original_support_used",
    "counter_support_used",
    "total_support_used",
    "support_cluster_count",
    "dominant_cluster_fraction",
    "duplicate_support_score",
    "collision_count",
    "leave_one_out_flip_count",
    "entropy",
    "top1_top2_margin",
    "scalar_pred_family",
    "vector_pred_family",
    "scalar_correct",
    "vector_correct",
    "counter_support_requested",
    "counter_support_target",
    "counter_support_resolved",
    "correlated_support_detected",
    "adversarial_support_detected",
    "view_signals",
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
    budgets = set()
    spaces = set()
    rows = 0
    with Path(path).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            rows += 1
            missing = [field for field in ROW_FIELDS if field not in row]
            require(not missing, f"{path} row {idx} missing {missing}")
            require(isinstance(row["view_signals"], dict), f"{path} row {idx} view_signals not dict")
            policies.add(row["policy"])
            regimes.add(row["support_regime"])
            budgets.add(row["support_budget_cap"])
            spaces.add(row["primitive_space"])
    require(rows > 0, f"{path} empty")
    require("FULL_MULTI_VIEW_ECF_POLICY" in policies, f"{path} missing full multiview rows")
    require("SCALAR_ARGMAX_ONLY" in policies, f"{path} missing scalar rows")
    require("CORRELATED_NOISE_SUPPORT" in regimes, f"{path} missing correlated regime")
    require("ADVERSARIAL_DISTRACTOR_SUPPORT" in regimes, f"{path} missing adversarial regime")
    require(5 in budgets, f"{path} missing support budget 5")
    require("ALL28_UNORDERED" in spaces, f"{path} missing primary space")


def source_guardrails(repo_root):
    runner = (repo_root / "scripts/probes/run_d46x_multi_view_ecf_control_barrage.py").read_text()
    compact = runner.replace(" ", "")
    require("hash(" not in runner, "Python hash forbidden")
    require("random.random()<" not in compact, "fake random threshold sampling forbidden")
    require("expected_selected" not in runner, "expected_selected feature forbidden")
    require("fixed_synthetic_accuracy" not in runner, "fixed synthetic accuracy dict forbidden")
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
    aggregate = load(out / "aggregate_metrics.json")
    decision = load(out / "decision.json")
    diagnostics = load(out / "view_diagnostics_report.json")
    synergy = load(out / "view_synergy_report.json")
    support = load(out / "support_cost_frontier_report.json")
    report = (out / "report.md").read_text()

    require(dataset["candidate_family_equivalence_metrics_separated"], "candidate/family/equivalence metrics not separated")
    require(set(ARMS).issubset(set(dataset["arms"])), "dataset manifest missing arms")
    require(set(map(str, [1, 2, 3, 4, 5])).issubset(set(support["FULL_MULTI_VIEW_ECF_POLICY"].keys())), "support budgets incomplete")
    primary = aggregate["primary_policy_metrics"]
    for arm in ARMS:
        require(arm in primary, f"primary metrics missing {arm}")
        for field in ["candidate_accuracy", "family_accuracy", "equivalence_accuracy", "average_total_support_used"]:
            require(field in primary[arm], f"{arm} missing {field}")
    full = primary["FULL_MULTI_VIEW_ECF_POLICY"]
    scalar = primary["SCALAR_ARGMAX_ONLY"]
    random_control = primary["RANDOM_EXTRA_SUPPORT_CONTROL"]
    bad = primary["BAD_VIEW_CONTROL"]
    shuffled = primary["SHUFFLED_VECTOR_FIELD_CONTROL"]
    no_counter = primary["NO_COUNTERFACTUAL_CONTROL"]
    require("scalar_argmax_failure_rate" in diagnostics, "scalar diagnostic missing")
    require("support_independence_precision_recall" in diagnostics, "support-independence diagnostic missing")
    require("best_single_view" in synergy, "best single view missing")
    require("best_pair_or_triple" in synergy, "best pair/triple missing")
    require(aggregate["failed_jobs"] == [], "failed jobs not empty")
    require("no raw visual Raven" in decision["boundary"], "boundary wording missing in decision")
    require("D46X only tests multi-view ECF control" in report, "boundary wording missing in report")

    allowed = {
        "multi_view_ecf_control_barrage_positive",
        "multi_view_ecf_positive_high_cost",
        "single_view_dominates_multiview_redundant",
        "multiview_ecf_not_robust",
    }
    require(decision["decision"] in allowed, f"unexpected decision {decision['decision']}")
    if decision["decision"] == "multi_view_ecf_control_barrage_positive":
        require(full["clean_accuracy"] >= 0.995, "full clean below positive gate")
        require(full["correlated_accuracy"] >= 0.95, "full correlated below positive gate")
        require(full["adversarial_accuracy"] >= 0.95, "full adversarial below positive gate")
        require(scalar["clean_accuracy"] - full["clean_accuracy"] <= 0.005, "clean regression too high")
        require(full["average_total_support_used"] <= 6.863675, "support cost above D46 robust combined")
        for control in [random_control, bad, shuffled, no_counter]:
            require(full["correlated_accuracy"] > control["correlated_accuracy"], "a control is not worse on correlated support")
        require(synergy["full_beats_all_single_views_and_controls"], "full multiview does not beat single/control arms")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
