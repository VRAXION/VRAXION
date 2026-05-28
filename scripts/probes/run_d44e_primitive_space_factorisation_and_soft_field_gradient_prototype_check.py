#!/usr/bin/env python3
"""Artifact checker for D44E primitive-space factorisation prototype."""
import argparse
import json
from pathlib import Path

REQUIRED = [
    "d44d2_upstream_manifest.json",
    "dataset_manifest.json",
    "primitive_space_report.json",
    "soft_field_metrics_report.json",
    "raw_candidate_id_argmax_report.json",
    "canonical_unordered_pair_factorisation_report.json",
    "family_group_score_factorisation_report.json",
    "soft_field_clustering_factorisation_report.json",
    "topk_soft_prefilter_staged_support_report.json",
    "entropy_margin_support_policy_report.json",
    "collision_aware_group_vote_report.json",
    "learned_grouping_lightweight_report.json",
    "equivalence_class_report.json",
    "family_vs_candidate_accuracy_report.json",
    "support_count_report.json",
    "support4_policy_report.json",
    "candidate_order_sensitivity_report.json",
    "true_candidate_rank_report.json",
    "ambiguity_prediction_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "row_outputs_train.jsonl",
    "row_outputs_test.jsonl",
    "row_outputs_ood.jsonl",
]
ROW_FIELDS = {"primitive_space", "candidate_id", "family_id", "equivalence_class_id", "support_used", "policy", "truth_family", "pred_family"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")
    aggregate = json.loads((out / "aggregate_metrics.json").read_text())
    required_key = "ALL28_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9::FAMILY_GROUP_SCORE_FACTORISATION"
    if required_key not in aggregate:
        raise SystemExit(f"missing aggregate key {required_key}")
    decision = json.loads((out / "decision.json").read_text())
    if "TRUE_LABEL_ECHO" in json.dumps(aggregate):
        raise SystemExit("label-echo arm is not allowed in D44E fair reports")
    for row_file in ("row_outputs_train.jsonl", "row_outputs_test.jsonl", "row_outputs_ood.jsonl"):
        with (out / row_file).open() as handle:
            first = handle.readline()
        if not first:
            raise SystemExit(f"{row_file} is empty")
        row = json.loads(first)
        missing_fields = sorted(ROW_FIELDS - set(row))
        if missing_fields:
            raise SystemExit(f"{row_file} missing fields: {missing_fields}")
    primitive = json.loads((out / "primitive_space_report.json").read_text())
    if "ALL28_UNORDERED_NONCENTER_CELL_PAIRS_ADD_MOD9" not in primitive:
        raise SystemExit("all28 primitive-space report missing")
    print(json.dumps({"status": "ok", "decision": decision["decision"], "next": decision["next"]}, indent=2))


if __name__ == "__main__":
    main()
