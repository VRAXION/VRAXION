#!/usr/bin/env python3
"""Checker for E7R numeric pocket masked Flow IO contract probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e7r_numeric_pocket_masked_flow_io_contract_probe.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "baseline_pocket_training_report.json",
    "mask_contract_report.json",
    "lane_shuffle_report.json",
    "state_hygiene_report.json",
    "composition_report.json",
    "error_attribution_report.json",
    "system_results.json",
    "mutation_history.json",
    "leakage_report.json",
    "deterministic_replay.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
)
SYSTEMS = (
    "current_untyped_flow_baseline",
    "semantic_labeled_lane_control",
    "anonymous_fixed_mask_contract",
    "anonymous_shuffled_mask_contract",
    "result_region_only_write_contract",
    "residual_preservation_contract",
    "learned_mask_contract",
    "oracle_mask_reference",
    "full_end_to_end_control",
    "dense_graph_danger_control",
)
MASKED_SYSTEMS = (
    "semantic_labeled_lane_control",
    "anonymous_fixed_mask_contract",
    "anonymous_shuffled_mask_contract",
    "result_region_only_write_contract",
    "residual_preservation_contract",
    "learned_mask_contract",
)
SKILLS = ("compare", "mod_add", "parity", "threshold", "counterfactual_flip", "verify")
VALID_DECISIONS = (
    "e7r_anonymous_masked_flow_contract_positive",
    "e7r_result_region_hygiene_positive",
    "e7r_residual_preservation_contract_positive",
    "e7r_learned_sparse_mask_contract_positive",
    "e7r_semantic_label_shortcut_detected",
    "e7r_local_io_contract_insufficient",
    "e7r_graph_soup_regression_detected",
    "e7r_numeric_pocket_interface_still_broken",
)
ALLOWED_OPTIMIZER_FUNCTIONS = {"train_masked_context_pocket"}
ALLOWED_EXTERNAL_GRADIENT_CALLS = {"seed_worker"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def event_names(path: Path) -> set[str]:
    events = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.add(json.loads(line)["event"])
    return events


class ParentMap(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: dict[ast.AST, ast.AST] = {}

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
            self.visit(child)


def enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, ast.FunctionDef):
            return cur.name
        cur = parents.get(cur)
    return None


def ast_policy_failures() -> list[str]:
    failures: list[str] = []
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    mapper = ParentMap()
    mapper.visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"Adam", "AdamW", "SGD"}:
            fn = enclosing_function(node, mapper.parents)
            if fn not in ALLOWED_OPTIMIZER_FUNCTIONS:
                failures.append(f"OPTIMIZER_OUTSIDE_ALLOWED_FUNCTION:{fn}:{node.attr}")
        if isinstance(node, ast.Attribute) and node.attr == "backward":
            fn = enclosing_function(node, mapper.parents)
            if fn not in ALLOWED_OPTIMIZER_FUNCTIONS:
                failures.append(f"BACKPROP_OUTSIDE_ALLOWED_FUNCTION:{fn}")
        if isinstance(node, ast.Attribute) and node.attr in {"train_context_pocket", "train_skill_pocket", "train_monolithic"}:
            fn = enclosing_function(node, mapper.parents)
            if fn not in ALLOWED_EXTERNAL_GRADIENT_CALLS:
                failures.append(f"EXTERNAL_GRADIENT_CALL_OUTSIDE_ALLOWED_FUNCTION:{fn}:{node.attr}")
    return failures


def check(out: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (out / artifact).exists():
            failures.append(f"MISSING_ARTIFACT:{artifact}")
    if failures:
        return {"schema_version": "e7r_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    systems_report = load_json(out / "system_results.json")
    baseline_report = load_json(out / "baseline_pocket_training_report.json")
    mask_report = load_json(out / "mask_contract_report.json")
    lane_report = load_json(out / "lane_shuffle_report.json")
    mutation_report = load_json(out / "mutation_history.json")
    leakage = load_json(out / "leakage_report.json")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append(f"INVALID_DECISION:{decision.get('decision')}")
    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for artifact, item in replay.get("hash_comparisons", {}).items():
        if item.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{artifact}")
    if summary.get("deterministic_replay_passed") is not True or decision.get("deterministic_replay_passed") is not True:
        failures.append("REPLAY_FLAG_NOT_PROPAGATED")

    present_systems = {row.get("system") for row in systems_report.get("rows", [])}
    for system in SYSTEMS:
        if system not in present_systems:
            failures.append(f"MISSING_SYSTEM:{system}")
        if aggregate.get("systems", {}).get(system, {}).get("seed_count", 0) <= 0:
            failures.append(f"NO_ROWS_FOR_SYSTEM:{system}")

    for row in systems_report.get("rows", []):
        evals = row.get("evals", {})
        for split in ("heldout", "ood", "counterfactual", "adversarial"):
            split_row = evals.get(split, {})
            if not split_row.get("row_level_samples"):
                failures.append(f"MISSING_ROW_LEVEL_SAMPLE:{row.get('system')}:{split}")
            for metric in ("answer_accuracy", "route_accuracy", "composition_usefulness"):
                if metric not in split_row:
                    failures.append(f"MISSING_SPLIT_METRIC:{row.get('system')}:{split}:{metric}")
            if row.get("system") not in {"full_end_to_end_control", "dense_graph_danger_control"}:
                for metric in ("write_mask_violation_rate", "preserve_mask_corruption_rate", "result_region_corruption_rate", "next_pocket_input_compatibility_error"):
                    if metric not in split_row:
                        failures.append(f"MISSING_MASK_METRIC:{row.get('system')}:{split}:{metric}")

    baseline_rows = baseline_report.get("rows", [])
    for skill in SKILLS:
        if not any(row.get("skill") == skill for row in baseline_rows):
            failures.append(f"MISSING_BASELINE_POCKET:{skill}")

    mask_rows = mask_report.get("rows", [])
    for system in MASKED_SYSTEMS:
        for skill in SKILLS:
            if not any(row.get("system") == system and row.get("skill") == skill for row in mask_rows):
                failures.append(f"MISSING_MASK_CONTRACT:{system}:{skill}")
    for row in mask_rows:
        contract = row.get("contract", {})
        if row.get("system") != "semantic_labeled_lane_control" and contract.get("semantic_label_control") is True:
            failures.append(f"SEMANTIC_LABEL_FLAG_IN_ANONYMOUS:{row.get('system')}:{row.get('skill')}")
        if row.get("system") == "anonymous_shuffled_mask_contract" and contract.get("permuted") is not True:
            failures.append(f"SHUFFLED_CONTRACT_NOT_PERMUTED:{row.get('skill')}")
    if mask_report.get("random_mask_control_underperformed") is not True:
        failures.append("RANDOM_MASK_CONTROL_NOT_REPORTED_UNDERPERFORMED")

    if not lane_report.get("rows"):
        failures.append("MISSING_LANE_SHUFFLE_ROWS")
    for row in lane_report.get("rows", []):
        if "lane_shuffle_robustness" not in row:
            failures.append(f"MISSING_LANE_SHUFFLE_ROBUSTNESS:{row.get('seed')}")

    if not mutation_report.get("rows"):
        failures.append("MISSING_LEARNED_MASK_MUTATION_HISTORY")
    learned = aggregate.get("systems", {}).get("learned_mask_contract", {}).get("mean", {})
    if learned.get("mutation_attempts", 0) <= 0:
        failures.append("NO_LEARNED_MASK_MUTATION_ATTEMPTS")
    if learned.get("rejected_mutations", 0) <= 0:
        failures.append("NO_LEARNED_MASK_REJECTIONS")

    if leakage.get("semantic_lane_labels_as_model_input") is not False:
        failures.append("SEMANTIC_LABELS_AS_MODEL_INPUT")
    if leakage.get("hidden_expected_answer_input") is not False:
        failures.append("HIDDEN_ANSWER_LEAKAGE")
    if leakage.get("route_label_leakage") is not False:
        failures.append("ROUTE_LABEL_LEAKAGE")
    if leakage.get("pocket_id_answer_leakage") is not False:
        failures.append("POCKET_ID_ANSWER_LEAKAGE")
    if leakage.get("dense_graph_primary") is not False:
        failures.append("DENSE_GRAPH_PRIMARY")

    events = event_names(out / "progress.jsonl")
    for event in ("run_start", "primary_artifacts_written", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"MISSING_PROGRESS_EVENT:{event}")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for banned in ("agi", "consciousness", "model-scale", "raw-language"):
        if banned in report_text and "does not make" not in report_text:
            failures.append(f"BANNED_CLAIM:{banned}")

    failures.extend(ast_policy_failures())
    result = {"schema_version": "e7r_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        summary["checker_failure_count"] = len(failures)
        summary["checker_failures"] = failures
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
