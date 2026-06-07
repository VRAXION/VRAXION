#!/usr/bin/env python3
"""Checker for E7V RAM read-context selection audit."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e7v_ram_read_context_selection_audit.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "read_map_report.json",
    "read_budget_curve_report.json",
    "system_results.json",
    "mutation_history.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
    "deterministic_replay.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "partial_aggregate_snapshot.json",
)
SYSTEMS = (
    "broad_read_next_free_write_baseline",
    "fixed_small_read_control",
    "random_read_map_control",
    "progressive_add_read_cells",
    "prune_from_broad_read",
    "swap_mutation_read_map",
    "grid_neighborhood_read_map",
    "sensitivity_guided_read_map_mutation",
    "learned_sparse_mask_reference",
    "oracle_read_map_reference",
    "dense_graph_danger_control",
)
SKILLS = ("compare", "mod_add", "parity", "threshold", "counterfactual_flip", "verify")
VALID_DECISIONS = (
    "e7v_compact_read_map_positive",
    "e7v_read_context_pruning_positive",
    "e7v_progressive_read_growth_positive",
    "e7v_read_map_swap_mutation_positive",
    "e7v_ram_grid_topology_positive",
    "e7v_broad_context_still_required",
    "e7v_sparse_mask_still_preferred",
    "e7v_graph_soup_regression_detected",
    "e7v_read_context_selection_no_advantage",
)
ALLOWED_TRAINING_CALL_FUNCTIONS = {"seed_worker", "train_read_map_library"}


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
        if isinstance(node, ast.Attribute) and node.attr in {
            "train_masked_context_pocket",
            "train_context_pocket",
            "train_skill_pocket",
            "train_monolithic",
            "mutate_contracts",
        }:
            fn = enclosing_function(node, mapper.parents)
            if fn not in ALLOWED_TRAINING_CALL_FUNCTIONS:
                failures.append(f"TRAINING_CALL_OUTSIDE_ALLOWED_FUNCTION:{fn}:{node.attr}")
        if isinstance(node, ast.Attribute) and node.attr in {"Adam", "AdamW", "SGD", "backward"}:
            fn = enclosing_function(node, mapper.parents)
            failures.append(f"DIRECT_OPTIMIZER_OR_BACKPROP:{fn}:{node.attr}")
    return failures


def check(out: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (out / artifact).exists():
            failures.append(f"MISSING_ARTIFACT:{artifact}")
    if failures:
        return {"schema_version": "e7v_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    manifest = load_json(out / "backend_manifest.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    system_results = load_json(out / "system_results.json")
    read_maps = load_json(out / "read_map_report.json")
    curves = load_json(out / "read_budget_curve_report.json")
    mutation = load_json(out / "mutation_history.json")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append(f"INVALID_DECISION:{decision.get('decision')}")
    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for artifact, row in replay.get("hash_comparisons", {}).items():
        if row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{artifact}")
    if decision.get("deterministic_replay_passed") is not True or summary.get("deterministic_replay_passed") is not True:
        failures.append("REPLAY_FLAG_NOT_PROPAGATED")

    fixed = manifest.get("fixed_write_policy", {})
    if fixed.get("allocator") != "deterministic_next_free":
        failures.append("FIXED_WRITE_ALLOCATOR_NOT_NEXT_FREE")
    if fixed.get("output_cells_per_pocket") != 1:
        failures.append("WRITE_CELL_COUNT_NOT_ONE")
    if fixed.get("direct_shared_write") is not False:
        failures.append("DIRECT_SHARED_WRITE_ENABLED")
    if manifest.get("semantic_lane_labels_as_model_input") is not False:
        failures.append("SEMANTIC_LABELS_AS_MODEL_INPUT")
    if manifest.get("flow_dim") != 40:
        failures.append(f"UNEXPECTED_FLOW_DIM:{manifest.get('flow_dim')}")

    present_systems = {row.get("system") for row in system_results.get("rows", [])}
    for system in SYSTEMS:
        if system not in present_systems:
            failures.append(f"MISSING_SYSTEM:{system}")
        if aggregate.get("systems", {}).get(system, {}).get("seed_count", 0) <= 0:
            failures.append(f"NO_ROWS_FOR_SYSTEM:{system}")

    for row in system_results.get("rows", []):
        evals = row.get("evals", {})
        for split in ("heldout", "ood", "counterfactual", "adversarial"):
            split_row = evals.get(split, {})
            if not split_row.get("row_level_samples"):
                failures.append(f"MISSING_ROW_LEVEL_SAMPLE:{row.get('system')}:{split}")
            for metric in ("answer_accuracy", "route_accuracy", "composition_usefulness"):
                if metric not in split_row:
                    failures.append(f"MISSING_SPLIT_METRIC:{row.get('system')}:{split}:{metric}")
        for metric in (
            "eval_mean_read_cell_count",
            "eval_mean_write_cell_count",
            "eval_mean_read_map_sparsity",
            "eval_mean_grid_locality_score",
            "eval_mean_write_spread",
            "eval_mean_preserve_mask_corruption_rate",
            "eval_mean_write_mask_violation_rate",
            "eval_mean_next_pocket_input_compatibility_error",
        ):
            if metric not in row:
                failures.append(f"MISSING_READ_METRIC:{row.get('system')}:{metric}")

    trained_systems = [system for system in SYSTEMS if system not in {"oracle_read_map_reference", "dense_graph_danger_control"}]
    for system in trained_systems:
        for skill in SKILLS:
            if not any(row.get("system") == system and row.get("skill") == skill for row in read_maps.get("rows", [])):
                failures.append(f"MISSING_READ_MAP:{system}:{skill}")
    for row in read_maps.get("rows", []):
        if not row.get("read_indices"):
            failures.append(f"EMPTY_READ_MAP:{row.get('system')}:{row.get('skill')}")
        if len(row.get("write_indices", [])) != 1:
            failures.append(f"WRITE_MAP_NOT_ONE_CELL:{row.get('system')}:{row.get('skill')}")
        contract = row.get("contract", {})
        if contract.get("semantic_label_control") is True:
            failures.append(f"SEMANTIC_READ_MAP:{row.get('system')}:{row.get('skill')}")
        if contract.get("fixed_next_free_write") is not True:
            failures.append(f"NON_FIXED_WRITE_MAP:{row.get('system')}:{row.get('skill')}")

    curve_systems = {row.get("system") for row in curves.get("rows", [])}
    for system in ("progressive_add_read_cells", "prune_from_broad_read"):
        if system not in curve_systems:
            failures.append(f"MISSING_READ_BUDGET_CURVE:{system}")
    for row in curves.get("rows", []):
        if len(row.get("curve", [])) < 1 or "chosen_read_budget" not in row:
            failures.append(f"BAD_CURVE_ROW:{row.get('seed')}:{row.get('system')}")

    mut_rows = mutation.get("rows", [])
    if not mut_rows:
        failures.append("MISSING_MUTATION_HISTORY")
    for system in ("swap_mutation_read_map", "learned_sparse_mask_reference", "progressive_add_read_cells", "prune_from_broad_read"):
        rows = [row for row in mut_rows if row.get("system") == system]
        if not rows:
            failures.append(f"MISSING_MUTATION_ROWS:{system}")
        if system in {"swap_mutation_read_map", "learned_sparse_mask_reference"}:
            if max((row.get("accepted", 0) for row in rows), default=0) <= 0:
                failures.append(f"NO_ACCEPTED_MUTATION:{system}")
            if max((row.get("rejected", 0) for row in rows), default=0) <= 0:
                failures.append(f"NO_REJECTED_MUTATION:{system}")

    events = event_names(out / "progress.jsonl")
    for event in ("run_start", "primary_artifacts_written", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"MISSING_PROGRESS_EVENT:{event}")

    report = (out / "report.md").read_text(encoding="utf-8").lower()
    for banned_label in ("truth slot", "memory slot", "confidence slot", "answer slot"):
        if banned_label in report:
            failures.append(f"SEMANTIC_LANE_LABEL:{banned_label}")
    for banned_claim in ("agi", "consciousness", "model-scale", "raw-language"):
        if banned_claim in report and "does not make" not in report:
            failures.append(f"BANNED_CLAIM:{banned_claim}")

    failures.extend(ast_policy_failures())
    result = {"schema_version": "e7v_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
