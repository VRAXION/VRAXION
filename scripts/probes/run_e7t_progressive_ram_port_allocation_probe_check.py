#!/usr/bin/env python3
"""Checker for E7T progressive RAM port allocation probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e7t_progressive_ram_port_allocation_probe.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "port_map_report.json",
    "slot_plateau_report.json",
    "shared_write_report.json",
    "flow_grid_frames.json",
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
    "untyped_flow_baseline",
    "output_write_map_only",
    "input_read_map_only",
    "input_plus_output_port_map",
    "learned_sparse_mask_reference",
    "progressive_write_slot_allocation",
    "progressive_read_write_slot_allocation",
    "learned_port_map_then_freeze",
    "shared_write_control",
    "integrator_shared_write_control",
    "oracle_port_map_reference",
    "dense_graph_danger_control",
)
SKILLS = ("compare", "mod_add", "parity", "threshold", "counterfactual_flip", "verify")
VALID_DECISIONS = (
    "e7t_output_port_map_positive",
    "e7t_input_output_port_map_positive",
    "e7t_progressive_write_slot_allocation_positive",
    "e7t_progressive_read_write_slot_allocation_positive",
    "e7t_learned_frozen_port_map_positive",
    "e7t_direct_shared_write_collision_detected",
    "e7t_integrator_shared_write_positive",
    "e7t_sparse_mask_contract_still_preferred",
    "e7t_graph_soup_regression_detected",
    "e7t_ram_port_allocation_no_advantage",
)
ALLOWED_TRAINING_CALL_FUNCTIONS = {"seed_worker", "train_masked_library", "train_progressive_write", "train_progressive_read_write"}


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
        return {"schema_version": "e7t_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    manifest = load_json(out / "backend_manifest.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    system_results = load_json(out / "system_results.json")
    port_map = load_json(out / "port_map_report.json")
    plateau = load_json(out / "slot_plateau_report.json")
    shared = load_json(out / "shared_write_report.json")
    mutation = load_json(out / "mutation_history.json")
    flow_grid = load_json(out / "flow_grid_frames.json")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append(f"INVALID_DECISION:{decision.get('decision')}")
    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for artifact, item in replay.get("hash_comparisons", {}).items():
        if item.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{artifact}")
    if summary.get("deterministic_replay_passed") is not True or decision.get("deterministic_replay_passed") is not True:
        failures.append("REPLAY_FLAG_NOT_PROPAGATED")

    if manifest.get("semantic_lane_labels_as_model_input") is not False:
        failures.append("SEMANTIC_LABELS_AS_MODEL_INPUT")
    if manifest.get("training_performed") is not True:
        failures.append("TRAINING_NOT_REPORTED")
    if manifest.get("flow_dim") != 40:
        failures.append(f"UNEXPECTED_FLOW_DIM:{manifest.get('flow_dim')}")

    present_systems = {row.get("system") for row in system_results.get("rows", [])}
    for system in SYSTEMS:
        if system not in present_systems:
            failures.append(f"MISSING_SYSTEM:{system}")
        if aggregate.get("systems", {}).get(system, {}).get("seed_count", 0) <= 0:
            failures.append(f"NO_ROWS_FOR_SYSTEM:{system}")
        mean = aggregate.get("systems", {}).get(system, {}).get("mean", {})
        for metric in ("eval_mean_composition_usefulness", "eval_mean_answer_accuracy"):
            if metric not in mean:
                failures.append(f"MISSING_AGGREGATE_METRIC:{system}:{metric}")

    for row in system_results.get("rows", []):
        evals = row.get("evals", {})
        for split in ("heldout", "ood", "counterfactual", "adversarial"):
            split_row = evals.get(split, {})
            if not split_row.get("row_level_samples"):
                failures.append(f"MISSING_ROW_LEVEL_SAMPLE:{row.get('system')}:{split}")
            for metric in ("answer_accuracy", "route_accuracy", "composition_usefulness"):
                if metric not in split_row:
                    failures.append(f"MISSING_SPLIT_METRIC:{row.get('system')}:{split}:{metric}")
        for metric in ("eval_mean_write_spread", "eval_mean_changed_cell_count", "eval_mean_delta_magnitude", "eval_mean_read_cell_count", "eval_mean_write_cell_count", "ram_collision_rate"):
            if metric not in row:
                failures.append(f"MISSING_PORT_RUNTIME_METRIC:{row.get('system')}:{metric}")

    for system in (
        "output_write_map_only",
        "input_read_map_only",
        "input_plus_output_port_map",
        "learned_sparse_mask_reference",
        "learned_port_map_then_freeze",
        "progressive_write_slot_allocation",
        "progressive_read_write_slot_allocation",
        "shared_write_control",
        "integrator_shared_write_control",
    ):
        for skill in SKILLS:
            if not any(row.get("system") == system or str(row.get("system", "")).startswith(system + "_") for row in port_map.get("rows", []) if row.get("skill") == skill):
                failures.append(f"MISSING_PORT_MAP:{system}:{skill}")

    for row in port_map.get("rows", []):
        contract = row.get("contract", {})
        if contract.get("semantic_label_control") is True:
            failures.append(f"SEMANTIC_PORT_MAP:{row.get('system')}:{row.get('skill')}")
        if row.get("system") not in {"untyped_flow_baseline", "input_read_map_only"} and contract.get("enforce") is False:
            failures.append(f"UNENFORCED_PRIMARY_PORT_MAP:{row.get('system')}:{row.get('skill')}")

    plateau_systems = {row.get("system") for row in plateau.get("rows", [])}
    for system in ("progressive_write_slot_allocation", "progressive_read_write_slot_allocation"):
        if system not in plateau_systems:
            failures.append(f"MISSING_PLATEAU:{system}")
    for row in plateau.get("rows", []):
        curve = row.get("curve", [])
        if len(curve) < 2:
            failures.append(f"PLATEAU_CURVE_TOO_SHORT:{row.get('system')}:{row.get('seed')}")
        if "chosen_budget" not in row.get("plateau", {}):
            failures.append(f"MISSING_CHOSEN_BUDGET:{row.get('system')}:{row.get('seed')}")

    if not shared.get("rows"):
        failures.append("MISSING_SHARED_WRITE_ROWS")
    for row in shared.get("rows", []):
        if "shared_write_collision_rate" not in row or "integrator_collision_rate" not in row:
            failures.append(f"MISSING_SHARED_COLLISION_METRIC:{row.get('seed')}")

    if not mutation.get("rows"):
        failures.append("MISSING_MUTATION_HISTORY")
    learned_mean = aggregate.get("systems", {}).get("learned_sparse_mask_reference", {}).get("mean", {})
    if learned_mean.get("mutation_attempts", 0) <= 0 or learned_mean.get("rejected_mutations", 0) <= 0:
        failures.append("LEARNED_MASK_MUTATION_COUNTS_MISSING")

    if flow_grid.get("schema_version") != "e7t_flow_grid_frames_v1" or not flow_grid.get("frames"):
        failures.append("MISSING_FLOW_GRID_COMPAT_FRAMES")

    events = event_names(out / "progress.jsonl")
    for event in ("run_start", "primary_artifacts_written", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"MISSING_PROGRESS_EVENT:{event}")

    report = (out / "report.md").read_text(encoding="utf-8").lower()
    for banned_label in ("truth slot", "memory slot", "confidence slot", "result slot"):
        if banned_label in report:
            failures.append(f"SEMANTIC_LANE_LABEL:{banned_label}")
    for banned_claim in ("agi", "consciousness", "model-scale", "raw-language"):
        if banned_claim in report and "does not make" not in report:
            failures.append(f"BANNED_CLAIM:{banned_claim}")

    failures.extend(ast_policy_failures())
    result = {"schema_version": "e7t_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
