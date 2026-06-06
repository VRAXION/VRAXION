#!/usr/bin/env python3
"""Checker for E7Q router-pocket joint binding probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e7q_router_pocket_joint_binding_probe.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "baseline_pocket_training_report.json",
    "router_training_report.json",
    "joint_binding_training_report.json",
    "flow_contract_report.json",
    "reuse_after_binding_report.json",
    "private_protocol_leakage_report.json",
    "composition_report.json",
    "error_attribution_report.json",
    "system_results.json",
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
    "frozen_router_trained_pocket",
    "trained_router_frozen_pocket",
    "trained_router_trained_pocket",
    "trained_router_trained_pocket_slot_guard",
    "full_end_to_end_training_control",
    "random_router_control",
    "oracle_route_reference",
)
REQUIRED_TRAINING_SYSTEMS = (
    "frozen_router_trained_pocket",
    "trained_router_frozen_pocket",
    "trained_router_trained_pocket",
    "trained_router_trained_pocket_slot_guard",
    "full_end_to_end_training_control",
)
SKILLS = ("compare", "mod_add", "parity", "threshold", "counterfactual_flip", "verify")
VALID_DECISIONS = (
    "e7q_router_pocket_joint_binding_positive",
    "e7q_slot_guard_joint_binding_positive",
    "e7q_slot_guard_improves_but_not_solved",
    "e7q_router_discovery_not_interface_fix",
    "e7q_private_router_pocket_protocol_detected",
    "e7q_full_end_to_end_control_preferred",
    "e7q_joint_binding_not_yet_viable",
    "e7q_artifact_or_task_too_easy",
)
ALLOWED_OPTIMIZER_FUNCTIONS = {"train_router_only", "train_joint_binding"}
ALLOWED_EXTERNAL_GRADIENT_CALLS = {"train_frozen_router_slot_pockets", "seed_worker"}


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
        if isinstance(node, ast.Attribute) and node.attr == "train_monolithic":
            fn = enclosing_function(node, mapper.parents)
            if fn not in ALLOWED_EXTERNAL_GRADIENT_CALLS:
                failures.append(f"MONOLITHIC_CONTROL_OUTSIDE_ALLOWED_FUNCTION:{fn}")
        if isinstance(node, ast.Attribute) and node.attr == "train_context_pocket":
            fn = enclosing_function(node, mapper.parents)
            if fn not in ALLOWED_EXTERNAL_GRADIENT_CALLS:
                failures.append(f"LOCAL_POCKET_TRAINING_OUTSIDE_ALLOWED_FUNCTION:{fn}")
    return failures


def check(out: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (out / artifact).exists():
            failures.append(f"MISSING_ARTIFACT:{artifact}")
    if failures:
        return {"schema_version": "e7q_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    systems_report = load_json(out / "system_results.json")
    baseline_report = load_json(out / "baseline_pocket_training_report.json")
    router_report = load_json(out / "router_training_report.json")
    joint_report = load_json(out / "joint_binding_training_report.json")
    reuse_report = load_json(out / "reuse_after_binding_report.json")
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
            if row.get("system") != "full_end_to_end_training_control":
                for metric in ("state_preservation_error", "result_slot_corruption_rate", "next_pocket_input_compatibility_error"):
                    if metric not in split_row:
                        failures.append(f"MISSING_FLOW_METRIC:{row.get('system')}:{split}:{metric}")

    baseline_rows = baseline_report.get("rows", [])
    for skill in SKILLS:
        if not any(row.get("skill") == skill for row in baseline_rows):
            failures.append(f"MISSING_BASELINE_POCKET:{skill}")

    router_rows = router_report.get("rows", [])
    if not any(row.get("system") == "trained_router_frozen_pocket" and row.get("skill") == "router" for row in router_rows):
        failures.append("MISSING_ROUTER_ONLY_TRAINING")
    if not any(row.get("system") == "full_end_to_end_training_control" for row in router_rows):
        failures.append("MISSING_FULL_CONTROL_TRAINING")

    joint_rows = joint_report.get("rows", [])
    for system in ("frozen_router_trained_pocket", "trained_router_trained_pocket", "trained_router_trained_pocket_slot_guard"):
        for skill in SKILLS:
            if not any(row.get("system") == system and row.get("skill") == skill for row in joint_rows):
                failures.append(f"MISSING_JOINT_TRAINING_ROW:{system}:{skill}")

    reuse_rows = reuse_report.get("rows", [])
    for system in ("trained_router_trained_pocket", "trained_router_trained_pocket_slot_guard"):
        if not any(row.get("system") == system and "pocket_reuse_after_binding_usefulness" in row for row in reuse_rows):
            failures.append(f"MISSING_REUSE_ROW:{system}")

    if leakage.get("symbolic_proxy_primary") is not False:
        failures.append("SYMBOLIC_PROXY_PRIMARY")
    if leakage.get("hidden_expected_answer_input") is not False:
        failures.append("LEAKAGE_FLAGGED")
    if leakage.get("dense_graph_primary") is not False:
        failures.append("DENSE_GRAPH_PRIMARY")
    if leakage.get("full_end_to_end_is_diagnostic_only") is not True:
        failures.append("FULL_E2E_NOT_MARKED_DIAGNOSTIC")

    events = event_names(out / "progress.jsonl")
    for event in ("run_start", "primary_artifacts_written", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"MISSING_PROGRESS_EVENT:{event}")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for banned in ("agi", "consciousness", "model-scale", "raw-language"):
        if banned in report_text and "does not make" not in report_text:
            failures.append(f"BANNED_CLAIM:{banned}")

    failures.extend(ast_policy_failures())
    result = {"schema_version": "e7q_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
