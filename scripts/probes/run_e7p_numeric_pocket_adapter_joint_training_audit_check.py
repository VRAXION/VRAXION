#!/usr/bin/env python3
"""Checker for E7P numeric pocket adapter joint training audit."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_e7p_numeric_pocket_adapter_joint_training_audit.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "baseline_pocket_training_report.json",
    "adapter_training_report.json",
    "flow_contract_report.json",
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
    "standalone_pocket_then_fixed_adapter",
    "adapter_only_training",
    "pocket_core_only_training",
    "joint_adapter_plus_pocket_training",
    "joint_adapter_plus_pocket_with_slot_contract",
    "full_end_to_end_training_control",
    "oracle_intermediate_state_reference",
)
TRAINING_SYSTEMS = (
    "adapter_only_training",
    "pocket_core_only_training",
    "joint_adapter_plus_pocket_training",
    "joint_adapter_plus_pocket_with_slot_contract",
    "full_end_to_end_training_control",
)
SKILLS = ("compare", "mod_add", "parity", "threshold", "counterfactual_flip", "verify")
VALID_DECISIONS = (
    "e7p_joint_adapter_pocket_training_positive",
    "e7p_adapter_contract_bottleneck_confirmed",
    "e7p_pocket_core_training_bottleneck_confirmed",
    "e7p_typed_slot_contract_required",
    "e7p_local_pocket_training_insufficient",
    "e7p_numeric_pocket_composition_not_yet_viable",
)
ALLOWED_OPTIMIZER_FUNCTIONS = {"train_context_pocket"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def event_names(progress_path: Path) -> set[str]:
    events = set()
    for line in progress_path.read_text(encoding="utf-8").splitlines():
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
    return failures


def check(out: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (out / artifact).exists():
            failures.append(f"MISSING_ARTIFACT:{artifact}")
    if failures:
        return {"schema_version": "e7p_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    systems_report = load_json(out / "system_results.json")
    adapter_report = load_json(out / "adapter_training_report.json")
    baseline_report = load_json(out / "baseline_pocket_training_report.json")
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
    system_seed_counts = aggregate.get("systems", {})
    for system in SYSTEMS:
        if system_seed_counts.get(system, {}).get("seed_count", 0) <= 0:
            failures.append(f"NO_ROWS_FOR_SYSTEM:{system}")

    for row in systems_report.get("rows", []):
        evals = row.get("evals", {})
        for split in ("heldout", "ood", "counterfactual", "adversarial"):
            split_row = evals.get(split, {})
            if not split_row.get("row_level_samples"):
                failures.append(f"MISSING_ROW_LEVEL_SAMPLE:{row.get('system')}:{split}")
            for metric in ("state_preservation_error", "result_slot_corruption_rate", "next_pocket_input_compatibility_error"):
                if metric not in split_row:
                    failures.append(f"MISSING_FLOW_METRIC:{row.get('system')}:{split}:{metric}")

    training_rows = adapter_report.get("rows", [])
    for system in TRAINING_SYSTEMS:
        for skill in SKILLS:
            if not any(row.get("system") == system and row.get("skill") == skill for row in training_rows):
                failures.append(f"MISSING_TRAINING_ROW:{system}:{skill}")
    baseline_rows = baseline_report.get("rows", [])
    for skill in SKILLS:
        if not any(row.get("skill") == skill for row in baseline_rows):
            failures.append(f"MISSING_BASELINE_POCKET:{skill}")

    if leakage.get("router_frozen") is not True:
        failures.append("ROUTER_NOT_FROZEN")
    if leakage.get("symbolic_proxy_primary") is not False:
        failures.append("SYMBOLIC_PROXY_PRIMARY")
    if leakage.get("hidden_expected_answer_input") is not False:
        failures.append("LEAKAGE_FLAGGED")

    events = event_names(out / "progress.jsonl")
    for event in ("run_start", "primary_artifacts_written", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"MISSING_PROGRESS_EVENT:{event}")

    report_text = (out / "report.md").read_text(encoding="utf-8").lower()
    for banned in ("agi", "consciousness", "model-scale", "raw-language"):
        if banned in report_text and "does not make" not in report_text:
            failures.append(f"BANNED_CLAIM:{banned}")

    failures.extend(ast_policy_failures())
    result = {"schema_version": "e7p_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
