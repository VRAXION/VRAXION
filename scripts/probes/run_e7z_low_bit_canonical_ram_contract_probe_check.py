#!/usr/bin/env python3
"""Checker for E7Z low-bit canonical RAM contract probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e7z_low_bit_canonical_ram_contract_probe.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "projected_oracle_report.json",
    "low_bit_boundary_report.json",
    "progressive_freeze_report.json",
    "mutation_repair_report.json",
    "bit_budget_report.json",
    "system_results.json",
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
    "continuous_direct_write_baseline",
    "oracle_write_continuous_reference",
    "oracle_write_binary_projected",
    "oracle_write_ternary_projected",
    "oracle_write_int4_projected",
    "learned_binary_ram_boundary",
    "learned_ternary_ram_boundary",
    "learned_int4_ram_boundary",
    "learned_binary_ram_boundary_plus_mutation_repair",
    "learned_ternary_ram_boundary_plus_mutation_repair",
    "learned_int4_ram_boundary_plus_mutation_repair",
    "pure_binary_pocket_and_ram",
    "pure_ternary_pocket_and_ram",
    "int4_pocket_and_ram",
    "mixed_precision_pocket_float_ram_lowbit",
    "dense_graph_danger_control",
)
ORACLE_SYSTEMS = {
    "oracle_write_continuous_reference",
    "oracle_write_binary_projected",
    "oracle_write_ternary_projected",
    "oracle_write_int4_projected",
}
REPAIR_SYSTEMS = {
    "learned_binary_ram_boundary_plus_mutation_repair",
    "learned_ternary_ram_boundary_plus_mutation_repair",
    "learned_int4_ram_boundary_plus_mutation_repair",
}
VALID_DECISIONS = (
    "e7z_binary_canonical_ram_contract_positive",
    "e7z_ternary_canonical_ram_contract_positive",
    "e7z_int4_canonical_ram_contract_positive",
    "e7z_low_bit_ram_contract_partially_positive",
    "e7z_low_bit_training_or_commit_learning_bottleneck",
    "e7z_full_low_bit_pocket_ram_preferred",
    "e7z_external_low_bit_ram_boundary_sufficient",
    "e7z_low_bit_mutation_repair_positive",
    "e7z_low_bit_canonical_ram_contract_not_sufficient",
    "e7z_graph_soup_regression_detected",
)
ALLOWED_TRAINING_CALL_FUNCTIONS = {"train_code_library", "seed_worker"}


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
            "train_read_map_library",
            "train_monolithic",
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
        return {"schema_version": "e7z_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    manifest = load_json(out / "backend_manifest.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    system_results = load_json(out / "system_results.json")
    projected = load_json(out / "projected_oracle_report.json")
    low_bit = load_json(out / "low_bit_boundary_report.json")
    freeze = load_json(out / "progressive_freeze_report.json")
    repair = load_json(out / "mutation_repair_report.json")
    bit_budget = load_json(out / "bit_budget_report.json")

    if decision.get("decision") not in VALID_DECISIONS:
        failures.append(f"INVALID_DECISION:{decision.get('decision')}")
    if replay.get("internal_replay_passed") is not True:
        failures.append("DETERMINISTIC_REPLAY_FAILED")
    for artifact, row in replay.get("hash_comparisons", {}).items():
        if row.get("match") is not True:
            failures.append(f"REPLAY_HASH_MISMATCH:{artifact}")
    if decision.get("deterministic_replay_passed") is not True or summary.get("deterministic_replay_passed") is not True:
        failures.append("REPLAY_FLAG_NOT_PROPAGATED")

    if manifest.get("semantic_lane_labels_as_model_input") is not False:
        failures.append("SEMANTIC_LABELS_AS_MODEL_INPUT")
    if manifest.get("new_router") is not False:
        failures.append("NEW_ROUTER_FLAG_ENABLED")
    if manifest.get("dense_graph_primary") is not False:
        failures.append("DENSE_GRAPH_PRIMARY_ENABLED")
    if manifest.get("oracle_used_as_reference_only") is not True:
        failures.append("ORACLE_REFERENCE_FLAG_MISSING")
    if manifest.get("mutation_repair_uses_backprop") is not False:
        failures.append("MUTATION_REPAIR_BACKPROP_ENABLED")
    if manifest.get("flow_dim") != 40:
        failures.append(f"UNEXPECTED_FLOW_DIM:{manifest.get('flow_dim')}")

    present = {row.get("system") for row in system_results.get("rows", [])}
    for system in SYSTEMS:
        if system not in present:
            failures.append(f"MISSING_SYSTEM:{system}")
        if aggregate.get("systems", {}).get(system, {}).get("seed_count", 0) <= 0:
            failures.append(f"NO_ROWS_FOR_SYSTEM:{system}")

    for row in system_results.get("rows", []):
        system = row.get("system")
        evals = row.get("evals", {})
        for split in ("heldout", "ood", "counterfactual", "adversarial"):
            split_row = evals.get(split, {})
            if not split_row.get("row_level_samples"):
                failures.append(f"MISSING_ROW_LEVEL_SAMPLE:{system}:{split}")
            for metric in (
                "answer_accuracy",
                "composition_usefulness",
                "canonical_state_validity",
                "oracle_write_similarity",
                "bundle_mean_absolute_error_to_oracle",
                "multi_cell_pattern_correlation",
                "support_channel_sign_mismatch_rate",
                "support_channel_silence_rate",
                "next_pocket_compatibility_error",
                "bit_budget",
                "boundary_bit_budget",
            ):
                if metric not in split_row:
                    failures.append(f"MISSING_SPLIT_METRIC:{system}:{split}:{metric}")
        for metric in (
            "eval_mean_composition_usefulness",
            "eval_mean_answer_accuracy",
            "eval_mean_canonical_state_validity",
            "eval_mean_bundle_mean_absolute_error_to_oracle",
            "eval_mean_next_pocket_compatibility_error",
            "bit_budget",
            "boundary_bit_budget",
        ):
            if metric not in row:
                failures.append(f"MISSING_RESULT_METRIC:{system}:{metric}")
        if system in REPAIR_SYSTEMS:
            for metric in ("mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "parameter_diff_hash"):
                if metric not in row:
                    failures.append(f"MISSING_REPAIR_METRIC:{system}:{metric}")

    projected_systems = {row.get("system") for row in projected.get("rows", [])}
    for system in ORACLE_SYSTEMS:
        if system not in projected_systems:
            failures.append(f"MISSING_PROJECTED_ORACLE:{system}")
    if not low_bit.get("contract_rows"):
        failures.append("MISSING_LOW_BIT_CONTRACT_ROWS")
    if not low_bit.get("sample_rows"):
        failures.append("MISSING_LOW_BIT_SAMPLE_ROWS")
    if not freeze.get("rows"):
        failures.append("MISSING_PROGRESSIVE_FREEZE_ROWS")
    if not repair.get("rows"):
        failures.append("MISSING_MUTATION_REPAIR_ROWS")
    if not bit_budget.get("rows"):
        failures.append("MISSING_BIT_BUDGET_ROWS")

    repair_systems_seen = {row.get("system") for row in repair.get("rows", [])}
    for system in REPAIR_SYSTEMS:
        if system not in repair_systems_seen:
            failures.append(f"MISSING_REPAIR_HISTORY:{system}")

    events = event_names(out / "progress.jsonl")
    for event in ("run_start", "primary_artifacts_written", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in events:
            failures.append(f"MISSING_PROGRESS_EVENT:{event}")

    report = (out / "report.md").read_text(encoding="utf-8").lower()
    for banned_label in ("truth slot", "memory slot", "confidence slot", "answer slot", "semantic lane"):
        if banned_label in report:
            failures.append(f"SEMANTIC_LANE_LABEL:{banned_label}")
    for banned_claim in ("agi", "consciousness", "model-scale", "raw-language"):
        if banned_claim in report and "does not make" not in report:
            failures.append(f"BANNED_CLAIM:{banned_claim}")

    failures.extend(ast_policy_failures())
    result = {"schema_version": "e7z_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
