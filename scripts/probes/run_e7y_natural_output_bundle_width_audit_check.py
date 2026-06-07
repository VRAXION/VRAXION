#!/usr/bin/env python3
"""Checker for E7Y natural output bundle width audit."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e7y_natural_output_bundle_width_audit.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "bundle_contract_report.json",
    "output_width_curve_report.json",
    "channel_morphology_report.json",
    "oracle_bundle_similarity_report.json",
    "ram_bundle_frame_report.json",
    "dense_graph_control_report.json",
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
    "single_value_write_baseline",
    "output_bundle_N2",
    "output_bundle_N3",
    "output_bundle_N4",
    "output_bundle_N5",
    "output_bundle_N6",
    "output_bundle_N8",
    "output_bundle_N12",
    "oracle_write_reference",
    "dense_graph_danger_control",
)
VALID_DECISIONS = (
    "e7y_natural_output_bundle_width_detected",
    "e7y_single_output_cell_sufficient",
    "e7y_large_output_bundle_required",
    "e7y_output_bundle_width_not_sufficient",
    "e7y_graph_soup_regression_detected",
)
ALLOWED_TRAINING_CALL_FUNCTIONS = {"train_bundle_library", "seed_worker"}


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
        return {"schema_version": "e7y_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    manifest = load_json(out / "backend_manifest.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    system_results = load_json(out / "system_results.json")
    contracts = load_json(out / "bundle_contract_report.json")
    curve = load_json(out / "output_width_curve_report.json")
    morphology = load_json(out / "channel_morphology_report.json")
    similarity = load_json(out / "oracle_bundle_similarity_report.json")
    frames = load_json(out / "ram_bundle_frame_report.json")
    dense = load_json(out / "dense_graph_control_report.json")

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
    if manifest.get("runtime_random_output_placement") is not False:
        failures.append("RUNTIME_RANDOM_OUTPUT_PLACEMENT")
    if manifest.get("new_router") is not False:
        failures.append("NEW_ROUTER_FLAG_ENABLED")
    if manifest.get("oracle_used_as_reference_only") is not True:
        failures.append("ORACLE_REFERENCE_FLAG_MISSING")
    if manifest.get("flow_dim") != 40:
        failures.append(f"UNEXPECTED_FLOW_DIM:{manifest.get('flow_dim')}")

    present_systems = {row.get("system") for row in system_results.get("rows", [])}
    for system in SYSTEMS:
        if system not in present_systems:
            failures.append(f"MISSING_SYSTEM:{system}")
        if aggregate.get("systems", {}).get(system, {}).get("seed_count", 0) <= 0:
            failures.append(f"NO_ROWS_FOR_SYSTEM:{system}")

    expected_widths = {
        "single_value_write_baseline": 1,
        "output_bundle_N2": 2,
        "output_bundle_N3": 3,
        "output_bundle_N4": 4,
        "output_bundle_N5": 5,
        "output_bundle_N6": 6,
        "output_bundle_N8": 8,
        "output_bundle_N12": 12,
    }
    for row in system_results.get("rows", []):
        system = row.get("system")
        if system in expected_widths and int(row.get("output_bundle_width", -1)) != expected_widths[system]:
            failures.append(f"WRONG_OUTPUT_WIDTH:{system}:{row.get('output_bundle_width')}")
        evals = row.get("evals", {})
        for split in ("heldout", "ood", "counterfactual", "adversarial"):
            split_row = evals.get(split, {})
            if not split_row.get("row_level_samples"):
                failures.append(f"MISSING_ROW_LEVEL_SAMPLE:{system}:{split}")
            for metric in (
                "answer_accuracy",
                "composition_usefulness",
                "output_bundle_width",
                "ram_cells_used",
                "oracle_bundle_similarity",
                "bundle_mean_absolute_error_to_oracle",
                "bundle_cellwise_correlation_with_oracle",
                "output_channel_redundancy",
                "next_pocket_input_compatibility",
            ):
                if metric not in split_row:
                    failures.append(f"MISSING_SPLIT_METRIC:{system}:{split}:{metric}")
        for metric in (
            "eval_mean_composition_usefulness",
            "eval_mean_answer_accuracy",
            "eval_mean_output_bundle_width",
            "eval_mean_ram_cells_used",
            "eval_mean_oracle_bundle_similarity",
            "eval_mean_bundle_mean_absolute_error_to_oracle",
            "eval_mean_bundle_cellwise_correlation_with_oracle",
            "eval_mean_output_channel_redundancy",
        ):
            if metric not in row:
                failures.append(f"MISSING_RESULT_METRIC:{system}:{metric}")

    contract_systems = {row.get("system") for row in contracts.get("rows", [])}
    for system in expected_widths:
        if system not in contract_systems:
            failures.append(f"MISSING_CONTRACT_SYSTEM:{system}")
    for row in contracts.get("rows", []):
        contract = row.get("contract", {})
        if contract.get("semantic_label_control") is not False:
            failures.append(f"SEMANTIC_CONTRACT_LABEL:{row.get('system')}:{row.get('skill')}")
        if int(contract.get("output_bundle_width", 0)) != int(row.get("contract", {}).get("ram_cells_used", 0)):
            failures.append(f"WIDTH_RAM_CELL_MISMATCH:{row.get('system')}:{row.get('skill')}")
        cells = contract.get("bundle_cells", [])
        if len(cells) != len(set(cells)):
            failures.append(f"DUPLICATE_BUNDLE_CELLS:{row.get('system')}:{row.get('skill')}")

    if len(curve.get("rows", [])) < len(expected_widths):
        failures.append("OUTPUT_WIDTH_CURVE_INCOMPLETE")
    if not morphology.get("rows"):
        failures.append("MISSING_CHANNEL_MORPHOLOGY_ROWS")
    if not similarity.get("rows"):
        failures.append("MISSING_ORACLE_BUNDLE_SIMILARITY_ROWS")
    if not frames.get("rows"):
        failures.append("MISSING_RAM_BUNDLE_FRAME_ROWS")
    if not dense.get("rows"):
        failures.append("MISSING_DENSE_GRAPH_CONTROL_ROWS")

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
    result = {"schema_version": "e7y_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
