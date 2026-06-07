#!/usr/bin/env python3
"""Checker for E7X output write value-format contrastive audit."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e7x_output_write_value_format_contrastive_audit.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "pocket_training_report.json",
    "read_map_report.json",
    "write_transform_report.json",
    "write_morphology_report.json",
    "write_histogram_report.json",
    "oracle_real_scatter_report.json",
    "ram_grid_frame_report.json",
    "top_failing_rows_report.json",
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
    "baseline_real_write",
    "oracle_write_reference",
    "affine_calibrated_write",
    "monotonic_calibrated_write",
    "zscore_normalized_write",
    "codebook_write",
    "sign_or_quantized_write",
    "residual_delta_write",
    "router_integrated_write",
)
SKILLS = ("compare", "mod_add", "parity", "threshold", "counterfactual_flip", "verify")
VALID_DECISIONS = (
    "e7x_output_scale_bias_calibration_bottleneck",
    "e7x_output_nonlinear_calibration_bottleneck",
    "e7x_canonical_value_code_required",
    "e7x_delta_write_format_required",
    "e7x_flow_integrator_required",
    "e7x_output_value_format_not_sufficient",
)
ALLOWED_TRAINING_CALL_FUNCTIONS = {"seed_worker"}


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
            "train_read_map_library",
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
        return {"schema_version": "e7x_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    manifest = load_json(out / "backend_manifest.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    system_results = load_json(out / "system_results.json")
    transforms = load_json(out / "write_transform_report.json")
    morphology = load_json(out / "write_morphology_report.json")
    histograms = load_json(out / "write_histogram_report.json")
    scatter = load_json(out / "oracle_real_scatter_report.json")
    frames = load_json(out / "ram_grid_frame_report.json")
    failing = load_json(out / "top_failing_rows_report.json")

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
    if manifest.get("new_architecture") is not False:
        failures.append("NEW_ARCHITECTURE_FLAG_ENABLED")
    if manifest.get("diagnostic_value_format_audit") is not True:
        failures.append("VALUE_FORMAT_AUDIT_NOT_DECLARED")
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

    for row in system_results.get("rows", []):
        evals = row.get("evals", {})
        for split in ("heldout", "ood", "counterfactual", "adversarial"):
            split_row = evals.get(split, {})
            if not split_row.get("row_level_samples"):
                failures.append(f"MISSING_ROW_LEVEL_SAMPLE:{row.get('system')}:{split}")
            for metric in ("answer_accuracy", "composition_usefulness", "oracle_write_similarity", "cellwise_correlation_with_oracle", "mean_absolute_error_to_oracle", "next_pocket_input_compatibility"):
                if metric not in split_row:
                    failures.append(f"MISSING_SPLIT_METRIC:{row.get('system')}:{split}:{metric}")
        for metric in (
            "eval_mean_oracle_write_similarity",
            "eval_mean_cellwise_correlation_with_oracle",
            "eval_mean_cosine_similarity_with_oracle",
            "eval_mean_absolute_error_to_oracle",
            "eval_mean_scale_ratio",
            "eval_mean_bias_offset",
            "eval_mean_saturation_rate",
            "eval_mean_sign_mismatch_rate",
            "eval_mean_entropy",
            "eval_mean_effective_value_levels",
            "eval_mean_noise_floor",
            "eval_mean_delta_magnitude",
            "eval_mean_next_pocket_input_compatibility",
        ):
            if metric not in row:
                failures.append(f"MISSING_VALUE_METRIC:{row.get('system')}:{metric}")

    for skill in SKILLS:
        if not any(row.get("skill") == skill for row in transforms.get("rows", [])):
            failures.append(f"MISSING_TRANSFORM_ROW:{skill}")
    if len(morphology.get("rows", [])) < len(SYSTEMS):
        failures.append("MORPHOLOGY_ROWS_INCOMPLETE")
    if not histograms.get("rows"):
        failures.append("MISSING_HISTOGRAM_ROWS")
    if not scatter.get("rows"):
        failures.append("MISSING_SCATTER_ROWS")
    if not frames.get("rows"):
        failures.append("MISSING_RAM_GRID_FRAME_ROWS")
    if "rows" not in failing:
        failures.append("MISSING_FAILING_ROWS_KEY")

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
    result = {"schema_version": "e7x_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
