#!/usr/bin/env python3
"""Checker for E8A canonical RAM code learning and smoothness probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e8a_canonical_ram_code_learning_and_smoothness_probe.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "producer_distillation_report.json",
    "consumer_distillation_report.json",
    "staged_composition_report.json",
    "smoothness_report.json",
    "mutation_repair_report.json",
    "code_teacher_comparison_report.json",
    "system_results.json",
    "row_level_samples.json",
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
    "current_best_baseline",
    "oracle_low_bit_reference",
    "producer_distill_binary",
    "producer_distill_ternary",
    "producer_distill_int4",
    "consumer_distill_binary",
    "producer_consumer_staged_binary",
    "producer_consumer_staged_ternary",
    "producer_consumer_staged_int4",
    "soft_to_hard_int4_to_ternary_to_binary",
    "contrastive_ram_code_alignment",
    "progressive_code_freeze",
    "mutation_only_from_random_lowbit",
    "mutation_repair_after_distillation",
    "full_end_to_end_control",
    "dense_graph_danger_control",
)
VALID_DECISIONS = (
    "e8a_canonical_ram_code_distillation_positive",
    "e8a_consumer_read_bottleneck",
    "e8a_producer_write_bottleneck",
    "e8a_soft_to_hard_code_curriculum_required",
    "e8a_int4_code_required",
    "e8a_binary_canonical_code_learned",
    "e8a_mutation_repair_after_distillation_positive",
    "e8a_mutation_only_code_learning_viable",
    "e8a_current_oracle_code_too_jagged",
    "e8a_canonical_ram_code_learning_failed",
    "e8a_graph_soup_regression_detected",
)
MUTATION_SYSTEMS = {"mutation_only_from_random_lowbit", "mutation_repair_after_distillation"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def event_names(path: Path) -> set[str]:
    events: set[str] = set()
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
        return {"schema_version": "e8a_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    manifest = load_json(out / "backend_manifest.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    system_results = load_json(out / "system_results.json")
    producer = load_json(out / "producer_distillation_report.json")
    consumer = load_json(out / "consumer_distillation_report.json")
    staged = load_json(out / "staged_composition_report.json")
    smooth = load_json(out / "smoothness_report.json")
    repair = load_json(out / "mutation_repair_report.json")
    teacher = load_json(out / "code_teacher_comparison_report.json")
    samples = load_json(out / "row_level_samples.json")

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
    if manifest.get("oracle_write_at_inference_for_learned_systems") is not False:
        failures.append("ORACLE_WRITE_AT_LEARNED_INFERENCE")
    if manifest.get("oracle_used_as_teacher_target") is not True:
        failures.append("ORACLE_TEACHER_TARGET_FLAG_MISSING")
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
        for metric in (
            "eval_mean_composition_usefulness",
            "eval_mean_answer_accuracy",
            "eval_mean_oracle_code_similarity",
            "eval_mean_bundle_mean_absolute_error_to_oracle",
            "eval_mean_next_pocket_compatibility_error",
        ):
            if metric not in row:
                failures.append(f"MISSING_RESULT_METRIC:{system}:{metric}")
        evals = row.get("evals", {})
        for split in ("heldout", "ood", "counterfactual", "adversarial"):
            split_row = evals.get(split, {})
            if not split_row.get("row_level_samples"):
                failures.append(f"MISSING_ROW_LEVEL_SAMPLE:{system}:{split}")
            for metric in ("answer_accuracy", "composition_usefulness", "oracle_code_similarity", "bundle_mean_absolute_error_to_oracle", "next_pocket_compatibility_error"):
                if metric not in split_row:
                    failures.append(f"MISSING_SPLIT_METRIC:{system}:{split}:{metric}")
        if system in MUTATION_SYSTEMS:
            for metric in ("mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "parameter_diff_hash"):
                if metric not in row:
                    failures.append(f"MISSING_MUTATION_METRIC:{system}:{metric}")

    if not producer.get("rows"):
        failures.append("MISSING_PRODUCER_DISTILLATION_ROWS")
    else:
        for row in producer.get("rows", []):
            if row.get("system") != "baseline_standalone_pocket" and row.get("oracle_used_as_teacher_target") is not True:
                failures.append(f"PRODUCER_ROW_NOT_TEACHER_MARKED:{row.get('system')}")
            if row.get("oracle_used_at_inference") is True:
                failures.append(f"PRODUCER_ORACLE_INFERENCE:{row.get('system')}")
    if not consumer.get("rows"):
        failures.append("MISSING_CONSUMER_DISTILLATION_ROWS")
    if not staged.get("rows"):
        failures.append("MISSING_STAGED_COMPOSITION_ROWS")
    if not smooth.get("rows"):
        failures.append("MISSING_SMOOTHNESS_ROWS")
    else:
        for row in smooth.get("rows", []):
            for metric in ("one_bit_flip_average_fitness_drop_proxy", "two_bit_flip_average_fitness_drop_proxy", "local_neighborhood_valid_rate_1bit", "capture_basin_radius_proxy"):
                if metric not in row:
                    failures.append(f"MISSING_SMOOTHNESS_METRIC:{row.get('system')}:{metric}")
    repair_systems = {row.get("system") for row in repair.get("rows", [])}
    for system in MUTATION_SYSTEMS:
        if system not in repair_systems:
            failures.append(f"MISSING_MUTATION_HISTORY:{system}")
    teacher_styles = {row.get("teacher_style") for row in teacher.get("rows", [])}
    for style in ("current_oracle_projection_code", "simplified_canonical_code"):
        if style not in teacher_styles:
            failures.append(f"MISSING_TEACHER_STYLE:{style}")
    if not samples.get("rows"):
        failures.append("MISSING_ROW_LEVEL_SAMPLE_ARTIFACT")

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
    result = {"schema_version": "e8a_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
