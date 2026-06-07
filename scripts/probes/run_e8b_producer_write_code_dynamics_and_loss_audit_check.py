#!/usr/bin/env python3
"""Checker for E8B producer write-code dynamics and loss audit."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e8b_producer_write_code_dynamics_and_loss_audit.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "producer_dynamics_report.json",
    "gradient_diagnostics_report.json",
    "teacher_comparison_report.json",
    "loss_variant_report.json",
    "curriculum_report.json",
    "mutation_repair_report.json",
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
    "current_projection_teacher_baseline",
    "simplified_canonical_teacher",
    "factorized_teacher_code",
    "codebook_teacher",
    "local_smooth_teacher_code",
    "per_cell_supervised_loss",
    "weighted_per_cell_loss",
    "contrastive_plus_per_cell_loss",
    "consumer_aware_loss",
    "code_similarity_loss",
    "soft_continuous_to_int4",
    "int4_to_ternary_to_binary",
    "progressive_cell_freeze_after_similarity_threshold",
    "mutation_repair_after_similarity_threshold",
    "mutation_only_from_random_lowbit",
    "dense_graph_danger_control",
    "consumer_distill_reference",
    "oracle_low_bit_reference",
)
VALID_DECISIONS = (
    "e8b_current_projection_teacher_too_hard",
    "e8b_factorized_or_codebook_teacher_positive",
    "e8b_per_cell_supervision_positive",
    "e8b_producer_generalization_bottleneck",
    "e8b_producer_capacity_or_loss_bottleneck",
    "e8b_gradient_conflict_or_jagged_target_confirmed",
    "e8b_teacher_or_architecture_mismatch",
    "e8b_seeded_mutation_repair_positive",
    "e8b_mutation_only_producer_learning_viable",
    "e8b_graph_soup_regression_detected",
)
MUTATION_SYSTEMS = {"mutation_repair_after_similarity_threshold", "mutation_only_from_random_lowbit"}
TRAINED_REQUIRED = set(SYSTEMS) - {"mutation_only_from_random_lowbit", "dense_graph_danger_control", "consumer_distill_reference", "oracle_low_bit_reference"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def events(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(json.loads(line)["event"])
    return out


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
            if fn != "train_producer_diagnostic":
                failures.append(f"UNEXPECTED_OPTIMIZER_OR_BACKPROP:{fn}:{node.attr}")
    return failures


def check(out: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (out / artifact).exists():
            failures.append(f"MISSING_ARTIFACT:{artifact}")
    if failures:
        return {"schema_version": "e8b_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}

    manifest = load_json(out / "backend_manifest.json")
    decision = load_json(out / "decision.json")
    summary = load_json(out / "summary.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    systems = load_json(out / "system_results.json")
    dynamics = load_json(out / "producer_dynamics_report.json")
    gradients = load_json(out / "gradient_diagnostics_report.json")
    teachers = load_json(out / "teacher_comparison_report.json")
    losses = load_json(out / "loss_variant_report.json")
    curriculum = load_json(out / "curriculum_report.json")
    repair = load_json(out / "mutation_repair_report.json")
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
    if manifest.get("oracle_write_at_inference_for_learned_systems") is not False:
        failures.append("ORACLE_WRITE_AT_LEARNED_INFERENCE")
    if manifest.get("gradient_diagnostics_logged") is not True:
        failures.append("GRADIENT_DIAGNOSTICS_FLAG_MISSING")
    if manifest.get("mutation_repair_uses_backprop") is not False:
        failures.append("MUTATION_REPAIR_BACKPROP_FLAG_ENABLED")

    present = {row.get("system") for row in systems.get("rows", [])}
    for system in SYSTEMS:
        if system not in present:
            failures.append(f"MISSING_SYSTEM:{system}")
        if aggregate.get("systems", {}).get(system, {}).get("seed_count", 0) <= 0:
            failures.append(f"NO_AGGREGATE_ROWS:{system}")

    for row in systems.get("rows", []):
        system = row.get("system")
        for metric in (
            "eval_mean_composition_usefulness",
            "eval_mean_answer_accuracy",
            "eval_mean_oracle_code_similarity",
            "eval_mean_bundle_mean_absolute_error_to_oracle",
            "eval_mean_next_pocket_compatibility_error",
            "eval_mean_write_entropy",
        ):
            if metric not in row:
                failures.append(f"MISSING_SYSTEM_METRIC:{system}:{metric}")
        for split in ("heldout", "ood", "counterfactual", "adversarial"):
            split_row = row.get("evals", {}).get(split, {})
            if not split_row.get("row_level_samples"):
                failures.append(f"MISSING_ROW_LEVEL_SAMPLE:{system}:{split}")
        if system in MUTATION_SYSTEMS:
            for metric in ("mutation_attempts", "accepted_mutations", "rejected_mutations", "rollback_count", "parameter_diff_hash"):
                if metric not in row:
                    failures.append(f"MISSING_MUTATION_METRIC:{system}:{metric}")

    dyn_systems = {row.get("system") for row in dynamics.get("rows", [])}
    for system in TRAINED_REQUIRED:
        if system not in dyn_systems:
            failures.append(f"MISSING_DYNAMICS_FOR_SYSTEM:{system}")
    for row in dynamics.get("rows", []):
        summary_row = row.get("summary", {})
        for metric in ("loss_drop", "validation_code_similarity_best", "tail_gain", "tail_range", "gradient_norm_mean", "gradient_cosine_mean"):
            if metric not in summary_row:
                failures.append(f"MISSING_DYNAMICS_SUMMARY:{row.get('system')}:{metric}")
        if row.get("oracle_used_at_inference") is True:
            failures.append(f"DYNAMICS_ORACLE_INFERENCE:{row.get('system')}")

    if not gradients.get("rows"):
        failures.append("MISSING_GRADIENT_ROWS")
    else:
        for row in gradients.get("rows", [])[: min(200, len(gradients.get("rows", [])))]:
            for metric in (
                "train_loss",
                "validation_loss",
                "train_code_similarity",
                "validation_code_similarity",
                "ood_code_similarity",
                "per_cell_mae",
                "per_cell_sign_accuracy",
                "gradient_norm",
                "gradient_variance",
                "gradient_cosine",
                "gradient_cosine_negative_rate",
                "validation_write_entropy",
            ):
                if metric not in row:
                    failures.append(f"MISSING_GRADIENT_METRIC:{row.get('system')}:{metric}")
    if len({row.get("teacher_style") for row in teachers.get("rows", [])}) < 5:
        failures.append("MISSING_TEACHER_VARIANTS")
    if not losses.get("rows"):
        failures.append("MISSING_LOSS_VARIANT_ROWS")
    if not curriculum.get("rows"):
        failures.append("MISSING_CURRICULUM_ROWS")
    repair_systems = {row.get("system") for row in repair.get("rows", [])}
    for system in MUTATION_SYSTEMS:
        if system not in repair_systems:
            failures.append(f"MISSING_REPAIR_HISTORY:{system}")
    if not samples.get("rows"):
        failures.append("MISSING_ROW_LEVEL_SAMPLE_ARTIFACT")

    event_set = events(out / "progress.jsonl")
    for event in ("run_start", "e8b_producer_epoch", "primary_artifacts_written", "deterministic_replay_start", "deterministic_replay_complete", "final_artifacts_written"):
        if event not in event_set:
            failures.append(f"MISSING_PROGRESS_EVENT:{event}")

    report = (out / "report.md").read_text(encoding="utf-8").lower()
    for banned in ("truth slot", "memory slot", "confidence slot", "answer slot", "semantic lane"):
        if banned in report:
            failures.append(f"SEMANTIC_LANE_LABEL:{banned}")
    for banned_claim in ("agi", "consciousness", "model-scale", "raw-language"):
        if banned_claim in report and "no raw-language" not in report:
            failures.append(f"BANNED_CLAIM:{banned_claim}")

    failures.extend(ast_policy_failures())
    result = {"schema_version": "e8b_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
