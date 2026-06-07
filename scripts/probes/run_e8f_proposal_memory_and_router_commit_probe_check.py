#!/usr/bin/env python3
"""Checker for E8F Proposal Memory And Router Commit Probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e8f_proposal_memory_and_router_commit_probe.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "proposal_memory_report.json",
    "commit_controller_report.json",
    "temporal_trace_report.json",
    "dense_graph_control_report.json",
    "producer_dynamics_report.json",
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
    "direct_overwrite_baseline",
    "output_feedback_only",
    "proposal_memory_no_commit",
    "proposal_memory_plus_simple_commit",
    "proposal_memory_plus_router_commit_gate",
    "proposal_memory_plus_learned_commit",
    "proposal_memory_plus_per_skill_commit",
    "proposal_memory_ring_buffer",
    "proposal_memory_plus_verifier_pocket",
    "proposal_memory_plus_stepwise_renormalization",
    "oracle_stepwise_commit_reference",
    "dense_graph_danger_control",
)
VALID_DECISIONS = (
    "e8f_proposal_memory_commit_positive",
    "e8f_output_feedback_sufficient",
    "e8f_commit_controller_required",
    "e8f_shared_commit_controller_positive",
    "e8f_per_skill_commit_required",
    "e8f_proposal_trace_memory_positive",
    "e8f_verifier_commit_required",
    "e8f_stepwise_renormalization_positive",
    "e8f_proposal_memory_not_sufficient",
    "e8f_answer_shortcut_trace_invalid",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def ast_policy(failures: list[dict[str, Any]]) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    forbidden_backward: list[int] = []
    semantic_name_hits: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> Any:
            name = ""
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name == "backward":
                forbidden_backward.append(node.lineno)
            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> Any:
            if isinstance(node.value, str):
                lower = node.value.lower()
                for token in ("confidence slot", "truth slot", "memory label", "reason slot"):
                    if token in lower:
                        semantic_name_hits.append((node.lineno, token))
            self.generic_visit(node)

    Visitor().visit(tree)
    for lineno in forbidden_backward:
        add_failure(failures, "UNEXPECTED_BACKWARD_IN_E8F_RUNNER", f"backward call at line {lineno}")
    for lineno, token in semantic_name_hits:
        add_failure(failures, "SEMANTIC_LABEL_STRING", f"{token} at line {lineno}")


def check(out: Path, write_summary: bool = False) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            add_failure(failures, "MISSING_ARTIFACT", name)
    if failures:
        result = {"schema_version": "e8f_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
        if write_summary:
            (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result

    manifest = load_json(out / "backend_manifest.json")
    decision = load_json(out / "decision.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    systems = aggregate.get("systems", {})
    system_rows = load_json(out / "system_results.json").get("rows", [])
    proposal_rows = load_json(out / "proposal_memory_report.json").get("rows", [])
    commit_rows = load_json(out / "commit_controller_report.json").get("rows", [])
    trace_rows = load_json(out / "temporal_trace_report.json").get("rows", [])
    dense_rows = load_json(out / "dense_graph_control_report.json").get("rows", [])
    samples = load_json(out / "row_level_samples.json").get("rows", [])
    dynamics_rows = load_json(out / "producer_dynamics_report.json").get("rows", [])

    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if not decision.get("deterministic_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_NOT_MARKED_PASS", "decision.json")
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "deterministic_replay.json")
    for name, item in replay.get("hash_comparisons", {}).items():
        if not item.get("match", False):
            add_failure(failures, "REPLAY_HASH_MISMATCH", name)

    if manifest.get("proposal_memory") is not True:
        add_failure(failures, "PROPOSAL_MEMORY_FLAG_MISSING", "backend_manifest.json")
    if manifest.get("stable_flow_requires_commit") is not True:
        add_failure(failures, "COMMIT_GUARD_FLAG_MISSING", "backend_manifest.json")
    if manifest.get("new_router_architecture") is not False:
        add_failure(failures, "NEW_ROUTER_ARCHITECTURE_FLAGGED", "E8F must only add commit-controller variants")
    if manifest.get("semantic_lane_labels_as_model_input") is not False:
        add_failure(failures, "SEMANTIC_LABELS_FLAGGED", "semantic labels cannot be model input")
    if manifest.get("oracle_write_at_inference_for_learned_systems") is not False:
        add_failure(failures, "ORACLE_LEARNED_INFERENCE_FLAGGED", "learned systems cannot use oracle writes")
    if manifest.get("dense_graph_primary_success_allowed") is not False:
        add_failure(failures, "DENSE_GRAPH_PRIMARY_ALLOWED", "dense graph cannot be primary success")
    if manifest.get("row_level_eval") is not True:
        add_failure(failures, "ROW_LEVEL_EVAL_MISSING", "backend_manifest.json")

    result_systems = {row.get("system") for row in system_rows}
    trace_systems = {row.get("system") for row in trace_rows}
    for system in SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_SYSTEM_IN_AGGREGATE", system)
        if system not in result_systems:
            add_failure(failures, "MISSING_SYSTEM_RESULT", system)
        if system not in trace_systems:
            add_failure(failures, "MISSING_TRACE_ROWS_FOR_SYSTEM", system)

    required_mean_metrics = (
        "eval_mean_composition_usefulness",
        "eval_mean_answer_accuracy",
        "eval_mean_trace_validity",
        "eval_mean_frame_mae_to_oracle",
        "eval_mean_delta_mae_to_oracle",
        "eval_mean_read_mae_on_next_pocket_cells",
        "eval_mean_drift_slope",
        "eval_mean_first_divergence_step",
    )
    for system, payload in systems.items():
        mean = payload.get("mean", {})
        for metric in required_mean_metrics:
            if metric not in mean:
                add_failure(failures, "MISSING_SYSTEM_METRIC", f"{system}:{metric}")

    if not proposal_rows:
        add_failure(failures, "MISSING_PROPOSAL_ROWS", "proposal_memory_report.json")
    else:
        required = ("proposal_memory_utilization", "proposal_acceptance_rate", "proposal_rejection_rate", "commit_correction_magnitude")
        for row in proposal_rows[: min(300, len(proposal_rows))]:
            for metric in required:
                if metric not in row:
                    add_failure(failures, "MISSING_PROPOSAL_METRIC", metric)
    if not commit_rows:
        add_failure(failures, "MISSING_COMMIT_ROWS", "commit_controller_report.json")
    else:
        if not any(row.get("system") == "proposal_memory_plus_learned_commit" for row in commit_rows):
            add_failure(failures, "MISSING_SHARED_COMMIT_REPORT", "proposal_memory_plus_learned_commit")
        if not any(row.get("system") == "proposal_memory_plus_per_skill_commit" for row in commit_rows):
            add_failure(failures, "MISSING_PER_SKILL_COMMIT_REPORT", "proposal_memory_plus_per_skill_commit")
        for row in commit_rows:
            if row.get("semantic_labels_used") is not False:
                add_failure(failures, "COMMIT_SEMANTIC_LABEL_FLAGGED", str(row))
            if row.get("oracle_used_at_inference") is not False:
                add_failure(failures, "COMMIT_ORACLE_INFERENCE_FLAGGED", str(row))
    if not trace_rows:
        add_failure(failures, "MISSING_TEMPORAL_TRACE_ROWS", "temporal_trace_report.json")
    if not dense_rows:
        add_failure(failures, "MISSING_DENSE_GRAPH_CONTROL_ROWS", "dense_graph_control_report.json")
    if not samples:
        add_failure(failures, "MISSING_ROW_LEVEL_SAMPLES", "row_level_samples.json")
    if not dynamics_rows:
        add_failure(failures, "MISSING_PRODUCER_DYNAMICS", "producer_dynamics_report.json")

    progress_text = (out / "progress.jsonl").read_text(encoding="utf-8")
    for event in ("run_start", "e8f_system_evaluated", "primary_artifacts_written", "deterministic_replay_complete"):
        if event not in progress_text:
            add_failure(failures, "MISSING_PROGRESS_EVENT", event)

    ast_policy(failures)
    result = {"schema_version": "e8f_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
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
