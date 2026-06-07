#!/usr/bin/env python3
"""Checker for E8E Temporal Thought-Frame Divergence Audit."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e8e_temporal_thought_frame_divergence_audit.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "trace_divergence_report.json",
    "intervention_report.json",
    "attractor_report.json",
    "local_editability_report.json",
    "mutation_history_report.json",
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
    "oracle_trace_reference",
    "current_best_learned_trace",
    "consumer_distill_trace_reference",
    "substrate_first_trace",
    "mutation_only_trace",
    "dense_graph_danger_trace",
)
VALID_DECISIONS = (
    "e8e_first_step_write_divergence",
    "e8e_temporal_drift_accumulation",
    "e8e_consumer_sensitive_state_mismatch",
    "e8e_recoverable_state_drift",
    "e8e_wrong_attractor_trace",
    "e8e_answer_shortcut_trace_invalid",
)
MUTATION_SYSTEMS = {"mutation_only_trace"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def ast_policy(failures: list[dict[str, Any]]) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    forbidden_backward: list[int] = []
    forbidden_optimizer: list[int] = []
    semantic_name_hits: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> Any:
            name = ""
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if isinstance(node.func.value, ast.Attribute):
                    name = f"{node.func.value.attr}.{name}"
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name == "backward":
                forbidden_backward.append(node.lineno)
            if "optim" in name.lower() or name in {"AdamW", "SGD"}:
                forbidden_optimizer.append(node.lineno)
            self.generic_visit(node)

        def visit_Constant(self, node: ast.Constant) -> Any:
            if isinstance(node.value, str):
                lower = node.value.lower()
                for token in ("confidence slot", "truth slot", "memory slot", "reason slot"):
                    if token in lower:
                        semantic_name_hits.append((node.lineno, token))
            self.generic_visit(node)

    Visitor().visit(tree)
    for lineno in forbidden_backward:
        add_failure(failures, "UNEXPECTED_BACKWARD_IN_E8E_RUNNER", f"backward call at line {lineno}")
    for lineno in forbidden_optimizer:
        add_failure(failures, "UNEXPECTED_OPTIMIZER_IN_E8E_RUNNER", f"optimizer call at line {lineno}")
    for lineno, token in semantic_name_hits:
        add_failure(failures, "SEMANTIC_LANE_LABEL_STRING", f"{token} at line {lineno}")


def check(out: Path, write_summary: bool = False) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            add_failure(failures, "MISSING_ARTIFACT", name)
    if failures:
        result = {"schema_version": "e8e_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
        if write_summary:
            (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result

    manifest = load_json(out / "backend_manifest.json")
    decision = load_json(out / "decision.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    systems = aggregate.get("systems", {})
    system_rows = load_json(out / "system_results.json").get("rows", [])
    trace_rows = load_json(out / "trace_divergence_report.json").get("rows", [])
    intervention_rows = load_json(out / "intervention_report.json").get("rows", [])
    attractor_rows = load_json(out / "attractor_report.json").get("rows", [])
    edit_rows = load_json(out / "local_editability_report.json").get("rows", [])
    mutation_rows = load_json(out / "mutation_history_report.json").get("rows", [])
    sample_rows = load_json(out / "row_level_samples.json").get("rows", [])
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

    if manifest.get("diagnostic_only") is not True:
        add_failure(failures, "DIAGNOSTIC_ONLY_FLAG_MISSING", "backend_manifest.json")
    if manifest.get("new_architecture") is not False:
        add_failure(failures, "NEW_ARCHITECTURE_FLAGGED", "E8E must not add architecture")
    if manifest.get("new_router") is not False:
        add_failure(failures, "NEW_ROUTER_FLAGGED", "E8E must not add router")
    if manifest.get("semantic_lane_labels_as_model_input") is not False:
        add_failure(failures, "SEMANTIC_LABELS_FLAGGED", "semantic labels cannot be model input")
    if manifest.get("oracle_write_at_inference_for_learned_systems") is not False:
        add_failure(failures, "ORACLE_LEARNED_INFERENCE_FLAGGED", "learned systems cannot use oracle writes")
    if manifest.get("oracle_writes_allowed_only_in_reference_and_intervention_arms") is not True:
        add_failure(failures, "ORACLE_INTERVENTION_GUARD_MISSING", "manifest guard missing")
    if manifest.get("row_level_eval") is not True:
        add_failure(failures, "ROW_LEVEL_EVAL_FLAG_MISSING", "backend_manifest.json")

    for system in SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_SYSTEM_IN_AGGREGATE", system)
    result_systems = {row.get("system") for row in system_rows}
    trace_systems = {row.get("system") for row in trace_rows}
    for system in SYSTEMS:
        if system not in result_systems:
            add_failure(failures, "MISSING_SYSTEM_RESULT", system)
        if system not in trace_systems:
            add_failure(failures, "MISSING_TRACE_SYSTEM_ROWS", system)

    required_mean_metrics = (
        "eval_mean_composition_usefulness",
        "eval_mean_answer_accuracy",
        "eval_mean_trace_similarity",
        "eval_mean_flow_frame_mae_to_oracle",
        "eval_mean_flow_delta_mae_to_oracle",
        "eval_mean_consumer_read_mask_mae",
        "eval_mean_first_divergence_step",
        "eval_mean_drift_slope",
        "eval_mean_transition_validity",
        "eval_mean_wrong_attractor_rate",
    )
    for system, payload in systems.items():
        mean = payload.get("mean", {})
        for metric in required_mean_metrics:
            if metric not in mean:
                add_failure(failures, "MISSING_SYSTEM_METRIC", f"{system}:{metric}")

    if not trace_rows:
        add_failure(failures, "MISSING_TRACE_ROWS", "trace_divergence_report.json")
    else:
        required_trace_metrics = (
            "frame_mae_to_oracle",
            "frame_cosine_to_oracle",
            "frame_correlation_to_oracle",
            "delta_mae_to_oracle",
            "consumer_read_mask_mae",
            "result_cell_error",
            "support_cell_sign_mismatch",
            "transition_validity",
        )
        for row in trace_rows[: min(200, len(trace_rows))]:
            for metric in required_trace_metrics:
                if metric not in row:
                    add_failure(failures, "MISSING_TRACE_METRIC", f"{row.get('system')}:{metric}")

    if not intervention_rows:
        add_failure(failures, "MISSING_INTERVENTION_ROWS", "intervention_report.json")
    else:
        interventions = {row.get("intervention") for row in intervention_rows}
        for name in (
            "oracle_reset_after_step_1",
            "oracle_reset_after_each_step",
            "learned_step_1_oracle_rest",
            "oracle_step_1_learned_rest",
            "one_learned_pocket_at_a_time",
            "consumer_sensitive_cell_replacement_only",
        ):
            if name not in interventions:
                add_failure(failures, "MISSING_INTERVENTION_ARM", name)

    if not attractor_rows:
        add_failure(failures, "MISSING_ATTRACTOR_ROWS", "attractor_report.json")
    if not edit_rows:
        add_failure(failures, "MISSING_LOCAL_EDITABILITY_ROWS", "local_editability_report.json")
    if not sample_rows:
        add_failure(failures, "MISSING_ROW_LEVEL_SAMPLES", "row_level_samples.json")
    if not dynamics_rows:
        add_failure(failures, "MISSING_PRODUCER_DYNAMICS", "producer_dynamics_report.json")

    mutation_by_system = {row.get("system") for row in mutation_rows}
    for system in MUTATION_SYSTEMS:
        if system not in mutation_by_system:
            add_failure(failures, "MISSING_MUTATION_HISTORY", system)
        subset = [row for row in mutation_rows if row.get("system") == system]
        accepted = sum(int(row.get("accepted", 0)) for row in subset)
        rejected = sum(int(row.get("rejected", 0)) for row in subset)
        rollback = sum(int(row.get("rollback", 0)) for row in subset)
        if accepted + rejected <= 0 or rejected <= 0 or rollback <= 0:
            add_failure(failures, "MISSING_ACCEPT_REJECT_ROLLBACK", f"{system}: accepted={accepted} rejected={rejected} rollback={rollback}")

    progress_text = (out / "progress.jsonl").read_text(encoding="utf-8")
    for event in ("run_start", "e8e_trace_system_evaluated", "primary_artifacts_written", "deterministic_replay_complete"):
        if event not in progress_text:
            add_failure(failures, "MISSING_PROGRESS_EVENT", event)

    ast_policy(failures)
    result = {"schema_version": "e8e_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
