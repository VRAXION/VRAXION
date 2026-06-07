#!/usr/bin/env python3
"""Checker for E8D Substrate-First RAM Language Pretraining Probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e8d_substrate_first_ram_language_pretraining_probe.py")
REQUIRED_ARTIFACTS = (
    "backend_manifest.json",
    "task_generation_report.json",
    "substrate_pretraining_report.json",
    "ram_validity_report.json",
    "producer_dynamics_report.json",
    "consumer_read_report.json",
    "compatibility_report.json",
    "mutation_repair_report.json",
    "gradient_diagnostics_report.json",
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
    "no_substrate_baseline",
    "bridge_only_baseline",
    "substrate_autoencoder",
    "substrate_transition_model",
    "low_bit_substrate_codebook",
    "frozen_substrate_then_producer",
    "frozen_substrate_then_consumer",
    "frozen_substrate_then_pocket_composition",
    "jointly_mutable_substrate_and_pockets",
    "oracle_substrate_reference",
    "dense_graph_danger_control",
)
VALID_DECISIONS = (
    "e8d_substrate_first_positive",
    "e8d_bridge_adapter_sufficient",
    "e8d_pocket_to_substrate_write_bottleneck",
    "e8d_substrate_consumer_read_bottleneck",
    "e8d_frozen_substrate_too_rigid",
    "e8d_graph_soup_regression_detected",
    "e8d_substrate_language_not_helpful",
)
MUTATION_SYSTEMS = {"jointly_mutable_substrate_and_pockets"}
TRAINED_REQUIRED = set(SYSTEMS) - {"oracle_substrate_reference", "dense_graph_danger_control"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def ast_policy(failures: list[dict[str, Any]]) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    optimizer_calls: list[tuple[int, str]] = []
    backward_calls: list[int] = []
    function_stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        def visit_Call(self, node: ast.Call) -> Any:
            name = ""
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
                if isinstance(node.func.value, ast.Attribute):
                    name = f"{node.func.value.attr}.{name}"
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            current = function_stack[-1] if function_stack else "<module>"
            if "optim" in name.lower() or name in {"AdamW", "SGD"}:
                optimizer_calls.append((node.lineno, current))
            if name == "backward":
                backward_calls.append(node.lineno)
            self.generic_visit(node)

    Visitor().visit(tree)
    allowed_optimizer_functions = {"train_library_on_contexts"}
    for lineno, func in optimizer_calls:
        if func not in allowed_optimizer_functions:
            add_failure(failures, "UNEXPECTED_OPTIMIZER_PATH", f"optimizer call at line {lineno} inside {func}")
    for lineno in backward_calls:
        # E8D delegates producer training to the imported E8C helper; the runner
        # itself should not introduce ad-hoc backward paths.
        add_failure(failures, "UNEXPECTED_BACKWARD_IN_E8D_RUNNER", f"backward call at line {lineno}")


def check(out: Path, write_summary: bool = False) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            add_failure(failures, "MISSING_ARTIFACT", name)
    if failures:
        result = {"schema_version": "e8d_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
        if write_summary:
            (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result

    manifest = load_json(out / "backend_manifest.json")
    decision = load_json(out / "decision.json")
    replay = load_json(out / "deterministic_replay.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    systems = aggregate.get("systems", {})
    system_results = load_json(out / "system_results.json").get("rows", [])
    substrate_rows = load_json(out / "substrate_pretraining_report.json").get("rows", [])
    validity_rows = load_json(out / "ram_validity_report.json").get("rows", [])
    compatibility_rows = load_json(out / "compatibility_report.json").get("rows", [])
    gradient_rows = load_json(out / "gradient_diagnostics_report.json").get("rows", [])
    repair_rows = load_json(out / "mutation_repair_report.json").get("rows", [])
    sample_rows = load_json(out / "row_level_samples.json").get("rows", [])

    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if not decision.get("deterministic_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_NOT_MARKED_PASS", "decision.json")
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "deterministic_replay.json")
    for name, item in replay.get("hash_comparisons", {}).items():
        if not item.get("match", False):
            add_failure(failures, "REPLAY_HASH_MISMATCH", name)
    if manifest.get("new_router") is not False:
        add_failure(failures, "NEW_ROUTER_FLAGGED", "E8D must not add a new router")
    if manifest.get("semantic_lane_labels_as_model_input") is not False:
        add_failure(failures, "SEMANTIC_LABELS_FLAGGED", "semantic labels must not be model input")
    if manifest.get("substrate_final_answer_objective") is not False:
        add_failure(failures, "FINAL_ANSWER_OBJECTIVE_FLAGGED", "substrate must not learn RAM -> final answer")
    if manifest.get("oracle_write_at_inference_for_learned_systems") is not False:
        add_failure(failures, "ORACLE_INFERENCE_FLAGGED", "learned systems cannot use oracle writes at inference")

    for system in SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_SYSTEM_IN_AGGREGATE", system)
    result_systems = {row.get("system") for row in system_results}
    for system in SYSTEMS:
        if system not in result_systems:
            add_failure(failures, "MISSING_SYSTEM_RESULT", system)

    for system, payload in systems.items():
        mean = payload.get("mean", {})
        for metric in (
            "eval_mean_composition_usefulness",
            "eval_mean_answer_accuracy",
            "eval_mean_producer_write_compatibility",
            "eval_mean_consumer_read_compatibility",
            "eval_mean_next_pocket_compatibility_error",
            "eval_mean_state_drift_per_step",
        ):
            if metric not in mean:
                add_failure(failures, "MISSING_SYSTEM_METRIC", f"{system}:{metric}")

    if not sample_rows:
        add_failure(failures, "MISSING_ROW_LEVEL_SAMPLES", "row_level_samples.json")
    if not gradient_rows:
        add_failure(failures, "MISSING_GRADIENT_DIAGNOSTICS", "gradient_diagnostics_report.json")
    elif not any(abs(float(row.get("gradient_cosine", 0.0))) + abs(float(row.get("gradient_variance", 0.0))) > 1e-9 for row in gradient_rows):
        add_failure(failures, "DEGENERATE_GRADIENT_DIAGNOSTICS", "all gradient diagnostics are zero")

    if not substrate_rows:
        add_failure(failures, "MISSING_SUBSTRATE_ROWS", "substrate_pretraining_report.json")
    for row in substrate_rows:
        if row.get("semantic_labels_used") is not False:
            add_failure(failures, "SUBSTRATE_SEMANTIC_LABEL", str(row))
        if row.get("uses_target_answer_label") is not False:
            add_failure(failures, "SUBSTRATE_TARGET_LABEL_LEAK", str(row))
        if row.get("no_final_answer_objective") is not True:
            add_failure(failures, "SUBSTRATE_FINAL_ANSWER_OBJECTIVE_MISSING_GUARD", str(row))
    if not validity_rows:
        add_failure(failures, "MISSING_RAM_VALIDITY_ROWS", "ram_validity_report.json")
    else:
        for row in validity_rows:
            for metric in ("ram_validity_score", "reconstruction_mae", "code_entropy", "low_bit_code_utilization"):
                if metric not in row:
                    add_failure(failures, "MISSING_RAM_VALIDITY_METRIC", f"{row.get('system')}:{metric}")

    if not compatibility_rows:
        add_failure(failures, "MISSING_COMPATIBILITY_ROWS", "compatibility_report.json")
    else:
        for row in compatibility_rows:
            for metric in ("producer_write_compatibility", "consumer_read_compatibility", "consumer_read_accuracy", "next_pocket_compatibility_error"):
                if metric not in row:
                    add_failure(failures, "MISSING_COMPATIBILITY_METRIC", f"{row.get('system')}:{metric}")

    mutation_by_system = {row.get("system") for row in repair_rows}
    for system in MUTATION_SYSTEMS:
        if system not in mutation_by_system:
            add_failure(failures, "MISSING_MUTATION_HISTORY", system)
        subset = [row for row in repair_rows if row.get("system") == system]
        accepted = sum(int(row.get("accepted", 0)) for row in subset)
        rejected = sum(int(row.get("rejected", 0)) for row in subset)
        rollback = sum(int(row.get("rollback", 0)) for row in subset)
        if accepted + rejected <= 0 or rejected <= 0 or rollback <= 0:
            add_failure(failures, "MISSING_ACCEPT_REJECT_ROLLBACK", f"{system}: accepted={accepted} rejected={rejected} rollback={rollback}")

    progress_text = (out / "progress.jsonl").read_text(encoding="utf-8")
    for event in ("run_start", "e8d_system_evaluated", "primary_artifacts_written", "deterministic_replay_complete"):
        if event not in progress_text:
            add_failure(failures, "MISSING_PROGRESS_EVENT", event)

    ast_policy(failures)
    result = {"schema_version": "e8d_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
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
