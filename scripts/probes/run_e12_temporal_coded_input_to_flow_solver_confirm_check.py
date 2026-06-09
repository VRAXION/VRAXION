#!/usr/bin/env python3
"""Checker for E12 temporal-coded input to Flow solver confirm."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e12_temporal_coded_input_to_flow_solver_confirm.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E12_TEMPORAL_CODED_INPUT_TO_FLOW_SOLVER_CONFIRM_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E12_TEMPORAL_CODED_INPUT_TO_FLOW_SOLVER_CONFIRM_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e12_input_stream_report.json",
    "e12_system_comparison_report.json",
    "e12_split_robustness_report.json",
    "e12_writeback_safety_report.json",
    "e12_trace_report.json",
    "e12_semantic_leak_report.json",
    "e12_deterministic_replay_report.json",
)
VALID_DECISIONS = (
    "e12_temporal_coded_input_to_flow_solver_confirmed",
    "e12_input_retention_or_temporal_order_failure",
    "e12_output_stream_decode_failure",
    "e12_trace_validity_failure",
    "e12_binding_or_rewrite_failure",
    "e12_noise_or_decoy_repair_failure",
    "e12_codebook_generalization_failure",
    "e12_writeback_safety_failure",
    "e12_semantic_slot_leak_detected",
    "e12_invalid_or_incomplete_run",
)
PRIMARY = "TEMPORAL_FLOW_PRUNED_SCHEDULED_POCKET_PRIMARY"
BASELINE = "OBSERVED_STREAM_DIRECT_BASELINE"
NO_GATE = "TEMPORAL_FLOW_NO_GATE"
FORBIDDEN_RUNTIME_SLOTS = ("ACTION", "DIRECTION", "ENTITY", "CATEGORY", "MOVE", "NORTH", "RED", "OBJECT")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def rate(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return round(float(num) / float(den), 6)


def check_imports(failures: list[dict[str, Any]]) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    allowed = {"__future__", "argparse", "dataclasses", "hashlib", "json", "pathlib", "random", "subprocess", "typing"}
    blocked = {"torch", "tensorflow", "keras", "jax", "numpy", "sklearn", "pandas"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root in blocked:
                add_failure(failures, "NEURAL_OR_EXTERNAL_IMPORT", name)
            elif root and root not in allowed:
                add_failure(failures, "NON_STDLIB_IMPORT_REVIEW_REQUIRED", name)


def check_boundaries(out: Path, failures: list[dict[str, Any]]) -> None:
    blocked = ("A" + "GI", "conscious" + "ness", "production-readiness", "production readiness", "D" + "99", "D" + "100")
    for path in (out / "report.md", *DOCS):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in blocked:
            if token.lower() in text:
                add_failure(failures, "BOUNDARY_TOKEN_FOUND", f"{path}:{token}")


def check_stream_report(stream: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    if stream.get("runtime_receives_semantic_labels") is not False:
        add_failure(failures, "RUNTIME_SEMANTIC_LABEL_FLAG_NOT_FALSE", "e12_input_stream_report.json")
    if stream.get("runtime_receives_text") is not None:
        add_failure(failures, "UNEXPECTED_TEXT_INPUT_FIELD", "e12_input_stream_report.json")
    sample = stream.get("sample_stream", [])
    for idx, tick in enumerate(sample):
        for key in ("clock", "boundary", "separator"):
            if tick.get(key) not in (0, 1):
                add_failure(failures, "NON_BINARY_TICK_FIELD", f"{idx}:{key}")
        for key in ("payload", "noise", "struct"):
            values = tick.get(key, [])
            if not isinstance(values, list) or any(value not in (0, 1) for value in values):
                add_failure(failures, "NON_BINARY_VECTOR_FIELD", f"{idx}:{key}")


def check_gate(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e12_deterministic_replay_report.json")
    stream = load_json(out / "e12_input_stream_report.json")
    semantic = load_json(out / "e12_semantic_leak_report.json")
    systems = aggregate.get("systems", {})
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_SYSTEM_MISMATCH", str(decision.get("primary_system")))
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e12_deterministic_replay_report.json")
    if semantic.get("no_semantic_slot_leak_detected") is not True:
        add_failure(failures, "SEMANTIC_LEAK_FLAG_NOT_TRUE", "e12_semantic_leak_report.json")
    if semantic.get("runtime_receives_forbidden_semantic_slots") is not False:
        add_failure(failures, "FORBIDDEN_RUNTIME_SLOT_FLAG_NOT_FALSE", "e12_semantic_leak_report.json")
    text = json.dumps(stream, sort_keys=True)
    for token in FORBIDDEN_RUNTIME_SLOTS:
        if token in text and "forbidden_slots_checked" not in text:
            add_failure(failures, "FORBIDDEN_SLOT_IN_STREAM_REPORT", token)
    check_stream_report(stream, failures)
    if not all(system in systems for system in (PRIMARY, BASELINE, NO_GATE)):
        add_failure(failures, "MISSING_REQUIRED_SYSTEM", "primary/baseline/no_gate")
        return
    primary = systems[PRIMARY]
    expected = {
        "exact_task_success_rate_at_least_095": primary["exact_task_success_rate"] >= 0.95,
        "output_sequence_accuracy_at_least_098": primary["output_sequence_accuracy"] >= 0.98,
        "trace_validity_at_least_095": primary["trace_validity"] >= 0.95,
        "temporal_order_accuracy_at_least_098": primary["temporal_order_accuracy"] >= 0.98,
        "binding_accuracy_at_least_095": primary["binding_accuracy"] >= 0.95,
        "conditional_control_accuracy_at_least_095": primary["conditional_control_accuracy"] >= 0.95,
        "rewrite_map_accuracy_at_least_095": primary["rewrite_map_accuracy"] >= 0.95,
        "noise_rejection_rate_at_least_090": primary["noise_rejection_rate"] >= 0.90,
        "decoy_rejection_rate_at_least_090": primary["decoy_rejection_rate"] >= 0.90,
        "heldout_codebook_accuracy_at_least_090": primary["heldout_codebook_accuracy"] >= 0.90,
        "randomized_codebook_generalization_at_least_090": primary["randomized_codebook_generalization"] >= 0.90,
        "wrong_writeback_rate_at_most_002": primary["wrong_writeback_rate"] <= 0.02,
        "destructive_overwrite_rate_at_most_002": primary["destructive_overwrite_rate"] <= 0.02,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "no_semantic_slot_leak_detected": primary["no_semantic_slot_leak_detected"] is True,
        "deterministic_replay_passed": replay.get("internal_replay_passed", False) is True,
    }
    reported = aggregate.get("positive_gate", {}).get("checks", {})
    for name, value in expected.items():
        if reported.get(name) is not value:
            add_failure(failures, "POSITIVE_GATE_MATH_MISMATCH", name)
    if aggregate.get("positive_gate", {}).get("passed") is not all(expected.values()):
        add_failure(failures, "POSITIVE_GATE_FLAG_MISMATCH", "aggregate_metrics.json")
    deltas = aggregate.get("positive_gate", {}).get("deltas", {})
    expected_cost_reduction = round(1.0 - rate(primary["cost_per_tick"], systems[NO_GATE]["cost_per_tick"]), 6)
    if deltas.get("cost_reduction_vs_no_gate") != expected_cost_reduction:
        add_failure(failures, "DELTA_COST_MISMATCH", str(deltas.get("cost_reduction_vs_no_gate")))


def check(out: Path, write_summary: bool = False) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            add_failure(failures, "MISSING_ARTIFACT", name)
    if RUNNER.exists():
        check_imports(failures)
    else:
        add_failure(failures, "MISSING_RUNNER", str(RUNNER))
    if not failures:
        check_gate(out, failures)
    check_boundaries(out, failures)
    result = {"schema_version": "e12_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e12_temporal_coded_input_to_flow_solver_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
