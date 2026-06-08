#!/usr/bin/env python3
"""Checker for E08 Flow Matrix Writeback Schema Confirm."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e08_flow_matrix_writeback_schema_confirm.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E08_FLOW_MATRIX_WRITEBACK_SCHEMA_CONFIRM_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E08_FLOW_MATRIX_WRITEBACK_SCHEMA_CONFIRM_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e08_search_report.json",
    "e08_writeback_schema_report.json",
    "e08_stress_cases_report.json",
    "e08_gate_commit_report.json",
    "e08_rollback_report.json",
    "e08_branch_contamination_report.json",
    "e08_temporal_stability_report.json",
    "e08_deterministic_replay_report.json",
)
VALID_DECISIONS = (
    "e08_flow_matrix_writeback_schema_confirmed",
    "e08_direct_overwrite_remains_best",
    "e08_common_schema_not_sufficient",
    "e08_gate_too_conservative",
    "e08_branch_contamination_not_fixed",
    "e08_stale_write_rollback_failure",
    "e08_trace_validity_failure",
    "e08_invalid_or_incomplete_run",
)
BASELINE = "DIRECT_OVERWRITE_BASELINE"
POSITIVE_CANDIDATES = ("COMMON_SCHEMA_GATED_WITH_ROLLBACK", "REGION_OPERATOR_SCHEMA")
SHARED_SCHEMA_ARMS = (
    "COMMON_SCHEMA_NO_GATE",
    "COMMON_SCHEMA_GATED",
    "COMMON_SCHEMA_GATED_WITH_ROLLBACK",
    "REGION_OPERATOR_SCHEMA",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def check_runner_imports(failures: list[dict[str, Any]]) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    allowed = {"__future__", "argparse", "dataclasses", "hashlib", "json", "pathlib", "random", "subprocess", "typing"}
    blocked = {"torch", "tensorflow", "keras", "jax", "numpy", "sklearn", "pandas"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in blocked:
                    add_failure(failures, "NEURAL_OR_EXTERNAL_IMPORT", alias.name)
                elif root not in allowed:
                    add_failure(failures, "NON_STDLIB_IMPORT_REVIEW_REQUIRED", alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in blocked:
                add_failure(failures, "NEURAL_OR_EXTERNAL_IMPORT", node.module or "")
            elif root and root not in allowed:
                add_failure(failures, "NON_STDLIB_IMPORT_REVIEW_REQUIRED", node.module or "")


def check_claim_boundaries(out: Path, failures: list[dict[str, Any]]) -> None:
    blocked_tokens = ("A" + "GI", "conscious" + "ness", "production-readiness", "production readiness", "D" + "99", "D" + "100")
    for path in (out / "report.md", *DOCS):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in blocked_tokens:
            if token.lower() in text:
                add_failure(failures, "BOUNDARY_TOKEN_FOUND", f"{path}:{token}")


def check_positive_gate(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e08_deterministic_replay_report.json")
    schema = load_json(out / "e08_writeback_schema_report.json")
    stress = load_json(out / "e08_stress_cases_report.json")
    search = load_json(out / "e08_search_report.json")

    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if not decision.get("deterministic_replay_passed", False):
        add_failure(failures, "DECISION_REPLAY_NOT_TRUE", "decision.json")
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e08_deterministic_replay_report.json")
    if search.get("equivalent_existing_probe_found") is not False:
        add_failure(failures, "SEARCH_EQUIVALENT_FLAG_UNEXPECTED", "expected no equivalent E08 probe")
    if stress.get("all_required_cases_present") is not True:
        add_failure(failures, "MISSING_STRESS_CASE", "e08_stress_cases_report.json")
    if schema.get("shared_schema_zero_violations") is not True:
        add_failure(failures, "SHARED_SCHEMA_VIOLATION", "e08_writeback_schema_report.json")

    arms = aggregate.get("arms", {})
    if BASELINE not in arms:
        add_failure(failures, "MISSING_BASELINE_METRICS", BASELINE)
        return
    best = decision.get("best_schema_arm")
    if best not in POSITIVE_CANDIDATES:
        add_failure(failures, "BEST_SCHEMA_ARM_INVALID", str(best))
        return
    baseline = arms[BASELINE]
    chosen = arms[best]
    shared_zero = all(arms[arm]["schema_violation_rate"] == 0.0 for arm in SHARED_SCHEMA_ARMS)
    stale_safe = chosen["stale_write_rejection_rate"] >= 0.8 or chosen["rollback_success_rate"] >= 0.8
    expected_checks = {
        "final_state_accuracy_improved_or_preserved": chosen["final_state_accuracy"] >= baseline["final_state_accuracy"],
        "useful_writeback_recall_at_least_075": chosen["useful_writeback_recall"] >= 0.75,
        "destructive_overwrite_rate_reduced_70pct": chosen["destructive_overwrite_rate"] <= baseline["destructive_overwrite_rate"] * 0.3,
        "branch_contamination_rate_reduced_90pct": chosen["branch_contamination_rate"] <= baseline["branch_contamination_rate"] * 0.1,
        "wrong_writeback_rate_reduced_50pct": chosen["wrong_writeback_rate"] <= baseline["wrong_writeback_rate"] * 0.5,
        "shared_schema_arms_schema_violation_zero": shared_zero,
        "trace_validity_higher_than_direct": chosen["trace_validity"] > baseline["trace_validity"],
        "stale_writes_rejected_or_rolled_back": stale_safe,
        "temporal_drift_rate_lower_than_direct": chosen["temporal_drift_rate"] < baseline["temporal_drift_rate"],
        "deterministic_replay_passed": replay.get("internal_replay_passed", False) is True,
        "no_neural_dependency_detected": chosen["no_neural_dependency_detected"] is True,
        "no_overclaim_boundary_preserved": chosen["no_overclaim_boundary_preserved"] is True,
    }
    reported = aggregate.get("positive_gate", {}).get("checks", {})
    for name, expected in expected_checks.items():
        if reported.get(name) is not expected:
            add_failure(failures, "POSITIVE_GATE_MATH_MISMATCH", name)
    if aggregate.get("positive_gate", {}).get("passed") is not all(expected_checks.values()):
        add_failure(failures, "POSITIVE_GATE_PASS_FLAG_MISMATCH", "aggregate_metrics.json")


def check(out: Path, write_summary: bool = False) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            add_failure(failures, "MISSING_ARTIFACT", name)
    if RUNNER.exists():
        check_runner_imports(failures)
    else:
        add_failure(failures, "MISSING_RUNNER", str(RUNNER))
    if not failures:
        check_positive_gate(out, failures)
    check_claim_boundaries(out, failures)
    result = {"schema_version": "e08_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e08_flow_matrix_writeback_schema_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
