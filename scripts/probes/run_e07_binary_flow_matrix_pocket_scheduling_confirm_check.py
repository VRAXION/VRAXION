#!/usr/bin/env python3
"""Checker for E07 Binary Flow Matrix Pocket Scheduling Confirm."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e07_binary_flow_matrix_pocket_scheduling_confirm.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E07_BINARY_FLOW_MATRIX_POCKET_SCHEDULING_CONFIRM_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E07_BINARY_FLOW_MATRIX_POCKET_SCHEDULING_CONFIRM_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e07_search_report.json",
    "e07_snapshot_vs_rollout_report.json",
    "e07_default_vs_complex_scheduling_report.json",
    "e07_common_matrix_language_report.json",
    "e07_branch_contamination_report.json",
    "e07_temporal_stability_report.json",
    "e07_deterministic_replay_report.json",
)
VALID_DECISIONS = (
    "e07_binary_flow_matrix_pocket_scheduling_confirmed",
    "e07_snapshot_selection_temporal_failure_detected",
    "e07_trigger_policy_too_conservative",
    "e07_branch_contamination_not_fixed",
    "e07_common_matrix_language_contract_failure",
    "e07_invalid_or_incomplete_run",
)
DEFAULT_ARMS = (
    "DEFAULT_ALWAYS_ON_TRIGGERED_COMPLEX_GATED",
    "DEFAULT_ADAPTIVE_TRIGGERED_COMPLEX_GATED",
)
BASELINE_ARM = "ALL_COMPLEX_ALWAYS_NO_GATE"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def check_runner_imports(failures: list[dict[str, Any]]) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "pathlib",
        "random",
        "subprocess",
        "typing",
    }
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
    claim_tokens = ("A" + "GI", "conscious" + "ness", "production-readiness", "production readiness")
    route_tokens = ("D" + "99", "D" + "100")
    paths = [out / "report.md", *DOCS]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in claim_tokens:
            if token.lower() in text:
                add_failure(failures, "BROAD_CLAIM_TOKEN_FOUND", f"{path}:{token}")
        for token in route_tokens:
            if token.lower() in text and "recurrent" in text:
                add_failure(failures, "UNRELATED_ROUTING_CLAIM_TOKEN_FOUND", f"{path}:{token}")


def check_positive_gate(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e07_deterministic_replay_report.json")
    common = load_json(out / "e07_common_matrix_language_report.json")
    snapshot = load_json(out / "e07_snapshot_vs_rollout_report.json")
    search = load_json(out / "e07_search_report.json")
    branch = load_json(out / "e07_branch_contamination_report.json")

    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if not decision.get("deterministic_replay_passed", False):
        add_failure(failures, "DECISION_REPLAY_NOT_TRUE", "decision.json")
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e07_deterministic_replay_report.json")
    if search.get("equivalent_existing_probe_found") is not False:
        add_failure(failures, "SEARCH_EQUIVALENT_FLAG_UNEXPECTED", "expected no equivalent existing E07 probe")

    arms = aggregate.get("arms", {})
    for arm in (BASELINE_ARM, *DEFAULT_ARMS):
        if arm not in arms:
            add_failure(failures, "MISSING_ARM_METRICS", arm)
            return
    best = decision.get("best_default_triggered_gated_arm")
    if best not in DEFAULT_ARMS:
        add_failure(failures, "BEST_DEFAULT_ARM_INVALID", str(best))
        return

    baseline = arms[BASELINE_ARM]
    chosen = arms[best]
    expected_checks = {
        "final_state_accuracy_improved_or_preserved": chosen["final_state_accuracy"] >= baseline["final_state_accuracy"],
        "false_positive_rate_lower": chosen["false_positive_rate"] < baseline["false_positive_rate"],
        "wrong_commit_rate_reduced_50pct": chosen["wrong_commit_rate"] <= baseline["wrong_commit_rate"] * 0.5,
        "destructive_overwrite_rate_reduced_70pct": chosen["destructive_overwrite_rate"] <= baseline["destructive_overwrite_rate"] * 0.3,
        "branch_contamination_rate_reduced_90pct": chosen["branch_contamination_rate"] <= baseline["branch_contamination_rate"] * 0.1,
        "avg_cost_per_tick_reduced_30pct": chosen["avg_cost_per_tick"] <= baseline["avg_cost_per_tick"] * 0.7,
        "useful_update_recall_at_least_075": chosen["useful_update_recall"] >= 0.75,
        "deterministic_replay_passed": replay.get("internal_replay_passed", False) is True,
    }
    reported_checks = aggregate.get("positive_gate", {}).get("checks", {})
    for name, expected in expected_checks.items():
        if reported_checks.get(name) is not expected:
            add_failure(failures, "POSITIVE_GATE_MATH_MISMATCH", name)
    if aggregate.get("positive_gate", {}).get("passed") is not all(expected_checks.values()):
        add_failure(failures, "POSITIVE_GATE_PASS_FLAG_MISMATCH", "aggregate_metrics.json")

    if common.get("contract_passed") is not True:
        add_failure(failures, "COMMON_LANGUAGE_CONTRACT_FAILED", "e07_common_matrix_language_report.json")
    for arm, count in common.get("direct_dialect_mutation_commits_by_arm", {}).items():
        if arm != BASELINE_ARM and count != 0:
            add_failure(failures, "DIRECT_DIALECT_COMMITTED_IN_GATED_ARM", f"{arm}:{count}")
    if snapshot.get("snapshot_temporal_failure_detected") is not True:
        add_failure(failures, "SNAPSHOT_TEMPORAL_FAILURE_NOT_DETECTED", "e07_snapshot_vs_rollout_report.json")
    if snapshot.get("rollout_more_stable_over_time") is not True:
        add_failure(failures, "ROLLOUT_STABILITY_NOT_CONFIRMED", "e07_snapshot_vs_rollout_report.json")
    if branch.get("positive_gate_branch_reduction", 0.0) < 0.9:
        add_failure(failures, "BRANCH_REDUCTION_GATE_FAILED", str(branch.get("positive_gate_branch_reduction")))


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
    result = {
        "schema_version": "e07_checker_result_v1",
        "out": str(out),
        "failure_count": len(failures),
        "failures": failures,
    }
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e07_binary_flow_matrix_pocket_scheduling_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
