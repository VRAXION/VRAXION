#!/usr/bin/env python3
"""Checker for E10 operator-library transfer and noisy-route confirm."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e10_operator_library_transfer_noisy_route_confirm.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E10_OPERATOR_LIBRARY_TRANSFER_AND_NOISY_ROUTE_CONFIRM_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E10_OPERATOR_LIBRARY_TRANSFER_AND_NOISY_ROUTE_CONFIRM_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e10_transfer_report.json",
    "e10_noisy_route_report.json",
    "e10_writeback_safety_report.json",
    "e10_split_robustness_report.json",
    "e10_operator_reuse_report.json",
    "e10_trace_report.json",
    "e10_deterministic_replay_report.json",
)
VALID_DECISIONS = (
    "e10_operator_library_transfer_and_noisy_route_confirmed",
    "e10_noisy_route_repair_insufficient",
    "e10_transfer_trace_validity_failure",
    "e10_writeback_safety_failure",
    "e10_operator_reuse_or_coverage_failure",
    "e10_usefulness_trace_tradeoff_unresolved",
    "e10_invalid_or_incomplete_run",
)
PRIMARY = "TRANSFER_LIBRARY_SCHEDULED_SCHEMA_GATED_PRUNED"
BASELINE = "DIRECT_OVERWRITE_NOISY_ROUTE"
NO_REPAIR = "OBSERVED_ROUTE_SCHEMA_GATED_NO_REPAIR"
NO_GATE = "REUSE_LIBRARY_NOISY_NO_GATE"
EVAL_SPLITS = ("heldout_transfer", "noisy_route", "partial_corruption", "ood_mixture", "adversarial_noise")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


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


def check_gate(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e10_deterministic_replay_report.json")
    splits = load_json(out / "e10_split_robustness_report.json")["split_metrics"]
    transfer = load_json(out / "e10_transfer_report.json")
    systems = aggregate.get("systems", {})
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_SYSTEM_MISMATCH", str(decision.get("primary_system")))
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e10_deterministic_replay_report.json")
    if transfer.get("mutation_discovery_rerun") is not False:
        add_failure(failures, "MUTATION_DISCOVERY_RERUN_NOT_FALSE", "e10_transfer_report.json")
    if not all(system in systems for system in (PRIMARY, BASELINE, NO_REPAIR, NO_GATE)):
        add_failure(failures, "MISSING_REQUIRED_SYSTEM", "primary/baseline/no_repair/no_gate")
        return
    primary = systems[PRIMARY]
    baseline = systems[BASELINE]
    no_repair = systems[NO_REPAIR]
    no_gate = systems[NO_GATE]
    robust = all(splits[PRIMARY][split]["usefulness"] >= 0.78 and splits[PRIMARY][split]["trace_validity"] >= 0.86 for split in EVAL_SPLITS)
    expected = {
        "beats_direct_usefulness": primary["usefulness"] > baseline["usefulness"],
        "beats_direct_trace_validity": primary["trace_validity"] > baseline["trace_validity"],
        "beats_no_repair_usefulness": primary["usefulness"] > no_repair["usefulness"],
        "beats_no_gate_wrong_writeback": primary["wrong_writeback_rate"] < no_gate["wrong_writeback_rate"],
        "trace_validity_at_least_090": primary["trace_validity"] >= 0.90,
        "usefulness_at_least_085": primary["usefulness"] >= 0.85,
        "useful_writeback_recall_at_least_085": primary["useful_writeback_recall"] >= 0.85,
        "wrong_writeback_rate_at_most_005": primary["wrong_writeback_rate"] <= 0.05,
        "destructive_overwrite_rate_at_most_005": primary["destructive_overwrite_rate"] <= 0.05,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "stale_write_rejection_at_least_090": primary["stale_write_rejection_rate"] >= 0.90,
        "route_repair_rate_at_least_080": primary["route_repair_rate"] >= 0.80,
        "noisy_route_false_accept_at_most_005": primary["noisy_route_false_accept_rate"] <= 0.05,
        "transfer_coverage_at_least_085": primary["transfer_coverage"] >= 0.85,
        "operator_reuse_rate_at_least_090": primary["operator_reuse_rate"] >= 0.90,
        "cost_lower_than_direct": primary["cost_per_tick"] < baseline["cost_per_tick"],
        "noisy_transfer_splits_not_collapsed": robust,
        "deterministic_replay_passed": replay.get("internal_replay_passed", False) is True,
        "no_neural_dependency_detected": primary["no_neural_dependency_detected"] is True,
        "no_overclaim_boundary_preserved": primary["no_overclaim_boundary_preserved"] is True,
    }
    reported = aggregate.get("positive_gate", {}).get("checks", {})
    for name, value in expected.items():
        if reported.get(name) is not value:
            add_failure(failures, "POSITIVE_GATE_MATH_MISMATCH", name)
    if aggregate.get("positive_gate", {}).get("passed") is not all(expected.values()):
        add_failure(failures, "POSITIVE_GATE_FLAG_MISMATCH", "aggregate_metrics.json")


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
    result = {"schema_version": "e10_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e10_operator_library_transfer_noisy_route_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
