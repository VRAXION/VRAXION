#!/usr/bin/env python3
"""Validate D128B optimizer backend and metric reality audit artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_REPORTS = [
    "d128b_static_code_inventory.json",
    "d128b_backend_classification_report.json",
    "d128b_claim_vs_code_audit.json",
    "d128b_metric_source_audit.json",
    "d128b_parameter_diff_audit.json",
    "d128b_mutation_algorithm_audit.json",
    "d128b_gradient_backprop_audit.json",
    "d128b_synthetic_harness_audit.json",
    "d128b_deterministic_replay_report.json",
    "aggregate_metrics.json",
    "decision.json",
    "summary.json",
    "report.md",
]

VALID_SYNTHETIC_DECISION = "d128b_synthetic_harness_backend_confirmed"
VALID_NEXT = "D128_CONTROLLED_SYMBOLIC_BRIDGE_FRONTIER_CONSOLIDATION_WITH_BACKEND_BOUNDARY"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(out: Path) -> None:
    missing = [name for name in REQUIRED_REPORTS if not (out / name).exists()]
    assert_true(not missing, f"missing required reports: {missing}")

    inventory = read_json(out / "d128b_static_code_inventory.json")
    backend = read_json(out / "d128b_backend_classification_report.json")
    claims = read_json(out / "d128b_claim_vs_code_audit.json")
    metric_sources = read_json(out / "d128b_metric_source_audit.json")
    params = read_json(out / "d128b_parameter_diff_audit.json")
    mutation = read_json(out / "d128b_mutation_algorithm_audit.json")
    gradient = read_json(out / "d128b_gradient_backprop_audit.json")
    synthetic = read_json(out / "d128b_synthetic_harness_audit.json")
    replay = read_json(out / "d128b_deterministic_replay_report.json")
    aggregate = read_json(out / "aggregate_metrics.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    report = (out / "report.md").read_text(encoding="utf-8").lower()

    assert_true(inventory.get("scanned_file_count", 0) > 0, "static inventory scanned no files")
    assert_true(backend.get("classifications"), "backend classification is absent")
    assert_true("conclusion" in mutation, "mutation audit is absent")
    assert_true("conclusion" in gradient, "gradient audit is absent")
    assert_true(metric_sources.get("metrics"), "metric source audit is absent")
    assert_true(claims.get("tasks"), "claim-vs-code audit is absent")
    assert_true("actual_parameter_diff_found" in params, "parameter diff audit is absent")
    assert_true("conclusion" in synthetic, "synthetic harness audit is absent")

    no_real_optimizer = not backend.get("real_optimizer_detected")
    no_mutation = mutation.get("conclusion") in {"mutation_algorithm_not_present", "mutation_algorithm_present_but_not_invoked"}
    no_gradient = gradient.get("conclusion") in {"no_gradient_backprop_detected", "gradient_backprop_imported_but_not_used"}
    synthetic_confirmed = synthetic.get("conclusion") == "mostly_synthetic_report_harness"

    if no_real_optimizer and no_mutation and no_gradient and synthetic_confirmed:
        assert_true(decision.get("decision") == VALID_SYNTHETIC_DECISION, "final decision does not match synthetic harness evidence")
        assert_true(decision.get("next") == VALID_NEXT, "next task does not match synthetic harness boundary")

    assert_true(aggregate.get("fallback_rows") == 0, "fallback_rows != 0")
    assert_true(aggregate.get("failed_jobs") == [], "failed_jobs non-empty")
    assert_true(summary.get("failed_jobs") == [], "summary failed_jobs non-empty")
    assert_true(replay.get("replay_passed") is True, "deterministic replay report did not pass")
    assert_true(params.get("actual_parameter_diff_found") is False, "parameter diff audit unexpectedly found real diffs")
    assert_true("synthetic" in report or "hardcoded" in report or "deterministic" in report, "report hides synthetic/hardcoded/formulaic nature")
    assert_true("real adapter training" not in report, "report implies real adapter training")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check D128B optimizer backend and metric reality audit artifacts")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validate(args.out)
    print(f"D128B audit check passed for {args.out}")


if __name__ == "__main__":
    main()
