#!/usr/bin/env python3
"""Checker for E11B neural baseline inference-cost compare."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e11b_neural_baseline_inference_cost_compare.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E11B_NEURAL_BASELINE_INFERENCE_COST_COMPARE_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E11B_NEURAL_BASELINE_INFERENCE_COST_COMPARE_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e11b_quality_report.json",
    "e11b_cost_model_report.json",
    "e11b_walltime_report.json",
    "e11b_neural_baseline_report.json",
    "e11b_deterministic_replay_report.json",
)
VALID_DECISIONS = (
    "e11b_flow_proxy_cost_advantage_vs_quality_matched_neural_confirmed",
    "e11b_neural_quality_mismatch_no_cost_claim",
    "e11b_flow_proxy_cost_advantage_not_confirmed",
    "e11b_invalid_or_incomplete_run",
)
FLOW = "FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED_PYTHON"
FLOW_BITSET = "FLOW_E10_BITSET_COST_MODEL"
NEURAL_SYSTEMS = (
    "TINY_MLP_ROUTE_ONLY_CONTROLLER",
    "TINY_MLP_TRACE_CONTROLLER",
    "TINY_GRU_TRACE_CONTROLLER",
    "SMALL_TRANSFORMER_TRACE_CONTROLLER",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def check_imports(failures: list[dict[str, Any]]) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "importlib.util",
        "json",
        "pathlib",
        "statistics",
        "subprocess",
        "sys",
        "time",
        "tracemalloc",
        "typing",
    }
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
                add_failure(failures, "EXTERNAL_ML_IMPORT", name)
            elif root and name not in allowed and root not in {"importlib"}:
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
    replay = load_json(out / "e11b_deterministic_replay_report.json")
    neural = load_json(out / "e11b_neural_baseline_report.json")
    wall = load_json(out / "e11b_walltime_report.json")
    systems = aggregate.get("systems", {})
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != FLOW:
        add_failure(failures, "PRIMARY_SYSTEM_MISMATCH", str(decision.get("primary_system")))
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e11b_deterministic_replay_report.json")
    if neural.get("training_run") is not False or neural.get("raw_grid_neural_model") is not False:
        add_failure(failures, "RAW_GRID_OR_TRAINING_CLAIM_NOT_FALSE", "e11b_neural_baseline_report.json")
    required = (FLOW, FLOW_BITSET, *NEURAL_SYSTEMS)
    if not all(system in systems for system in required):
        add_failure(failures, "MISSING_REQUIRED_SYSTEM", ",".join(system for system in required if system not in systems))
        return
    flow = systems[FLOW]
    for system in (FLOW, FLOW_BITSET, *NEURAL_SYSTEMS):
        row = systems[system]
        expected_quality = (
            row["trace_validity"] >= 0.90
            and row["usefulness"] >= 0.85
            and row["useful_writeback_recall"] >= 0.85
            and row["wrong_writeback_rate"] <= 0.05
        )
        if row.get("quality_matched") is not expected_quality:
            add_failure(failures, "QUALITY_MATCHED_FLAG_MISMATCH", system)
    quality_matched = [system for system in NEURAL_SYSTEMS if systems[system]["quality_matched"]]
    cheapest = min(quality_matched, key=lambda system: systems[system]["proxy_ops_per_tick"]) if quality_matched else None
    cheapest_ops = systems[cheapest]["proxy_ops_per_tick"] if cheapest else 0.0
    advantage = round(float(cheapest_ops) / float(flow["proxy_ops_per_tick"]), 6) if cheapest else 0.0
    expected = {
        "flow_quality_matched": flow["quality_matched"] is True,
        "at_least_one_quality_matched_neural": bool(quality_matched),
        "flow_wrong_writeback_zero": flow["wrong_writeback_rate"] == 0.0,
        "flow_destructive_zero": flow["destructive_overwrite_rate"] == 0.0,
        "flow_ops_at_least_3x_lower_than_cheapest_quality_neural": advantage >= 3.0,
        "flow_cost_per_valid_writeback_lower_than_cheapest_quality_neural": bool(cheapest and flow["cost_per_valid_writeback"] < systems[cheapest]["cost_per_valid_writeback"]),
        "bitset_cost_model_lower_than_flow_python": systems[FLOW_BITSET]["proxy_ops_per_tick"] < flow["proxy_ops_per_tick"],
        "no_raw_grid_neural_claim": True,
        "deterministic_replay_passed": replay.get("internal_replay_passed", False) is True,
    }
    reported = aggregate.get("positive_gate", {}).get("checks", {})
    for name, value in expected.items():
        if reported.get(name) is not value:
            add_failure(failures, "POSITIVE_GATE_MATH_MISMATCH", name)
    if aggregate.get("positive_gate", {}).get("passed") is not all(expected.values()):
        add_failure(failures, "POSITIVE_GATE_FLAG_MISMATCH", "aggregate_metrics.json")
    if aggregate.get("positive_gate", {}).get("cheapest_quality_matched_neural") != cheapest:
        add_failure(failures, "CHEAPEST_NEURAL_MISMATCH", str(cheapest))
    if wall.get("skipped") is not True and FLOW not in wall.get("systems", {}):
        add_failure(failures, "MISSING_WALLTIME_FLOW_MEASUREMENT", "e11b_walltime_report.json")


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
    result = {"schema_version": "e11b_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e11b_neural_baseline_inference_cost_compare")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
