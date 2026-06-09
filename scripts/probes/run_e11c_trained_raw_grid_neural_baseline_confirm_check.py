#!/usr/bin/env python3
"""Checker for E11C trained raw-grid neural baseline confirm."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e11c_trained_raw_grid_neural_baseline_confirm.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E11C_TRAINED_RAW_GRID_NEURAL_BASELINE_CONFIRM_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E11C_TRAINED_RAW_GRID_NEURAL_BASELINE_CONFIRM_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e11c_training_report.json",
    "e11c_quality_report.json",
    "e11c_cost_report.json",
    "e11c_split_report.json",
    "e11c_deterministic_replay_report.json",
)
VALID_DECISIONS = (
    "e11c_flow_advantage_vs_trained_raw_grid_neural_confirmed",
    "e11c_trained_raw_grid_neural_baseline_not_quality_matched",
    "e11c_trained_raw_grid_neural_beats_or_matches_flow",
    "e11c_flow_quality_failure",
    "e11c_invalid_or_incomplete_run",
)
FLOW = "FLOW_E10_SCHEDULED_SCHEMA_GATED_PRUNED"
SOFTMAX = "TRAINED_RAW_GRID_ROUTE_SOFTMAX"
MLP = "TRAINED_RAW_GRID_ROUTE_MLP"
TRAINED_NEURAL_SYSTEMS = (SOFTMAX, MLP)
SYSTEMS = (FLOW, "OBSERVED_ROUTE_NO_TRAIN_BASELINE", SOFTMAX, MLP)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def quality_matched(row: dict[str, Any]) -> bool:
    return bool(row["trace_validity"] >= 0.90 and row["usefulness"] >= 0.85 and row["useful_writeback_recall"] >= 0.85 and row["wrong_writeback_rate"] <= 0.05)


def check_imports(failures: list[dict[str, Any]]) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    allowed = {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "importlib.util",
        "json",
        "math",
        "pathlib",
        "random",
        "subprocess",
        "sys",
        "time",
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


def expected_decision(gate: dict[str, Any], systems: dict[str, dict[str, Any]]) -> str:
    if not systems[FLOW]["quality_matched"]:
        return "e11c_flow_quality_failure"
    if gate.get("passed") is True:
        return "e11c_flow_advantage_vs_trained_raw_grid_neural_confirmed"
    if not gate["checks"]["at_least_one_trained_raw_grid_neural_quality_matched"]:
        return "e11c_trained_raw_grid_neural_baseline_not_quality_matched"
    if any(systems[system]["quality_matched"] and systems[system]["cost_per_valid_writeback"] <= systems[FLOW]["cost_per_valid_writeback"] for system in TRAINED_NEURAL_SYSTEMS):
        return "e11c_trained_raw_grid_neural_beats_or_matches_flow"
    return "e11c_invalid_or_incomplete_run"


def check_gate(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e11c_deterministic_replay_report.json")
    training = load_json(out / "e11c_training_report.json")
    systems = aggregate.get("systems", {})
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != FLOW:
        add_failure(failures, "PRIMARY_SYSTEM_MISMATCH", str(decision.get("primary_system")))
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e11c_deterministic_replay_report.json")
    if training.get("raw_grid_neural_training") is not True:
        add_failure(failures, "RAW_GRID_NEURAL_TRAINING_NOT_TRUE", "e11c_training_report.json")
    if training.get("detector_evidence_used_by_neural") is not False:
        add_failure(failures, "DETECTOR_EVIDENCE_FLAG_NOT_FALSE", "e11c_training_report.json")
    if not all(system in systems for system in SYSTEMS):
        add_failure(failures, "MISSING_REQUIRED_SYSTEM", ",".join(system for system in SYSTEMS if system not in systems))
        return
    for system in SYSTEMS:
        expected_quality = quality_matched(systems[system])
        if systems[system].get("quality_matched") is not expected_quality:
            add_failure(failures, "QUALITY_MATCHED_FLAG_MISMATCH", system)
    flow = systems[FLOW]
    quality_neural = [system for system in TRAINED_NEURAL_SYSTEMS if systems[system]["quality_matched"]]
    cheapest = min(quality_neural, key=lambda system: systems[system]["proxy_ops_per_tick"]) if quality_neural else None
    expected = {
        "flow_quality_matched": flow["quality_matched"] is True,
        "flow_wrong_writeback_zero": flow["wrong_writeback_rate"] == 0.0,
        "flow_destructive_zero": flow["destructive_overwrite_rate"] == 0.0,
        "at_least_one_trained_raw_grid_neural_quality_matched": bool(quality_neural),
        "flow_cost_lower_than_cheapest_quality_neural": bool(cheapest and flow["proxy_ops_per_tick"] < systems[cheapest]["proxy_ops_per_tick"]),
        "flow_cost_per_valid_writeback_lower_than_cheapest_quality_neural": bool(cheapest and flow["cost_per_valid_writeback"] < systems[cheapest]["cost_per_valid_writeback"]),
        "no_detector_evidence_to_neural": True,
        "deterministic_replay_passed": replay.get("internal_replay_passed", False) is True,
    }
    reported = aggregate.get("positive_gate", {}).get("checks", {})
    for name, value in expected.items():
        if reported.get(name) is not value:
            add_failure(failures, "POSITIVE_GATE_MATH_MISMATCH", name)
    if aggregate.get("positive_gate", {}).get("passed") is not all(expected.values()):
        add_failure(failures, "POSITIVE_GATE_FLAG_MISMATCH", "aggregate_metrics.json")
    expected_label = expected_decision(aggregate["positive_gate"], systems)
    if decision.get("decision") != expected_label:
        add_failure(failures, "DECISION_MISMATCH", f"expected {expected_label}, got {decision.get('decision')}")


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
    result = {"schema_version": "e11c_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e11c_trained_raw_grid_neural_baseline_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
