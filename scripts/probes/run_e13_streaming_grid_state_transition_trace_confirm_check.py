#!/usr/bin/env python3
"""Checker for E13 streaming grid state-transition trace confirm."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e13_streaming_grid_state_transition_trace_confirm.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E13_STREAMING_GRID_STATE_TRANSITION_TRACE_CONFIRM_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E13_STREAMING_GRID_STATE_TRANSITION_TRACE_CONFIRM_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e13_search_report.json",
    "e13_dataset_report.json",
    "e13_system_comparison_report.json",
    "e13_trace_accuracy_report.json",
    "e13_noisy_repair_report.json",
    "e13_missing_frame_report.json",
    "e13_heldout_composition_report.json",
    "e13_long_horizon_report.json",
    "e13_ood_grid_report.json",
    "e13_destructive_license_report.json",
    "e13_writeback_safety_report.json",
    "e13_semantic_leak_report.json",
    "e13_deterministic_replay_report.json",
)
VALID_DECISIONS = (
    "e13_streaming_grid_state_transition_trace_confirmed",
    "e13_clean_trace_failure",
    "e13_noisy_trace_repair_failure",
    "e13_missing_frame_repair_failure",
    "e13_heldout_composition_failure",
    "e13_long_horizon_drift_failure",
    "e13_ood_grid_generalization_failure",
    "e13_destructive_license_failure",
    "e13_writeback_safety_failure",
    "e13_semantic_slot_leak_detected",
    "e13_invalid_or_incomplete_run",
)
PRIMARY = "FLOW_GRID_PRUNED_SCHEDULED_POCKET_PRIMARY"
BASELINE = "OBSERVED_FRAME_DIFF_BASELINE"
DIRECT = "DIRECT_OVERWRITE_GRID_BASELINE"
NO_STATE = "NO_INTERNAL_STATE_BASELINE"
ORACLE = "ORACLE_TRACE_REFERENCE"
REQUIRED_SYSTEMS = (
    BASELINE,
    DIRECT,
    NO_STATE,
    ORACLE,
    "FLOW_GRID_GATED_WRITEBACK",
    "FLOW_GRID_TRACE_REPAIR",
    "FLOW_GRID_SCHEDULED_POCKET_PRIMARY",
    PRIMARY,
    "TINY_GRID_MLP_CONTROL",
)
REQUIRED_SPLITS = (
    "train_like",
    "validation",
    "heldout_composition",
    "noisy",
    "adversarial_noise",
    "missing_frame",
    "ood_grid_size",
    "long_horizon",
    "branch_switch",
)
FORBIDDEN_RUNTIME_SLOTS = ("SHIFT_RIGHT", "EXPAND", "CONTRACT", "SPLIT", "MERGE", "OBJECT", "DIRECTION", "ACTION")


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


def check_dataset(dataset: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    if dataset.get("runtime_receives_semantic_labels") is not False:
        add_failure(failures, "RUNTIME_SEMANTIC_LABEL_FLAG_NOT_FALSE", "e13_dataset_report.json")
    if sorted(dataset.get("grid_sizes", [])) != [8, 12, 16, 24]:
        add_failure(failures, "GRID_SIZE_SET_MISMATCH", str(dataset.get("grid_sizes")))
    for split in REQUIRED_SPLITS:
        if split not in dataset.get("splits", []):
            add_failure(failures, "MISSING_DATASET_SPLIT", split)
    for horizon in (1, 3, 6, 12, 24):
        if horizon not in dataset.get("route_lengths", []):
            add_failure(failures, "MISSING_ROUTE_LENGTH", str(horizon))


def check_semantic_report(semantic: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    if semantic.get("runtime_receives_forbidden_semantic_slots") is not False:
        add_failure(failures, "FORBIDDEN_RUNTIME_SLOT_FLAG_NOT_FALSE", "e13_semantic_leak_report.json")
    if semantic.get("no_semantic_slot_leak_detected") is not True:
        add_failure(failures, "SEMANTIC_LEAK_FLAG_NOT_TRUE", "e13_semantic_leak_report.json")
    runtime_text = json.dumps(semantic.get("primary_runtime_config", {}), sort_keys=True).upper()
    for token in FORBIDDEN_RUNTIME_SLOTS:
        if token in runtime_text:
            add_failure(failures, "FORBIDDEN_SLOT_IN_RUNTIME_CONFIG", token)


def expected_gate(aggregate: dict[str, Any], replay_passed: bool) -> dict[str, bool]:
    systems = aggregate["systems"]
    splits = aggregate["split_metrics"]
    primary = systems[PRIMARY]
    direct = systems[DIRECT]
    noisy_beats = (
        splits[PRIMARY]["noisy"]["final_grid_similarity"] > splits[BASELINE]["noisy"]["final_grid_similarity"]
        and splits[PRIMARY]["adversarial_noise"]["final_grid_similarity"] > splits[BASELINE]["adversarial_noise"]["final_grid_similarity"]
        and splits[PRIMARY]["missing_frame"]["operator_trace_exact_accuracy"] > splits[BASELINE]["missing_frame"]["operator_trace_exact_accuracy"]
    )
    no_state_beats = (
        splits[PRIMARY]["noisy"]["operator_trace_exact_accuracy"] > splits[NO_STATE]["noisy"]["operator_trace_exact_accuracy"]
        and splits[PRIMARY]["long_horizon"]["operator_trace_exact_accuracy"] > splits[NO_STATE]["long_horizon"]["operator_trace_exact_accuracy"]
    )
    direct_beats = primary["wrong_writeback_rate"] < direct["wrong_writeback_rate"] and primary["destructive_overwrite_rate"] < direct["destructive_overwrite_rate"] and primary["trace_validity"] > direct["trace_validity"]
    return {
        "final_grid_exact_accuracy_at_least_095": primary["final_grid_exact_accuracy"] >= 0.95,
        "final_grid_similarity_at_least_098": primary["final_grid_similarity"] >= 0.98,
        "operator_trace_exact_accuracy_at_least_090": primary["operator_trace_exact_accuracy"] >= 0.90,
        "per_step_operator_accuracy_at_least_095": primary["per_step_operator_accuracy"] >= 0.95,
        "trace_validity_at_least_095": primary["trace_validity"] >= 0.95,
        "internal_state_consistency_at_least_095": primary["internal_state_consistency"] >= 0.95,
        "noisy_repair_rate_at_least_090": primary["noisy_repair_rate"] >= 0.90,
        "missing_frame_repair_rate_at_least_085": primary["missing_frame_repair_rate"] >= 0.85,
        "decoy_rejection_rate_at_least_090": primary["decoy_rejection_rate"] >= 0.90,
        "heldout_composition_accuracy_at_least_090": primary["heldout_composition_accuracy"] >= 0.90,
        "long_horizon_survival_rate_at_least_085": primary["long_horizon_survival_rate"] >= 0.85,
        "ood_grid_generalization_at_least_085": primary["ood_grid_generalization"] >= 0.85,
        "branch_switch_accuracy_at_least_095": primary["branch_switch_accuracy"] >= 0.95,
        "licensed_destructive_accept_rate_at_least_090": primary["licensed_destructive_accept_rate"] >= 0.90,
        "unlicensed_destructive_reject_rate_at_least_095": primary["unlicensed_destructive_reject_rate"] >= 0.95,
        "wrong_writeback_rate_at_most_002": primary["wrong_writeback_rate"] <= 0.02,
        "destructive_overwrite_rate_at_most_002": primary["destructive_overwrite_rate"] <= 0.02,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "temporal_drift_not_worse_than_direct": primary["temporal_drift_rate"] <= direct["temporal_drift_rate"],
        "drift_slope_not_explosive": primary["drift_slope_explosive"] is False,
        "beats_observed_on_noisy_missing_adversarial": noisy_beats,
        "beats_no_internal_state_on_noisy_and_long": no_state_beats,
        "beats_direct_on_safety_and_trace": direct_beats,
        "deterministic_replay_passed": replay_passed,
        "no_semantic_slot_leak_detected": primary["no_semantic_slot_leak_detected"] is True,
        "no_neural_dependency_detected": primary["no_neural_dependency_detected"] is True,
    }


def check_gate(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e13_deterministic_replay_report.json")
    dataset = load_json(out / "e13_dataset_report.json")
    semantic = load_json(out / "e13_semantic_leak_report.json")
    systems = aggregate.get("systems", {})
    splits = aggregate.get("split_metrics", {})
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_SYSTEM_MISMATCH", str(decision.get("primary_system")))
    if decision.get("primary_system") == ORACLE:
        add_failure(failures, "ORACLE_AS_PRIMARY", ORACLE)
    if decision.get("decision") == "e13_streaming_grid_state_transition_trace_confirmed" and decision.get("next") != "E14_REGION_AWARE_PARALLEL_POCKET_SCHEDULER_CONFIRM":
        add_failure(failures, "NEXT_MISMATCH_FOR_CONFIRMED", str(decision.get("next")))
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e13_deterministic_replay_report.json")
    check_dataset(dataset, failures)
    check_semantic_report(semantic, failures)
    for system in REQUIRED_SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_REQUIRED_SYSTEM", system)
    for system in (PRIMARY, BASELINE, DIRECT, NO_STATE):
        for split in REQUIRED_SPLITS:
            if system not in splits or split not in splits[system]:
                add_failure(failures, "MISSING_SPLIT_METRICS", f"{system}:{split}")
    if failures:
        return
    expected = expected_gate(aggregate, replay.get("internal_replay_passed", False))
    reported = aggregate.get("positive_gate", {}).get("checks", {})
    for name, value in expected.items():
        if reported.get(name) is not value:
            add_failure(failures, "POSITIVE_GATE_MATH_MISMATCH", name)
    if aggregate.get("positive_gate", {}).get("passed") is not all(expected.values()):
        add_failure(failures, "POSITIVE_GATE_FLAG_MISMATCH", "aggregate_metrics.json")
    primary = systems[PRIMARY]
    deltas = aggregate.get("positive_gate", {}).get("deltas", {})
    expected_cost_reduction = round(1.0 - rate(primary["cost_per_tick"], systems["FLOW_GRID_TRACE_REPAIR"]["cost_per_tick"]), 6)
    if deltas.get("cost_reduction_vs_trace_repair") != expected_cost_reduction:
        add_failure(failures, "DELTA_COST_MISMATCH", str(deltas.get("cost_reduction_vs_trace_repair")))
    expected_wrong_reduction = round(1.0 - rate(primary["wrong_writeback_rate"], systems[DIRECT]["wrong_writeback_rate"]), 6)
    if deltas.get("wrong_writeback_reduction_vs_direct") != expected_wrong_reduction:
        add_failure(failures, "DELTA_WRITEBACK_MISMATCH", str(deltas.get("wrong_writeback_reduction_vs_direct")))


def check(out: Path, write_summary: bool = False) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        if not (out / name).exists():
            add_failure(failures, "MISSING_ARTIFACT", name)
    for doc in DOCS:
        if not doc.exists():
            add_failure(failures, "MISSING_DOC", str(doc))
    if RUNNER.exists():
        check_imports(failures)
    else:
        add_failure(failures, "MISSING_RUNNER", str(RUNNER))
    if not failures:
        check_gate(out, failures)
    check_boundaries(out, failures)
    result = {"schema_version": "e13_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e13_streaming_grid_state_transition_trace_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
