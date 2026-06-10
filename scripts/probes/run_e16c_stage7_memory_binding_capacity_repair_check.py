#!/usr/bin/env python3
"""Checker for E16C Stage 7 memory binding capacity repair probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e16c_stage7_memory_binding_capacity_repair.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E16C_STAGE7_MEMORY_BINDING_CAPACITY_REPAIR_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E16C_STAGE7_MEMORY_BINDING_CAPACITY_REPAIR_RESULT.md",
)
PRIMARY = "MUTATION_TRAINED_PRUNED_MEMORY_POLICY_PRIMARY"
UNPRUNED = "MUTATION_TRAINED_MEMORY_POLICY_PRIMARY"
BASELINE = "E16C_BASELINE_STAGE7_POLICY"
NO_GATE = "LAST_WRITE_MEMORY_NO_GATE"
BOUNDARY = (
    "This is a deterministic synthetic controlled text-flow Stage 7 memory binding repair probe. "
    "It tests targeted mutation/search over memory policies. It does not prove general natural-language AI."
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e16c_stage7_search_report.json",
    "e16c_stage7_memory_policy_report.json",
    "e16c_stage7_training_curve_report.json",
    "e16c_stage7_capacity_sweep_report.json",
    "e16c_stage7_system_comparison_report.json",
    "e16c_stage7_task_family_report.json",
    "e16c_stage7_ablation_report.json",
    "e16c_stage7_trace_validity_report.json",
    "e16c_stage7_writeback_safety_report.json",
    "e16c_stage7_heldout_generalization_report.json",
    "e16c_stage7_downstream_stage8_probe_report.json",
    "e16c_stage7_semantic_macro_leak_audit_report.json",
    "e16c_stage7_deterministic_replay_report.json",
    "e16c_stage7_boundary_claims_report.json",
    "e16c_stage7_next_recommendation.json",
)
REQUIRED_SYSTEMS = (
    BASELINE,
    NO_GATE,
    "VALID_LAST_MEMORY",
    "MAJORITY_MEMORY_NO_ABSTAIN",
    "FIXED_SLOT_FIFO_MEMORY",
    "FIXED_SLOT_LRU_MEMORY",
    "KEY_ADDRESSED_MEMORY_POLICY",
    UNPRUNED,
    PRIMARY,
    "NO_MEMORY_SLOTS_ABLATION",
    "LOW_MEMORY_CAPACITY_ABLATION",
    "NO_STALE_REJECTION_ABLATION",
    "NO_REPAIR_EVIDENCE_ABLATION",
    "NO_AMBIGUITY_ABSTAIN_ABLATION",
    "NO_NESTED_RESOLUTION_ABLATION",
)
VALID_DECISIONS = (
    "e16c_stage7_memory_binding_capacity_repair_confirmed",
    "e16c_stage7_memory_binding_capacity_repair_partial",
    "e16c_stage7_memory_binding_capacity_repair_failed",
    "e16c_stage7_memory_binding_capacity_repair_invalid_or_incomplete",
)
GATE_THRESHOLDS = {
    "multi_sentence_binding_accuracy": (0.75, "min"),
    "long_horizon_recall": (0.75, "min"),
    "ambiguous_abstain_accuracy": (0.80, "min"),
    "nested_depth2_accuracy": (0.75, "min"),
    "nested_depth3_accuracy": (0.65, "min"),
    "capacity_pressure_accuracy": (0.70, "min"),
    "stale_update_rejection_rate": (0.85, "min"),
    "corrupt_then_repair_success_rate": (0.80, "min"),
    "distractor_gap_survival": (0.80, "min"),
    "trace_validity": (0.95, "min"),
    "wrong_writeback_rate": (0.02, "max"),
    "destructive_overwrite_rate": (0.02, "max"),
    "branch_contamination_rate": (0.0, "eq"),
    "semantic_slot_leak_detected": (False, "eq"),
    "macro_leak_detected": (False, "eq"),
    "privileged_control_selected_as_primary": (False, "eq"),
    "deterministic_replay_passed": (True, "eq"),
    "checker_failure_count": (0, "eq"),
}
ALLOWED_MICRO_OPS = {
    "READ_POS",
    "WRITE_POS",
    "COPY_POS",
    "COMPARE_EQ",
    "IF_EQ",
    "ROUTE_TOKEN",
    "OPEN_MEMORY_SLOT",
    "WRITE_MEMORY_SLOT",
    "READ_MEMORY_SLOT",
    "CLEAR_MEMORY_SLOT",
    "MEMORY_SLOT_SCORE",
    "TRACE_CHECK",
    "GATED_COMMIT",
    "ABSTAIN_OUTPUT",
    "REPAIR_COMMIT",
}
FORBIDDEN_MACROS = {
    "BIND",
    "QUERY",
    "MEMORY_LOOKUP_MACRO",
    "KEY_VALUE_BIND_MACRO",
    "REVERSE",
    "MAP",
    "FILTER",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


def rounded(value: float) -> float:
    return round(float(value), 6)


def check_imports(failures: list[dict[str, Any]]) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    allowed = {"__future__", "argparse", "hashlib", "json", "pathlib", "random", "typing"}
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


def expected_gate(primary: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key, (threshold, mode) in GATE_THRESHOLDS.items():
        value = primary.get(key)
        if mode == "min":
            checks[f"{key}_at_least_{threshold}"] = value >= threshold
        elif mode == "max":
            checks[f"{key}_at_most_{threshold}"] = value <= threshold
        else:
            checks[f"{key}_equals_{threshold}"] = value == threshold
    return checks


def expected_decision(primary: dict[str, Any], gate_passed: bool) -> str:
    if gate_passed:
        return "e16c_stage7_memory_binding_capacity_repair_confirmed"
    materially_improved = (
        primary.get("delta_binding_accuracy", 0.0) >= 0.05
        or primary.get("delta_long_horizon_recall", 0.0) >= 0.05
        or primary.get("trace_validity", 0.0) > 0.928
    )
    if materially_improved:
        return "e16c_stage7_memory_binding_capacity_repair_partial"
    return "e16c_stage7_memory_binding_capacity_repair_failed"


def check_policy_ops(policy_report: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    if policy_report.get("macro_free") is not True:
        add_failure(failures, "POLICY_REPORT_MACRO_FREE_NOT_TRUE", "e16c_stage7_memory_policy_report.json")
    for group_name in ("discovered_policies", "pruned_policies"):
        policies = policy_report.get(group_name, [])
        if not policies:
            add_failure(failures, "EMPTY_POLICY_GROUP", group_name)
        for policy in policies:
            ops = policy.get("micro_program", [])
            if not ops:
                add_failure(failures, "POLICY_WITHOUT_MICRO_PROGRAM", str(policy.get("policy_id")))
            for op in ops:
                if op not in ALLOWED_MICRO_OPS:
                    add_failure(failures, "NON_MICRO_OP_IN_POLICY", f"{group_name}:{op}")
                if op in FORBIDDEN_MACROS:
                    add_failure(failures, "FORBIDDEN_MACRO_IN_POLICY", f"{group_name}:{op}")
    primary_policy = policy_report.get("primary_policy", {})
    for op in primary_policy.get("micro_program", []):
        if op in FORBIDDEN_MACROS or op not in ALLOWED_MICRO_OPS:
            add_failure(failures, "PRIMARY_POLICY_BAD_OP", str(op))
    if policy_report.get("primary_is_privileged_control") is not False:
        add_failure(failures, "PRIMARY_POLICY_PRIVILEGED", "e16c_stage7_memory_policy_report.json")


def check_boundary(out: Path, failures: list[dict[str, Any]]) -> None:
    paths = [out / "report.md", out / "e16c_stage7_boundary_claims_report.json", *DOCS]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if BOUNDARY not in text:
            add_failure(failures, "BOUNDARY_TEXT_MISSING", str(path))
        scrubbed = text.replace(BOUNDARY, "")
        lower = scrubbed.lower()
        blocked = ("agi", "consciousness", "open-natural-language", "open natural language", "general natural-language", "general natural language", "d99", "d100")
        for token in blocked:
            if token in lower:
                add_failure(failures, "BROAD_CLAIM_TOKEN_FOUND", f"{path}:{token}")


def check_reports(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    summary = load_json(out / "summary.json")
    policy_report = load_json(out / "e16c_stage7_memory_policy_report.json")
    sweep = load_json(out / "e16c_stage7_capacity_sweep_report.json")
    ablation = load_json(out / "e16c_stage7_ablation_report.json")
    semantic = load_json(out / "e16c_stage7_semantic_macro_leak_audit_report.json")
    replay = load_json(out / "e16c_stage7_deterministic_replay_report.json")
    writeback = load_json(out / "e16c_stage7_writeback_safety_report.json")
    stage8 = load_json(out / "e16c_stage7_downstream_stage8_probe_report.json")
    systems = aggregate.get("systems", {})
    for system in REQUIRED_SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_SYSTEM", system)
    if PRIMARY not in systems or BASELINE not in systems:
        return
    primary = systems[PRIMARY]
    baseline = systems[BASELINE]
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_SYSTEM_MISMATCH", str(decision.get("primary_system")))
    if summary.get("primary_system") != PRIMARY:
        add_failure(failures, "SUMMARY_PRIMARY_MISMATCH", str(summary.get("primary_system")))
    if primary.get("baseline_stage7_binding_accuracy") != baseline.get("multi_sentence_binding_accuracy"):
        add_failure(failures, "BASELINE_BINDING_NOT_RECORDED", "baseline_stage7_binding_accuracy")
    if primary.get("baseline_long_horizon_recall") != baseline.get("long_horizon_recall"):
        add_failure(failures, "BASELINE_RECALL_NOT_RECORDED", "baseline_long_horizon_recall")
    if primary.get("delta_binding_accuracy") != rounded(primary.get("repaired_stage7_binding_accuracy") - baseline.get("multi_sentence_binding_accuracy")):
        add_failure(failures, "DELTA_BINDING_MISMATCH", str(primary.get("delta_binding_accuracy")))
    if primary.get("delta_long_horizon_recall") != rounded(primary.get("repaired_long_horizon_recall") - baseline.get("long_horizon_recall")):
        add_failure(failures, "DELTA_RECALL_MISMATCH", str(primary.get("delta_long_horizon_recall")))

    expected = expected_gate(primary)
    reported = aggregate.get("positive_gate", {}).get("checks", {})
    for key, value in expected.items():
        if reported.get(key) is not value:
            add_failure(failures, "GATE_CHECK_MISMATCH", key)
    gate_passed = all(expected.values())
    if aggregate.get("positive_gate", {}).get("passed") is not gate_passed:
        add_failure(failures, "GATE_PASSED_FLAG_MISMATCH", "aggregate_metrics.json")
    if decision.get("positive_gate_passed") is not gate_passed:
        add_failure(failures, "DECISION_GATE_FLAG_MISMATCH", "decision.json")
    expected_label = expected_decision(primary, gate_passed)
    if decision.get("decision") != expected_label:
        add_failure(failures, "DECISION_MATH_MISMATCH", f"{decision.get('decision')} != {expected_label}")
    if expected_label == "e16c_stage7_memory_binding_capacity_repair_confirmed" and decision.get("next") != "E16C_STAGE8_NOISY_MULTI_SENTENCE_REPAIR_CONFIRM":
        add_failure(failures, "NEXT_MISMATCH_FOR_CONFIRMED", str(decision.get("next")))

    rows = sweep.get("rows", [])
    if [row.get("slot_count") for row in rows] != [1, 2, 3, 4, 6, 8, 12]:
        add_failure(failures, "CAPACITY_SWEEP_SLOT_SET_MISMATCH", str([row.get("slot_count") for row in rows]))
    passing = [row["slot_count"] for row in rows if row.get("stage7_gate_cleared")]
    first_passing = min(passing) if passing else None
    if sweep.get("first_passing_memory_slot_count") != first_passing:
        add_failure(failures, "FIRST_PASSING_SLOT_MISMATCH", str(sweep.get("first_passing_memory_slot_count")))
    if primary.get("first_passing_memory_slot_count") != first_passing:
        add_failure(failures, "PRIMARY_FIRST_PASSING_SLOT_MISMATCH", str(primary.get("first_passing_memory_slot_count")))
    if not rows:
        add_failure(failures, "CAPACITY_SWEEP_EMPTY", "e16c_stage7_capacity_sweep_report.json")
    for row in rows:
        for key in ("binding_accuracy", "nested_depth2_accuracy", "nested_depth3_accuracy", "long_horizon_recall", "capacity_pressure_accuracy", "cost_per_episode"):
            if key not in row:
                add_failure(failures, "CAPACITY_SWEEP_FIELD_MISSING", key)

    expectations = ablation.get("expectations", {})
    for key in (
        "no_memory_slots_fails_stage7",
        "low_capacity_fails_capacity_pressure",
        "no_stale_rejection_fails_stale_family",
        "no_repair_evidence_fails_repair_family",
        "no_ambiguity_abstain_wrong_commit_risk",
        "no_nested_resolution_fails_depth2_depth3",
    ):
        if expectations.get(key) is not True:
            add_failure(failures, "ABLATION_EXPECTATION_FAILED", key)
    if semantic.get("semantic_slot_leak_detected") is not False:
        add_failure(failures, "SEMANTIC_LEAK_FLAG_NOT_FALSE", "e16c_stage7_semantic_macro_leak_audit_report.json")
    if semantic.get("macro_leak_detected") is not False:
        add_failure(failures, "MACRO_LEAK_FLAG_NOT_FALSE", "e16c_stage7_semantic_macro_leak_audit_report.json")
    if semantic.get("runtime_receives_macro_bind_or_query") is not False:
        add_failure(failures, "RUNTIME_MACRO_BIND_QUERY_FLAG_NOT_FALSE", "e16c_stage7_semantic_macro_leak_audit_report.json")
    if semantic.get("privileged_control_selected_as_primary") is not False:
        add_failure(failures, "PRIVILEGED_PRIMARY_FLAG_NOT_FALSE", "e16c_stage7_semantic_macro_leak_audit_report.json")
    if replay.get("internal_replay_passed") is not True:
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e16c_stage7_deterministic_replay_report.json")
    if writeback.get("no_gate_worse_trace") is not True:
        add_failure(failures, "NO_GATE_TRACE_CONTRAST_MISSING", "e16c_stage7_writeback_safety_report.json")
    if writeback.get("no_gate_worse_wrong_writeback") is not True:
        add_failure(failures, "NO_GATE_WRITEBACK_CONTRAST_MISSING", "e16c_stage7_writeback_safety_report.json")
    if stage8.get("stage8_full_pass_required") is not False:
        add_failure(failures, "STAGE8_MARKED_REQUIRED", "e16c_stage7_downstream_stage8_probe_report.json")
    check_policy_ops(policy_report, failures)


def check(out: Path, write_summary: bool = False) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for artifact in REQUIRED_ARTIFACTS:
        if not (out / artifact).exists():
            add_failure(failures, "MISSING_ARTIFACT", artifact)
    for doc in DOCS:
        if not doc.exists():
            add_failure(failures, "MISSING_DOC", str(doc))
    if RUNNER.exists():
        check_imports(failures)
    else:
        add_failure(failures, "MISSING_RUNNER", str(RUNNER))
    if not failures:
        check_reports(out, failures)
    check_boundary(out, failures)
    result = {"schema_version": "e16c_stage7_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e16c_stage7_memory_binding_capacity_repair")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
