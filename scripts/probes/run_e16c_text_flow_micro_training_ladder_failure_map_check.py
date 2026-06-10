#!/usr/bin/env python3
"""Checker for E16C text-flow micro-training ladder failure-map probe."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e16c_text_flow_micro_training_ladder_failure_map.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E16C_TEXT_FLOW_MICRO_TRAINING_LADDER_FAILURE_MAP_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E16C_TEXT_FLOW_MICRO_TRAINING_LADDER_FAILURE_MAP_RESULT.md",
)
PRIMARY = "MICRO_TRAINING_PRUNED_PRIMARY"
NO_GATE = "MICRO_TRAINING_NO_GATE"
HAND_CONTROL = "HAND_MICRO_REFERENCE_CONTROL"
BOUNDARY = (
    "This is a deterministic synthetic controlled text-flow micro-training ladder. "
    "It maps how far micro-program discovery gets from a minimal micro-VM. "
    "It does not prove general natural language AI or unconstrained invention from absolute nothing."
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e16c_search_report.json",
    "e16c_micro_vm_report.json",
    "e16c_curriculum_report.json",
    "e16c_training_curve_report.json",
    "e16c_stage_metric_report.json",
    "e16c_best_stage_report.json",
    "e16c_failure_map_report.json",
    "e16c_discovered_program_library_report.json",
    "e16c_pruned_library_report.json",
    "e16c_ablation_report.json",
    "e16c_heldout_generalization_report.json",
    "e16c_trace_validity_report.json",
    "e16c_writeback_safety_report.json",
    "e16c_semantic_macro_leak_audit_report.json",
    "e16c_deterministic_replay_report.json",
    "e16c_boundary_claims_report.json",
    "e16c_next_repair_recommendation.json",
)
REQUIRED_SYSTEMS = (
    "RANDOM_MICRO_PROGRAM_BASELINE",
    "GREEDY_SUPPORT_FIT_BASELINE",
    HAND_CONTROL,
    NO_GATE,
    "MICRO_TRAINING_PRIMARY",
    PRIMARY,
    "NO_REWRITE_MICRO_ABLATION",
    "NO_VALIDITY_MICRO_ABLATION",
    "NO_MEMORY_MICRO_ABLATION",
    "NO_CONDITIONAL_MICRO_ABLATION",
    "TOO_SHORT_PROGRAM_BUDGET_ABLATION",
)
STAGE_GATES = {
    0: {"char_stream_recovery_accuracy": (0.98, "min")},
    1: {"token_boundary_accuracy": (0.95, "min"), "token_recovery_accuracy": (0.95, "min")},
    2: {"order_program_discovery_accuracy": (0.85, "min"), "output_sequence_accuracy": (0.90, "min")},
    3: {"rewrite_evidence_fit_accuracy": (0.85, "min"), "heldout_rewrite_accuracy": (0.85, "min")},
    4: {"filter_program_accuracy": (0.85, "min"), "decoy_rejection_rate": (0.85, "min"), "wrong_writeback_rate": (0.05, "max")},
    5: {"phrase_composition_accuracy": (0.80, "min"), "chain_order_accuracy": (0.80, "min")},
    6: {"sentence_template_accuracy": (0.75, "min"), "heldout_template_accuracy": (0.70, "min")},
    7: {"multi_sentence_binding_accuracy": (0.70, "min"), "long_horizon_recall": (0.70, "min"), "ambiguous_abstain_accuracy": (0.75, "min")},
    8: {
        "repair_success_rate": (0.70, "min"),
        "noise_rejection_rate": (0.75, "min"),
        "canonical_decoder_exact_accuracy": (0.75, "min"),
        "trace_validity": (0.90, "min"),
    },
}
ALLOWED_MICRO_OPS = {
    "READ_POS",
    "WRITE_POS",
    "COPY_POS",
    "COMPARE_EQ",
    "IF_EQ",
    "IF_VALID_EVIDENCE",
    "IF_REWRITE_EVIDENCE",
    "ROUTE_TOKEN",
    "KEEP_TOKEN",
    "DROP_TOKEN",
    "COMMIT_OUTPUT",
    "OPEN_MEMORY_SLOT",
    "WRITE_MEMORY_SLOT",
    "READ_MEMORY_SLOT",
    "CLEAR_MEMORY_SLOT",
    "TRACE_CHECK",
    "GATED_COMMIT",
}
FORBIDDEN_MACROS = {
    "REVERSE",
    "ROTATE",
    "SWAP01",
    "SWAP12",
    "SWAP23",
    "MAP",
    "FILTER",
    "BIND",
    "QUERY",
    "MAP_THEN_REVERSE",
    "REVERSE_THEN_MAP",
    "FILTER_THEN_REVERSE",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_failure(failures: list[dict[str, Any]], code: str, detail: str) -> None:
    failures.append({"code": code, "detail": detail})


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


def stage_pass(stage_id: int, metrics: dict[str, float]) -> bool:
    for key, (threshold, mode) in STAGE_GATES[stage_id].items():
        value = metrics.get(key, 0.0)
        if mode == "min" and value < threshold:
            return False
        if mode == "max" and value > threshold:
            return False
    return True


def recompute_best(rows: list[dict[str, Any]]) -> tuple[int, int | None, str | None]:
    best = -1
    first_fail: int | None = None
    first_fail_name: str | None = None
    for row in rows:
        if row["passed"] and first_fail is None:
            best = row["stage"]
        elif not row["passed"] and first_fail is None:
            first_fail = row["stage"]
            first_fail_name = row["stage_name"]
    return best, first_fail, first_fail_name


def valid_decision_label(label: str) -> bool:
    return label in {
        "e16c_text_flow_micro_training_ladder_confirmed",
        "e16c_text_flow_micro_training_ladder_partial_confirmed",
        "e16c_invalid_or_incomplete_run",
    } or label.startswith("e16c_text_flow_micro_training_ladder_failed_at_stage_")


def check_boundary(out: Path, failures: list[dict[str, Any]]) -> None:
    paths = [out / "report.md", out / "e16c_boundary_claims_report.json", *DOCS]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if BOUNDARY not in text:
            add_failure(failures, "BOUNDARY_TEXT_MISSING", str(path))
        scrubbed = text.replace(BOUNDARY, "")
        lower = scrubbed.lower()
        blocked = ("agi", "consciousness", "open-natural-language", "open natural language", "general natural language", "d99", "d100")
        for token in blocked:
            if token in lower:
                add_failure(failures, "BROAD_CLAIM_TOKEN_FOUND", f"{path}:{token}")


def check_program_library(report: dict[str, Any], failures: list[dict[str, Any]], label: str) -> None:
    programs = report.get("programs", [])
    if not programs:
        add_failure(failures, "EMPTY_PROGRAM_LIBRARY", label)
    for program in programs:
        ops = program.get("micro_ops", [])
        if not ops:
            add_failure(failures, "PROGRAM_WITHOUT_OPS", str(program.get("program_id")))
        for op in ops:
            if op not in ALLOWED_MICRO_OPS:
                add_failure(failures, "NON_MICRO_OP_IN_LIBRARY", f"{label}:{op}")
            if op in FORBIDDEN_MACROS:
                add_failure(failures, "FORBIDDEN_MACRO_IN_LIBRARY", f"{label}:{op}")


def check_stage_logic(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    best_report = load_json(out / "e16c_best_stage_report.json")
    failure_map = load_json(out / "e16c_failure_map_report.json")
    training = load_json(out / "e16c_training_curve_report.json")
    discovered = load_json(out / "e16c_discovered_program_library_report.json")
    pruned = load_json(out / "e16c_pruned_library_report.json")
    ablation = load_json(out / "e16c_ablation_report.json")
    semantic = load_json(out / "e16c_semantic_macro_leak_audit_report.json")
    replay = load_json(out / "e16c_deterministic_replay_report.json")
    writeback = load_json(out / "e16c_writeback_safety_report.json")
    systems = aggregate.get("systems", {})
    primary = systems.get(PRIMARY, {})

    if not valid_decision_label(str(decision.get("decision"))):
        add_failure(failures, "INVALID_DECISION_LABEL", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_SYSTEM_MISMATCH", str(decision.get("primary_system")))
    for system in REQUIRED_SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_SYSTEM", system)
    if not primary:
        return

    rows = primary.get("stage_pass_vector", [])
    if len(rows) != 9:
        add_failure(failures, "STAGE_VECTOR_LENGTH_MISMATCH", str(len(rows)))
        return
    for row in rows:
        stage_id = row["stage"]
        expected = stage_pass(stage_id, row.get("metrics", {}))
        if row.get("passed") is not expected:
            add_failure(failures, "STAGE_GATE_MATH_MISMATCH", str(stage_id))
    best, first_fail, first_fail_name = recompute_best(rows)
    if primary.get("best_stage_passed") != best:
        add_failure(failures, "BEST_STAGE_MISMATCH", str(primary.get("best_stage_passed")))
    if primary.get("first_failing_stage") != first_fail:
        add_failure(failures, "FIRST_FAIL_STAGE_MISMATCH", str(primary.get("first_failing_stage")))
    if primary.get("first_failing_stage_name") != first_fail_name:
        add_failure(failures, "FIRST_FAIL_NAME_MISMATCH", str(primary.get("first_failing_stage_name")))
    if best_report.get("best_stage_passed") != best or best_report.get("first_failing_stage") != first_fail:
        add_failure(failures, "BEST_STAGE_REPORT_MISMATCH", "e16c_best_stage_report.json")
    if first_fail is not None and not failure_map.get("failure_signature"):
        add_failure(failures, "FAILURE_SIGNATURE_MISSING", "e16c_failure_map_report.json")
    if first_fail is not None and not failure_map.get("recommended_next_repair"):
        add_failure(failures, "RECOMMENDED_REPAIR_MISSING", "e16c_failure_map_report.json")
    if decision.get("best_stage_passed") != best or decision.get("first_failing_stage") != first_fail:
        add_failure(failures, "DECISION_STAGE_FIELDS_MISMATCH", "decision.json")

    expected_decision = "e16c_text_flow_micro_training_ladder_confirmed" if first_fail is None else (
        "e16c_text_flow_micro_training_ladder_partial_confirmed" if best >= 5 else f"e16c_text_flow_micro_training_ladder_failed_at_stage_{first_fail}"
    )
    if decision.get("decision") != expected_decision:
        add_failure(failures, "DECISION_LOGIC_MISMATCH", f"{decision.get('decision')} != {expected_decision}")
    if decision.get("positive_gate_passed") is not (expected_decision.endswith("confirmed")):
        add_failure(failures, "POSITIVE_GATE_FLAG_MISMATCH", "decision.json")

    curves = training.get("learning_curve_by_stage", {})
    for stage_id in range(9):
        curve = curves.get(str(stage_id), [])
        if len(curve) < 2:
            add_failure(failures, "MISSING_LEARNING_CURVE", str(stage_id))
        for row in curve:
            if "generation" not in row or "train_accuracy" not in row or "heldout_accuracy" not in row:
                add_failure(failures, "MALFORMED_LEARNING_CURVE_ROW", str(stage_id))

    check_program_library(discovered, failures, "discovered")
    check_program_library(pruned, failures, "pruned")
    if pruned.get("pruned_library_size") != primary.get("discovered_library_size"):
        add_failure(failures, "PRUNED_LIBRARY_SIZE_MISMATCH", "e16c_pruned_library_report.json")
    if semantic.get("semantic_slot_leak_detected") is not False:
        add_failure(failures, "SEMANTIC_LEAK_FLAG_NOT_FALSE", "e16c_semantic_macro_leak_audit_report.json")
    if semantic.get("macro_leak_detected") is not False:
        add_failure(failures, "MACRO_LEAK_FLAG_NOT_FALSE", "e16c_semantic_macro_leak_audit_report.json")
    if semantic.get("privileged_control_selected_as_primary") is not False:
        add_failure(failures, "PRIVILEGED_PRIMARY_FLAG_NOT_FALSE", "e16c_semantic_macro_leak_audit_report.json")
    if replay.get("internal_replay_passed") is not True:
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e16c_deterministic_replay_report.json")
    if writeback.get("no_gate_worse_trace_safety") is not True:
        add_failure(failures, "NO_GATE_TRACE_CONTRAST_MISSING", "e16c_writeback_safety_report.json")
    if writeback.get("no_gate_worse_writeback_safety") is not True:
        add_failure(failures, "NO_GATE_WRITEBACK_CONTRAST_MISSING", "e16c_writeback_safety_report.json")
    if systems[HAND_CONTROL].get("best_stage_passed") < primary.get("best_stage_passed"):
        add_failure(failures, "HAND_REFERENCE_WEAKER_THAN_PRIMARY", HAND_CONTROL)
    if primary.get("privileged_control_selected_as_primary") is not False:
        add_failure(failures, "PRIMARY_PRIVILEGE_FLAG_NOT_FALSE", PRIMARY)

    expectations = ablation.get("expectations", {})
    expected_ablations = {
        "no_rewrite_fails_stage_3_or_later": True,
        "no_validity_fails_stage_4_or_later": True,
        "no_memory_fails_stage_7_or_later": True,
        "no_conditional_fails_template_stage": True,
        "too_short_fails_composition_stage": True,
    }
    for key, expected in expected_ablations.items():
        if expectations.get(key) is not expected:
            add_failure(failures, "ABLATION_EXPECTATION_FAILED", key)


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
        check_stage_logic(out, failures)
    check_boundary(out, failures)
    result = {"schema_version": "e16c_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e16c_text_flow_micro_training_ladder_failure_map")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
