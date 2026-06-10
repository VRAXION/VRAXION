#!/usr/bin/env python3
"""Checker for E13Z text-stream to temporal Flow capability confirm."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e13z_text_stream_to_temporal_flow_capability_confirm.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E13Z_TEXT_STREAM_TO_TEMPORAL_FLOW_CAPABILITY_CONFIRM_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E13Z_TEXT_STREAM_TO_TEMPORAL_FLOW_CAPABILITY_CONFIRM_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e13z_search_report.json",
    "e13z_input_stream_report.json",
    "e13z_vocab_codebook_report.json",
    "e13z_support_query_episode_report.json",
    "e13z_system_comparison_report.json",
    "e13z_task_family_report.json",
    "e13z_trace_validity_report.json",
    "e13z_writeback_safety_report.json",
    "e13z_noise_decoy_report.json",
    "e13z_heldout_generalization_report.json",
    "e13z_semantic_leak_audit_report.json",
    "e13z_deterministic_replay_report.json",
    "e13z_boundary_claims_report.json",
)
VALID_DECISIONS = (
    "e13z_text_stream_to_temporal_flow_capability_confirmed",
    "e13z_text_stream_input_recovery_failure",
    "e13z_token_boundary_failure",
    "e13z_support_fit_failure",
    "e13z_transform_selection_failure",
    "e13z_output_decoder_failure",
    "e13z_noise_decoy_failure",
    "e13z_codebook_generalization_failure",
    "e13z_semantic_slot_leak_detected",
    "e13z_writeback_safety_failure",
    "e13z_invalid_or_incomplete_run",
)
PRIMARY = "TEMPORAL_TEXT_FLOW_SUPPORT_FIT_PRUNED_PRIMARY"
DIRECT = "DIRECT_TEXT_REGEX_BASELINE"
ORACLE = "PRIVILEGED_ORACLE_TASK_FAMILY_CONTROL"
STATIC = "STATIC_CODEBOOK_LOOKUP_CONTROL"
NO_GATE = "TEMPORAL_TEXT_FLOW_NO_GATE"
SUPPORT_FIT = "TEMPORAL_TEXT_FLOW_SUPPORT_FIT_GATED"
REQUIRED_SYSTEMS = (
    DIRECT,
    ORACLE,
    STATIC,
    NO_GATE,
    "TEMPORAL_TEXT_FLOW_GATED",
    SUPPORT_FIT,
    PRIMARY,
    "TINY_SEQUENCE_MLP_CONTROL",
)
REQUIRED_FAMILIES = (
    "COPY_SEQUENCE",
    "REVERSE_SEQUENCE",
    "ROTATE_OR_SHIFT_SEQUENCE",
    "REWRITE_MAP",
    "BIND_QUERY",
    "CONDITIONAL_MARKER",
    "MULTI_STEP_COMPOSITION",
    "NOISE_AND_DECOY_STREAM",
    "HELDOUT_VOCABULARY",
    "RANDOMIZED_CODEBOOK_COUNTERFACTUAL",
)
FORBIDDEN_RUNTIME_SLOTS = ("ACTION", "DIRECTION", "ENTITY", "MOVE", "NORTH", "RED", "CATEGORY")
FORBIDDEN_INPUT_WORDS = ("copy", "reverse", "bind", "map", "rotate", "query")


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


def function_source(name: str) -> str:
    text = RUNNER.read_text(encoding="utf-8")
    marker = f"def {name}("
    start = text.find(marker)
    if start < 0:
        return ""
    next_def = text.find("\ndef ", start + 1)
    if next_def < 0:
        next_def = len(text)
    return text[start:next_def]


def check_runtime_no_family_branch(failures: list[dict[str, Any]]) -> None:
    source = function_source("run_flow_runtime")
    if not source:
        add_failure(failures, "MISSING_PRIMARY_RUNTIME_FUNCTION", "run_flow_runtime")
        return
    blocked = ("episode.family", ".family", "task_family")
    for token in blocked:
        if token in source:
            add_failure(failures, "TASK_FAMILY_VISIBLE_TO_PRIMARY_RUNTIME", token)


def check_boundaries(out: Path, failures: list[dict[str, Any]]) -> None:
    blocked = ("A" + "GI", "conscious" + "ness", "natural-language", "natural language", "production-readiness", "production readiness", "D" + "99", "D" + "100")
    for path in (out / "report.md", *DOCS):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in blocked:
            if token.lower() in text:
                add_failure(failures, "BOUNDARY_TOKEN_FOUND", f"{path}:{token}")


def check_input_stream(stream: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    if stream.get("runtime_receives_semantic_labels") is not False:
        add_failure(failures, "RUNTIME_SEMANTIC_LABEL_FLAG_NOT_FALSE", "e13z_input_stream_report.json")
    if stream.get("forbidden_task_words_present") is not False:
        add_failure(failures, "FORBIDDEN_TASK_WORD_FLAG_NOT_FALSE", "e13z_input_stream_report.json")
    if stream.get("active_instruction_tokens") != []:
        add_failure(failures, "ACTIVE_INSTRUCTION_TOKENS_NOT_EMPTY", str(stream.get("active_instruction_tokens")))
    text = json.dumps(stream.get("active_instruction_tokens", []), sort_keys=True).lower()
    for word in FORBIDDEN_INPUT_WORDS:
        if word in text:
            add_failure(failures, "FORBIDDEN_INPUT_WORD_ACTIVE", word)
    for idx, pulse in enumerate(stream.get("sample_stream", [])):
        for key in ("clock", "start", "end", "boundary", "separator", "guard"):
            if not isinstance(pulse.get(key), int):
                add_failure(failures, "NON_INTEGER_PULSE_FIELD", f"{idx}:{key}")
        for key in ("char", "struct"):
            values = pulse.get(key)
            if not isinstance(values, list) or any(value not in (0, 1) for value in values):
                add_failure(failures, "NON_BINARY_PULSE_VECTOR", f"{idx}:{key}")


def check_codebooks(report: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    if report.get("randomized_vocab_used") is not True:
        add_failure(failures, "RANDOMIZED_VOCAB_NOT_TRUE", "e13z_vocab_codebook_report.json")
    if report.get("randomized_codebook_used") is not True:
        add_failure(failures, "RANDOMIZED_CODEBOOK_NOT_TRUE", "e13z_vocab_codebook_report.json")
    if report.get("heldout_vocabulary_used") is not True:
        add_failure(failures, "HELDOUT_VOCAB_NOT_TRUE", "e13z_vocab_codebook_report.json")
    if len(report.get("unique_codebook_hashes", [])) < 2:
        add_failure(failures, "CODEBOOK_HASHES_NOT_RANDOMIZED", str(report.get("unique_codebook_hashes")))


def check_semantic(semantic: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    if semantic.get("runtime_receives_forbidden_semantic_slots") is not False:
        add_failure(failures, "FORBIDDEN_RUNTIME_SLOT_FLAG_NOT_FALSE", "e13z_semantic_leak_audit_report.json")
    if semantic.get("input_stream_contains_task_words") is not False:
        add_failure(failures, "INPUT_TASK_WORD_FLAG_NOT_FALSE", "e13z_semantic_leak_audit_report.json")
    if semantic.get("task_family_used_by_primary_runtime") is not False:
        add_failure(failures, "TASK_FAMILY_RUNTIME_FLAG_NOT_FALSE", "e13z_semantic_leak_audit_report.json")
    if semantic.get("semantic_slot_leak_detected") is not False:
        add_failure(failures, "SEMANTIC_LEAK_FLAG_NOT_FALSE", "e13z_semantic_leak_audit_report.json")
    if semantic.get("privileged_control_selected_as_primary") is not False:
        add_failure(failures, "PRIVILEGED_PRIMARY_FLAG_NOT_FALSE", "e13z_semantic_leak_audit_report.json")
    runtime_text = json.dumps(semantic.get("primary_runtime_config", {}), sort_keys=True).upper()
    for token in FORBIDDEN_RUNTIME_SLOTS:
        if token in runtime_text:
            add_failure(failures, "FORBIDDEN_SLOT_IN_RUNTIME_CONFIG", token)


def expected_gate(aggregate: dict[str, Any], replay_passed: bool) -> dict[str, bool]:
    systems = aggregate["systems"]
    primary = systems[PRIMARY]
    direct = systems[DIRECT]
    no_gate = systems[NO_GATE]
    return {
        "query_output_exact_accuracy_at_least_092": primary["query_output_exact_accuracy"] >= 0.92,
        "output_sequence_accuracy_at_least_095": primary["output_sequence_accuracy"] >= 0.95,
        "output_token_accuracy_at_least_098": primary["output_token_accuracy"] >= 0.98,
        "trace_validity_at_least_095": primary["trace_validity"] >= 0.95,
        "char_stream_recovery_accuracy_at_least_098": primary["char_stream_recovery_accuracy"] >= 0.98,
        "token_boundary_accuracy_at_least_098": primary["token_boundary_accuracy"] >= 0.98,
        "candidate_transform_fit_accuracy_at_least_090": primary["candidate_transform_fit_accuracy"] >= 0.90,
        "latent_transform_selection_accuracy_at_least_090": primary["latent_transform_selection_accuracy"] >= 0.90,
        "copy_family_accuracy_at_least_095": primary["copy_family_accuracy"] >= 0.95,
        "reverse_family_accuracy_at_least_095": primary["reverse_family_accuracy"] >= 0.95,
        "rotate_shift_family_accuracy_at_least_090": primary["rotate_shift_family_accuracy"] >= 0.90,
        "rewrite_map_family_accuracy_at_least_090": primary["rewrite_map_family_accuracy"] >= 0.90,
        "bind_query_family_accuracy_at_least_090": primary["bind_query_family_accuracy"] >= 0.90,
        "conditional_marker_family_accuracy_at_least_085": primary["conditional_marker_family_accuracy"] >= 0.85,
        "multi_step_composition_accuracy_at_least_080": primary["multi_step_composition_accuracy"] >= 0.80,
        "noise_rejection_rate_at_least_085": primary["noise_rejection_rate"] >= 0.85,
        "decoy_rejection_rate_at_least_085": primary["decoy_rejection_rate"] >= 0.85,
        "heldout_vocabulary_accuracy_at_least_085": primary["heldout_vocabulary_accuracy"] >= 0.85,
        "randomized_codebook_generalization_at_least_085": primary["randomized_codebook_generalization"] >= 0.85,
        "wrong_writeback_rate_at_most_002": primary["wrong_writeback_rate"] <= 0.02,
        "destructive_overwrite_rate_at_most_002": primary["destructive_overwrite_rate"] <= 0.02,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "semantic_slot_leak_detected_false": primary["semantic_slot_leak_detected"] is False,
        "privileged_control_selected_as_primary_false": primary["privileged_control_selected_as_primary"] is False,
        "deterministic_replay_passed": replay_passed,
        "beats_direct_text_regex_on_query_exact": primary["query_output_exact_accuracy"] > direct["query_output_exact_accuracy"],
        "beats_no_gate_on_trace_validity": primary["trace_validity"] > no_gate["trace_validity"],
        "beats_no_gate_on_wrong_writeback": primary["wrong_writeback_rate"] < no_gate["wrong_writeback_rate"],
        "static_control_not_primary": PRIMARY != STATIC,
        "oracle_control_not_primary": PRIMARY != ORACLE,
    }


def check_gate(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e13z_deterministic_replay_report.json")
    stream = load_json(out / "e13z_input_stream_report.json")
    codebooks = load_json(out / "e13z_vocab_codebook_report.json")
    semantic = load_json(out / "e13z_semantic_leak_audit_report.json")
    boundary = load_json(out / "e13z_boundary_claims_report.json")
    systems = aggregate.get("systems", {})
    family_metrics = aggregate.get("family_metrics", {})
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_SYSTEM_MISMATCH", str(decision.get("primary_system")))
    if decision.get("primary_system") in {ORACLE, STATIC}:
        add_failure(failures, "PRIVILEGED_CONTROL_AS_PRIMARY", str(decision.get("primary_system")))
    if decision.get("decision") == "e13z_text_stream_to_temporal_flow_capability_confirmed" and decision.get("next") != "E14_TEXT_STREAM_COMPOSITION_AND_CANONICAL_DECODER_CONFIRM":
        add_failure(failures, "NEXT_MISMATCH_FOR_CONFIRMED", str(decision.get("next")))
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e13z_deterministic_replay_report.json")
    for system in REQUIRED_SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_REQUIRED_SYSTEM", system)
    for family in REQUIRED_FAMILIES:
        if PRIMARY not in family_metrics or family not in family_metrics[PRIMARY]:
            add_failure(failures, "MISSING_PRIMARY_FAMILY_METRICS", family)
    check_input_stream(stream, failures)
    check_codebooks(codebooks, failures)
    check_semantic(semantic, failures)
    if "deterministic synthetic controlled text-stream proxy" not in boundary.get("boundary", ""):
        add_failure(failures, "BOUNDARY_TEXT_MISSING", str(boundary.get("boundary")))
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
    expected_cost_reduction = round(1.0 - rate(primary["cost_per_tick"], systems[SUPPORT_FIT]["cost_per_tick"]), 6)
    if deltas.get("cost_reduction_vs_support_fit") != expected_cost_reduction:
        add_failure(failures, "DELTA_COST_MISMATCH", str(deltas.get("cost_reduction_vs_support_fit")))


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
        check_runtime_no_family_branch(failures)
    else:
        add_failure(failures, "MISSING_RUNNER", str(RUNNER))
    if not failures:
        check_gate(out, failures)
    check_boundaries(out, failures)
    result = {"schema_version": "e13z_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e13z_text_stream_to_temporal_flow_capability_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
