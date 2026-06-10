#!/usr/bin/env python3
"""Checker for E14 text-stream composition and canonical decoder confirm."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e14_text_stream_composition_and_canonical_decoder_confirm.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E14_TEXT_STREAM_COMPOSITION_AND_CANONICAL_DECODER_CONFIRM_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E14_TEXT_STREAM_COMPOSITION_AND_CANONICAL_DECODER_CONFIRM_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e14_search_report.json",
    "e14_input_stream_report.json",
    "e14_vocab_codebook_report.json",
    "e14_support_query_episode_report.json",
    "e14_transform_chain_report.json",
    "e14_system_comparison_report.json",
    "e14_task_family_report.json",
    "e14_canonical_decoder_report.json",
    "e14_renderer_faithfulness_report.json",
    "e14_trace_validity_report.json",
    "e14_writeback_safety_report.json",
    "e14_noise_decoy_report.json",
    "e14_heldout_generalization_report.json",
    "e14_semantic_leak_audit_report.json",
    "e14_deterministic_replay_report.json",
    "e14_boundary_claims_report.json",
)
VALID_DECISIONS = (
    "e14_text_stream_composition_and_canonical_decoder_confirmed",
    "e14_input_recovery_failure",
    "e14_support_parse_failure",
    "e14_transform_chain_selection_failure",
    "e14_chain_order_failure",
    "e14_decoder_failure",
    "e14_renderer_faithfulness_failure",
    "e14_ambiguous_case_failure",
    "e14_noise_decoy_failure",
    "e14_codebook_generalization_failure",
    "e14_semantic_slot_leak_detected",
    "e14_writeback_safety_failure",
    "e14_invalid_or_incomplete_run",
)
PRIMARY = "COMPOSITION_FLOW_PRUNED_GATED_CANONICAL_DECODER_PRIMARY"
DIRECT = "DIRECT_TEXT_REGEX_BASELINE"
ORACLE = "PRIVILEGED_ORACLE_CHAIN_CONTROL"
STATIC = "STATIC_CODEBOOK_LOOKUP_CONTROL"
SINGLE = "E13Z_SINGLE_TRANSFORM_FALLBACK"
NO_GATE = "COMPOSITION_FLOW_NO_GATE"
DECODER = "COMPOSITION_FLOW_GATED_CANONICAL_DECODER"
CHEAT = "RENDERER_ORACLE_CHEAT_CONTROL"
REQUIRED_SYSTEMS = (
    DIRECT,
    ORACLE,
    STATIC,
    SINGLE,
    NO_GATE,
    "COMPOSITION_FLOW_GATED",
    DECODER,
    PRIMARY,
    CHEAT,
    "TINY_SEQUENCE_MLP_CONTROL",
)
REQUIRED_FAMILIES = (
    "TWO_STEP_REVERSE_THEN_MAP",
    "TWO_STEP_MAP_THEN_REVERSE",
    "ROTATE_THEN_MAP",
    "MAP_THEN_ROTATE",
    "BIND_THEN_QUERY_THEN_MAP",
    "CONDITIONAL_COMPOSITION",
    "MULTI_SUPPORT_CHAIN_SELECTION",
    "AMBIGUOUS_SUPPORT_ABSTAIN_OR_REPAIR",
    "NOISE_AND_DECOY_COMPOSITION",
    "HELDOUT_CHAIN_COMPOSITION",
    "RANDOMIZED_CODEBOOK_COUNTERFACTUAL",
    "CANONICAL_DECODER_STRESS",
)
FORBIDDEN_RUNTIME_SLOTS = ("ACTION", "DIRECTION", "ENTITY", "MOVE", "NORTH", "RED", "CATEGORY", "COPY", "REVERSE", "BIND", "MAP", "ROTATE")
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


def check_primary_runtime_source(failures: list[dict[str, Any]]) -> None:
    runtime = function_source("run_composition_runtime")
    renderer = function_source("render_from_canonical")
    if not runtime:
        add_failure(failures, "MISSING_PRIMARY_RUNTIME_FUNCTION", "run_composition_runtime")
    for token in ("episode.family", ".family", "task_family", "chain_id", "oracle", "episode.expected", "expected_status"):
        if token in runtime:
            add_failure(failures, "FORBIDDEN_PRIMARY_RUNTIME_SOURCE_TOKEN", token)
    for token in ("oracle", "expected", "family", "chain_id"):
        if token in renderer:
            add_failure(failures, "FORBIDDEN_RENDERER_SOURCE_TOKEN", token)


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
        add_failure(failures, "RUNTIME_SEMANTIC_LABEL_FLAG_NOT_FALSE", "e14_input_stream_report.json")
    if stream.get("forbidden_task_words_present") is not False:
        add_failure(failures, "FORBIDDEN_TASK_WORD_FLAG_NOT_FALSE", "e14_input_stream_report.json")
    if stream.get("active_instruction_tokens") != []:
        add_failure(failures, "ACTIVE_INSTRUCTION_TOKENS_NOT_EMPTY", str(stream.get("active_instruction_tokens")))
    active_text = json.dumps(stream.get("active_instruction_tokens", []), sort_keys=True).lower()
    for word in FORBIDDEN_INPUT_WORDS:
        if word in active_text:
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
        add_failure(failures, "RANDOMIZED_VOCAB_NOT_TRUE", "e14_vocab_codebook_report.json")
    if report.get("randomized_codebook_used") is not True:
        add_failure(failures, "RANDOMIZED_CODEBOOK_NOT_TRUE", "e14_vocab_codebook_report.json")
    if report.get("heldout_vocabulary_used") is not True:
        add_failure(failures, "HELDOUT_VOCAB_NOT_TRUE", "e14_vocab_codebook_report.json")
    if report.get("heldout_chain_compositions_used") is not True:
        add_failure(failures, "HELDOUT_CHAIN_NOT_TRUE", "e14_vocab_codebook_report.json")
    if len(report.get("unique_codebook_hashes", [])) < 2:
        add_failure(failures, "CODEBOOK_HASHES_NOT_RANDOMIZED", str(report.get("unique_codebook_hashes")))


def check_semantic(semantic: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    if semantic.get("runtime_receives_forbidden_semantic_slots") is not False:
        add_failure(failures, "FORBIDDEN_RUNTIME_SLOT_FLAG_NOT_FALSE", "e14_semantic_leak_audit_report.json")
    if semantic.get("input_stream_contains_task_words") is not False:
        add_failure(failures, "INPUT_TASK_WORD_FLAG_NOT_FALSE", "e14_semantic_leak_audit_report.json")
    if semantic.get("task_family_used_by_primary_runtime") is not False:
        add_failure(failures, "TASK_FAMILY_RUNTIME_FLAG_NOT_FALSE", "e14_semantic_leak_audit_report.json")
    if semantic.get("chain_id_used_by_primary_runtime") is not False:
        add_failure(failures, "CHAIN_ID_RUNTIME_FLAG_NOT_FALSE", "e14_semantic_leak_audit_report.json")
    if semantic.get("renderer_oracle_access_in_primary_runtime") is not False:
        add_failure(failures, "RENDERER_ORACLE_ACCESS_FLAG_NOT_FALSE", "e14_semantic_leak_audit_report.json")
    if semantic.get("semantic_slot_leak_detected") is not False:
        add_failure(failures, "SEMANTIC_LEAK_FLAG_NOT_FALSE", "e14_semantic_leak_audit_report.json")
    runtime_text = json.dumps(semantic.get("primary_runtime_config", {}), sort_keys=True).upper()
    for token in FORBIDDEN_RUNTIME_SLOTS:
        if token in runtime_text:
            add_failure(failures, "FORBIDDEN_SLOT_IN_RUNTIME_CONFIG", token)


def expected_gate(aggregate: dict[str, Any], replay_passed: bool) -> dict[str, bool]:
    systems = aggregate["systems"]
    primary = systems[PRIMARY]
    return {
        "query_output_exact_accuracy_at_least_090": primary["query_output_exact_accuracy"] >= 0.90,
        "output_sequence_accuracy_at_least_094": primary["output_sequence_accuracy"] >= 0.94,
        "output_token_accuracy_at_least_097": primary["output_token_accuracy"] >= 0.97,
        "canonical_output_schema_validity_at_least_099": primary["canonical_output_schema_validity"] >= 0.99,
        "canonical_decoder_exact_accuracy_at_least_095": primary["canonical_decoder_exact_accuracy"] >= 0.95,
        "renderer_faithfulness_exact_100": primary["renderer_faithfulness"] == 1.0,
        "renderer_oracle_leak_detected_false": primary["renderer_oracle_leak_detected"] is False,
        "trace_validity_at_least_095": primary["trace_validity"] >= 0.95,
        "candidate_transform_fit_accuracy_at_least_090": primary["candidate_transform_fit_accuracy"] >= 0.90,
        "latent_transform_selection_accuracy_at_least_090": primary["latent_transform_selection_accuracy"] >= 0.90,
        "transform_chain_selection_accuracy_at_least_088": primary["transform_chain_selection_accuracy"] >= 0.88,
        "chain_order_accuracy_at_least_090": primary["chain_order_accuracy"] >= 0.90,
        "chain_step_accuracy_at_least_092": primary["chain_step_accuracy"] >= 0.92,
        "composition_exact_accuracy_at_least_088": primary["composition_exact_accuracy"] >= 0.88,
        "order_sensitive_pair_accuracy_at_least_090": primary["order_sensitive_pair_accuracy"] >= 0.90,
        "heldout_chain_composition_accuracy_at_least_082": primary["heldout_chain_composition_accuracy"] >= 0.82,
        "ambiguous_case_abstain_or_repair_accuracy_at_least_080": primary["ambiguous_case_abstain_or_repair_accuracy"] >= 0.80,
        "noise_decoy_composition_accuracy_at_least_082": primary["noise_decoy_composition_accuracy"] >= 0.82,
        "heldout_vocab_accuracy_at_least_085": primary["heldout_vocab_accuracy"] >= 0.85,
        "randomized_codebook_generalization_at_least_085": primary["randomized_codebook_generalization"] >= 0.85,
        "wrong_writeback_rate_at_most_002": primary["wrong_writeback_rate"] <= 0.02,
        "destructive_overwrite_rate_at_most_002": primary["destructive_overwrite_rate"] <= 0.02,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "semantic_slot_leak_detected_false": primary["semantic_slot_leak_detected"] is False,
        "privileged_control_selected_as_primary_false": primary["privileged_control_selected_as_primary"] is False,
        "deterministic_replay_passed": replay_passed,
        "beats_direct_on_query_exact": primary["query_output_exact_accuracy"] > systems[DIRECT]["query_output_exact_accuracy"],
        "beats_single_transform_on_composition": primary["composition_exact_accuracy"] > systems[SINGLE]["composition_exact_accuracy"],
        "beats_single_transform_on_order_sensitive": primary["order_sensitive_pair_accuracy"] > systems[SINGLE]["order_sensitive_pair_accuracy"],
        "beats_no_gate_on_trace_validity": primary["trace_validity"] > systems[NO_GATE]["trace_validity"],
        "beats_no_gate_on_wrong_writeback": primary["wrong_writeback_rate"] < systems[NO_GATE]["wrong_writeback_rate"],
        "static_control_not_primary": PRIMARY != STATIC,
        "oracle_control_not_primary": PRIMARY != ORACLE,
        "renderer_oracle_cheat_control_detected_invalid": systems[CHEAT]["renderer_oracle_leak_detected"] is True,
    }


def check_gate(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    replay = load_json(out / "e14_deterministic_replay_report.json")
    stream = load_json(out / "e14_input_stream_report.json")
    codebooks = load_json(out / "e14_vocab_codebook_report.json")
    semantic = load_json(out / "e14_semantic_leak_audit_report.json")
    boundary = load_json(out / "e14_boundary_claims_report.json")
    renderer = load_json(out / "e14_renderer_faithfulness_report.json")
    systems = aggregate.get("systems", {})
    family_metrics = aggregate.get("family_metrics", {})
    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_SYSTEM_MISMATCH", str(decision.get("primary_system")))
    if decision.get("primary_system") in {ORACLE, STATIC, CHEAT}:
        add_failure(failures, "PRIVILEGED_CONTROL_AS_PRIMARY", str(decision.get("primary_system")))
    if decision.get("decision") == "e14_text_stream_composition_and_canonical_decoder_confirmed" and decision.get("next") != "E15_TEXT_STREAM_LONG_HORIZON_MEMORY_AND_REPAIR_CONFIRM":
        add_failure(failures, "NEXT_MISMATCH_FOR_CONFIRMED", str(decision.get("next")))
    if not replay.get("internal_replay_passed", False):
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e14_deterministic_replay_report.json")
    for system in REQUIRED_SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_REQUIRED_SYSTEM", system)
    for family in REQUIRED_FAMILIES:
        if PRIMARY not in family_metrics or family not in family_metrics[PRIMARY]:
            add_failure(failures, "MISSING_PRIMARY_FAMILY_METRICS", family)
    check_input_stream(stream, failures)
    check_codebooks(codebooks, failures)
    check_semantic(semantic, failures)
    if "deterministic synthetic controlled text-stream composition proxy" not in boundary.get("boundary", ""):
        add_failure(failures, "BOUNDARY_TEXT_MISSING", str(boundary.get("boundary")))
    if renderer.get("renderer_oracle_leak_detected", {}).get(CHEAT) is not True:
        add_failure(failures, "CHEAT_RENDERER_NOT_DETECTED", CHEAT)
    if renderer.get("renderer_faithfulness", {}).get(PRIMARY) != 1.0:
        add_failure(failures, "PRIMARY_RENDERER_NOT_FAITHFUL", str(renderer.get("renderer_faithfulness", {}).get(PRIMARY)))
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
    expected_cost_delta = round(1.0 - rate(primary["cost_per_tick"], systems[DECODER]["cost_per_tick"]), 6)
    if aggregate.get("positive_gate", {}).get("deltas", {}).get("cost_reduction_vs_decoder") != expected_cost_delta:
        add_failure(failures, "DELTA_COST_MISMATCH", str(aggregate.get("positive_gate", {}).get("deltas", {}).get("cost_reduction_vs_decoder")))


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
        check_primary_runtime_source(failures)
    else:
        add_failure(failures, "MISSING_RUNNER", str(RUNNER))
    if not failures:
        check_gate(out, failures)
    check_boundaries(out, failures)
    result = {"schema_version": "e14_checker_result_v1", "out": str(out), "failure_count": len(failures), "failures": failures}
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e14_text_stream_composition_and_canonical_decoder_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
