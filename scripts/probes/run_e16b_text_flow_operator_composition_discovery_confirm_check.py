#!/usr/bin/env python3
"""Checker for E16B text-flow operator composition discovery confirm."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_e16b_text_flow_operator_composition_discovery_confirm.py")
DOCS = (
    Path(__file__).resolve().parents[2] / "docs/research/E16B_TEXT_FLOW_OPERATOR_COMPOSITION_DISCOVERY_CONFIRM_CONTRACT.md",
    Path(__file__).resolve().parents[2] / "docs/research/E16B_TEXT_FLOW_OPERATOR_COMPOSITION_DISCOVERY_CONFIRM_RESULT.md",
)
REQUIRED_ARTIFACTS = (
    "decision.json",
    "summary.json",
    "aggregate_metrics.json",
    "report.md",
    "e16b_search_report.json",
    "e16b_operator_grammar_report.json",
    "e16b_macro_removal_audit_report.json",
    "e16b_discovered_library_report.json",
    "e16b_program_chain_report.json",
    "e16b_system_comparison_report.json",
    "e16b_task_family_report.json",
    "e16b_ablation_report.json",
    "e16b_support_disambiguation_report.json",
    "e16b_heldout_generalization_report.json",
    "e16b_trace_validity_report.json",
    "e16b_writeback_safety_report.json",
    "e16b_semantic_leak_audit_report.json",
    "e16b_deterministic_replay_report.json",
    "e16b_boundary_claims_report.json",
)
VALID_DECISIONS = (
    "e16b_text_flow_operator_composition_discovery_confirmed",
    "e16b_search_failed_to_discover_composition",
    "e16b_chain_length_insufficient",
    "e16b_order_primitive_dependency_failure",
    "e16b_map_primitive_dependency_failure",
    "e16b_filter_primitive_dependency_failure",
    "e16b_support_ambiguity_failure",
    "e16b_holdout_generalization_failure",
    "e16b_semantic_or_macro_leak_detected",
    "e16b_writeback_safety_failure",
    "e16b_invalid_or_incomplete_run",
)
PRIMARY = "COMPOSITION_DISCOVERY_PRUNED_PRIMARY"
UNPRUNED = "COMPOSITION_DISCOVERY_PRIMARY"
NO_GATE = "COMPOSITION_DISCOVERY_NO_GATE"
RANDOM_MATCHED = "RANDOM_LIBRARY_MATCHED_BUDGET"
RANDOM_BEST = "RANDOM_LIBRARY_BEST_OF_N_CONTROL"
MACRO_CONTROL = "TRUE_MACRO_LIBRARY_CONTROL"
HAND_AUTHORED = "TRUE_PRIMITIVE_HAND_AUTHORED_CONTROL"
REQUIRED_SYSTEMS = (
    "RANDOM_LIBRARY_SMALL",
    RANDOM_MATCHED,
    RANDOM_BEST,
    MACRO_CONTROL,
    HAND_AUTHORED,
    NO_GATE,
    UNPRUNED,
    PRIMARY,
    "INSUFFICIENT_CHAIN_LEN_ABLATION",
    "MISSING_ORDER_PRIMITIVES_ABLATION",
    "MISSING_MAP_PRIMITIVE_ABLATION",
    "MISSING_FILTER_PRIMITIVE_ABLATION",
    "AMBIGUOUS_SUPPORT_NO_ABSTAIN_ABLATION",
)
REQUIRED_FAMILIES = (
    "REVERSE_FROM_SWAPS",
    "MAP_THEN_REVERSE_FROM_PRIMITIVES",
    "REVERSE_THEN_MAP_FROM_PRIMITIVES",
    "FILTER_THEN_REVERSE",
    "ROTATE_THEN_MAP",
    "MAP_THEN_ROTATE",
    "SWAP_OUTER_THEN_MAP",
    "SUPPORT_AMBIGUITY_ABSTAIN_OR_REPAIR",
    "SUPPORT_DISAMBIGUATION",
    "HELDOUT_VOCAB_CODEBOOK",
    "DECOY_HEAVY_COMPOSITION",
)
ALLOWED_PRIMITIVES = {
    "SWAP01",
    "SWAP12",
    "SWAP23",
    "ROTL",
    "ROTR",
    "MAP",
    "FILTER_VALID",
    "COPY",
    "COMMIT_OUTPUT",
}
FORBIDDEN_MACROS = {
    "REVERSE",
    "MAP_THEN_REVERSE",
    "REVERSE_THEN_MAP",
    "FILTER_THEN_REVERSE",
    "ROTATE_THEN_MAP",
    "MAP_THEN_ROTATE",
    "SWAP_OUTER_THEN_MAP",
    "PRIVILEGED_DIRECT_MACRO",
}
BOUNDARY = (
    "This confirms grammar-level operator composition discovery from lower-level primitives in a deterministic synthetic "
    "controlled text-flow proxy. It does not confirm unconstrained operator invention or general natural-language AI."
)


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
    allowed = {"__future__", "argparse", "collections", "dataclasses", "hashlib", "json", "pathlib", "random", "typing"}
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
    source = function_source("run_primitive_composition_runtime")
    if not source:
        add_failure(failures, "MISSING_PRIMARY_RUNTIME_FUNCTION", "run_primitive_composition_runtime")
        return
    for token in ("family", "gold", "expected", "oracle", "true_sequence", "task_family", "PRIVILEGED_DIRECT_MACRO"):
        if token in source:
            add_failure(failures, "FORBIDDEN_PRIMARY_RUNTIME_SOURCE_TOKEN", token)


def check_boundary_text(out: Path, failures: list[dict[str, Any]]) -> None:
    paths = [out / "report.md", out / "e16b_boundary_claims_report.json", *DOCS]
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if BOUNDARY not in text:
            add_failure(failures, "REQUIRED_BOUNDARY_MISSING", str(path))
        scrubbed = text.replace(BOUNDARY, "")
        lower = scrubbed.lower()
        blocked = ("agi", "consciousness", "open-natural-language", "open natural-language", "open natural language", "d99", "d100")
        for token in blocked:
            if token in lower:
                add_failure(failures, "BROAD_BOUNDARY_TOKEN_FOUND", f"{path}:{token}")


def expected_gate(aggregate: dict[str, Any], replay_passed: bool) -> dict[str, bool]:
    systems = aggregate["systems"]
    primary = systems[PRIMARY]
    return {
        "macro_removed_confirmed_true": primary["macro_removed_confirmed"] is True,
        "direct_macro_leak_detected_false": primary["direct_macro_leak_detected"] is False,
        "discovery_test_exact_accuracy_at_least_092": primary["discovery_test_exact_accuracy"] >= 0.92,
        "composition_exact_accuracy_at_least_090": primary["composition_exact_accuracy"] >= 0.90,
        "heldout_vocab_accuracy_at_least_088": primary["heldout_vocab_accuracy"] >= 0.88,
        "randomized_codebook_generalization_at_least_088": primary["randomized_codebook_generalization"] >= 0.88,
        "support_fit_accuracy_at_least_090": primary["support_fit_accuracy"] >= 0.90,
        "program_selection_accuracy_at_least_088": primary["program_selection_accuracy"] >= 0.88,
        "order_sensitive_pair_accuracy_at_least_090": primary["order_sensitive_pair_accuracy"] >= 0.90,
        "reverse_from_swaps_accuracy_at_least_092": primary["reverse_from_swaps_accuracy"] >= 0.92,
        "map_then_reverse_accuracy_at_least_088": primary["map_then_reverse_accuracy"] >= 0.88,
        "reverse_then_map_accuracy_at_least_088": primary["reverse_then_map_accuracy"] >= 0.88,
        "filter_then_reverse_accuracy_at_least_085": primary["filter_then_reverse_accuracy"] >= 0.85,
        "rotate_then_map_accuracy_at_least_085": primary["rotate_then_map_accuracy"] >= 0.85,
        "map_then_rotate_accuracy_at_least_085": primary["map_then_rotate_accuracy"] >= 0.85,
        "support_disambiguation_accuracy_at_least_090": primary["support_disambiguation_accuracy"] >= 0.90,
        "ambiguous_case_abstain_or_repair_accuracy_at_least_080": primary["ambiguous_case_abstain_or_repair_accuracy"] >= 0.80,
        "decoy_heavy_composition_accuracy_at_least_082": primary["decoy_heavy_composition_accuracy"] >= 0.82,
        "trace_validity_at_least_095": primary["trace_validity"] >= 0.95,
        "wrong_writeback_rate_at_most_002": primary["wrong_writeback_rate"] <= 0.02,
        "destructive_overwrite_rate_at_most_002": primary["destructive_overwrite_rate"] <= 0.02,
        "branch_contamination_zero": primary["branch_contamination_rate"] == 0.0,
        "semantic_slot_leak_detected_false": primary["semantic_slot_leak_detected"] is False,
        "privileged_control_selected_as_primary_false": primary["privileged_control_selected_as_primary"] is False,
        "deterministic_replay_passed": replay_passed,
        "beats_random_matched_on_discovery_test_exact": primary["discovery_test_exact_accuracy"] > systems[RANDOM_MATCHED]["discovery_test_exact_accuracy"],
        "beats_random_best_of_n_on_discovery_test_exact": primary["discovery_test_exact_accuracy"] > systems[RANDOM_BEST]["discovery_test_exact_accuracy"],
        "beats_no_gate_on_trace_validity": primary["trace_validity"] > systems[NO_GATE]["trace_validity"],
        "beats_no_gate_on_wrong_writeback": primary["wrong_writeback_rate"] < systems[NO_GATE]["wrong_writeback_rate"],
    }


def check_gate(out: Path, failures: list[dict[str, Any]]) -> None:
    decision = load_json(out / "decision.json")
    aggregate = load_json(out / "aggregate_metrics.json")
    grammar = load_json(out / "e16b_operator_grammar_report.json")
    macro = load_json(out / "e16b_macro_removal_audit_report.json")
    library = load_json(out / "e16b_discovered_library_report.json")
    ablation = load_json(out / "e16b_ablation_report.json")
    semantic = load_json(out / "e16b_semantic_leak_audit_report.json")
    replay = load_json(out / "e16b_deterministic_replay_report.json")
    writeback = load_json(out / "e16b_writeback_safety_report.json")
    systems = aggregate.get("systems", {})
    family_metrics = aggregate.get("family_metrics", {})

    if decision.get("decision") not in VALID_DECISIONS:
        add_failure(failures, "INVALID_DECISION", str(decision.get("decision")))
    if decision.get("primary_system") != PRIMARY:
        add_failure(failures, "PRIMARY_SYSTEM_MISMATCH", str(decision.get("primary_system")))
    if decision.get("decision") == "e16b_text_flow_operator_composition_discovery_confirmed":
        if decision.get("next") != "E16C_TEXT_FLOW_OPERATOR_INVENTION_FROM_MICRO_PRIMITIVES_CONFIRM":
            add_failure(failures, "NEXT_MISMATCH_FOR_CONFIRMED", str(decision.get("next")))
    for system in REQUIRED_SYSTEMS:
        if system not in systems:
            add_failure(failures, "MISSING_SYSTEM", system)
    for family in REQUIRED_FAMILIES:
        if PRIMARY not in family_metrics or family not in family_metrics[PRIMARY]:
            add_failure(failures, "MISSING_PRIMARY_FAMILY", family)

    if grammar.get("primary_grammar_contains_only_allowed_primitives") is not True:
        add_failure(failures, "PRIMARY_GRAMMAR_NOT_ALLOWED_ONLY", "e16b_operator_grammar_report.json")
    for primitive in grammar.get("allowed_primitives", []):
        if primitive not in ALLOWED_PRIMITIVES:
            add_failure(failures, "UNEXPECTED_ALLOWED_PRIMITIVE", str(primitive))
    if grammar.get("direct_macro_operators_in_primary_grammar") not in ([], None):
        add_failure(failures, "DIRECT_MACRO_IN_PRIMARY_GRAMMAR", str(grammar.get("direct_macro_operators_in_primary_grammar")))
    if macro.get("macro_removed_confirmed") is not True:
        add_failure(failures, "MACRO_REMOVAL_NOT_CONFIRMED", "e16b_macro_removal_audit_report.json")
    if macro.get("direct_macro_leak_detected") is not False:
        add_failure(failures, "DIRECT_MACRO_LEAK_FLAG_NOT_FALSE", "e16b_macro_removal_audit_report.json")

    for record in library.get("primary_library", []):
        sequence = record.get("primitive_sequence", [])
        for item in sequence:
            if item not in ALLOWED_PRIMITIVES:
                add_failure(failures, "NON_PRIMITIVE_IN_PRIMARY_LIBRARY", str(record))
            if item in FORBIDDEN_MACROS:
                add_failure(failures, "FORBIDDEN_MACRO_IN_PRIMARY_LIBRARY", str(record))
        for field in (
            "program_id",
            "primitive_sequence",
            "chain_len",
            "evidence_fit_score",
            "support_coverage",
            "heldout_coverage",
            "conflict_count",
            "cost",
            "trace_validity",
            "reason_code",
        ):
            if field not in record:
                add_failure(failures, "MISSING_PROGRAM_FIELD", field)

    if semantic.get("semantic_slot_leak_detected") is not False:
        add_failure(failures, "SEMANTIC_LEAK_FLAG_NOT_FALSE", "e16b_semantic_leak_audit_report.json")
    if semantic.get("runtime_receives_task_family_labels") is not False:
        add_failure(failures, "RUNTIME_TASK_FAMILY_LABEL_FLAG_NOT_FALSE", "e16b_semantic_leak_audit_report.json")
    if semantic.get("runtime_receives_oracle_expected_output") is not False:
        add_failure(failures, "RUNTIME_ORACLE_FLAG_NOT_FALSE", "e16b_semantic_leak_audit_report.json")
    if semantic.get("privileged_control_selected_as_primary") is not False:
        add_failure(failures, "PRIVILEGED_CONTROL_PRIMARY_FLAG_NOT_FALSE", "e16b_semantic_leak_audit_report.json")
    if replay.get("internal_replay_passed") is not True:
        add_failure(failures, "DETERMINISTIC_REPLAY_FAILED", "e16b_deterministic_replay_report.json")
    if writeback.get("no_gate_failed_trace_or_writeback_safety") is not True:
        add_failure(failures, "NO_GATE_SAFETY_CONTRAST_MISSING", "e16b_writeback_safety_report.json")

    if failures:
        return

    expected = expected_gate(aggregate, replay.get("internal_replay_passed", False))
    reported = aggregate.get("positive_gate", {}).get("checks", {})
    for name, value in expected.items():
        if reported.get(name) is not value:
            add_failure(failures, "POSITIVE_GATE_MATH_MISMATCH", name)
    if aggregate.get("positive_gate", {}).get("passed") is not all(expected.values()):
        add_failure(failures, "POSITIVE_GATE_FLAG_MISMATCH", "aggregate_metrics.json")
    if decision.get("positive_gate_passed") is not aggregate.get("positive_gate", {}).get("passed"):
        add_failure(failures, "DECISION_GATE_FLAG_MISMATCH", "decision.json")

    primary = systems[PRIMARY]
    expected_cost_delta = round(1.0 - rate(primary["cost_per_tick"], systems[UNPRUNED]["cost_per_tick"]), 6)
    if aggregate.get("positive_gate", {}).get("deltas", {}).get("pruned_cost_reduction_vs_unpruned") != expected_cost_delta:
        add_failure(failures, "PRUNED_COST_DELTA_MISMATCH", str(aggregate.get("positive_gate", {}).get("deltas", {}).get("pruned_cost_reduction_vs_unpruned")))
    if primary["discovery_test_exact_accuracy"] <= systems[RANDOM_MATCHED]["discovery_test_exact_accuracy"]:
        add_failure(failures, "RANDOM_MATCHED_NOT_BEATEN", "discovery_test_exact_accuracy")
    if primary["discovery_test_exact_accuracy"] <= systems[RANDOM_BEST]["discovery_test_exact_accuracy"]:
        add_failure(failures, "RANDOM_BEST_NOT_BEATEN", "discovery_test_exact_accuracy")
    if primary["trace_validity"] <= systems[NO_GATE]["trace_validity"]:
        add_failure(failures, "NO_GATE_TRACE_NOT_WORSE", "trace_validity")
    if primary["wrong_writeback_rate"] >= systems[NO_GATE]["wrong_writeback_rate"]:
        add_failure(failures, "NO_GATE_WRITEBACK_NOT_WORSE", "wrong_writeback_rate")

    expectations = ablation.get("expectations", {})
    for key in (
        "insufficient_chain_len_materially_below_primary",
        "missing_order_fails_reverse_or_order",
        "missing_map_fails_map_families",
        "missing_filter_fails_filter_or_decoy",
        "ambiguous_no_abstain_higher_wrong_commit",
    ):
        if expectations.get(key) is not True:
            add_failure(failures, "ABLATION_EXPECTATION_FAILED", key)
    if systems["INSUFFICIENT_CHAIN_LEN_ABLATION"]["discovery_test_exact_accuracy"] > primary["discovery_test_exact_accuracy"] - 0.20:
        add_failure(failures, "INSUFFICIENT_CHAIN_LEN_TOO_STRONG", "discovery_test_exact_accuracy")
    if systems["MISSING_ORDER_PRIMITIVES_ABLATION"]["reverse_from_swaps_accuracy"] >= 0.50:
        add_failure(failures, "MISSING_ORDER_DID_NOT_FAIL_REVERSE", "reverse_from_swaps_accuracy")
    if systems["MISSING_MAP_PRIMITIVE_ABLATION"]["map_then_reverse_accuracy"] >= 0.50:
        add_failure(failures, "MISSING_MAP_DID_NOT_FAIL_MAP_FAMILY", "map_then_reverse_accuracy")
    if systems["MISSING_FILTER_PRIMITIVE_ABLATION"]["decoy_heavy_composition_accuracy"] >= 0.50:
        add_failure(failures, "MISSING_FILTER_DID_NOT_FAIL_DECOY", "decoy_heavy_composition_accuracy")
    if systems["AMBIGUOUS_SUPPORT_NO_ABSTAIN_ABLATION"]["ambiguous_no_abstain_wrong_commit_rate"] <= primary["ambiguous_no_abstain_wrong_commit_rate"]:
        add_failure(failures, "NO_ABSTAIN_WRONG_COMMIT_NOT_HIGHER", "ambiguous_no_abstain_wrong_commit_rate")
    if systems[MACRO_CONTROL]["direct_macro_leak_detected"] is not True:
        add_failure(failures, "MACRO_CONTROL_NOT_MARKED_PRIVILEGED_LEAK", MACRO_CONTROL)
    if systems[PRIMARY]["privileged_control_selected_as_primary"] is not False:
        add_failure(failures, "PRIVILEGED_PRIMARY_SELECTED", PRIMARY)


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
    check_boundary_text(out, failures)
    result = {
        "schema_version": "e16b_checker_result_v1",
        "out": str(out),
        "failure_count": len(failures),
        "failures": failures,
    }
    if write_summary:
        (out / "checker_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="target/pilot_wave/e16b_text_flow_operator_composition_discovery_confirm")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    result = check(Path(args.out), write_summary=args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
