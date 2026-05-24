#!/usr/bin/env python3
"""145H mixed structured-rule composition priority binding scale confirm."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_145h_mixed_structured_rule_composition_priority_binding_scale_confirm/smoke")
DEFAULT_145A_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_145a_mixed_structured_rule_composition_priority_binding_prototype/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
PHASE_145A_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_145a_mixed_structured_rule_composition_priority_binding_prototype.py"
NEW_DECODER = "deterministic_pocket_gated_mixed_structured_rule_composition_binding_decoder"
OLD_SELECTED_POCKET_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
OLD_STRUCTURED_RULE_DECODER = "deterministic_pocket_gated_structured_rule_metadata_binding_decoder"
POSITIVE_DECISION = "mixed_structured_rule_composition_priority_binding_scale_confirmed"
POSITIVE_VERDICT = "INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRMED"
POSITIVE_NEXT = "145Z_MIXED_STRUCTURED_RULE_COMPOSITION_NEXT_DECISION_PLAN"
REQUIRED_PRIORITY_ORDERS = {
    "recency>quorum>tie_break",
    "quorum>recency>tie_break",
    "tie_break>quorum>recency",
}
SUPPORTED_BLOCK_TYPES = {"quorum", "recency", "tie_break"}
POCKETS = {"pocket_a", "pocket_b", "pocket_c"}
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "rule_metadata_reasoning_claimed": False,
    "natural_language_rule_reasoning_claimed": False,
    "open_ended_arbitration_claimed": False,
    "gpt_like_readiness_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
    "architecture_superiority_claimed": False,
}
BOUNDARY_TEXT = (
    "145H is constrained helper/backend evidence only: mixed structured-rule composition with explicit priority over block types only, "
    "not natural-language rule reasoning, not open-ended arbitration, not GPT-like/open-domain capability, "
    "not production readiness, and not architecture superiority."
)
FALLBACK_FAMILIES = {
    "ALL_BLOCKS_INVALID_FALLBACK",
    "MISSING_PRIORITY_CONTROL",
    "DUPLICATE_PRIORITY_ENTRY_CONTROL",
    "UNKNOWN_PRIORITY_ENTRY_CONTROL",
    "PRIORITY_REFERENCES_MISSING_BLOCK_CONTROL",
    "MULTIPLE_PRIORITY_LINES_CONTROL",
    "DUPLICATE_RULE_BLOCK_TYPE_CONTROL",
    "MALFORMED_BLOCK_BOUNDARY_CONTROL",
    "METADATA_OUTSIDE_BLOCK_CONTROL",
    "NESTED_BLOCK_BOUNDARY_CONTROL",
    "EMPTY_RULE_BLOCK_CONTROL",
    "STRUCTURAL_INVALID_PROMPT_NO_FALLTHROUGH_CONTROL",
    "PRIORITY_POCKET_ORACLE_CONTROL",
    "RULE_COMPOSITION_CORRUPTION_CONTROL",
}
POSITIVE_FAMILIES = {
    "SINGLE_VALID_BLOCK_BASELINE",
    "MULTI_BLOCK_PRIORITY_RECENCY_WINS",
    "MULTI_BLOCK_PRIORITY_QUORUM_WINS",
    "MULTI_BLOCK_PRIORITY_TIE_BREAK_WINS",
    "INVALID_HIGH_PRIORITY_FALLS_THROUGH_TO_LOWER_PRIORITY",
    "SEMANTIC_INVALID_HIGH_PRIORITY_BLOCK_FALLTHROUGH_CONTROL",
    "SAME_BLOCKS_DIFFERENT_PRIORITY",
    "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
    "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_phase_145a() -> Any:
    spec = importlib.util.spec_from_file_location("phase_145a_mixed_rule_reuse", PHASE_145A_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load 145A runner module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase145a = load_phase_145a()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def resolve_target_out(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    relative = resolved.relative_to(REPO_ROOT)
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise ValueError("--out must stay under target/pilot_wave")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def append_progress(out: Path, event: str, **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "details": details})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def git_show_head(path: str) -> str:
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else ""


def require_145a(root: Path) -> dict[str, Any]:
    required = ["decision.json", "aggregate_metrics.json", "shared_helper_diff_audit.json", "summary.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 145A artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    helper_audit = read_json(root / "shared_helper_diff_audit.json")
    expected_exact = {
        "decision": "mixed_structured_rule_composition_priority_binding_prototype_positive",
        "verdict": "INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE_POSITIVE",
        "next": "145H_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM",
    }
    for key, expected in expected_exact.items():
        if decision.get(key) != expected:
            raise RuntimeError(f"bad 145A {key}: {decision.get(key)}")
    metric_exact = {
        "end_to_end_answer_accuracy": 1.0,
        "final_selected_pocket_derivation_accuracy": 1.0,
        "selected_pocket_to_marker_binding_accuracy": 1.0,
        "semantic_invalid_high_priority_fallthrough_accuracy": 1.0,
        "structural_invalid_prompt_fallback_rate": 1.0,
        "priority_pocket_oracle_rejection_rate": 1.0,
        "rule_composition_ablation_accuracy": 0.0,
        "distinct_block_candidate_coverage": True,
        "legacy_structured_rule_metadata_regression_passed": True,
        "legacy_selected_pocket_binding_regression_passed": True,
        "deterministic_replay_passed": True,
    }
    for key, expected in metric_exact.items():
        if metrics.get(key) != expected:
            raise RuntimeError(f"bad 145A metric {key}: {metrics.get(key)}")
    if not helper_audit.get("helper_source_sha256_after"):
        raise RuntimeError("145A helper diff audit missing helper_source_sha256_after")
    return {
        "schema_version": "phase_145h_upstream_145a_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": metrics,
        "shared_helper_diff_audit": helper_audit,
    }


def shared_helper_no_change_audit(upstream: dict[str, Any]) -> dict[str, Any]:
    current_source = HELPER_PATH.read_text(encoding="utf-8")
    head_source = git_show_head("scripts/probes/shared_raw_generation_helper.py")
    current_hash = sha256_text(current_source)
    upstream_hash = upstream["shared_helper_diff_audit"]["helper_source_sha256_after"]
    audit = {
        "schema_version": "phase_145h_shared_helper_no_change_audit_v1",
        "current_shared_helper_sha256": current_hash,
        "upstream_145a_shared_helper_sha256": upstream_hash,
        "shared_helper_no_change_since_145a": current_hash == upstream_hash,
        "shared_helper_modified_by_145h": current_source != head_source,
        "shared_raw_generation_helper_unchanged_from_head": current_source == head_source,
        "passed": False,
    }
    audit["passed"] = (
        audit["shared_helper_no_change_since_145a"] is True
        and audit["shared_helper_modified_by_145h"] is False
        and audit["shared_raw_generation_helper_unchanged_from_head"] is True
    )
    return audit


def helper_mixed_rule_semantics_audit() -> dict[str, Any]:
    source = HELPER_PATH.read_text(encoding="utf-8")
    required_terms = {
        "mixed_decoder_string_present": NEW_DECODER in source,
        "mixed_decoder_constant_present": "MIXED_STRUCTURED_RULE_COMPOSITION_BINDING_DECODER" in source,
        "mixed_selection_function_present": "_instnct_select_mixed_structured_rule_composition_value" in source,
        "mixed_parser_helper_present": "_instnct_parse_mixed_rule_composition" in source,
        "mixed_priority_helper_present": "_instnct_parse_mixed_priority" in source,
        "priority_block_type_guard_present": "priority_pocket_oracle" in source and "unknown_priority_entry" in source,
        "strict_boundary_terms_present": all(term in source for term in ["missing_block_end", "nested_rule_block_before_block_end", "metadata_outside_block", "duplicate_rule_block_type", "empty_rule_block"]),
        "old_selected_pocket_decoder_present": OLD_SELECTED_POCKET_DECODER in source,
        "old_structured_rule_decoder_present": OLD_STRUCTURED_RULE_DECODER in source,
        "request_validation_present": "def validate_request" in source and "ALLOWED_REQUEST_KEYS" in source,
        "no_training_or_network_terms_present": not any(term in source for term in ["torch.optim", "import socket", "import requests", "urllib.request", "http.client"]),
    }
    return {
        "schema_version": "phase_145h_helper_mixed_rule_semantics_audit_v1",
        **required_terms,
        "passed": all(required_terms.values()),
    }


def shared_helper_diff_audit_compat(no_change: dict[str, Any], semantics: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_145h_shared_helper_diff_audit_v1",
        "source_changed": False,
        "new_mixed_decoder_string_present": semantics["mixed_decoder_string_present"],
        "new_mixed_selection_function_present": semantics["mixed_selection_function_present"],
        "new_mixed_parser_helpers_present": semantics["mixed_parser_helper_present"] and semantics["mixed_priority_helper_present"],
        "new_behavior_manifest_gated": semantics["mixed_decoder_constant_present"],
        "old_selected_pocket_decoder_still_present": semantics["old_selected_pocket_decoder_present"],
        "old_structured_metadata_decoder_still_present": semantics["old_structured_rule_decoder_present"],
        "validate_request_unchanged": no_change["shared_raw_generation_helper_unchanged_from_head"],
        "allowed_request_keys_unchanged": no_change["shared_raw_generation_helper_unchanged_from_head"],
        "forbidden_request_keys_not_loosened": no_change["shared_raw_generation_helper_unchanged_from_head"],
        "no_training_import_added": semantics["no_training_or_network_terms_present"],
        "no_network_or_io_added": semantics["no_training_or_network_terms_present"],
        "helper_source_sha256_after": no_change["current_shared_helper_sha256"],
        "shared_helper_no_change_since_145a": no_change["shared_helper_no_change_since_145a"],
        "passed": no_change["passed"] and semantics["passed"],
    }


def positive_vs_fallback_denominator_report(main_scored: list[dict[str, Any]]) -> dict[str, Any]:
    mixed = [row for row in main_scored if row["decoder_arm"] == "mixed_rule_composition"]
    positive = [row for row in mixed if row["expected_derive_success"] is True]
    fallback = [row for row in mixed if row["expected_fallback"] is True]
    positive_ids = {row["row_id"] for row in positive}
    fallback_ids = {row["row_id"] for row in fallback}
    overlap = sorted(positive_ids & fallback_ids)
    positive_accuracy = phase145a.fraction(positive, "final_answer_correct")
    fallback_accuracy = phase145a.fallback_rate(fallback)
    report = {
        "schema_version": "phase_145h_positive_vs_fallback_denominator_report_v1",
        "positive_composition_row_count": len(positive),
        "expected_fallback_row_count": len(fallback),
        "end_to_end_answer_denominator_row_count": len(positive),
        "fallback_rows_in_positive_denominator": len(overlap),
        "fallback_rows_in_positive_denominator_sample": overlap[:20],
        "expected_fallback_rows_excluded_from_end_to_end_answer_accuracy": not overlap,
        "positive_composition_subset_accuracy": positive_accuracy,
        "fallback_control_subset_accuracy": fallback_accuracy,
        "passed": not overlap and positive_accuracy >= 0.98 and fallback_accuracy >= 0.98,
    }
    return report


def priority_order_coverage_report(rows: list[dict[str, Any]], main_scored: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["row_id"]: row for row in rows}
    successful = [row for row in main_scored if row["decoder_arm"] == "mixed_rule_composition" and row["expected_derive_success"] is True and row.get("final_selected_pocket_derivation_correct") is True]
    orders = {">".join(row.get("expected_priority_order") or []) for row in successful if row.get("expected_priority_order")}
    winning_block_types: set[str] = set()
    winning_pockets_by_block = {block: set() for block in SUPPORTED_BLOCK_TYPES}
    for scored in successful:
        original = by_id[scored["row_id"]]
        candidates = original.get("expected_block_candidates") or {}
        for block in original.get("expected_priority_order") or []:
            candidate = candidates.get(block)
            if candidate:
                winning_block_types.add(block)
                winning_pockets_by_block.setdefault(block, set()).add(candidate)
                break
    all_orders = REQUIRED_PRIORITY_ORDERS <= orders
    all_block_types = SUPPORTED_BLOCK_TYPES <= winning_block_types
    all_pockets = all(POCKETS <= winning_pockets_by_block.get(block, set()) for block in SUPPORTED_BLOCK_TYPES)
    return {
        "schema_version": "phase_145h_priority_order_coverage_report_v1",
        "covered_priority_orders": sorted(orders),
        "required_priority_orders": sorted(REQUIRED_PRIORITY_ORDERS),
        "winning_block_types": sorted(winning_block_types),
        "winning_pockets_by_block_type": {key: sorted(value) for key, value in winning_pockets_by_block.items()},
        "all_priority_orders_covered": all_orders,
        "all_block_types_win_under_priority": all_block_types,
        "all_pockets_win_under_each_supported_block_type": all_pockets,
        "passed": all_orders and all_block_types and all_pockets,
    }


def block_type_candidate_coverage_report(rows: list[dict[str, Any]], main_scored: list[dict[str, Any]]) -> dict[str, Any]:
    candidates_by_block = {block: set() for block in SUPPORTED_BLOCK_TYPES}
    for row in rows:
        if row.get("decoder_arm") != "mixed_rule_composition":
            continue
        for block, candidate in (row.get("expected_block_candidates") or {}).items():
            if block in candidates_by_block and candidate:
                candidates_by_block[block].add(candidate)
    same_blocks = [row for row in main_scored if row["family"] == "SAME_BLOCKS_DIFFERENT_PRIORITY"]
    distinct = all(row.get("distinct_block_candidate_row") is True for row in same_blocks)
    priority_changes = phase145a.fraction(same_blocks, "priority_only_changes_winner_correct")
    all_pockets = all(POCKETS <= candidates_by_block.get(block, set()) for block in SUPPORTED_BLOCK_TYPES)
    return {
        "schema_version": "phase_145h_block_type_candidate_coverage_report_v1",
        "candidate_pockets_by_block_type": {key: sorted(value) for key, value in candidates_by_block.items()},
        "same_blocks_different_priority_row_count": len(same_blocks),
        "same_blocks_distinct_candidate_coverage": distinct,
        "priority_only_changes_winner_accuracy": priority_changes,
        "all_pockets_win_under_each_supported_block_type": all_pockets,
        "distinct_block_candidate_coverage": distinct,
        "passed": all_pockets and distinct and priority_changes >= 0.98,
    }


def aggregate_metrics_145h(
    main_scored: list[dict[str, Any]],
    ablation_scored: list[dict[str, Any]],
    legacy_structured: dict[str, Any],
    legacy_selected: dict[str, Any],
    request_audit: dict[str, Any],
    static_report: dict[str, Any],
    deterministic: bool,
    denominator: dict[str, Any],
    priority_coverage: dict[str, Any],
    block_coverage: dict[str, Any],
    no_change: dict[str, Any],
) -> dict[str, Any]:
    metrics = phase145a.aggregate_metrics(main_scored, ablation_scored, legacy_structured, legacy_selected, request_audit, static_report, deterministic)
    metrics["schema_version"] = "phase_145h_aggregate_metrics_v1"
    metrics["positive_composition_subset_accuracy"] = denominator["positive_composition_subset_accuracy"]
    metrics["fallback_control_subset_accuracy"] = denominator["fallback_control_subset_accuracy"]
    metrics["all_priority_orders_covered"] = priority_coverage["all_priority_orders_covered"]
    metrics["all_block_types_win_under_priority"] = priority_coverage["all_block_types_win_under_priority"]
    metrics["all_pockets_win_under_each_supported_block_type"] = priority_coverage["all_pockets_win_under_each_supported_block_type"] and block_coverage["all_pockets_win_under_each_supported_block_type"]
    metrics["shared_helper_no_change_since_145a"] = no_change["shared_helper_no_change_since_145a"]
    return metrics


def per_seed_gate_report(main_scored: list[dict[str, Any]]) -> dict[str, Any]:
    reports = []
    for seed in sorted({row["seed"] for row in main_scored}):
        subset = [row for row in main_scored if row["seed"] == seed]
        mixed = [row for row in subset if row["decoder_arm"] == "mixed_rule_composition"]
        positive = [row for row in mixed if row["expected_derive_success"] is True]
        fallback = [row for row in mixed if row["expected_fallback"] is True]
        row = {
            "seed": seed,
            "row_count": len(subset),
            "positive_composition_subset_accuracy": phase145a.fraction(positive, "final_answer_correct"),
            "fallback_control_subset_accuracy": phase145a.fallback_rate(fallback),
            "semantic_invalid_high_priority_fallthrough_accuracy": phase145a.fraction([item for item in subset if item["family"] == "SEMANTIC_INVALID_HIGH_PRIORITY_BLOCK_FALLTHROUGH_CONTROL"], "final_selected_pocket_derivation_correct"),
            "structural_invalid_prompt_fallback_rate": phase145a.fallback_rate([item for item in subset if item["family"] == "STRUCTURAL_INVALID_PROMPT_NO_FALLTHROUGH_CONTROL"]),
            "priority_pocket_oracle_rejection_rate": phase145a.fallback_rate([item for item in subset if item["family"] == "PRIORITY_POCKET_ORACLE_CONTROL"]),
        }
        row["passed"] = all(row[key] >= 0.98 for key in row if key.endswith("_accuracy") or key.endswith("_rate"))
        reports.append(row)
    return {
        "schema_version": "phase_145h_per_seed_gate_report_v1",
        "seed_reports": reports,
        "passed": all(row["passed"] for row in reports),
    }


def per_family_gate_report(main_scored: list[dict[str, Any]]) -> dict[str, Any]:
    reports = []
    for family in sorted({row["family"] for row in main_scored}):
        subset = [row for row in main_scored if row["family"] == family]
        if family in FALLBACK_FAMILIES:
            metric_name = "fallback_rate"
            value = phase145a.fallback_rate(subset)
        else:
            metric_name = "final_answer_accuracy"
            value = phase145a.fraction(subset, "final_answer_correct")
        reports.append({"family": family, "row_count": len(subset), metric_name: value, "passed": value >= 0.98})
    return {
        "schema_version": "phase_145h_per_family_gate_report_v1",
        "family_reports": reports,
        "passed": all(row["passed"] for row in reports),
    }


def fallback_gates_pass(metrics: dict[str, Any]) -> bool:
    keys = [
        "all_blocks_invalid_fallback_rate",
        "missing_priority_fallback_rate",
        "duplicate_priority_rejection_rate",
        "unknown_priority_rejection_rate",
        "missing_block_reference_rejection_rate",
        "multiple_priority_lines_rejection_rate",
        "duplicate_rule_block_type_rejection_rate",
        "malformed_block_boundary_rejection_rate",
        "metadata_outside_block_rejection_rate",
        "nested_block_boundary_rejection_rate",
        "empty_rule_block_rejection_rate",
        "structural_invalid_prompt_fallback_rate",
        "priority_pocket_oracle_rejection_rate",
    ]
    return all(metrics.get(key, 0.0) >= 0.98 for key in keys)


def choose_decision(
    metrics: dict[str, Any],
    no_change: dict[str, Any],
    semantics: dict[str, Any],
    request_audit: dict[str, Any],
    prompt_report: dict[str, Any],
    static_report: dict[str, Any],
    denominator: dict[str, Any],
    priority_coverage: dict[str, Any],
    block_coverage: dict[str, Any],
    per_seed: dict[str, Any],
    per_family: dict[str, Any],
) -> dict[str, Any]:
    integrity = all(
        report.get("passed") is True
        for report in [no_change, semantics, request_audit, prompt_report, static_report, denominator, priority_coverage, block_coverage, per_seed, per_family]
    )
    gates = (
        metrics["mixed_rule_block_parse_accuracy"] >= 0.98
        and metrics["per_block_candidate_derivation_accuracy"] >= 0.98
        and metrics["priority_policy_parse_accuracy"] >= 0.98
        and metrics["final_selected_pocket_derivation_accuracy"] >= 0.98
        and metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.98
        and metrics["same_line_value_extraction_accuracy"] >= 0.98
        and metrics["end_to_end_answer_accuracy"] >= 0.98
        and metrics["positive_composition_subset_accuracy"] >= 0.98
        and metrics["fallback_control_subset_accuracy"] >= 0.98
        and metrics["invalid_high_priority_fallthrough_accuracy"] >= 0.98
        and metrics["semantic_invalid_high_priority_fallthrough_accuracy"] >= 0.98
        and fallback_gates_pass(metrics)
        and metrics["same_blocks_different_priority_accuracy"] >= 0.98
        and metrics["priority_only_changes_winner_accuracy"] >= 0.98
        and metrics["all_priority_orders_covered"] is True
        and metrics["all_block_types_win_under_priority"] is True
        and metrics["all_pockets_win_under_each_supported_block_type"] is True
        and metrics["distinct_block_candidate_coverage"] is True
        and metrics["rule_composition_ablation_accuracy"] <= 0.05
        and metrics["helper_request_forbidden_metadata_count"] == 0
        and metrics["per_row_manifest_switch_rate"] == 0.0
        and metrics["per_row_payload_marker_switch_rate"] == 0.0
        and metrics["shared_helper_no_change_since_145a"] is True
        and metrics["legacy_structured_rule_metadata_regression_passed"] is True
        and metrics["legacy_selected_pocket_binding_regression_passed"] is True
        and metrics["deterministic_replay_passed"] is True
    )
    positive = integrity and gates
    if not integrity or no_change.get("passed") is not True or semantics.get("passed") is not True:
        decision = "helper_integrity_failure"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif metrics["mixed_rule_block_parse_accuracy"] < 0.98:
        decision = "mixed_rule_block_parse_scale_failure"; next_step = "145B_MIXED_RULE_BLOCK_PARSE_FAILURE_ANALYSIS"
    elif metrics["priority_policy_parse_accuracy"] < 0.98:
        decision = "priority_policy_parse_scale_failure"; next_step = "145C_PRIORITY_POLICY_PARSE_FAILURE_ANALYSIS"
    elif metrics["final_selected_pocket_derivation_accuracy"] < 0.98:
        decision = "final_selected_pocket_derivation_scale_failure"; next_step = "145D_MIXED_RULE_FINAL_SELECTION_FAILURE_ANALYSIS"
    elif metrics["rule_composition_ablation_accuracy"] > 0.05 or metrics["priority_pocket_oracle_rejection_rate"] < 0.98:
        decision = "rule_composition_oracle_shortcut_detected"; next_step = "145E_RULE_COMPOSITION_ORACLE_SHORTCUT_ANALYSIS"
    elif not fallback_gates_pass(metrics):
        decision = "priority_ambiguity_not_rejected"; next_step = "145F_PRIORITY_AMBIGUITY_ANALYSIS"
    elif metrics["invalid_high_priority_fallthrough_accuracy"] < 0.98 or metrics["semantic_invalid_high_priority_fallthrough_accuracy"] < 0.98:
        decision = "invalid_block_fallthrough_scale_failure"; next_step = "145G_INVALID_BLOCK_FALLTHROUGH_ANALYSIS"
    elif metrics["legacy_structured_rule_metadata_regression_passed"] is not True:
        decision = "legacy_structured_rule_metadata_regression"; next_step = "144C_STRUCTURED_RULE_METADATA_PARSE_FAILURE_ANALYSIS"
    elif metrics["selected_pocket_to_marker_binding_accuracy"] < 0.98 or metrics["legacy_selected_pocket_binding_regression_passed"] is not True:
        decision = "selected_pocket_binding_regression"; next_step = "143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS"
    elif positive:
        decision = POSITIVE_DECISION; next_step = POSITIVE_NEXT
    else:
        decision = "helper_integrity_failure"; next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    return {
        "schema_version": "phase_145h_decision_v1",
        "decision": decision,
        "verdict": POSITIVE_VERDICT if decision == POSITIVE_DECISION else "INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRM_BLOCKED",
        "next": next_step,
        "positive_gate_passed": positive,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_metric_report(out: Path, name: str, metric_name: str, value: Any, passed: bool, extra: dict[str, Any] | None = None) -> None:
    write_json(out / name, {"schema_version": f"phase_145h_{name.replace('.json', '')}_v1", metric_name: value, "passed": passed, **(extra or {})})


def write_family_report(out: Path, name: str, scored: list[dict[str, Any]], family: str, metric_name: str, metric_key: str | None = None, *, fallback_metric: bool = False) -> None:
    subset = phase145a.scoped(scored, family)
    value = phase145a.fallback_rate(subset) if fallback_metric else phase145a.fraction(subset, metric_key or metric_name)
    write_json(out / name, {"schema_version": f"phase_145h_{metric_name}_report_v1", "family": family, "row_count": len(subset), metric_name: value, "fallback_rate": phase145a.fallback_rate(subset), "passed": value >= 0.98})


def write_report(out: Path, decision: dict[str, Any], metrics: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Verdict: `{decision['verdict']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Key Metrics

- main eval rows: `{metrics['main_eval_rows']}`
- mixed rule block parse accuracy: `{metrics['mixed_rule_block_parse_accuracy']}`
- per-block candidate derivation accuracy: `{metrics['per_block_candidate_derivation_accuracy']}`
- priority policy parse accuracy: `{metrics['priority_policy_parse_accuracy']}`
- final selected pocket derivation accuracy: `{metrics['final_selected_pocket_derivation_accuracy']}`
- selected pocket to marker binding accuracy: `{metrics['selected_pocket_to_marker_binding_accuracy']}`
- same-line value extraction accuracy: `{metrics['same_line_value_extraction_accuracy']}`
- end-to-end answer accuracy: `{metrics['end_to_end_answer_accuracy']}`
- positive composition subset accuracy: `{metrics['positive_composition_subset_accuracy']}`
- fallback control subset accuracy: `{metrics['fallback_control_subset_accuracy']}`
- semantic invalid high-priority fallthrough accuracy: `{metrics['semantic_invalid_high_priority_fallthrough_accuracy']}`
- structural invalid prompt fallback rate: `{metrics['structural_invalid_prompt_fallback_rate']}`
- priority pocket oracle rejection rate: `{metrics['priority_pocket_oracle_rejection_rate']}`
- shared helper no change since 145A: `{metrics['shared_helper_no_change_since_145a']}`
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 145H mixed structured-rule composition scale confirm")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-145a-root", type=Path, default=DEFAULT_145A_ROOT)
    parser.add_argument("--seeds", default="5301,5302,5303,5304")
    parser.add_argument("--groups-per-family", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_145h_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_145a(resolve_repo_path(args.upstream_145a_root))
    write_json(out / "upstream_145a_manifest.json", upstream)
    append_progress(out, "upstream verified", upstream_decision=upstream["decision"]["decision"])

    helper = phase145a.load_helper()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    rows = phase145a.build_rows(seeds, args.groups_per_family, args.group_size)
    append_progress(out, "rows built", row_count=len(rows), families=len(phase145a.FAMILIES))
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_145h_analysis_config_v1",
            "milestone": MILESTONE,
            "boundary": BOUNDARY_TEXT,
            "seeds": seeds,
            "families": phase145a.FAMILIES,
            "groups_per_family": args.groups_per_family,
            "group_size": args.group_size,
            "main_eval_rows": len(rows),
            "decoder": NEW_DECODER,
            "scale_confirm_only": True,
            "shared_helper_modification_allowed": False,
            **FALSE_FLAGS,
        },
    )

    mixed_path, mixed_manifest = phase145a.build_manifest(out, "mixed_rule_composition_145h", decoder_type=NEW_DECODER)
    structured_path, structured_manifest = phase145a.build_manifest(out, "legacy_structured_rule_145h", decoder_type=OLD_STRUCTURED_RULE_DECODER, gate_marker=phase145a.STRUCTURED_GATE)
    selected_path, selected_manifest = phase145a.build_manifest(out, "legacy_selected_pocket_145h", decoder_type=OLD_SELECTED_POCKET_DECODER, gate_marker=phase145a.STRUCTURED_GATE)
    request_audit_rows: list[dict[str, Any]] = []
    mixed_rows = [row for row in rows if row["decoder_arm"] == "mixed_rule_composition"]
    structured_rows = [row for row in rows if row["decoder_arm"] == "old_structured_rule_metadata"]
    selected_rows = [row for row in rows if row["decoder_arm"] == "old_selected_pocket"]

    mixed_results = phase145a.run_arm(helper, out, "mixed_rule_composition_scale_main", mixed_rows, mixed_path, mixed_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    structured_results = phase145a.run_arm(helper, out, "legacy_structured_rule_metadata_scale", structured_rows, structured_path, structured_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    selected_results = phase145a.run_arm(helper, out, "legacy_selected_pocket_binding_scale", selected_rows, selected_path, selected_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    main_results = mixed_results + structured_results + selected_results
    main_scored = phase145a.score(mixed_rows, mixed_results) + phase145a.score(structured_rows, structured_results) + phase145a.score(selected_rows, selected_results)
    replay_results = phase145a.run_arm(helper, out, "mixed_rule_composition_scale_replay", mixed_rows, mixed_path, mixed_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    deterministic = [row["generated_text_hash"] for row in mixed_results] == [row["generated_text_hash"] for row in replay_results]

    ablation_rows = []
    for row in mixed_rows:
        if row["expected_derive_success"]:
            ablated = dict(row)
            ablated["prompt"] = phase145a.build_mixed_prompt(row, row["mixed_rule_lines"], omit_rules=True)
            ablated["expected_fallback"] = True
            ablated["expected_parse_success"] = False
            ablated["expected_derive_success"] = False
            ablated["expected_failure_reason"] = "missing_rule_block"
            ablation_rows.append(ablated)
    ablation_results = phase145a.run_arm(helper, out, "rule_composition_scale_ablation", ablation_rows, mixed_path, mixed_manifest["checkpoint_sha256"], args.max_new_tokens, args.heartbeat_sec, request_audit_rows)
    ablation_scored = phase145a.score(ablation_rows, ablation_results)
    append_progress(out, "generation complete", deterministic=deterministic)

    no_change = shared_helper_no_change_audit(upstream)
    semantics = helper_mixed_rule_semantics_audit()
    helper_diff = shared_helper_diff_audit_compat(no_change, semantics)
    request_audit = phase145a.helper_request_audit(request_audit_rows)
    static_report = phase145a.static_manifest_integrity_report([mixed_manifest, structured_manifest, selected_manifest])
    legacy_structured_accuracy = phase145a.fraction(phase145a.scoped(main_scored, "LEGACY_STRUCTURED_RULE_METADATA_REGRESSION_CONTROL"), "final_answer_correct")
    legacy_selected_accuracy = phase145a.fraction(phase145a.scoped(main_scored, "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL"), "final_answer_correct")
    legacy_structured = {
        "schema_version": "phase_145h_legacy_structured_rule_metadata_regression_report_v1",
        "legacy_structured_rule_metadata_accuracy": legacy_structured_accuracy,
        "legacy_structured_rule_metadata_regression_passed": legacy_structured_accuracy >= 0.98,
        "passed": legacy_structured_accuracy >= 0.98,
    }
    legacy_selected = {
        "schema_version": "phase_145h_legacy_selected_pocket_binding_regression_report_v1",
        "legacy_selected_pocket_binding_accuracy": legacy_selected_accuracy,
        "legacy_selected_pocket_binding_regression_passed": legacy_selected_accuracy >= 0.98,
        "passed": legacy_selected_accuracy >= 0.98,
    }
    denominator = positive_vs_fallback_denominator_report(main_scored)
    priority_coverage = priority_order_coverage_report(rows, main_scored)
    block_coverage = block_type_candidate_coverage_report(rows, main_scored)
    metrics = aggregate_metrics_145h(main_scored, ablation_scored, legacy_structured, legacy_selected, request_audit, static_report, deterministic, denominator, priority_coverage, block_coverage, no_change)
    prompt_report = phase145a.prompt_scanner_report(rows, main_scored)
    per_seed = per_seed_gate_report(main_scored)
    per_family = per_family_gate_report(main_scored)
    decision = choose_decision(metrics, no_change, semantics, request_audit, prompt_report, static_report, denominator, priority_coverage, block_coverage, per_seed, per_family)
    summary = {"schema_version": "phase_145h_summary_v1", "milestone": MILESTONE, "status": "complete", "boundary": BOUNDARY_TEXT, "decision": decision, "aggregate_metrics": metrics, **FALSE_FLAGS}

    write_jsonl(out / "main_results.jsonl", main_results)
    write_jsonl(out / "main_scoring.jsonl", main_scored)
    write_json(out / "shared_helper_no_change_audit.json", no_change)
    write_json(out / "helper_mixed_rule_semantics_audit.json", semantics)
    write_json(out / "shared_helper_diff_audit.json", helper_diff)
    write_json(out / "positive_vs_fallback_denominator_report.json", denominator)
    write_json(out / "priority_order_coverage_report.json", priority_coverage)
    write_json(out / "block_type_candidate_coverage_report.json", block_coverage)
    write_json(out / "per_seed_gate_report.json", per_seed)
    write_json(out / "per_family_gate_report.json", per_family)
    write_json(out / "prompt_scanner_report.json", prompt_report)
    write_json(out / "helper_request_audit.json", request_audit)
    write_json(out / "static_manifest_integrity_report.json", static_report)
    write_json(out / "legacy_structured_rule_metadata_regression_report.json", legacy_structured)
    write_json(out / "legacy_selected_pocket_binding_regression_report.json", legacy_selected)
    write_metric_report(out, "mixed_rule_block_parser_report.json", "mixed_rule_block_parse_accuracy", metrics["mixed_rule_block_parse_accuracy"], metrics["mixed_rule_block_parse_accuracy"] >= 0.98)
    write_metric_report(out, "per_block_candidate_derivation_report.json", "per_block_candidate_derivation_accuracy", metrics["per_block_candidate_derivation_accuracy"], metrics["per_block_candidate_derivation_accuracy"] >= 0.98)
    write_metric_report(out, "priority_policy_report.json", "priority_policy_parse_accuracy", metrics["priority_policy_parse_accuracy"], metrics["priority_policy_parse_accuracy"] >= 0.98)
    write_metric_report(out, "final_selected_pocket_derivation_report.json", "final_selected_pocket_derivation_accuracy", metrics["final_selected_pocket_derivation_accuracy"], metrics["final_selected_pocket_derivation_accuracy"] >= 0.98)
    write_metric_report(out, "selected_pocket_binding_report.json", "selected_pocket_to_marker_binding_accuracy", metrics["selected_pocket_to_marker_binding_accuracy"], metrics["selected_pocket_to_marker_binding_accuracy"] >= 0.98)
    write_metric_report(out, "same_line_value_extraction_report.json", "same_line_value_extraction_accuracy", metrics["same_line_value_extraction_accuracy"], metrics["same_line_value_extraction_accuracy"] >= 0.98)
    family_reports = [
        ("invalid_high_priority_fallthrough_report.json", "INVALID_HIGH_PRIORITY_FALLS_THROUGH_TO_LOWER_PRIORITY", "invalid_high_priority_fallthrough_accuracy", "final_selected_pocket_derivation_correct", False),
        ("all_blocks_invalid_fallback_report.json", "ALL_BLOCKS_INVALID_FALLBACK", "all_blocks_invalid_fallback_rate", None, True),
        ("missing_priority_report.json", "MISSING_PRIORITY_CONTROL", "missing_priority_fallback_rate", None, True),
        ("duplicate_priority_entry_report.json", "DUPLICATE_PRIORITY_ENTRY_CONTROL", "duplicate_priority_rejection_rate", None, True),
        ("unknown_priority_entry_report.json", "UNKNOWN_PRIORITY_ENTRY_CONTROL", "unknown_priority_rejection_rate", None, True),
        ("missing_block_reference_report.json", "PRIORITY_REFERENCES_MISSING_BLOCK_CONTROL", "missing_block_reference_rejection_rate", None, True),
        ("multiple_priority_lines_report.json", "MULTIPLE_PRIORITY_LINES_CONTROL", "multiple_priority_lines_rejection_rate", None, True),
        ("duplicate_rule_block_type_report.json", "DUPLICATE_RULE_BLOCK_TYPE_CONTROL", "duplicate_rule_block_type_rejection_rate", None, True),
        ("malformed_block_boundary_report.json", "MALFORMED_BLOCK_BOUNDARY_CONTROL", "malformed_block_boundary_rejection_rate", None, True),
        ("metadata_outside_block_report.json", "METADATA_OUTSIDE_BLOCK_CONTROL", "metadata_outside_block_rejection_rate", None, True),
        ("nested_block_boundary_report.json", "NESTED_BLOCK_BOUNDARY_CONTROL", "nested_block_boundary_rejection_rate", None, True),
        ("empty_rule_block_report.json", "EMPTY_RULE_BLOCK_CONTROL", "empty_rule_block_rejection_rate", None, True),
        ("priority_pocket_oracle_report.json", "PRIORITY_POCKET_ORACLE_CONTROL", "priority_pocket_oracle_rejection_rate", None, True),
        ("same_blocks_different_priority_report.json", "SAME_BLOCKS_DIFFERENT_PRIORITY", "same_blocks_different_priority_accuracy", "final_selected_pocket_derivation_correct", False),
        ("same_priority_different_block_values_report.json", "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES", "same_priority_different_block_values_accuracy", "final_answer_correct", False),
        ("same_template_opposite_priority_winner_report.json", "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER", "same_template_opposite_priority_winner_accuracy", "final_answer_correct", False),
    ]
    for filename, family, metric_name, key, is_fallback in family_reports:
        write_family_report(out, filename, main_scored, family, metric_name, key, fallback_metric=is_fallback)
    write_metric_report(out, "semantic_invalid_high_priority_fallthrough_report.json", "semantic_invalid_high_priority_fallthrough_accuracy", metrics["semantic_invalid_high_priority_fallthrough_accuracy"], metrics["semantic_invalid_high_priority_fallthrough_accuracy"] >= 0.98)
    write_metric_report(out, "structural_invalid_prompt_fallback_report.json", "structural_invalid_prompt_fallback_rate", metrics["structural_invalid_prompt_fallback_rate"], metrics["structural_invalid_prompt_fallback_rate"] >= 0.98)
    write_json(out / "rule_composition_ablation_report.json", {"schema_version": "phase_145h_rule_composition_ablation_report_v1", "row_count": len(ablation_scored), "rule_composition_ablation_accuracy": metrics["rule_composition_ablation_accuracy"], "rule_composition_ablation_fallback_rate": metrics["rule_composition_ablation_fallback_rate"], "passed": metrics["rule_composition_ablation_accuracy"] <= 0.05})
    write_json(out / "rule_composition_oracle_shortcut_report.json", {"schema_version": "phase_145h_rule_composition_oracle_shortcut_report_v1", "priority_pocket_oracle_rejection_rate": metrics["priority_pocket_oracle_rejection_rate"], "rule_composition_ablation_accuracy": metrics["rule_composition_ablation_accuracy"], "passed": metrics["priority_pocket_oracle_rejection_rate"] >= 0.98 and metrics["rule_composition_ablation_accuracy"] <= 0.05})
    write_json(out / "aggregate_metrics.json", metrics)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, metrics)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_145h_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
