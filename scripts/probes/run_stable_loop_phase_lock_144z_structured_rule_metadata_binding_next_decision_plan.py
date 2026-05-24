#!/usr/bin/env python3
"""144Z planning-only next decision after structured rule metadata scale confirm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_144Z_STRUCTURED_RULE_METADATA_BINDING_NEXT_DECISION_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_144z_structured_rule_metadata_binding_next_decision_plan/smoke")
DEFAULT_144H_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_144h_structured_rule_metadata_to_selected_pocket_binding_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_144z_structured_rule_metadata_binding_next_decision_plan_check.py"
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
DECISION = "structured_rule_composition_priority_binding_prototype_plan_recommended"
SELECTED_OPTION = "mixed_structured_rule_composition_priority_binding_prototype"
NEXT = "145A_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_PROTOTYPE"
NEW_DECODER = "deterministic_pocket_gated_mixed_structured_rule_composition_binding_decoder"
OLD_SELECTED_POCKET_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
OLD_STRUCTURED_RULE_DECODER = "deterministic_pocket_gated_structured_rule_metadata_binding_decoder"
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
    "144Z is planning-only and artifact-only after the positive 144H scale confirm. "
    "It is constrained helper/backend evidence only for structured rule metadata binding next decisions; "
    "it is not natural-language rule reasoning, not open-ended arbitration, "
    "not GPT-like/open-domain/broad assistant capability, "
    "not production/public API/deployment/safety readiness, and not architecture superiority."
)
EXPECTED_144H_METRICS = {
    "main_eval_rows": 6528,
    "rule_metadata_parse_accuracy": 1.0,
    "derived_selected_pocket_accuracy": 1.0,
    "selected_pocket_to_marker_binding_accuracy": 1.0,
    "same_line_value_extraction_accuracy": 1.0,
    "end_to_end_answer_accuracy": 1.0,
    "rule_metadata_ablation_accuracy": 0.0,
    "wrong_family_extra_key_rejection_rate": 1.0,
    "quorum_clear_winner_ignores_tie_break_accuracy": 1.0,
    "legacy_143w_binding_regression_passed": True,
    "shared_helper_no_change_since_144b": True,
    "deterministic_replay_passed": True,
}
REQUIRED_145A_SUBSETS = [
    "SINGLE_VALID_BLOCK_BASELINE",
    "MULTI_BLOCK_PRIORITY_RECENCY_WINS",
    "MULTI_BLOCK_PRIORITY_QUORUM_WINS",
    "MULTI_BLOCK_PRIORITY_TIE_BREAK_WINS",
    "INVALID_HIGH_PRIORITY_FALLS_THROUGH_TO_LOWER_PRIORITY",
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
    "PRIORITY_POCKET_ORACLE_CONTROL",
    "SAME_BLOCKS_DIFFERENT_PRIORITY",
    "SAME_PRIORITY_DIFFERENT_BLOCK_VALUES",
    "SAME_TEMPLATE_OPPOSITE_PRIORITY_WINNER",
    "RULE_COMPOSITION_CORRUPTION_CONTROL",
    "LEGACY_STRUCTURED_RULE_METADATA_REGRESSION_CONTROL",
    "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL",
]
REQUIRED_145A_METRICS = [
    "mixed_rule_block_parse_accuracy",
    "per_block_candidate_derivation_accuracy",
    "priority_policy_parse_accuracy",
    "final_selected_pocket_derivation_accuracy",
    "selected_pocket_to_marker_binding_accuracy",
    "same_line_value_extraction_accuracy",
    "end_to_end_answer_accuracy",
    "invalid_high_priority_fallthrough_accuracy",
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
    "priority_pocket_oracle_rejection_rate",
    "same_blocks_different_priority_accuracy",
    "same_priority_different_block_values_accuracy",
    "same_template_opposite_priority_winner_accuracy",
    "rule_composition_ablation_accuracy",
    "helper_request_forbidden_metadata_count",
    "per_row_manifest_switch_rate",
    "per_row_payload_marker_switch_rate",
    "legacy_structured_rule_metadata_regression_passed",
    "legacy_selected_pocket_binding_regression_passed",
    "deterministic_replay_passed",
]
TRACE_FIELDS = [
    "parsed_rule_blocks",
    "parsed_priority_order",
    "per_block_parse_success",
    "per_block_derived_candidate_pocket",
    "final_selected_pocket_id",
    "binding_marker",
    "extracted_value",
    "generated_answer",
    "failure_reason",
]
PROMPT_SCANNER_FORBIDDEN = [
    "final_selected",
    "derived_selected",
    "selected_pocket",
    "selected_pocket_id",
    "winner=pocket_*",
    "winner value",
    "selected value",
    "answer value",
    "target value",
    "resolved output",
    "expected output",
    "gold output",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def helper_unchanged_from_head() -> bool:
    result = subprocess.run(
        ["git", "show", "HEAD:scripts/probes/shared_raw_generation_helper.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return result.stdout == HELPER_PATH.read_text(encoding="utf-8")


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [RUNNER_PATH, CHECKER_PATH]:
        if not path.exists():
            failures.append(f"missing:{rel(path)}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("run_stable_loop_phase_lock_") or module == "shared_raw_generation_helper":
                    failures.append(f"forbidden_import:{rel(path)}:{module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"torch", "shared_raw_generation_helper"}:
                        failures.append(f"forbidden_import:{rel(path)}:{alias.name}")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "backward", "step", "forward", "raw_generate", "load_checkpoint"}:
                    failures.append(f"forbidden_call:{rel(path)}:{name}")
    return {"schema_version": "phase_144z_ast_scan_v1", "passed": not failures, "failures": failures}


def require_144h(root: Path) -> tuple[dict[str, Any], list[str]]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "summary.json",
        "shared_helper_no_change_audit.json",
        "helper_structured_rule_semantics_audit.json",
        "per_seed_gate_report.json",
        "per_family_gate_report.json",
        "static_manifest_integrity_report.json",
        "helper_request_audit.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return {"root": rel(root), "missing_artifacts": missing}, [f"missing:{name}" for name in missing]
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    summary = read_json(root / "summary.json")
    helper_no_change = read_json(root / "shared_helper_no_change_audit.json")
    helper_semantics = read_json(root / "helper_structured_rule_semantics_audit.json")
    seed = read_json(root / "per_seed_gate_report.json")
    family = read_json(root / "per_family_gate_report.json")
    static = read_json(root / "static_manifest_integrity_report.json")
    request = read_json(root / "helper_request_audit.json")
    checks = {
        "decision": decision.get("decision") == "structured_rule_metadata_to_selected_pocket_binding_scale_confirmed",
        "verdict": decision.get("verdict") == "INSTNCT_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_SCALE_CONFIRMED",
        "next": decision.get("next") == "144Z_STRUCTURED_RULE_METADATA_BINDING_NEXT_DECISION_PLAN",
        "summary_boundary_present": "structured rule metadata to selected-pocket binding only" in json.dumps(summary),
        "helper_no_change": helper_no_change.get("shared_helper_no_change_since_144b") is True,
        "helper_not_modified_by_144h": helper_no_change.get("shared_helper_modified_by_144h") is False,
        "helper_semantics_passed": helper_semantics.get("passed") is True,
        "per_seed_gate_passed": seed.get("passed") is True,
        "per_family_gate_passed": family.get("passed") is True,
        "static_manifest_passed": static.get("passed") is True,
        "request_allowed_keys": request.get("all_requests_allowed_keys_only") is True,
        "request_forbidden_metadata_count": request.get("helper_request_forbidden_metadata_count") == 0,
    }
    for key, expected in EXPECTED_144H_METRICS.items():
        checks[f"metric:{key}"] = metrics.get(key) == expected
    failed = [key for key, passed in checks.items() if not passed]
    manifest = {
        "schema_version": "phase_144z_upstream_144h_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": {key: metrics.get(key) for key in EXPECTED_144H_METRICS},
        "shared_helper_no_change_audit": helper_no_change,
        "helper_structured_rule_semantics_audit": helper_semantics,
        "per_seed_gate_passed": seed.get("passed"),
        "per_family_gate_passed": family.get("passed"),
        "static_manifest_integrity_passed": static.get("passed"),
        "helper_request_audit": {
            "all_requests_allowed_keys_only": request.get("all_requests_allowed_keys_only"),
            "helper_request_forbidden_metadata_count": request.get("helper_request_forbidden_metadata_count"),
            "raw_generate_allowed_in_runner": request.get("raw_generate_allowed_in_runner"),
            "raw_generate_allowed_in_checker": request.get("raw_generate_allowed_in_checker"),
        },
        "gate_checks": checks,
        "failed_gate_checks": failed,
    }
    return manifest, failed


def evidence_chain_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_144z_evidence_chain_summary_v1",
        "current_state": "single structured rule metadata binding scale confirmed",
        "upstream_144h": upstream,
        "interpretation": [
            "143W scale-confirmed explicit selected-pocket binding and selected-marker occurrence rejection.",
            "144B introduced a manifest-gated structured rule metadata decoder.",
            "144H scale-confirmed single canonical rule families: quorum, recency, tie_break, and hierarchy combiner.",
            "The confirmed layer is single structured rule metadata -> selected pocket -> static marker -> same-line value extraction.",
            "The next untested bridge is mixed structured rule composition with an explicit priority policy.",
        ],
    }


def structured_rule_binding_state_report() -> dict[str, Any]:
    return {
        "schema_version": "phase_144z_structured_rule_binding_state_report_v1",
        "confirmed_layer": [
            "canonical single structured rule metadata parsing",
            "single rule selected pocket derivation",
            "selected pocket to static marker binding",
            "same-line value extraction",
            "wrong-family extra key rejection",
            "clear quorum winner ignores irrelevant tie_break_order",
            "legacy 143W binding regression preservation",
        ],
        "single_structured_rule_binding_scale_confirmed": True,
        "structured_rule_metadata_ablation_control_passed": True,
        "legacy_selected_pocket_binding_preserved": True,
        "remaining_gap": "multiple structured rule candidates -> explicit priority/conflict policy -> final selected pocket id",
        "natural_language_rule_reasoning_claimed": False,
        "open_ended_arbitration_claimed": False,
    }


def mixed_rule_composition_gap_analysis() -> dict[str, Any]:
    return {
        "schema_version": "phase_144z_mixed_rule_composition_gap_analysis_v1",
        "single_rule_structured_metadata_scale_confirmed": True,
        "mixed_structured_rule_composition_untested": True,
        "priority_policy_final_selection_untested": True,
        "invalid_high_priority_fallthrough_untested": True,
        "block_boundary_parser_untested": True,
        "priority_oracle_shortcut_claimed": False,
        "natural_language_rule_reasoning_claimed": False,
        "next_gap": "multiple canonical structured rule blocks -> explicit priority policy -> final selected pocket identity",
    }


def next_decision_matrix() -> dict[str, Any]:
    options = [
        {
            "option_id": SELECTED_OPTION,
            "mechanism": "Plan the first executable mixed structured rule composition prototype with explicit priority policy.",
            "diagnostic_value": "high",
            "oracle_risk": "medium",
            "scope_cost": "medium",
            "recommendation": "recommended",
            "reason": "144H closed single-rule parsing; the next missing bridge is controlled composition across multiple rule candidates.",
        },
        {
            "option_id": "structured_rule_metadata_robustness_extension",
            "mechanism": "Add more single-rule grammar variants and malformed metadata controls.",
            "diagnostic_value": "medium",
            "oracle_risk": "low",
            "scope_cost": "low",
            "recommendation": "defer",
            "reason": "The next bottleneck is composition, not more single-rule coverage.",
        },
        {
            "option_id": "integration_into_broader_multi_pocket_arbitration_suite",
            "mechanism": "Move directly to a broader arbitration harness.",
            "diagnostic_value": "medium",
            "oracle_risk": "high",
            "scope_cost": "high",
            "recommendation": "defer",
            "reason": "A narrow mixed-rule composition prototype should precede broader suite integration.",
        },
        {
            "option_id": "stop_at_single_rule_structured_metadata_binding",
            "mechanism": "Stop after single structured rule metadata binding scale confirm.",
            "diagnostic_value": "low",
            "oracle_risk": "low",
            "scope_cost": "low",
            "recommendation": "not_selected",
            "reason": "It would leave explicit priority composition untested.",
        },
    ]
    return {
        "schema_version": "phase_144z_next_decision_matrix_v1",
        "selected_option": SELECTED_OPTION,
        "recommended_next": NEXT,
        "options": options,
    }


def anti_oracle_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_144z_anti_oracle_requirements_v1",
        "rule_composition_subsets_forbid": [
            *PROMPT_SCANNER_FORBIDDEN,
            "per-row selected pocket request metadata",
            "per-row manifest switching",
            "payload marker list narrowed to correct pocket",
            "post-generation repair",
            "priority=pocket_*",
            "selected block oracle",
        ],
        "request_metadata_forbid": [
            "selected_pocket_id",
            "winner",
            "final_selected",
            "derived_selected",
            "expected_answer",
            "gold_output",
            "target_json",
            "scorer_metadata",
        ],
        "allow": [
            "canonical structured rule blocks",
            "priority over block types only",
            "static pocket marker map",
            "existing selected-pocket binding layer",
        ],
    }


def mixed_rule_block_format_spec() -> dict[str, Any]:
    return {
        "schema_version": "phase_144z_mixed_rule_block_format_spec_v1",
        "canonical_format_example": [
            "rule_block=quorum",
            "votes=pocket_a,pocket_b,pocket_a",
            "block_end",
            "",
            "rule_block=recency",
            "recency_order=pocket_c>pocket_b>pocket_a",
            "block_end",
            "",
            "rule_block=tie_break",
            "tied=pocket_a,pocket_c",
            "tie_break_order=pocket_c>pocket_a>pocket_b",
            "block_end",
            "",
            "priority=recency>quorum>tie_break",
        ],
        "valid_block_types": ["quorum", "recency", "tie_break"],
        "priority_entries_are_block_types_not_pockets": True,
        "block_boundary_policy": {
            "missing_block_end": "fallback",
            "nested_rule_block_before_block_end": "fallback",
            "metadata_outside_rule_block_except_priority": "fallback",
            "duplicate_rule_block_types": "fallback_for_145a",
            "unknown_rule_block_types": "fallback",
            "empty_rule_blocks": "fallback",
        },
        "priority_policy": {
            "priority_line_required": True,
            "multiple_priority_lines": "fallback",
            "duplicate_priority_entries": "fallback",
            "unknown_priority_entries": "fallback",
            "priority_references_missing_block": "fallback",
            "malformed_priority_separators": "fallback",
            "priority_pocket_oracle_entries": "fallback",
            "final_selected_pocket": "first valid candidate in priority order",
            "invalid_high_priority_block": "ignored if a lower-priority valid block exists",
            "all_priority_referenced_blocks_invalid": "fallback",
        },
    }


def target_145a_milestone_plan(anti_oracle: dict[str, Any]) -> dict[str, Any]:
    positive_gates = {
        "mixed_rule_block_parse_accuracy": ">= 0.90",
        "per_block_candidate_derivation_accuracy": ">= 0.90",
        "priority_policy_parse_accuracy": ">= 0.90",
        "final_selected_pocket_derivation_accuracy": ">= 0.90",
        "selected_pocket_to_marker_binding_accuracy": ">= 0.95",
        "same_line_value_extraction_accuracy": ">= 0.95",
        "end_to_end_answer_accuracy": ">= 0.90",
        "invalid_high_priority_fallthrough_accuracy": ">= 0.90",
        "all_blocks_invalid_fallback_rate": ">= 0.90",
        "missing_priority_fallback_rate": ">= 0.90",
        "duplicate_priority_rejection_rate": ">= 0.90",
        "unknown_priority_rejection_rate": ">= 0.90",
        "missing_block_reference_rejection_rate": ">= 0.90",
        "multiple_priority_lines_rejection_rate": ">= 0.90",
        "duplicate_rule_block_type_rejection_rate": ">= 0.90",
        "malformed_block_boundary_rejection_rate": ">= 0.90",
        "metadata_outside_block_rejection_rate": ">= 0.90",
        "nested_block_boundary_rejection_rate": ">= 0.90",
        "empty_rule_block_rejection_rate": ">= 0.90",
        "priority_pocket_oracle_rejection_rate": ">= 0.90",
        "same_blocks_different_priority_accuracy": ">= 0.90",
        "same_priority_different_block_values_accuracy": ">= 0.90",
        "same_template_opposite_priority_winner_accuracy": ">= 0.90",
        "rule_composition_ablation_accuracy": "<= 0.15",
        "helper_request_forbidden_metadata_count": "= 0",
        "per_row_manifest_switch_rate": "= 0.0",
        "per_row_payload_marker_switch_rate": "= 0.0",
        "legacy_structured_rule_metadata_regression_passed": "= true",
        "legacy_selected_pocket_binding_regression_passed": "= true",
        "deterministic_replay_passed": "= true",
    }
    return {
        "schema_version": "phase_144z_target_145a_milestone_plan_v1",
        "milestone": NEXT,
        "implementation_ready": True,
        "planning_only": False,
        "helper_changing": True,
        "decoder_name": NEW_DECODER,
        "helper_change_scope": "new manifest-gated mixed structured rule composition decoder only",
        "old_decoders_must_remain_unchanged": [
            OLD_SELECTED_POCKET_DECODER,
            OLD_STRUCTURED_RULE_DECODER,
        ],
        "request_key_change_allowed": False,
        "natural_language_rule_parsing_allowed": False,
        "free_form_arbitration_allowed": False,
        "post_generation_repair_allowed": False,
        "intended_primitive": [
            "parse multiple canonical structured rule blocks",
            "derive candidate winner for each valid rule block",
            "apply explicit priority policy over block types",
            "derive final selected pocket id",
            "reuse existing static marker binding and same-line value extraction",
            "emit ANSWER=E<value>",
        ],
        "mixed_rule_block_format_spec": mixed_rule_block_format_spec(),
        "policy": [
            "Each block parses independently.",
            "Each valid block derives one candidate pocket.",
            "Invalid high-priority blocks are ignored if a lower-priority valid block exists.",
            "If all priority-referenced blocks are invalid: fallback.",
            "priority= is mandatory and must reference known block types.",
            "Final selected pocket is the first valid candidate in priority order.",
            "Duplicate priority entries, unknown priority entries, priority entries referencing missing block types, malformed priority separators, and multiple priority lines all fallback.",
            "Duplicate block types, unknown block types, empty blocks, missing block_end, nested rule_block before block_end, and metadata outside a block except priority= all fallback.",
            "Conflicting valid candidates are allowed only when valid priority resolves them.",
            "Conflict without valid priority: fallback.",
        ],
        "required_subsets": REQUIRED_145A_SUBSETS,
        "required_metrics": REQUIRED_145A_METRICS,
        "positive_gates": positive_gates,
        "required_trace_fields": TRACE_FIELDS,
        "prompt_scanner_forbidden": PROMPT_SCANNER_FORBIDDEN,
        "anti_oracle_requirements": anti_oracle,
        "clean_negative_routes": {
            "mixed_rule_block_parse_failure": "145B_MIXED_RULE_BLOCK_PARSE_FAILURE_ANALYSIS",
            "priority_policy_parse_failure": "145C_PRIORITY_POLICY_PARSE_FAILURE_ANALYSIS",
            "final_selected_pocket_derivation_failure": "145D_MIXED_RULE_FINAL_SELECTION_FAILURE_ANALYSIS",
            "rule_composition_oracle_shortcut_detected": "145E_RULE_COMPOSITION_ORACLE_SHORTCUT_ANALYSIS",
            "priority_ambiguity_not_rejected": "145F_PRIORITY_AMBIGUITY_ANALYSIS",
            "invalid_block_fallthrough_failure": "145G_INVALID_BLOCK_FALLTHROUGH_ANALYSIS",
            "legacy_structured_rule_metadata_regression": "144C_STRUCTURED_RULE_METADATA_PARSE_FAILURE_ANALYSIS",
            "selected_pocket_binding_regression": "143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
        "claim_limit": (
            "A future positive 145A would prove constrained mixed structured-rule composition with explicit priority policy only; "
            "it would not prove natural-language reasoning, open-ended arbitration, GPT-like capability, production readiness, or architecture superiority."
        ),
    }


def risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_144z_risk_register_v1",
        "risks": [
            {
                "risk_id": "priority_line_becomes_pocket_oracle",
                "risk": "priority= could accidentally encode pocket IDs instead of block types.",
                "mitigation": "145A prompt scanner and parser require priority entries to be known block types and reject pocket entries.",
            },
            {
                "risk_id": "loose_block_boundaries",
                "risk": "A loose block parser could accept nested, empty, or unterminated blocks.",
                "mitigation": "145A must reject missing block_end, nested rule_block, metadata outside blocks, and empty blocks.",
            },
            {
                "risk_id": "invalid_high_priority_overfallback",
                "risk": "The prototype could fallback whenever a high-priority block is invalid, even if a lower-priority block is valid.",
                "mitigation": "145A must measure invalid_high_priority_fallthrough_accuracy separately.",
            },
            {
                "risk_id": "legacy_decoder_regression",
                "risk": "The new mixed-rule decoder could alter the structured single-rule or selected-pocket decoders.",
                "mitigation": "145A must keep old decoders unchanged and include both legacy regression controls.",
            },
        ],
    }


def decision_payload(upstream_failed: list[str], ast_report: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    positive = not upstream_failed and ast_report.get("passed") is True and target.get("implementation_ready") is True
    return {
        "schema_version": "phase_144z_decision_v1",
        "decision": DECISION if positive else "structured_rule_composition_priority_plan_blocked",
        "selected_option": SELECTED_OPTION if positive else "blocked",
        "next": NEXT if positive else "144Z_BLOCKER_ANALYSIS",
        "positive_gate_passed": positive,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], gap: dict[str, Any], target: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Selected option: `{decision['selected_option']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Evidence

- Single structured rule binding scale confirmed: `{gap['single_rule_structured_metadata_scale_confirmed']}`
- Mixed structured rule composition untested: `{gap['mixed_structured_rule_composition_untested']}`
- Priority policy final selection untested: `{gap['priority_policy_final_selection_untested']}`

## Target 145A

`{target['milestone']}` is implementation-ready and uses the new manifest-gated decoder `{target['decoder_name']}`. Existing decoders remain unchanged: `{', '.join(target['old_decoders_must_remain_unchanged'])}`.

The target primitive is multiple canonical structured rule blocks -> explicit priority policy -> final selected pocket -> existing static marker binding -> same-line value extraction.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 144Z structured rule metadata binding next decision plan")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-144h-root", type=Path, default=DEFAULT_144H_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_144z_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream, upstream_failed = require_144h(resolve_repo_path(args.upstream_144h_root))
    append_progress(out, "upstream verified", failed_gate_count=len(upstream_failed))
    ast_report = scan_ast()
    evidence = evidence_chain_summary(upstream)
    state = structured_rule_binding_state_report()
    gap = mixed_rule_composition_gap_analysis()
    matrix = next_decision_matrix()
    anti_oracle = anti_oracle_requirements()
    risks = risk_register()
    target = target_145a_milestone_plan(anti_oracle)
    decision = decision_payload(upstream_failed, ast_report, target)
    helper_source = HELPER_PATH.read_text(encoding="utf-8")
    config = {
        "schema_version": "phase_144z_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "planning_only": True,
        "artifact_only": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "helper_generation_called": False,
        "raw_generate_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "helper_modified": False,
        "backend_modified": False,
        "request_key_change": False,
        "public_request_key_change": False,
        "runtime_surface_mutated": False,
        "release_surface_mutated": False,
        "product_surface_mutated": False,
        "root_license_changed": False,
        "shared_helper_sha256": sha256_text(helper_source),
        "shared_helper_unchanged_from_head": helper_unchanged_from_head(),
        **FALSE_FLAGS,
    }
    summary = {
        "schema_version": "phase_144z_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "decision": decision,
        "structured_rule_binding_state_report": state,
        "mixed_rule_composition_gap_analysis": gap,
        "target_145a_milestone_plan": target,
        **FALSE_FLAGS,
    }

    write_json(out / "analysis_config.json", config)
    write_json(out / "upstream_144h_manifest.json", upstream)
    write_json(out / "evidence_chain_summary.json", evidence)
    write_json(out / "structured_rule_binding_state_report.json", state)
    write_json(out / "next_decision_matrix.json", matrix)
    write_json(out / "mixed_rule_composition_gap_analysis.json", gap)
    write_json(out / "target_145a_milestone_plan.json", target)
    write_json(out / "anti_oracle_requirements.json", anti_oracle)
    write_json(out / "risk_register.json", risks)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, gap, target)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_144z_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
