#!/usr/bin/env python3
"""144A planning-only structured rule metadata to selected-pocket binding plan."""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_144a_structured_rule_metadata_to_selected_pocket_binding_plan/smoke")
DEFAULT_143Z_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_143z_rule_selected_pocket_binding_next_decision_plan/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_144a_structured_rule_metadata_to_selected_pocket_binding_plan_check.py"
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
DECISION = "structured_rule_metadata_to_selected_pocket_binding_prototype_plan_recommended"
SELECTED_OPTION = "canonical_structured_rule_metadata_parser_plus_existing_selected_pocket_binding"
NEXT = "144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE"
NEW_DECODER = "deterministic_pocket_gated_structured_rule_metadata_binding_decoder"
OLD_SELECTED_POCKET_DECODER = "deterministic_pocket_gated_rule_selected_pocket_binding_decoder"
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
    "144A is planning-only and artifact-only after 143Z. It is constrained helper/backend evidence only "
    "and structured rule metadata to selected-pocket binding only; "
    "it is not natural-language rule reasoning, not open-ended arbitration, "
    "not GPT-like/open-domain/broad assistant capability, "
    "not production/public API/deployment/safety readiness, and not architecture superiority."
)
REQUIRED_144B_METRICS = [
    "rule_metadata_parse_accuracy",
    "derived_selected_pocket_accuracy",
    "selected_pocket_to_marker_binding_accuracy",
    "same_line_value_extraction_accuracy",
    "end_to_end_answer_accuracy",
    "rule_derived_no_winner_label_accuracy",
    "explicit_winner_baseline_accuracy",
    "rule_metadata_ablation_accuracy",
    "corrupt_rule_metadata_rejection_rate",
    "missing_rule_metadata_fallback_rate",
    "ambiguous_rule_metadata_rejection_rate",
    "helper_request_forbidden_metadata_count",
    "per_row_manifest_switch_rate",
    "per_row_payload_marker_switch_rate",
    "legacy_selected_pocket_binding_regression_passed",
    "deterministic_replay_passed",
]
REQUIRED_144B_SUBSETS = [
    "EXPLICIT_WINNER_LABEL_BASELINE",
    "RULE_METADATA_DERIVED_NO_WINNER_LABEL",
    "QUORUM_RULE_DERIVED",
    "RECENCY_RULE_DERIVED",
    "TIE_BREAK_RULE_DERIVED",
    "HIERARCHY_RULE_DERIVED",
    "SAME_VALUES_DIFFERENT_RULE",
    "SAME_RULE_DIFFERENT_VALUES",
    "SAME_TEMPLATE_OPPOSITE_RULE_WINNER",
    "RULE_METADATA_CORRUPTION_CONTROL",
    "MISSING_RULE_METADATA_CONTROL",
    "AMBIGUOUS_RULE_METADATA_CONTROL",
    "LEGACY_SELECTED_POCKET_BINDING_REGRESSION_CONTROL",
]
RULE_DERIVED_FORBIDDEN = [
    "winner=pocket_*",
    "selected_pocket_id",
    "final winner",
    "winner value",
    "answer value",
    "gold value",
    "target value",
    "resolved output",
    "expected output",
    "per-row selected pocket request metadata",
    "per-row manifest switching",
    "payload marker list narrowed to correct pocket",
    "post-generation repair",
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
    return {"schema_version": "phase_144a_ast_scan_v1", "passed": not failures, "failures": failures}


def require_143z(root: Path) -> tuple[dict[str, Any], list[str]]:
    required = [
        "decision.json",
        "rule_metadata_bridge_gap_analysis.json",
        "selected_pocket_binding_state_report.json",
        "target_144a_milestone_plan.json",
        "anti_oracle_requirements.json",
        "summary.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return {"root": rel(root), "missing_artifacts": missing}, [f"missing:{name}" for name in missing]
    decision = read_json(root / "decision.json")
    gap = read_json(root / "rule_metadata_bridge_gap_analysis.json")
    state = read_json(root / "selected_pocket_binding_state_report.json")
    target = read_json(root / "target_144a_milestone_plan.json")
    anti = read_json(root / "anti_oracle_requirements.json")
    summary = read_json(root / "summary.json")
    checks = {
        "decision": decision.get("decision") == "rule_metadata_to_selected_pocket_binding_plan_recommended",
        "selected_option": decision.get("selected_option") == "structured_rule_metadata_to_selected_pocket_binding_plan",
        "next": decision.get("next") == "144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN",
        "selected_pocket_binding_scale_confirmed": gap.get("selected_pocket_binding_scale_confirmed") is True,
        "selected_marker_occurrence_rejection_scale_confirmed": gap.get("selected_marker_occurrence_rejection_scale_confirmed") is True,
        "rule_metadata_to_selected_pocket_identity_untested": gap.get("rule_metadata_to_selected_pocket_identity_untested") is True,
        "natural_language_rule_reasoning_untested": gap.get("natural_language_rule_reasoning_untested") is True,
        "open_ended_arbitration_claimed": gap.get("open_ended_arbitration_claimed") is False,
        "state_binding_confirmed": state.get("selected_pocket_binding_scale_confirmed") is True,
        "target_144a_planning_only": target.get("planning_only") is True,
        "target_144a_first_executable": target.get("first_executable_prototype") == NEXT,
        "summary_boundary": "not natural-language rule reasoning" in json.dumps(summary),
    }
    for forbidden in [
        "winner=pocket_* in rule-derived subsets",
        "selected_pocket_id in prompt or request metadata",
        "final/winner value markers",
        "answer/gold/target/resolved output markers",
        "per-row selected pocket request metadata",
        "per-row manifest switching",
        "payload marker list narrowed to the correct pocket",
        "post-generation repair",
    ]:
        checks[f"anti_oracle:{forbidden}"] = forbidden in anti.get("forbid", [])
    failed = [key for key, passed in checks.items() if not passed]
    manifest = {
        "schema_version": "phase_144a_upstream_143z_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "rule_metadata_bridge_gap_analysis": gap,
        "selected_pocket_binding_state_report": state,
        "target_144a_milestone_plan": target,
        "anti_oracle_requirements": anti,
        "gate_checks": checks,
        "failed_gate_checks": failed,
    }
    return manifest, failed


def grammar_spec() -> dict[str, Any]:
    return {
        "schema_version": "phase_144a_structured_rule_metadata_grammar_spec_v1",
        "scope": "canonical structured rule metadata only",
        "free_form_natural_language_rule_parsing_allowed": False,
        "general_rules": {
            "exact_keys_only_per_rule_family": True,
            "duplicate_keys_policy": "fallback",
            "unknown_keys_policy": "fallback",
            "missing_required_keys_policy": "fallback",
            "multiple_rule_type_lines_policy": "fallback",
            "invalid_pocket_ids_policy": "fallback",
            "malformed_separators_policy": "fallback",
            "rule_derived_subsets_forbid_winner_labels": True,
            "rule_derived_subsets_forbid_selected_pocket_id": True,
        },
        "valid_pocket_ids": ["pocket_a", "pocket_b", "pocket_c"],
        "families": {
            "quorum": {
                "required_keys": ["rule_type", "votes"],
                "optional_keys": ["tie_break_order"],
                "example": ["rule_type=quorum", "votes=pocket_a,pocket_b,pocket_a"],
                "policy": [
                    "winner = most frequent pocket",
                    "tie without tie_break_order -> fallback",
                    "if tie_break_order is present, winner = first tied pocket appearing in tie_break_order",
                    "invalid/duplicate/missing votes -> fallback",
                ],
            },
            "recency": {
                "required_keys": ["rule_type", "recency_order"],
                "optional_keys": [],
                "example": ["rule_type=recency", "recency_order=pocket_c>pocket_b>pocket_a"],
                "policy": [
                    "winner = first valid pocket in recency_order",
                    "duplicate pockets in recency_order -> fallback",
                    "missing pockets allowed only if at least one valid pocket exists",
                    "invalid pocket ids -> fallback",
                ],
            },
            "tie_break": {
                "required_keys": ["rule_type", "tied", "tie_break_order"],
                "optional_keys": [],
                "example": ["rule_type=tie_break", "tied=pocket_a,pocket_b", "tie_break_order=pocket_b>pocket_a>pocket_c"],
                "policy": [
                    "winner = first tied pocket appearing in tie_break_order",
                    "no tied pocket appears in tie_break_order -> fallback",
                    "invalid tied values or duplicate tied values -> fallback",
                ],
            },
            "hierarchy": {
                "required_keys": ["rule_type", "hierarchy", "stale", "recency_winner", "quorum_winner", "tie_break_winner"],
                "optional_keys": [],
                "example": [
                    "rule_type=hierarchy",
                    "hierarchy=stale_rejection>recency>quorum>tie_break",
                    "stale=pocket_a",
                    "recency_winner=pocket_b",
                    "quorum_winner=pocket_c",
                    "tie_break_winner=pocket_b",
                ],
                "policy": [
                    "hierarchy tests priority ordering over precomputed sub-rule winners",
                    "it does not claim nested derivation of recency/quorum/tie_break from raw rule fields",
                    "winner = first non-stale, non-none sub-rule winner in hierarchy order after stale rejection",
                    "if all candidate winners are stale/none/invalid -> fallback",
                ],
            },
        },
    }


def rule_derivation_policy_matrix(grammar: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for family, spec in grammar["families"].items():
        rows.append(
            {
                "rule_family": family,
                "required_keys": spec["required_keys"],
                "optional_keys": spec["optional_keys"],
                "derive_policy": spec["policy"],
                "fallback_on_duplicate_keys": True,
                "fallback_on_unknown_keys": True,
                "fallback_on_invalid_pocket_ids": True,
            }
        )
    return {
        "schema_version": "phase_144a_rule_derivation_policy_matrix_v1",
        "rows": rows,
        "hierarchy_is_combiner_fixture_not_nested_derivation": True,
    }


def anti_oracle_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_144a_anti_oracle_requirements_v1",
        "rule_derived_subsets_forbid": RULE_DERIVED_FORBIDDEN,
        "request_metadata_forbid": ["selected_pocket_id", "winner", "expected_answer", "gold_output", "target_json", "scorer_metadata"],
        "manifest_forbid": ["per-row manifest switching", "payload marker list narrowed to correct pocket", "per-row selected pocket metadata"],
        "allow": ["static pocket marker map", "canonical structured rule metadata", "explicit winner label baseline in baseline subset only"],
    }


def prototype_design_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_144a_prototype_design_requirements_v1",
        "decoder_name": NEW_DECODER,
        "existing_selected_pocket_decoder_to_preserve": OLD_SELECTED_POCKET_DECODER,
        "helper_change_allowed": True,
        "helper_change_scope": "new manifest-gated decoder only",
        "request_key_change_allowed": False,
        "post_generation_repair_allowed": False,
        "trace_fields_required": [
            "parsed_rule_type",
            "parsed_rule_fields",
            "parse_success",
            "derived_selected_pocket_id",
            "binding_marker",
            "extracted_value",
            "generated_answer",
            "failure_reason",
        ],
        "intended_primitive": [
            "parse canonical structured rule metadata",
            "derive selected pocket id",
            "reuse existing selected pocket -> static marker -> same-line value extraction",
            "emit ANSWER=E<value>",
        ],
    }


def prototype_options_matrix() -> dict[str, Any]:
    return {
        "schema_version": "phase_144a_prototype_options_matrix_v1",
        "selected_option": SELECTED_OPTION,
        "options": [
            {
                "option_id": SELECTED_OPTION,
                "mechanism": "new manifest-gated decoder parses canonical structured rule metadata and reuses existing selected-pocket binding",
                "diagnostic_value": "high",
                "oracle_risk": "medium",
                "implementation_scope": "helper prototype",
                "recommendation": "recommended",
            },
            {
                "option_id": "prompt_level_free_form_rule_parser",
                "mechanism": "parse natural language rule prose",
                "diagnostic_value": "low_for_current_milestone",
                "oracle_risk": "high",
                "implementation_scope": "too_broad",
                "recommendation": "reject_for_144B",
            },
            {
                "option_id": "request_metadata_selected_pocket_binding",
                "mechanism": "pass selected pocket through request metadata",
                "diagnostic_value": "invalid",
                "oracle_risk": "critical",
                "implementation_scope": "forbidden",
                "recommendation": "reject",
            },
        ],
    }


def selected_prototype_recommendation() -> dict[str, Any]:
    return {
        "schema_version": "phase_144a_selected_prototype_recommendation_v1",
        "selected_option": SELECTED_OPTION,
        "decoder_name": NEW_DECODER,
        "next": NEXT,
        "why": "This is the smallest implementation-ready bridge from structured rule metadata to the already-confirmed selected-pocket binding layer.",
        "no_intermediate_micro_plan_after_144a": True,
    }


def target_144b_milestone_plan(grammar: dict[str, Any], anti_oracle: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_144a_target_144b_milestone_plan_v1",
        "milestone": NEXT,
        "decoder_name": NEW_DECODER,
        "helper_changing": True,
        "helper_change_scope": "new manifest-gated decoder only",
        "old_selected_pocket_binding_decoder_must_remain_unchanged": True,
        "existing_selected_pocket_decoder": OLD_SELECTED_POCKET_DECODER,
        "request_key_change_allowed": False,
        "post_generation_repair_allowed": False,
        "implementation_ready": True,
        "no_additional_planning_milestone_expected": True,
        "structured_rule_metadata_grammar_spec": grammar,
        "required_trace_fields": prototype_design_requirements()["trace_fields_required"],
        "required_subsets": REQUIRED_144B_SUBSETS,
        "rule_derived_subset_forbidden": anti_oracle["rule_derived_subsets_forbid"],
        "required_metrics": REQUIRED_144B_METRICS,
        "positive_gates": {
            "rule_metadata_parse_accuracy": ">= 0.90",
            "derived_selected_pocket_accuracy": ">= 0.90",
            "selected_pocket_to_marker_binding_accuracy": ">= 0.95",
            "same_line_value_extraction_accuracy": ">= 0.95",
            "end_to_end_answer_accuracy": ">= 0.90",
            "rule_derived_no_winner_label_accuracy": ">= 0.90",
            "explicit_winner_baseline_accuracy": ">= 0.95",
            "rule_metadata_ablation_accuracy": "<= 0.15",
            "corrupt_rule_metadata_rejection_rate": ">= 0.90",
            "missing_rule_metadata_fallback_rate": ">= 0.90",
            "ambiguous_rule_metadata_rejection_rate": ">= 0.90",
            "helper_request_forbidden_metadata_count": "= 0",
            "per_row_manifest_switch_rate": "= 0.0",
            "per_row_payload_marker_switch_rate": "= 0.0",
            "legacy_selected_pocket_binding_regression_passed": "= true",
            "deterministic_replay_passed": "= true",
        },
        "clean_negative_routes": {
            "structured_rule_metadata_parse_failure": "144C_STRUCTURED_RULE_METADATA_PARSE_FAILURE_ANALYSIS",
            "derived_selected_pocket_failure": "144D_DERIVED_SELECTED_POCKET_FAILURE_ANALYSIS",
            "rule_metadata_oracle_shortcut_detected": "144E_RULE_METADATA_ORACLE_SHORTCUT_ANALYSIS",
            "rule_metadata_ambiguity_not_rejected": "144F_RULE_METADATA_AMBIGUITY_ANALYSIS",
            "hierarchy_priority_policy_failure": "144G_HIERARCHY_RULE_POLICY_ANALYSIS",
            "selected_pocket_binding_regression": "143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
    }


def risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_144a_risk_register_v1",
        "risks": [
            {
                "risk_id": "hidden_winner_oracle",
                "risk": "Rule-derived subset accidentally includes winner=pocket_* or selected_pocket_id.",
                "mitigation": "Forbid these strings in rule-derived prompt scanner and request metadata audit.",
            },
            {
                "risk_id": "grammar_ambiguity",
                "risk": "Duplicate keys, unknown keys, or malformed separators silently derive a pocket.",
                "mitigation": "Exact-key grammar with fallback on ambiguity or malformed metadata.",
            },
            {
                "risk_id": "hierarchy_overclaim",
                "risk": "Hierarchy combiner could be misread as nested derivation of sub-rule winners.",
                "mitigation": "State hierarchy tests priority over precomputed sub-rule winners only.",
            },
            {
                "risk_id": "legacy_decoder_regression",
                "risk": "New decoder changes existing selected-pocket binding behavior.",
                "mitigation": "Require legacy selected-pocket binding regression control and source diff audit.",
            },
        ],
    }


def decision_payload(upstream_failed: list[str], ast_report: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    positive = not upstream_failed and ast_report.get("passed") is True and target.get("implementation_ready") is True
    return {
        "schema_version": "phase_144a_decision_v1",
        "decision": DECISION if positive else "structured_rule_metadata_to_selected_pocket_binding_plan_blocked",
        "selected_option": SELECTED_OPTION if positive else "blocked",
        "next": NEXT if positive else "144A_BLOCKER_ANALYSIS",
        "positive_gate_passed": positive,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any], grammar: dict[str, Any], target: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Selected option: `{decision['selected_option']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## 144B Decoder

`{target['decoder_name']}`

## Grammar Families

`{', '.join(sorted(grammar['families']))}`

144B is implementation-ready and routes directly to `{target['milestone']}` with no additional micro-plan expected unless the 144A checker fails.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 144A structured rule metadata planning milestone")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-143z-root", type=Path, default=DEFAULT_143Z_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_144a_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream, upstream_failed = require_143z(resolve_repo_path(args.upstream_143z_root))
    append_progress(out, "upstream verified", failed_gate_count=len(upstream_failed))
    ast_report = scan_ast()
    grammar = grammar_spec()
    policies = rule_derivation_policy_matrix(grammar)
    anti_oracle = anti_oracle_requirements()
    requirements = prototype_design_requirements()
    options = prototype_options_matrix()
    recommendation = selected_prototype_recommendation()
    target = target_144b_milestone_plan(grammar, anti_oracle)
    risks = risk_register()
    decision = decision_payload(upstream_failed, ast_report, target)
    helper_source = HELPER_PATH.read_text(encoding="utf-8")
    config = {
        "schema_version": "phase_144a_analysis_config_v1",
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
        "schema_version": "phase_144a_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "decision": decision,
        "structured_rule_metadata_grammar_spec": grammar,
        "target_144b_milestone_plan": target,
        **FALSE_FLAGS,
    }

    write_json(out / "analysis_config.json", config)
    write_json(out / "upstream_143z_manifest.json", upstream)
    write_json(out / "prototype_design_requirements.json", requirements)
    write_json(out / "structured_rule_metadata_grammar_spec.json", grammar)
    write_json(out / "rule_derivation_policy_matrix.json", policies)
    write_json(out / "prototype_options_matrix.json", options)
    write_json(out / "selected_prototype_recommendation.json", recommendation)
    write_json(out / "target_144b_milestone_plan.json", target)
    write_json(out / "anti_oracle_requirements.json", anti_oracle)
    write_json(out / "risk_register.json", risks)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, grammar, target)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_144a_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
