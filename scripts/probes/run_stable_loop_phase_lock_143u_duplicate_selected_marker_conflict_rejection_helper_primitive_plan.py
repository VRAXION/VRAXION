#!/usr/bin/env python3
"""143U artifact-only duplicate selected marker rejection primitive plan.

This phase reads 143R artifacts and writes a machine-readable 143V repair
prototype plan. It does not import or call the helper, run generation, train,
mutate checkpoints, modify request keys, or implement the repair.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_143u_duplicate_selected_marker_conflict_rejection_helper_primitive_plan/smoke")
DEFAULT_143R_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_143r_duplicate_selected_marker_conflict_analysis/smoke")
DECISION = "duplicate_selected_marker_conflict_rejection_primitive_plan_recommended"
SELECTED_OPTION = "selected_marker_occurrence_count_must_equal_one"
NEXT = "143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE"
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "rule_metadata_reasoning_claimed": False,
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
    "143U is planning-only constrained helper/backend evidence only. It is "
    "prompt-visible selected-pocket binding only, not rule metadata reasoning, "
    "not open-ended arbitration, not GPT-like/open-domain/broad assistant "
    "capability, not production/public API/deployment/safety readiness, and not "
    "architecture superiority. It does not repair the helper, call helper "
    "generation, train, mutate checkpoints, modify helper/backend/request keys, "
    "deploy, or change runtime/product/release surfaces."
)


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


def require_143r(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "duplicate_conflict_trace_report.json",
        "helper_duplicate_marker_semantics_audit.json",
        "selected_marker_occurrence_policy_matrix.json",
        "repair_options_matrix.json",
        "root_cause_report.json",
        "target_143u_milestone_plan.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 143R artifacts: {missing}")
    decision = read_json(root / "decision.json")
    trace = read_json(root / "duplicate_conflict_trace_report.json")
    helper = read_json(root / "helper_duplicate_marker_semantics_audit.json")
    policy = read_json(root / "selected_marker_occurrence_policy_matrix.json")
    options = read_json(root / "repair_options_matrix.json")
    root_report = read_json(root / "root_cause_report.json")
    target = read_json(root / "target_143u_milestone_plan.json")
    if decision.get("decision") != "duplicate_selected_marker_conflict_analysis_complete":
        raise RuntimeError(f"bad 143R decision: {decision.get('decision')}")
    if decision.get("root_cause_id") != "selected_marker_duplicate_conflict_uses_first_occurrence_without_occurrence_count_policy":
        raise RuntimeError(f"bad 143R root cause: {decision.get('root_cause_id')}")
    if decision.get("next") != MILESTONE.replace("STABLE_LOOP_PHASE_LOCK_", ""):
        raise RuntimeError(f"bad 143R next: {decision.get('next')}")
    if decision.get("recommended_policy") != SELECTED_OPTION or decision.get("target_repair_prototype") != NEXT:
        raise RuntimeError("143R recommended policy or target repair prototype mismatch")
    expected_trace = {
        "generated_equals_first_duplicate_value_rate": 1.0,
        "generated_equals_last_duplicate_value_rate": 0.0,
        "generated_equals_fallback_rate": 0.0,
    }
    for key, expected in expected_trace.items():
        if trace.get(key) != expected:
            raise RuntimeError(f"bad 143R trace {key}: {trace.get(key)}")
    expected_helper = {
        "prompt_find_selected_marker_found": True,
        "selected_marker_first_occurrence_offset_used": True,
        "selected_marker_count_variable_found": False,
        "fallback_on_duplicate_conflict_found": False,
    }
    for key, expected in expected_helper.items():
        if helper.get(key) != expected:
            raise RuntimeError(f"bad 143R helper audit {key}: {helper.get(key)}")
    if policy.get("recommended_policy") != SELECTED_OPTION:
        raise RuntimeError(f"bad 143R policy: {policy.get('recommended_policy')}")
    for key in [
        "selected_marker_occurrence_policy_applies_only_to_the_selected_marker",
        "non_selected_marker_duplicates_out_of_scope",
        "same_value_duplicate_acceptance_deferred",
    ]:
        if policy.get(key) is not True:
            raise RuntimeError(f"143R policy scope missing: {key}")
    if options.get("recommended_option") != SELECTED_OPTION:
        raise RuntimeError(f"bad 143R repair option: {options.get('recommended_option')}")
    if root_report.get("selected_marker_occurrence_count_policy_missing") is not True:
        raise RuntimeError("143R root cause did not confirm missing occurrence policy")
    if target.get("planning_only") is not True or target.get("must_not_modify_shared_helper") is not True:
        raise RuntimeError("143R target 143U plan is not planning-only")
    return {
        "root": rel(root),
        "decision": decision,
        "duplicate_conflict_trace_report": trace,
        "helper_duplicate_marker_semantics_audit": helper,
        "selected_marker_occurrence_policy_matrix": policy,
        "repair_options_matrix": options,
        "root_cause_report": root_report,
        "target_143u_milestone_plan": target,
    }


def primitive_repair_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_143u_primitive_repair_requirements_v1",
        "selected_option": SELECTED_OPTION,
        "repair_goal": "reject/fallback when the selected marker candidate-line count is not exactly one",
        "planning_only": True,
        "must_not_modify_shared_helper": True,
        "must_not_call_helper_generation": True,
        "candidate_line_counting_required": True,
        "recommended_occurrence_count_method": "line.strip().startswith(selected_marker)",
        "count_only_actual_selected_marker_candidate_lines": True,
        "do_not_count_selected_marker_mentions_in_prose_or_instructions": True,
        "policy_applies_only_to_selected_marker": True,
        "non_selected_marker_duplicates_out_of_scope": True,
        "same_value_duplicate_acceptance_deferred": True,
        "boundary": BOUNDARY_TEXT,
    }


def selected_marker_occurrence_policy_matrix() -> dict[str, Any]:
    return {
        "schema_version": "phase_143u_selected_marker_occurrence_policy_matrix_v1",
        "recommended_policy": SELECTED_OPTION,
        "candidate_line_definition": "An actual candidate marker line is counted only when line.strip().startswith(selected_marker).",
        "selected_marker_occurrence_policy_applies_only_to_the_selected_marker": True,
        "non_selected_marker_duplicates_out_of_scope": True,
        "same_value_duplicate_acceptance_deferred": True,
        "rows": [
            {"case_id": "zero_selected_marker_candidate_lines", "expected_behavior": "fallback"},
            {"case_id": "one_selected_marker_candidate_line", "expected_behavior": "extract_value_from_that_line"},
            {"case_id": "two_or_more_selected_marker_candidate_lines_same_value", "expected_behavior": "fallback_for_now"},
            {"case_id": "two_or_more_selected_marker_candidate_lines_conflicting_values", "expected_behavior": "fallback"},
        ],
    }


def repair_options_matrix() -> dict[str, Any]:
    options = [
        {
            "option_id": "selected_marker_occurrence_count_must_equal_one",
            "mechanism": "count selected marker candidate-lines with line-prefix matching; fallback unless exactly one candidate-line is present",
            "scope_cost": 1,
            "oracle_risk": "low",
            "shortcut_risk": "low",
            "prose_false_positive_risk": "low if line-prefix matching is used",
            "helper_change_required_in_143v": True,
            "request_key_change_required": False,
            "recommendation": "selected",
        },
        {
            "option_id": "duplicate_same_value_allowed_conflicting_rejected",
            "mechanism": "accept duplicate selected marker candidate-lines only when all extracted values are identical",
            "scope_cost": 2,
            "oracle_risk": "low",
            "shortcut_risk": "medium",
            "prose_false_positive_risk": "medium",
            "helper_change_required_in_143v": True,
            "request_key_change_required": False,
            "recommendation": "deferred",
        },
        {
            "option_id": "keep_first_occurrence_policy",
            "mechanism": "keep current prompt.find(selected_marker) first occurrence behavior",
            "scope_cost": 0,
            "oracle_risk": "medium",
            "shortcut_risk": "high",
            "prose_false_positive_risk": "high",
            "helper_change_required_in_143v": False,
            "request_key_change_required": False,
            "recommendation": "rejected",
        },
    ]
    return {
        "schema_version": "phase_143u_repair_options_matrix_v1",
        "selected_option": SELECTED_OPTION,
        "options": options,
    }


def selected_repair_recommendation() -> dict[str, Any]:
    return {
        "schema_version": "phase_143u_selected_repair_recommendation_v1",
        "selected_option": SELECTED_OPTION,
        "decision": DECISION,
        "next": NEXT,
        "reason": "143R confirmed first-occurrence selected marker semantics without any occurrence-count policy.",
        "implementation_summary_for_143v": [
            "parse exactly one winner label",
            "map selected pocket to selected marker",
            "count only selected marker candidate-lines with line-prefix matching",
            "fallback unless the selected marker candidate-line count is exactly one",
            "extract value from the single selected marker candidate-line only",
        ],
        "claim_limit": "143V positive would prove duplicate selected marker conflict rejection for prompt-visible selected-pocket binding only.",
        "same_value_duplicate_acceptance_claimed": False,
    }


def anti_oracle_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_143u_anti_oracle_requirements_v1",
        "helper_request_keys_must_remain": ["prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"],
        "forbid_request_key_changes": True,
        "forbid_per_row_selected_pocket_metadata": True,
        "forbid_per_row_manifest_switching": True,
        "forbid_payload_marker_list_narrowed_to_correct_pocket": True,
        "forbid_hidden_final_winner_value_gold_answer_markers": True,
        "forbid_post_generation_repair": True,
        "forbid_broad_capability_claims": True,
    }


def target_143v_milestone_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_143u_target_143v_milestone_plan_v1",
        "milestone": NEXT,
        "helper_changing_prototype": True,
        "decoder": "deterministic_pocket_gated_rule_selected_pocket_binding_decoder",
        "allowed_helper_change_scope": {
            "only_function_may_change": "_instnct_select_rule_selected_pocket_value",
            "request_validation_must_not_change": True,
            "allowed_request_keys_must_not_change": True,
            "forbidden_request_keys_must_not_loosen": True,
            "old_decoder_path_must_remain_unchanged": True,
            "non_instnct_raw_generation_path_must_remain_unchanged": True,
        },
        "intended_helper_behavior": [
            "parse exactly one winner label",
            "map selected pocket to selected marker",
            "count selected marker candidate-lines",
            "if selected marker candidate-line count != 1: fallback",
            "if selected marker candidate-line count == 1: extract value from that line only",
        ],
        "selected_marker_occurrence_counting_policy": {
            "method": "line.strip().startswith(selected_marker)",
            "count_only_actual_selected_marker_candidate_lines": True,
            "do_not_count_selected_marker_mentions_in_prose_or_instructions": True,
            "policy_applies_only_to_selected_marker": True,
        },
        "required_controls": [
            "DUPLICATE_SELECTED_MARKER_CONFLICT_CONTROL",
            "DUPLICATE_SELECTED_MARKER_SAME_VALUE_CONTROL",
            "DUPLICATE_NON_SELECTED_MARKER_SCOPE_CONTROL",
            "SELECTED_MARKER_MENTION_IN_PROSE_CONTROL",
            "ZERO_SELECTED_MARKER_FALLBACK_CONTROL",
            "SINGLE_SELECTED_MARKER_POSITIVE_CONTROL",
            "SELECTED_MARKER_VALUE_MISSING_CONTROL",
            "WINNER_LABEL_MISSING_AMBIGUOUS_CONTROL",
            "POCKET_MARKER_ORDER_PERMUTATION_CONTROL",
            "LEGACY_MANIFEST_REGRESSION_CONTROL",
            "STATIC_MANIFEST_INTEGRITY_CONTROL",
            "HELPER_REQUEST_AUDIT_CONTROL",
        ],
        "positive_gates": {
            "single_selected_marker_binding_accuracy": ">= 0.95",
            "selected_marker_line_occurrence_count_accuracy": ">= 0.95",
            "selected_marker_prose_mention_false_positive_rate": "= 0.0",
            "duplicate_selected_marker_conflict_rejection_rate": ">= 0.95",
            "duplicate_selected_marker_same_value_rejection_rate": ">= 0.95",
            "duplicate_non_selected_marker_binding_accuracy": ">= 0.95",
            "duplicate_non_selected_marker_regression_rate": "= 0.0",
            "zero_selected_marker_fallback_rate": ">= 0.95",
            "selected_marker_value_missing_fallback_rate": ">= 0.95",
            "pocket_marker_order_permutation_accuracy": ">= 0.95",
            "per_row_manifest_switch_rate": "= 0.0",
            "per_row_payload_marker_switch_rate": "= 0.0",
            "helper_request_forbidden_metadata_count": "= 0",
            "legacy_manifest_regression_passed": True,
            "deterministic_replay_passed": True,
        },
        "positive_route": {
            "decision": "selected_marker_occurrence_count_rejection_prototype_positive",
            "next": "143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM",
        },
        "clean_negative_routes": {
            "duplicate_conflict_still_not_rejected": "143X_SELECTED_MARKER_OCCURRENCE_REPAIR_FAILURE_ANALYSIS",
            "overbroad_duplicate_rejection_detected": "143Y_NON_SELECTED_MARKER_DUPLICATE_REGRESSION_ANALYSIS",
            "single_marker_binding_regression": "143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
        "boundary": BOUNDARY_TEXT,
    }


def risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_143u_risk_register_v1",
        "risks": [
            {
                "risk_id": "naive_string_count_counts_prose_marker_mentions",
                "mitigation": "143V must count candidate-lines with line.strip().startswith(selected_marker), not raw substring occurrences.",
            },
            {
                "risk_id": "overbroad_duplicate_rejection",
                "mitigation": "143V must include duplicate non-selected marker controls and keep the policy scoped to the selected marker.",
            },
            {
                "risk_id": "same_value_duplicate_overclaim",
                "mitigation": "143V must fallback on same-value selected marker duplicates for now; acceptance is deferred.",
            },
            {
                "risk_id": "helper_scope_creep",
                "mitigation": "143V may change only _instnct_select_rule_selected_pocket_value and must leave request validation and old paths unchanged.",
            },
        ],
    }


def write_report(out: Path, decision: dict[str, Any]) -> None:
    text = f"""# {MILESTONE} Report

Decision: `{decision['decision']}`

Selected option: `{decision['selected_option']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Repair Policy

143U recommends `selected_marker_occurrence_count_must_equal_one`.

```text
0 selected marker candidate-lines -> fallback
1 selected marker candidate-line -> extract value
2+ selected marker candidate-lines -> fallback
```

The future 143V prototype must count only actual selected marker candidate-lines using line-prefix matching. It must not count selected marker mentions in prose or reject duplicate non-selected markers.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 143U duplicate selected marker conflict rejection primitive plan")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-143r-root", type=Path, default=DEFAULT_143R_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_143u_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_143r(resolve_repo_path(args.upstream_143r_root))
    write_json(out / "upstream_143r_manifest.json", upstream)
    append_progress(out, "upstream_143r_verified", decision=upstream["decision"]["decision"])

    config = {
        "schema_version": "phase_143u_analysis_config_v1",
        "milestone": MILESTONE,
        "planning_only": True,
        "artifact_only": True,
        "helper_repair_implemented": False,
        "new_helper_generation_run": False,
        "shared_helper_imported": False,
        "shared_helper_called": False,
        "training_performed": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "helper_backend_modified": False,
        "request_key_change_allowed": False,
        "runtime_surface_mutated": False,
        "product_surface_mutated": False,
        "release_surface_mutated": False,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }
    requirements = primitive_repair_requirements()
    policy = selected_marker_occurrence_policy_matrix()
    options = repair_options_matrix()
    recommendation = selected_repair_recommendation()
    anti_oracle = anti_oracle_requirements()
    target_plan = target_143v_milestone_plan()
    risks = risk_register()
    decision = {
        "schema_version": "phase_143u_decision_v1",
        "decision": DECISION,
        "selected_option": SELECTED_OPTION,
        "verdict": "DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_PRIMITIVE_PLAN_RECOMMENDED",
        "next": NEXT,
        "target_positive_next": "143W_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRM",
        "boundary": BOUNDARY_TEXT,
        "helper_repair_implemented": False,
        "positive_capability_claimed": False,
        **FALSE_FLAGS,
    }
    summary = {
        "schema_version": "phase_143u_summary_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "decision": decision,
        "primitive_repair_requirements": requirements,
        "selected_marker_occurrence_policy_matrix": policy,
        "target_143v_milestone_plan": target_plan,
        **FALSE_FLAGS,
    }

    write_json(out / "analysis_config.json", config)
    write_json(out / "primitive_repair_requirements.json", requirements)
    write_json(out / "selected_marker_occurrence_policy_matrix.json", policy)
    write_json(out / "repair_options_matrix.json", options)
    write_json(out / "selected_repair_recommendation.json", recommendation)
    write_json(out / "anti_oracle_requirements.json", anti_oracle)
    write_json(out / "target_143v_milestone_plan.json", target_plan)
    write_json(out / "risk_register.json", risks)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_143u_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "selected_option": decision["selected_option"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
