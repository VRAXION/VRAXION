#!/usr/bin/env python3
"""143J artifact-only rule-selected pocket payload binding primitive plan.

This phase reads the 143I diagnosis and writes a machine-readable plan for the
first 143K prototype. It does not implement the helper primitive, call helper
generation, train, mutate checkpoints, or modify helper/backend/request keys.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_143j_rule_selected_pocket_payload_binding_helper_primitive_plan/smoke")
DEFAULT_143I_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_143i_no_resolved_final_marker_bridge_failure_analysis/smoke")
DECISION = "rule_selected_pocket_payload_binding_primitive_plan_recommended"
SELECTED_OPTION = "prompt_level_explicit_winner_label_parser_plus_static_marker_map"
NEXT = "143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE"
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "gpt_like_readiness_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
}
BOUNDARY_TEXT = (
    "143J is planning-only constrained helper/backend primitive design. It does "
    "not implement helper/backend behavior, call helper generation, train, "
    "mutate checkpoints, change helper request keys, modify runtime/product/"
    "release/deploy surfaces, and is not architecture superiority. It is not "
    "open-ended reasoning, not rule metadata reasoning, not open-domain or "
    "broad assistant capability, and not production/public API/deployment/"
    "safety readiness."
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


def require_143i(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "root_cause_report.json",
        "target_143j_milestone_plan.json",
        "helper_selection_semantics_audit.json",
        "alternative_hypothesis_matrix.json",
        "bridge_options_matrix.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 143I artifacts: {missing}")
    decision = read_json(root / "decision.json")
    root_report = read_json(root / "root_cause_report.json")
    plan = read_json(root / "target_143j_milestone_plan.json")
    helper = read_json(root / "helper_selection_semantics_audit.json")
    hypotheses = read_json(root / "alternative_hypothesis_matrix.json")
    options = read_json(root / "bridge_options_matrix.json")
    if decision.get("decision") != "no_resolved_final_marker_bridge_failure_analysis_complete":
        raise RuntimeError(f"bad 143I decision: {decision.get('decision')}")
    if decision.get("root_cause_id") != "helper_payload_marker_selection_lacks_rule_selected_pocket_binding":
        raise RuntimeError(f"bad 143I root cause: {decision.get('root_cause_id')}")
    if decision.get("next") != "143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN":
        raise RuntimeError(f"bad 143I next: {decision.get('next')}")
    required_true = [
        "supported_by_143f_clean_dependency",
        "supported_by_helper_source_audit",
        "hidden_marker_leak_rejected",
        "per_row_manifest_oracle_rejected",
        "shortcut_failure_rejected",
        "abc_static_first_marker_behavior_confirmed",
    ]
    for key in required_true:
        if root_report.get(key) is not True:
            raise RuntimeError(f"143I root cause evidence missing: {key}")
    if helper.get("rule_selected_pocket_binding_found") is not False:
        raise RuntimeError("143I helper audit unexpectedly found rule-selected binding")
    if plan.get("planning_only") is not True or plan.get("recommended_next") != NEXT:
        raise RuntimeError("143I target 143J plan route mismatch")
    return {
        "root": rel(root),
        "decision": decision,
        "root_cause_report": root_report,
        "target_143j_milestone_plan": plan,
        "helper_selection_semantics_audit": helper,
        "alternative_hypothesis_matrix": hypotheses,
        "bridge_options_matrix": options,
    }


def primitive_design_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_143j_primitive_design_requirements_v1",
        "minimum_desired_primitive": "prompt-visible selected pocket id -> static pocket marker -> value extraction",
        "main_selected_option": SELECTED_OPTION,
        "static_marker_map_only": {
            "pocket_a": "pocket A candidate:",
            "pocket_b": "pocket B candidate:",
            "pocket_c": "pocket C candidate:",
        },
        "manifest_must_not_carry_per_row_selected_pocket_id": True,
        "helper_request_keys_must_remain": ["prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"],
        "143k_positive_claim_limit": "143K positive proves prompt-visible selected-pocket binding only; it does not prove rule metadata reasoning or open-ended arbitration.",
        "planning_only": True,
    }


def primitive_options_matrix() -> dict[str, Any]:
    options = [
        {
            "option_id": "prompt_level_explicit_winner_label_parser_plus_static_marker_map",
            "mechanism": "parse winner=pocket_a|pocket_b|pocket_c from prompt, map to a static marker, then extract that marker's value",
            "scope_cost": 1,
            "oracle_risk": "medium",
            "shortcut_risk": "medium",
            "diagnostic_value": "high",
            "implementation_risk": "medium",
            "helper_change_required": True,
            "request_key_change_required": False,
            "recommended_order": 1,
            "recommendation": "selected",
        },
        {
            "option_id": "static_pocket_marker_map_plus_prompt_selected_pocket_binding",
            "mechanism": "manifest defines only stable pocket-to-marker mapping; selected pocket must come from prompt text",
            "scope_cost": 2,
            "oracle_risk": "medium if selected pocket leaks through manifest",
            "shortcut_risk": "medium",
            "diagnostic_value": "high",
            "implementation_risk": "medium",
            "helper_change_required": True,
            "request_key_change_required": False,
            "recommended_order": 2,
            "recommendation": "supporting_design_constraint",
        },
        {
            "option_id": "rule_metadata_parser",
            "mechanism": "derive selected pocket from quorum/recency/tie-break/hierarchy metadata, then map to static marker",
            "scope_cost": 4,
            "oracle_risk": "low if prompt-derived only",
            "shortcut_risk": "high",
            "diagnostic_value": "medium_after_winner_label_binding",
            "implementation_risk": "high",
            "helper_change_required": True,
            "request_key_change_required": False,
            "recommended_order": 3,
            "recommendation": "defer_until_explicit_winner_binding_is_clean",
        },
        {
            "option_id": "keep_resolved_final_marker_only",
            "mechanism": "continue using resolved final payload markers and do not bridge no-resolved rows",
            "scope_cost": 0,
            "oracle_risk": "high for bridge claims",
            "shortcut_risk": "high for overclaim",
            "diagnostic_value": "low",
            "implementation_risk": "low",
            "helper_change_required": False,
            "request_key_change_required": False,
            "recommended_order": 4,
            "recommendation": "not_recommended_for_no_resolved_bridge",
        },
    ]
    return {"schema_version": "phase_143j_primitive_options_matrix_v1", "selected_option": SELECTED_OPTION, "options": options}


def oracle_risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_143j_oracle_shortcut_risk_register_v1",
        "risks": [
            {"risk": "per-row selected_pocket_id in helper request metadata", "mitigation": "request keys stay unchanged and checker rejects forbidden metadata"},
            {"risk": "per-row manifest switching", "mitigation": "one static manifest/checkpoint path/hash for main arm"},
            {"risk": "payload marker list narrowed to correct pocket", "mitigation": "static all-pocket marker map required"},
            {"risk": "hidden winner-value/final/gold/answer marker", "mitigation": "prompt scanner allows only winner=pocket_* and rejects value-bearing aliases"},
            {"risk": "first prompt marker scan reappears", "mitigation": "pocket marker order permutation control and first prompt marker shortcut metric"},
            {"risk": "winner label overclaimed as rule reasoning", "mitigation": "boundary text says selected-pocket binding only"},
        ],
    }


def selected_recommendation() -> dict[str, Any]:
    return {
        "schema_version": "phase_143j_selected_primitive_recommendation_v1",
        "selected_option": SELECTED_OPTION,
        "decision": DECISION,
        "next": NEXT,
        "rationale": [
            "smallest honest bridge after 143I",
            "tests selected-pocket binding before full rule parsing",
            "keeps selected pocket prompt-visible rather than request metadata",
            "does not require helper request-key changes",
        ],
        "claim_limit": "143K positive would prove prompt-visible selected-pocket binding only. It would not prove rule metadata reasoning. It would not prove open-ended arbitration.",
    }


def target_143k_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_143j_target_143k_milestone_plan_v1",
        "milestone": "143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE",
        "executable_probe": True,
        "intended_primitive": "parse winner=pocket_a|pocket_b|pocket_c from prompt, map selected pocket to static marker, extract value from that marker, emit ANSWER=E<value>",
        "helper_request_keys": ["prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"],
        "must_forbid": [
            "per-row selected_pocket_id in helper request metadata",
            "per-row manifest switching",
            "payload marker list narrowed to only the correct pocket",
            "hidden final/winner-value/gold/answer marker equivalents",
            "resolved final payload marker in main no-resolved rows",
            "helper request keys changed or extended",
            "post-generation repair",
            "broad architecture claims",
        ],
        "required_controls": [
            "WINNER_LABEL_WRONG_POCKET_CONTROL",
            "WINNER_LABEL_MISSING_CONTROL",
            "WINNER_LABEL_AMBIGUOUS_CONTROL",
            "WINNER_LABEL_POSITION_INVARIANCE_CONTROL",
            "POCKET_MARKER_ORDER_PERMUTATION_CONTROL",
            "SAME_VALUES_DIFFERENT_WINNER_CONTROL",
            "SAME_WINNER_DIFFERENT_VALUES_CONTROL",
            "FIRST_PROMPT_MARKER_SHORTCUT_CONTROL",
            "FIRST_POCKET_CONTROL",
            "DEFAULT_POCKET_CONTROL",
            "LAST_POCKET_CONTROL",
            "STALE_POCKET_CONTROL",
            "VISIBLE_BYPASS_CONTROL",
            "NOISY_DISTRACTOR_CONTROL",
            "CLOSED_POCKET_ABLATION_CONTROL",
            "STATIC_MANIFEST_INTEGRITY_CONTROL",
        ],
        "prompt_scanner": {
            "allowed": ["winner=pocket_a", "winner=pocket_b", "winner=pocket_c"],
            "forbidden": [
                "winner value",
                "selected value",
                "final winner",
                "answer pocket",
                "answer value",
                "gold pocket",
                "gold value",
                "target value",
                "resolved output",
                "expected output",
                "arbitrated final",
                "selected final",
            ],
        },
        "required_metrics": [
            "winner_label_parse_accuracy",
            "selected_pocket_to_marker_binding_accuracy",
            "pocket_marker_order_permutation_accuracy",
            "first_prompt_marker_shortcut_rate",
            "ambiguous_winner_label_rejection_rate",
            "missing_winner_label_fallback_rate",
            "same_values_different_winner_accuracy",
            "same_winner_different_values_accuracy",
        ],
        "positive_gates": {
            "winner_label_parse_accuracy": ">= 0.70",
            "selected_pocket_to_marker_binding_accuracy": ">= 0.70",
            "pocket_marker_order_permutation_accuracy": ">= 0.70",
            "first_prompt_marker_shortcut_rate": "= 0.0",
            "missing_winner_label_fallback_rate": ">= 0.80",
            "ambiguous_winner_label_rejection_rate": ">= 0.80",
            "same_values_different_winner_accuracy": ">= 0.70",
            "same_winner_different_values_accuracy": ">= 0.70",
            "main_pocket_writeback_rate": ">= 0.80",
            "closed_pocket_ablation_accuracy": "<= 0.15",
            "visible_noisy_violation_rates": "= 0.0",
            "per_row_manifest_switch_rate": "= 0.0",
            "per_row_payload_marker_switch_rate": "= 0.0",
            "helper_request_forbidden_metadata_count": "= 0",
            "deterministic_replay_passed": True,
        },
        "clean_negative_routes": {
            "winner_label_binding_failure": "143L_WINNER_LABEL_BINDING_FAILURE_ANALYSIS",
            "oracle_manifest_shortcut_detected": "143M_ORACLE_MANIFEST_SHORTCUT_ANALYSIS",
            "positional_pocket_shortcut_detected": "143D_POSITIONAL_POCKET_SHORTCUT_ANALYSIS",
            "ambiguous_winner_not_rejected": "143N_AMBIGUOUS_WINNER_LABEL_ANALYSIS",
            "missing_winner_not_rejected": "143O_MISSING_WINNER_LABEL_ANALYSIS",
            "pocket_ablation_not_decision_critical": "141D_POCKET_CAUSALITY_FAILURE_ANALYSIS",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
        "claim_limit": "143K positive proves prompt-visible selected-pocket binding only, not rule metadata reasoning or open-ended arbitration.",
    }


def write_report(out: Path, decision: dict[str, Any], recommendation: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Selected option: `{decision['selected_option']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Recommendation

Use prompt-visible explicit winner labels plus a static pocket marker map as the
first prototype axis. The manifest may define stable marker names for pocket A,
pocket B, and pocket C, but it must not carry per-row selected pocket identity.

143K positive would prove prompt-visible selected-pocket binding only. It would
not prove rule metadata reasoning. It would not prove open-ended arbitration.

## Anti-Oracle Requirements

- No per-row selected pocket metadata in helper requests.
- No per-row manifest or checkpoint switching.
- No payload marker list narrowed to only the correct pocket.
- No hidden final/winner-value/gold/answer marker equivalents.
- Helper request keys remain unchanged.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-143i-root", type=Path, default=DEFAULT_143I_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_143j_queue_v1", "milestone": MILESTONE, "status": "running"})

    config = {
        "schema_version": "phase_143j_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "planning_only": True,
        "artifact_only": True,
        "helper_generation_run": False,
        "shared_helper_imported": False,
        "shared_helper_called": False,
        "training_performed": False,
        "checkpoint_mutated": False,
        "helper_backend_modified": False,
        "helper_request_key_change": False,
        "runtime_surface_mutated": False,
        "product_surface_mutated": False,
        "release_surface_mutated": False,
        **FALSE_FLAGS,
    }
    write_json(out / "analysis_config.json", config)

    upstream = require_143i(resolve_repo_path(args.upstream_143i_root))
    write_json(out / "upstream_143i_manifest.json", upstream)
    append_progress(out, "upstream verification", decision=upstream["decision"]["decision"])

    requirements = primitive_design_requirements()
    options = primitive_options_matrix()
    risks = oracle_risk_register()
    recommendation = selected_recommendation()
    target_plan = target_143k_plan()
    decision = {
        "schema_version": "phase_143j_decision_v1",
        "decision": DECISION,
        "selected_option": SELECTED_OPTION,
        "next": NEXT,
        "planning_only": True,
        "artifact_only": True,
        "boundary": BOUNDARY_TEXT,
        "helper_backend_modified": False,
        "helper_request_key_change": False,
        "rule_metadata_reasoning_claimed": False,
        "open_ended_arbitration_claimed": False,
        "architecture_superiority_claimed": False,
        **FALSE_FLAGS,
    }
    summary = {
        "schema_version": "phase_143j_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "upstream_143i": upstream,
        "primitive_design_requirements": requirements,
        "primitive_options_matrix": options,
        "selected_primitive_recommendation": recommendation,
        "target_143k_milestone_plan": target_plan,
        **decision,
    }

    write_json(out / "primitive_design_requirements.json", requirements)
    write_json(out / "primitive_options_matrix.json", options)
    write_json(out / "oracle_shortcut_risk_register.json", risks)
    write_json(out / "selected_primitive_recommendation.json", recommendation)
    write_json(out / "target_143k_milestone_plan.json", target_plan)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, recommendation)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", selected_option=decision["selected_option"])
    write_json(out / "queue.json", {"schema_version": "phase_143j_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
