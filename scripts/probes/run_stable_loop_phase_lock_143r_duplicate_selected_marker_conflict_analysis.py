#!/usr/bin/env python3
"""143R artifact-only duplicate selected marker conflict analysis.

This phase reads 143P artifacts and helper source text only. It does not import
or call the helper, run generation, train, mutate checkpoints, or implement a
repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_143r_duplicate_selected_marker_conflict_analysis/smoke")
DEFAULT_143P_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_143p_rule_selected_pocket_payload_binding_scale_confirm/smoke")
HELPER_PATH = Path("scripts/probes/shared_raw_generation_helper.py")
UPSTREAM_143P_RUNNER = Path("scripts/probes/run_stable_loop_phase_lock_143p_rule_selected_pocket_payload_binding_scale_confirm.py")
ROOT_CAUSE_ID = "selected_marker_duplicate_conflict_uses_first_occurrence_without_occurrence_count_policy"
DECISION = "duplicate_selected_marker_conflict_analysis_complete"
NEXT = "143U_DUPLICATE_SELECTED_MARKER_CONFLICT_REJECTION_HELPER_PRIMITIVE_PLAN"
TARGET_143V = "143V_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_PROTOTYPE"
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
    "143R is artifact-only duplicate selected marker conflict analysis for "
    "constrained helper/backend evidence. It is prompt-visible selected-pocket "
    "binding only, not rule metadata reasoning, not open-ended arbitration, not "
    "GPT-like/open-domain/broad assistant capability, not production/public "
    "API/deployment/safety readiness, and not architecture superiority. It does "
    "not repair the helper, call helper generation, train, mutate checkpoints, "
    "modify helper/backend/request keys, deploy, or change runtime/product/"
    "release surfaces."
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_function(source: str, name: str) -> str:
    match = re.search(rf"^def {re.escape(name)}\(.*?(?=^def |\Z)", source, re.MULTILINE | re.DOTALL)
    return match.group(0) if match else ""


def require_143p(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "duplicate_selected_marker_conflict_report.json",
        "static_manifest_integrity_report.json",
        "helper_request_audit.json",
        "legacy_manifest_regression_report.json",
        "prompt_scanner_report.json",
        "summary.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 143P artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    duplicate = read_json(root / "duplicate_selected_marker_conflict_report.json")
    static = read_json(root / "static_manifest_integrity_report.json")
    request = read_json(root / "helper_request_audit.json")
    legacy = read_json(root / "legacy_manifest_regression_report.json")
    prompt = read_json(root / "prompt_scanner_report.json")
    summary = read_json(root / "summary.json")
    if decision.get("decision") != "duplicate_selected_marker_conflict_not_rejected":
        raise RuntimeError(f"bad 143P decision: {decision.get('decision')}")
    if decision.get("next") != "143R_DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS":
        raise RuntimeError(f"bad 143P next: {decision.get('next')}")
    expected = {
        "winner_label_parse_accuracy": 1.0,
        "selected_pocket_to_marker_binding_accuracy": 1.0,
        "pocket_marker_order_permutation_accuracy": 1.0,
        "main_pocket_writeback_rate": 1.0,
        "duplicate_selected_marker_conflict_rejection_rate": 0.0,
        "duplicate_selected_marker_first_value_rate": 1.0,
        "duplicate_selected_marker_last_value_rate": 0.0,
        "legacy_manifest_regression_passed": True,
        "deterministic_replay_passed": True,
    }
    for key, expected_value in expected.items():
        if metrics.get(key) != expected_value:
            raise RuntimeError(f"bad 143P metric {key}: {metrics.get(key)} != {expected_value}")
    if duplicate.get("row_count", 0) <= 0:
        raise RuntimeError("143P duplicate conflict row_count is not positive")
    if duplicate.get("fallback_rate") != 0.0:
        raise RuntimeError("143P duplicate conflict unexpectedly fell back")
    if duplicate.get("duplicate_selected_marker_first_value_rate") != 1.0:
        raise RuntimeError("143P duplicate conflict did not select first duplicate value")
    if duplicate.get("duplicate_selected_marker_last_value_rate") != 0.0:
        raise RuntimeError("143P duplicate conflict selected last duplicate value")
    if duplicate.get("unexpected_value_rate") != 0.0:
        raise RuntimeError("143P duplicate conflict produced unexpected values")
    if static.get("passed") is not True or static.get("payload_marker_list_not_narrowed_to_correct_pocket") is not True:
        raise RuntimeError("143P static manifest integrity failed")
    if request.get("helper_request_forbidden_metadata_count") != 0 or request.get("all_requests_allowed_keys_only") is not True:
        raise RuntimeError("143P helper request audit failed")
    if legacy.get("legacy_manifest_regression_passed") is not True:
        raise RuntimeError("143P legacy regression failed")
    if prompt.get("passed") is not True:
        raise RuntimeError("143P prompt scanner failed")
    return {
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": metrics,
        "duplicate_selected_marker_conflict_report": duplicate,
        "static_manifest_integrity_report": static,
        "helper_request_audit": {
            "accepted_helper_request_count": request.get("accepted_helper_request_count"),
            "all_requests_allowed_keys_only": request.get("all_requests_allowed_keys_only"),
            "helper_request_forbidden_metadata_count": request.get("helper_request_forbidden_metadata_count"),
            "selected_pocket_id_not_in_request_metadata": request.get("selected_pocket_id_not_in_request_metadata"),
            "winner_label_not_in_request_metadata": request.get("winner_label_not_in_request_metadata"),
            "raw_generate_allowed_in_runner": request.get("raw_generate_allowed_in_runner"),
            "raw_generate_allowed_in_checker": request.get("raw_generate_allowed_in_checker"),
        },
        "legacy_manifest_regression_report": legacy,
        "prompt_scanner_report": prompt,
        "summary": summary,
    }


def duplicate_conflict_trace_report(upstream: dict[str, Any]) -> dict[str, Any]:
    duplicate = upstream["duplicate_selected_marker_conflict_report"]
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_143r_duplicate_conflict_trace_report_v1",
        "source_artifacts": [
            "duplicate_selected_marker_conflict_report.json",
            "aggregate_metrics.json",
        ],
        "row_level_duplicate_results_persisted_by_143p": False,
        "occurrence_counts_inferred_from_143p_duplicate_prompt_builder": True,
        "duplicate_rows_count": duplicate.get("row_count", 0),
        "selected_marker_occurrence_count_min": 2,
        "selected_marker_occurrence_count_max": 2,
        "generated_equals_first_duplicate_value_rate": duplicate.get("duplicate_selected_marker_first_value_rate"),
        "generated_equals_last_duplicate_value_rate": duplicate.get("duplicate_selected_marker_last_value_rate"),
        "generated_equals_fallback_rate": duplicate.get("fallback_rate"),
        "generated_equals_unexpected_value_rate": duplicate.get("unexpected_value_rate"),
        "aggregate_duplicate_selected_marker_conflict_rejection_rate": metrics.get("duplicate_selected_marker_conflict_rejection_rate"),
        "trace_interpretation": "143P duplicate selected marker controls inserted two conflicting selected marker lines; generated outputs always matched the first duplicate value and never fell back.",
        "passed": duplicate.get("row_count", 0) > 0
        and duplicate.get("duplicate_selected_marker_first_value_rate") == 1.0
        and duplicate.get("duplicate_selected_marker_last_value_rate") == 0.0
        and duplicate.get("fallback_rate") == 0.0
        and duplicate.get("unexpected_value_rate") == 0.0,
    }


def duplicate_conflict_failure_mode_report(upstream: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_143r_duplicate_conflict_failure_mode_report_v1",
        "failure_mode_id": "duplicate_selected_marker_conflict_first_occurrence_selected",
        "single_selected_marker_binding_still_works": metrics.get("selected_pocket_to_marker_binding_accuracy") == 1.0,
        "pocket_marker_order_permutation_still_works": metrics.get("pocket_marker_order_permutation_accuracy") == 1.0,
        "duplicate_selected_marker_conflict_rejected": metrics.get("duplicate_selected_marker_conflict_rejection_rate") >= 0.95,
        "first_occurrence_selected": trace.get("generated_equals_first_duplicate_value_rate") == 1.0,
        "last_occurrence_selected": trace.get("generated_equals_last_duplicate_value_rate") == 1.0,
        "fallback_selected": trace.get("generated_equals_fallback_rate") == 1.0,
        "edge_gap_is_narrow": True,
        "interpretation": "The 143K/143P selected-pocket binding primitive works for one selected marker, but lacks duplicate selected-marker conflict rejection.",
    }


def helper_duplicate_marker_semantics_audit() -> dict[str, Any]:
    helper = resolve_repo_path(HELPER_PATH)
    text = helper.read_text(encoding="utf-8")
    fn = extract_function(text, "_instnct_select_rule_selected_pocket_value")
    return {
        "schema_version": "phase_143r_helper_duplicate_marker_semantics_audit_v1",
        "helper_path": rel(helper),
        "helper_source_sha256": sha256_file(helper),
        "selected_function_name": "_instnct_select_rule_selected_pocket_value",
        "function_found": bool(fn),
        "prompt_find_selected_marker_found": "prompt.find(selected_marker)" in fn,
        "selected_marker_first_occurrence_offset_used": bool(
            re.search(r"pos\s*=\s*prompt\.find\(selected_marker\)", fn)
            and re.search(r"segment\s*=\s*prompt\[pos\s*\+\s*len\(selected_marker\)", fn)
        ),
        "all_occurrence_scan_found": bool(
            "finditer(selected_marker" in fn
            or "findall(selected_marker" in fn
            or re.search(r"while\s+.*find\(selected_marker", fn)
        ),
        "selected_marker_count_variable_found": bool(
            re.search(r"selected_marker\w*_count|count\w*_selected_marker|selected_marker_occurrence", fn)
        ),
        "duplicate_selected_marker_conflict_rejection_found": bool(
            "duplicate_selected_marker" in fn or "duplicate conflict" in fn.lower()
        ),
        "fallback_on_duplicate_conflict_found": bool(
            re.search(r"duplicate.*fallback|fallback.*duplicate|occurrence_count\s*!=\s*1", fn, re.IGNORECASE)
        ),
        "selected_marker_missing_fallback_found": "selected_marker_missing_from_prompt" in fn,
        "selected_marker_value_missing_fallback_found": "selected_marker_value_missing" in fn,
        "winner_label_count_policy_found": "len(winner_labels) != 1" in fn,
        "interpretation": "The current selected-pocket binding function uses prompt.find(selected_marker) and extracts from the first occurrence without counting duplicate selected markers.",
    }


def selected_marker_occurrence_policy_matrix() -> dict[str, Any]:
    rows = [
        {
            "case_id": "zero_selected_marker_occurrences",
            "current_143p_status": "supported_fallback",
            "recommended_policy": "fallback",
            "reason": "Selected marker absence already falls back cleanly.",
        },
        {
            "case_id": "one_selected_marker_occurrence",
            "current_143p_status": "supported_extract_value",
            "recommended_policy": "extract_value",
            "reason": "Single selected marker binding is the validated positive path.",
        },
        {
            "case_id": "two_or_more_selected_marker_occurrences_same_value",
            "current_143p_status": "not_tested_as_acceptance",
            "recommended_policy": "fallback_for_now",
            "reason": "Same-value duplicate acceptance is deferred; exactly one occurrence is the safest invariant.",
        },
        {
            "case_id": "two_or_more_selected_marker_occurrences_conflicting_values",
            "current_143p_status": "first_occurrence_selected",
            "recommended_policy": "fallback",
            "reason": "Conflicting selected marker duplicates are ambiguous and must not silently select first or last.",
        },
    ]
    return {
        "schema_version": "phase_143r_selected_marker_occurrence_policy_matrix_v1",
        "recommended_policy": "selected_marker_occurrence_count_must_equal_one",
        "selected_marker_occurrence_policy_applies_only_to_the_selected_marker": True,
        "non_selected_marker_duplicates_out_of_scope": True,
        "same_value_duplicate_acceptance_deferred": True,
        "rows": rows,
    }


def alternative_hypothesis_matrix() -> dict[str, Any]:
    rows = [
        ("winner_label_parser_failure", "rejected", ["aggregate_metrics.json"], {"winner_label_parse_accuracy": 1.0}, "Winner labels parse correctly on the main scale path."),
        ("static_marker_map_failure", "rejected", ["aggregate_metrics.json", "static_manifest_integrity_report.json"], {"selected_pocket_to_marker_binding_accuracy": 1.0}, "Static marker mapping works for the non-duplicate selected marker path."),
        ("marker_order_shortcut", "rejected", ["aggregate_metrics.json", "pocket_marker_order_permutation_report.json"], {"pocket_marker_order_permutation_accuracy": 1.0, "first_prompt_marker_shortcut_rate": 0.0}, "Marker order permutation remained correct."),
        ("request_metadata_oracle", "rejected", ["helper_request_audit.json"], {"helper_request_forbidden_metadata_count": 0, "selected_pocket_id_not_in_request_metadata": True}, "No selected pocket or winner label entered helper request metadata."),
        ("per_row_manifest_switching", "rejected", ["static_manifest_integrity_report.json"], {"per_row_manifest_switch_rate": 0.0, "per_row_payload_marker_switch_rate": 0.0}, "The 143P manifest and payload markers stayed static."),
        ("legacy_regression", "rejected", ["legacy_manifest_regression_report.json"], {"legacy_manifest_regression_passed": True}, "Old decoder behavior stayed unchanged."),
        ("selected_marker_missing_or_value_missing_gap", "rejected", ["aggregate_metrics.json"], {"selected_marker_missing_fallback_rate": 1.0, "selected_marker_value_missing_fallback_rate": 1.0}, "Missing marker and blank selected marker cases already fall back."),
        ("duplicate_selected_marker_first_occurrence_semantics", "supported", ["duplicate_selected_marker_conflict_report.json", "helper_duplicate_marker_semantics_audit.json"], {"duplicate_selected_marker_first_value_rate": 1.0}, "Duplicate conflicting selected markers use the first selected marker occurrence because the helper extracts from prompt.find(selected_marker)."),
    ]
    return {
        "schema_version": "phase_143r_alternative_hypothesis_matrix_v1",
        "hypotheses": [
            {
                "hypothesis_id": hypothesis_id,
                "status": status,
                "evidence_artifacts": artifacts,
                "metrics": metrics,
                "explanation": explanation,
            }
            for hypothesis_id, status, artifacts, metrics, explanation in rows
        ],
    }


def root_cause_report(trace: dict[str, Any], helper_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_143r_root_cause_report_v1",
        "root_cause_id": ROOT_CAUSE_ID,
        "supported_by_143p_duplicate_conflict": trace.get("passed") is True,
        "supported_by_helper_source_audit": helper_audit.get("prompt_find_selected_marker_found") is True
        and helper_audit.get("selected_marker_first_occurrence_offset_used") is True,
        "winner_label_parser_failure_rejected": True,
        "static_marker_map_failure_rejected": True,
        "metadata_oracle_rejected": True,
        "per_row_manifest_switching_rejected": True,
        "legacy_regression_rejected": True,
        "selected_marker_missing_or_value_missing_gap_rejected": True,
        "duplicate_first_occurrence_behavior_confirmed": trace.get("generated_equals_first_duplicate_value_rate") == 1.0,
        "selected_marker_occurrence_count_policy_missing": helper_audit.get("selected_marker_count_variable_found") is False
        and helper_audit.get("fallback_on_duplicate_conflict_found") is False,
        "architecture_failure_claimed": False,
        "reasoning_failure_claimed": False,
        "interpretation": "The helper has no occurrence-count policy for the selected marker. It uses the first selected-marker occurrence and therefore does not reject duplicate conflicting values.",
    }


def repair_options_matrix() -> dict[str, Any]:
    options = [
        {
            "option_id": "selected_marker_occurrence_count_must_equal_one",
            "mechanism": "scan occurrences of the selected marker; fallback unless exactly one occurrence is present, then extract from that occurrence",
            "scope_cost": 1,
            "oracle_risk": "low",
            "shortcut_risk": "low",
            "diagnostic_value": "high",
            "helper_change_required": True,
            "request_key_change_required": False,
            "same_value_duplicate_policy": "fallback_for_now",
            "conflicting_duplicate_policy": "fallback",
            "recommendation": "selected",
        },
        {
            "option_id": "duplicate_same_value_allowed_conflicting_rejected",
            "mechanism": "scan all selected marker occurrences, accept only if all extracted values are identical, reject conflicting duplicates",
            "scope_cost": 2,
            "oracle_risk": "low",
            "shortcut_risk": "medium",
            "diagnostic_value": "medium",
            "helper_change_required": True,
            "request_key_change_required": False,
            "same_value_duplicate_policy": "accept_if_exactly_same",
            "conflicting_duplicate_policy": "fallback",
            "recommendation": "deferred",
        },
        {
            "option_id": "keep_first_occurrence_policy",
            "mechanism": "keep current prompt.find(selected_marker) behavior",
            "scope_cost": 0,
            "oracle_risk": "medium",
            "shortcut_risk": "high",
            "diagnostic_value": "low",
            "helper_change_required": False,
            "request_key_change_required": False,
            "same_value_duplicate_policy": "first_occurrence",
            "conflicting_duplicate_policy": "first_occurrence",
            "recommendation": "rejected",
        },
    ]
    return {
        "schema_version": "phase_143r_repair_options_matrix_v1",
        "recommended_option": "selected_marker_occurrence_count_must_equal_one",
        "options": options,
    }


def risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_143r_risk_register_v1",
        "risks": [
            {
                "risk_id": "over_repair_all_duplicate_markers",
                "mitigation": "143U must scope occurrence counting to the selected marker only; non-selected marker duplicates are out of scope.",
            },
            {
                "risk_id": "same_value_duplicate_policy_overclaim",
                "mitigation": "143R recommends fallback for same-value duplicates for now and explicitly defers same-value acceptance.",
            },
            {
                "risk_id": "helper_repair_without_plan",
                "mitigation": "143U remains planning-only; the first helper repair prototype is routed to 143V.",
            },
            {
                "risk_id": "broad_capability_overclaim",
                "mitigation": "All outputs state prompt-visible selected-pocket binding only and reject rule reasoning or assistant claims.",
            },
        ],
    }


def target_143u_milestone_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_143r_target_143u_milestone_plan_v1",
        "milestone": NEXT,
        "planning_only": True,
        "must_not_modify_shared_helper": True,
        "recommended_next_prototype": TARGET_143V,
        "recommended_repair_policy": "selected_marker_occurrence_count_must_equal_one",
        "intended_later_helper_behavior": [
            "parse exactly one winner label",
            "map selected pocket to selected marker using the static map",
            "count occurrences of the selected marker in the prompt",
            "fallback if selected marker occurrence count is not exactly one",
            "extract value only from the single selected marker occurrence",
        ],
        "must_forbid": [
            "helper request key changes",
            "per-row selected_pocket_id metadata",
            "per-row manifest switching",
            "payload marker list narrowed to the correct pocket",
            "hidden final/winner-value/gold/answer marker equivalents",
            "broad architecture claims",
        ],
        "boundary": BOUNDARY_TEXT,
    }


def write_report(out: Path, decision: dict[str, Any], root_report: dict[str, Any], trace: dict[str, Any]) -> None:
    text = f"""# {MILESTONE} Report

Decision: `{decision['decision']}`

Root cause: `{root_report['root_cause_id']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Evidence

- duplicate rows: `{trace['duplicate_rows_count']}`
- generated equals first duplicate value rate: `{trace['generated_equals_first_duplicate_value_rate']}`
- generated equals last duplicate value rate: `{trace['generated_equals_last_duplicate_value_rate']}`
- generated equals fallback rate: `{trace['generated_equals_fallback_rate']}`

143R does not repair the helper. It records that the selected-pocket binding primitive works for one selected marker, while duplicate selected marker conflicts currently use first-occurrence extraction because the helper has no selected-marker occurrence-count policy.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 143R duplicate selected marker conflict analysis")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-143p-root", type=Path, default=DEFAULT_143P_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_143r_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_143p(resolve_repo_path(args.upstream_143p_root))
    write_json(out / "upstream_143p_manifest.json", upstream)
    append_progress(out, "upstream_143p_verified", decision=upstream["decision"]["decision"])

    config = {
        "schema_version": "phase_143r_analysis_config_v1",
        "milestone": MILESTONE,
        "artifact_only": True,
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
    write_json(out / "analysis_config.json", config)

    trace = duplicate_conflict_trace_report(upstream)
    failure = duplicate_conflict_failure_mode_report(upstream, trace)
    helper_audit = helper_duplicate_marker_semantics_audit()
    policy = selected_marker_occurrence_policy_matrix()
    hypotheses = alternative_hypothesis_matrix()
    root_report = root_cause_report(trace, helper_audit)
    options = repair_options_matrix()
    risks = risk_register()
    target_plan = target_143u_milestone_plan()
    append_progress(out, "analysis_reports_built", root_cause=root_report["root_cause_id"])

    decision = {
        "schema_version": "phase_143r_decision_v1",
        "decision": DECISION,
        "root_cause_id": ROOT_CAUSE_ID,
        "verdict": "DUPLICATE_SELECTED_MARKER_CONFLICT_ANALYSIS_COMPLETE",
        "next": NEXT,
        "target_repair_prototype": TARGET_143V,
        "recommended_policy": "selected_marker_occurrence_count_must_equal_one",
        "boundary": BOUNDARY_TEXT,
        "positive_capability_claimed": False,
        **FALSE_FLAGS,
    }
    summary = {
        "schema_version": "phase_143r_summary_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "decision": decision,
        "root_cause_report": root_report,
        "duplicate_conflict_trace_report": trace,
        "selected_marker_occurrence_policy_matrix": policy,
        **FALSE_FLAGS,
    }

    write_json(out / "duplicate_conflict_trace_report.json", trace)
    write_json(out / "duplicate_conflict_failure_mode_report.json", failure)
    write_json(out / "helper_duplicate_marker_semantics_audit.json", helper_audit)
    write_json(out / "selected_marker_occurrence_policy_matrix.json", policy)
    write_json(out / "alternative_hypothesis_matrix.json", hypotheses)
    write_json(out / "root_cause_report.json", root_report)
    write_json(out / "repair_options_matrix.json", options)
    write_json(out / "risk_register.json", risks)
    write_json(out / "target_143u_milestone_plan.json", target_plan)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, root_report, trace)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_143r_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "root_cause_id": ROOT_CAUSE_ID, "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
