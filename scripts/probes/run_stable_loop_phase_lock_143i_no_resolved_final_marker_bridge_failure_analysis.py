#!/usr/bin/env python3
"""143I artifact-only no-resolved-final-marker bridge failure analysis.

This phase reads 143F artifacts and helper source text only. It does not import
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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_143I_NO_RESOLVED_FINAL_MARKER_BRIDGE_FAILURE_ANALYSIS"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_143i_no_resolved_final_marker_bridge_failure_analysis/smoke")
DEFAULT_143F_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_143f_instnct_pocket_gated_multi_pocket_arbitration_scale_confirm/smoke")
HELPER_PATH = Path("scripts/probes/shared_raw_generation_helper.py")
ROOT_CAUSE_ID = "helper_payload_marker_selection_lacks_rule_selected_pocket_binding"
NEXT = "143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN"
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
    "143I is artifact-only failure analysis for constrained helper/backend "
    "evidence. It is not architecture failure, not open-ended reasoning, not "
    "general composition, not GPT-like/open-domain/broad assistant capability, "
    "not production/public API/deployment/safety readiness, and not architecture "
    "superiority. It does not run helper generation, train, mutate checkpoints, "
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


def require_143f(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "arm_comparison.json",
        "resolved_marker_dependency_report.json",
        "no_resolved_final_marker_subset_report.json",
        "no_resolved_abc_static_marker_control_report.json",
        "no_resolved_explicit_winner_label_subset_report.json",
        "no_resolved_rule_derived_winner_subset_report.json",
        "no_resolved_final_marker_subset_manifest.json",
        "no_resolved_prompt_scan_report.json",
        "helper_request_audit.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 143F artifacts: {missing}")
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "arm_comparison.json")
    dependency = read_json(root / "resolved_marker_dependency_report.json")
    no_resolved = read_json(root / "no_resolved_final_marker_subset_report.json")
    abc = read_json(root / "no_resolved_abc_static_marker_control_report.json")
    explicit = read_json(root / "no_resolved_explicit_winner_label_subset_report.json")
    rule = read_json(root / "no_resolved_rule_derived_winner_subset_report.json")
    manifest = read_json(root / "no_resolved_final_marker_subset_manifest.json")
    prompt_scan = read_json(root / "no_resolved_prompt_scan_report.json")
    helper_audit = read_json(root / "helper_request_audit.json")
    expected = {
        "resolved_marker_present_subset_accuracy": 1.0,
        "no_resolved_final_marker_subset_accuracy": 0.0,
        "no_resolved_final_marker_subset_fallback_rate": 1.0,
        "no_resolved_final_marker_subset_shortcut_rate": 0.0,
        "no_resolved_final_marker_subset_unexpected_value_rate": 0.0,
        "no_resolved_final_marker_subset_visible_rate": 0.0,
        "no_resolved_final_marker_subset_noisy_rate": 0.0,
        "no_resolved_final_marker_subset_train_namespace_rate": 0.0,
        "no_resolved_abc_static_first_pocket_rate": 1.0,
        "resolved_marker_dependency_delta": 1.0,
    }
    if decision.get("decision") != "resolved_final_marker_dependency_confirmed":
        raise RuntimeError(f"bad 143F decision: {decision.get('decision')}")
    if decision.get("next") != "143I_NO_RESOLVED_FINAL_MARKER_BRIDGE_FAILURE_ANALYSIS":
        raise RuntimeError(f"bad 143F next: {decision.get('next')}")
    if decision.get("clean_negative_valid") is not True:
        raise RuntimeError("143F clean_negative_valid is not true")
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"bad 143F metric {key}: {metrics.get(key)} != {value}")
    if dependency.get("clean_dependency_passed") is not True:
        raise RuntimeError("143F dependency report is not clean")
    if no_resolved.get("clean_dependency_passed") is not True:
        raise RuntimeError("143F no-resolved subset did not pass clean dependency gate")
    if manifest.get("payload_markers_static") is not True or manifest.get("per_row_manifest_switching_allowed") is not False:
        raise RuntimeError("143F no-resolved static manifest integrity failed")
    if prompt_scan.get("passed") is not True:
        raise RuntimeError("143F no-resolved prompt scan failed")
    if helper_audit.get("all_requests_allowed_keys_only") is not True or helper_audit.get("forbidden_keys_present_count") != 0:
        raise RuntimeError("143F helper request audit failed")
    return {
        "root": rel(root),
        "decision": decision,
        "metrics": metrics,
        "dependency": dependency,
        "no_resolved": no_resolved,
        "abc": abc,
        "explicit": explicit,
        "rule": rule,
        "manifest": manifest,
        "prompt_scan": prompt_scan,
        "helper_request_audit": {
            "accepted_helper_request_count": helper_audit.get("accepted_helper_request_count"),
            "all_requests_allowed_keys_only": helper_audit.get("all_requests_allowed_keys_only"),
            "forbidden_keys_present_count": helper_audit.get("forbidden_keys_present_count"),
            "raw_generate_allowed_in_runner": helper_audit.get("raw_generate_allowed_in_runner"),
            "raw_generate_allowed_in_checker": helper_audit.get("raw_generate_allowed_in_checker"),
        },
    }


def helper_selection_semantics_audit() -> dict[str, Any]:
    helper = resolve_repo_path(HELPER_PATH)
    text = helper.read_text(encoding="utf-8")
    fn = extract_function(text, "_instnct_select_open_pocket_value")
    return {
        "schema_version": "phase_143i_helper_selection_semantics_audit_v1",
        "helper_path": rel(helper),
        "helper_source_sha256": sha256_file(helper),
        "selected_function_name": "_instnct_select_open_pocket_value",
        "function_found": bool(fn),
        "open_gate_check_found": "gate_open = bool(gate_marker and gate_marker in prompt)" in fn,
        "payload_marker_loop_found": "for marker in pocket_markers" in fn,
        "first_present_marker_return_found": "return value" in fn and "open_pocket_writeback" in fn,
        "fallback_if_no_marker_found": "closed_pocket_fallback" in fn and "return fallback" in fn,
        "rule_selected_pocket_binding_found": any(token in fn for token in ["selected_pocket_id", "winner=", "arbitration_rule_id"]),
        "prompt_level_rule_parser_found": any(token in fn for token in ["quorum", "recency", "tie_break", "hierarchy", "arbitration rule"]),
        "per_row_manifest_binding_found": any(token in fn for token in ["expected_answer", "expected_values", "oracle_data", "scorer_metadata"]),
        "mechanism_summary": "open gate -> iterate configured payload markers -> return first value after first present marker -> fallback if no marker found",
    }


def failure_mode_report(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["metrics"]
    return {
        "schema_version": "phase_143i_no_resolved_failure_mode_report_v1",
        "resolved_marker_present_subset_accuracy": metrics["resolved_marker_present_subset_accuracy"],
        "no_resolved_final_marker_subset_accuracy": metrics["no_resolved_final_marker_subset_accuracy"],
        "no_resolved_final_marker_subset_fallback_rate": metrics["no_resolved_final_marker_subset_fallback_rate"],
        "no_resolved_final_marker_subset_shortcut_rate": metrics["no_resolved_final_marker_subset_shortcut_rate"],
        "no_resolved_final_marker_subset_unexpected_value_rate": metrics["no_resolved_final_marker_subset_unexpected_value_rate"],
        "no_resolved_final_marker_subset_visible_rate": metrics["no_resolved_final_marker_subset_visible_rate"],
        "no_resolved_final_marker_subset_noisy_rate": metrics["no_resolved_final_marker_subset_noisy_rate"],
        "no_resolved_final_marker_subset_train_namespace_rate": metrics["no_resolved_final_marker_subset_train_namespace_rate"],
        "resolved_marker_dependency_delta": metrics["resolved_marker_dependency_delta"],
        "failure_mode": "closed_pocket_fallback_due_to_missing_configured_marker",
        "clean_failure": True,
    }


def abc_static_report(upstream: dict[str, Any]) -> dict[str, Any]:
    abc = upstream["abc"]
    return {
        "schema_version": "phase_143i_abc_static_marker_diagnostic_report_v1",
        "diagnostic_only": True,
        "abc_static_first_pocket_rate": abc["no_resolved_abc_static_first_pocket_rate"],
        "abc_static_default_pocket_rate": abc["no_resolved_abc_static_default_pocket_rate"],
        "abc_static_last_pocket_rate": abc["no_resolved_abc_static_last_pocket_rate"],
        "abc_static_accuracy": abc["no_resolved_abc_static_marker_accuracy"],
        "payload_markers_static": abc["payload_markers_static"],
        "interpretation": "static A/B/C payload markers select the first present marker, not the rule-selected pocket",
        "first_marker_scan_behavior_confirmed": abc["no_resolved_abc_static_first_pocket_rate"] == 1.0,
    }


def explicit_vs_rule_report(upstream: dict[str, Any]) -> dict[str, Any]:
    explicit = upstream["explicit"]
    rule = upstream["rule"]
    return {
        "schema_version": "phase_143i_explicit_vs_rule_derived_winner_report_v1",
        "explicit_winner_label_subset": explicit,
        "rule_derived_winner_subset": rule,
        "explicit_winner_label_ignored_by_current_helper": explicit.get("accuracy") == 0.0 and explicit.get("fallback_rate") == 1.0,
        "rule_derived_winner_text_ignored_by_current_helper": rule.get("accuracy") == 0.0 and rule.get("fallback_rate") == 1.0,
        "interpretation": "winner labels and rule metadata are prompt text only under current helper selection semantics",
    }


def alternative_hypotheses(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["metrics"]
    manifest = upstream["manifest"]
    prompt_scan = upstream["prompt_scan"]
    rows = [
        ("hidden_final_marker_leak", "rejected", ["no_resolved_prompt_scan_report.json", "no_resolved_final_marker_subset_manifest.json"], {"prompt_scan_passed": prompt_scan.get("passed"), "final_payload_marker_removed": manifest.get("final_payload_marker_removed")}, "No-resolved prompts passed banned-marker scan and the final payload marker was removed."),
        ("per_row_manifest_oracle", "rejected", ["no_resolved_final_marker_subset_report.json"], {"no_resolved_unique_checkpoint_path_count": metrics["no_resolved_unique_checkpoint_path_count"], "no_resolved_per_row_manifest_switch_rate": metrics["no_resolved_per_row_manifest_switch_rate"], "no_resolved_per_row_payload_marker_switch_rate": metrics["no_resolved_per_row_payload_marker_switch_rate"]}, "The subset used one checkpoint/hash and a static payload marker list."),
        ("first_pocket_shortcut", "rejected", ["no_resolved_final_marker_shortcut_report.json"], {"first_pocket_rate": metrics["no_resolved_final_marker_subset_shortcut_rate"]}, "FINAL_MARKERS-static no-resolved subset did not select pocket A."),
        ("default_pocket_shortcut", "rejected", ["no_resolved_final_marker_shortcut_report.json"], {"default_pocket_rate": metrics["no_resolved_final_marker_subset_shortcut_rate"]}, "FINAL_MARKERS-static no-resolved subset did not select default pocket values."),
        ("stale_visible_noisy_shortcut", "rejected", ["no_resolved_final_marker_subset_report.json"], {"visible_rate": metrics["no_resolved_final_marker_subset_visible_rate"], "noisy_rate": metrics["no_resolved_final_marker_subset_noisy_rate"]}, "Visible, noisy, and stale shortcut channels stayed at zero."),
        ("random_unexpected_value", "rejected", ["no_resolved_final_marker_subset_report.json"], {"unexpected_value_rate": metrics["no_resolved_final_marker_subset_unexpected_value_rate"], "train_namespace_rate": metrics["no_resolved_final_marker_subset_train_namespace_rate"]}, "The failure was fallback, not random or train-namespace output."),
        ("closed_pocket_fallback_due_to_missing_configured_marker", "supported", ["no_resolved_final_marker_subset_report.json", "helper_selection_semantics_audit.json"], {"fallback_rate": metrics["no_resolved_final_marker_subset_fallback_rate"], "accuracy": metrics["no_resolved_final_marker_subset_accuracy"]}, "Static FINAL_MARKERS were configured but no final marker was present, so helper fell back."),
        ("abc_static_first_marker_scan", "supported", ["no_resolved_abc_static_marker_control_report.json"], {"first_pocket_rate": metrics["no_resolved_abc_static_first_pocket_rate"], "abc_static_accuracy": metrics["no_resolved_abc_static_marker_accuracy"]}, "When static A/B/C markers are configured, the first prompt marker is selected."),
        ("explicit_winner_label_ignored_by_current_helper", "supported", ["no_resolved_explicit_winner_label_subset_report.json"], {"explicit_winner_label_subset_accuracy": upstream["explicit"].get("accuracy"), "explicit_winner_label_subset_fallback_rate": upstream["explicit"].get("fallback_rate")}, "Explicit winner text did not alter marker selection."),
        ("rule_derived_winner_text_ignored_by_current_helper", "supported", ["no_resolved_rule_derived_winner_subset_report.json"], {"rule_derived_winner_subset_accuracy": upstream["rule"].get("accuracy"), "rule_derived_winner_subset_fallback_rate": upstream["rule"].get("fallback_rate")}, "Rule metadata text did not alter marker selection."),
    ]
    return {
        "schema_version": "phase_143i_alternative_hypothesis_matrix_v1",
        "hypotheses": [
            {
                "hypothesis_id": hypothesis_id,
                "status": status,
                "evidence_artifacts": artifacts,
                "metrics": hypothesis_metrics,
                "explanation": explanation,
            }
            for hypothesis_id, status, artifacts, hypothesis_metrics, explanation in rows
        ],
    }


def bridge_options_matrix() -> dict[str, Any]:
    options = [
        {
            "option_id": "manifest_level_selected_pocket_binding_primitive",
            "mechanism": "static mapping from parsed selected pocket id to static pocket marker",
            "expected_benefit": "smallest bridge from winner text to corresponding pocket value extraction",
            "oracle_risk": "medium if selected pocket is supplied per-row outside prompt",
            "shortcut_risk": "medium; must prevent marker list narrowing to the correct pocket",
            "helper_change_required": True,
            "request_key_change_required": False,
            "required_controls": ["per_row_manifest_switching_forbidden", "all_pocket_markers_static", "first_default_last_controls", "closed_pocket_ablation"],
            "recommendation": "recommended_for_143J_design",
        },
        {
            "option_id": "prompt_level_explicit_winner_label_parser",
            "mechanism": "parse winner=pocket_* from prompt, map to static marker, extract value",
            "expected_benefit": "directly tests selected-pocket binding without full rule parser scope",
            "oracle_risk": "medium; winner label is explicit but remains prompt-visible rather than request metadata",
            "shortcut_risk": "medium; requires same-template opposite-winner and label corruption controls",
            "helper_change_required": True,
            "request_key_change_required": False,
            "required_controls": ["winner_label_corruption_control", "same_values_different_winner_control", "abc_static_first_marker_control"],
            "recommendation": "recommended_first_prototype_axis",
        },
        {
            "option_id": "rule_metadata_parser",
            "mechanism": "derive selected pocket from quorum/recency/tie-break/hierarchy metadata",
            "expected_benefit": "closer to rule-bound arbitration",
            "oracle_risk": "low if derived from prompt only",
            "shortcut_risk": "high; larger parser surface and more shortcut controls required",
            "helper_change_required": True,
            "request_key_change_required": False,
            "required_controls": ["rule_inversion", "same_values_different_rule", "same_rule_different_values", "priority_hierarchy_conflict"],
            "recommendation": "defer_until_explicit_winner_binding_is_clean",
        },
        {
            "option_id": "keep_resolved_final_marker_and_stop_bridge_expansion",
            "mechanism": "continue using resolved final payload markers only",
            "expected_benefit": "keeps current scale-confirmed fixture stable",
            "oracle_risk": "high for bridge claims",
            "shortcut_risk": "low for fixture, high for overclaim",
            "helper_change_required": False,
            "request_key_change_required": False,
            "required_controls": ["resolved_final_marker_echo_control"],
            "recommendation": "not_recommended_if_goal_is_no_resolved_bridge",
        },
    ]
    return {"schema_version": "phase_143i_bridge_options_matrix_v1", "options": options}


def target_143j_plan(options: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_143i_target_143j_milestone_plan_v1",
        "milestone": "143J_RULE_SELECTED_POCKET_PAYLOAD_BINDING_HELPER_PRIMITIVE_PLAN",
        "planning_only": True,
        "recommended_decision": "rule_selected_pocket_payload_binding_primitive_plan_recommended",
        "recommended_next": "143K_RULE_SELECTED_POCKET_PAYLOAD_BINDING_PROTOTYPE_PROBE",
        "minimum_desired_primitive": "rule-selected pocket id -> static pocket marker -> value extraction",
        "options_to_compare": [option["option_id"] for option in options["options"]],
        "must_forbid": [
            "per-row oracle selected_pocket_id in helper request metadata",
            "per-row manifest switching",
            "payload marker list set to only the correct pocket",
            "hidden final/winner/gold/answer marker equivalents",
            "helper request key changes",
            "broad architecture claims",
        ],
        "required_controls_for_143k": [
            "closed-pocket ablation",
            "first/default/last/stale shortcut controls",
            "visible/noisy distractor controls",
            "winner-label corruption control",
            "same-template opposite-winner controls",
            "static manifest integrity audit",
        ],
    }


def root_cause_report(helper_audit: dict[str, Any], hypotheses: dict[str, Any], abc: dict[str, Any]) -> dict[str, Any]:
    statuses = {row["hypothesis_id"]: row["status"] for row in hypotheses["hypotheses"]}
    return {
        "schema_version": "phase_143i_root_cause_report_v1",
        "root_cause_id": ROOT_CAUSE_ID,
        "root_cause_summary": "current constrained helper/backend path lacks rule-selected pocket payload binding",
        "supported_by_143f_clean_dependency": statuses.get("closed_pocket_fallback_due_to_missing_configured_marker") == "supported",
        "supported_by_helper_source_audit": helper_audit["payload_marker_loop_found"] and helper_audit["fallback_if_no_marker_found"] and not helper_audit["rule_selected_pocket_binding_found"],
        "hidden_marker_leak_rejected": statuses.get("hidden_final_marker_leak") == "rejected",
        "per_row_manifest_oracle_rejected": statuses.get("per_row_manifest_oracle") == "rejected",
        "shortcut_failure_rejected": all(statuses.get(key) == "rejected" for key in ["first_pocket_shortcut", "default_pocket_shortcut", "stale_visible_noisy_shortcut", "random_unexpected_value"]),
        "abc_static_first_marker_behavior_confirmed": abc["first_marker_scan_behavior_confirmed"],
        "architecture_failure_claimed": False,
    }


def write_report(out: Path, decision: dict[str, Any], root: dict[str, Any], failure: dict[str, Any], abc: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Root cause: `{root['root_cause_id']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Evidence Chain

- Resolved-marker-present accuracy: `{failure['resolved_marker_present_subset_accuracy']}`
- No-resolved accuracy: `{failure['no_resolved_final_marker_subset_accuracy']}`
- No-resolved fallback rate: `{failure['no_resolved_final_marker_subset_fallback_rate']}`
- No-resolved shortcut rate: `{failure['no_resolved_final_marker_subset_shortcut_rate']}`
- ABC static first-pocket rate: `{abc['abc_static_first_pocket_rate']}`

## Interpretation

The current helper opens the pocket gate, scans configured payload markers in
order, returns the first value after the first present marker, and falls back
when no configured marker is present. Therefore the no-resolved FINAL_MARKERS
subset fails cleanly because no configured final marker exists in the prompt.

This is a helper-semantics bottleneck, not an architecture failure. The next
planning milestone should design a rule-selected pocket payload binding
primitive without introducing per-row oracle metadata or marker-list shortcuts.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-143f-root", type=Path, default=DEFAULT_143F_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_143i_queue_v1", "milestone": MILESTONE, "status": "running"})

    config = {
        "schema_version": "phase_143i_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "artifact_only": True,
        "training_performed": False,
        "new_helper_generation_run": False,
        "shared_helper_imported": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "helper_backend_modified": False,
        "request_key_change_allowed": False,
        "runtime_surface_mutated": False,
        "product_surface_mutated": False,
        "release_surface_mutated": False,
        **FALSE_FLAGS,
    }
    write_json(out / "analysis_config.json", config)

    upstream = require_143f(resolve_repo_path(args.upstream_143f_root))
    write_json(out / "upstream_143f_manifest.json", upstream)
    append_progress(out, "upstream verification", decision=upstream["decision"]["decision"])

    helper_audit = helper_selection_semantics_audit()
    failure = failure_mode_report(upstream)
    abc = abc_static_report(upstream)
    explicit_rule = explicit_vs_rule_report(upstream)
    static_manifest = {
        "schema_version": "phase_143i_static_manifest_integrity_report_v1",
        "payload_markers_static": upstream["manifest"].get("payload_markers_static"),
        "per_row_manifest_switching_allowed": upstream["manifest"].get("per_row_manifest_switching_allowed"),
        "per_row_payload_marker_switching_allowed": upstream["manifest"].get("per_row_payload_marker_switching_allowed"),
        "no_resolved_unique_checkpoint_path_count": upstream["metrics"].get("no_resolved_unique_checkpoint_path_count"),
        "no_resolved_unique_checkpoint_hash_count": upstream["metrics"].get("no_resolved_unique_checkpoint_hash_count"),
        "no_resolved_per_row_manifest_switch_rate": upstream["metrics"].get("no_resolved_per_row_manifest_switch_rate"),
        "no_resolved_per_row_payload_marker_switch_rate": upstream["metrics"].get("no_resolved_per_row_payload_marker_switch_rate"),
        "passed": True,
    }
    prompt_scan = {
        "schema_version": "phase_143i_prompt_scan_integrity_report_v1",
        **upstream["prompt_scan"],
    }
    hypotheses = alternative_hypotheses(upstream)
    options = bridge_options_matrix()
    target_plan = target_143j_plan(options)
    root = root_cause_report(helper_audit, hypotheses, abc)
    risk_register = {
        "schema_version": "phase_143i_risk_register_v1",
        "risks": [
            {"risk": "selected pocket becomes oracle metadata", "mitigation": "forbid request metadata and per-row manifest switching"},
            {"risk": "payload marker list narrowed to correct pocket", "mitigation": "require static all-pocket marker list"},
            {"risk": "resolved final marker reintroduced under another name", "mitigation": "regex prompt scan and corrupted marker controls"},
            {"risk": "claim overextended to architecture reasoning", "mitigation": "boundary flags and report wording"},
        ],
    }
    decision = {
        "schema_version": "phase_143i_decision_v1",
        "decision": "no_resolved_final_marker_bridge_failure_analysis_complete",
        "root_cause_id": ROOT_CAUSE_ID,
        "next": NEXT,
        "clean_negative_valid": True,
        "artifact_only": True,
        "architecture_failure_claimed": False,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }
    summary = {
        "schema_version": "phase_143i_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "upstream_143f": upstream,
        "helper_selection_semantics_audit": helper_audit,
        "no_resolved_failure_mode_report": failure,
        "abc_static_marker_diagnostic_report": abc,
        "root_cause_report": root,
        "target_143j_milestone_plan": target_plan,
        **decision,
    }

    write_json(out / "helper_selection_semantics_audit.json", helper_audit)
    write_json(out / "no_resolved_failure_mode_report.json", failure)
    write_json(out / "abc_static_marker_diagnostic_report.json", abc)
    write_json(out / "explicit_vs_rule_derived_winner_report.json", explicit_rule)
    write_json(out / "static_manifest_integrity_report.json", static_manifest)
    write_json(out / "prompt_scan_integrity_report.json", prompt_scan)
    write_json(out / "alternative_hypothesis_matrix.json", hypotheses)
    write_json(out / "root_cause_report.json", root)
    write_json(out / "bridge_options_matrix.json", options)
    write_json(out / "risk_register.json", risk_register)
    write_json(out / "target_143j_milestone_plan.json", target_plan)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, root, failure, abc)
    append_progress(out, "analysis complete", root_cause=ROOT_CAUSE_ID, next=NEXT)
    append_progress(out, "final verdict", decision=decision["decision"])
    write_json(out / "queue.json", {"schema_version": "phase_143i_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
