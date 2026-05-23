#!/usr/bin/env python3
"""141Z planning-only next decision after hardened multi-field scale confirm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_141Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_NEXT_DECISION_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_141z_instnct_pocket_gated_multi_field_transfer_next_decision_plan/smoke")
DEFAULT_141F_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_141f_instnct_pocket_gated_multi_field_transfer_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_141z_instnct_pocket_gated_multi_field_transfer_next_decision_plan_check.py"
POSITIVE_NEXT = "142A_INSTNCT_POCKET_GATED_CONFLICT_PRIORITY_TRANSFER_PROBE"
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
    "141Z is planning-only and artifact-only after the hardened positive 141F "
    "scale confirm. It does not run helper generation, train, mutate checkpoints, "
    "modify shared_raw_generation_helper.py, modify helper/backend/request-key/"
    "runtime/release/product surfaces, deploy, change root LICENSE, or claim "
    "GPT-like readiness, open-domain reasoning, broad assistant capability, "
    "production/public API/deployment/safety readiness, or architecture superiority."
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
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("--out must stay inside repo") from exc
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


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def scan_ast() -> dict[str, Any]:
    failures: list[str] = []
    for path in [RUNNER_PATH, CHECKER_PATH]:
        if not path.exists():
            failures.append(f"missing:{rel(path)}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("run_stable_loop_phase_lock_"):
                failures.append(f"old_runner_import:{rel(path)}:{node.module}")
            if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names):
                failures.append(f"torch_import:{rel(path)}")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "backward", "step", "raw_generate"}:
                    failures.append(f"forbidden_call:{rel(path)}:{name}")
    return {"schema_version": "phase_141z_ast_scan_v1", "passed": not failures, "failures": failures}


def require_141f(root: Path) -> tuple[dict[str, Any], list[str]]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "helper_request_audit.json",
        "canonical_metric_alias_report.json",
        "per_seed_gate_report.json",
        "per_family_gate_report.json",
        "summary.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return {"root": rel(root), "missing_artifacts": missing}, [f"missing:{name}" for name in missing]

    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    request_audit = read_json(root / "helper_request_audit.json")
    alias_report = read_json(root / "canonical_metric_alias_report.json")
    seed_report = read_json(root / "per_seed_gate_report.json")
    family_report = read_json(root / "per_family_gate_report.json")
    summary = read_json(root / "summary.json")

    checks = {
        "decision": decision.get("decision") == "instnct_pocket_gated_multi_field_transfer_scale_confirmed",
        "next": decision.get("next") == "141Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_NEXT_DECISION_PLAN",
        "eval_row_count": metrics.get("eval_row_count") == 2880,
        "family_count": metrics.get("family_count") == 6,
        "scaffold_variant_count": metrics.get("scaffold_variant_count") == 72,
        "main_final_answer_accuracy": metrics.get("main_final_answer_accuracy") == 1.0,
        "main_multi_field_binding_accuracy": metrics.get("main_multi_field_binding_accuracy") == 1.0,
        "main_pocket_writeback_rate": metrics.get("main_pocket_writeback_rate") == 1.0,
        "main_contrast_group_accuracy": metrics.get("main_contrast_group_accuracy") == 1.0,
        "ablation_final_answer_accuracy": metrics.get("ablation_final_answer_accuracy") == 0.0,
        "pocket_ablation_delta_final_answer_accuracy": metrics.get("pocket_ablation_delta_final_answer_accuracy") == 1.0,
        "single_field_shortcut_rate": metrics.get("single_field_shortcut_rate") == 0.0,
        "field_a_shortcut_rate": metrics.get("field_a_shortcut_rate") == 0.0,
        "field_b_shortcut_rate": metrics.get("field_b_shortcut_rate") == 0.0,
        "intermediate_copy_shortcut_rate": metrics.get("intermediate_copy_shortcut_rate") == 0.0,
        "priority_conflict_wrong_field_rate": metrics.get("priority_conflict_wrong_field_rate") == 0.0,
        "visible_bypass_violation_rate": metrics.get("visible_bypass_violation_rate") == 0.0,
        "noisy_distractor_violation_rate": metrics.get("noisy_distractor_violation_rate") == 0.0,
        "direct_pocket_value_marker_rate": metrics.get("direct_pocket_value_marker_rate") == 0.0,
        "deterministic_replay_passed": metrics.get("deterministic_replay_passed") is True,
        "infrastructure_gates": all(metrics.get("infrastructure_gates", {}).values()),
        "request_allowed_keys": request_audit.get("all_requests_allowed_keys_only") is True,
        "request_forbidden_key_count": request_audit.get("forbidden_keys_present_count") == 0,
        "request_no_forbidden_metadata": request_audit.get("no_forbidden_keys_in_accepted_generation_requests") is True,
        "request_runner_generation_allowed": request_audit.get("raw_generate_allowed_in_runner") is True,
        "request_checker_generation_forbidden": request_audit.get("raw_generate_allowed_in_checker") is False,
        "seed_gate_report": seed_report.get("passed") is True,
        "family_gate_report": family_report.get("passed") is True,
        "decision_seed_failures": decision.get("per_seed_gate_failures") == [],
        "decision_family_failures": decision.get("per_family_gate_failures") == [],
        "summary_complete": summary.get("status") == "complete",
    }
    required_aliases = {
        "direct_POCKET_VALUE_rate": "direct_pocket_value_marker_rate",
        "pocket_ablation_delta": "pocket_ablation_delta_final_answer_accuracy",
        "main_final_accuracy": "main_final_answer_accuracy",
        "main_binding_accuracy": "main_multi_field_binding_accuracy",
    }
    aliases = alias_report.get("aliases_normalized", {})
    for alias, canonical in required_aliases.items():
        checks[f"alias:{alias}"] = aliases.get(alias) == canonical

    failed = [key for key, passed in checks.items() if not passed]
    manifest = {
        "schema_version": "phase_141z_upstream_141f_manifest_v1",
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "metrics": {key: metrics.get(key) for key in [
            "eval_row_count",
            "family_count",
            "scaffold_variant_count",
            "main_final_answer_accuracy",
            "main_multi_field_binding_accuracy",
            "main_pocket_writeback_rate",
            "main_contrast_group_accuracy",
            "ablation_final_answer_accuracy",
            "pocket_ablation_delta_final_answer_accuracy",
            "single_field_shortcut_rate",
            "field_a_shortcut_rate",
            "field_b_shortcut_rate",
            "intermediate_copy_shortcut_rate",
            "priority_conflict_wrong_field_rate",
            "visible_bypass_violation_rate",
            "noisy_distractor_violation_rate",
            "direct_pocket_value_marker_rate",
            "deterministic_replay_passed",
        ]},
        "helper_request_audit": {
            "all_requests_allowed_keys_only": request_audit.get("all_requests_allowed_keys_only"),
            "forbidden_keys_present_count": request_audit.get("forbidden_keys_present_count"),
            "no_forbidden_keys_in_accepted_generation_requests": request_audit.get("no_forbidden_keys_in_accepted_generation_requests"),
            "raw_generate_allowed_in_runner": request_audit.get("raw_generate_allowed_in_runner"),
            "raw_generate_allowed_in_checker": request_audit.get("raw_generate_allowed_in_checker"),
            "accepted_helper_request_count": request_audit.get("accepted_helper_request_count"),
        },
        "canonical_metric_alias_report": alias_report,
        "per_seed_gate_passed": seed_report.get("passed"),
        "per_family_gate_passed": family_report.get("passed"),
        "gate_checks": checks,
        "failed_gate_checks": failed,
    }
    return manifest, failed


def evidence_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_141z_evidence_chain_summary_v1",
        "current_state": "hardened multi-field transfer scale confirmed",
        "upstream_141f": upstream,
        "interpretation": [
            "Multi-field final selection survived 2880 rows, 6 families, and 72 scaffold variants.",
            "Closed-pocket ablation remained decision-critical.",
            "Field A, field B, intermediate, priority-wrong, visible, noisy, and direct marker shortcuts stayed at zero.",
            "Helper request audit showed allowed keys only and no expected/scorer/oracle metadata.",
            "The next useful falsification is explicit conflict and priority inversion, not more multi-field scale.",
        ],
        "not_claimed": ["open-ended reasoning", "general composition", "GPT-like readiness", "architecture superiority"],
    }


def gap_analysis() -> dict[str, Any]:
    return {
        "schema_version": "phase_141z_multi_field_to_conflict_priority_gap_analysis_v1",
        "closed_gaps": [
            "multi-field final selection smoke",
            "multi-field final selection scale confirm",
            "helper request audit and alias hardening",
            "single-field shortcut rejection",
            "closed-pocket causal ablation",
        ],
        "remaining_gaps": [
            "no dedicated priority inversion stress",
            "no balanced A-wins/B-wins/table-override/rule-override suite",
            "no priority-default shortcut measurement",
            "no same-template opposite-winner contrast gate",
        ],
        "next_gap_to_test": "conflict and priority transfer with inversion pairs",
    }


def decision_matrix() -> dict[str, Any]:
    return {
        "schema_version": "phase_141z_next_decision_matrix_v1",
        "selected_option": "conflict_priority_transfer",
        "recommended_next": POSITIVE_NEXT,
        "options": [
            {"option": "more_multi_field_scale", "recommendation": "defer", "reason": "141F already scale-confirmed multi-field transfer with hardened audits."},
            {"option": "conflict_priority_transfer", "recommendation": "select", "reason": "Priority inversion is the next untested failure mode after stable multi-field selection."},
            {"option": "multi_pocket_arbitration", "recommendation": "later", "reason": "Useful after explicit priority conflict is measured."},
            {"option": "real_task_mixed_suite", "recommendation": "later", "reason": "Premature before conflict/priority shortcut routes are characterized."},
            {"option": "blocker_analysis", "recommendation": "not_selected", "reason": "No hardened 141F blocker is present."},
        ],
    }


def conflict_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_141z_conflict_priority_transfer_requirements_v1",
        "milestone": POSITIVE_NEXT,
        "helper_only_final_eval": True,
        "training_allowed": False,
        "must_not_be_always_b_wins": True,
        "required_row_types": [
            "A wins rows",
            "B wins rows",
            "table override wins rows",
            "rule override wins rows",
            "visible value loses rows",
            "noisy distractor loses rows",
            "same-template different-priority contrast rows",
            "priority inversion pairs",
        ],
        "families": [
            "TWO_FIELD_PRIORITY_RULE",
            "DUAL_POCKET_CONFLICT",
            "TABLE_RULE_PRIORITY_OVERRIDE",
            "VISIBLE_VALUE_LOSES_TO_PRIORITY",
            "NOISY_DISTRACTOR_PRIORITY_TRAP",
            "SAME_TEMPLATE_DIFFERENT_PRIORITY_CONTRAST",
        ],
        "required_metrics": [
            "priority_rule_accuracy",
            "conflict_resolution_accuracy",
            "wrong_priority_field_rate",
            "priority_default_shortcut_rate",
            "priority_inversion_accuracy",
            "same_template_opposite_winner_accuracy",
        ],
        "positive_gates": {
            "main_final_answer_accuracy_min": 0.70,
            "priority_rule_accuracy_min": 0.70,
            "conflict_resolution_accuracy_min": 0.70,
            "priority_inversion_accuracy_min": 0.65,
            "same_template_opposite_winner_accuracy_min": 0.65,
            "main_pocket_writeback_rate_min": 0.80,
            "main_contrast_group_accuracy_min": 0.70,
            "ablation_final_answer_accuracy_max": 0.15,
            "pocket_ablation_delta_final_answer_accuracy_min": 0.50,
            "wrong_priority_field_rate": 0.0,
            "priority_default_shortcut_rate": 0.0,
            "single_field_shortcut_rate": 0.0,
            "visible_bypass_violation_rate": 0.0,
            "noisy_distractor_violation_rate": 0.0,
            "direct_pocket_value_marker_rate": 0.0,
            "deterministic_replay_passed": True,
        },
    }


def anti_shortcuts() -> dict[str, Any]:
    return {
        "schema_version": "phase_141z_anti_shortcut_requirements_v1",
        "reject": [
            "always B wins",
            "always A wins",
            "always table override wins",
            "always rule override wins",
            "single-field shortcut success",
            "visible value bypass",
            "noisy distractor copy",
            "closed-pocket correct answer",
            "wrong-priority field selection",
            "priority-default shortcut",
            "same-template priority inversion failure",
            "prefix-only success",
            "post-generation repair",
            "expected/scorer/oracle metadata in helper requests",
        ],
    }


def target_142a_plan(reqs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_141z_target_142a_milestone_plan_v1",
        "milestone": POSITIVE_NEXT,
        "type": "executable helper-only conflict/priority probe",
        "helper_only_final_eval": True,
        "training_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "helper_backend_modification_allowed": False,
        "public_request_key_change_allowed": False,
        "selected_task": "conflict_priority_transfer",
        "row_design": reqs["required_row_types"],
        "families": reqs["families"],
        "required_metrics": reqs["required_metrics"],
        "positive_gates": reqs["positive_gates"],
        "required_artifacts": [
            "priority_rule_manifest.json",
            "conflict_pair_manifest.json",
            "priority_inversion_report.json",
            "wrong_priority_field_report.json",
            "priority_control_report.json",
            "helper_request_audit.json",
            "canonical_metric_alias_report.json",
            "per_seed_gate_report.json",
            "per_family_gate_report.json",
        ],
        "required_controls": [
            "A_ONLY_CONTROL",
            "B_ONLY_CONTROL",
            "TABLE_DEFAULT_CONTROL",
            "RULE_DEFAULT_CONTROL",
            "VISIBLE_VALUE_CONTROL",
            "NOISY_DISTRACTOR_CONTROL",
            "CLOSED_POCKET_ABLATION_CONTROL",
            "PRIORITY_DEFAULT_SHORTCUT_CONTROL",
            "SAME_TEMPLATE_PRIORITY_INVERSION_CONTROL",
            "PREFIX_ONLY_CONTROL",
        ],
        "failure_routes": {
            "wrong_priority_field_selected": "142B_PRIORITY_SELECTION_FAILURE_ANALYSIS",
            "single_field_shortcut_detected": "141B_SINGLE_FIELD_SHORTCUT_ANALYSIS",
            "conflict_resolution_failure": "142C_CONFLICT_RESOLUTION_FAILURE_ANALYSIS",
            "priority_default_shortcut_detected": "142D_PRIORITY_DEFAULT_SHORTCUT_ANALYSIS",
            "priority_inversion_failure": "142E_PRIORITY_INVERSION_FAILURE_ANALYSIS",
            "pocket_ablation_not_decision_critical": "141D_POCKET_CAUSALITY_FAILURE_ANALYSIS",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
        "boundary": "constrained helper/backend evidence only; not open-ended reasoning or general composition",
    }


def risks() -> dict[str, Any]:
    return {
        "schema_version": "phase_141z_risk_register_v1",
        "risks": [
            {"risk": "priority task degenerates into always-B shortcut", "mitigation": "require A-wins, B-wins, table override, rule override, and priority inversion pairs"},
            {"risk": "same-template contrast still exposes final marker directly", "mitigation": "track priority inversion and same-template opposite-winner accuracy separately"},
            {"risk": "broad capability overclaim", "mitigation": "keep all readiness flags false and boundary text explicit"},
        ],
    }


def write_report(out: Path, decision: dict[str, Any], matrix: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Selected option: `{matrix['selected_option']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

This is constrained helper/backend evidence only, not open-ended reasoning, not
general composition, not GPT-like readiness, not open-domain reasoning, not broad
assistant capability, not production/public API/deployment/safety readiness, and
not architecture superiority.

Boundary phrases: not general composition; not broad assistant capability.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-141f-root", type=Path, default=DEFAULT_141F_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_141z_queue_v1", "milestone": MILESTONE, "status": "running"})

    config = {
        "schema_version": "phase_141z_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "planning_only": True,
        "artifact_only": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "helper_generation_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "helper_modified": False,
        "backend_modified": False,
        "public_request_key_change": False,
        "runtime_surface_mutated": False,
        "release_surface_mutated": False,
        "product_surface_mutated": False,
        "root_license_changed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "analysis_config.json", config)

    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)
    upstream, upstream_failures = require_141f(resolve_repo_path(args.upstream_141f_root))
    write_json(out / "upstream_141f_manifest.json", upstream)
    append_progress(out, "upstream verification", failures=upstream_failures)

    evidence = evidence_summary(upstream)
    gaps = gap_analysis()
    matrix = decision_matrix()
    reqs = conflict_requirements()
    shortcuts = anti_shortcuts()
    target = target_142a_plan(reqs)
    risk_register = risks()
    for name, payload in [
        ("evidence_chain_summary.json", evidence),
        ("multi_field_to_conflict_priority_gap_analysis.json", gaps),
        ("next_decision_matrix.json", matrix),
        ("conflict_priority_transfer_requirements.json", reqs),
        ("anti_shortcut_requirements.json", shortcuts),
        ("target_142a_milestone_plan.json", target),
        ("risk_register.json", risk_register),
    ]:
        write_json(out / name, payload)
        append_progress(out, f"wrote {name}", hash=stable_hash(payload))

    if upstream_failures or ast_report["passed"] is not True:
        decision_value = "upstream_141f_evidence_incomplete"
        next_step = "141Z_UPSTREAM_141F_EVIDENCE_REPAIR"
    else:
        decision_value = "conflict_priority_transfer_probe_recommended"
        next_step = POSITIVE_NEXT
    decision = {
        "schema_version": "phase_141z_decision_v1",
        "decision": decision_value,
        "selected_option": matrix["selected_option"] if decision_value == "conflict_priority_transfer_probe_recommended" else "blocker_analysis",
        "recommended_next": next_step,
        "next": next_step,
        "planning_only": True,
        "artifact_only": True,
        "capability_claimed": False,
        "clean_negative_valid": True,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    summary = {
        "schema_version": "phase_141z_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "upstream_141f": upstream,
        "upstream_failures": upstream_failures,
        "decision_matrix": matrix,
        "target_142a": target,
        **decision,
    }
    write_json(out / "summary.json", summary)
    write_report(out, decision, matrix)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    append_progress(out, "final verdict", status="complete")
    write_json(out / "queue.json", {"schema_version": "phase_141z_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
