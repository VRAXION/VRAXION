#!/usr/bin/env python3
"""143Z planning-only next decision after selected-pocket binding scale confirm."""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_143z_rule_selected_pocket_binding_next_decision_plan/smoke")
DEFAULT_143W_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_143w_selected_marker_occurrence_count_rejection_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_143z_rule_selected_pocket_binding_next_decision_plan_check.py"
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
DECISION = "rule_metadata_to_selected_pocket_binding_plan_recommended"
SELECTED_OPTION = "structured_rule_metadata_to_selected_pocket_binding_plan"
NEXT = "144A_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PLAN"
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
    "143Z is planning-only and artifact-only after the positive 143W selected-pocket "
    "binding scale confirm. It is constrained helper/backend evidence only and "
    "prompt-visible selected-pocket binding only; it is not rule metadata reasoning yet, "
    "not natural-language rule reasoning, not open-ended arbitration, "
    "not GPT-like/open-domain/broad assistant capability, "
    "not production/public API/deployment/safety readiness, and not architecture superiority."
)
EXPECTED_143W_METRICS = {
    "main_eval_rows": 3072,
    "single_selected_marker_binding_accuracy": 1.0,
    "positive_binding_subset_writeback_rate": 1.0,
    "duplicate_selected_marker_conflict_rejection_rate": 1.0,
    "duplicate_selected_marker_same_value_rejection_rate": 1.0,
    "duplicate_non_selected_marker_conflict_binding_accuracy": 1.0,
    "selected_marker_invalid_value_fallback_rate": 1.0,
    "selected_marker_multi_value_same_line_fallback_rate": 1.0,
    "selected_marker_prose_plus_one_valid_line_accuracy": 1.0,
    "following_line_value_leak_rate": 0.0,
    "legacy_manifest_regression_passed": True,
    "shared_helper_no_change_since_143v": True,
    "deterministic_replay_passed": True,
}


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
    return {"schema_version": "phase_143z_ast_scan_v1", "passed": not failures, "failures": failures}


def require_143w(root: Path) -> tuple[dict[str, Any], list[str]]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "summary.json",
        "shared_helper_no_change_audit.json",
        "helper_repair_semantics_audit.json",
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
    helper_semantics = read_json(root / "helper_repair_semantics_audit.json")
    seed = read_json(root / "per_seed_gate_report.json")
    family = read_json(root / "per_family_gate_report.json")
    static = read_json(root / "static_manifest_integrity_report.json")
    request = read_json(root / "helper_request_audit.json")
    checks = {
        "decision": decision.get("decision") == "selected_marker_occurrence_count_rejection_scale_confirmed",
        "verdict": decision.get("verdict") == "INSTNCT_SELECTED_MARKER_OCCURRENCE_COUNT_REJECTION_SCALE_CONFIRMED",
        "next": decision.get("next") == "143Z_RULE_SELECTED_POCKET_BINDING_NEXT_DECISION_PLAN",
        "summary_boundary_present": "prompt-visible selected-pocket binding only" in json.dumps(summary),
        "helper_no_change": helper_no_change.get("shared_helper_no_change_since_143v") is True,
        "helper_not_modified_by_143w": helper_no_change.get("shared_helper_modified_by_143w") is False,
        "helper_semantics_passed": helper_semantics.get("passed") is True,
        "per_seed_gate_passed": seed.get("passed") is True,
        "per_family_gate_passed": family.get("passed") is True,
        "static_manifest_passed": static.get("passed") is True,
        "request_allowed_keys": request.get("all_requests_allowed_keys_only") is True,
        "request_forbidden_metadata_count": request.get("helper_request_forbidden_metadata_count") == 0,
    }
    for key, expected in EXPECTED_143W_METRICS.items():
        checks[f"metric:{key}"] = metrics.get(key) == expected
    failed = [key for key, passed in checks.items() if not passed]
    manifest = {
        "schema_version": "phase_143z_upstream_143w_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": {key: metrics.get(key) for key in EXPECTED_143W_METRICS},
        "shared_helper_no_change_audit": helper_no_change,
        "helper_repair_semantics_audit": helper_semantics,
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
        "schema_version": "phase_143z_evidence_chain_summary_v1",
        "current_state": "selected-pocket binding and selected-marker occurrence rejection scale confirmed",
        "upstream_143w": upstream,
        "interpretation": [
            "143K established prompt-visible winner labels can bind to static pocket markers.",
            "143V repaired duplicate selected-marker ambiguity with line-level candidate parsing.",
            "143W scale-confirmed that repair over 3072 main rows with no helper change since 143V.",
            "The confirmed layer is selected pocket identity to static marker to same-line value extraction.",
            "The untested layer is constrained structured rule metadata to selected pocket identity.",
        ],
    }


def selected_pocket_binding_state_report() -> dict[str, Any]:
    return {
        "schema_version": "phase_143z_selected_pocket_binding_state_report_v1",
        "confirmed_layer": [
            "prompt-visible winner=pocket_* -> static marker -> value extraction",
            "selected marker occurrence-count rejection",
            "duplicate selected marker conflict rejection",
            "non-selected duplicate marker scope preservation",
        ],
        "selected_pocket_binding_scale_confirmed": True,
        "selected_marker_occurrence_rejection_scale_confirmed": True,
        "remaining_gap": "constrained structured rule metadata -> derived selected pocket id",
        "rule_metadata_reasoning_claimed": False,
    }


def rule_metadata_bridge_gap_analysis() -> dict[str, Any]:
    return {
        "schema_version": "phase_143z_rule_metadata_bridge_gap_analysis_v1",
        "selected_pocket_binding_scale_confirmed": True,
        "selected_marker_occurrence_rejection_scale_confirmed": True,
        "rule_metadata_to_selected_pocket_identity_untested": True,
        "natural_language_rule_reasoning_untested": True,
        "open_ended_arbitration_claimed": False,
        "next_gap": "structured rule metadata -> selected pocket identity before static marker binding",
        "excluded_next_step": "free-form natural-language rule parsing",
    }


def next_decision_matrix() -> dict[str, Any]:
    options = [
        {
            "option_id": "structured_rule_metadata_to_selected_pocket_binding_plan",
            "mechanism": "Plan constrained rule metadata grammars that derive selected pocket identity before existing marker binding.",
            "diagnostic_value": "high",
            "oracle_risk": "medium",
            "scope_cost": "medium",
            "recommendation": "recommended",
            "reason": "It targets the only untested bridge layer after 143W.",
        },
        {
            "option_id": "binding_robustness_extension_plan",
            "mechanism": "Add more whitespace, casing, marker, and value parser robustness around the already-confirmed binding layer.",
            "diagnostic_value": "medium",
            "oracle_risk": "low",
            "scope_cost": "low",
            "recommendation": "defer",
            "reason": "143W already closed the immediate duplicate-marker and parser edge gaps.",
        },
        {
            "option_id": "integrate_selected_pocket_binding_into_broader_arbitration_suite",
            "mechanism": "Use the confirmed binding primitive inside a larger arbitration fixture.",
            "diagnostic_value": "medium",
            "oracle_risk": "medium",
            "scope_cost": "high",
            "recommendation": "defer",
            "reason": "Integration should follow a structured metadata derivation plan.",
        },
        {
            "option_id": "keep_prompt_visible_winner_label_only_and_stop_bridge_expansion",
            "mechanism": "Stop at explicit winner labels and keep resolved marker expansion closed.",
            "diagnostic_value": "low",
            "oracle_risk": "low",
            "scope_cost": "low",
            "recommendation": "not_selected",
            "reason": "It would leave the structured rule metadata bridge question untested.",
        },
    ]
    return {
        "schema_version": "phase_143z_next_decision_matrix_v1",
        "selected_option": SELECTED_OPTION,
        "recommended_next": NEXT,
        "options": options,
    }


def anti_oracle_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_143z_anti_oracle_requirements_v1",
        "forbid": [
            "winner=pocket_* in rule-derived subsets",
            "selected_pocket_id in prompt or request metadata",
            "final/winner value markers",
            "answer/gold/target/resolved output markers",
            "per-row selected pocket request metadata",
            "per-row manifest switching",
            "payload marker list narrowed to the correct pocket",
            "post-generation repair",
            "broad architecture claims",
        ],
        "allow": [
            "static pocket marker map",
            "canonical structured rule metadata",
            "explicit winner label baseline as a separate baseline subset only",
        ],
    }


def target_144a_milestone_plan() -> dict[str, Any]:
    grammars = {
        "quorum": ["rule_type=quorum", "votes=pocket_a,pocket_b,pocket_a"],
        "recency": ["rule_type=recency", "recency_order=pocket_c>pocket_b>pocket_a"],
        "tie_break": ["rule_type=tie_break", "tied=pocket_a,pocket_b", "tie_break_order=pocket_b>pocket_a>pocket_c"],
        "hierarchy": [
            "hierarchy=stale_rejection>recency>quorum>tie_break",
            "stale=pocket_a",
            "recency_winner=pocket_b",
            "quorum_winner=pocket_c",
            "tie_break_winner=pocket_b",
        ],
    }
    return {
        "schema_version": "phase_143z_target_144a_milestone_plan_v1",
        "milestone": NEXT,
        "planning_only": True,
        "artifact_only": True,
        "helper_generation_allowed": False,
        "helper_modification_allowed": False,
        "training_allowed": False,
        "first_executable_prototype": "144B_STRUCTURED_RULE_METADATA_TO_SELECTED_POCKET_BINDING_PROTOTYPE",
        "rule_metadata_scope": "canonical structured rule metadata only",
        "natural_language_rule_parsing_allowed": False,
        "candidate_rule_metadata_grammars": grammars,
        "target_144b_required_metrics": [
            "rule_metadata_parse_accuracy",
            "derived_selected_pocket_accuracy",
            "selected_pocket_to_marker_binding_accuracy",
            "end_to_end_answer_accuracy",
            "rule_metadata_ablation_accuracy",
            "explicit_winner_baseline_accuracy",
        ],
        "target_144b_required_subsets": [
            "EXPLICIT_WINNER_LABEL_BASELINE",
            "RULE_METADATA_DERIVED_NO_WINNER_LABEL",
            "SAME_VALUES_DIFFERENT_RULE",
            "SAME_RULE_DIFFERENT_VALUES",
            "SAME_TEMPLATE_OPPOSITE_RULE_WINNER",
            "RULE_METADATA_CORRUPTION_CONTROL",
        ],
        "rule_derived_subset_forbidden": anti_oracle_requirements()["forbid"],
        "claim_limit": (
            "A positive 144B would prove constrained structured rule metadata -> selected pocket binding only; "
            "it would not prove natural-language reasoning, open-ended arbitration, GPT-like capability, "
            "production readiness, or architecture superiority."
        ),
    }


def risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_143z_risk_register_v1",
        "risks": [
            {
                "risk_id": "hidden_winner_label_reintroduced",
                "risk": "Rule-derived subsets could accidentally include winner=pocket_* labels.",
                "mitigation": "144A/144B must ban winner labels outside the explicit baseline subset.",
            },
            {
                "risk_id": "oracle_manifest_selected_pocket",
                "risk": "A per-row manifest or request field could provide selected pocket identity.",
                "mitigation": "Keep static marker maps only and forbid selected_pocket_id request metadata.",
            },
            {
                "risk_id": "natural_language_reasoning_overclaim",
                "risk": "Structured metadata parsing could be misread as natural-language reasoning.",
                "mitigation": "Use canonical structured grammars only and keep boundary wording in all artifacts.",
            },
        ],
    }


def decision_payload(upstream_failed: list[str], ast_report: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    positive = not upstream_failed and ast_report.get("passed") is True and target.get("planning_only") is True
    decision = DECISION if positive else "structured_rule_metadata_next_decision_blocked"
    next_step = NEXT if positive else "143Z_BLOCKER_ANALYSIS"
    return {
        "schema_version": "phase_143z_decision_v1",
        "decision": decision,
        "selected_option": SELECTED_OPTION if positive else "blocked",
        "next": next_step,
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

- Selected-pocket binding scale confirmed: `{gap['selected_pocket_binding_scale_confirmed']}`
- Selected-marker occurrence rejection scale confirmed: `{gap['selected_marker_occurrence_rejection_scale_confirmed']}`
- Rule metadata to selected pocket identity untested: `{gap['rule_metadata_to_selected_pocket_identity_untested']}`
- Natural-language rule reasoning untested: `{gap['natural_language_rule_reasoning_untested']}`

## Target 144A

`{target['milestone']}` is planning-only and uses canonical structured rule metadata only. It routes the first executable prototype to `{target['first_executable_prototype']}`.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 143Z selected-pocket binding next decision plan")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-143w-root", type=Path, default=DEFAULT_143W_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_143z_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream, upstream_failed = require_143w(resolve_repo_path(args.upstream_143w_root))
    append_progress(out, "upstream verified", failed_gate_count=len(upstream_failed))
    ast_report = scan_ast()
    evidence = evidence_chain_summary(upstream)
    binding_state = selected_pocket_binding_state_report()
    gap = rule_metadata_bridge_gap_analysis()
    matrix = next_decision_matrix()
    anti_oracle = anti_oracle_requirements()
    risks = risk_register()
    target = target_144a_milestone_plan()
    decision = decision_payload(upstream_failed, ast_report, target)
    helper_source = HELPER_PATH.read_text(encoding="utf-8")
    config = {
        "schema_version": "phase_143z_analysis_config_v1",
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
        "schema_version": "phase_143z_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "decision": decision,
        "selected_pocket_binding_state_report": binding_state,
        "rule_metadata_bridge_gap_analysis": gap,
        "target_144a_milestone_plan": target,
        **FALSE_FLAGS,
    }

    write_json(out / "analysis_config.json", config)
    write_json(out / "upstream_143w_manifest.json", upstream)
    write_json(out / "evidence_chain_summary.json", evidence)
    write_json(out / "selected_pocket_binding_state_report.json", binding_state)
    write_json(out / "rule_metadata_bridge_gap_analysis.json", gap)
    write_json(out / "next_decision_matrix.json", matrix)
    write_json(out / "anti_oracle_requirements.json", anti_oracle)
    write_json(out / "risk_register.json", risks)
    write_json(out / "target_144a_milestone_plan.json", target)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision, gap, target)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_143z_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
