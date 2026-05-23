#!/usr/bin/env python3
"""140Z planning-only decision after multi-step transfer scale confirm."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_140Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_DECISION_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_140z_instnct_pocket_gated_multi_field_transfer_decision_plan/smoke")
DEFAULT_140Y_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_140y_instnct_pocket_gated_multi_step_transfer_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_140z_instnct_pocket_gated_multi_field_transfer_decision_plan_check.py"
POSITIVE_NEXT = "141A_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_PROBE"
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
    "140Z is planning-only and artifact-only after the positive 140Y multi-step "
    "transfer scale confirm. It does not train, run new helper generation, mutate "
    "checkpoints, modify shared_raw_generation_helper.py, modify helper/backend/"
    "runtime/release/product surfaces, change public request keys, start services, "
    "deploy, change root LICENSE, or claim GPT-like or broad assistant readiness."
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
    return {"schema_version": "phase_140z_ast_scan_v1", "passed": not failures, "failures": failures}


def require_140y(root: Path) -> dict[str, Any]:
    required = ["decision.json", "arm_comparison.json", "determinism_replay_report.json", "summary.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing 140Y artifacts: {missing}")
    decision = read_json(root / "decision.json")
    comparison = read_json(root / "arm_comparison.json")
    replay = read_json(root / "determinism_replay_report.json")
    summary = read_json(root / "summary.json")
    if decision.get("decision") != "instnct_pocket_gated_multi_step_transfer_scale_confirmed":
        raise RuntimeError(f"bad 140Y decision: {decision.get('decision')}")
    if decision.get("next") != "140Z_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_DECISION_PLAN":
        raise RuntimeError(f"bad 140Y next: {decision.get('next')}")
    exact = {
        "eval_row_count": comparison.get("eval_row_count") == 2880,
        "family_count": comparison.get("family_count") == 6,
        "scaffold_variant_count": comparison.get("scaffold_variant_count") == 48,
        "main_final_answer_accuracy": comparison.get("main_final_answer_accuracy") == 1.0,
        "main_step1_intermediate_accuracy": comparison.get("main_step1_intermediate_accuracy") == 1.0,
        "main_step2_final_accuracy": comparison.get("main_step2_final_accuracy") == 1.0,
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate") == 1.0,
        "main_contrast_group_accuracy": comparison.get("main_contrast_group_accuracy") == 1.0,
        "ablation_final_answer_accuracy": comparison.get("ablation_final_answer_accuracy") == 0.0,
        "pocket_ablation_delta_final_answer_accuracy": comparison.get("pocket_ablation_delta_final_answer_accuracy") == 1.0,
        "source_copy_shortcut_rate": comparison.get("source_copy_shortcut_rate") == 0.0,
        "intermediate_copy_shortcut_rate": comparison.get("intermediate_copy_shortcut_rate") == 0.0,
        "visible_bypass_violation_rate": comparison.get("visible_bypass_violation_rate") == 0.0,
        "noisy_distractor_violation_rate": comparison.get("noisy_distractor_violation_rate") == 0.0,
        "direct_pocket_value_marker_rate": comparison.get("direct_pocket_value_marker_rate") == 0.0,
        "deterministic_replay_passed": replay.get("deterministic_replay_passed") is True,
        "summary_status": summary.get("status") == "complete",
    }
    failed = [key for key, passed in exact.items() if not passed]
    if failed:
        raise RuntimeError(f"140Y exact profile mismatch: {failed}")
    return {
        "root": rel(root),
        "decision": decision.get("decision"),
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "eval_row_count": comparison.get("eval_row_count"),
        "family_count": comparison.get("family_count"),
        "scaffold_variant_count": comparison.get("scaffold_variant_count"),
        "main_final_answer_accuracy": comparison.get("main_final_answer_accuracy"),
        "main_step1_intermediate_accuracy": comparison.get("main_step1_intermediate_accuracy"),
        "main_step2_final_accuracy": comparison.get("main_step2_final_accuracy"),
        "main_pocket_writeback_rate": comparison.get("main_pocket_writeback_rate"),
        "main_contrast_group_accuracy": comparison.get("main_contrast_group_accuracy"),
        "ablation_final_answer_accuracy": comparison.get("ablation_final_answer_accuracy"),
        "pocket_ablation_delta_final_answer_accuracy": comparison.get("pocket_ablation_delta_final_answer_accuracy"),
        "source_copy_shortcut_rate": comparison.get("source_copy_shortcut_rate"),
        "intermediate_copy_shortcut_rate": comparison.get("intermediate_copy_shortcut_rate"),
        "visible_bypass_violation_rate": comparison.get("visible_bypass_violation_rate"),
        "noisy_distractor_violation_rate": comparison.get("noisy_distractor_violation_rate"),
        "direct_pocket_value_marker_rate": comparison.get("direct_pocket_value_marker_rate"),
        "deterministic_replay_passed": replay.get("deterministic_replay_passed"),
        "gate_checks": exact,
    }


def evidence_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_140z_evidence_chain_summary_v1",
        "current_state": "multi-step transfer scale confirmed",
        "evidence": upstream,
        "interpretation": [
            "A->B->C multi-step transfer survived 2880 rows, 6 families, and 48 scaffold variants.",
            "Closed-pocket ablation remained decision-critical.",
            "Source, intermediate, visible bypass, noisy distractor, and direct POCKET_VALUE shortcuts stayed at zero.",
            "The next useful falsification is whether final answers require multiple fields or pocket sources, not just one final payload marker.",
        ],
        "not_claimed": ["GPT-like readiness", "broad assistant capability", "open-domain reasoning", "architecture superiority"],
    }


def gap_analysis() -> dict[str, Any]:
    return {
        "schema_version": "phase_140z_multi_step_to_multi_field_gap_analysis_v1",
        "closed_gaps": [
            "single-field transform bridge scale",
            "A->B->C multi-step transfer scale",
            "source/intermediate copy rejection",
            "closed-pocket causal ablation",
            "visible/noisy bypass rejection",
        ],
        "remaining_gaps": [
            "no measured two-field final binding",
            "no measured multi-pocket composition",
            "no measured priority conflict resolution between fields",
            "no measured same-template multi-field contrast groups",
        ],
        "next_gap_to_test": "multi-field and multi-pocket transfer bridge",
    }


def decision_matrix() -> dict[str, Any]:
    return {
        "schema_version": "phase_140z_transfer_decision_matrix_v1",
        "selected_option": "multi_field_multi_pocket_transfer",
        "recommended_next": POSITIVE_NEXT,
        "options": [
            {"option": "more_multi_step_scale", "recommendation": "reject_for_now", "diagnostic_value": "medium_low", "reason": "140Y already scale-confirmed multi-step transfer with perfect shortcut profile"},
            {"option": "multi_field_multi_pocket_transfer", "recommendation": "select", "diagnostic_value": "high", "reason": "tests whether the pocket-gated path survives multiple required fields and same-template field rebinding"},
            {"option": "conflict_priority_transfer", "recommendation": "include_in_141a", "diagnostic_value": "high", "reason": "priority conflict should be a family inside 141A rather than a separate first probe"},
            {"option": "real_task_mixed_suite", "recommendation": "defer", "diagnostic_value": "medium_later", "reason": "mixed-suite breadth should follow a focused multi-field falsification"},
            {"option": "blocker_analysis", "recommendation": "not_selected", "diagnostic_value": "low_now", "reason": "140Y upstream evidence is complete and positive"},
        ],
    }


def multi_field_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_140z_multi_field_transfer_requirements_v1",
        "target_milestone": POSITIVE_NEXT,
        "helper_only_final_eval": True,
        "training_allowed": False,
        "helper_backend_modification_allowed": False,
        "public_request_key_change_allowed": False,
        "must_test_multi_field_not_single_field": True,
        "families": [
            "FIELD_A_PLUS_FIELD_B_TO_FINAL",
            "POCKET_SOURCE_TABLE_RULE_FIELD",
            "DUAL_POCKET_PRIORITY_CONFLICT",
            "MULTI_FIELD_SAME_TEMPLATE_CONTRAST",
            "DISTRACTOR_FIELD_MIX",
            "INTERMEDIATE_FIELD_CHAIN",
        ],
        "row_fields": [
            "field_a_value",
            "field_b_value",
            "optional_table_field_value",
            "optional_rule_field_value",
            "intermediate_value",
            "final_target_value",
            "visible_wrong_value",
            "noisy_distractor_values",
            "priority_rule_id",
            "contrast_group_id",
        ],
        "positive_gates": {
            "main_final_answer_accuracy_min": 0.55,
            "multi_field_binding_accuracy_min": 0.55,
            "main_pocket_writeback_rate_min": 0.70,
            "main_contrast_group_accuracy_min": 0.55,
            "ablation_final_answer_accuracy_max": 0.20,
            "pocket_ablation_delta_min": 0.30,
            "single_field_shortcut_rate": 0.0,
            "visible_bypass_violation_rate": 0.0,
            "noisy_distractor_violation_rate": 0.0,
            "direct_pocket_value_marker_rate": 0.0,
            "deterministic_replay_passed": True,
        },
    }


def anti_shortcuts() -> dict[str, Any]:
    return {
        "schema_version": "phase_140z_anti_shortcut_requirements_v1",
        "reject": [
            "field A only",
            "field B only",
            "intermediate only",
            "visible wrong value",
            "noisy distractor value",
            "closed pocket correct answer",
            "single-field shortcut success",
            "priority conflict wrong field",
            "prefix-only success",
            "direct POCKET_VALUE main path",
            "expected-output construction",
            "old runner imports",
            "oracle/rerank/verifier/LLM judge",
            "post-generation repair",
        ],
        "required_controls": [
            "FIELD_A_ONLY_CONTROL",
            "FIELD_B_ONLY_CONTROL",
            "INTERMEDIATE_COPY_CONTROL",
            "VISIBLE_TARGET_BYPASS_CONTROL",
            "NOISY_DISTRACTOR_CONTROL",
            "CLOSED_POCKET_ABLATION_CONTROL",
            "SINGLE_FIELD_SHORTCUT_CONTROL",
            "PRIORITY_CONFLICT_WRONG_FIELD_CONTROL",
            "PREFIX_ONLY_CONTROL",
        ],
    }


def target_141a_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_140z_target_141a_milestone_plan_v1",
        "milestone": POSITIVE_NEXT,
        "type": "helper-only executable probe",
        "train_allowed": False,
        "helper_generation_allowed": True,
        "helper_backend_modification_allowed": False,
        "public_request_key_change_allowed": False,
        "source_checkpoint_mutation_allowed": False,
        "clean_negative_accepted": True,
        "selected_candidate": "open_multi_field_final_all_markers",
        "candidate_manifests": [
            "closed_pocket_no_multi_field",
            "wrong_gate_marker_no_multi_field",
            "field_a_only_candidate",
            "field_b_only_candidate",
            "intermediate_copy_candidate",
            "priority_wrong_candidate",
            "visible_bypass_candidate",
            "open_multi_field_final_all_markers",
        ],
        "main_payload_markers": [
            "resolved multi-field final:",
            "priority-selected final:",
            "joined field result:",
            "verified combined target:",
        ],
        "required_artifacts": [
            "multi_field_eval_manifest.json",
            "multi_field_binding_manifest.json",
            "multi_field_transfer_metrics.json",
            "field_shortcut_report.json",
            "priority_conflict_report.json",
            "single_field_shortcut_report.json",
            "arm_comparison.json",
            "decision.json",
            "summary.json",
            "report.md",
        ],
        "failure_routes": {
            "single_field_shortcut_detected": "141B_SINGLE_FIELD_SHORTCUT_ANALYSIS",
            "multi_field_binding_failure": "141C_MULTI_FIELD_BINDING_FAILURE_ANALYSIS",
            "pocket_ablation_not_decision_critical": "141D_POCKET_CAUSALITY_FAILURE_ANALYSIS",
            "priority_conflict_failure": "141E_PRIORITY_CONFLICT_FAILURE_ANALYSIS",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
        },
        "positive_route": "141F_INSTNCT_POCKET_GATED_MULTI_FIELD_TRANSFER_SCALE_CONFIRM",
    }


def risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_140z_risk_register_v1",
        "risks": [
            {"risk": "current helper selects one payload value", "mitigation": "score 141A as constrained helper evidence and use candidate/control failures to reject single-field shortcuts"},
            {"risk": "multi-field prompt may hide a direct final marker", "mitigation": "direct POCKET_VALUE remains zero and single-field candidates/controls must fail"},
            {"risk": "priority conflict becomes narrative-only", "mitigation": "require explicit priority conflict report and wrong-priority candidate failure"},
        ],
    }


def write_report(out: Path, decision: dict[str, Any], matrix: dict[str, Any], upstream: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

140Y evidence:

- eval rows: `{upstream['eval_row_count']}`
- families: `{upstream['family_count']}`
- scaffold variants: `{upstream['scaffold_variant_count']}`
- final/step1/step2/writeback/contrast: `1.0`
- ablation final answer accuracy: `{upstream['ablation_final_answer_accuracy']}`
- ablation delta: `{upstream['pocket_ablation_delta_final_answer_accuracy']}`
- source/intermediate copy rates: `{upstream['source_copy_shortcut_rate']}` / `{upstream['intermediate_copy_shortcut_rate']}`
- visible/noisy/direct marker rates: `{upstream['visible_bypass_violation_rate']}` / `{upstream['noisy_distractor_violation_rate']}` / `{upstream['direct_pocket_value_marker_rate']}`
- deterministic replay passed: `{upstream['deterministic_replay_passed']}`

Selected option: `{matrix['selected_option']}`.

This remains planning-only constrained helper evidence: not GPT-like readiness,
not broad assistant capability, not production readiness, not public API
readiness, not deployment readiness, and not safety alignment.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-140y-root", type=Path, default=DEFAULT_140Y_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_140z_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream = require_140y(resolve_repo_path(args.upstream_140y_root))
    write_json(out / "upstream_140y_manifest.json", upstream)
    append_progress(out, "upstream verification", upstream_hash=stable_hash(upstream))

    config = {
        "schema_version": "phase_140z_analysis_config_v1",
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
        "public_request_key_change_allowed": False,
        "runtime_surface_mutated": False,
        "release_surface_mutated": False,
        "product_surface_mutated": False,
        "root_license_changed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "analysis_config.json", config)
    append_progress(out, "artifact loading", artifact_count=4)

    ast_report = scan_ast()
    write_json(out / "ast_shortcut_scan_report.json", ast_report)
    evidence = evidence_summary(upstream)
    gaps = gap_analysis()
    matrix = decision_matrix()
    reqs = multi_field_requirements()
    shortcuts = anti_shortcuts()
    target = target_141a_plan()
    risks = risk_register()
    write_json(out / "evidence_chain_summary.json", evidence)
    write_json(out / "multi_step_to_multi_field_gap_analysis.json", gaps)
    write_json(out / "transfer_decision_matrix.json", matrix)
    write_json(out / "multi_field_transfer_requirements.json", reqs)
    write_json(out / "anti_shortcut_requirements.json", shortcuts)
    write_json(out / "target_141a_milestone_plan.json", target)
    write_json(out / "risk_register.json", risks)
    append_progress(out, "decision matrix", selected=matrix["selected_option"], next=matrix["recommended_next"])

    if not ast_report["passed"]:
        decision = "helper_integrity_failure"
        next_step = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    elif matrix["selected_option"] != "multi_field_multi_pocket_transfer":
        decision = "upstream_evidence_blocked"
        next_step = "140Z_UPSTREAM_EVIDENCE_REPAIR"
    else:
        decision = "multi_field_transfer_probe_recommended"
        next_step = POSITIVE_NEXT
    decision_payload = {
        "schema_version": "phase_140z_decision_v1",
        "decision": decision,
        "next": next_step,
        "planning_only": True,
        "artifact_only": True,
        "clean_negative_valid": True,
        "selected_option": matrix["selected_option"],
        "recommended_next": matrix["recommended_next"],
        "capability_claimed": False,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision_payload)
    summary = {
        "schema_version": "phase_140z_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "upstream": upstream,
        "decision_matrix": matrix,
        **decision_payload,
    }
    write_json(out / "summary.json", summary)
    write_report(out, decision_payload, matrix, upstream)
    append_progress(out, "decision", decision=decision, next=next_step)
    append_progress(out, "final verdict", decision=decision)
    write_json(out / "queue.json", {"schema_version": "phase_140z_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision, "next": next_step})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
