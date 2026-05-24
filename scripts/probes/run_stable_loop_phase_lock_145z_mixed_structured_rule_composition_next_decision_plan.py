#!/usr/bin/env python3
"""145Z planning-only next decision after mixed structured-rule composition scale confirm."""

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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_145Z_MIXED_STRUCTURED_RULE_COMPOSITION_NEXT_DECISION_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_145z_mixed_structured_rule_composition_next_decision_plan/smoke")
DEFAULT_145H_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_145h_mixed_structured_rule_composition_priority_binding_scale_confirm/smoke")
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_145z_mixed_structured_rule_composition_next_decision_plan_check.py"
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
DECISION = "trainable_structured_reasoning_distillation_bridge_plan_recommended"
SELECTED_OPTION = "trainable_structured_reasoning_distillation_bridge_plan"
NEXT = "146A_TRAINABLE_STRUCTURED_REASONING_DISTILLATION_BRIDGE_PROTOTYPE"
FALSE_FLAGS = {
    "reasoning_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "rule_metadata_reasoning_claimed": False,
    "natural_language_rule_reasoning_claimed": False,
    "open_ended_arbitration_claimed": False,
    "gpt_like_readiness_claimed": False,
    "gemma_like_capability_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
    "architecture_superiority_claimed": False,
}
BOUNDARY_TEXT = (
    "145Z is planning-only and artifact-only after the positive 145H scale confirm. "
    "It is constrained helper/backend evidence only for deciding the next model-facing bridge; "
    "it does not prove natural-language reasoning, not open-ended arbitration, "
    "not GPT-like/Gemma-like capability, not production readiness, and not architecture superiority."
)
EXPECTED_145H_METRICS = {
    "main_eval_rows": 9600,
    "end_to_end_answer_accuracy": 1.0,
    "positive_composition_subset_accuracy": 1.0,
    "fallback_control_subset_accuracy": 1.0,
    "final_selected_pocket_derivation_accuracy": 1.0,
    "selected_pocket_to_marker_binding_accuracy": 1.0,
    "priority_pocket_oracle_rejection_rate": 1.0,
    "rule_composition_ablation_accuracy": 0.0,
    "shared_helper_no_change_since_145a": True,
    "legacy_structured_rule_metadata_regression_passed": True,
    "legacy_selected_pocket_binding_regression_passed": True,
    "deterministic_replay_passed": True,
}
REQUIRED_146A_OUTPUTS = [
    "curriculum_train.jsonl",
    "curriculum_validation.jsonl",
    "curriculum_test.jsonl",
    "curriculum_ood_test.jsonl",
    "teacher_trace_manifest.json",
    "training_config.json",
    "evaluation_report.json",
    "oracle_shortcut_audit.json",
    "decision.json",
    "summary.json",
    "report.md",
]
REQUIRED_146A_METRICS = [
    "teacher_label_reproduction_accuracy",
    "selected_pocket_prediction_accuracy",
    "final_value_prediction_accuracy",
    "heldout_template_accuracy",
    "ood_composition_accuracy",
    "oracle_ablation_accuracy",
    "shortcut_scanner_violation_count",
    "train_validation_leakage_count",
    "test_template_overlap_rate",
    "deterministic_replay_passed",
]
REQUIRED_146A_SHORTCUT_FORBIDDEN = [
    "selected_pocket_id",
    "winner=pocket_*",
    "final_selected",
    "derived_selected",
    "answer value",
    "gold value",
    "target value",
    "resolved output",
    "expected output",
    "teacher trace fields in input",
    "per-row oracle metadata",
]
NEGATIVE_ROUTES_146A = {
    "curriculum_generation_failure": "146B_CURRICULUM_GENERATION_FAILURE_ANALYSIS",
    "train_eval_leakage_detected": "146C_TRAIN_EVAL_LEAKAGE_ANALYSIS",
    "model_shortcut_detected": "146D_MODEL_SHORTCUT_ANALYSIS",
    "teacher_label_reproduction_failure": "146E_TEACHER_DISTILLATION_FAILURE_ANALYSIS",
    "ood_generalization_failure": "146F_OOD_COMPOSITION_FAILURE_ANALYSIS",
    "helper_stack_regression": "145B_MIXED_RULE_BLOCK_PARSE_FAILURE_ANALYSIS",
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
                    if alias.name in {"torch", "tensorflow", "shared_raw_generation_helper"}:
                        failures.append(f"forbidden_import:{rel(path)}:{alias.name}")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in {"train", "fit", "backward", "step", "forward", "raw_generate", "load_checkpoint", "save_checkpoint"}:
                    failures.append(f"forbidden_call:{rel(path)}:{name}")
    return {"schema_version": "phase_145z_ast_scan_v1", "passed": not failures, "failures": failures}


def require_145h(root: Path) -> tuple[dict[str, Any], list[str]]:
    required = [
        "decision.json",
        "aggregate_metrics.json",
        "summary.json",
        "shared_helper_no_change_audit.json",
        "positive_vs_fallback_denominator_report.json",
        "priority_order_coverage_report.json",
        "block_type_candidate_coverage_report.json",
        "helper_request_audit.json",
        "static_manifest_integrity_report.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        return {"root": rel(root), "missing_artifacts": missing}, [f"missing:{name}" for name in missing]
    decision = read_json(root / "decision.json")
    metrics = read_json(root / "aggregate_metrics.json")
    summary = read_json(root / "summary.json")
    no_change = read_json(root / "shared_helper_no_change_audit.json")
    denominator = read_json(root / "positive_vs_fallback_denominator_report.json")
    priority = read_json(root / "priority_order_coverage_report.json")
    block = read_json(root / "block_type_candidate_coverage_report.json")
    request = read_json(root / "helper_request_audit.json")
    static = read_json(root / "static_manifest_integrity_report.json")
    checks = {
        "decision": decision.get("decision") == "mixed_structured_rule_composition_priority_binding_scale_confirmed",
        "verdict": decision.get("verdict") == "INSTNCT_MIXED_STRUCTURED_RULE_COMPOSITION_PRIORITY_BINDING_SCALE_CONFIRMED",
        "next": decision.get("next") == "145Z_MIXED_STRUCTURED_RULE_COMPOSITION_NEXT_DECISION_PLAN",
        "helper_no_change_audit_passed": no_change.get("passed") is True,
        "denominator_report_passed": denominator.get("passed") is True,
        "priority_order_coverage_passed": priority.get("passed") is True,
        "block_type_candidate_coverage_passed": block.get("passed") is True,
        "request_audit_passed": request.get("passed") is True and request.get("helper_request_forbidden_metadata_count") == 0,
        "static_manifest_integrity_passed": static.get("passed") is True and static.get("per_row_manifest_switch_rate") == 0.0 and static.get("per_row_payload_marker_switch_rate") == 0.0,
    }
    for key, expected in EXPECTED_145H_METRICS.items():
        checks[f"metric_{key}"] = metrics.get(key) == expected
    failures = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "phase_145z_upstream_145h_manifest_v1",
        "root": rel(root),
        "decision": decision,
        "aggregate_metrics": metrics,
        "summary": summary,
        "shared_helper_no_change_audit": no_change,
        "positive_vs_fallback_denominator_report": denominator,
        "priority_order_coverage_report": priority,
        "block_type_candidate_coverage_report": block,
        "helper_request_audit": request,
        "static_manifest_integrity_report": static,
        "checks": checks,
        "failed_checks": failures,
        "passed": not failures,
    }, failures


def build_evidence_chain_summary(upstream: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_145z_evidence_chain_summary_v1",
        "143w_explicit_selected_pocket_binding_scale_confirmed": True,
        "144h_single_structured_rule_metadata_binding_scale_confirmed": True,
        "145h_mixed_structured_rule_composition_scale_confirmed": True,
        "145h_main_eval_rows": upstream["aggregate_metrics"]["main_eval_rows"],
        "145h_positive_composition_subset_accuracy": upstream["aggregate_metrics"]["positive_composition_subset_accuracy"],
        "145h_fallback_control_subset_accuracy": upstream["aggregate_metrics"]["fallback_control_subset_accuracy"],
        "145h_oracle_ablation_accuracy": upstream["aggregate_metrics"]["rule_composition_ablation_accuracy"],
        "remaining_gap": "trainable/model-facing internalization of the structured helper scaffold is untested",
        "passed": upstream["passed"],
    }


def build_structured_helper_stack_state_report(upstream: dict[str, Any]) -> dict[str, Any]:
    metrics = upstream["aggregate_metrics"]
    return {
        "schema_version": "phase_145z_structured_helper_stack_state_report_v1",
        "explicit_selected_pocket_binding_scale_confirmed": True,
        "single_structured_rule_metadata_binding_scale_confirmed": True,
        "mixed_structured_rule_composition_scale_confirmed": True,
        "mixed_composition_end_to_end_answer_accuracy": metrics["end_to_end_answer_accuracy"],
        "mixed_composition_positive_subset_accuracy": metrics["positive_composition_subset_accuracy"],
        "mixed_composition_fallback_subset_accuracy": metrics["fallback_control_subset_accuracy"],
        "priority_pocket_oracle_rejection_rate": metrics["priority_pocket_oracle_rejection_rate"],
        "shared_helper_no_change_since_145a": metrics["shared_helper_no_change_since_145a"],
        "trainable_model_internalization_untested": True,
        "natural_language_rule_reasoning_untested": True,
        "passed": True,
    }


def build_model_facing_gap_analysis() -> dict[str, Any]:
    return {
        "schema_version": "phase_145z_model_facing_bridge_gap_analysis_v1",
        "structured_helper_stack_scale_confirmed": True,
        "trainable_model_internalization_untested": True,
        "natural_language_rule_reasoning_untested": True,
        "open_ended_arbitration_claimed": False,
        "gpt_like_or_gemma_like_capability_claimed": False,
        "recommended_bridge": "controlled supervised curriculum from the confirmed helper scaffold to a model-facing predictor",
        "passed": True,
    }


def build_next_decision_matrix() -> dict[str, Any]:
    options = [
        {
            "option": "trainable_structured_reasoning_distillation_bridge_plan",
            "recommended": True,
            "score": 1.0,
            "rationale": "Moves from scale-confirmed deterministic scaffold to controlled trainable/model-facing imitation with anti-oracle audits.",
        },
        {
            "option": "structured_helper_engine_extension_plan",
            "recommended": False,
            "score": 0.55,
            "rationale": "Would add more deterministic engine surface, but does not test model-facing internalization.",
        },
        {
            "option": "language_interface_wrapper_around_helper_plan",
            "recommended": False,
            "score": 0.45,
            "rationale": "Could expose a language-like interface too early while natural-language reasoning remains untested.",
        },
        {
            "option": "stop_at_mixed_structured_rule_helper_primitive",
            "recommended": False,
            "score": 0.25,
            "rationale": "Preserves current claims but leaves the model-facing bridge untested.",
        },
    ]
    return {
        "schema_version": "phase_145z_next_decision_matrix_v1",
        "options": options,
        "selected_option": SELECTED_OPTION,
        "decision": DECISION,
        "passed": True,
    }


def build_target_146a_milestone_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_145z_target_146a_milestone_plan_v1",
        "milestone": NEXT,
        "implementation_ready": True,
        "planning_only": False,
        "first_model_facing_bridge_prototype": True,
        "intended_primitive": [
            "use the confirmed structured helper scaffold as teacher",
            "generate controlled supervised curriculum",
            "train/evaluate a small model-facing predictor on LM-compatible canonical structured inputs",
            "test selected-pocket and final-value imitation without hidden oracle shortcuts",
        ],
        "input_policy": {
            "canonical_structured_prompts_only": True,
            "free_form_natural_language_rule_parsing": False,
            "heldout_adversarial_templates_required": True,
            "train_validation_test_split_required": True,
            "ood_test_required": True,
        },
        "required_outputs": REQUIRED_146A_OUTPUTS,
        "required_metrics": REQUIRED_146A_METRICS,
        "positive_gates": {
            "teacher_label_reproduction_accuracy": ">= 0.80",
            "selected_pocket_prediction_accuracy": ">= 0.80",
            "final_value_prediction_accuracy": ">= 0.80",
            "heldout_template_accuracy": ">= 0.70",
            "ood_composition_accuracy": ">= 0.60",
            "oracle_ablation_accuracy": "<= 0.20",
            "shortcut_scanner_violation_count": "= 0",
            "train_validation_leakage_count": "= 0",
            "test_template_overlap_rate": "<= 0.05",
            "deterministic_replay_passed": "= true",
        },
        "model_facing_input_forbidden_shortcuts": REQUIRED_146A_SHORTCUT_FORBIDDEN,
        "teacher_trace_policy": {
            "teacher_trace_manifest_allowed_as_eval_artifact": True,
            "teacher_trace_fields_forbidden_in_model_input": True,
            "teacher_outputs_must_be_split_safe": True,
        },
        "clean_negative_routes": NEGATIVE_ROUTES_146A,
        "claim_limit": "A positive 146A proves only limited supervised imitation of the structured scaffold under controlled inputs.",
        "not_claimed": [
            "natural-language rule reasoning",
            "open-ended arbitration",
            "GPT-like/Gemma-like capability",
            "production readiness",
            "architecture superiority",
        ],
    }


def build_anti_oracle_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_145z_anti_oracle_requirements_v1",
        "model_facing_inputs_must_forbid": REQUIRED_146A_SHORTCUT_FORBIDDEN,
        "hidden_shortcuts_forbidden": [
            "selected pocket request metadata",
            "winner labels",
            "answer/gold/target/resolved/expected output fields",
            "teacher trace fields in input",
            "train/validation/test leakage",
            "template overlap beyond allowed threshold",
            "per-row oracle metadata",
        ],
        "oracle_ablation_required": True,
        "shortcut_scanner_required": True,
        "passed": True,
    }


def build_risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_145z_risk_register_v1",
        "risks": [
            {
                "risk": "model learns marker/template shortcuts rather than scaffold behavior",
                "mitigation": "heldout templates, OOD composition split, shortcut scanner, and oracle ablation",
            },
            {
                "risk": "teacher trace leaks into model-facing input",
                "mitigation": "teacher traces remain artifacts only and are forbidden in input scanner",
            },
            {
                "risk": "positive 146A is overclaimed as language reasoning",
                "mitigation": "canonical structured prompts only and explicit broad-capability false flags",
            },
            {
                "risk": "train/test leakage inflates reproduction metrics",
                "mitigation": "train_validation_leakage_count and test_template_overlap_rate gates",
            },
        ],
        "passed": True,
    }


def build_decision() -> dict[str, Any]:
    return {
        "schema_version": "phase_145z_decision_v1",
        "decision": DECISION,
        "selected_option": SELECTED_OPTION,
        "next": NEXT,
        "positive_gate_passed": True,
        "boundary": BOUNDARY_TEXT,
        **FALSE_FLAGS,
    }


def write_report(out: Path, decision: dict[str, Any]) -> None:
    text = f"""# {MILESTONE}

Decision: `{decision['decision']}`

Selected option: `{decision['selected_option']}`

Next: `{decision['next']}`

Boundary: {BOUNDARY_TEXT}

## Summary

145H scale-confirmed mixed structured-rule composition with explicit priority over block types. The selected next direction is a trainable structured reasoning distillation bridge, not another deterministic helper-engine extension.

The target 146A milestone is model-facing but still constrained: it starts from canonical structured prompts, uses the confirmed helper scaffold as teacher, and evaluates supervised imitation with anti-oracle, leakage, heldout-template, and OOD checks.

## Claim Limit

145Z is planning-only. A future positive 146A would prove only limited supervised imitation of the structured scaffold under controlled inputs. It would not prove natural-language rule reasoning, open-ended arbitration, GPT-like/Gemma-like capability, production readiness, or architecture superiority.
"""
    write_text(out / "report.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 145Z mixed structured-rule composition next decision plan")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--upstream-145h-root", type=Path, default=DEFAULT_145H_ROOT)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()

    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    append_progress(out, "startup", milestone=MILESTONE, heartbeat_sec=args.heartbeat_sec)
    write_json(out / "queue.json", {"schema_version": "phase_145z_queue_v1", "milestone": MILESTONE, "status": "running"})

    upstream, upstream_failures = require_145h(resolve_repo_path(args.upstream_145h_root))
    if upstream_failures:
        raise RuntimeError(f"145H upstream verification failed: {upstream_failures}")
    write_json(out / "upstream_145h_manifest.json", upstream)
    append_progress(out, "upstream verified", upstream_decision=upstream["decision"]["decision"])

    ast_scan = scan_ast()
    if not ast_scan["passed"]:
        raise RuntimeError(f"145Z static scan failed: {ast_scan['failures']}")
    if not helper_unchanged_from_head():
        raise RuntimeError("shared_raw_generation_helper.py changed from HEAD")
    append_progress(out, "static integrity verified", helper_unchanged=True)

    decision = build_decision()
    evidence = build_evidence_chain_summary(upstream)
    state = build_structured_helper_stack_state_report(upstream)
    gap = build_model_facing_gap_analysis()
    matrix = build_next_decision_matrix()
    target = build_target_146a_milestone_plan()
    anti = build_anti_oracle_requirements()
    risk = build_risk_register()
    config = {
        "schema_version": "phase_145z_analysis_config_v1",
        "milestone": MILESTONE,
        "boundary": BOUNDARY_TEXT,
        "planning_only": True,
        "artifact_only": True,
        "raw_generate_allowed": False,
        "shared_helper_import_allowed": False,
        "helper_modification_allowed": False,
        "training_or_checkpoint_mutation_allowed": False,
        "selected_option": SELECTED_OPTION,
        "next": NEXT,
        "runner_sha256": sha256_text(RUNNER_PATH.read_text(encoding="utf-8")),
        **FALSE_FLAGS,
    }
    summary = {
        "schema_version": "phase_145z_summary_v1",
        "milestone": MILESTONE,
        "status": "complete",
        "boundary": BOUNDARY_TEXT,
        "decision": decision,
        "selected_option": SELECTED_OPTION,
        "next": NEXT,
        "upstream_145h_verified": True,
        "target_146a_implementation_ready": True,
        **FALSE_FLAGS,
    }

    write_json(out / "analysis_config.json", config)
    write_json(out / "evidence_chain_summary.json", evidence)
    write_json(out / "structured_helper_stack_state_report.json", state)
    write_json(out / "model_facing_bridge_gap_analysis.json", gap)
    write_json(out / "next_decision_matrix.json", matrix)
    write_json(out / "target_146a_milestone_plan.json", target)
    write_json(out / "anti_oracle_requirements.json", anti)
    write_json(out / "risk_register.json", risk)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", summary)
    write_report(out, decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    write_json(out / "queue.json", {"schema_version": "phase_145z_queue_v1", "milestone": MILESTONE, "status": "complete", "decision": decision["decision"], "next": decision["next"]})
    print(json.dumps({"decision": decision["decision"], "selected_option": decision["selected_option"], "next": decision["next"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
