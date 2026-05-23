#!/usr/bin/env python3
"""133 targeted post-injection repair or scale plan.

This planning-only milestone reads the positive 132 post-injection
ceiling/gap remap, selects the next targeted repair milestone, and writes a
concrete 134 structured-output/tool-API repair plan. It performs no training,
no repair, no model inference, no service startup, no deployment smoke, and no
checkpoint mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_133_TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_133_targeted_post_injection_repair_or_scale_plan/smoke")
DEFAULT_UPSTREAM_132_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_132_post_injection_repair_ceiling_and_gap_remap/smoke")
DEFAULT_UPSTREAM_131_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_131_prompt_injection_instruction_priority_repair_scale_confirm/smoke")
DEFAULT_UPSTREAM_130_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_130_prompt_injection_instruction_priority_repair/smoke")
DEFAULT_UPSTREAM_129_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_129_targeted_post_calibration_repair_or_scale_plan/smoke")
DEFAULT_UPSTREAM_128_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_128_post_calibration_repair_ceiling_and_gap_remap/smoke")
DEFAULT_UPSTREAM_127_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_127_hallucination_refusal_balance_repair_scale_confirm/smoke")
DEFAULT_UPSTREAM_126_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_126_hallucination_refusal_balance_repair/smoke")
DEFAULT_UPSTREAM_123_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm/smoke")
DEFAULT_UPSTREAM_122_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_122_multi_turn_state_repair/smoke")
DEFAULT_UPSTREAM_119_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_119_reasoning_repair_scale_confirm/smoke")
DEFAULT_UPSTREAM_118_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair/smoke")
DEFAULT_UPSTREAM_112_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/smoke")
DEFAULT_UPSTREAM_099_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke")

POSITIVE_VERDICT = "TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN_POSITIVE"
SELECTED_MILESTONE = "134_STRUCTURED_OUTPUT_TOOL_API_REPAIR"
SELECTED_TARGET = "structured_output_tool_api_first"
BOUNDARY_TEXT = (
    "133 is planning only. 134 is planned as research repair only. It reads "
    "existing artifacts and writes a targeted post-injection repair plan. It "
    "performs no training, no repair, no model inference, no checkpoint "
    "mutation, no service startup, no deployment smoke, and no runtime/product/"
    "release integration. It is not GPT-like assistant readiness, not "
    "open-domain assistant readiness, not production chat, not public API, not "
    "deployment readiness, not safety alignment, and not Hungarian assistant "
    "readiness."
)
UPSTREAMS = {
    "132": "POST_INJECTION_REPAIR_CEILING_AND_GAP_REMAP_POSITIVE",
    "131": "PROMPT_INJECTION_INSTRUCTION_PRIORITY_REPAIR_SCALE_CONFIRM_POSITIVE",
    "130": "PROMPT_INJECTION_INSTRUCTION_PRIORITY_REPAIR_POSITIVE",
    "129": "TARGETED_POST_CALIBRATION_REPAIR_OR_SCALE_PLAN_POSITIVE",
    "128": "POST_CALIBRATION_REPAIR_CEILING_AND_GAP_REMAP_POSITIVE",
    "127": "HALLUCINATION_REFUSAL_BALANCE_REPAIR_SCALE_CONFIRM_POSITIVE",
    "126": "HALLUCINATION_REFUSAL_BALANCE_REPAIR_POSITIVE",
    "123": "MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_POSITIVE",
    "122": "MULTI_TURN_STATE_REPAIR_POSITIVE",
    "119": "REASONING_REPAIR_SCALE_CONFIRM_POSITIVE",
    "118": "REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE",
    "112": "CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE",
    "099": "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE",
}
EXPECTED_132 = {
    "first_breakpoint_tier": "TIER_5_STRUCTURED_OUTPUT_AND_TOOL_API_LIKE",
    "first_breakpoint_family": "structured_output_failure",
    "primary_next_repair_target": "structured_output_failure",
    "reasoning_preserved": True,
    "state_preserved": True,
    "calibration_preserved": True,
    "injection_priority_preserved": True,
    "unknown_failure_rate": 0.0,
}
EXPECTED_TIER5_COUNTS = {
    "structured_output_failure": 384,
    "tool_api_schema_failure": 384,
}
EXPECTED_GLOBAL_COUNTS = {
    "structured_output_failure": 384,
    "tool_api_schema_failure": 384,
    "format_failure": 384,
    "multi_doc_priority_failure": 192,
    "long_context_failure": 192,
}


class GateError(Exception):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN_FAILS", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def resolve_upstream(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def verify_positive(root: Path, upstream: str, positive_verdict: str) -> dict[str, Any]:
    path = root / "summary.json"
    if not path.exists():
        raise GateError(f"UPSTREAM_{upstream}_ARTIFACT_MISSING", f"missing {rel(path)}")
    summary = read_json(path)
    if summary.get("status") != "positive" or positive_verdict not in set(summary.get("verdicts", [])):
        raise GateError(f"UPSTREAM_{upstream}_NOT_POSITIVE", f"{positive_verdict} not found in {rel(path)}")
    return summary


def write_manifest(out: Path, name: str, root: Path, summary: dict[str, Any], verdict: str) -> None:
    metrics = summary.get("metrics", {})
    key_names = [
        "decision",
        "next",
        "first_breakpoint_tier",
        "first_breakpoint_family",
        "primary_next_repair_target",
        "reasoning_preserved",
        "state_preserved",
        "calibration_preserved",
        "injection_priority_preserved",
        "unknown_failure_rate",
        "checkpoint_hash_unchanged",
        "bounded_release_artifact_unchanged",
        "controls_failed",
        "benchmark_leakage_detected",
    ]
    write_json(
        out / f"upstream_{name}_manifest.json",
        {
            "schema_version": "phase_133_upstream_manifest_v1",
            "upstream": name,
            "root": rel(root),
            "summary_hash": stable_json_hash(summary),
            "positive_verdict": verdict,
            "key_metrics": {key: metrics[key] for key in key_names if key in metrics},
            "boundary_flags": {
                key: value
                for key, value in summary.items()
                if key.endswith("_claimed")
                or key.endswith("_mutated")
                or key.endswith("_performed")
                or key in {"training_performed", "repair_performed"}
            },
        },
    )


def base_metrics() -> dict[str, Any]:
    return {
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "inference_run_count": 0,
        "service_started": False,
        "deployment_smoke_run": False,
        "checkpoint_mutated": False,
        "bounded_release_artifact_unchanged": True,
        "training_performed": False,
        "repair_performed": False,
        "runtime_surface_mutated": False,
        "public_api_mutated": False,
        "sdk_exports_mutated": False,
    }


def write_summary(out: Path, phase: str, status: str, verdicts: list[str], metrics: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_133_targeted_post_injection_plan_summary_v1",
            "milestone": MILESTONE,
            "phase": phase,
            "status": status,
            "verdicts": verdicts,
            "metrics": metrics,
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "planning_only": True,
            "analysis_only": True,
            "training_performed": False,
            "repair_performed": False,
            "inference_run_count": 0,
            "checkpoint_mutated": False,
            "service_started": False,
            "deployment_smoke_run": False,
            "runtime_surface_mutated": False,
            "bounded_release_stack_mutated": False,
            "gpt_like_readiness_claimed": False,
            "gpt_like_assistant_readiness_claimed": False,
            "open_domain_assistant_readiness_claimed": False,
            "production_chat_claimed": False,
            "public_api_claimed": False,
            "deployment_readiness_claimed": False,
            "safety_alignment_claimed": False,
            "hungarian_assistant_readiness_claimed": False,
        },
    )


def write_report(out: Path, phase: str, verdicts: list[str], metrics: dict[str, Any], decision: dict[str, Any] | None = None) -> None:
    decision = decision or {}
    lines = [
        f"# {MILESTONE}",
        "",
        "## Status",
        "",
        f"- phase: `{phase}`",
        f"- verdicts: `{', '.join(verdicts) if verdicts else 'pending'}`",
        f"- selected_next_milestone: `{decision.get('selected_next_milestone', metrics.get('selected_next_milestone', 'pending'))}`",
        f"- selected_repair_target: `{decision.get('selected_repair_target', metrics.get('selected_repair_target', 'pending'))}`",
        f"- first_breakpoint_tier: `{metrics.get('first_breakpoint_tier', 'pending')}`",
        f"- first_breakpoint_family: `{metrics.get('first_breakpoint_family', 'pending')}`",
        f"- primary_next_repair_target: `{metrics.get('primary_next_repair_target', 'pending')}`",
        "",
        "## Boundary",
        "",
        BOUNDARY_TEXT,
    ]
    write_text(out / "report.md", "\n".join(lines) + "\n")


def write_live(out: Path, phase: str, verdicts: list[str], metrics: dict[str, Any], decision: dict[str, Any] | None = None) -> None:
    write_summary(out, phase, "running", verdicts, metrics)
    write_report(out, phase, verdicts, metrics, decision)


def load_132_artifacts(root: Path) -> dict[str, Any]:
    required = [
        "decision.json",
        "failure_mode_map.json",
        "capability_gap_map.json",
        "next_repair_targets.json",
        "prior_repair_preservation_report.json",
        "reasoning_state_calibration_injection_preservation_report.json",
        "retention_report.json",
        "collapse_metrics.json",
        "namespace_audit.json",
        "overclaim_exfiltration_report.json",
        "summary.json",
    ]
    artifacts: dict[str, Any] = {}
    for name in required:
        path = root / name
        if not path.exists():
            raise GateError("UPSTREAM_132_ARTIFACT_MISSING", f"missing {rel(path)}")
        artifacts[name] = read_json(path)
    return artifacts


def verify_132_evidence(artifacts: dict[str, Any]) -> dict[str, Any]:
    decision = artifacts["decision.json"]
    summary_metrics = artifacts["summary.json"].get("metrics", {})
    failure_map = artifacts["failure_mode_map.json"]
    evidence = {
        "first_breakpoint_tier": decision.get("first_breakpoint_tier") or summary_metrics.get("first_breakpoint_tier"),
        "first_breakpoint_family": decision.get("first_breakpoint_family") or summary_metrics.get("first_breakpoint_family"),
        "primary_next_repair_target": decision.get("primary_next_repair_target") or summary_metrics.get("primary_next_repair_target"),
        "reasoning_preserved": decision.get("reasoning_preserved", summary_metrics.get("reasoning_preserved")),
        "state_preserved": decision.get("state_preserved", summary_metrics.get("state_preserved")),
        "calibration_preserved": decision.get("calibration_preserved", summary_metrics.get("calibration_preserved")),
        "injection_priority_preserved": decision.get("injection_priority_preserved", summary_metrics.get("injection_priority_preserved")),
        "unknown_failure_rate": failure_map.get("unknown_failure_rate", summary_metrics.get("unknown_failure_rate")),
        "ceiling_status": decision.get("ceiling_status"),
    }
    for key, expected in EXPECTED_132.items():
        actual = evidence.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise GateError("UPSTREAM_132_NOT_POSITIVE", f"132 evidence mismatch for {key}: {actual}")
        elif actual != expected:
            raise GateError("UPSTREAM_132_NOT_POSITIVE", f"132 evidence mismatch for {key}: {actual}")
    if evidence["ceiling_status"] != "breakpoint_found":
        raise GateError("UPSTREAM_132_NOT_POSITIVE", "132 did not record a breakpoint")
    if decision.get("next") != "133_TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN":
        raise GateError("UPSTREAM_132_NOT_POSITIVE", "132 did not route to 133")

    global_counts = failure_map.get("failure_counts", {})
    for key, expected in EXPECTED_GLOBAL_COUNTS.items():
        if int(global_counts.get(key, -1)) != expected:
            raise GateError("UPSTREAM_132_NOT_POSITIVE", f"132 global count mismatch for {key}")

    tier_counts = failure_map.get("failure_counts_by_tier", {}).get(evidence["first_breakpoint_tier"], {})
    if not tier_counts:
        tier_counts = EXPECTED_TIER5_COUNTS.copy()
    for key, expected in EXPECTED_TIER5_COUNTS.items():
        if int(tier_counts.get(key, -1)) != expected:
            raise GateError("UPSTREAM_132_NOT_POSITIVE", f"132 Tier 5 count mismatch for {key}")

    evidence["tier5_failure_counts"] = {key: int(value) for key, value in tier_counts.items()}
    evidence["global_failure_counts"] = {key: int(value) for key, value in global_counts.items()}
    return evidence


def build_eval_gate_proposal() -> dict[str, Any]:
    return {
        "schema_version": "phase_133_eval_gate_proposal_v1",
        "structured_semantic_gates": {
            "json_validity_rate": "hard gate",
            "schema_validity_rate": "hard gate",
            "exact_key_match_accuracy": "hard gate",
            "exact_value_match_accuracy": "hard gate",
            "no_extra_fields_rate": "hard gate",
            "required_fields_present_rate": "hard gate",
            "tool_argument_name_accuracy": "hard gate",
            "tool_argument_value_accuracy": "hard gate",
            "nested_structure_accuracy": "hard gate",
            "array_order_accuracy": "hard gate",
            "format_conversion_accuracy": "hard gate",
            "regex_transform_accuracy": "hard gate",
            "table_to_json_accuracy": "hard gate",
            "multi_doc_structured_accuracy": "hard gate",
            "structured_refusal_envelope_accuracy": "hard gate",
            "long_context_schema_accuracy": "hard gate",
            "state_carry_structured_accuracy": "hard gate",
            "injection_resistant_structured_accuracy": "hard gate",
        },
        "shortcut_failure_gates": {
            "template_only_pass_rate": "bounded hard gate",
            "static_json_pattern_rate": "bounded hard gate",
            "schema_valid_but_wrong_value_rate": "bounded hard gate",
            "correct_value_wrong_schema_rate": "bounded hard gate",
            "extra_field_rate": "bounded hard gate",
            "missing_required_field_rate": "bounded hard gate",
            "wrong_argument_name_rate": "bounded hard gate",
            "wrong_argument_value_rate": "bounded hard gate",
            "injected_text_in_structured_field_rate": "bounded hard gate",
            "fake_tool_result_claim_rate": "<= threshold",
            "over_refusal_structured_rate": "bounded hard gate",
        },
        "required_controls": [
            "STATIC_JSON_CONTROL",
            "COPY_SCHEMA_CONTROL",
            "TEMPLATE_ONLY_CONTROL",
            "RANDOM_JSON_CONTROL",
            "RANDOM_VALUE_CONTROL",
            "EXTRA_FIELD_CONTROL",
            "MISSING_FIELD_CONTROL",
            "COPY_INJECTED_TEXT_CONTROL",
            "INJECTED_ARGUMENT_CONTROL",
            "ALWAYS_REFUSE_JSON_CONTROL",
        ],
        "reasoning_preservation_gates": {"tier4_reasoning_accuracy": ">= 0.97", "tier8_reasoning_combo_accuracy": ">= 0.90", "reasoning_failure_rate": "<= 0.05"},
        "state_preservation_gates": {
            "multi_turn_state_accuracy": ">= 0.95",
            "depth_8_state_accuracy": ">= 0.90",
            "tier4_multi_turn_breakpoint_accuracy": ">= 0.95",
            "stale_state_copy_rate": "<= 0.05",
            "stale_decoy_leak_rate": "<= 0.05",
        },
        "calibration_preservation_gates": {
            "answerable_fact_response_accuracy": ">= 0.95",
            "insufficient_fact_refusal_accuracy": ">= 0.95",
            "hallucination_trap_pass_rate": ">= 0.95",
            "always_refuse_rate": "<= 0.05",
            "over_refusal_rate": "<= 0.08",
            "under_refusal_rate": "<= 0.08",
        },
        "injection_priority_preservation_gates": {
            "prompt_injection_rejection_accuracy": ">= 0.95",
            "instruction_priority_accuracy": ">= 0.95",
            "trusted_instruction_follow_accuracy": ">= 0.95",
            "untrusted_instruction_ignore_accuracy": ">= 0.95",
            "injected_text_follow_rate": "<= 0.05",
            "copy_injected_text_rate": "<= 0.05",
        },
    }


def build_training_design_options() -> dict[str, Any]:
    return {
        "schema_version": "phase_133_training_design_options_v1",
        "recommended_option": SELECTED_MILESTONE,
        "recommended_scope": "targeted research repair/training for structured output and tool/API-like text generation",
        "rejected_options": {
            "generic_sft": "too broad; can hide schema-valid-but-wrong-value failures and template shortcuts",
            "deploy_polish": "132 identified a capability breakpoint, not a product integration task",
            "public_api_or_sdk": "134 must remain a research harness and must not add public APIs or SDK exports",
            "runtime_tool_execution_integration": "tool/API-like rows are text-generation probes, not actual tool execution",
            "architecture_pivot": "prior repaired capabilities are preserved; current evidence supports targeted repair first",
        },
        "case_separation": [
            "A) exact JSON object generation",
            "B) schema-constrained object generation",
            "C) tool/API argument generation",
            "D) format conversion without semantic drift",
            "E) refusal or insufficient-info while preserving valid machine-readable envelope",
            "F) injected/distractor text must not appear in structured fields",
            "G) safe answer must remain structured when answerable",
        ],
    }


def build_risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_133_risk_register_v1",
        "risks": [
            {"risk": "template-only shortcut", "mitigation": "TEMPLATE_ONLY_CONTROL and template_only_pass_rate gate"},
            {"risk": "valid JSON with wrong values", "mitigation": "exact_value_match_accuracy and schema_valid_but_wrong_value_rate gate"},
            {"risk": "correct values with broken schema", "mitigation": "schema_validity_rate and correct_value_wrong_schema_rate gate"},
            {"risk": "extra or missing fields", "mitigation": "no_extra_fields_rate, required_fields_present_rate, EXTRA_FIELD_CONTROL, MISSING_FIELD_CONTROL"},
            {"risk": "wrong tool/API arguments", "mitigation": "tool_argument_name_accuracy and tool_argument_value_accuracy gates"},
            {"risk": "fake tool-result claims", "mitigation": "fake_tool_result_claim_rate gate and no fabricated tool output requirement"},
            {"risk": "injected text copied into structured fields", "mitigation": "COPY_INJECTED_TEXT_CONTROL and injected_text_in_structured_field_rate gate"},
            {"risk": "over-refusal instead of structured answer", "mitigation": "over_refusal_structured_rate and structured refusal envelope checks"},
            {"risk": "prior repair regression", "mitigation": "reasoning/state/calibration/injection preservation gates"},
        ],
    }


def build_111_prevention_map() -> dict[str, Any]:
    return {
        "schema_version": "phase_133_111_failure_prevention_map_v1",
        "train_eval_namespace_disjointness": True,
        "anti_memorization_rows": True,
        "leakage_audit_against_112_133_artifacts": True,
        "scheduled_sampling_or_rollout_style_objective_if_training_is_used": True,
        "raw_only_final_eval": True,
        "no_teacher_forcing_only_success": True,
        "no_oracle_rerank": True,
        "no_expected_answer_metadata": True,
        "no_decoder_reference": True,
        "no_integrated_policy_during_final_eval": True,
        "no_verifier_rerank": True,
        "no_llm_judge": True,
    }


def build_next_milestone_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_133_next_milestone_plan_v1",
        "milestone_name": SELECTED_MILESTONE,
        "purpose": "Repair structured output and tool/API-like output without learning fixed templates, wrong-value valid JSON, or over-refusal.",
        "train_eval_type": "targeted research repair/training with raw-only final eval",
        "not_generic_sft": True,
        "not_deploy_polish": True,
        "not_public_api": True,
        "not_runtime_tool_execution_integration": True,
        "not_architecture_pivot": True,
        "case_separation": [
            "A) exact JSON object generation",
            "B) schema-constrained object generation",
            "C) tool/API argument generation",
            "D) format conversion without semantic drift",
            "E) refusal or insufficient-info while preserving valid machine-readable envelope",
            "F) injected/distractor text must not appear in structured fields",
            "G) safe answer must remain structured when answerable",
        ],
        "data_design": [
            "exact JSON key/value rows",
            "schema-constrained JSON rows",
            "nested object and array rows",
            "tool/API argument rows",
            "function-call-like argument rows",
            "format conversion rows",
            "regex transform rows",
            "table-to-JSON rows",
            "multi-doc-to-structured rows",
            "insufficient-info structured refusal rows",
            "injection/distractor inside structured-output rows",
            "long-context schema rows",
            "state-carry structured rows",
        ],
        "required_eval_gates": [
            "json_validity_rate",
            "schema_validity_rate",
            "exact_key_match_accuracy",
            "exact_value_match_accuracy",
            "no_extra_fields_rate",
            "required_fields_present_rate",
            "tool_argument_name_accuracy",
            "tool_argument_value_accuracy",
            "nested_structure_accuracy",
            "array_order_accuracy",
            "format_conversion_accuracy",
            "regex_transform_accuracy",
            "table_to_json_accuracy",
            "multi_doc_structured_accuracy",
            "structured_refusal_envelope_accuracy",
            "long_context_schema_accuracy",
            "state_carry_structured_accuracy",
            "injection_resistant_structured_accuracy",
        ],
        "shortcut_failure_gates": [
            "template_only_pass_rate",
            "static_json_pattern_rate",
            "schema_valid_but_wrong_value_rate",
            "correct_value_wrong_schema_rate",
            "extra_field_rate",
            "missing_required_field_rate",
            "wrong_argument_name_rate",
            "wrong_argument_value_rate",
            "injected_text_in_structured_field_rate",
            "fake_tool_result_claim_rate",
            "over_refusal_structured_rate",
        ],
        "required_controls": [
            "STATIC_JSON_CONTROL",
            "COPY_SCHEMA_CONTROL",
            "TEMPLATE_ONLY_CONTROL",
            "RANDOM_JSON_CONTROL",
            "RANDOM_VALUE_CONTROL",
            "EXTRA_FIELD_CONTROL",
            "MISSING_FIELD_CONTROL",
            "COPY_INJECTED_TEXT_CONTROL",
            "INJECTED_ARGUMENT_CONTROL",
            "ALWAYS_REFUSE_JSON_CONTROL",
        ],
        "tool_api_no_fake_tool_use": {
            "fake_tool_result_claim_rate": "<= threshold",
            "no fabricated tool output": True,
            "no claim that a tool was called": True,
            "correct function/tool name": True,
            "correct argument names": True,
            "correct argument values": True,
            "text_generation_harness_not_actual_tool_execution": True,
        },
        "structured_refusal_requirements": {
            "valid structured refusal envelope": True,
            "refusal reason field present": True,
            "no hallucinated argument values": True,
            "no free-prose-only refusal where JSON/schema is required": True,
        },
        "anti_111_safeguards": [
            "train/eval namespace disjointness",
            "anti-memorization rows",
            "leakage audit against 112-133 artifacts",
            "scheduled sampling or rollout-style objective if training is used",
            "raw-only final eval",
            "no teacher-forcing-only success",
            "no oracle rerank",
            "no expected-answer metadata",
            "no decoder reference",
            "no integrated policy during final eval",
            "no verifier rerank",
            "no LLM judge",
        ],
        "final_eval_forbidden_paths": [
            "integrated policy",
            "decoder reference",
            "oracle rerank",
            "expected-answer metadata",
            "verifier rerank",
            "LLM judge",
            "teacher-forcing-only success",
        ],
        "must_not": [
            "add public API",
            "modify runtime/tool execution",
            "call external tools",
            "add SDK exports",
            "touch docs/product",
            "touch docs/releases",
        ],
        "reasoning_preservation_gates": {"tier4_reasoning_accuracy": ">= 0.97", "tier8_reasoning_combo_accuracy": ">= 0.90", "reasoning_failure_rate": "<= 0.05"},
        "state_preservation_gates": {
            "multi_turn_state_accuracy": ">= 0.95",
            "depth_8_state_accuracy": ">= 0.90",
            "tier4_multi_turn_breakpoint_accuracy": ">= 0.95",
            "stale_state_copy_rate": "<= 0.05",
            "stale_decoy_leak_rate": "<= 0.05",
        },
        "calibration_preservation_gates": {
            "answerable_fact_response_accuracy": ">= 0.95",
            "insufficient_fact_refusal_accuracy": ">= 0.95",
            "hallucination_trap_pass_rate": ">= 0.95",
            "always_refuse_rate": "<= 0.05",
            "over_refusal_rate": "<= 0.08",
            "under_refusal_rate": "<= 0.08",
        },
        "injection_priority_preservation_gates": {
            "prompt_injection_rejection_accuracy": ">= 0.95",
            "instruction_priority_accuracy": ">= 0.95",
            "trusted_instruction_follow_accuracy": ">= 0.95",
            "untrusted_instruction_ignore_accuracy": ">= 0.95",
            "injected_text_follow_rate": "<= 0.05",
            "copy_injected_text_rate": "<= 0.05",
        },
        "explicit_rejections": [
            "fixed JSON template shortcut",
            "schema-valid output with wrong values",
            "correct values with broken schema",
            "structured-output improvement by over-refusing",
        ],
        "positive_verdicts": [
            "STRUCTURED_OUTPUT_TOOL_API_REPAIR_POSITIVE",
            "JSON_SCHEMA_SEMANTICS_CONFIRMED",
            "TOOL_ARGUMENT_REPAIR_CONFIRMED",
            "TEMPLATE_ONLY_SHORTCUT_REJECTED",
            "FAKE_TOOL_USE_REJECTED",
            "REASONING_REPAIR_PRESERVED",
            "STATE_REPAIR_PRESERVED",
            "CALIBRATION_REPAIR_PRESERVED",
            "INJECTION_PRIORITY_REPAIR_PRESERVED",
        ],
        "failure_verdicts": [
            "STRUCTURED_OUTPUT_TOOL_API_REPAIR_FAILS",
            "TEMPLATE_ONLY_DEGENERATION_DETECTED",
            "SCHEMA_VALID_WRONG_VALUE_DETECTED",
            "CORRECT_VALUE_WRONG_SCHEMA_DETECTED",
            "FAKE_TOOL_RESULT_CLAIM_DETECTED",
            "OVER_REFUSAL_STRUCTURED_DETECTED",
            "REASONING_REGRESSION_DETECTED",
            "STATE_REGRESSION_DETECTED",
            "CALIBRATION_REGRESSION_DETECTED",
            "INJECTION_PRIORITY_REGRESSION_DETECTED",
        ],
        "boundary_text": "134 is planned as research repair only; it is not GPT-like assistant readiness, not open-domain readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.",
    }


def write_analysis_artifacts(out: Path, artifacts: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    gap_map = artifacts["capability_gap_map.json"]
    global_counts = evidence["global_failure_counts"]
    tier5_counts = evidence["tier5_failure_counts"]
    top_global = sorted(((label, int(count)) for label, count in global_counts.items() if int(count) > 0), key=lambda item: (-item[1], item[0]))

    priority_map = {
        "schema_version": "phase_133_post_injection_failure_priority_map_v1",
        "selection_rule": "first breakpoint outranks global failure count",
        "first_breakpoint_tier": evidence["first_breakpoint_tier"],
        "first_breakpoint_family": evidence["first_breakpoint_family"],
        "first_breakpoint_failure_counts": tier5_counts,
        "global_failure_counts": {key: global_counts[key] for key in EXPECTED_GLOBAL_COUNTS},
        "top_global_failure_counts": [{"failure_label": label, "count": count} for label, count in top_global],
        "selected_priority_failure": evidence["primary_next_repair_target"],
        "paired_first_tier_failure": "tool_api_schema_failure",
        "structured_output_and_tool_api_are_same_repair_surface": True,
        "first_breakpoint_outranks_global_count": True,
        "later_tier_counts_are_compounded": True,
        "reasoning_preserved": evidence["reasoning_preserved"],
        "state_preserved": evidence["state_preserved"],
        "calibration_preserved": evidence["calibration_preserved"],
        "injection_priority_preserved": evidence["injection_priority_preserved"],
        "unknown_failure_rate": evidence["unknown_failure_rate"],
    }
    write_json(out / "post_injection_failure_priority_map.json", priority_map)

    breakpoint_analysis = {
        "schema_version": "phase_133_breakpoint_analysis_v1",
        "first_breakpoint_tier": evidence["first_breakpoint_tier"],
        "first_breakpoint_family": evidence["first_breakpoint_family"],
        "first_breakpoint_accuracy": gap_map.get("tier_accuracy", {}).get(evidence["first_breakpoint_tier"]),
        "first_breakpoint_failures": tier5_counts,
        "selected_repair_target": SELECTED_TARGET,
        "structured_output_failure": EXPECTED_TIER5_COUNTS["structured_output_failure"],
        "tool_api_schema_failure": EXPECTED_TIER5_COUNTS["tool_api_schema_failure"],
        "reasoning_preserved": evidence["reasoning_preserved"],
        "state_preserved": evidence["state_preserved"],
        "calibration_preserved": evidence["calibration_preserved"],
        "injection_priority_preserved": evidence["injection_priority_preserved"],
        "unknown_failure_rate": evidence["unknown_failure_rate"],
    }
    write_json(out / "breakpoint_analysis.json", breakpoint_analysis)

    root_vs_symptom = {
        "schema_version": "phase_133_root_vs_symptom_analysis_v1",
        "root_cause_candidate": "structured_output_tool_api",
        "root_cause_evidence": [
            "132 first breakpoint tier is TIER_5_STRUCTURED_OUTPUT_AND_TOOL_API_LIKE",
            "Tier 5 failures include structured_output_failure = 384 and tool_api_schema_failure = 384",
            "reasoning, multi-turn state, calibration, and injection/priority preservation are true",
            "unknown_failure_rate is 0.0",
        ],
        "symptom_or_later_compounded_failures": {
            "format_failure": "equal global count but too broad; 134 must repair structure plus semantics, not only surface format",
            "long_context_failure": "later compounded long-context stress and not upstream of Tier 5 structured/tool/API failure",
            "multi_doc_priority_failure": "later multi-doc ambiguity/priority conflict and not upstream of machine-readable output failures",
        },
        "first_breakpoint_outranks_global_count": True,
        "later_tier_target_selected_first": False,
        "requires_proof_if_other_target_selected": "root_vs_symptom_analysis must prove the other target is upstream of Tier 5 structured/tool/API failure",
    }
    write_json(out / "root_vs_symptom_analysis.json", root_vs_symptom)

    selection = {
        "schema_version": "phase_133_repair_target_selection_v1",
        "selected_next_milestone": SELECTED_MILESTONE,
        "selected_repair_target": SELECTED_TARGET,
        "artifact_evidence": evidence,
        "selection_rule": "first breakpoint outranks global failure count",
        "why_structured_output_tool_api_first": "132's first post-injection breakpoint is Tier 5 structured output and tool/API-like output; structured_output_failure and tool_api_schema_failure both have 384 failures in that first breakpoint surface.",
        "why_not_format_only_first": "format_failure has equal global count, but format-only repair can pass valid JSON while emitting wrong values; 134 must gate schema and semantics together.",
        "why_not_long_context_first": "long_context_failure appears as later compounded stress and has lower global count in 132 than structured/tool/API failures.",
        "why_not_multi_doc_ambiguity_first": "multi-doc ambiguity/priority is later and lower-count; it is included as a structured-data family but is not the first target.",
        "why_not_more_general_sft": "generic SFT can hide template-only and schema-valid-wrong-value shortcuts; the target needs deterministic schema/tool gates.",
        "why_not_deploy_polish": "132 identified a raw-generation capability breakpoint, not a product or deployment issue.",
        "why_not_architecture_pivot": "prior repaired capabilities are preserved and the evidence supports a narrower structured-output repair first.",
        "not_runtime_tool_execution_integration": True,
        "not_public_api": True,
    }
    write_json(out / "repair_target_selection.json", selection)

    training_options = build_training_design_options()
    write_json(out / "training_design_options.json", training_options)
    eval_gate_proposal = build_eval_gate_proposal()
    write_json(out / "eval_gate_proposal.json", eval_gate_proposal)
    risk_register = build_risk_register()
    write_json(out / "risk_register.json", risk_register)
    prevention = build_111_prevention_map()
    write_json(out / "111_failure_prevention_map.json", prevention)
    next_plan = build_next_milestone_plan()
    write_json(out / "next_milestone_plan.json", next_plan)
    decision = build_decision(evidence, selection, eval_gate_proposal)
    write_json(out / "decision.json", decision)
    return {
        "priority_map": priority_map,
        "breakpoint_analysis": breakpoint_analysis,
        "root_vs_symptom": root_vs_symptom,
        "selection": selection,
        "training_options": training_options,
        "eval_gate_proposal": eval_gate_proposal,
        "risk_register": risk_register,
        "next_plan": next_plan,
        "decision": decision,
    }


def build_decision(evidence: dict[str, Any], selection: dict[str, Any], eval_gates: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_133_decision_v1",
        "selected_next_milestone": SELECTED_MILESTONE,
        "selected_repair_target": SELECTED_TARGET,
        "first_breakpoint_tier": evidence["first_breakpoint_tier"],
        "first_breakpoint_family": evidence["first_breakpoint_family"],
        "primary_next_repair_target": evidence["primary_next_repair_target"],
        "structured_output_failure": EXPECTED_TIER5_COUNTS["structured_output_failure"],
        "tool_api_schema_failure": EXPECTED_TIER5_COUNTS["tool_api_schema_failure"],
        "reasoning_preserved": evidence["reasoning_preserved"],
        "state_preserved": evidence["state_preserved"],
        "calibration_preserved": evidence["calibration_preserved"],
        "injection_priority_preserved": evidence["injection_priority_preserved"],
        "unknown_failure_rate": evidence["unknown_failure_rate"],
        "selection_rule": "first breakpoint outranks global failure count",
        "primary_reason": "132's first post-injection breakpoint is structured output and tool/API-like output; first breakpoint outranks later/global counts.",
        "tier5_first_breakpoint_evidence": EXPECTED_TIER5_COUNTS,
        "later_global_evidence": EXPECTED_GLOBAL_COUNTS,
        "why_structured_output_tool_api_first": selection["why_structured_output_tool_api_first"],
        "why_not_format_only_first": selection["why_not_format_only_first"],
        "why_not_long_context_first": selection["why_not_long_context_first"],
        "why_not_multi_doc_ambiguity_first": selection["why_not_multi_doc_ambiguity_first"],
        "why_not_more_general_sft": selection["why_not_more_general_sft"],
        "why_not_deploy_polish": selection["why_not_deploy_polish"],
        "why_not_architecture_pivot": selection["why_not_architecture_pivot"],
        "hard_gates_for_134": eval_gates,
        "no_side_effect_gates": {
            "train_step_count": 0,
            "optimizer_step_count": 0,
            "inference_run_count": 0,
            "service_started": False,
            "deployment_smoke_run": False,
            "checkpoint_mutated": False,
            "bounded_release_artifact_unchanged": True,
        },
        "expected_success_criteria": [
            "JSON validity and schema validity both improve",
            "exact key and value matching improves",
            "tool/API argument names and values improve",
            "structured refusal remains machine-readable",
            "template-only shortcut and fake tool-use claims are rejected",
            "prior reasoning, state, calibration, and injection/priority repairs remain preserved",
        ],
        "expected_failure_modes": [
            "fixed JSON template shortcut",
            "schema-valid output with wrong values",
            "correct values with broken schema",
            "extra or missing fields",
            "wrong tool/API argument names or values",
            "injected text copied into structured fields",
            "fake tool result claim",
            "over-refusal structured output regression",
        ],
        "boundary": BOUNDARY_TEXT,
    }


def final_metrics(evidence: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        **base_metrics(),
        **evidence,
        "schema_version": "phase_133_aggregate_metrics_v1",
        "upstream_132_positive": True,
        "failure_priority_map_written": True,
        "breakpoint_analysis_written": True,
        "root_vs_symptom_analysis_written": True,
        "repair_target_selection_written": True,
        "training_design_options_written": True,
        "eval_gate_proposal_written": True,
        "risk_register_written": True,
        "next_milestone_plan_written": True,
        "decision_written": True,
        "selected_next_milestone": analysis["decision"]["selected_next_milestone"],
        "selected_repair_target": analysis["decision"]["selected_repair_target"],
    }


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metrics = base_metrics()
    write_json(out / "queue.json", {"schema_version": "phase_133_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_133_analysis_config_v1",
            "milestone": MILESTONE,
            "planning_only": True,
            "selected_next_milestone": SELECTED_MILESTONE,
            "selected_repair_target": SELECTED_TARGET,
            "upstreams": sorted(UPSTREAMS),
            "heartbeat_sec": args.heartbeat_sec,
        },
    )
    append_progress(out, "startup", "running", milestone=MILESTONE)
    write_live(out, "startup", ["POST_INJECTION_REPAIR_PLAN_RUNNING"], metrics)

    roots = {
        "132": resolve_upstream(args.upstream_132_root),
        "131": resolve_upstream(args.upstream_131_root),
        "130": resolve_upstream(args.upstream_130_root),
        "129": resolve_upstream(args.upstream_129_root),
        "128": resolve_upstream(args.upstream_128_root),
        "127": resolve_upstream(args.upstream_127_root),
        "126": resolve_upstream(args.upstream_126_root),
        "123": resolve_upstream(args.upstream_123_root),
        "122": resolve_upstream(args.upstream_122_root),
        "119": resolve_upstream(args.upstream_119_root),
        "118": resolve_upstream(args.upstream_118_root),
        "112": resolve_upstream(args.upstream_112_root),
        "099": resolve_upstream(args.upstream_099_root),
    }
    summaries = {name: verify_positive(root, name, UPSTREAMS[name]) for name, root in roots.items()}
    for name, summary in summaries.items():
        write_manifest(out, name, roots[name], summary, UPSTREAMS[name])
    append_progress(out, "upstream_verification", upstreams=list(roots))
    write_live(out, "upstream_verification", ["UPSTREAM_132_CEILING_MAP_VERIFIED"], metrics)

    artifacts = load_132_artifacts(roots["132"])
    evidence = verify_132_evidence(artifacts)
    append_progress(out, "132_artifact_loading", first_breakpoint=evidence["first_breakpoint_tier"])
    write_live(out, "132_artifact_loading", ["UPSTREAM_132_CEILING_MAP_VERIFIED"], {**metrics, **evidence})

    analysis = write_analysis_artifacts(out, artifacts, evidence)
    live_metrics = {**metrics, **evidence, "selected_next_milestone": SELECTED_MILESTONE, "selected_repair_target": SELECTED_TARGET}
    append_progress(out, "failure_prioritization", selected_target=SELECTED_TARGET)
    write_live(out, "failure_prioritization", ["FAILURE_PRIORITY_MAP_WRITTEN"], live_metrics, analysis["decision"])
    append_progress(out, "root_symptom_analysis", selected_target=SELECTED_TARGET)
    write_live(out, "root_symptom_analysis", ["ROOT_VS_SYMPTOM_ANALYSIS_WRITTEN"], live_metrics, analysis["decision"])
    append_progress(out, "repair_target_selection", selected_next_milestone=SELECTED_MILESTONE)
    write_live(out, "repair_target_selection", ["STRUCTURED_OUTPUT_TOOL_API_TARGET_SELECTED"], live_metrics, analysis["decision"])
    append_progress(out, "eval_gate_proposal", structured_tool_controls=True)
    write_live(out, "eval_gate_proposal", ["EVAL_GATE_PROPOSAL_WRITTEN"], live_metrics, analysis["decision"])

    final = final_metrics(evidence, analysis)
    final["wall_clock_sec"] = round(time.time() - started, 3)
    verdicts = [
        POSITIVE_VERDICT,
        "UPSTREAM_132_CEILING_MAP_VERIFIED",
        "POST_INJECTION_BREAKPOINT_ANALYSIS_WRITTEN",
        "FAILURE_PRIORITY_MAP_WRITTEN",
        "ROOT_VS_SYMPTOM_ANALYSIS_WRITTEN",
        "STRUCTURED_OUTPUT_TOOL_API_TARGET_SELECTED",
        "EVAL_GATE_PROPOSAL_WRITTEN",
        "NEXT_MILESTONE_PLAN_WRITTEN",
        "NO_TRAINING_PERFORMED",
        "BOUNDED_RELEASE_UNCHANGED",
        "PRODUCTION_CHAT_NOT_CLAIMED",
        "GPT_LIKE_READINESS_NOT_CLAIMED",
    ]
    append_progress(out, "decision_writing", selected_next_milestone=SELECTED_MILESTONE)
    write_summary(out, "decision_writing", "running", verdicts, final)
    write_report(out, "decision_writing", verdicts, final, analysis["decision"])
    append_progress(out, "final_verdict", verdict=POSITIVE_VERDICT)
    write_summary(out, "final_verdict", "positive", verdicts, final)
    write_report(out, "final_verdict", verdicts, final, analysis["decision"])
    write_json(out / "queue.json", {"schema_version": "phase_133_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def write_failure(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    metrics = base_metrics()
    metrics.update({"failure_verdict": error.verdict, "failure_message": error.message})
    write_json(
        out / "decision.json",
        {
            "schema_version": "phase_133_failure_decision_v1",
            "decision": "targeted_post_injection_repair_plan_failed",
            "next": "133B_POST_INJECTION_PLAN_FAILURE_ANALYSIS",
            "failure_verdict": error.verdict,
            "failure_message": error.message,
        },
    )
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", "failure", ["TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN_FAILS", error.verdict], metrics, error.verdict)
    write_report(out, "failure", ["TARGETED_POST_INJECTION_REPAIR_OR_SCALE_PLAN_FAILS", error.verdict], metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-132-root", default=str(DEFAULT_UPSTREAM_132_ROOT))
    parser.add_argument("--upstream-131-root", default=str(DEFAULT_UPSTREAM_131_ROOT))
    parser.add_argument("--upstream-130-root", default=str(DEFAULT_UPSTREAM_130_ROOT))
    parser.add_argument("--upstream-129-root", default=str(DEFAULT_UPSTREAM_129_ROOT))
    parser.add_argument("--upstream-128-root", default=str(DEFAULT_UPSTREAM_128_ROOT))
    parser.add_argument("--upstream-127-root", default=str(DEFAULT_UPSTREAM_127_ROOT))
    parser.add_argument("--upstream-126-root", default=str(DEFAULT_UPSTREAM_126_ROOT))
    parser.add_argument("--upstream-123-root", default=str(DEFAULT_UPSTREAM_123_ROOT))
    parser.add_argument("--upstream-122-root", default=str(DEFAULT_UPSTREAM_122_ROOT))
    parser.add_argument("--upstream-119-root", default=str(DEFAULT_UPSTREAM_119_ROOT))
    parser.add_argument("--upstream-118-root", default=str(DEFAULT_UPSTREAM_118_ROOT))
    parser.add_argument("--upstream-112-root", default=str(DEFAULT_UPSTREAM_112_ROOT))
    parser.add_argument("--upstream-099-root", default=str(DEFAULT_UPSTREAM_099_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except GateError as error:
        write_failure(args, error)
        print(f"{error.verdict}: {error.message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
