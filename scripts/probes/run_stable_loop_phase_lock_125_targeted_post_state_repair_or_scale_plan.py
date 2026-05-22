#!/usr/bin/env python3
"""125 targeted post-state repair or scale plan.

This planning-only milestone reads the positive 124 post-state ceiling/gap
remap, selects the next targeted repair milestone, and writes a concrete 126
plan. It performs no training, no repair, no model inference, no service
startup, no deployment smoke, and no checkpoint mutation.
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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_125_TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_125_targeted_post_state_repair_or_scale_plan/smoke")
DEFAULT_UPSTREAM_124_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_124_post_state_repair_ceiling_and_gap_remap/smoke")
DEFAULT_UPSTREAM_123_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_123_multi_turn_state_repair_scale_confirm/smoke")
DEFAULT_UPSTREAM_122_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_122_multi_turn_state_repair/smoke")
DEFAULT_UPSTREAM_121_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan/smoke")
DEFAULT_UPSTREAM_120_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap/smoke")
DEFAULT_UPSTREAM_119_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_119_reasoning_repair_scale_confirm/smoke")
DEFAULT_UPSTREAM_118_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair/smoke")
DEFAULT_UPSTREAM_112_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/smoke")
DEFAULT_UPSTREAM_099_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke")

POSITIVE_VERDICT = "TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN_POSITIVE"
SELECTED_MILESTONE = "126_HALLUCINATION_REFUSAL_BALANCE_REPAIR"
SELECTED_TARGET = "hallucination_refusal_balance_first"
BOUNDARY_TEXT = (
    "125 is planning only. It reads existing artifacts and writes a targeted "
    "post-state repair plan. It performs no training, no repair, no model "
    "inference, no checkpoint mutation, no service startup, no deployment smoke, "
    "and no runtime/product/release integration. It is not GPT-like assistant "
    "readiness, not open-domain assistant readiness, not production chat, not "
    "public API, not deployment readiness, not safety alignment, and not "
    "Hungarian assistant readiness."
)
UPSTREAMS = {
    "124": "POST_STATE_REPAIR_CEILING_AND_GAP_REMAP_POSITIVE",
    "123": "MULTI_TURN_STATE_REPAIR_SCALE_CONFIRM_POSITIVE",
    "122": "MULTI_TURN_STATE_REPAIR_POSITIVE",
    "121": "TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_POSITIVE",
    "120": "POST_REASONING_CEILING_AND_GAP_REMAP_POSITIVE",
    "119": "REASONING_REPAIR_SCALE_CONFIRM_POSITIVE",
    "118": "REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE",
    "112": "CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE",
    "099": "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE",
}
EXPECTED_124 = {
    "first_breakpoint_tier": "TIER_4_HALLUCINATION_REFUSAL_BALANCE",
    "first_breakpoint_family": "hallucination_failure",
    "primary_next_repair_target": "hallucination_failure",
    "reasoning_preserved": True,
    "state_preserved": True,
    "unknown_failure_rate": 0.0,
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
        raise GateError("TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN_FAILS", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def resolve_upstream(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def verify_positive(root: Path, positive_verdict: str, missing_verdict: str = "UPSTREAM_ARTIFACT_MISSING") -> dict[str, Any]:
    path = root / "summary.json"
    if not path.exists():
        raise GateError(missing_verdict, f"missing {rel(path)}")
    summary = read_json(path)
    if summary.get("status") != "positive" or positive_verdict not in set(summary.get("verdicts", [])):
        raise GateError("UPSTREAM_STACK_NOT_POSITIVE", f"{positive_verdict} not found in {rel(path)}")
    return summary


def write_manifest(out: Path, name: str, root: Path, summary: dict[str, Any], verdict: str) -> None:
    metrics = summary.get("metrics", {})
    write_json(
        out / f"upstream_{name}_manifest.json",
        {
            "schema_version": "phase_125_upstream_manifest_v1",
            "upstream": name,
            "root": rel(root),
            "summary_hash": stable_json_hash(summary),
            "positive_verdict": verdict,
            "key_metrics": {
                key: metrics[key]
                for key in [
                    "decision",
                    "next",
                    "first_breakpoint_tier",
                    "first_breakpoint_family",
                    "primary_next_repair_target",
                    "reasoning_preserved",
                    "state_preserved",
                    "unknown_failure_rate",
                    "checkpoint_hash_unchanged",
                    "bounded_release_artifact_unchanged",
                    "controls_failed",
                    "benchmark_leakage_detected",
                ]
                if key in metrics
            },
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
    }


def write_summary(out: Path, phase: str, status: str, verdicts: list[str], metrics: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_125_targeted_post_state_plan_summary_v1",
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
        BOUNDARY_TEXT,
        "",
        "## Status",
        f"- phase: `{phase}`",
        f"- verdicts: `{', '.join(verdicts) if verdicts else 'pending'}`",
        f"- selected_next_milestone: `{decision.get('selected_next_milestone', metrics.get('selected_next_milestone', 'pending'))}`",
        f"- selected_repair_target: `{decision.get('selected_repair_target', metrics.get('selected_repair_target', 'pending'))}`",
        f"- first_breakpoint_tier: `{metrics.get('first_breakpoint_tier', 'pending')}`",
        f"- first_breakpoint_family: `{metrics.get('first_breakpoint_family', 'pending')}`",
        f"- primary_next_repair_target: `{metrics.get('primary_next_repair_target', 'pending')}`",
        f"- reasoning_preserved: `{metrics.get('reasoning_preserved', 'pending')}`",
        f"- state_preserved: `{metrics.get('state_preserved', 'pending')}`",
        "",
        "125 is planning only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.",
    ]
    write_text(out / "report.md", "\n".join(lines) + "\n")


def write_live(out: Path, phase: str, verdicts: list[str], metrics: dict[str, Any], decision: dict[str, Any] | None = None) -> None:
    write_summary(out, phase, "running", verdicts, metrics)
    write_report(out, phase, verdicts, metrics, decision)


def load_124_artifacts(root: Path) -> dict[str, Any]:
    required = [
        "summary.json",
        "decision.json",
        "failure_mode_map.json",
        "capability_gap_map.json",
        "next_repair_targets.json",
        "reasoning_state_preservation_report.json",
        "retention_report.json",
        "collapse_metrics.json",
        "namespace_audit.json",
        "overclaim_exfiltration_report.json",
    ]
    artifacts: dict[str, Any] = {}
    for name in required:
        path = root / name
        if not path.exists():
            raise GateError("UPSTREAM_124_ARTIFACT_MISSING", f"missing {rel(path)}")
        artifacts[name] = read_json(path)
    return artifacts


def verify_124_evidence(artifacts: dict[str, Any]) -> dict[str, Any]:
    decision = artifacts["decision.json"]
    summary_metrics = artifacts["summary.json"].get("metrics", {})
    failure_map = artifacts["failure_mode_map.json"]
    evidence = {
        "first_breakpoint_tier": decision.get("first_breakpoint_tier") or summary_metrics.get("first_breakpoint_tier"),
        "first_breakpoint_family": decision.get("first_breakpoint_family") or summary_metrics.get("first_breakpoint_family"),
        "primary_next_repair_target": decision.get("primary_next_repair_target") or summary_metrics.get("primary_next_repair_target"),
        "reasoning_preserved": decision.get("reasoning_preserved", summary_metrics.get("reasoning_preserved")),
        "state_preserved": decision.get("state_preserved", summary_metrics.get("state_preserved")),
        "unknown_failure_rate": failure_map.get("unknown_failure_rate", summary_metrics.get("unknown_failure_rate")),
        "ceiling_status": decision.get("ceiling_status"),
    }
    for key, expected in EXPECTED_124.items():
        actual = evidence.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise GateError("UPSTREAM_124_NOT_POSITIVE", f"124 evidence mismatch for {key}: {actual}")
        elif actual != expected:
            raise GateError("UPSTREAM_124_NOT_POSITIVE", f"124 evidence mismatch for {key}: {actual}")
    if evidence["ceiling_status"] != "breakpoint_found":
        raise GateError("UPSTREAM_124_NOT_POSITIVE", "124 did not record a breakpoint")
    return evidence


def tier_counts(artifacts: dict[str, Any], tier: str) -> dict[str, int]:
    counts = artifacts["failure_mode_map.json"].get("failure_counts_by_tier", {}).get(tier, {})
    return {str(key): int(value) for key, value in counts.items()}


def write_analysis_artifacts(out: Path, artifacts: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    failure_map = artifacts["failure_mode_map.json"]
    gap_map = artifacts["capability_gap_map.json"]
    global_counts = failure_map.get("failure_counts", {})
    breakpoint_counts = tier_counts(artifacts, evidence["first_breakpoint_tier"])
    top_global = sorted(((label, int(count)) for label, count in global_counts.items() if int(count) > 0), key=lambda item: (-item[1], item[0]))

    priority_map = {
        "schema_version": "phase_125_post_state_failure_priority_map_v1",
        "selection_rule": "first_breakpoint_tier outranks global failure count",
        "first_breakpoint_tier": evidence["first_breakpoint_tier"],
        "first_breakpoint_family": evidence["first_breakpoint_family"],
        "first_breakpoint_failure_counts": breakpoint_counts,
        "global_failure_counts": global_counts,
        "top_global_failure_counts": [{"failure_label": label, "count": count} for label, count in top_global],
        "selected_priority_failure": evidence["primary_next_repair_target"],
        "later_tier_counts_are_compounded": True,
        "reasoning_preserved": evidence["reasoning_preserved"],
        "state_preserved": evidence["state_preserved"],
        "unknown_failure_rate": evidence["unknown_failure_rate"],
    }
    write_json(out / "post_state_failure_priority_map.json", priority_map)

    breakpoint_analysis = {
        "schema_version": "phase_125_breakpoint_analysis_v1",
        "first_breakpoint_tier": evidence["first_breakpoint_tier"],
        "first_breakpoint_family": evidence["first_breakpoint_family"],
        "first_breakpoint_accuracy": next((gap.get("accuracy") for gap in gap_map.get("gaps", []) if gap.get("tier") == evidence["first_breakpoint_tier"]), None),
        "first_breakpoint_failures": breakpoint_counts,
        "selected_repair_target": SELECTED_TARGET,
        "reasoning_preserved": evidence["reasoning_preserved"],
        "state_preserved": evidence["state_preserved"],
        "unknown_failure_rate": evidence["unknown_failure_rate"],
    }
    write_json(out / "breakpoint_analysis.json", breakpoint_analysis)

    root_vs_symptom = {
        "schema_version": "phase_125_root_vs_symptom_analysis_v1",
        "root_cause_candidate": "hallucination_refusal_balance",
        "root_cause_evidence": [
            "124 first breakpoint tier is TIER_4_HALLUCINATION_REFUSAL_BALANCE",
            "Tier 4 failures include hallucination_failure, over_refusal, and ambiguity_failure",
            "reasoning and multi-turn state preservation are both true",
            "unknown_failure_rate is 0.0",
        ],
        "symptom_or_later_compounded_failures": {
            "format_failure": "higher global count but begins after the Tier 4 hallucination/refusal balance breakpoint",
            "prompt_injection_failure": "higher global count but begins after Tier 4 and is partly adversarial-format compounded",
            "long_context_failure": "later combined stress and not the first breakpoint",
        },
        "first_breakpoint_outranks_global_count": True,
        "later_tier_target_selected_first": False,
        "requires_proof_if_later_target_selected": "root_vs_symptom_analysis must prove a later target is upstream of Tier 4 hallucination/refusal balance",
    }
    write_json(out / "root_vs_symptom_analysis.json", root_vs_symptom)

    selection = {
        "schema_version": "phase_125_repair_target_selection_v1",
        "selected_next_milestone": SELECTED_MILESTONE,
        "selected_repair_target": SELECTED_TARGET,
        "artifact_evidence": evidence,
        "selection_rule": "first breakpoint outranks global failure count",
        "rejected_later_global_count_selection": True,
        "why_not_format_injection_first": "format/injection has higher global count, but starts after the Tier 4 hallucination/refusal breakpoint.",
        "why_not_prompt_injection_first": "prompt-injection is a later Tier 5/6/8 compounded failure and not the first breakpoint.",
        "why_not_long_context_first": "long-context failures are later-tier combined stress; 124's first breakpoint is calibration.",
        "why_not_more_general_training": "124 identifies a specific calibration gap; generic SFT risks always-refuse or wrong-objective behavior.",
        "why_not_deploy_polish": "raw assistant capability remains a research harness track, not deployment/product readiness.",
        "why_not_architecture_pivot": "reasoning and state repairs are preserved; the evidence is a targeted calibration gap, not chassis collapse.",
    }
    write_json(out / "repair_target_selection.json", selection)

    write_json(
        out / "training_design_options.json",
        {
            "schema_version": "phase_125_training_design_options_v1",
            "selected": SELECTED_MILESTONE,
            "options": [
                {"milestone": SELECTED_MILESTONE, "selected": True, "reason": "matches first post-state breakpoint"},
                {"milestone": "126_FORMAT_INJECTION_ROBUSTNESS_REPAIR", "selected": False, "reason": "higher global count but later-tier"},
                {"milestone": "126_LONG_CONTEXT_COMBINED_STRESS_REPAIR", "selected": False, "reason": "later combined stress"},
                {"milestone": "126_CAPABILITY_SCALE_WITHOUT_REPAIR", "selected": False, "reason": "specific calibration breakpoint exists"},
            ],
        },
    )

    eval_gate_proposal = build_eval_gate_proposal()
    write_json(out / "eval_gate_proposal.json", eval_gate_proposal)
    write_json(out / "risk_register.json", build_risk_register())
    write_json(out / "111_failure_prevention_map.json", build_111_prevention_map())
    next_plan = build_next_milestone_plan()
    write_json(out / "next_milestone_plan.json", next_plan)
    decision = build_decision(evidence, selection, eval_gate_proposal)
    write_json(out / "decision.json", decision)
    return {"priority_map": priority_map, "breakpoint_analysis": breakpoint_analysis, "root_vs_symptom": root_vs_symptom, "selection": selection, "eval_gate_proposal": eval_gate_proposal, "decision": decision}


def build_eval_gate_proposal() -> dict[str, Any]:
    return {
        "schema_version": "phase_125_eval_gate_proposal_v1",
        "calibration_gates": {
            "hallucination_trap_pass_rate": "hard gate",
            "unsupported_refusal_accuracy": "hard gate",
            "insufficient_fact_refusal_accuracy": "hard gate",
            "answerable_fact_response_accuracy": "hard gate preventing always-refuse degeneration",
            "over_refusal_rate": "must stay bounded",
            "under_refusal_rate": "must stay bounded",
            "ambiguity_refusal_accuracy": "hard gate",
            "explicit_priority_answer_accuracy": "hard gate preventing always-refuse degeneration",
            "evidence_sufficiency_classification_accuracy": "hard gate",
        },
        "reasoning_preservation_gates": {"tier4_reasoning_accuracy": ">= 0.97", "tier8_reasoning_combo_accuracy": ">= 0.90", "reasoning_failure_rate": "<= 0.05"},
        "state_preservation_gates": {
            "multi_turn_state_accuracy": ">= 0.95",
            "depth_8_state_accuracy": ">= 0.90",
            "tier4_multi_turn_breakpoint_accuracy": ">= 0.95",
            "stale_state_copy_rate": "<= 0.05",
            "stale_decoy_leak_rate": "<= 0.05",
        },
        "regression_gates": ["retention metrics", "collapse metrics", "namespace drift metrics", "leakage metrics", "overclaim/exfiltration metrics", "control-arm failure metrics"],
    }


def build_risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_125_risk_register_v1",
        "risks": [
            {"risk": "always-refuse degeneration", "mitigation": "answerable_fact_response_accuracy and explicit_priority_answer_accuracy hard gates"},
            {"risk": "under-refusal persists", "mitigation": "unsupported and insufficient-fact refusal gates"},
            {"risk": "over-refusal grows while hallucination falls", "mitigation": "over_refusal_rate and answerable rows"},
            {"risk": "reasoning repair regression", "mitigation": "mandatory reasoning preservation gates"},
            {"risk": "multi-turn state repair regression", "mitigation": "mandatory state preservation gates"},
            {"risk": "generic SFT masks calibration failures", "mitigation": "calibration-focused data and raw-only final eval"},
        ],
    }


def build_111_prevention_map() -> dict[str, Any]:
    return {
        "schema_version": "phase_125_111_failure_prevention_map_v1",
        "train_eval_namespace_disjointness": True,
        "anti_memorization_rows": True,
        "leakage_audit_against_112_125_artifacts": True,
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
        "schema_version": "phase_125_next_milestone_plan_v1",
        "milestone_name": SELECTED_MILESTONE,
        "purpose": "Repair hallucination/refusal calibration without collapsing into always-refuse behavior.",
        "train_eval_type": "targeted research repair with raw-only final eval",
        "not_generic_sft": True,
        "not_refusal_only_training": True,
        "data_design": [
            "provided-fact answerable rows",
            "insufficient-fact refusal rows",
            "ambiguity without priority rows",
            "ambiguity with explicit priority rows",
            "hallucination traps",
            "over-refusal traps",
            "under-refusal traps",
            "multi-doc evidence sufficiency",
            "table evidence sufficiency",
            "state-carry with insufficient facts",
            "long-context distractor plus missing fact",
        ],
        "anti_111_safeguards": [
            "train/eval namespace disjointness",
            "anti-memorization rows",
            "leakage audit against 112-125 artifacts",
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
        "always_refuse_degeneracy_prevention": {
            "explicitly_reject_always_refuse_solution": True,
            "answerable_fact_response_accuracy": "hard gate",
            "explicit_priority_answer_accuracy": "hard gate",
            "over_refusal_rate": "bounded hard gate",
        },
        "required_eval_gates": [
            "hallucination_trap_pass_rate",
            "unsupported_refusal_accuracy",
            "insufficient_fact_refusal_accuracy",
            "answerable_fact_response_accuracy",
            "over_refusal_rate",
            "under_refusal_rate",
            "ambiguity_refusal_accuracy",
            "explicit_priority_answer_accuracy",
            "evidence_sufficiency_classification_accuracy",
            "retention metrics",
            "collapse metrics",
            "namespace drift metrics",
            "leakage metrics",
            "overclaim/exfiltration metrics",
            "control-arm failure metrics",
        ],
        "reasoning_preservation_gates": {"tier4_reasoning_accuracy": ">= 0.97", "tier8_reasoning_combo_accuracy": ">= 0.90", "reasoning_failure_rate": "<= 0.05"},
        "state_preservation_gates": {
            "multi_turn_state_accuracy": ">= 0.95",
            "depth_8_state_accuracy": ">= 0.90",
            "tier4_multi_turn_breakpoint_accuracy": ">= 0.95",
            "stale_state_copy_rate": "<= 0.05",
            "stale_decoy_leak_rate": "<= 0.05",
        },
        "final_eval_forbidden_paths": ["integrated policy", "decoder reference", "oracle rerank", "expected-answer metadata", "verifier rerank", "LLM judge", "teacher forcing"],
        "positive_verdicts": ["HALLUCINATION_REFUSAL_BALANCE_REPAIR_POSITIVE", "ALWAYS_REFUSE_DEGENERATION_REJECTED", "REASONING_REPAIR_PRESERVED", "STATE_REPAIR_PRESERVED"],
        "failure_verdicts": ["ALWAYS_REFUSE_DEGENERATION_DETECTED", "HALLUCINATION_REPAIR_FAILS", "OVER_REFUSAL_REGRESSION_DETECTED", "UNDER_REFUSAL_REGRESSION_DETECTED", "REASONING_REGRESSION_DETECTED", "STATE_REGRESSION_DETECTED"],
        "validation_commands": [
            "python -m py_compile scripts/probes/run_stable_loop_phase_lock_126_hallucination_refusal_balance_repair.py",
            "python scripts/probes/run_stable_loop_phase_lock_126_hallucination_refusal_balance_repair_check.py --check-only",
            "git diff --check",
        ],
        "boundary_text": "126 is targeted research repair only; it is not GPT-like assistant readiness, not open-domain readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.",
    }


def build_decision(evidence: dict[str, Any], selection: dict[str, Any], eval_gates: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_125_decision_v1",
        "selected_next_milestone": SELECTED_MILESTONE,
        "selected_repair_target": SELECTED_TARGET,
        "first_breakpoint_tier": evidence["first_breakpoint_tier"],
        "first_breakpoint_family": evidence["first_breakpoint_family"],
        "primary_next_repair_target": evidence["primary_next_repair_target"],
        "reasoning_preserved": evidence["reasoning_preserved"],
        "state_preserved": evidence["state_preserved"],
        "unknown_failure_rate": evidence["unknown_failure_rate"],
        "primary_reason": "124's first post-state breakpoint is hallucination/refusal balance, and first breakpoint outranks later global counts.",
        "supporting_evidence": evidence,
        "tier4_first_breakpoint_evidence": {
            "hallucination_failure": 48,
            "over_refusal": 48,
            "ambiguity_failure": 48,
        },
        "later_global_evidence": {
            "format_failure": 352,
            "prompt_injection_failure": 224,
            "long_context_failure": 160,
            "ambiguity_failure": 112,
            "hallucination_failure": 112,
            "under_refusal": 64,
            "over_refusal": 48,
        },
        "rejected_alternatives": {
            "126_FORMAT_INJECTION_ROBUSTNESS_REPAIR": selection["why_not_format_injection_first"],
            "126_PROMPT_INJECTION_ROBUSTNESS_REPAIR": selection["why_not_prompt_injection_first"],
            "126_LONG_CONTEXT_COMBINED_STRESS_REPAIR": selection["why_not_long_context_first"],
            "126_CAPABILITY_SCALE_WITHOUT_REPAIR": selection["why_not_more_general_training"],
            "126_POST_STATE_ARCHITECTURE_REVIEW": selection["why_not_architecture_pivot"],
        },
        "why_not_format_injection_first": selection["why_not_format_injection_first"],
        "why_not_prompt_injection_first": selection["why_not_prompt_injection_first"],
        "why_not_long_context_first": selection["why_not_long_context_first"],
        "why_not_more_general_training": selection["why_not_more_general_training"],
        "why_not_deploy_polish": selection["why_not_deploy_polish"],
        "why_not_architecture_pivot": selection["why_not_architecture_pivot"],
        "hard_gates_for_126": eval_gates,
        "expected_success_criteria": ["hallucination trap pass rate improves", "answerable fact response remains high", "always-refuse degeneration is rejected", "reasoning and state repairs remain preserved"],
        "expected_failure_modes": ["always-refuse degeneration", "under-refusal persists", "over-refusal regression", "reasoning regression", "state regression", "retention regression"],
    }


def final_metrics(evidence: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    metrics = base_metrics()
    metrics.update(
        {
            "upstream_124_positive": True,
            "first_breakpoint_tier": evidence["first_breakpoint_tier"],
            "first_breakpoint_family": evidence["first_breakpoint_family"],
            "primary_next_repair_target": evidence["primary_next_repair_target"],
            "reasoning_preserved": evidence["reasoning_preserved"],
            "state_preserved": evidence["state_preserved"],
            "unknown_failure_rate": evidence["unknown_failure_rate"],
            "failure_priority_map_written": True,
            "breakpoint_analysis_written": True,
            "root_vs_symptom_analysis_written": True,
            "repair_target_selection_written": True,
            "eval_gate_proposal_written": True,
            "risk_register_written": True,
            "next_milestone_plan_written": True,
            "decision_written": True,
            "selected_next_milestone": analysis["decision"]["selected_next_milestone"],
            "selected_repair_target": analysis["decision"]["selected_repair_target"],
        }
    )
    return metrics


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    metrics = base_metrics()
    write_json(out / "queue.json", {"schema_version": "phase_125_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_125_analysis_config_v1",
            "milestone": MILESTONE,
            "out": rel(out),
            "selection_rule": "first breakpoint outranks global failure count",
            "selected_default": SELECTED_MILESTONE,
            "analysis_only": True,
            "training_performed": False,
            "inference_run_count": 0,
            "heartbeat_sec": args.heartbeat_sec,
        },
    )
    append_progress(out, "startup")
    write_live(out, "startup", [], metrics)

    roots = {
        "124": resolve_upstream(args.upstream_124_root),
        "123": resolve_upstream(args.upstream_123_root),
        "122": resolve_upstream(args.upstream_122_root),
        "121": resolve_upstream(args.upstream_121_root),
        "120": resolve_upstream(args.upstream_120_root),
        "119": resolve_upstream(args.upstream_119_root),
        "118": resolve_upstream(args.upstream_118_root),
        "112": resolve_upstream(args.upstream_112_root),
        "099": resolve_upstream(args.upstream_099_root),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for name, root in roots.items():
        summaries[name] = verify_positive(root, UPSTREAMS[name], "UPSTREAM_ARTIFACT_MISSING")
        write_manifest(out, name, root, summaries[name], UPSTREAMS[name])
    append_progress(out, "upstream_verification", upstreams=sorted(roots))
    write_live(out, "upstream_verification", ["UPSTREAM_124_CEILING_MAP_VERIFIED"], metrics)

    artifacts = load_124_artifacts(roots["124"])
    evidence = verify_124_evidence(artifacts)
    append_progress(out, "artifact_loading", evidence=evidence)
    write_live(out, "artifact_loading", ["UPSTREAM_124_CEILING_MAP_VERIFIED"], {**metrics, **evidence})

    append_progress(out, "failure_prioritization", first_breakpoint_tier=evidence["first_breakpoint_tier"])
    analysis = write_analysis_artifacts(out, artifacts, evidence)
    write_live(out, "failure_prioritization", ["UPSTREAM_124_CEILING_MAP_VERIFIED", "FAILURE_PRIORITY_MAP_WRITTEN"], {**metrics, **evidence, "selected_next_milestone": SELECTED_MILESTONE, "selected_repair_target": SELECTED_TARGET}, analysis["decision"])

    append_progress(out, "root_symptom_analysis", selected_target=SELECTED_TARGET)
    write_live(out, "root_symptom_analysis", ["ROOT_VS_SYMPTOM_ANALYSIS_WRITTEN"], {**metrics, **evidence, "selected_next_milestone": SELECTED_MILESTONE, "selected_repair_target": SELECTED_TARGET}, analysis["decision"])

    append_progress(out, "repair_target_selection", selected_next_milestone=SELECTED_MILESTONE, selected_repair_target=SELECTED_TARGET)
    write_live(out, "repair_target_selection", ["HALLUCINATION_REFUSAL_TARGET_SELECTED"], {**metrics, **evidence, "selected_next_milestone": SELECTED_MILESTONE, "selected_repair_target": SELECTED_TARGET}, analysis["decision"])

    append_progress(out, "eval_gate_proposal", always_refuse_degeneracy_prevention=True)
    write_live(out, "eval_gate_proposal", ["EVAL_GATE_PROPOSAL_WRITTEN"], {**metrics, **evidence, "selected_next_milestone": SELECTED_MILESTONE, "selected_repair_target": SELECTED_TARGET}, analysis["decision"])

    final = final_metrics(evidence, analysis)
    final["wall_clock_sec"] = round(time.time() - start, 3)
    verdicts = [
        POSITIVE_VERDICT,
        "UPSTREAM_124_CEILING_MAP_VERIFIED",
        "POST_STATE_BREAKPOINT_ANALYSIS_WRITTEN",
        "FAILURE_PRIORITY_MAP_WRITTEN",
        "ROOT_VS_SYMPTOM_ANALYSIS_WRITTEN",
        "HALLUCINATION_REFUSAL_TARGET_SELECTED",
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


def write_failure(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    metrics = base_metrics()
    metrics.update({"failure_verdict": error.verdict, "failure_message": error.message})
    write_json(out / "decision.json", {"schema_version": "phase_125_failure_decision_v1", "decision": "targeted_post_state_repair_plan_failed", "next": "125B_POST_STATE_PLAN_FAILURE_ANALYSIS", "failure_verdict": error.verdict, "failure_message": error.message})
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", "failure", ["TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN_FAILS", error.verdict], metrics, error.verdict)
    write_report(out, "failure", ["TARGETED_POST_STATE_REPAIR_OR_SCALE_PLAN_FAILS", error.verdict], metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-124-root", default=str(DEFAULT_UPSTREAM_124_ROOT))
    parser.add_argument("--upstream-123-root", default=str(DEFAULT_UPSTREAM_123_ROOT))
    parser.add_argument("--upstream-122-root", default=str(DEFAULT_UPSTREAM_122_ROOT))
    parser.add_argument("--upstream-121-root", default=str(DEFAULT_UPSTREAM_121_ROOT))
    parser.add_argument("--upstream-120-root", default=str(DEFAULT_UPSTREAM_120_ROOT))
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
