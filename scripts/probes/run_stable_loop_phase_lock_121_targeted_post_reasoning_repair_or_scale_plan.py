#!/usr/bin/env python3
"""121 targeted post-reasoning repair or scale plan.

This analysis-only milestone reads the positive 120 post-reasoning ceiling/gap
remap, selects the next targeted repair milestone, and writes a concrete 122
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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_121_TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_121_targeted_post_reasoning_repair_or_scale_plan/smoke")
DEFAULT_UPSTREAM_120_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_120_post_reasoning_ceiling_and_gap_remap/smoke")
DEFAULT_UPSTREAM_119_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_119_reasoning_repair_scale_confirm/smoke")
DEFAULT_UPSTREAM_118_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_118_reasoning_first_raw_assistant_repair/smoke")
DEFAULT_UPSTREAM_116_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_116_raw_assistant_capability_ceiling_and_gap_map/smoke")
DEFAULT_UPSTREAM_115_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_115_external_style_raw_assistant_stress_confirm/smoke")
DEFAULT_UPSTREAM_112_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/smoke")
DEFAULT_UPSTREAM_099_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke")

POSITIVE_VERDICT = "TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_POSITIVE"
BOUNDARY_TEXT = (
    "121 is planning only. It reads existing artifacts and writes a targeted "
    "post-reasoning repair plan. It performs no training, no repair, no model "
    "inference, no checkpoint mutation, no service startup, no deployment smoke, "
    "and no runtime/product/release integration. It is not GPT-like assistant "
    "readiness, not open-domain assistant readiness, not production chat, not "
    "public API, not deployment readiness, not safety alignment, and not "
    "Hungarian assistant readiness."
)

UPSTREAMS = {
    "120": "POST_REASONING_CEILING_AND_GAP_REMAP_POSITIVE",
    "119": "REASONING_REPAIR_SCALE_CONFIRM_POSITIVE",
    "118": "REASONING_FIRST_RAW_ASSISTANT_REPAIR_POSITIVE",
    "116": "RAW_ASSISTANT_CAPABILITY_CEILING_AND_GAP_MAP_POSITIVE",
    "115": "EXTERNAL_STYLE_RAW_ASSISTANT_STRESS_CONFIRM_POSITIVE",
    "112": "CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE",
    "099": "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE",
}

EXPECTED_120 = {
    "first_breakpoint_tier": "TIER_4_MULTI_TURN_STATE_UPDATE",
    "primary_next_repair_target": "multi_turn_state_failure",
    "reasoning_regression_rejected": True,
    "reasoning_failure_rate": 0.0,
}

SELECTED_MILESTONE = "122_MULTI_TURN_STATE_REPAIR"
SELECTED_TARGET = "multi_turn_state_first"


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


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


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
        raise GateError("TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_FAILS", "--out must stay under target/pilot_wave")
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
            "schema_version": "phase_121_upstream_manifest_v1",
            "upstream": name,
            "root": rel(root),
            "summary_hash": stable_json_hash(summary),
            "positive_verdict": verdict,
            "key_metrics": {
                key: metrics[key]
                for key in [
                    "decision",
                    "next",
                    "eval_count",
                    "first_breakpoint_tier",
                    "primary_next_repair_target",
                    "reasoning_regression_rejected",
                    "reasoning_failure_rate",
                    "tier4_reasoning_accuracy",
                    "tier8_reasoning_combo_accuracy",
                    "controls_failed",
                    "benchmark_leakage_detected",
                    "retention_preserved",
                    "collapse_rejected",
                    "checkpoint_hash_unchanged",
                    "bounded_release_artifact_unchanged",
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
            "schema_version": "phase_121_targeted_post_reasoning_plan_summary_v1",
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
        f"- primary_next_repair_target: `{metrics.get('primary_next_repair_target', 'pending')}`",
        f"- reasoning_failure_rate: `{metrics.get('reasoning_failure_rate', 'pending')}`",
        "",
        "121 is planning only. It is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.",
    ]
    write_text(out / "report.md", "\n".join(lines) + "\n")


def write_live(out: Path, phase: str, verdicts: list[str], metrics: dict[str, Any], decision: dict[str, Any] | None = None) -> None:
    write_summary(out, phase, "running", verdicts, metrics)
    write_report(out, phase, verdicts, metrics, decision)


def load_120_artifacts(root: Path) -> dict[str, Any]:
    required = [
        "summary.json",
        "decision.json",
        "failure_mode_map.json",
        "capability_gap_map.json",
        "post_reasoning_delta_vs_116.json",
        "reasoning_regression_report.json",
        "retention_report.json",
        "collapse_metrics.json",
        "overclaim_exfiltration_report.json",
        "control_arm_report.json",
        "next_repair_targets.json",
    ]
    artifacts: dict[str, Any] = {}
    for name in required:
        path = root / name
        if not path.exists():
            raise GateError("UPSTREAM_120_ARTIFACT_MISSING", f"missing {rel(path)}")
        artifacts[name] = read_json(path)
    artifacts["human_samples"] = read_jsonl(root / "human_readable_samples.jsonl", limit=80)
    artifacts["failure_samples"] = read_jsonl(root / "failure_case_samples.jsonl", limit=200)
    return artifacts


def verify_120_evidence(artifacts: dict[str, Any]) -> dict[str, Any]:
    decision = artifacts["decision.json"]
    summary_metrics = artifacts["summary.json"].get("metrics", {})
    evidence = {
        "first_breakpoint_tier": decision.get("first_breakpoint_tier") or summary_metrics.get("first_breakpoint_tier"),
        "primary_next_repair_target": decision.get("primary_next_repair_target") or summary_metrics.get("primary_next_repair_target"),
        "reasoning_regression_rejected": decision.get("reasoning_regression_rejected", summary_metrics.get("reasoning_regression_rejected")),
        "reasoning_failure_rate": summary_metrics.get("reasoning_failure_rate", artifacts["reasoning_regression_report.json"].get("reasoning_failure_rate")),
        "ceiling_status": decision.get("ceiling_status"),
    }
    for key, expected in EXPECTED_120.items():
        actual = evidence.get(key)
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise GateError("UPSTREAM_120_NOT_POSITIVE", f"120 evidence mismatch for {key}: {actual}")
        elif actual != expected:
            raise GateError("UPSTREAM_120_NOT_POSITIVE", f"120 evidence mismatch for {key}: {actual}")
    if evidence["ceiling_status"] != "breakpoint_found":
        raise GateError("UPSTREAM_120_NOT_POSITIVE", "120 did not record a breakpoint")
    return evidence


def first_breakpoint_counts(artifacts: dict[str, Any], tier: str) -> dict[str, int]:
    counts = artifacts["failure_mode_map.json"].get("failure_counts_by_tier", {}).get(tier, {})
    return {str(key): int(value) for key, value in counts.items()}


def write_analysis_artifacts(out: Path, artifacts: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    failure_map = artifacts["failure_mode_map.json"]
    gap_map = artifacts["capability_gap_map.json"]
    delta = artifacts["post_reasoning_delta_vs_116.json"]
    global_counts = failure_map.get("failure_counts", {})
    breakpoint_counts = first_breakpoint_counts(artifacts, evidence["first_breakpoint_tier"])
    top_global = sorted(
        ((label, int(count)) for label, count in global_counts.items() if int(count) > 0),
        key=lambda item: (-item[1], item[0]),
    )

    priority_map = {
        "schema_version": "phase_121_post_reasoning_failure_priority_map_v1",
        "selection_rule": "first_breakpoint_tier outranks global failure count",
        "first_breakpoint_tier": evidence["first_breakpoint_tier"],
        "first_breakpoint_failure_counts": breakpoint_counts,
        "global_failure_counts": global_counts,
        "top_global_failure_counts": [{"failure_label": label, "count": count} for label, count in top_global],
        "selected_priority_failure": evidence["primary_next_repair_target"],
        "later_tier_counts_are_compounded": True,
        "reasoning_regression_rejected": evidence["reasoning_regression_rejected"],
        "reasoning_failure_rate": evidence["reasoning_failure_rate"],
    }
    write_json(out / "post_reasoning_failure_priority_map.json", priority_map)

    breakpoint_analysis = {
        "schema_version": "phase_121_breakpoint_analysis_v1",
        "first_breakpoint_tier": evidence["first_breakpoint_tier"],
        "first_breakpoint_accuracy": next(
            (gap.get("accuracy") for gap in gap_map.get("gaps", []) if gap.get("tier") == evidence["first_breakpoint_tier"]),
            None,
        ),
        "first_breakpoint_failures": breakpoint_counts,
        "selected_repair_target": SELECTED_TARGET,
        "reasoning_breakpoint_resolved": delta.get("reasoning_breakpoint_resolved") is True,
        "reasoning_failure_count_116": delta.get("reasoning_failure_count_116"),
        "reasoning_failure_count_120": delta.get("reasoning_failure_count_120"),
        "reasoning_regression_rejected": evidence["reasoning_regression_rejected"],
        "reasoning_failure_rate": evidence["reasoning_failure_rate"],
    }
    write_json(out / "breakpoint_analysis.json", breakpoint_analysis)

    root_vs_symptom = {
        "schema_version": "phase_121_root_vs_symptom_analysis_v1",
        "root_cause_candidate": "multi_turn_state_failure",
        "root_cause_evidence": [
            "120 first breakpoint tier is TIER_4_MULTI_TURN_STATE_UPDATE",
            "Tier 4 failures are multi_turn_state_failure only",
            "reasoning regression is rejected and reasoning_failure_rate is 0.0",
        ],
        "symptom_or_later_compounded_failures": {
            "hallucination_refusal_balance": "appears at Tier 5 after the multi-turn breakpoint",
            "format_and_prompt_injection": "appears at Tier 6 after the multi-turn breakpoint",
            "long_context_combined_stress": "dominates later global counts but appears in Tier 7/8 combined stress",
        },
        "first_breakpoint_outranks_global_count": True,
        "later_tier_target_selected_first": False,
        "requires_proof_if_later_target_selected": "root_vs_symptom_analysis must prove a later target is upstream of multi-turn failure",
    }
    write_json(out / "root_vs_symptom_analysis.json", root_vs_symptom)

    selection = {
        "schema_version": "phase_121_repair_target_selection_v1",
        "selected_next_milestone": SELECTED_MILESTONE,
        "selected_repair_target": SELECTED_TARGET,
        "artifact_evidence": evidence,
        "selection_rule": "first breakpoint outranks global failure count",
        "rejected_later_global_count_selection": True,
        "why_not_hallucination_refusal_first": "hallucination/refusal failures begin at Tier 5, after the Tier 4 multi-turn state breakpoint.",
        "why_not_format_injection_first": "format/injection failures begin at Tier 6 and must be guarded, not selected before the first breakpoint.",
        "why_not_long_context_first": "long-context has a higher global count in later combined tiers, but 120 shows it is not the first breakpoint.",
        "why_not_more_general_training": "120 identifies a specific post-reasoning breakpoint; generic SFT risks another 111-style wrong-objective failure.",
        "why_not_deploy_polish": "raw assistant capability remains a research harness track, not a deploy/product readiness track.",
        "why_not_architecture_pivot": "111X/112/119/120 support current-chassis viability; the evidence is a targeted capability gap, not chassis collapse.",
    }
    write_json(out / "repair_target_selection.json", selection)

    training_design_options = {
        "schema_version": "phase_121_training_design_options_v1",
        "selected": SELECTED_MILESTONE,
        "options": [
            {"milestone": "122_MULTI_TURN_STATE_REPAIR", "selected": True, "reason": "matches first post-reasoning breakpoint"},
            {"milestone": "122_HALLUCINATION_REFUSAL_BALANCE_REPAIR", "selected": False, "reason": "later Tier 5 gap; keep as regression gate"},
            {"milestone": "122_FORMAT_INJECTION_ROBUSTNESS_REPAIR", "selected": False, "reason": "later Tier 6 gap; keep as regression gate"},
            {"milestone": "122_LONG_CONTEXT_COMBINED_STRESS_REPAIR", "selected": False, "reason": "higher global count but later compounded stress"},
            {"milestone": "122_CAPABILITY_SCALE_WITHOUT_REPAIR", "selected": False, "reason": "specific breakpoint exists"},
            {"milestone": "122_POST_REASONING_ARCHITECTURE_REVIEW", "selected": False, "reason": "no evidence of architecture collapse"},
        ],
    }
    write_json(out / "training_design_options.json", training_design_options)

    eval_gate_proposal = {
        "schema_version": "phase_121_eval_gate_proposal_v1",
        "required_multi_turn_gates": {
            "multi_turn_state_accuracy": "improves against 120 baseline and meets 122 threshold",
            "state_tracking_accuracy": "improves against 120 baseline",
            "multi_turn_correction_accuracy": "improves against 120 baseline",
            "stale_state_rejection_accuracy": "hard gate",
            "override_chain_accuracy": "hard gate",
            "active_slot_after_update_accuracy": "hard gate",
        },
        "reasoning_preservation_gates": {
            "tier4_reasoning_accuracy": ">= 0.97",
            "tier8_reasoning_combo_accuracy": ">= 0.90",
            "reasoning_failure_rate": "<= 0.05",
        },
        "regression_gates": [
            "retention metrics",
            "collapse metrics",
            "namespace drift metrics",
            "leakage metrics",
            "overclaim/exfiltration metrics",
            "static/copy/random controls fail",
        ],
    }
    write_json(out / "eval_gate_proposal.json", eval_gate_proposal)

    risk_register = {
        "schema_version": "phase_121_risk_register_v1",
        "risks": [
            {"risk": "teacher-forcing-only state success", "mitigation": "scheduled sampling or rollout-style objective if training is used"},
            {"risk": "state namespace memorization", "mitigation": "fresh namespace splits and anti-memorization rows"},
            {"risk": "reasoning repair regression", "mitigation": "mandatory reasoning preservation gates"},
            {"risk": "overfitting stale-state templates", "mitigation": "varied correction depths, override orders, and decoy positions"},
            {"risk": "long-context global counts distract from first breakpoint", "mitigation": "first-breakpoint outranks global count rule"},
        ],
    }
    write_json(out / "risk_register.json", risk_register)

    prevention = {
        "schema_version": "phase_121_111_failure_prevention_map_v1",
        "train_eval_namespace_disjointness": True,
        "anti_memorization_rows": True,
        "leakage_audit_against_112_121_artifacts": True,
        "scheduled_sampling_or_rollout_style_objective_if_training_is_used": True,
        "raw_only_final_eval": True,
        "no_teacher_forcing_only_success": True,
        "no_oracle_rerank": True,
        "no_expected_answer_metadata": True,
        "no_decoder_reference": True,
        "no_integrated_policy_during_final_eval": True,
    }
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
        "eval_gate_proposal": eval_gate_proposal,
        "decision": decision,
    }


def build_next_milestone_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_121_next_milestone_plan_v1",
        "milestone_name": SELECTED_MILESTONE,
        "purpose": "Repair the first post-reasoning breakpoint: multi-turn state tracking under corrections, overrides, stale-state decoys, and state carry.",
        "train_eval_type": "targeted research repair with raw-only final eval",
        "upstreams": [
            "121 positive",
            "120 positive",
            "119 positive",
            "118 positive",
            "112 positive",
            "099 positive",
        ],
        "data_design": [
            "multi-turn corrections",
            "active vs stale state tracking",
            "override chains",
            "slot updates across turns",
            "table/doc facts plus state updates",
            "bounded refusal with state carry",
            "stale-state decoys",
        ],
        "anti_leakage_rules": [
            "fresh train/eval rows",
            "train/eval namespace disjointness",
            "exact and near-duplicate leakage audit against 112-121 artifacts",
            "disjoint row hashes across train/eval",
        ],
        "anti_memorization_rules": [
            "anti-memorization rows",
            "randomized namespaces",
            "anti-copy rows",
            "stale-state decoys",
            "case-id and active-slot copy gates",
        ],
        "objective_guardrails": [
            "scheduled sampling or rollout-style objective if training is used",
            "no teacher-forcing-only success",
            "raw rollout state metrics must improve",
            "generic SFT is not sufficient for the selected 122 plan",
        ],
        "final_eval_forbidden_paths": [
            "integrated policy",
            "decoder reference",
            "oracle rerank",
            "expected-answer metadata",
            "verifier rerank",
            "LLM judge",
            "teacher forcing",
        ],
        "required_eval_gates": [
            "multi_turn_state_accuracy",
            "state_tracking_accuracy",
            "multi_turn_correction_accuracy",
            "stale_state_rejection_accuracy",
            "override_chain_accuracy",
            "active_slot_after_update_accuracy",
            "retention metrics",
            "collapse metrics",
            "namespace drift metrics",
            "leakage metrics",
        ],
        "reasoning_preservation_gates": {
            "tier4_reasoning_accuracy": ">= 0.97",
            "tier8_reasoning_combo_accuracy": ">= 0.90",
            "reasoning_failure_rate": "<= 0.05",
        },
        "retention_gates": [
            "bounded chat retention >= 0.90",
            "finite-label AnchorRoute retention >= 0.90",
            "unsupported refusal retention >= 0.80",
        ],
        "collapse_gates": [
            "empty output <= 0.02",
            "static output <= 0.10",
            "repetition <= 0.20",
            "copy prompt <= 0.15",
        ],
        "positive_verdicts": [
            "MULTI_TURN_STATE_REPAIR_POSITIVE",
            "MULTI_TURN_STATE_BREAKPOINT_IMPROVED",
            "REASONING_REPAIR_PRESERVED",
            "RETENTION_PRESERVED",
            "COLLAPSE_REJECTED",
            "NO_OVERCLAIM",
        ],
        "failure_verdicts": [
            "MULTI_TURN_STATE_REPAIR_FAILS",
            "TRAIN_EVAL_LEAKAGE_DETECTED",
            "NAMESPACE_MEMORIZATION_DETECTED",
            "TEACHER_FORCING_ONLY_SUCCESS_DETECTED",
            "REASONING_PRESERVATION_GATE_MISSING",
            "RETENTION_REGRESSION_DETECTED",
            "COLLAPSE_DETECTED",
        ],
        "validation_commands": [
            "python -m py_compile scripts/probes/run_stable_loop_phase_lock_122_multi_turn_state_repair.py",
            "python scripts/probes/run_stable_loop_phase_lock_122_multi_turn_state_repair_check.py --check-only",
            "git diff --check",
        ],
        "boundary_text": "122 is targeted research repair only; it is not GPT-like assistant readiness, not open-domain readiness, not production chat, not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.",
    }


def build_decision(evidence: dict[str, Any], selection: dict[str, Any], eval_gates: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_121_decision_v1",
        "selected_next_milestone": SELECTED_MILESTONE,
        "selected_repair_target": SELECTED_TARGET,
        "first_breakpoint_tier": evidence["first_breakpoint_tier"],
        "primary_next_repair_target": evidence["primary_next_repair_target"],
        "reasoning_regression_rejected": evidence["reasoning_regression_rejected"],
        "reasoning_failure_rate": evidence["reasoning_failure_rate"],
        "primary_reason": "120's first post-reasoning breakpoint is multi-turn state update, and first breakpoint outranks later global failure counts.",
        "supporting_evidence": evidence,
        "rejected_alternatives": {
            "122_HALLUCINATION_REFUSAL_BALANCE_REPAIR": selection["why_not_hallucination_refusal_first"],
            "122_FORMAT_INJECTION_ROBUSTNESS_REPAIR": selection["why_not_format_injection_first"],
            "122_LONG_CONTEXT_COMBINED_STRESS_REPAIR": selection["why_not_long_context_first"],
            "122_CAPABILITY_SCALE_WITHOUT_REPAIR": selection["why_not_more_general_training"],
            "122_POST_REASONING_ARCHITECTURE_REVIEW": selection["why_not_architecture_pivot"],
        },
        "why_not_hallucination_refusal_first": selection["why_not_hallucination_refusal_first"],
        "why_not_format_injection_first": selection["why_not_format_injection_first"],
        "why_not_long_context_first": selection["why_not_long_context_first"],
        "why_not_more_general_training": selection["why_not_more_general_training"],
        "why_not_deploy_polish": selection["why_not_deploy_polish"],
        "why_not_architecture_pivot": selection["why_not_architecture_pivot"],
        "hard_gates_for_122": eval_gates,
        "expected_success_criteria": [
            "Tier 4 multi-turn state update improves against 120 baseline",
            "state tracking and multi-turn correction improve",
            "stale-state rejection and override-chain accuracy pass",
            "reasoning repair remains preserved",
            "retention/collapse/boundary gates remain clean",
        ],
        "expected_failure_modes": [
            "teacher-forcing-only state success",
            "namespace memorization",
            "reasoning regression",
            "retention regression",
            "collapse",
            "long-context combined stress remains unresolved",
        ],
    }


def final_metrics(evidence: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    metrics = base_metrics()
    metrics.update(
        {
            "upstream_120_positive": True,
            "first_breakpoint_tier": evidence["first_breakpoint_tier"],
            "primary_next_repair_target": evidence["primary_next_repair_target"],
            "reasoning_regression_rejected": evidence["reasoning_regression_rejected"],
            "reasoning_failure_rate": evidence["reasoning_failure_rate"],
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
    write_json(
        out / "queue.json",
        {
            "schema_version": "phase_121_queue_v1",
            "milestone": MILESTONE,
            "status": "started",
            "started_at": utc_now(),
            "heartbeat_sec": args.heartbeat_sec,
        },
    )
    write_json(
        out / "analysis_config.json",
        {
            "schema_version": "phase_121_analysis_config_v1",
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
        "120": resolve_upstream(args.upstream_120_root),
        "119": resolve_upstream(args.upstream_119_root),
        "118": resolve_upstream(args.upstream_118_root),
        "116": resolve_upstream(args.upstream_116_root),
        "115": resolve_upstream(args.upstream_115_root),
        "112": resolve_upstream(args.upstream_112_root),
        "099": resolve_upstream(args.upstream_099_root),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for name, root in roots.items():
        summaries[name] = verify_positive(root, UPSTREAMS[name], "UPSTREAM_ARTIFACT_MISSING")
        write_manifest(out, name, root, summaries[name], UPSTREAMS[name])
    append_progress(out, "upstream_verification", upstreams=sorted(roots))
    write_live(out, "upstream_verification", ["UPSTREAM_120_POST_REASONING_MAP_VERIFIED"], metrics)

    artifacts = load_120_artifacts(roots["120"])
    evidence = verify_120_evidence(artifacts)
    append_progress(out, "artifact_loading", evidence=evidence)
    write_live(out, "artifact_loading", ["UPSTREAM_120_POST_REASONING_MAP_VERIFIED"], {**metrics, **evidence})

    append_progress(out, "failure_prioritization", first_breakpoint_tier=evidence["first_breakpoint_tier"])
    analysis = write_analysis_artifacts(out, artifacts, evidence)
    write_live(
        out,
        "failure_prioritization",
        ["UPSTREAM_120_POST_REASONING_MAP_VERIFIED", "FAILURE_PRIORITY_MAP_WRITTEN"],
        {**metrics, **evidence, "selected_next_milestone": SELECTED_MILESTONE, "selected_repair_target": SELECTED_TARGET},
        analysis["decision"],
    )

    append_progress(out, "root_symptom_analysis", selected_target=SELECTED_TARGET)
    write_live(
        out,
        "root_symptom_analysis",
        ["ROOT_VS_SYMPTOM_ANALYSIS_WRITTEN"],
        {**metrics, **evidence, "selected_next_milestone": SELECTED_MILESTONE, "selected_repair_target": SELECTED_TARGET},
        analysis["decision"],
    )

    append_progress(out, "repair_target_selection", selected_next_milestone=SELECTED_MILESTONE, selected_repair_target=SELECTED_TARGET)
    write_live(
        out,
        "repair_target_selection",
        ["MULTI_TURN_STATE_REPAIR_TARGET_SELECTED"],
        {**metrics, **evidence, "selected_next_milestone": SELECTED_MILESTONE, "selected_repair_target": SELECTED_TARGET},
        analysis["decision"],
    )

    append_progress(out, "eval_gate_proposal", reasoning_preservation_gates=True)
    write_live(
        out,
        "eval_gate_proposal",
        ["EVAL_GATE_PROPOSAL_WRITTEN"],
        {**metrics, **evidence, "selected_next_milestone": SELECTED_MILESTONE, "selected_repair_target": SELECTED_TARGET},
        analysis["decision"],
    )

    final = final_metrics(evidence, analysis)
    final["wall_clock_sec"] = round(time.time() - start, 3)
    verdicts = [
        POSITIVE_VERDICT,
        "UPSTREAM_120_POST_REASONING_MAP_VERIFIED",
        "POST_REASONING_BREAKPOINT_ANALYSIS_WRITTEN",
        "FAILURE_PRIORITY_MAP_WRITTEN",
        "ROOT_VS_SYMPTOM_ANALYSIS_WRITTEN",
        "MULTI_TURN_STATE_REPAIR_TARGET_SELECTED",
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
    write_json(
        out / "decision.json",
        {
            "schema_version": "phase_121_failure_decision_v1",
            "decision": "targeted_post_reasoning_repair_plan_failed",
            "next": "121B_POST_REASONING_PLAN_FAILURE_ANALYSIS",
            "failure_verdict": error.verdict,
            "failure_message": error.message,
        },
    )
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", "failure", ["TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_FAILS", error.verdict], metrics, error.verdict)
    write_report(out, "failure", ["TARGETED_POST_REASONING_REPAIR_OR_SCALE_PLAN_FAILS", error.verdict], metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-120-root", default=str(DEFAULT_UPSTREAM_120_ROOT))
    parser.add_argument("--upstream-119-root", default=str(DEFAULT_UPSTREAM_119_ROOT))
    parser.add_argument("--upstream-118-root", default=str(DEFAULT_UPSTREAM_118_ROOT))
    parser.add_argument("--upstream-116-root", default=str(DEFAULT_UPSTREAM_116_ROOT))
    parser.add_argument("--upstream-115-root", default=str(DEFAULT_UPSTREAM_115_ROOT))
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
