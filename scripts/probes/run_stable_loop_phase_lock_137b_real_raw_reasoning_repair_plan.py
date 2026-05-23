#!/usr/bin/env python3
"""137B real-raw reasoning repair plan.

Planning-only forensic diagnosis after the 137R clean negative. This runner
reads existing artifacts only; it does not train, repair, run model inference,
call the shared helper for generation, or mutate checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_137B_REAL_RAW_REASONING_REPAIR_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_137b_real_raw_reasoning_repair_plan/smoke")
DEFAULT_UPSTREAM_137R_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_137r_real_raw_reasoning_rebuild/smoke")
DEFAULT_UPSTREAM_136R_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_136r_real_raw_core_capability_minimal_rebuild/smoke")
DEFAULT_UPSTREAM_135E_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_135e_shared_raw_generation_helper_and_canary_gate/smoke")
DEFAULT_UPSTREAM_135D_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_135d_global_raw_evidence_rebuild_plan/smoke")
EXPECTED_DECISION = "real_raw_reasoning_repair_plan_complete"
EXPECTED_NEXT = "138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE"
BOUNDARY_TEXT = (
    "137B is planning only. It diagnoses the 137R clean negative and writes the "
    "next repair/probe plan. It does not train, repair, run new model inference, "
    "call shared_raw_generation_helper.py for new generations, mutate "
    "checkpoints, modify helper/backend code, delete files, consolidate old "
    "runners, start services, deploy, modify runtime/release/product surfaces, "
    "or change root LICENSE. It does not restore reasoning, raw assistant "
    "capability, structured/tool capability, GPT-like readiness, open-domain "
    "assistant readiness, production readiness, deployment readiness, public "
    "API readiness, or safety alignment."
)
REQUIRED_137R_ARTIFACTS = [
    "summary.json",
    "decision.json",
    "aggregate_metrics.json",
    "per_family_metrics.json",
    "per_seed_metrics.jsonl",
    "raw_generation_results.jsonl",
    "raw_generation_trace.jsonl",
    "scoring_results.jsonl",
    "control_results.jsonl",
    "generated_before_scoring_report.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "helper_provenance_verification.json",
    "freshness_leakage_audit.json",
    "failure_case_samples.jsonl",
    "reasoning_dataset.jsonl",
]
FALSE_BOUNDARY_FLAGS = {
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


class GateError(Exception):
    def __init__(self, verdict: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.verdict = verdict
        self.message = message
        self.details = details or {}


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("137B_BOUNDARY_FAILURE", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("137B_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_137b_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "planning_only": True,
            "training_performed": False,
            "repair_performed": False,
            "new_model_inference_run": False,
            "shared_helper_called_for_new_generation": False,
            "checkpoint_mutated": False,
            "runtime_surface_mutated": False,
            "root_license_changed": False,
            **FALSE_BOUNDARY_FLAGS,
            "metrics": decision,
        },
    )


def write_report(out: Path, verdicts: list[str], decision: dict[str, Any]) -> None:
    lines = [
        f"# {MILESTONE}",
        "",
        "## Boundary",
        "",
        BOUNDARY_TEXT,
        "",
        "## Verdicts",
        "",
    ]
    lines.extend(f"- `{verdict}`" for verdict in verdicts)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `decision`: `{decision.get('decision')}`",
            f"- `next`: `{decision.get('next')}`",
            f"- `primary_diagnosis`: `{decision.get('primary_diagnosis')}`",
            f"- `checkpoint_capability_gap_likelihood`: `{decision.get('checkpoint_capability_gap_likelihood')}`",
            "",
            "137B is planning only.",
            "Reasoning is not restored.",
            "Raw assistant capability remains quarantined.",
            "Structured/tool capability remains invalidated as model evidence.",
            "Not GPT-like readiness.",
            "Not open-domain assistant readiness.",
            "Not production chat.",
            "Not public API.",
            "Not deployment readiness.",
            "Not safety alignment.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def refresh_status(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision)


def verify_upstreams(out: Path, root_137r: Path, root_136r: Path, root_135e: Path, root_135d: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = [name for name in REQUIRED_137R_ARTIFACTS if not (root_137r / name).exists()]
    if missing:
        raise GateError("137B_UPSTREAM_137R_ARTIFACT_INCOMPLETE", "137R artifacts missing", {"missing": missing})
    decision_137r = read_json(root_137r / "decision.json")
    aggregate_137r = read_json(root_137r / "aggregate_metrics.json")
    canary_137r = read_json(root_137r / "expected_output_canary_report.json")
    ast_137r = read_json(root_137r / "ast_shortcut_scan_report.json")
    controls_137r = read_json(root_137r / "control_arm_report.json")
    leakage_137r = read_json(root_137r / "freshness_leakage_audit.json")
    provenance_137r = read_json(root_137r / "helper_provenance_verification.json")
    before_137r = read_json(root_137r / "generated_before_scoring_report.json")
    if decision_137r.get("verdict") != "REAL_RAW_REASONING_REBUILD_FAILS":
        raise GateError("137B_UPSTREAM_137R_ARTIFACT_INCOMPLETE", "137R verdict is not clean negative")
    if decision_137r.get("decision") != "real_raw_reasoning_not_restored" or decision_137r.get("next") != "137B_REAL_RAW_REASONING_REPAIR_PLAN":
        raise GateError("137B_UPSTREAM_137R_ARTIFACT_INCOMPLETE", "137R decision route mismatch")
    required_truth = [
        aggregate_137r.get("mean_real_raw_reasoning_accuracy") == 0.0,
        canary_137r.get("expected_output_canary_passed") is True,
        ast_137r.get("ast_shortcut_scan_passed") is True,
        controls_137r.get("controls_failed") is True,
        leakage_137r.get("leakage_rejected") is True,
        provenance_137r.get("checkpoint_hash_unchanged") is True,
        before_137r.get("helper_requests_built_without_expected_or_scorer_metadata") is True,
        decision_137r.get("expected_output_used_for_generation") is False,
        decision_137r.get("scorer_metadata_used_for_generation") is False,
    ]
    if not all(required_truth):
        raise GateError("137B_UPSTREAM_137R_ARTIFACT_INCOMPLETE", "137R clean-negative guardrail signals missing")

    decision_136r = read_json(root_136r / "decision.json")
    decision_135e = read_json(root_135e / "decision.json")
    decision_135d = read_json(root_135d / "decision.json")
    if decision_136r.get("verdict") != "REAL_RAW_CORE_CAPABILITY_MINIMAL_REBUILD_POSITIVE":
        raise GateError("UPSTREAM_136R_NOT_POSITIVE", "136R not positive")
    if decision_135e.get("verdict") != "SHARED_RAW_GENERATION_HELPER_AND_CANARY_GATE_POSITIVE":
        raise GateError("UPSTREAM_135E_NOT_POSITIVE", "135E not positive")
    if decision_135d.get("decision") != "global_raw_evidence_rebuild_plan_complete":
        raise GateError("UPSTREAM_135D_NOT_POSITIVE", "135D not complete")

    manifest_137r = {
        "schema_version": "phase_137b_upstream_137r_manifest_v1",
        "upstream_137r_root": rel(root_137r),
        "upstream_137r_clean_negative_verified": True,
        "verdict": decision_137r.get("verdict"),
        "decision": decision_137r.get("decision"),
        "next": decision_137r.get("next"),
        "mean_real_raw_reasoning_accuracy": aggregate_137r.get("mean_real_raw_reasoning_accuracy"),
        "canary_passed": True,
        "ast_scan_passed": True,
        "controls_failed": True,
        "leakage_rejected": True,
        "checkpoint_hash_unchanged": True,
        "no_expected_or_scorer_metadata_reached_generation": True,
    }
    manifest_136r = {
        "schema_version": "phase_137b_upstream_136r_manifest_v1",
        "upstream_136r_root": rel(root_136r),
        "upstream_136r_verified": True,
        "upstream_135e_root": rel(root_135e),
        "upstream_135e_verified": True,
        "upstream_135d_root": rel(root_135d),
        "upstream_135d_verified": True,
        "verdict_136r": decision_136r.get("verdict"),
        "verdict_135e": decision_135e.get("verdict"),
        "decision_135d": decision_135d.get("decision"),
    }
    write_json(out / "upstream_137r_manifest.json", manifest_137r)
    write_json(out / "upstream_136r_manifest.json", manifest_136r)
    return manifest_137r, manifest_136r


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def token_jaccard(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def generation_quality_report(root_137r: Path) -> dict[str, Any]:
    raw = read_jsonl(root_137r / "raw_generation_results.jsonl")
    traces = read_jsonl(root_137r / "raw_generation_trace.jsonl")
    rows = {row["row_id"]: row for row in read_jsonl(root_137r / "reasoning_dataset.jsonl")}
    trace_by_row = {row["row_id"]: row for row in traces}
    generated = [str(row.get("generated_text", "")) for row in raw]
    token_counts = [int(trace_by_row[row["row_id"]]["response"].get("token_count", 0)) for row in raw if row["row_id"] in trace_by_row]
    row_count = len(raw)
    nonempty = [bool(text.strip()) for text in generated]
    output_hashes = [row.get("generated_text_hash") for row in raw]
    stale_fragments = [bool(re.search(r"(^|\n)\s*(User|Assistant):", text)) for text in generated]
    expected_inclusion = []
    numeric_inclusion = []
    prompt_copy = []
    distractor_copy = []
    refusal_fragments = []
    policy_fragments = []
    utf8_replacements = []
    off_prompt = []
    for item, text in zip(raw, generated):
        source = rows[item["row_id"]]
        expected = source["expected_output"]
        prompt = source["prompt"]
        expected_inclusion.append(expected in text)
        expected_nums = numeric_tokens(expected)
        generated_nums = numeric_tokens(text)
        numeric_inclusion.append(bool(expected_nums) and expected_nums.issubset(generated_nums))
        prompt_copy.append(token_jaccard(prompt, text) >= 0.50)
        distractor_copy.append(source.get("forbidden_distractor", "") in text)
        lower = text.lower()
        refusal_fragments.append(any(token in lower for token in ["unsupported", "insufficient", "cannot", "refuse", "unanswerable"]))
        policy_fragments.append(any(token in lower for token in ["policy", "system override", "hidden policy", "gpt-like", "ignore limits"]))
        utf8_replacements.append("\ufffd" in text)
        off_prompt.append((not expected_inclusion[-1]) and (stale_fragments[-1] or token_jaccard(prompt, text) < 0.20))
    unique_count = len(set(output_hashes))
    return {
        "schema_version": "phase_137b_generation_quality_report_v1",
        "source_artifact": "137R raw_generation_results.jsonl",
        "row_count": row_count,
        "generated_text_exists_rate": sum(text is not None for text in generated) / row_count,
        "nonempty_rate": sum(nonempty) / row_count,
        "token_count_min": min(token_counts) if token_counts else 0,
        "token_count_mean": mean(token_counts) if token_counts else 0.0,
        "token_count_max": max(token_counts) if token_counts else 0,
        "unique_output_hash_count": unique_count,
        "repeated_output_rate": 1.0 - (unique_count / row_count if row_count else 0.0),
        "stale_user_assistant_fragment_rate": sum(stale_fragments) / row_count,
        "expected_token_inclusion_rate": sum(expected_inclusion) / row_count,
        "numeric_expected_token_inclusion_rate": sum(numeric_inclusion) / row_count,
        "prompt_copy_rate": sum(prompt_copy) / row_count,
        "distractor_copy_rate": sum(distractor_copy) / row_count,
        "refusal_fragment_rate": sum(refusal_fragments) / row_count,
        "policy_fragment_rate": sum(policy_fragments) / row_count,
        "utf8_replacement_rate": sum(utf8_replacements) / row_count,
        "off_prompt_output_rate": sum(off_prompt) / row_count,
        "artifact_derived_only": True,
    }


def scoring_mismatch_report(root_137r: Path, quality: dict[str, Any]) -> dict[str, Any]:
    scoring = read_jsonl(root_137r / "scoring_results.jsonl")
    raw = {row["row_id"]: row for row in read_jsonl(root_137r / "raw_generation_results.jsonl")}
    near_matches = []
    for row in scoring:
        generated = raw[row["row_id"]]["generated_text"]
        expected = row["expected_output"]
        near_matches.append(token_jaccard(generated, expected) >= 0.35)
    controls = read_json(root_137r / "control_arm_report.json")
    leakage = read_json(root_137r / "freshness_leakage_audit.json")
    expected_rate = quality["expected_token_inclusion_rate"]
    near_rate = sum(near_matches) / len(near_matches) if near_matches else 0.0
    strict_but_valid = expected_rate <= 0.01 and near_rate <= 0.05 and controls.get("controls_failed") is True and leakage.get("leakage_rejected") is True
    return {
        "schema_version": "phase_137b_scoring_mismatch_report_v1",
        "expected_token_inclusion_rate": expected_rate,
        "near_match_rate": near_rate,
        "controls_failed": controls.get("controls_failed"),
        "leakage_rejected": leakage.get("leakage_rejected"),
        "scoring_too_strict_but_directionally_correct_output": False,
        "scoring_strict_but_valid": strict_but_valid,
        "route_if_near_match_nontrivial": "137E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS" if near_rate > 0.05 else None,
        "primary_scorer_issue_detected": not strict_but_valid,
    }


def reasoning_failure_diagnosis(root_137r: Path, quality: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    decision = read_json(root_137r / "decision.json")
    leakage = read_json(root_137r / "freshness_leakage_audit.json")
    controls = read_json(root_137r / "control_arm_report.json")
    provenance = read_json(root_137r / "helper_provenance_verification.json")
    helper_integrity_passed = (
        decision.get("expected_output_canary_passed") is True
        and decision.get("ast_shortcut_scan_passed") is True
        and decision.get("generated_text_produced_before_scoring") is True
        and provenance.get("checkpoint_hash_unchanged") is True
    )
    checkpoint_gap = (
        helper_integrity_passed
        and controls.get("controls_failed") is True
        and leakage.get("leakage_rejected") is True
        and quality["expected_token_inclusion_rate"] <= 0.01
        and scoring["near_match_rate"] <= 0.05
        and (quality["off_prompt_output_rate"] >= 0.50 or quality["stale_user_assistant_fragment_rate"] >= 0.50)
    )
    return {
        "schema_version": "phase_137b_reasoning_failure_diagnosis_v1",
        "helper_integrity_failure": not helper_integrity_passed,
        "scorer_task_weakness": scoring["primary_scorer_issue_detected"],
        "leakage_eval_contamination": leakage.get("leakage_rejected") is not True,
        "checkpoint_model_capability_gap": checkpoint_gap,
        "prompt_distribution_mismatch": quality["off_prompt_output_rate"] >= 0.50,
        "decoding_config_mismatch": quality["repeated_output_rate"] >= 0.90,
        "byte_level_unreadable_output": quality["utf8_replacement_rate"] > 0.10,
        "exact_match_too_strict_but_directionally_correct": scoring["scoring_too_strict_but_directionally_correct_output"],
        "primary_diagnosis": "checkpoint_model_capability_gap_with_prompt_distribution_mismatch" if checkpoint_gap else "non_model_issue_requires_review",
        "evidence": {
            "mean_real_raw_reasoning_accuracy": decision.get("mean_real_raw_reasoning_accuracy"),
            "helper_integrity_passed": helper_integrity_passed,
            "controls_failed": controls.get("controls_failed"),
            "leakage_rejected": leakage.get("leakage_rejected"),
            "expected_token_inclusion_rate": quality["expected_token_inclusion_rate"],
            "near_match_rate": scoring["near_match_rate"],
            "off_prompt_output_rate": quality["off_prompt_output_rate"],
            "stale_user_assistant_fragment_rate": quality["stale_user_assistant_fragment_rate"],
        },
    }


def checkpoint_gap_report(root_137r: Path, diagnosis: dict[str, Any], quality: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    provenance = read_json(root_137r / "helper_provenance_verification.json")
    likelihood = "high" if diagnosis["checkpoint_model_capability_gap"] else "not_primary"
    return {
        "schema_version": "phase_137b_checkpoint_capability_gap_report_v1",
        "checkpoint_capability_gap_likelihood": likelihood,
        "selected_checkpoint_path": provenance.get("selected_checkpoint_path"),
        "selected_checkpoint_sha256": provenance.get("selected_checkpoint_sha256"),
        "backend_name": provenance.get("backend_name"),
        "backend_load_status": provenance.get("backend_load_status"),
        "helper_integrity_passed": not diagnosis["helper_integrity_failure"],
        "controls_failed": True,
        "leakage_rejected": True,
        "expected_token_inclusion_rate": quality["expected_token_inclusion_rate"],
        "near_match_rate": scoring["near_match_rate"],
        "off_prompt_output_rate": quality["off_prompt_output_rate"],
        "evidence_supports_model_or_checkpoint_gap": likelihood == "high",
    }


def repair_option_matrix(primary_diagnosis: str) -> dict[str, Any]:
    options = [
        {
            "option": "targeted_real_raw_reasoning_training",
            "recommended": primary_diagnosis == "checkpoint_model_capability_gap_with_prompt_distribution_mismatch",
            "benefit": "Directly addresses absent reasoning behavior under real raw generation.",
            "risk": "Can overfit unless final eval uses shared helper, canary, controls, and leakage gates.",
            "next": "138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE",
        },
        {
            "option": "prompt_format_alignment",
            "recommended": False,
            "benefit": "Can test whether prompts are outside the checkpoint distribution.",
            "risk": "May hide weak reasoning if prompts are simplified too far.",
            "next": "138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE",
        },
        {
            "option": "generation_config_review_without_helper_changes",
            "recommended": False,
            "benefit": "Can isolate deterministic decoding mismatch.",
            "risk": "Must not alter helper or use best-of-n/retry loops.",
            "next": "138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE",
        },
        {
            "option": "provenance_safe_checkpoint_selection_review",
            "recommended": False,
            "benefit": "Can test if selected checkpoint is wrong for reasoning.",
            "risk": "Checkpoint swaps need strict provenance and cannot be hidden in eval.",
            "next": "138R_REAL_RAW_REASONING_REPAIR_TRAINING_PLAN_OR_PROBE",
        },
        {
            "option": "eval_design_repair",
            "recommended": primary_diagnosis != "checkpoint_model_capability_gap_with_prompt_distribution_mismatch",
            "benefit": "Appropriate if near matches or scorer weakness appear.",
            "risk": "Not supported by current 0% exact/near evidence unless metrics change.",
            "next": "137C_REAL_RAW_REASONING_EVAL_DESIGN_REPAIR",
        },
    ]
    return {"schema_version": "phase_137b_repair_option_matrix_v1", "options": options}


def next_milestone_plan() -> dict[str, Any]:
    return {
        "schema_version": "phase_137b_next_milestone_plan_v1",
        "milestone": EXPECTED_NEXT,
        "purpose": "Plan or run a bounded targeted repair/probe for real-raw reasoning under the shared helper trust root.",
        "repair_hypothesis": "The 102 byte-GRU checkpoint can generate text but lacks the reasoning/prompt-following behavior needed for 137R-style reasoning rows.",
        "train_eval_distinction": "Training/probe work may update only a target checkpoint under target/, while final eval must use shared_raw_generation_helper.py and generated-before-scoring proof.",
        "allowed_target_checkpoint_path": "target/pilot_wave/stable_loop_phase_lock_138r_real_raw_reasoning_repair_training_plan_or_probe/",
        "source_checkpoint_immutability": True,
        "raw_helper_final_eval": True,
        "generated_before_scoring_proof": True,
        "anti_oracle_canary": True,
        "ast_shortcut_scan": True,
        "leakage_audit": True,
        "scorer_controls": ["STATIC_OUTPUT_CONTROL", "COPY_PROMPT_CONTROL", "RANDOM_ANSWER_CONTROL", "DISTRACTOR_COPY_CONTROL"],
        "success_gates": [
            "shared helper only",
            "generated_text before scoring",
            "canary passes",
            "AST scan passes",
            "controls fail",
            "leakage rejected",
            "per-seed reasoning gates pass",
        ],
        "failure_routes": {
            "clean_negative": "137B_REAL_RAW_REASONING_REPAIR_PLAN or targeted failure analysis",
            "helper_integrity_failure": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL",
            "scorer_weakness": "137E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS",
            "leakage": "137L_REASONING_EVAL_LEAKAGE_REDESIGN",
        },
        "clean_negative_accepted": True,
        "no_threshold_weakening_to_force_positive": True,
        "no_helper_alteration_to_improve_score": True,
        "no_checkpoint_swap_without_provenance_safe_selection": True,
        "no_expected_or_scorer_metadata_in_helper_requests": True,
    }


def anti_shortcut_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_137b_anti_shortcut_requirements_v1",
        "requirements": [
            "shared_raw_generation_helper.py only",
            "generated_text before scoring",
            "expected-output canary",
            "AST shortcut scan",
            "no expected/scorer metadata in helper request",
            "no old runner imports",
            "no deterministic positive-arm construction",
            "no LLM judge",
            "no oracle/rerank/verifier",
            "controls must fail",
            "leakage rejected",
            "clean negative accepted",
        ],
    }


def risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_137b_risk_register_v1",
        "risks": [
            {"risk": "Repair overfits synthetic reasoning tokens", "mitigation": "fresh held-out helper-only final eval with leakage audit"},
            {"risk": "Prompt simplification hides capability gap", "mitigation": "preserve per-family floors and clean-negative route"},
            {"risk": "Checkpoint swap masks provenance", "mitigation": "require source checkpoint hashes and target-only mutation"},
            {"risk": "Scorer weakness mistaken for model failure", "mitigation": "near-match and control analysis before repair"},
        ],
    }


def write_failure(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    decision = {
        "schema_version": "phase_137b_failure_decision_v1",
        "decision": "real_raw_reasoning_repair_plan_incomplete",
        "next": "137B_REPAIR_PLAN_INCOMPLETE_ANALYSIS",
        "failure_verdict": error.verdict,
        "failure_message": error.message,
        **FALSE_BOUNDARY_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", ["REAL_RAW_REASONING_REPAIR_PLAN_FAILS", error.verdict], decision, error.message)
    write_report(out, ["REAL_RAW_REASONING_REPAIR_PLAN_FAILS", error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "queue.json", {"schema_version": "phase_137b_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["REAL_RAW_REASONING_REPAIR_PLAN_RUNNING"], {"decision": "pending", "next": "pending"})

    root_137r = resolve_path(args.upstream_137r_root)
    root_136r = resolve_path(args.upstream_136r_root)
    root_135e = resolve_path(args.upstream_135e_root)
    root_135d = resolve_path(args.upstream_135d_root)
    verify_upstreams(out, root_137r, root_136r, root_135e, root_135d)
    append_progress(out, "upstream verification", upstream_137r=True, upstream_136r=True, upstream_135e=True, upstream_135d=True)
    refresh_status(out, "running", ["UPSTREAMS_VERIFIED"], {"decision": "pending", "next": "pending"})

    append_progress(out, "artifact loading", artifact_count=len(REQUIRED_137R_ARTIFACTS))
    quality = generation_quality_report(root_137r)
    write_json(out / "generation_quality_report.json", quality)
    append_progress(out, "generation-quality analysis", row_count=quality["row_count"], expected_token_inclusion_rate=quality["expected_token_inclusion_rate"])

    scoring = scoring_mismatch_report(root_137r, quality)
    write_json(out / "scoring_mismatch_report.json", scoring)
    append_progress(out, "scoring-mismatch analysis", near_match_rate=scoring["near_match_rate"], strict_but_valid=scoring["scoring_strict_but_valid"])

    diagnosis = reasoning_failure_diagnosis(root_137r, quality, scoring)
    write_json(out / "reasoning_failure_diagnosis.json", diagnosis)
    append_progress(out, "diagnosis", primary_diagnosis=diagnosis["primary_diagnosis"])

    gap = checkpoint_gap_report(root_137r, diagnosis, quality, scoring)
    write_json(out / "checkpoint_capability_gap_report.json", gap)
    options = repair_option_matrix(diagnosis["primary_diagnosis"])
    write_json(out / "repair_option_matrix.json", options)
    append_progress(out, "repair option selection", primary=diagnosis["primary_diagnosis"])

    if diagnosis["helper_integrity_failure"]:
        decision_name = "raw_helper_integrity_failure"
        next_name = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
        recommended_type = "helper_integrity_fix"
    elif diagnosis["leakage_eval_contamination"]:
        decision_name = "reasoning_eval_leakage"
        next_name = "137L_REASONING_EVAL_LEAKAGE_REDESIGN"
        recommended_type = "eval_leakage_redesign"
    elif scoring["primary_scorer_issue_detected"]:
        decision_name = "reasoning_scorer_or_task_weakness"
        next_name = "137E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"
        recommended_type = "scorer_or_eval_design_review"
    else:
        decision_name = EXPECTED_DECISION
        next_name = EXPECTED_NEXT
        recommended_type = "targeted_real_raw_reasoning_training_or_probe"

    rejected = [
        {"alternative": "helper integrity fix", "rejected": not diagnosis["helper_integrity_failure"], "reason": "137R canary, AST, provenance, and generated-before-scoring gates passed"},
        {"alternative": "scorer weakness", "rejected": not scoring["primary_scorer_issue_detected"], "reason": "expected and near-match rates are low while scorer controls failed"},
        {"alternative": "leakage redesign", "rejected": not diagnosis["leakage_eval_contamination"], "reason": "137R leakage audit rejected overlap"},
        {"alternative": "capability repair", "rejected": recommended_type != "targeted_real_raw_reasoning_training_or_probe", "reason": "selected only when helper/scorer/leakage are not primary"},
    ]
    recommendation = {
        "schema_version": "phase_137b_recommended_repair_target_v1",
        "recommended_repair_type": recommended_type,
        "recommended_next": next_name,
        "primary_diagnosis": diagnosis["primary_diagnosis"],
        "rejected_alternatives": rejected,
    }
    write_json(out / "recommended_repair_target.json", recommendation)
    write_json(out / "next_milestone_plan.json", next_milestone_plan())
    write_json(out / "anti_shortcut_requirements.json", anti_shortcut_requirements())
    write_json(out / "risk_register.json", risk_register())
    append_progress(out, "next milestone drafting", next=next_name)

    decision = {
        "schema_version": "phase_137b_decision_v1",
        "decision": decision_name,
        "next": next_name,
        "reason": "137R produced 0% reasoning accuracy with clean helper integrity, failed controls, rejected leakage, and off-prompt/stale raw outputs.",
        "primary_diagnosis": diagnosis["primary_diagnosis"],
        "evidence_summary": diagnosis["evidence"],
        "helper_integrity_status": "passed" if not diagnosis["helper_integrity_failure"] else "failed",
        "scorer_status": "strict_but_valid" if scoring["scoring_strict_but_valid"] else "requires_review",
        "leakage_status": "rejected" if not diagnosis["leakage_eval_contamination"] else "detected",
        "generation_quality_summary": {
            "row_count": quality["row_count"],
            "expected_token_inclusion_rate": quality["expected_token_inclusion_rate"],
            "near_match_rate": scoring["near_match_rate"],
            "stale_user_assistant_fragment_rate": quality["stale_user_assistant_fragment_rate"],
            "off_prompt_output_rate": quality["off_prompt_output_rate"],
        },
        "checkpoint_capability_gap_likelihood": gap["checkpoint_capability_gap_likelihood"],
        "recommended_repair_type": recommended_type,
        "rejected_alternatives": rejected,
        "required_gates_for_next": next_milestone_plan()["success_gates"],
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "inference_run_count": 0,
        "shared_helper_called_for_new_generation": False,
        "checkpoint_mutated": False,
        "runtime_surface_mutated": False,
        "root_license_changed": False,
        **FALSE_BOUNDARY_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "decision writing", decision=decision_name, next=next_name)
    verdicts = [
        "REAL_RAW_REASONING_REPAIR_PLAN_COMPLETE",
        "PLANNING_ONLY",
        "NO_NEW_INFERENCE",
        "UPSTREAM_137R_CLEAN_NEGATIVE_VERIFIED",
        "REASONING_NOT_RESTORED",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED",
    ]
    refresh_status(out, "positive", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_137b_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-137r-root", default=str(DEFAULT_UPSTREAM_137R_ROOT))
    parser.add_argument("--upstream-136r-root", default=str(DEFAULT_UPSTREAM_136R_ROOT))
    parser.add_argument("--upstream-135e-root", default=str(DEFAULT_UPSTREAM_135E_ROOT))
    parser.add_argument("--upstream-135d-root", default=str(DEFAULT_UPSTREAM_135D_ROOT))
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
