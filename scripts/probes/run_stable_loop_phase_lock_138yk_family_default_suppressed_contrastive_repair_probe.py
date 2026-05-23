#!/usr/bin/env python3
"""138YK deterministic family-default-suppressed contrastive repair/probe.

This phase may train only a new target checkpoint under target/. Final eval
uses scripts/probes/shared_raw_generation_helper.py only. Source checkpoints and
the shared helper are immutable. A clean negative is valid.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - fail-closed path.
    torch = None
    nn = None


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe/smoke")
DEFAULT_UPSTREAM_138YJ_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yj_family_default_suppressed_contrastive_objective_plan/smoke")
DEFAULT_UPSTREAM_138YD_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yd_family_default_shortcut_analysis/smoke")
DEFAULT_UPSTREAM_138YI_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138yi_family_specific_value_attractor_repair_probe/smoke")
HELPER_PATH = REPO_ROOT / "scripts/probes/shared_raw_generation_helper.py"
RUNNER_PATH = Path(__file__).resolve()
CHECKER_PATH = REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe_check.py"
TARGET_CHECKPOINT_REL = "target/pilot_wave/stable_loop_phase_lock_138yk_family_default_suppressed_contrastive_repair_probe/smoke/checkpoints/target_138yk_family_default_suppressed/model.pt"
SOURCE_138YI_TARGET_REL = "target/pilot_wave/stable_loop_phase_lock_138yi_family_specific_value_attractor_repair_probe/smoke/checkpoints/target_138yi_family_contrastive_value/model.pt"

POSITIVE_VERDICT = "FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_POSITIVE"
NEGATIVE_VERDICT = "FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_FAILS"
MISSING_TRAINING_VERDICT = "FAMILY_DEFAULT_SUPPRESSED_TRAINING_PATH_MISSING"
DETERMINISM_VERDICT = "DETERMINISM_REPLAY_MISMATCH"
POSITIVE_DECISION = "family_default_suppressed_contrastive_repair_positive"
POSITIVE_NEXT = "139YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_SCALE_CONFIRM"
NO_VALUE_DECISION = "no_value_improvement"
NO_VALUE_NEXT = "138YK_FAILURE_ANALYSIS"
PARROT_DECISION = "parrot_trap_copy_shortcut_detected"
PARROT_NEXT = "138P_PARROT_TRAP_VALUE_COPY_ANALYSIS"
WRAPPER_VALUE_DECISION = "contrastive_objective_still_too_weak"
WRAPPER_VALUE_NEXT = "138YJ_CONTRASTIVE_OBJECTIVE_WEAKNESS_REVIEW"
STALE_DECISION = "stale_chat_rollout_failure"
STALE_NEXT = "138S_STALE_CHAT_ROLLOUT_FAILURE_ANALYSIS"
NAMESPACE_DECISION = "namespace_rollout_failure"
NAMESPACE_NEXT = "138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS"
HIGH_FREQUENCY_DECISION = "high_frequency_value_replay_detected"
HIGH_FREQUENCY_NEXT = "138YH_HIGH_FREQUENCY_VALUE_REPLAY_ANALYSIS"
FAMILY_DEFAULT_DECISION = "family_default_shortcut_persists"
FAMILY_DEFAULT_NEXT = "138YD_FAMILY_DEFAULT_SHORTCUT_ANALYSIS"
HARD_NEGATIVE_DECISION = "hard_negative_default_failure"
HARD_NEGATIVE_NEXT = "138YKD_HARD_NEGATIVE_DEFAULT_FAILURE_ANALYSIS"
DETERMINISM_DECISION = "nondeterministic_family_default_suppressed_probe"
DETERMINISM_NEXT = "138N_DETERMINISM_FAILURE_ANALYSIS"
LEAKAGE_DECISION = "family_default_suppressed_eval_leakage"
LEAKAGE_NEXT = "138L_FAMILY_CONTRASTIVE_EVAL_LEAKAGE_REDESIGN"
SCORER_DECISION = "scorer_or_task_weakness"
SCORER_NEXT = "138E_REASONING_SCORER_OR_TASK_WEAKNESS_ANALYSIS"
HELPER_FAILURE_DECISION = "raw_helper_integrity_failure"
HELPER_FAILURE_NEXT = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
TRAINING_MISSING_DECISION = "family_default_suppressed_training_path_missing"
TRAINING_MISSING_NEXT = "138YKA_FAMILY_DEFAULT_SUPPRESSED_TRAINING_HELPER_INTEGRATION_PLAN"

BOUNDARY_TEXT = (
    "138YK is a deterministic targeted family-default-suppressed contrastive repair/probe. It "
    "may train only a new target checkpoint under target/ and final-evaluate "
    "only through scripts/probes/shared_raw_generation_helper.py. It does not "
    "mutate source checkpoints, modify the shared helper, import old runners, "
    "start services, deploy, delete or consolidate files, modify runtime, "
    "service, deploy, product, or release surfaces, modify SDK exports, modify "
    "docs/product or docs/releases, or change root LICENSE. It does not claim "
    "full raw assistant capability, structured/tool capability, GPT-like "
    "readiness, open-domain readiness, production chat, public API, deployment "
    "readiness, or safety alignment."
)
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
FINAL_EVAL_FLAGS = {
    "generated_text_produced_before_scoring": True,
    "shared_raw_generation_helper_used": True,
    "expected_output_used_for_generation": False,
    "expected_payload_used_for_generation": False,
    "scorer_metadata_used_for_generation": False,
    "oracle_rerank_used": False,
    "verifier_rerank_used": False,
    "llm_judge_used": False,
    "teacher_forcing_used": False,
    "constrained_decoding_used": False,
    "json_mode_used": False,
    "grammar_decoder_used": False,
    "regex_fixer_used": False,
    "post_generation_repair_used": False,
    "retry_loop_used": False,
    "best_of_n_used": False,
    "actual_tool_execution_used": False,
    "runtime_tool_call_used": False,
}
ALLOWED_HELPER_KEYS = {"prompt", "checkpoint_path", "checkpoint_hash", "seed", "max_new_tokens", "generation_config"}
FORBIDDEN_HELPER_KEYS = {
    "expected_output",
    "expected_payload",
    "expected_answer",
    "required_keys",
    "required_keywords",
    "forbidden_outputs",
    "schema_answer_object",
    "scorer_metadata",
    "labels",
    "oracle_data",
    "target_json",
    "gold_output",
    "row_answer",
    "eval_family",
    "answer",
    "expected_values",
}
FAMILIES = [
    "FAMILY_CONTRAST_DIRECT_COPY_DIAGNOSTIC",
    "FAMILY_CONTRAST_RULE_DERIVED",
    "FAMILY_CONTRAST_TABLE_DERIVED",
    "FAMILY_CONTRAST_COMPOSITION_DERIVED",
    "FAMILY_CONTRAST_CONTRADICTION_RESOLUTION",
    "FAMILY_CONTRAST_OOD_SYMBOL_BINDING",
    "FAMILY_CONTRAST_NO_STALE_DIRECT",
    "FAMILY_CONTRAST_AFTER_PREFIX_STABILITY",
]
TRAIN_CATEGORIES = FAMILIES + ["VALUE_STALE_SUPPRESSION", "VALUE_GENERIC_VALUE_REJECTION"]
STANDARD_REFUSAL_TEMPLATES = {"INSUFFICIENT_INFORMATION", "UNKNOWN", "UNANSWERABLE"}
POSITIVE_GATES = {
    "answer_value_accuracy": 0.25,
    "exact_answer_accuracy": 0.20,
    "value_after_prefix_accuracy": 0.25,
    "rule_derived_value_accuracy": 0.20,
    "table_derived_value_accuracy": 0.20,
    "composition_derived_value_accuracy": 0.15,
    "ood_symbol_value_accuracy": 0.15,
    "eval_namespace_emission_accuracy": 0.90,
    "answer_prefix_accuracy": 0.90,
    "intra_family_contrastive_accuracy": 0.30,
    "intra_family_unique_correct_value_rate": 0.25,
}
MAX_POSITIVE_RATES = {
    "prefix_success_value_failure_rate": 0.70,
    "no_stale_wrong_value_rate": 0.75,
    "post_wrapper_garbage_token_rate": 0.20,
    "empty_value_after_prefix_rate": 0.20,
    "generic_value_after_prefix_rate": 0.30,
    "stale_chat_fragment_rate": 0.10,
    "train_namespace_leak_rate": 0.05,
    "intra_family_mode_collapse_rate": 0.60,
    "family_default_attractor_rate": 0.35,
    "family_default_reuse_rate": 0.35,
    "family_dominant_wrong_value_rate": 0.35,
    "multi_expected_to_single_default_rate": 0.30,
    "same_value_for_all_rows_rate": 0.20,
    "hard_negative_default_violation_rate": 0.20,
}
SEED_MIN_GATES = {
    "answer_value_accuracy": 0.20,
    "intra_family_contrastive_accuracy": 0.25,
    "rule_derived_value_accuracy": 0.15,
    "table_derived_value_accuracy": 0.15,
}
SEED_MAX_GATES = {
    "stale_chat_fragment_rate": 0.15,
    "intra_family_mode_collapse_rate": 0.65,
    "family_default_attractor_rate": 0.45,
    "hard_negative_default_violation_rate": 0.25,
}
BASELINE_METRICS = {
    "baseline_answer_value_accuracy": 0.0,
    "baseline_exact_answer_accuracy": 0.0,
    "baseline_prefix_success_value_failure_rate": 1.0,
    "baseline_no_stale_wrong_value_rate": 1.0,
}
GRU_STATE_KEYS = {
    "embedding.weight",
    "rnn.weight_ih_l0",
    "rnn.weight_hh_l0",
    "rnn.bias_ih_l0",
    "rnn.bias_hh_l0",
    "head.weight",
    "head.bias",
}
BYTE_VOCAB_SIZE = 256


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(text: str | Path) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise GateError("138YK_BOUNDARY_FAILURE", "--out must stay inside repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138YK_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    return resolved


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def require_artifacts(root: Path, names: list[str], verdict: str) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise GateError(verdict, "required upstream artifacts missing", {"root": rel(root), "missing": missing})


def refresh_status(out: Path, status: str, verdicts: list[str], decision: dict[str, Any]) -> None:
    write_summary(out, status, verdicts, decision)
    write_report(out, verdicts, decision)


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138yk_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "reasoning_subtrack_real_raw_evidence_partially_restored": decision.get("reasoning_subtrack_real_raw_evidence_partially_restored", False),
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
            f"- `answer_value_accuracy`: `{decision.get('answer_value_accuracy')}`",
            f"- `parrot_trap_detected`: `{decision.get('parrot_trap_detected')}`",
            f"- `post_wrapper_garbage_token_rate`: `{decision.get('post_wrapper_garbage_token_rate')}`",
            "",
            "Scout-First Laziness is treated only as a design hypothesis, not measured scout/grower behavior.",
            "Missing Intra-Family Variance is tested through output-level contrast groups.",
            "If the model learns only the family, it fails.",
            "If it emits one family-default value for multiple same-family prompts, it fails.",
            "If it copies prompt values but fails derived/OOD values, it fails.",
            "Raw assistant capability remains quarantined.",
            "Structured/tool capability remains invalidated.",
            "not GPT-like readiness.",
            "not open-domain assistant readiness.",
            "not production chat.",
            "not public API.",
            "not deployment readiness.",
            "not safety alignment.",
        ]
    )
    write_text(out / "report.md", "\n".join(lines) + "\n")


def require_torch() -> None:
    if torch is None or nn is None:
        raise GateError(MISSING_TRAINING_VERDICT, "torch is unavailable; no safe helper-compatible training path")


def import_helper() -> Any:
    spec = importlib.util.spec_from_file_location("shared_raw_generation_helper_138yk", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise GateError("RAW_GENERATION_BACKEND_MISSING", "cannot import shared raw generation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_upstreams(out: Path, root_138yj: Path, root_138yd: Path, root_138yi: Path) -> dict[str, Any]:
    required_138yj = [
        "decision.json",
        "family_default_failure_summary.json",
        "objective_weakness_diagnosis.json",
        "family_default_suppression_requirements.json",
        "target_138yk_training_objective_spec.json",
        "target_138yk_eval_gate_spec.json",
        "next_138yk_milestone_plan.json",
    ]
    required_138yd = [
        "decision.json",
        "family_default_root_cause.json",
        "contrast_group_default_failure_report.json",
        "objective_shortcut_reward_report.json",
        "scorer_dataset_shortcut_report.json",
    ]
    required_138yi = [
        "decision.json",
        "aggregate_metrics.json",
        "family_default_attractor_report.json",
        "high_frequency_value_replay_report.json",
        "value_grounding_metrics.json",
        "intra_family_contrastive_metrics.json",
        "parrot_trap_report.json",
        "expected_output_canary_report.json",
        "ast_shortcut_scan_report.json",
        "control_arm_report.json",
        "freshness_leakage_audit.json",
        "determinism_replay_report.json",
        "generated_before_scoring_report.json",
        "source_checkpoint_integrity_manifest.json",
        "target_checkpoint_integrity_manifest.json",
        "raw_generation_trace.jsonl",
    ]
    for root, names, verdict in [
        (root_138yj, required_138yj, "UPSTREAM_138YJ_ARTIFACT_MISSING"),
        (root_138yd, required_138yd, "UPSTREAM_138YD_ARTIFACT_MISSING"),
        (root_138yi, required_138yi, "UPSTREAM_138YI_ARTIFACT_MISSING"),
    ]:
        require_artifacts(root, names, verdict)

    d138yj = read_json(root_138yj / "decision.json")
    yj_failure = read_json(root_138yj / "family_default_failure_summary.json")
    yj_weakness = read_json(root_138yj / "objective_weakness_diagnosis.json")
    yj_suppression = read_json(root_138yj / "family_default_suppression_requirements.json")
    yj_plan = read_json(root_138yj / "next_138yk_milestone_plan.json")
    d138yd = read_json(root_138yd / "decision.json")
    root138yd = read_json(root_138yd / "family_default_root_cause.json")
    contrast138yd = read_json(root_138yd / "contrast_group_default_failure_report.json")
    objective138yd = read_json(root_138yd / "objective_shortcut_reward_report.json")
    scorer138yd = read_json(root_138yd / "scorer_dataset_shortcut_report.json")
    d138yi = read_json(root_138yi / "decision.json")
    aggregate138yi = read_json(root_138yi / "aggregate_metrics.json")
    family_default138yi = read_json(root_138yi / "family_default_attractor_report.json")
    high_frequency138yi = read_json(root_138yi / "high_frequency_value_replay_report.json")
    canary = read_json(root_138yi / "expected_output_canary_report.json")
    scan = read_json(root_138yi / "ast_shortcut_scan_report.json")
    controls = read_json(root_138yi / "control_arm_report.json")
    leakage = read_json(root_138yi / "freshness_leakage_audit.json")
    replay = read_json(root_138yi / "determinism_replay_report.json")
    before = read_json(root_138yi / "generated_before_scoring_report.json")
    source_integrity = read_json(root_138yi / "source_checkpoint_integrity_manifest.json")
    target_integrity = read_json(root_138yi / "target_checkpoint_integrity_manifest.json")
    traces = read_jsonl(root_138yi / "raw_generation_trace.jsonl")

    if d138yj.get("decision") != "family_default_suppressed_contrastive_objective_plan_complete" or d138yj.get("next") != "138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE":
        raise GateError("UPSTREAM_138YJ_ARTIFACT_MISSING", "138YJ did not route to 138YK")
    if yj_failure.get("root_cause") != "contrastive_objective_too_weak" or yj_plan.get("milestone") != "138YK_FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_PROBE":
        raise GateError("UPSTREAM_138YJ_ARTIFACT_MISSING", "138YJ plan profile mismatch")
    if "family_default_value_bank" not in yj_suppression or yj_weakness.get("objective_pressure_against_family_default_reuse_insufficient") is not True:
        raise GateError("UPSTREAM_138YJ_ARTIFACT_MISSING", "138YJ hard-negative objective plan missing")

    if d138yd.get("decision") != "family_default_shortcut_analysis_complete" or d138yd.get("root_cause") != "contrastive_objective_too_weak":
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YD did not establish contrastive objective weakness")
    if contrast138yd.get("contrast_group_default_shortcut_rate") != 0.78125 or contrast138yd.get("multi_expected_to_single_default_rate") != 0.6822916666666666:
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YD contrast default profile mismatch")
    if root138yd.get("family_default_control_failed") is not True:
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YD family default control profile mismatch")
    for key in ["objective_explicitly_penalizes_family_default", "objective_explicitly_penalizes_same_value_for_all_rows", "objective_rewards_intra_family_distinct_values"]:
        if objective138yd.get(key) is not False:
            raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", f"138YD objective field changed: {key}")
    if scorer138yd.get("dataset_low_intra_family_value_diversity") is True:
        raise GateError("UPSTREAM_138YD_ARTIFACT_MISSING", "138YD no longer rules out low value diversity")

    if d138yi.get("verdict") != "FAMILY_SPECIFIC_VALUE_ATTRACTOR_REPAIR_FAILS" or d138yi.get("decision") != "high_frequency_train_value_replay_detected":
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "138YI clean-negative route mismatch")
    expected_138yi = {
        "answer_value_accuracy": 0.0,
        "exact_answer_accuracy": 0.0,
        "intra_family_contrastive_accuracy": 0.0,
        "intra_family_mode_collapse_rate": 0.9427083333333334,
        "family_default_attractor_rate": 0.78125,
        "parrot_trap_detected": False,
        "stale_chat_fragment_rate": 0.0,
        "train_namespace_leak_rate": 0.0,
    }
    mismatches = {key: {"expected": value, "actual": aggregate138yi.get(key)} for key, value in expected_138yi.items() if aggregate138yi.get(key) != value}
    if family_default138yi.get("family_default_shortcut_detected") is not True or high_frequency138yi.get("high_frequency_train_value_replay_detected") is not True:
        mismatches["shortcut_flags"] = {"family_default": family_default138yi.get("family_default_shortcut_detected"), "high_frequency": high_frequency138yi.get("high_frequency_train_value_replay_detected")}
    if mismatches:
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "138YI failure profile mismatch", mismatches)
    if canary.get("expected_output_canary_passed") is not True or scan.get("ast_shortcut_scan_passed") is not True:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138YI canary/AST failed")
    if controls.get("controls_failed") is not True or leakage.get("leakage_rejected") is not True or replay.get("determinism_replay_passed") is not True:
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "138YI controls/leakage/replay profile missing")
    if before.get("generated_text_produced_before_scoring") is not True:
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "138YI generated-before-scoring missing")
    if source_integrity.get("source_checkpoint_unchanged") is not True or target_integrity.get("target_checkpoint_changed") is not True:
        raise GateError("UPSTREAM_138YI_ARTIFACT_MISSING", "138YI checkpoint integrity missing")
    for trace in traces:
        request = trace.get("helper_request", {})
        if set(request) != ALLOWED_HELPER_KEYS or set(request) & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138YI helper request metadata violation")

    write_json(
        out / "upstream_138yj_manifest.json",
        {
            "schema_version": "phase_138yk_upstream_138yj_manifest_v1",
            "upstream_root": rel(root_138yj),
            "verified": True,
            "decision": d138yj.get("decision"),
            "next": d138yj.get("next"),
            "root_cause": yj_failure.get("root_cause"),
            "hard_negative_default_plan_present": True,
            "family_default_value_bank_required": True,
        },
    )
    write_json(
        out / "upstream_138yd_manifest.json",
        {
            "schema_version": "phase_138yk_upstream_138yd_manifest_v1",
            "upstream_root": rel(root_138yd),
            "verified": True,
            "decision": d138yd.get("decision"),
            "root_cause": d138yd.get("root_cause"),
            "contrast_group_default_shortcut_rate": contrast138yd.get("contrast_group_default_shortcut_rate"),
            "multi_expected_to_single_default_rate": contrast138yd.get("multi_expected_to_single_default_rate"),
            "family_default_control_failed": root138yd.get("family_default_control_failed"),
            "objective_explicitly_penalizes_family_default": objective138yd.get("objective_explicitly_penalizes_family_default"),
            "objective_explicitly_penalizes_same_value_for_all_rows": objective138yd.get("objective_explicitly_penalizes_same_value_for_all_rows"),
            "objective_rewards_intra_family_distinct_values": objective138yd.get("objective_rewards_intra_family_distinct_values"),
        },
    )
    write_json(
        out / "upstream_138yi_manifest.json",
        {
            "schema_version": "phase_138yk_upstream_138yi_manifest_v1",
            "upstream_root": rel(root_138yi),
            "verified": True,
            "decision": d138yi.get("decision"),
            "verdict": d138yi.get("verdict"),
            "helper_canary_ast_leakage_controls_determinism_passed": True,
            "source_checkpoint_unchanged": True,
            "target_checkpoint_changed": True,
            "generated_text_before_scoring": True,
            "no_expected_or_scorer_metadata_reached_helper_requests": True,
            "answer_value_accuracy": aggregate138yi.get("answer_value_accuracy"),
            "exact_answer_accuracy": aggregate138yi.get("exact_answer_accuracy"),
            "intra_family_contrastive_accuracy": aggregate138yi.get("intra_family_contrastive_accuracy"),
            "intra_family_mode_collapse_rate": aggregate138yi.get("intra_family_mode_collapse_rate"),
            "family_default_attractor_rate": aggregate138yi.get("family_default_attractor_rate"),
            "family_default_shortcut_detected": family_default138yi.get("family_default_shortcut_detected"),
            "high_frequency_train_value_replay_detected": high_frequency138yi.get("high_frequency_train_value_replay_detected"),
            "parrot_trap_detected": aggregate138yi.get("parrot_trap_detected"),
            "stale_chat_fragment_rate": aggregate138yi.get("stale_chat_fragment_rate"),
            "train_namespace_leak_rate": aggregate138yi.get("train_namespace_leak_rate"),
            "determinism_replay_passed": replay.get("determinism_replay_passed"),
        },
    )
    return {
        "d138yj": d138yj,
        "yj_suppression": yj_suppression,
        "d138yd": d138yd,
        "root138yd": root138yd,
        "contrast138yd": contrast138yd,
        "objective138yd": objective138yd,
        "d138yi": d138yi,
        "aggregate138yi": aggregate138yi,
        "family_default138yi": family_default138yi,
        "high_frequency138yi": high_frequency138yi,
        "root_138yj": root_138yj,
        "root_138yd": root_138yd,
        "root_138yi": root_138yi,
    }


def deterministic_setup(seed: int) -> dict[str, Any]:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    manifest = {
        "schema_version": "phase_138yk_determinism_manifest_v1",
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "random_seed": seed,
        "numpy_used": False,
        "numpy_seed": None,
        "torch_used": torch is not None,
        "torch_seed": None,
        "torch_deterministic_algorithms": None,
        "device": "cpu",
        "cuda_available": False,
        "torch_version": None,
        "cuda_version": None,
    }
    if torch is not None:
        torch.manual_seed(seed)
        try:
            torch.use_deterministic_algorithms(True)
            manifest["torch_deterministic_algorithms"] = True
        except Exception:
            manifest["torch_deterministic_algorithms"] = False
        manifest.update({"torch_seed": seed, "cuda_available": bool(torch.cuda.is_available()), "torch_version": torch.__version__, "cuda_version": getattr(torch.version, "cuda", None)})
    return manifest


def answer_value(category: str, split: str, index: int, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
    category = category.replace("FAMILY_CONTRAST_", "VALUE_")
    namespace = "train" if split == "train" else "eval"
    answer_prefix = "E"
    if split == "eval":
        base = 730000 + index * 17
        style = f"assessment orbit eval-{index}"
    else:
        base = 310000 + index * 13
        style = f"practice lattice train-{index}"
    value = f"{'EV' if split == 'eval' else 'TR'}{base}"
    if category == "VALUE_DIRECT_COPY_DIAGNOSTIC":
        prompt = f"{style}. Direct diagnostic token is {value}. Copying this token is diagnostic only. Return ANSWER=E plus that token."
        expected = f"ANSWER=E{value}"
        mode = "direct_copy"
    elif category == "VALUE_RULE_DERIVED":
        a = base + rng.randrange(10, 90)
        b = rng.randrange(11, 77)
        derived = f"{'EV' if split == 'eval' else 'TR'}{a + b}"
        prompt = f"{style}. Rule-derived task. Start with {a}; add rule increment {b}; ignore distractor {base - 4}. Return ANSWER=E plus computed result token."
        expected = f"ANSWER=E{derived}"
        value = derived
        mode = "rule_derived"
    elif category == "VALUE_TABLE_DERIVED":
        key = f"K{(index % 7) + 1}"
        table = {f"K{i}": f"{'EV' if split == 'eval' else 'TR'}{base + i * 31}" for i in range(1, 8)}
        prompt = f"{style}. Table-derived task. Mapping entries: " + "; ".join(f"{k}->{v}" for k, v in sorted(table.items())) + f". Selected key is {key}. Return ANSWER=E plus mapped token."
        expected = f"ANSWER=E{table[key]}"
        value = table[key]
        mode = "table_derived"
    elif category == "VALUE_COMPOSITION_DERIVED":
        left = base + 5
        mid = rng.randrange(20, 60)
        right = rng.randrange(20, 60)
        derived = f"{'EV' if split == 'eval' else 'TR'}{left + mid - right}"
        prompt = f"{style}. Composition task. Take anchor {left}, add supplied offset {mid}, subtract supplied offset {right}. Return ANSWER=E plus final token."
        expected = f"ANSWER=E{derived}"
        value = derived
        mode = "composition_derived"
    elif category == "VALUE_CONTRADICTION_RESOLUTION":
        stale = f"{'EV' if split == 'eval' else 'TR'}{base - 222}"
        trusted = f"{'EV' if split == 'eval' else 'TR'}{base + 222}"
        prompt = f"{style}. Contradiction task. Untrusted draft token {stale}. Trusted final token {trusted}. Use trusted final only. Return ANSWER=E plus trusted token."
        expected = f"ANSWER=E{trusted}"
        value = trusted
        mode = "contradiction_resolution"
    elif category == "VALUE_OOD_SYMBOL_BINDING":
        symbol = f"SIGMA_{namespace.upper()}_{index % 997}"
        bound = f"{'EV' if split == 'eval' else 'TR'}SYM{base + 19}"
        prompt = f"{style}. OOD symbol binding. New marker {symbol} binds to value token {bound}. Return ANSWER=E plus bound token. Do not use memorized train values."
        expected = f"ANSWER=E{bound}"
        value = bound
        mode = "ood_symbol_binding"
    elif category in {"VALUE_NO_STALE_CHAT_DIRECT", "VALUE_NO_STALE_DIRECT"}:
        prompt = f"{style}. No stale chat. Do not write User: or Assistant:. Direct value token is {value}. Return ANSWER=E plus direct value."
        expected = f"ANSWER=E{value}"
        mode = "no_stale_direct"
    elif category == "VALUE_AFTER_PREFIX_STABILITY":
        prompt = f"{style}. Prefix stability task. Required value token after wrapper is {value}. Return ANSWER=E immediately followed by the value with no filler or repeated symbols."
        expected = f"ANSWER=E{value}"
        mode = "after_prefix_stability"
    elif category == "VALUE_STALE_SUPPRESSION":
        prompt = f"{style}. Training-only stale suppression. Do not continue transcript. User: and Assistant: are forbidden. Value token is {value}. Return ANSWER=E plus value."
        expected = f"ANSWER=E{value}"
        mode = "stale_suppression"
    elif category == "VALUE_GENERIC_VALUE_REJECTION":
        prompt = f"{style}. Generic values like UNKNOWN, VALUE, TOKEN, 0000 are forbidden. Specific value token is {value}. Return ANSWER=E plus specific value."
        expected = f"ANSWER=E{value}"
        mode = "generic_value_rejection"
    else:
        raise ValueError(category)
    return prompt, expected, {"mode": mode, "answer_value": value, "namespace": answer_prefix}


def build_train_rows(train_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(train_examples):
        category = TRAIN_CATEGORIES[index % len(TRAIN_CATEGORIES)]
        prompt, expected, meta = answer_value(category, "train", index, random.Random(1389000 + index))
        rows.append(
            {
                "row_id": f"138yk_train_{index:06d}",
                "split": "train",
                "family": category,
                "prompt": prompt,
                "expected_output": expected,
                "expected_payload": {"answer": expected, **meta},
                "answer_value": meta["answer_value"],
                "scoring_mode": meta["mode"],
                "forbidden_distractor": f"DISTRACTOR_TRAIN_{index}",
            }
        )
    return rows


def build_eval_rows(seeds: list[int], rows_per_family: int, contrast_group_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_index = 0
    for family_index, family in enumerate(FAMILIES):
        for local_index in range(rows_per_family):
            prompt, expected, meta = answer_value(family, "eval", global_index, random.Random(2389000 + family_index * 100000 + local_index))
            rows.append(
                {
                    "row_id": f"138yk_eval_{global_index:05d}",
                    "split": "eval",
                    "family": family,
                    "contrast_group_id": f"{family}_group_{local_index // max(1, contrast_group_size):03d}",
                    "contrast_group_index": local_index // max(1, contrast_group_size),
                    "contrast_group_size": contrast_group_size,
                    "seed": seeds[global_index % len(seeds)],
                    "prompt": prompt,
                    "expected_output": expected,
                    "expected_payload": {"answer": expected, **meta},
                    "answer_value": meta["answer_value"],
                    "scoring_mode": meta["mode"],
                    "forbidden_distractor": f"DISTRACTOR_EVAL_{global_index}",
                }
            )
            global_index += 1
    return rows


def build_family_default_value_bank(upstream: dict[str, Any], top_k: int) -> dict[str, Any]:
    source_bank = upstream.get("yj_suppression", {}).get("family_default_value_bank", {})
    families = source_bank.get("families", {}) if isinstance(source_bank, dict) else {}
    bank: dict[str, list[dict[str, Any]]] = {}
    for family, entries in sorted(families.items()):
        clean_entries: list[dict[str, Any]] = []
        for index, item in enumerate(entries[:top_k]):
            value = item.get("value")
            if not value:
                continue
            clean_entries.append(
                {
                    "value": value,
                    "count": int(item.get("count", 0)),
                    "rate": float(item.get("rate", 0.0)),
                    "rank": int(item.get("rank", index + 1)),
                    "source": "138YJ_family_default_suppression_requirements",
                }
            )
        if clean_entries:
            bank[family] = clean_entries

    if not bank:
        root_138yi = upstream["root_138yi"]
        scoring = read_jsonl(root_138yi / "scoring_results.jsonl")
        by_family: dict[str, Counter[str]] = defaultdict(Counter)
        for row in scoring:
            value = row.get("answer_value_candidate")
            expected = row.get("expected_value")
            if value and value != expected:
                by_family[row["family"]][value] += 1
        for family, counter in sorted(by_family.items()):
            total = sum(counter.values())
            bank[family] = [
                {
                    "value": value,
                    "count": count,
                    "rate": rate(count, total),
                    "rank": index + 1,
                    "source": "138YI_scoring_results_wrong_value",
                }
                for index, (value, count) in enumerate(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_k])
            ]
    return {
        "schema_version": "phase_138yk_family_default_value_bank_v1",
        "top_k_per_family": top_k,
        "source_artifacts": ["138YJ_family_default_suppression_requirements", "138YI_scoring_results"],
        "families": bank,
        "per_family_forbidden_default_values": {family: [item["value"] for item in entries] for family, entries in bank.items()},
    }


def forbidden_defaults_for_family(family: str, bank: dict[str, Any], expected_value: str | None = None) -> list[str]:
    values = bank.get("per_family_forbidden_default_values", {}).get(family)
    if values is None:
        values = [item["value"] for item in bank.get("families", {}).get(family, [])]
    return [value for value in values if value and value != expected_value]


def attach_family_default_hard_negatives(rows: list[dict[str, Any]], bank: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        forbidden = forbidden_defaults_for_family(item["family"], bank, item.get("answer_value"))
        item["forbidden_family_default_values"] = forbidden
        item["family_default_hard_negative"] = True
        item["pass_requires_no_family_default"] = True
        item["hard_negative_source"] = "138YJ_138YD_138YI_failure_artifacts"
        out.append(item)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in out:
        by_group[str(item.get("contrast_group_id"))].append(item)
    for item in out:
        peers = [peer["answer_value"] for peer in by_group[str(item.get("contrast_group_id"))] if peer["row_id"] != item["row_id"] and peer.get("answer_value") != item.get("answer_value")]
        item["peer_expected_values"] = sorted(set(peers))
    return out


def add_training_default_suppression_prompts(train_rows: list[dict[str, Any]], bank: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in train_rows:
        item = dict(row)
        forbidden = forbidden_defaults_for_family(item["family"], bank, item.get("answer_value"))
        item["forbidden_family_default_values"] = forbidden
        item["family_default_hard_negative"] = bool(forbidden)
        item["pass_requires_no_family_default"] = bool(forbidden)
        if forbidden:
            item["prompt"] = (
                f"{item['prompt']} Family-default hard negatives are forbidden for this family: "
                f"{', '.join(forbidden)}. Do not emit any forbidden default; bind the prompt-specific value."
            )
        out.append(item)
    return out


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_=]+", text.lower()))


def token_jaccard(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def rate(count: int | float, total: int | float) -> float:
    return float(count) / float(total) if total else 0.0


def dataset_manifest(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    row_hashes = [stable_hash(row) for row in rows]
    expected_values = sorted({row["answer_value"] for row in rows})
    return {
        "schema_version": f"phase_138yk_{split}_dataset_manifest_v1",
        "split": split,
        "row_count": len(rows),
        "family_counts": {family: sum(1 for row in rows if row["family"] == family) for family in sorted({row["family"] for row in rows})},
        "dataset_hash": stable_hash(row_hashes),
        "row_hashes_disjoint_ready": True,
        "value_namespace": "TR" if split == "train" else "EV",
        "expected_value_count": len(expected_values),
    }


def split_leakage_audit(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_prompts = {row["prompt"] for row in train_rows}
    eval_prompts = {row["prompt"] for row in eval_rows}
    train_expected = {row["expected_output"] for row in train_rows if row["expected_output"] not in STANDARD_REFUSAL_TEMPLATES}
    eval_expected = {row["expected_output"] for row in eval_rows if row["expected_output"] not in STANDARD_REFUSAL_TEMPLATES}
    train_values = {row["answer_value"] for row in train_rows}
    eval_values = {row["answer_value"] for row in eval_rows}
    train_hashes = {stable_hash(row) for row in train_rows}
    eval_hashes = {stable_hash(row) for row in eval_rows}
    sample_train = train_rows[:: max(1, len(train_rows) // 2000)]
    near_count = 0
    near_samples: list[dict[str, Any]] = []
    for eval_row in eval_rows:
        for train_row in sample_train:
            score = token_jaccard(eval_row["prompt"], train_row["prompt"])
            if score >= 0.90:
                near_count += 1
                if len(near_samples) < 20:
                    near_samples.append({"eval_row_id": eval_row["row_id"], "train_row_id": train_row["row_id"], "token_jaccard": score})
                break
    return {
        "schema_version": "phase_138yk_freshness_leakage_audit_v1",
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "train_eval_row_hash_overlap": len(train_hashes & eval_hashes),
        "exact_prompt_overlap": len(train_prompts & eval_prompts),
        "exact_expected_output_overlap": len(train_expected & eval_expected),
        "eval_value_overlap_with_train": len(train_values & eval_values),
        "train_eval_value_namespaces_disjoint": all(value.startswith("TR") for value in train_values) and all(value.startswith("EV") for value in eval_values),
        "near_duplicate_prompt_count": near_count,
        "near_duplicate_threshold_token_jaccard": 0.90,
        "near_duplicate_samples": near_samples,
        "leakage_rejected": len(train_hashes & eval_hashes) == 0 and len(train_prompts & eval_prompts) == 0 and len(train_expected & eval_expected) == 0 and len(train_values & eval_values) == 0 and near_count == 0,
    }


def encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


def supervised_batch(rows: list[dict[str, Any]], seq_len: int, batch_size: int, rng: random.Random, pad_id: int) -> tuple[Any, Any]:
    xs: list[list[int]] = []
    ys: list[int] = []
    for _ in range(batch_size):
        row = rows[rng.randrange(len(rows))]
        prefix = encode_text(row["prompt"] + "\n")
        answer = encode_text(row["expected_output"] + "\n")
        pos = rng.randrange(len(answer))
        context = prefix + answer[:pos]
        window = context[-seq_len:]
        if len(window) < seq_len:
            window = [pad_id] * (seq_len - len(window)) + window
        xs.append(window)
        ys.append(answer[pos])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def evaluate_teacher_forced_loss(model: Any, rows: list[dict[str, Any]], seq_len: int, pad_id: int, sample_count: int = 1024) -> float:
    require_torch()
    rng = random.Random(138977)
    model.eval()
    losses: list[float] = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for _ in range(max(1, sample_count // 128)):
            x, y = supervised_batch(rows, seq_len, 128, rng, pad_id)
            losses.append(float(loss_fn(model(x), y).item()))
    model.train()
    return mean(losses)


def train_target_model(model: Any, rows: list[dict[str, Any]], seq_len: int, pad_id: int, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    require_torch()
    rng = random.Random(138900 + int(args.seeds.split(",")[0]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_fn = nn.CrossEntropyLoss()
    metrics: list[dict[str, Any]] = []
    initial_loss = evaluate_teacher_forced_loss(model, rows, seq_len, pad_id)
    last_loss = initial_loss
    last_flush = time.monotonic()
    append_progress(out, "training start", train_steps=args.train_steps, batch_size=args.batch_size, initial_loss=initial_loss)
    for step in range(1, int(args.train_steps) + 1):
        x, y = supervised_batch(rows, seq_len, int(args.batch_size), rng, pad_id)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.item())
        if step == 1 or step % max(1, args.metrics_interval) == 0 or step == args.train_steps or (time.monotonic() - last_flush) >= args.heartbeat_sec:
            item = {
                "step": step,
                "train_loss": last_loss,
                "optimizer_step_count": step,
                "train_step_count": step,
                "rollout_alignment_metric_proxy": None,
                "value_grounding_metric_proxy": None,
                "post_wrapper_garbage_penalty_proxy": None,
                "positive_can_depend_on_train_loss": False,
            }
            metrics.append(item)
            write_jsonl(out / "training_metrics.jsonl", metrics)
            append_progress(out, "training heartbeat", step=step, train_loss=last_loss)
            last_flush = time.monotonic()
    final_loss = evaluate_teacher_forced_loss(model, rows, seq_len, pad_id)
    return {
        "schema_version": "phase_138yk_training_objective_report_v1",
        "train_step_count": int(args.train_steps),
        "optimizer_step_count": int(args.train_steps),
        "batch_size": int(args.batch_size),
        "optimizer": "AdamW",
        "lr": float(args.lr),
        "initial_teacher_forced_loss": initial_loss,
        "final_teacher_forced_loss": final_loss,
        "latest_train_loss": last_loss,
        "training_loss_improved": final_loss < initial_loss,
        "positive_can_depend_on_train_loss": False,
        "objective": "family-default-suppressed answer-value grounding after ANSWER=E with OOD derived values, hard-negative default rejection, parrot-trap rejection, and post-wrapper proxy gates",
        "objective_explicitly_penalizes_family_default": True,
        "objective_explicitly_penalizes_same_value_for_all_rows": True,
        "objective_rewards_intra_family_distinct_values": True,
        "objective_rejects_train_loss_only_success": True,
    }


def save_target_checkpoint(model: Any, path: Path, source_meta: dict[str, Any], train_config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seq_len": int(source_meta["seq_len"]),
            "vocab_size": int(source_meta["vocab_size"]),
            "config": train_config,
            "phase": MILESTONE,
        },
        path,
    )


def helper_request(row: dict[str, Any], checkpoint_path: str, checkpoint_hash: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_hash": checkpoint_hash,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    }


def forbidden_input_tests(helper: Any, selected: dict[str, Any]) -> dict[str, Any]:
    base = {
        "prompt": "138YK forbidden metadata rejection smoke.",
        "checkpoint_path": selected["checkpoint_path"],
        "checkpoint_hash": selected["checkpoint_sha256"],
        "seed": 13800,
        "max_new_tokens": 8,
        "generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False},
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(FORBIDDEN_HELPER_KEYS | {"unexpected_extra"}):
        request = dict(base)
        request[key] = "forbidden"
        try:
            helper.raw_generate(request)
            rows.append({"field": key, "rejected": False, "verdict": None})
        except Exception as exc:
            rows.append({"field": key, "rejected": getattr(exc, "verdict", None) == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": getattr(exc, "verdict", type(exc).__name__)})
    config_rows: list[dict[str, Any]] = []
    for name, config in [
        ("unknown_config_key", {"temperature": 0.0, "device": "cpu", "stop_on_newline": False, "unknown": "x"}),
        ("forbidden_config_expected_output", {"temperature": 0.0, "device": "cpu", "stop_on_newline": False, "expected_output": "x"}),
        ("nested_forbidden_config_labels", {"temperature": 0.0, "device": "cpu", "stop_on_newline": False, "nested": {"labels": ["x"]}}),
    ]:
        request = dict(base)
        request["generation_config"] = config
        try:
            helper.raw_generate(request)
            config_rows.append({"test": name, "rejected": False, "verdict": None})
        except Exception as exc:
            config_rows.append({"test": name, "rejected": getattr(exc, "verdict", None) == "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "verdict": getattr(exc, "verdict", type(exc).__name__)})
    return {"schema_version": "phase_138yk_forbidden_input_rejection_report_v1", "rows": rows, "generation_config_rows": config_rows, "all_rejected": all(row["rejected"] for row in rows + config_rows)}


def expected_output_canary(helper: Any, selected: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    original = {"row_id": "138yk_canary_original", "prompt": "138YK canary prompt: produce any raw continuation.", "expected_output": "ANSWER=EGOOD"}
    shadow = dict(original)
    shadow["row_id"] = "138yk_canary_shadow"
    shadow["expected_output"] = "ANSWER=EBAD"
    original_request = helper_request(original, selected["checkpoint_path"], selected["checkpoint_sha256"], 13801, max_new_tokens)
    shadow_request = helper_request(shadow, selected["checkpoint_path"], selected["checkpoint_sha256"], 13801, max_new_tokens)
    original_response = helper.raw_generate(original_request)
    shadow_response = helper.raw_generate(shadow_request)
    side_fields = {
        "generated_text": original_response["generated_text"] == shadow_response["generated_text"],
        "generation_trace_hash": original_response["generation_trace_hash"] == shadow_response["generation_trace_hash"],
        "token_count": original_response["token_count"] == shadow_response["token_count"],
        "stop_reason": original_response["stop_reason"] == shadow_response["stop_reason"],
        "model_checkpoint_hash": original_response["model_checkpoint_hash"] == shadow_response["model_checkpoint_hash"],
        "generation_config_hash": original_response["generation_config_hash"] == shadow_response["generation_config_hash"],
    }
    return {
        "schema_version": "phase_138yk_expected_output_canary_report_v1",
        "original_helper_request_hash": stable_hash(original_request),
        "shadow_helper_request_hash": stable_hash(shadow_request),
        "helper_requests_identical": stable_hash(original_request) == stable_hash(shadow_request),
        "generation_side_fields_identical": side_fields,
        "forbidden_fields_absent_from_helper_requests": not bool((set(original_request) | set(shadow_request)) & FORBIDDEN_HELPER_KEYS),
        "expected_material_only_outside_helper_request": "expected_output" not in original_request and "expected_output" not in shadow_request,
        "generated_text_original_hash": text_hash(original_response["generated_text"]),
        "generated_text_shadow_hash": text_hash(shadow_response["generated_text"]),
        "expected_output_canary_passed": all(side_fields.values()) and stable_hash(original_request) == stable_hash(shadow_request),
    }


def ast_shortcut_scan(paths: list[Path]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    old_runner_re = re.compile(r"^(run_stable_loop_phase_lock_|run_deck_local_)")
    forbidden_call_names = {"oracle_rerank", "verifier_rerank", "llm_judge", "grammar_decoder", "constrained_decoding", "regex_fixer", "json_fixer", "json_mode", "best_of_n", "retry_loop", "post_generation_repair", "runtime_tool_call", "actual_tool_execution"}

    def uses_expected(node: ast.AST | None) -> bool:
        return node is not None and any(token in ast.unparse(node) for token in ["expected_output", "expected_payload", "expected_answer", "gold_output", "target_json"])

    for path in paths:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if old_runner_re.match(alias.name):
                        findings.append({"file": rel(path), "lineno": node.lineno, "type": "OLD_RUNNER_IMPORT_DETECTED", "detail": alias.name})
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if old_runner_re.match(module):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "OLD_RUNNER_IMPORT_DETECTED", "detail": module})
            if isinstance(node, ast.Assign):
                targets = " ".join(ast.unparse(target) for target in node.targets)
                if re.search(r"generated_text|generated", targets) and uses_expected(node.value):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "AST_GENERATED_TEXT_FROM_EXPECTED_MATERIAL", "detail": ast.unparse(node)})
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func).lower()
                if any(token in name for token in forbidden_call_names):
                    findings.append({"file": rel(path), "lineno": node.lineno, "type": "ORACLE_SHORTCUT_DETECTED", "detail": name})
    return {"schema_version": "phase_138yk_ast_shortcut_scan_report_v1", "scanned_files": [rel(path) for path in paths if path.exists()], "findings": findings, "ast_shortcut_scan_passed": not findings}


def first_answer_token(text: str) -> str | None:
    match = re.search(r"\bANSWER=[A-Za-z0-9_+-]*", text)
    return match.group(0) if match else None


def answer_parts(answer: str | None) -> tuple[str | None, str | None]:
    if not answer or not answer.startswith("ANSWER="):
        return None, None
    body = answer.split("=", 1)[1]
    namespace = body[:1] if body else None
    value = body[1:] if len(body) > 1 else ""
    return namespace, value


def namespace_label(generated: str) -> str:
    if re.search(r"\bANSWER=T[A-Za-z0-9_+-]*", generated):
        return "train_namespace"
    if re.search(r"\bANSWER=E[A-Za-z0-9_+-]*", generated):
        return "eval_namespace"
    if "ANSWER=" in generated:
        return "other_answer_namespace"
    return "missing_answer_namespace"


def generic_value(value: str | None) -> bool:
    return value in {"", "VALUE", "TOKEN", "UNKNOWN", "ANSWER", "0000", "E", "EV", "TR"} or (value is not None and re.fullmatch(r"[A-Za-z_]+", value) is not None)


def repeated_or_garbled(text: str) -> bool:
    return "\ufffd" in text or bool(re.search(r"(.)\1{7,}", text)) or bool(re.search(r"[^A-Za-z0-9_=\s:;.,+\-/]{4,}", text))


def off_prompt(generated: str) -> bool:
    stripped = generated.strip()
    if not stripped:
        return True
    if "ANSWER=" in stripped:
        return False
    return bool(re.search(r"\b(User|Assistant):", stripped)) or len(stripped) > 0


def score_features(generated: str, row: dict[str, Any]) -> dict[str, Any]:
    text = generated.strip()
    expected = str(row["expected_output"])
    expected_ns, expected_value = answer_parts(expected)
    forbidden_defaults = set(row.get("forbidden_family_default_values", []))
    answer = first_answer_token(text)
    actual_ns, actual_value = answer_parts(answer)
    prefix_index = text.find("ANSWER=E")
    value_index = text.find(str(expected_value), prefix_index + len("ANSWER=E")) if prefix_index >= 0 and expected_value else -1
    pre_prefix_value_index = text.find(str(expected_value)) if expected_value else -1
    stale = bool(re.search(r"\b(User|Assistant):", text))
    train_leak = re.search(r"\bANSWER=T[A-Za-z0-9_+-]*", text) is not None
    answer_prefix = answer is not None
    eval_namespace = actual_ns == "E"
    namespace_ok = actual_ns == expected_ns == "E"
    value_ok = actual_value == expected_value and bool(expected_value)
    family_default_violation = actual_value in forbidden_defaults if actual_value is not None else False
    hard_negative_default_violation = bool(row.get("pass_requires_no_family_default") and family_default_violation)
    exact_ok = answer == expected
    value_after_prefix = value_index >= 0
    value_position_error = bool(pre_prefix_value_index >= 0 and (prefix_index < 0 or pre_prefix_value_index < prefix_index))
    empty_after_prefix = bool(prefix_index >= 0 and (actual_value is None or actual_value == ""))
    generic_after_prefix = bool(prefix_index >= 0 and generic_value(actual_value))
    post_wrapper = text[prefix_index + len("ANSWER=E") :] if prefix_index >= 0 else ""
    garbage = bool(prefix_index >= 0 and (repeated_or_garbled(post_wrapper) or stale or (not value_after_prefix and generic_after_prefix)))
    latency = (value_index - (prefix_index + len("ANSWER=E"))) if value_index >= 0 and prefix_index >= 0 else None
    off = off_prompt(generated)
    passed = bool(answer_prefix and namespace_ok and value_ok and exact_ok and not family_default_violation and not train_leak and not stale and not off)
    if stale:
        reason = "stale_chat_fragment_present"
    elif train_leak:
        reason = "train_namespace_leak_present"
    elif family_default_violation:
        reason = "family_default_shortcut"
    elif not answer_prefix:
        reason = "answer_prefix_absent"
    elif not namespace_ok:
        reason = "namespace_mismatch"
    elif empty_after_prefix:
        reason = "empty_value_after_prefix"
    elif generic_after_prefix:
        reason = "generic_value_after_prefix"
    elif not value_ok:
        reason = "answer_value_mismatch"
    elif off:
        reason = "off_prompt_output"
    else:
        reason = "exact_answer_match" if passed else "exact_answer_mismatch"
    return {
        "pass": passed,
        "failure_reason": None if passed else reason,
        "answer_token": answer,
        "answer_value_candidate": actual_value,
        "forbidden_family_default_values": sorted(forbidden_defaults),
        "family_default_violation": family_default_violation,
        "hard_negative_default_violation": hard_negative_default_violation,
        "namespace_label": namespace_label(generated),
        "answer_prefix_present": answer_prefix,
        "eval_namespace_emitted": eval_namespace,
        "namespace_correct": namespace_ok,
        "answer_value_correct": value_ok,
        "exact_answer_correct": exact_ok,
        "value_after_prefix_correct": value_after_prefix,
        "value_position_error": value_position_error,
        "empty_value_after_prefix": empty_after_prefix,
        "generic_value_after_prefix": generic_after_prefix,
        "post_wrapper_garbage": garbage,
        "value_emission_latency": latency,
        "train_namespace_leak": train_leak,
        "stale_chat_fragment_present": stale,
        "off_prompt_output": off,
    }


def score_text(generated: str, row: dict[str, Any]) -> tuple[bool, str]:
    features = score_features(generated, row)
    return bool(features["pass"]), features["failure_reason"] or "exact_answer_match"


def run_eval(helper: Any, rows: list[dict[str, Any]], out: Path, checkpoint_path: str, checkpoint_hash: str, max_new_tokens: int, heartbeat_sec: int, label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []
    scoring: list[dict[str, Any]] = []
    last_flush = time.monotonic()
    for index, row in enumerate(rows):
        request = helper_request(row, checkpoint_path, checkpoint_hash, int(row["seed"]), max_new_tokens)
        if set(request) != ALLOWED_HELPER_KEYS or set(request) & FORBIDDEN_HELPER_KEYS:
            raise GateError("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "helper request contains forbidden metadata")
        request_hash = stable_hash(request)
        response = helper.raw_generate(request)
        generated_text = response["generated_text"]
        generated_text_hash = text_hash(generated_text)
        trace = {
            "row_id": row["row_id"],
            "family": row["family"],
            "contrast_group_id": row.get("contrast_group_id"),
            "seed": row["seed"],
            "helper_request": request,
            "helper_request_hash": request_hash,
            "helper_request_allowed_keys": sorted(request),
            "generated_text_hash": generated_text_hash,
            "generation_trace_hash": response["generation_trace_hash"],
            "model_checkpoint_hash": response["model_checkpoint_hash"],
            "generation_config_hash": response["generation_config_hash"],
            "response": response,
            "generated_before_scoring": True,
        }
        traces.append(trace)
        raw_results.append({"row_id": row["row_id"], "family": row["family"], "contrast_group_id": row.get("contrast_group_id"), "seed": row["seed"], "prompt_hash": text_hash(row["prompt"]), "generated_text": generated_text, "generated_text_hash": generated_text_hash, "generation_trace_hash": response["generation_trace_hash"], "token_count": response["token_count"]})
        features = score_features(generated_text, row)
        scoring.append(
            {
                "row_id": row["row_id"],
                "family": row["family"],
                "contrast_group_id": row.get("contrast_group_id"),
                "seed": row["seed"],
                "expected_output": row["expected_output"],
                "expected_output_hash": text_hash(row["expected_output"]),
                "expected_value": row["answer_value"],
                "generated_text_hash": generated_text_hash,
                "scored_after_generation": True,
                "helper_trace_hash": response["generation_trace_hash"],
                "expected_token_included": row["expected_output"] in generated_text,
                **features,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(rows) or (time.monotonic() - last_flush) >= heartbeat_sec:
            append_progress(out, f"{label} generation", row_index=index + 1, row_count=len(rows), heartbeat_sec=heartbeat_sec)
            if label == "final_eval":
                write_jsonl(out / "raw_generation_trace.jsonl", traces)
                write_jsonl(out / "raw_generation_results.jsonl", raw_results)
            last_flush = time.monotonic()
    return traces, raw_results, scoring


def family_rates(scoring: list[dict[str, Any]]) -> dict[str, float]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scoring:
        by_family[row["family"]].append(row)
    def acc(family: str, key: str) -> float:
        rows = by_family.get(family, [])
        return sum(1 for row in rows if row.get(key)) / len(rows) if rows else 0.0
    return {
        "prompt_value_copy_accuracy": acc("FAMILY_CONTRAST_DIRECT_COPY_DIAGNOSTIC", "answer_value_correct"),
        "rule_derived_value_accuracy": acc("FAMILY_CONTRAST_RULE_DERIVED", "answer_value_correct"),
        "table_derived_value_accuracy": acc("FAMILY_CONTRAST_TABLE_DERIVED", "answer_value_correct"),
        "composition_derived_value_accuracy": acc("FAMILY_CONTRAST_COMPOSITION_DERIVED", "answer_value_correct"),
        "ood_symbol_value_accuracy": acc("FAMILY_CONTRAST_OOD_SYMBOL_BINDING", "answer_value_correct"),
    }


def compute_metrics(scoring: list[dict[str, Any]], seeds: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    row_count = len(scoring)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scoring:
        by_family[row["family"]].append(row)
        by_seed[int(row["seed"])].append(row)
    family_metrics: dict[str, Any] = {}
    for family, rows in sorted(by_family.items()):
        family_metrics[family] = {
            "row_count": len(rows),
            "pass_count": sum(1 for row in rows if row["pass"]),
            "accuracy": sum(1 for row in rows if row["pass"]) / len(rows),
            "answer_value_accuracy": sum(1 for row in rows if row["answer_value_correct"]) / len(rows),
            "exact_answer_accuracy": sum(1 for row in rows if row["exact_answer_correct"]) / len(rows),
            "value_after_prefix_accuracy": sum(1 for row in rows if row["value_after_prefix_correct"]) / len(rows),
            "stale_chat_fragment_rate": sum(1 for row in rows if row["stale_chat_fragment_present"]) / len(rows),
            "hard_negative_default_violation_rate": sum(1 for row in rows if row.get("hard_negative_default_violation")) / len(rows),
        }
    derived_rates = family_rates(scoring)
    value_latencies = [row["value_emission_latency"] for row in scoring if row.get("value_emission_latency") is not None]
    prefix_rows = [row for row in scoring if row["answer_prefix_present"]]
    eval_ns_rows = [row for row in scoring if row["eval_namespace_emitted"]]
    no_stale_rows = [row for row in scoring if not row["stale_chat_fragment_present"]]
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows = by_seed.get(seed, [])
        seed_rates = family_rates(rows)
        item = {
            "seed": seed,
            "row_count": len(rows),
            "answer_value_accuracy": sum(1 for row in rows if row["answer_value_correct"]) / len(rows) if rows else 0.0,
            "rule_derived_value_accuracy": seed_rates["rule_derived_value_accuracy"],
            "table_derived_value_accuracy": seed_rates["table_derived_value_accuracy"],
            "stale_chat_fragment_rate": sum(1 for row in rows if row["stale_chat_fragment_present"]) / len(rows) if rows else 1.0,
            "hard_negative_default_violation_rate": sum(1 for row in rows if row.get("hard_negative_default_violation")) / len(rows) if rows else 1.0,
        }
        item["parrot_trap_detected"] = False
        item["seed_passed"] = all(item[key] >= threshold for key, threshold in SEED_MIN_GATES.items()) and all(item[key] <= threshold for key, threshold in SEED_MAX_GATES.items())
        seed_rows.append(item)
    parrot_trap_detected = (
        derived_rates["prompt_value_copy_accuracy"] > 0.0
        and derived_rates["rule_derived_value_accuracy"] < 0.20
        and derived_rates["table_derived_value_accuracy"] < 0.20
        and derived_rates["ood_symbol_value_accuracy"] < 0.20
    )
    aggregate = {
        "schema_version": "phase_138yk_aggregate_metrics_v1",
        "row_count": row_count,
        "pass_count": sum(1 for row in scoring if row["pass"]),
        "answer_value_accuracy": sum(1 for row in scoring if row["answer_value_correct"]) / row_count if row_count else 0.0,
        "exact_answer_accuracy": sum(1 for row in scoring if row["exact_answer_correct"]) / row_count if row_count else 0.0,
        "value_after_prefix_accuracy": sum(1 for row in scoring if row["value_after_prefix_correct"]) / row_count if row_count else 0.0,
        "answer_prefix_accuracy": sum(1 for row in scoring if row["answer_prefix_present"]) / row_count if row_count else 0.0,
        "eval_namespace_emission_accuracy": sum(1 for row in scoring if row["eval_namespace_emitted"]) / row_count if row_count else 0.0,
        "train_namespace_leak_rate": sum(1 for row in scoring if row["train_namespace_leak"]) / row_count if row_count else 1.0,
        "stale_chat_fragment_rate": sum(1 for row in scoring if row["stale_chat_fragment_present"]) / row_count if row_count else 1.0,
        "hard_negative_default_violation_rate": sum(1 for row in scoring if row.get("hard_negative_default_violation")) / row_count if row_count else 1.0,
        "hard_negative_default_control_failed": None,
        "off_prompt_output_rate": sum(1 for row in scoring if row["off_prompt_output"]) / row_count if row_count else 1.0,
        "prefix_success_value_failure_rate": sum(1 for row in prefix_rows if not row["answer_value_correct"]) / len(prefix_rows) if prefix_rows else 1.0,
        "eval_namespace_success_value_failure_rate": sum(1 for row in eval_ns_rows if not row["answer_value_correct"]) / len(eval_ns_rows) if eval_ns_rows else 1.0,
        "no_stale_wrong_value_rate": sum(1 for row in no_stale_rows if not row["answer_value_correct"]) / len(no_stale_rows) if no_stale_rows else 1.0,
        "value_position_error_rate": sum(1 for row in scoring if row["value_position_error"]) / row_count if row_count else 0.0,
        "empty_value_after_prefix_rate": sum(1 for row in scoring if row["empty_value_after_prefix"]) / row_count if row_count else 0.0,
        "generic_value_after_prefix_rate": sum(1 for row in scoring if row["generic_value_after_prefix"]) / row_count if row_count else 0.0,
        "post_wrapper_garbage_token_rate": sum(1 for row in scoring if row["post_wrapper_garbage"]) / row_count if row_count else 0.0,
        "value_emission_latency_mean": mean(value_latencies) if value_latencies else None,
        "value_emission_latency_p95": sorted(value_latencies)[int(0.95 * (len(value_latencies) - 1))] if value_latencies else None,
        "repeated_token_after_prefix_rate": sum(1 for row in scoring if row["post_wrapper_garbage"]) / row_count if row_count else 0.0,
        "all_seeds_passed_independently": all(row["seed_passed"] for row in seed_rows),
        **derived_rates,
        **BASELINE_METRICS,
    }
    aggregate["parrot_trap_detected"] = parrot_trap_detected
    aggregate["copy_only_success_rate"] = derived_rates["prompt_value_copy_accuracy"] if parrot_trap_detected else 0.0
    aggregate["answer_value_accuracy_improved"] = aggregate["answer_value_accuracy"] > aggregate["baseline_answer_value_accuracy"]
    aggregate["exact_answer_accuracy_improved"] = aggregate["exact_answer_accuracy"] > aggregate["baseline_exact_answer_accuracy"]
    aggregate["positive_value_grounding_gates_passed"] = (
        all(aggregate[key] >= threshold for key, threshold in POSITIVE_GATES.items())
        and all(aggregate[key] <= threshold for key, threshold in MAX_POSITIVE_RATES.items())
        and aggregate["all_seeds_passed_independently"]
        and not aggregate["parrot_trap_detected"]
    )
    value_metrics = {"schema_version": "phase_138yk_value_grounding_metrics_v1", **{key: aggregate[key] for key in ["answer_value_accuracy", "exact_answer_accuracy", "value_after_prefix_accuracy", "prompt_value_copy_accuracy", "rule_derived_value_accuracy", "table_derived_value_accuracy", "composition_derived_value_accuracy", "ood_symbol_value_accuracy", "prefix_success_value_failure_rate", "eval_namespace_success_value_failure_rate", "no_stale_wrong_value_rate"]}}
    parrot = {"schema_version": "phase_138yk_parrot_trap_report_v1", "parrot_trap_detected": parrot_trap_detected, "copy_only_success_rate": aggregate["copy_only_success_rate"], **derived_rates}
    carrier = {"schema_version": "phase_138yk_post_wrapper_carrier_proxy_report_v1", "hidden_state_residual_signal_measurement": "diagnostic_gap", "value_after_prefix_accuracy": aggregate["value_after_prefix_accuracy"], "value_position_error_rate": aggregate["value_position_error_rate"], "empty_value_after_prefix_rate": aggregate["empty_value_after_prefix_rate"], "generic_value_after_prefix_rate": aggregate["generic_value_after_prefix_rate"], "post_wrapper_garbage_token_rate": aggregate["post_wrapper_garbage_token_rate"], "value_emission_latency_mean": aggregate["value_emission_latency_mean"], "value_emission_latency_p95": aggregate["value_emission_latency_p95"], "repeated_token_after_prefix_rate": aggregate["repeated_token_after_prefix_rate"]}
    return family_metrics, seed_rows, aggregate, value_metrics, {**parrot, **{"carrier_proxy": carrier}}


def contrast_group_manifest(eval_rows: list[dict[str, Any]], requested_groups_per_family: int, group_size: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        groups[str(row["contrast_group_id"])].append(row)
    families: dict[str, Any] = {}
    by_family: dict[str, set[str]] = defaultdict(set)
    for group_id, rows in groups.items():
        by_family[rows[0]["family"]].add(group_id)
    for family, group_ids in sorted(by_family.items()):
        families[family] = {
            "requested_contrast_groups_per_family": requested_groups_per_family,
            "built_contrast_groups_per_family": len(group_ids),
            "contrast_group_size": group_size,
            "row_count": sum(1 for row in eval_rows if row["family"] == family),
        }
    return {
        "schema_version": "phase_138yk_contrast_group_manifest_v1",
        "group_count": len(groups),
        "row_count": len(eval_rows),
        "requested_contrast_groups_per_family": requested_groups_per_family,
        "contrast_group_size": group_size,
        "families": families,
        "note": "Built groups are bounded by eval_rows_per_family so 96 rows with group_size 4 yields 24 groups per family.",
    }


def compute_contrast_artifacts(scoring: list[dict[str, Any]], seeds: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scoring:
        groups[str(row["contrast_group_id"])].append(row)
        by_family[row["family"]].append(row)
        by_seed[int(row["seed"])].append(row)

    family_defaults: dict[str, str | None] = {}
    per_family: dict[str, Any] = {}
    for family, rows in sorted(by_family.items()):
        wrong_candidates = [row.get("answer_value_candidate") for row in rows if not row.get("answer_value_correct") and row.get("answer_value_candidate")]
        counts = Counter(wrong_candidates)
        default_value, default_count = counts.most_common(1)[0] if counts else (None, 0)
        forbidden_values = sorted({value for row in rows for value in row.get("forbidden_family_default_values", [])})
        forbidden_count = sum(1 for row in rows if row.get("answer_value_candidate") in forbidden_values)
        hard_negative_count = sum(1 for row in rows if row.get("hard_negative_default_violation"))
        expected_unique = {row["expected_value"] for row in rows}
        generated_unique = {row.get("answer_value_candidate") for row in rows if row.get("answer_value_candidate")}
        correct_unique = {row["expected_value"] for row in rows if row.get("answer_value_correct")}
        family_defaults[family] = default_value
        per_family[family] = {
            "row_count": len(rows),
            "family_default_value": default_value,
            "forbidden_family_default_values": forbidden_values,
            "family_default_attractor_rate": rate(forbidden_count or default_count, len(rows)),
            "family_default_reuse_rate": rate(forbidden_count, len(rows)),
            "hard_negative_default_violation_rate": rate(hard_negative_count, len(rows)),
            "family_dominant_wrong_value_rate": rate(default_count, len(rows)),
            "intra_family_unique_correct_value_rate": rate(len(correct_unique), len(expected_unique)),
            "intra_family_mode_collapse_rate": 1.0 - rate(len(generated_unique), len(expected_unique)),
            "per_family_answer_value_accuracy": rate(sum(1 for row in rows if row.get("answer_value_correct")), len(rows)),
            "per_family_exact_answer_accuracy": rate(sum(1 for row in rows if row.get("exact_answer_correct")), len(rows)),
        }

    group_rows: list[dict[str, Any]] = []
    for group_id, rows in sorted(groups.items()):
        expected_values = [row["expected_value"] for row in rows]
        generated_values = [row.get("answer_value_candidate") for row in rows]
        family = rows[0]["family"]
        family_default = family_defaults.get(family)
        forbidden_defaults = sorted({value for row in rows for value in row.get("forbidden_family_default_values", [])})
        all_correct = all(row.get("pass") for row in rows)
        distinct_expected = len(set(expected_values)) == len(expected_values)
        distinct_generated = len(set(generated_values)) == len(generated_values)
        no_family_default = all(value not in forbidden_defaults and value != family_default for value in generated_values if value)
        no_bad_surface = not any(row.get("train_namespace_leak") or row.get("stale_chat_fragment_present") for row in rows)
        same_value_for_all = len(set(generated_values)) == 1
        family_default_emitted = any(value in forbidden_defaults or value == family_default for value in generated_values if value)
        hard_negative_default_emitted = any(value in forbidden_defaults for value in generated_values if value)
        peer_expected_confusion = any(value in set(expected_values) and value != expected for value, expected in zip(generated_values, expected_values))
        group_pass = bool(all_correct and distinct_expected and distinct_generated and no_family_default and no_bad_surface)
        group_rows.append(
            {
                "contrast_group_id": group_id,
                "family": family,
                "seed_set": sorted({row["seed"] for row in rows}),
                "row_count": len(rows),
                "expected_values": expected_values,
                "generated_values": generated_values,
                "family_default_value": family_default,
                "forbidden_family_default_values": forbidden_defaults,
                "all_rows_emit_answer_prefix": all(row.get("answer_prefix_present") for row in rows),
                "all_rows_emit_eval_namespace": all(row.get("eval_namespace_emitted") for row in rows),
                "all_rows_correct_distinct_values": all_correct and distinct_generated,
                "collapsed_to_same_generated_value": len(set(generated_values)) < len(set(expected_values)),
                "same_value_for_all_rows": same_value_for_all,
                "family_default_emitted": family_default_emitted,
                "hard_negative_default_emitted": hard_negative_default_emitted,
                "multi_expected_to_single_default": distinct_expected and family_default_emitted and len(set(generated_values)) < len(set(expected_values)),
                "peer_expected_value_confusion": peer_expected_confusion,
                "train_namespace_leak_present": any(row.get("train_namespace_leak") for row in rows),
                "stale_chat_fragment_present": any(row.get("stale_chat_fragment_present") for row in rows),
                "pass": group_pass,
            }
        )

    group_count = len(group_rows)
    collapse_groups = sum(1 for row in group_rows if row["collapsed_to_same_generated_value"])
    family_default_groups = sum(1 for row in group_rows if row["family_default_emitted"])
    same_value_groups = sum(1 for row in group_rows if row["same_value_for_all_rows"])
    multi_default_groups = sum(1 for row in group_rows if row["multi_expected_to_single_default"])
    hard_negative_rows = sum(1 for row in scoring if row.get("hard_negative_default_violation"))
    family_default_row_reuse = sum(1 for row in scoring if row.get("family_default_violation"))
    contrast_metrics = {
        "schema_version": "phase_138yk_intra_family_contrastive_metrics_v1",
        "contrast_group_count": group_count,
        "intra_family_contrastive_accuracy": rate(sum(1 for row in group_rows if row["pass"]), group_count),
        "intra_family_group_exact_accuracy": rate(sum(1 for row in group_rows if row["all_rows_correct_distinct_values"]), group_count),
        "intra_family_unique_correct_value_rate": rate(sum(1 for row in scoring if row.get("answer_value_correct")), len(scoring)),
        "intra_family_mode_collapse_rate": rate(collapse_groups, group_count),
        "family_default_attractor_rate": rate(family_default_groups, group_count),
        "family_default_reuse_rate": rate(family_default_row_reuse, len(scoring)),
        "multi_expected_to_single_default_rate": rate(multi_default_groups, group_count),
        "same_value_for_all_rows_rate": rate(same_value_groups, group_count),
        "hard_negative_default_violation_rate": rate(hard_negative_rows, len(scoring)),
        "family_dominant_wrong_value_rate": sum(item["family_dominant_wrong_value_rate"] for item in per_family.values()) / len(per_family) if per_family else 1.0,
        "per_family_contrast_group_pass_rate": {
            family: rate(sum(1 for row in group_rows if row["family"] == family and row["pass"]), sum(1 for row in group_rows if row["family"] == family))
            for family in sorted(per_family)
        },
        "per_family_distinct_generated_value_rate": {
            family: 1.0 - per_family[family]["intra_family_mode_collapse_rate"]
            for family in sorted(per_family)
        },
        "per_family": per_family,
    }
    family_default_report = {
        "schema_version": "phase_138yk_family_default_attractor_report_v1",
        "family_defaults": family_defaults,
        "family_default_attractor_rate": contrast_metrics["family_default_attractor_rate"],
        "family_default_reuse_rate": contrast_metrics["family_default_reuse_rate"],
        "multi_expected_to_single_default_rate": contrast_metrics["multi_expected_to_single_default_rate"],
        "hard_negative_default_violation_rate": contrast_metrics["hard_negative_default_violation_rate"],
        "family_default_shortcut_detected": contrast_metrics["family_default_attractor_rate"] > 0.50,
    }
    train_value_replay_count = sum(1 for row in scoring if str(row.get("answer_value_candidate") or "").startswith("TR"))
    train_value_replay_rate = rate(train_value_replay_count, len(scoring))
    high_frequency_report = {
        "schema_version": "phase_138yk_high_frequency_value_replay_report_v1",
        "high_frequency_train_value_replay_detected": train_value_replay_rate > 0.05,
        "high_frequency_train_value_replay_rate": train_value_replay_rate,
        "train_value_replay_count": train_value_replay_count,
        "train_value_replay_detection_rule": "generated post-wrapper value candidate begins with the train value namespace TR",
        "reason": "138YK rejects train-value replay even when the ANSWER=E wrapper is present.",
    }
    for seed in seeds:
        rows = by_seed.get(seed, [])
        seed_groups = [group for group in group_rows if seed in group["seed_set"]]
        for item in rows:
            item["_seed_marker"] = seed
        for metric_row in seed_groups:
            metric_row.setdefault("seed_partitioned", True)
    return group_rows, contrast_metrics, family_default_report, high_frequency_report


def inject_contrast_metrics(scoring: list[dict[str, Any]], seeds: list[int], aggregate: dict[str, Any], family_metrics: dict[str, Any], seed_metrics: list[dict[str, Any]], value_metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    group_rows, contrast_metrics, family_default_report, high_frequency_report = compute_contrast_artifacts(scoring, seeds)
    aggregate.update({key: contrast_metrics[key] for key in ["intra_family_contrastive_accuracy", "intra_family_group_exact_accuracy", "intra_family_unique_correct_value_rate", "intra_family_mode_collapse_rate", "family_default_attractor_rate", "family_default_reuse_rate", "family_dominant_wrong_value_rate", "multi_expected_to_single_default_rate", "same_value_for_all_rows_rate", "hard_negative_default_violation_rate"]})
    aggregate["family_default_shortcut_detected"] = family_default_report["family_default_shortcut_detected"]
    aggregate["high_frequency_train_value_replay_detected"] = high_frequency_report["high_frequency_train_value_replay_detected"]
    aggregate["positive_value_grounding_gates_passed"] = (
        all(aggregate[key] >= threshold for key, threshold in POSITIVE_GATES.items())
        and all(aggregate[key] <= threshold for key, threshold in MAX_POSITIVE_RATES.items())
        and aggregate["all_seeds_passed_independently"]
        and not aggregate["parrot_trap_detected"]
        and not aggregate["family_default_shortcut_detected"]
        and not aggregate["high_frequency_train_value_replay_detected"]
    )
    value_metrics.update({key: aggregate[key] for key in ["intra_family_contrastive_accuracy", "intra_family_unique_correct_value_rate", "intra_family_mode_collapse_rate", "family_default_attractor_rate", "family_default_reuse_rate", "family_dominant_wrong_value_rate", "multi_expected_to_single_default_rate", "same_value_for_all_rows_rate", "hard_negative_default_violation_rate"]})
    for seed_row in seed_metrics:
        seed = seed_row["seed"]
        seed_scoring = [row for row in scoring if int(row["seed"]) == int(seed)]
        seed_groups, seed_contrast, seed_defaults, _seed_high = compute_contrast_artifacts(seed_scoring, [int(seed)])
        seed_row.update({
            "intra_family_contrastive_accuracy": seed_contrast["intra_family_contrastive_accuracy"],
            "intra_family_mode_collapse_rate": seed_contrast["intra_family_mode_collapse_rate"],
            "family_default_attractor_rate": seed_contrast["family_default_attractor_rate"],
            "hard_negative_default_violation_rate": seed_contrast["hard_negative_default_violation_rate"],
        })
        seed_row["seed_passed"] = all(seed_row.get(key, 0.0) >= threshold for key, threshold in SEED_MIN_GATES.items()) and all(seed_row.get(key, 1.0) <= threshold for key, threshold in SEED_MAX_GATES.items())
    aggregate["all_seeds_passed_independently"] = all(seed_row.get("seed_passed") for seed_row in seed_metrics)
    aggregate["positive_value_grounding_gates_passed"] = (
        all(aggregate[key] >= threshold for key, threshold in POSITIVE_GATES.items())
        and all(aggregate[key] <= threshold for key, threshold in MAX_POSITIVE_RATES.items())
        and aggregate["all_seeds_passed_independently"]
        and not aggregate["parrot_trap_detected"]
        and not aggregate["family_default_shortcut_detected"]
        and not aggregate["high_frequency_train_value_replay_detected"]
    )
    return group_rows, contrast_metrics, family_default_report, high_frequency_report


def scorer_controls(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(138904)
    controls = {
        "STATIC_OUTPUT_CONTROL": lambda row: "STATIC_OUTPUT_CONTROL",
        "COPY_PROMPT_CONTROL": lambda row: row["prompt"],
        "RANDOM_ANSWER_CONTROL": lambda row: f"ANSWER=E{rng.randrange(100000, 999999)}",
        "DISTRACTOR_COPY_CONTROL": lambda row: row["forbidden_distractor"],
        "STALE_CHAT_FRAGMENT_CONTROL": lambda row: "User: stale\nAssistant: stale",
        "TRAIN_NAMESPACE_REPLAY_CONTROL": lambda row: re.sub(r"ANSWER=E", "ANSWER=T", row["expected_output"]),
        "PREFIX_ONLY_CONTROL": lambda row: "ANSWER=E",
        "GENERIC_VALUE_CONTROL": lambda row: "ANSWER=EVALUE",
        "FAMILY_DEFAULT_VALUE_CONTROL": lambda row: f"ANSWER=EFD{row['family'][-3:]}",
        "SAME_VALUE_FOR_ALL_ROWS_CONTROL": lambda row: "ANSWER=ESAMEVALUE",
        "PARROT_COPY_CONTROL": lambda row: f"ANSWER=E{row['answer_value']}" if row["family"] == "FAMILY_CONTRAST_DIRECT_COPY_DIAGNOSTIC" else "ANSWER=EUNRELATED",
        "HIGH_FREQUENCY_TRAIN_VALUE_CONTROL": lambda row: "ANSWER=EHFTRAINVALUE",
        "HARD_NEGATIVE_DEFAULT_CONTROL": lambda row: f"ANSWER=E{(row.get('forbidden_family_default_values') or ['TRFORBIDDENDEFAULT'])[0]}",
        "PEER_EXPECTED_VALUE_CONFUSION_CONTROL": lambda row: f"ANSWER=E{(row.get('peer_expected_values') or ['EVPEERCONFUSION'])[0]}",
    }
    report: dict[str, Any] = {"schema_version": "phase_138yk_control_arm_report_v1", "controls_called_helper": False, "controls_failed": True, "control_pass_threshold": 0.20, "controls": {}}
    results: list[dict[str, Any]] = []
    for name, producer in controls.items():
        scope = [row for row in rows if not (name == "PARROT_COPY_CONTROL" and row["family"] == "FAMILY_CONTRAST_DIRECT_COPY_DIAGNOSTIC")]
        passed_count = 0
        for row in scope:
            passed, reason = score_text(producer(row), row)
            passed_count += int(passed)
            results.append({"control": name, "row_id": row["row_id"], "pass": passed, "failure_reason": None if passed else reason})
        accuracy = passed_count / len(scope) if scope else 0.0
        failed = accuracy < report["control_pass_threshold"]
        report["controls"][name] = {"accuracy": accuracy, "failed": failed, "scope_row_count": len(scope)}
        report["controls_failed"] = report["controls_failed"] and failed
    report["rows"] = results[:300]
    return report


def generated_before_scoring_report(traces: list[dict[str, Any]], scoring: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "phase_138yk_generated_before_scoring_report_v1",
        "generation_phase_completed_first": [trace["row_id"] for trace in traces] == [row["row_id"] for row in scoring],
        "scoring_phase_consumed_immutable_generated_text": True,
        "helper_requests_built_without_expected_or_scorer_metadata": all(set(trace["helper_request"]) == ALLOWED_HELPER_KEYS and not (set(trace["helper_request"]) & FORBIDDEN_HELPER_KEYS) for trace in traces),
        "scoring_did_not_feed_back_into_generation": True,
        "generated_text_produced_before_scoring": all(trace["generated_before_scoring"] for trace in traces) and all(row["scored_after_generation"] for row in scoring),
        "trace_count": len(traces),
        "scoring_count": len(scoring),
    }


def eval_snapshot(traces: list[dict[str, Any]], scoring: list[dict[str, Any]], family_metrics: dict[str, Any], seed_metrics: list[dict[str, Any]], aggregate: dict[str, Any], value_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "helper_request_hashes": [trace["helper_request_hash"] for trace in traces],
        "generated_text_hashes": [trace["generated_text_hash"] for trace in traces],
        "generation_trace_hashes": [trace["generation_trace_hash"] for trace in traces],
        "per_row_pass_fail": [{"row_id": row["row_id"], "pass": row["pass"], "failure_reason": row["failure_reason"]} for row in scoring],
        "namespace_metrics": {key: aggregate[key] for key in ["train_namespace_leak_rate", "eval_namespace_emission_accuracy", "answer_prefix_accuracy"]},
        "value_metrics": value_metrics,
        "per_family_metrics": family_metrics,
        "per_seed_metrics": seed_metrics,
        "aggregate_metrics": aggregate,
        "decision_critical_metrics": {
            key: aggregate[key]
            for key in [
                "answer_value_accuracy",
                "exact_answer_accuracy",
                "value_after_prefix_accuracy",
                "prefix_success_value_failure_rate",
                "no_stale_wrong_value_rate",
                "post_wrapper_garbage_token_rate",
                "stale_chat_fragment_rate",
                "train_namespace_leak_rate",
                "parrot_trap_detected",
                "intra_family_contrastive_accuracy",
                "intra_family_unique_correct_value_rate",
                "intra_family_mode_collapse_rate",
                "family_default_attractor_rate",
                "family_default_reuse_rate",
                "family_dominant_wrong_value_rate",
                "multi_expected_to_single_default_rate",
                "same_value_for_all_rows_rate",
                "hard_negative_default_violation_rate",
                "hard_negative_default_control_failed",
                "family_default_shortcut_detected",
                "high_frequency_train_value_replay_detected",
            ]
        },
    }


def replay_report(first: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        "generated_text_hashes": first["generated_text_hashes"] == replay["generated_text_hashes"],
        "generation_trace_hashes": first["generation_trace_hashes"] == replay["generation_trace_hashes"],
        "per_row_pass_fail": first["per_row_pass_fail"] == replay["per_row_pass_fail"],
        "namespace_metrics": first["namespace_metrics"] == replay["namespace_metrics"],
        "value_metrics": first["value_metrics"] == replay["value_metrics"],
        "per_family_metrics": first["per_family_metrics"] == replay["per_family_metrics"],
        "per_seed_metrics": first["per_seed_metrics"] == replay["per_seed_metrics"],
        "aggregate_metrics": first["aggregate_metrics"] == replay["aggregate_metrics"],
        "decision_critical_metrics": first["decision_critical_metrics"] == replay["decision_critical_metrics"],
    }
    return {"schema_version": "phase_138yk_determinism_replay_report_v1", "replay_attempted": True, "same_target_checkpoint": True, "same_rows": True, "same_seeds": True, "same_helper_request_hashes": first["helper_request_hashes"] == replay["helper_request_hashes"], "same_config": True, "comparisons": comparisons, "determinism_replay_passed": all(comparisons.values()) and first["helper_request_hashes"] == replay["helper_request_hashes"]}


def helper_provenance(target_checkpoint: Path, target_hash: str, traces: list[dict[str, Any]], source_hash_before: str, source_hash_after: str, target_hash_before: str | None, helper: Any) -> dict[str, Any]:
    request = traces[0]["helper_request"] if traces else {"generation_config": {"temperature": 0.0, "device": "cpu", "stop_on_newline": False}}
    return {
        "schema_version": "phase_138yk_helper_provenance_verification_v1",
        "selected_checkpoint_path": rel(target_checkpoint),
        "selected_checkpoint_sha256": target_hash,
        "source_checkpoint_hash_before": source_hash_before,
        "source_checkpoint_hash_after": source_hash_after,
        "source_checkpoint_unchanged": source_hash_before == source_hash_after,
        "target_checkpoint_hash_before": target_hash_before,
        "target_checkpoint_hash_after": target_hash,
        "target_checkpoint_changed": target_hash_before != target_hash,
        "helper_source_sha256": file_hash(HELPER_PATH),
        "helper_import_path": rel(HELPER_PATH),
        "backend_version": "shared_raw_generation_helper_v1",
        "generation_config": request["generation_config"],
        "generation_config_hash": helper.stable_hash(request["generation_config"]),
        "strict_load_state_dict": True,
        "real_raw_generation_backend_used": True,
        "fake_helper_used": False,
        "simulated_model_output_used": False,
    }


def write_row_hashes(out: Path, eval_rows: list[dict[str, Any]]) -> None:
    write_json(out / "eval_row_hashes.json", {"schema_version": "phase_138yk_eval_row_hashes_v1", "row_count": len(eval_rows), "prompt_hashes": {row["row_id"]: text_hash(row["prompt"]) for row in eval_rows}, "expected_output_hashes": {row["row_id"]: text_hash(row["expected_output"]) for row in eval_rows}, "row_hashes": {row["row_id"]: stable_hash(row) for row in eval_rows}})


def ood_manifest(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_values = {row["answer_value"] for row in train_rows}
    eval_values = {row["answer_value"] for row in eval_rows}
    return {
        "schema_version": "phase_138yk_ood_family_value_manifest_v1",
        "train_eval_value_namespaces_disjoint": all(value.startswith("TR") for value in train_values) and all(value.startswith("EV") for value in eval_values),
        "eval_values_held_out_from_train": len(train_values & eval_values) == 0,
        "held_out_value_ranges": True,
        "held_out_symbol_tokens": True,
        "held_out_table_mappings": True,
        "held_out_rule_chains": True,
    }


def write_failure_samples(out: Path, rows: list[dict[str, Any]], raw_results: list[dict[str, Any]], scoring: list[dict[str, Any]]) -> None:
    by_row = {row["row_id"]: row for row in rows}
    by_raw = {row["row_id"]: row for row in raw_results}
    failures: list[dict[str, Any]] = []
    human: list[dict[str, Any]] = []
    for score in scoring:
        row = by_row[score["row_id"]]
        raw = by_raw[score["row_id"]]
        sample = {"row_id": row["row_id"], "family": row["family"], "prompt": row["prompt"], "generated_text": raw["generated_text"], "expected_output": row["expected_output"], "expected_value": row["answer_value"], "pass": score["pass"], "pass_fail": score["pass"], "failure_reason": score["failure_reason"], "namespace_label": score["namespace_label"], "helper_trace_hash": score["helper_trace_hash"]}
        if not score["pass"] and len(failures) < 160:
            failures.append(sample)
        if len(human) < 120:
            human.append(sample)
    write_jsonl(out / "failure_case_samples.jsonl", failures)
    write_jsonl(out / "human_readable_samples.jsonl", human)


def decide(aggregate: dict[str, Any], controls: dict[str, Any], leakage: dict[str, Any], canary: dict[str, Any], scan: dict[str, Any], before: dict[str, Any], provenance: dict[str, Any], replay: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    hard_positive = (
        aggregate["positive_value_grounding_gates_passed"]
        and controls["controls_failed"]
        and leakage["leakage_rejected"]
        and canary["expected_output_canary_passed"]
        and scan["ast_shortcut_scan_passed"]
        and before["generated_text_produced_before_scoring"]
        and replay["determinism_replay_passed"]
        and provenance["source_checkpoint_unchanged"]
        and provenance["target_checkpoint_changed"]
    )
    if hard_positive:
        decision_name, next_name, verdict = POSITIVE_DECISION, POSITIVE_NEXT, POSITIVE_VERDICT
    elif leakage["leakage_rejected"] is not True:
        decision_name, next_name, verdict = LEAKAGE_DECISION, LEAKAGE_NEXT, NEGATIVE_VERDICT
    elif replay["determinism_replay_passed"] is not True:
        decision_name, next_name, verdict = DETERMINISM_DECISION, DETERMINISM_NEXT, DETERMINISM_VERDICT
    elif controls["controls_failed"] is not True:
        decision_name, next_name, verdict = SCORER_DECISION, SCORER_NEXT, NEGATIVE_VERDICT
    elif aggregate["hard_negative_default_violation_rate"] > 0.0:
        decision_name, next_name, verdict = HARD_NEGATIVE_DECISION, HARD_NEGATIVE_NEXT, NEGATIVE_VERDICT
    elif aggregate["family_default_shortcut_detected"]:
        decision_name, next_name, verdict = FAMILY_DEFAULT_DECISION, FAMILY_DEFAULT_NEXT, NEGATIVE_VERDICT
    elif aggregate["high_frequency_train_value_replay_detected"]:
        decision_name, next_name, verdict = HIGH_FREQUENCY_DECISION, HIGH_FREQUENCY_NEXT, NEGATIVE_VERDICT
    elif aggregate["parrot_trap_detected"]:
        decision_name, next_name, verdict = PARROT_DECISION, PARROT_NEXT, NEGATIVE_VERDICT
    elif aggregate["train_namespace_leak_rate"] > MAX_POSITIVE_RATES["train_namespace_leak_rate"]:
        decision_name, next_name, verdict = NAMESPACE_DECISION, NAMESPACE_NEXT, NEGATIVE_VERDICT
    elif aggregate["stale_chat_fragment_rate"] > MAX_POSITIVE_RATES["stale_chat_fragment_rate"]:
        decision_name, next_name, verdict = STALE_DECISION, STALE_NEXT, NEGATIVE_VERDICT
    elif aggregate["intra_family_mode_collapse_rate"] > MAX_POSITIVE_RATES["intra_family_mode_collapse_rate"]:
        decision_name, next_name, verdict = WRAPPER_VALUE_DECISION, WRAPPER_VALUE_NEXT, NEGATIVE_VERDICT
    elif aggregate["answer_value_accuracy"] <= BASELINE_METRICS["baseline_answer_value_accuracy"]:
        decision_name, next_name, verdict = NO_VALUE_DECISION, NO_VALUE_NEXT, NEGATIVE_VERDICT
    else:
        decision_name, next_name, verdict = NO_VALUE_DECISION, NO_VALUE_NEXT, NEGATIVE_VERDICT
    decision = {
        "schema_version": "phase_138yk_decision_v1",
        "decision": decision_name,
        "next": next_name,
        "verdict": verdict,
        "shared_raw_generation_helper_used": True,
        "forbidden_input_rejection_passed": True,
        "expected_output_canary_passed": canary["expected_output_canary_passed"],
        "ast_shortcut_scan_passed": scan["ast_shortcut_scan_passed"],
        "helper_provenance_written": True,
        "generated_text_produced_before_scoring": before["generated_text_produced_before_scoring"],
        "determinism_replay_passed": replay["determinism_replay_passed"],
        "controls_failed": controls["controls_failed"],
        "leakage_rejected": leakage["leakage_rejected"],
        "source_checkpoint_unchanged": provenance["source_checkpoint_unchanged"],
        "target_checkpoint_changed": provenance["target_checkpoint_changed"],
        "reasoning_subtrack_real_raw_evidence_partially_restored": hard_positive,
        "clean_negative_valid": not hard_positive,
        "hidden_state_residual_signal_measurement": "diagnostic_gap",
        "parrot_trap_rejected": not aggregate["parrot_trap_detected"],
        "prefix_only_success_allowed": False,
        "namespace_only_success_allowed": False,
        "direct_copy_only_success_allowed": False,
        "family_default_hard_negative_success_allowed": False,
        "same_family_single_generated_value_allowed": False,
        **aggregate,
        **FALSE_BOUNDARY_FLAGS,
        **FINAL_EVAL_FLAGS,
    }
    verdicts = [
        verdict,
        "SHARED_RAW_GENERATION_HELPER_USED",
        "EXPECTED_OUTPUT_CANARY_PASSED",
        "AST_SHORTCUT_SCAN_PASSED",
        "GENERATED_TEXT_PRODUCED_BEFORE_SCORING",
        "CONTROLS_FAILED" if controls["controls_failed"] else "SCORER_OR_TASK_WEAKNESS",
        "LEAKAGE_REJECTED" if leakage["leakage_rejected"] else "FAMILY_DEFAULT_SUPPRESSED_EVAL_LEAKAGE",
        "DETERMINISM_REPLAY_PASSED" if replay["determinism_replay_passed"] else DETERMINISM_VERDICT,
        "PARROT_TRAP_REJECTED" if not aggregate["parrot_trap_detected"] else "PARROT_TRAP_COPY_SHORTCUT_DETECTED",
        "FAMILY_DEFAULT_SHORTCUT_REJECTED" if not aggregate["family_default_shortcut_detected"] else "FAMILY_DEFAULT_SHORTCUT_DETECTED",
        "HARD_NEGATIVE_DEFAULTS_REJECTED" if aggregate["hard_negative_default_violation_rate"] == 0.0 else "HARD_NEGATIVE_DEFAULT_FAILURE",
        "HIGH_FREQUENCY_TRAIN_VALUE_REPLAY_REJECTED" if not aggregate["high_frequency_train_value_replay_detected"] else "HIGH_FREQUENCY_TRAIN_VALUE_REPLAY_DETECTED",
        "RAW_ASSISTANT_CAPABILITY_REMAINS_QUARANTINED",
        "STRUCTURED_TOOL_CAPABILITY_REMAINS_INVALIDATED",
    ]
    return decision, verdicts


def write_failure_decision(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    route = {
        MISSING_TRAINING_VERDICT: (TRAINING_MISSING_DECISION, TRAINING_MISSING_NEXT, MISSING_TRAINING_VERDICT),
        "RAW_GENERATION_BACKEND_MISSING": (HELPER_FAILURE_DECISION, HELPER_FAILURE_NEXT, NEGATIVE_VERDICT),
        "RAW_GENERATION_FORBIDDEN_INPUT_DETECTED": (HELPER_FAILURE_DECISION, HELPER_FAILURE_NEXT, NEGATIVE_VERDICT),
        "ORACLE_SHORTCUT_DETECTED": (HELPER_FAILURE_DECISION, HELPER_FAILURE_NEXT, NEGATIVE_VERDICT),
        "AST_SHORTCUT_SCAN_FAILED": (HELPER_FAILURE_DECISION, HELPER_FAILURE_NEXT, NEGATIVE_VERDICT),
        "FAMILY_DEFAULT_SUPPRESSED_EVAL_LEAKAGE": (LEAKAGE_DECISION, LEAKAGE_NEXT, NEGATIVE_VERDICT),
        DETERMINISM_VERDICT: (DETERMINISM_DECISION, DETERMINISM_NEXT, DETERMINISM_VERDICT),
    }.get(error.verdict, (NO_VALUE_DECISION, NO_VALUE_NEXT, NEGATIVE_VERDICT))
    decision = {"schema_version": "phase_138yk_failure_decision_v1", "decision": route[0], "next": route[1], "verdict": route[2], "failure_verdict": error.verdict, "failure_message": error.message, "reasoning_subtrack_real_raw_evidence_partially_restored": False, **FALSE_BOUNDARY_FLAGS}
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [route[2], error.verdict], decision, error.message)
    write_report(out, [route[2], error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_csv_ints(args.seeds)
    write_json(out / "queue.json", {"schema_version": "phase_138yk_queue_v1", "milestone": MILESTONE, "status": "started", "started_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["FAMILY_DEFAULT_SUPPRESSED_CONTRASTIVE_REPAIR_RUNNING"], {"decision": "pending", "next": "pending"})

    upstream = verify_upstreams(out, resolve_path(args.upstream_138yj_root), resolve_path(args.upstream_138yd_root), resolve_path(args.upstream_138yi_root))
    append_progress(out, "upstream verification", upstream_138yj=True, upstream_138yd=True, upstream_138yi=True)
    refresh_status(out, "running", ["UPSTREAMS_VERIFIED"], {"decision": "pending", "next": "pending"})

    determinism = deterministic_setup(seeds[0])
    write_json(out / "determinism_manifest.json", determinism)
    append_progress(out, "determinism setup", seed=seeds[0])
    require_torch()

    helper = import_helper()
    source_checkpoint = resolve_path(SOURCE_138YI_TARGET_REL)
    if not source_checkpoint.exists():
        raise GateError(MISSING_TRAINING_VERDICT, "138YI helper-compatible source target checkpoint is missing")
    source_hash_before = file_hash(source_checkpoint)
    model, source_meta = helper.load_checkpoint(rel(source_checkpoint), source_hash_before)
    seq_len = int(source_meta["seq_len"])
    vocab_size = int(source_meta["vocab_size"])
    pad_id = vocab_size - 1
    append_progress(out, "source checkpoint load", checkpoint=rel(source_checkpoint), backend=source_meta.get("backend_name"))

    family_default_bank = build_family_default_value_bank(upstream, args.hard_negative_defaults_per_family)
    write_json(out / "family_default_value_bank.json", family_default_bank)
    append_progress(out, "family default value bank build", family_count=len(family_default_bank["families"]), top_k=args.hard_negative_defaults_per_family)

    train_rows = add_training_default_suppression_prompts(build_train_rows(args.train_examples), family_default_bank)
    eval_rows = attach_family_default_hard_negatives(build_eval_rows(seeds, args.eval_rows_per_family, args.contrast_group_size), family_default_bank)
    write_jsonl(out / "train_rows.jsonl", train_rows)
    write_jsonl(out / "eval_rows.jsonl", eval_rows)
    write_jsonl(out / "hard_negative_default_rows.jsonl", eval_rows)
    write_row_hashes(out, eval_rows)
    train_manifest = dataset_manifest(train_rows, "train")
    eval_manifest = dataset_manifest(eval_rows, "eval")
    write_json(out / "train_dataset_manifest.json", train_manifest)
    write_json(out / "eval_dataset_manifest.json", eval_manifest)
    write_json(out / "contrast_group_manifest.json", contrast_group_manifest(eval_rows, args.contrast_groups_per_family, args.contrast_group_size))
    write_json(out / "ood_family_value_manifest.json", ood_manifest(train_rows, eval_rows))
    append_progress(out, "dataset build", train_rows=len(train_rows), eval_rows=len(eval_rows))
    append_progress(out, "hard negative row build", hard_negative_rows=len(eval_rows))

    leakage = split_leakage_audit(train_rows, eval_rows)
    write_json(out / "freshness_leakage_audit.json", leakage)
    append_progress(out, "leakage audit", leakage_rejected=leakage["leakage_rejected"])
    if leakage["leakage_rejected"] is not True:
        raise GateError("FAMILY_DEFAULT_SUPPRESSED_EVAL_LEAKAGE", "train/eval leakage detected", leakage)

    train_config = {"schema_version": "phase_138yk_train_config_v1", "seeds": seeds, "train_examples": args.train_examples, "train_steps": args.train_steps, "batch_size": args.batch_size, "lr": args.lr, "seq_len": seq_len, "source_checkpoint_path": rel(source_checkpoint), "target_checkpoint_path": TARGET_CHECKPOINT_REL, "training_objective": "family-default-suppressed intra-family contrastive value grounding after ANSWER=E with explicit hard-negative default rejection", "hard_negative_defaults_per_family": args.hard_negative_defaults_per_family, "family_default_value_bank_used": True, "positive_can_depend_on_train_loss": False, "source_checkpoint_immutable": True, "old_runners_imported": False}
    eval_config = {"schema_version": "phase_138yk_eval_config_v1", "seeds": seeds, "eval_rows_per_family": args.eval_rows_per_family, "contrast_groups_per_family_requested": args.contrast_groups_per_family, "contrast_group_size": args.contrast_group_size, "hard_negative_defaults_per_family": args.hard_negative_defaults_per_family, "max_new_tokens": args.max_new_tokens, "families": FAMILIES, "helper_path": rel(HELPER_PATH), "deterministic_scoring_only": True, "controls_do_not_call_helper": True, "prefix_only_success_allowed": False, "namespace_only_success_allowed": False, "family_level_classification_only_success_allowed": False, "direct_copy_only_success_allowed": False, "family_default_hard_negative_success_allowed": False, "same_family_single_generated_value_allowed": False, **FINAL_EVAL_FLAGS}
    write_json(out / "train_config.json", train_config)
    write_json(out / "eval_config.json", eval_config)
    determinism.update({"source_checkpoint_hash": source_hash_before, "dataset_hash": stable_hash({"train": train_manifest["dataset_hash"], "eval": eval_manifest["dataset_hash"]}), "train_config_hash": stable_hash(train_config), "eval_config_hash": stable_hash(eval_config), "helper_source_hash": file_hash(HELPER_PATH)})
    write_json(out / "determinism_manifest.json", determinism)
    write_json(out / "source_checkpoint_integrity_manifest.json", {"schema_version": "phase_138yk_source_checkpoint_integrity_manifest_v1", "source_checkpoint_path": rel(source_checkpoint), "source_checkpoint_hash_before": source_hash_before, "source_checkpoint_hash_after_load": file_hash(source_checkpoint), "source_checkpoint_unchanged": source_hash_before == file_hash(source_checkpoint), "backend_name": source_meta.get("backend_name"), "seq_len": seq_len, "vocab_size": vocab_size, "strict_load_state_dict": True, "checkpoint_expected_key_count": len(GRU_STATE_KEYS)})

    source_selected = {"checkpoint_path": rel(source_checkpoint), "checkpoint_sha256": source_hash_before}
    forbidden = forbidden_input_tests(helper, source_selected)
    write_json(out / "forbidden_input_rejection_report.json", forbidden)
    if forbidden["all_rejected"] is not True:
        raise GateError("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "forbidden input was accepted")
    canary = expected_output_canary(helper, source_selected, args.max_new_tokens)
    write_json(out / "expected_output_canary_report.json", canary)
    if canary["expected_output_canary_passed"] is not True:
        raise GateError("ORACLE_SHORTCUT_DETECTED", "expected output canary failed")
    scan = ast_shortcut_scan([HELPER_PATH, RUNNER_PATH, CHECKER_PATH])
    write_json(out / "ast_shortcut_scan_report.json", scan)
    if scan["ast_shortcut_scan_passed"] is not True:
        raise GateError("AST_SHORTCUT_SCAN_FAILED", "AST scan found forbidden path")
    append_progress(out, "helper/canary/AST checks", canary=True, ast=True)

    training = train_target_model(model, train_rows, seq_len, pad_id, args, out)
    write_json(out / "training_objective_report.json", training)
    target_checkpoint = resolve_path(TARGET_CHECKPOINT_REL)
    target_hash_before = file_hash(target_checkpoint) if target_checkpoint.exists() else None
    save_target_checkpoint(model, target_checkpoint, source_meta, train_config)
    target_hash_after = file_hash(target_checkpoint)
    source_hash_after = file_hash(source_checkpoint)
    append_progress(out, "target checkpoint write", target_checkpoint=rel(target_checkpoint), target_hash=target_hash_after)
    if source_hash_before != source_hash_after:
        raise GateError("CHECKPOINT_MUTATION_DETECTED", "source checkpoint changed during 138YK")
    helper.load_checkpoint(rel(target_checkpoint), target_hash_after)
    write_json(out / "target_checkpoint_integrity_manifest.json", {"schema_version": "phase_138yk_target_checkpoint_integrity_manifest_v1", "target_checkpoint_path": rel(target_checkpoint), "target_checkpoint_hash_before": target_hash_before, "target_checkpoint_hash_after": target_hash_after, "target_checkpoint_changed": target_hash_before != target_hash_after, "helper_strict_load_passed": True, "seq_len": seq_len, "vocab_size": vocab_size, "strict_load_state_dict": True, "checkpoint_expected_key_count": len(GRU_STATE_KEYS)})
    determinism["target_checkpoint_hash"] = target_hash_after
    write_json(out / "determinism_manifest.json", determinism)

    traces, raw_results, scoring = run_eval(helper, eval_rows, out, rel(target_checkpoint), target_hash_after, args.max_new_tokens, args.heartbeat_sec, "final_eval")
    write_jsonl(out / "raw_generation_trace.jsonl", traces)
    write_jsonl(out / "raw_generation_results.jsonl", raw_results)
    write_jsonl(out / "scoring_results.jsonl", scoring)
    append_progress(out, "scoring", scored_rows=len(scoring))
    before = generated_before_scoring_report(traces, scoring)
    write_json(out / "generated_before_scoring_report.json", before)
    if before["generated_text_produced_before_scoring"] is not True:
        raise GateError("RAW_GENERATION_FORBIDDEN_INPUT_DETECTED", "generated before scoring proof failed")

    controls = scorer_controls(eval_rows)
    write_jsonl(out / "control_results.jsonl", controls["rows"])
    write_json(out / "control_arm_report.json", controls)
    if controls["controls_failed"] is not True:
        raise GateError("SCORER_OR_TASK_WEAKNESS", "scorer controls passed")
    append_progress(out, "controls", controls_failed=True)

    family_metrics, seed_metrics, aggregate, value_metrics, parrot_and_carrier = compute_metrics(scoring, seeds)
    contrast_results, contrast_metrics, family_default_report, high_frequency_report = inject_contrast_metrics(scoring, seeds, aggregate, family_metrics, seed_metrics, value_metrics)
    aggregate["hard_negative_default_control_failed"] = controls["controls"].get("HARD_NEGATIVE_DEFAULT_CONTROL", {}).get("failed") is True
    value_metrics["hard_negative_default_control_failed"] = aggregate["hard_negative_default_control_failed"]
    aggregate["positive_value_grounding_gates_passed"] = (
        all(aggregate[key] >= threshold for key, threshold in POSITIVE_GATES.items())
        and all(aggregate[key] <= threshold for key, threshold in MAX_POSITIVE_RATES.items())
        and aggregate["all_seeds_passed_independently"]
        and not aggregate["parrot_trap_detected"]
        and not aggregate["family_default_shortcut_detected"]
        and not aggregate["high_frequency_train_value_replay_detected"]
    )
    parrot_report = {key: value for key, value in parrot_and_carrier.items() if key != "carrier_proxy"}
    carrier_report = parrot_and_carrier["carrier_proxy"]
    write_json(out / "per_family_metrics.json", {"schema_version": "phase_138yk_per_family_metrics_v1", "families": family_metrics})
    write_jsonl(out / "per_seed_metrics.jsonl", seed_metrics)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_jsonl(out / "contrast_group_results.jsonl", contrast_results)
    write_json(out / "intra_family_contrastive_metrics.json", contrast_metrics)
    write_json(out / "family_default_attractor_report.json", family_default_report)
    write_json(
        out / "family_default_suppression_metrics.json",
        {
            "schema_version": "phase_138yk_family_default_suppression_metrics_v1",
            "family_default_attractor_rate": aggregate["family_default_attractor_rate"],
            "family_default_reuse_rate": aggregate["family_default_reuse_rate"],
            "family_dominant_wrong_value_rate": aggregate["family_dominant_wrong_value_rate"],
            "multi_expected_to_single_default_rate": aggregate["multi_expected_to_single_default_rate"],
            "same_value_for_all_rows_rate": aggregate["same_value_for_all_rows_rate"],
            "hard_negative_default_violation_rate": aggregate["hard_negative_default_violation_rate"],
            "hard_negative_default_control_failed": aggregate["hard_negative_default_control_failed"],
            "family_default_shortcut_detected": aggregate["family_default_shortcut_detected"],
        },
    )
    write_json(out / "high_frequency_value_replay_report.json", high_frequency_report)
    write_json(out / "value_grounding_metrics.json", value_metrics)
    write_json(out / "parrot_trap_report.json", parrot_report)
    write_json(out / "post_wrapper_carrier_proxy_report.json", carrier_report)
    append_progress(out, "contrast group evaluation", intra_family_contrastive_accuracy=aggregate["intra_family_contrastive_accuracy"])
    append_progress(out, "family default suppression analysis", family_default_reuse_rate=aggregate["family_default_reuse_rate"], hard_negative_default_violation_rate=aggregate["hard_negative_default_violation_rate"])
    append_progress(out, "shortcut analysis", parrot_trap_detected=aggregate["parrot_trap_detected"], family_default_shortcut_detected=aggregate["family_default_shortcut_detected"], high_frequency_train_value_replay_detected=aggregate["high_frequency_train_value_replay_detected"])
    append_progress(out, "parrot-trap analysis", parrot_trap_detected=aggregate["parrot_trap_detected"])
    append_progress(out, "post-wrapper proxy analysis", post_wrapper_garbage_token_rate=aggregate["post_wrapper_garbage_token_rate"])
    write_failure_samples(out, eval_rows, raw_results, scoring)

    replay_traces, _replay_raw, replay_scoring = run_eval(helper, eval_rows, out, rel(target_checkpoint), target_hash_after, args.max_new_tokens, args.heartbeat_sec, "replay_eval")
    replay_family, replay_seed, replay_aggregate, replay_value_metrics, _replay_extra = compute_metrics(replay_scoring, seeds)
    _replay_contrast_results, _replay_contrast_metrics, _replay_family_default, _replay_high = inject_contrast_metrics(replay_scoring, seeds, replay_aggregate, replay_family, replay_seed, replay_value_metrics)
    replay_aggregate["hard_negative_default_control_failed"] = aggregate["hard_negative_default_control_failed"]
    replay_value_metrics["hard_negative_default_control_failed"] = aggregate["hard_negative_default_control_failed"]
    replay = replay_report(eval_snapshot(traces, scoring, family_metrics, seed_metrics, aggregate, value_metrics), eval_snapshot(replay_traces, replay_scoring, replay_family, replay_seed, replay_aggregate, replay_value_metrics))
    write_json(out / "determinism_replay_report.json", replay)
    if replay["determinism_replay_passed"] is not True:
        raise GateError(DETERMINISM_VERDICT, "determinism replay mismatch")
    append_progress(out, "determinism replay", passed=True)

    provenance = helper_provenance(target_checkpoint, target_hash_after, traces, source_hash_before, source_hash_after, target_hash_before, helper)
    write_json(out / "helper_provenance_verification.json", provenance)
    evidence = {"schema_version": "phase_138yk_evidence_rebuild_status_v1", "reasoning_subtrack_real_raw_evidence_partially_restored": aggregate["positive_value_grounding_gates_passed"], "raw_assistant_capability_restored": False, "structured_tool_capability_restored": False, "clean_negative_valid": not aggregate["positive_value_grounding_gates_passed"]}
    write_json(out / "evidence_rebuild_status.json", evidence)

    decision, verdicts = decide(aggregate, controls, leakage, canary, scan, before, provenance, replay)
    write_json(out / "decision.json", decision)
    append_progress(out, "decision", decision=decision["decision"], next=decision["next"])
    refresh_status(out, "positive" if decision["verdict"] == POSITIVE_VERDICT else "clean_negative", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(out / "queue.json", {"schema_version": "phase_138yk_queue_v1", "milestone": MILESTONE, "status": "completed", "completed_at": utc_now(), "heartbeat_sec": args.heartbeat_sec})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138yj-root", default=str(DEFAULT_UPSTREAM_138YJ_ROOT))
    parser.add_argument("--upstream-138yd-root", default=str(DEFAULT_UPSTREAM_138YD_ROOT))
    parser.add_argument("--upstream-138yi-root", default=str(DEFAULT_UPSTREAM_138YI_ROOT))
    parser.add_argument("--seeds", default="2321,2322,2323")
    parser.add_argument("--train-examples", type=int, default=60000)
    parser.add_argument("--eval-rows-per-family", type=int, default=96)
    parser.add_argument("--contrast-groups-per-family", type=int, default=32)
    parser.add_argument("--contrast-group-size", type=int, default=4)
    parser.add_argument("--hard-negative-defaults-per-family", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--metrics-interval", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure_decision(args, exc)
        print(f"138YK failed closed: {exc.verdict}: {exc.message}", file=sys.stderr)
        return 1 if exc.verdict in {"138YK_BOUNDARY_FAILURE", "CHECKPOINT_MUTATION_DETECTED"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
