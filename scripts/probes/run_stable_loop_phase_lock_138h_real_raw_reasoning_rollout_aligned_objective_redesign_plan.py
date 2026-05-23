#!/usr/bin/env python3
"""138H artifact-only rollout-aligned objective redesign plan.

This phase reads existing 138R/138G/138GA artifacts only. It does not train,
run inference, call shared_raw_generation_helper.py, run torch forward passes,
mutate checkpoints, modify helper/backend code, import old runners, start
services, deploy, delete or consolidate files, or modify runtime/release/product
surfaces.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_138h_real_raw_reasoning_rollout_aligned_objective_redesign_plan/smoke")
DEFAULT_UPSTREAM_138GA_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138ga_objective_failure_ambiguity_resolution/smoke")
DEFAULT_UPSTREAM_138G_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138g_real_raw_reasoning_objective_failure_analysis/smoke")
DEFAULT_UPSTREAM_138R_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_138r_real_raw_reasoning_repair_training_plan_or_probe/smoke")
BOUNDARY_TEXT = (
    "138H is planning-only rollout-aligned objective redesign. It does not "
    "train, run inference, call shared_raw_generation_helper.py, run torch "
    "forward passes, mutate checkpoints, modify helper/backend code, import "
    "old runners, start services, deploy, delete or consolidate files, modify "
    "runtime/release/product surfaces, or change root LICENSE. It does not "
    "restore reasoning, raw assistant capability, structured/tool capability, "
    "GPT-like readiness, open-domain readiness, production chat, public API, "
    "deployment readiness, or safety alignment."
)
FALSE_FLAGS = {
    "reasoning_restored": False,
    "reasoning_subtrack_real_raw_evidence_partially_restored": False,
    "raw_assistant_capability_restored": False,
    "structured_tool_capability_restored": False,
    "gpt_like_readiness_claimed": False,
    "open_domain_assistant_readiness_claimed": False,
    "production_chat_claimed": False,
    "public_api_claimed": False,
    "deployment_readiness_claimed": False,
    "safety_alignment_claimed": False,
}
TAG_TYPES = {"artifact_observed", "computed_from_artifact", "diagnostic_gap", "inference"}
REQUIRED_138GA_ARTIFACTS = [
    "decision.json",
    "summary.json",
    "near_match_classification_report.json",
    "objective_failure_disambiguation.json",
    "meaningful_partial_answer_report.json",
    "upstream_138g_manifest.json",
    "upstream_138r_manifest.json",
]
REQUIRED_138G_ARTIFACTS = [
    "decision.json",
    "teacher_forcing_vs_rollout_report.json",
    "scoring_strictness_recheck.json",
]
REQUIRED_138R_ARTIFACTS = [
    "aggregate_metrics.json",
    "expected_output_canary_report.json",
    "ast_shortcut_scan_report.json",
    "control_arm_report.json",
    "freshness_leakage_audit.json",
    "determinism_replay_report.json",
    "helper_provenance_verification.json",
    "generated_before_scoring_report.json",
]


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
        raise GateError("138H_BOUNDARY_FAILURE", "--out must stay inside the repo") from exc
    parts = [part.lower() for part in relative.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("138H_BOUNDARY_FAILURE", "--out must stay under target/pilot_wave")
    if any(part == ".." for part in relative.parts):
        raise GateError("138H_BOUNDARY_FAILURE", "--out must not escape target/pilot_wave")
    return resolved


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def tagged(value: Any, evidence_type: str, source: str, note: str = "") -> dict[str, Any]:
    if evidence_type not in TAG_TYPES:
        raise ValueError(evidence_type)
    payload = {"value": value, "evidence_type": evidence_type, "source": source}
    if note:
        payload["note"] = note
    return payload


def gap(field: str, source: str, note: str) -> dict[str, Any]:
    return tagged(None, "diagnostic_gap", source, f"{field}: {note}")


def require_files(root: Path, names: list[str], verdict: str) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise GateError(verdict, "upstream artifacts missing", {"root": rel(root), "missing": missing})


def write_summary(out: Path, status: str, verdicts: list[str], decision: dict[str, Any], failure: str | None = None) -> None:
    write_json(
        out / "summary.json",
        {
            "schema_version": "phase_138h_summary_v1",
            "milestone": MILESTONE,
            "status": status,
            "verdicts": verdicts,
            "decision": decision.get("decision"),
            "next": decision.get("next"),
            "failure": failure,
            "boundary": BOUNDARY_TEXT,
            "artifact_only_planning": True,
            "training_performed": False,
            "new_model_inference_run": False,
            "shared_helper_called": False,
            "torch_forward_pass_run": False,
            "checkpoint_mutated": False,
            "runtime_surface_mutated": False,
            "root_license_changed": False,
            **FALSE_FLAGS,
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
            f"- `primary_bottleneck`: `{decision.get('primary_bottleneck')}`",
            f"- `no_capability_restored`: `{decision.get('no_capability_restored')}`",
            "",
            "138H is planning-only objective redesign.",
            "Reasoning is not restored.",
            "The reasoning subtrack real-raw evidence is not partially restored.",
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


def verify_138ga(out: Path, root: Path) -> dict[str, Any]:
    require_files(root, REQUIRED_138GA_ARTIFACTS, "UPSTREAM_138GA_ARTIFACT_MISSING")
    decision = read_json(root / "decision.json")
    summary = read_json(root / "summary.json")
    classification = read_json(root / "near_match_classification_report.json")
    disambiguation = read_json(root / "objective_failure_disambiguation.json")
    required = {
        "decision": decision.get("decision") == "objective_failure_disambiguated",
        "next": decision.get("next") == "138H_REAL_RAW_REASONING_ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN",
        "near_match_row_count": decision.get("near_match_row_count") == 38,
        "total_scored_row_count": decision.get("total_scored_row_count") == 960,
        "primary_label_counts": decision.get("primary_label_counts") == {"train_namespace_overlap": 38},
        "meaningful_near_match_rate": decision.get("meaningful_near_match_rate") == 0.0,
        "classification_counts": classification.get("primary_label_counts") == {"train_namespace_overlap": 38},
        "objective_failure_disambiguated": disambiguation.get("objective_failure_disambiguated") is True,
    }
    for key in FALSE_FLAGS:
        required[f"decision_{key}"] = decision.get(key) is False
        required[f"summary_{key}"] = summary.get(key) is False
    failed = [key for key, passed in required.items() if not passed]
    if failed:
        raise GateError("UPSTREAM_138GA_ARTIFACT_MISSING", "138GA did not match expected 138H route", {"failed": failed})
    manifest = {
        "schema_version": "phase_138h_upstream_138ga_manifest_v1",
        "upstream_138ga_root": rel(root),
        "upstream_138ga_verified": True,
        "decision": decision.get("decision"),
        "next": decision.get("next"),
        "near_match_row_count": decision.get("near_match_row_count"),
        "total_scored_row_count": decision.get("total_scored_row_count"),
        "primary_label_counts": decision.get("primary_label_counts"),
        "meaningful_near_match_rate": decision.get("meaningful_near_match_rate"),
        "objective_failure_disambiguated": True,
        **FALSE_FLAGS,
    }
    write_json(out / "upstream_138ga_manifest.json", manifest)
    return manifest


def verify_138g(out: Path, root: Path) -> dict[str, Any]:
    require_files(root, REQUIRED_138G_ARTIFACTS, "UPSTREAM_138G_ARTIFACT_MISSING")
    decision = read_json(root / "decision.json")
    teacher = read_json(root / "teacher_forcing_vs_rollout_report.json")
    required = {
        "decision": decision.get("decision") == "objective_failure_ambiguous",
        "teacher_initial_gap": teacher.get("fields", {}).get("teacher_forced_loss_initial", {}).get("evidence_type") == "diagnostic_gap",
        "teacher_final_gap": teacher.get("fields", {}).get("teacher_forced_loss_final", {}).get("evidence_type") == "diagnostic_gap",
    }
    failed = [key for key, passed in required.items() if not passed]
    if failed:
        raise GateError("UPSTREAM_138G_ARTIFACT_MISSING", "138G did not match expected 138H route", {"failed": failed})
    manifest = {
        "schema_version": "phase_138h_upstream_138g_manifest_v1",
        "upstream_138g_root": rel(root),
        "upstream_138g_verified": True,
        "decision": decision.get("decision"),
        "teacher_forced_loss_fields_diagnostic_gap": True,
        "train_loss_decreased": teacher.get("computed", {}).get("train_loss_decreased", {}).get("value"),
        "teacher_forced_loss_improved_claim_allowed": teacher.get("computed", {}).get("teacher_forced_loss_improved_claim_allowed", {}).get("value"),
    }
    write_json(out / "upstream_138g_manifest.json", manifest)
    return manifest


def verify_138r(out: Path, root: Path) -> dict[str, Any]:
    require_files(root, REQUIRED_138R_ARTIFACTS, "UPSTREAM_138R_ARTIFACT_MISSING")
    aggregate = read_json(root / "aggregate_metrics.json")
    canary = read_json(root / "expected_output_canary_report.json")
    scan = read_json(root / "ast_shortcut_scan_report.json")
    controls = read_json(root / "control_arm_report.json")
    leakage = read_json(root / "freshness_leakage_audit.json")
    replay = read_json(root / "determinism_replay_report.json")
    provenance = read_json(root / "helper_provenance_verification.json")
    generated_before = read_json(root / "generated_before_scoring_report.json")
    required = {
        "mean_accuracy_zero": aggregate.get("mean_real_raw_reasoning_accuracy") == 0.0,
        "expected_token_rate_zero": aggregate.get("expected_token_inclusion_rate") == 0.0,
        "canary_passed": canary.get("expected_output_canary_passed") is True,
        "ast_passed": scan.get("ast_shortcut_scan_passed") is True,
        "controls_failed": controls.get("controls_failed") is True,
        "leakage_rejected": leakage.get("leakage_rejected") is True,
        "determinism_replay_passed": replay.get("determinism_replay_passed") is True,
        "source_checkpoint_unchanged": provenance.get("source_checkpoint_unchanged") is True,
        "target_checkpoint_changed": provenance.get("target_checkpoint_changed") is True,
        "generated_before_scoring": generated_before.get("generated_text_produced_before_scoring") is True,
    }
    failed = [key for key, passed in required.items() if not passed]
    if failed:
        raise GateError("RAW_HELPER_INTEGRITY_FAILURE", "138R helper/eval integrity regression", {"failed": failed})
    manifest = {
        "schema_version": "phase_138h_upstream_138r_manifest_v1",
        "upstream_138r_root": rel(root),
        "upstream_138r_verified": True,
        "mean_real_raw_reasoning_accuracy": aggregate.get("mean_real_raw_reasoning_accuracy"),
        "expected_token_inclusion_rate": aggregate.get("expected_token_inclusion_rate"),
        "near_match_rate": aggregate.get("near_match_rate"),
        "helper_canary_ast_leakage_controls_determinism_passed": True,
        "source_checkpoint_unchanged": True,
        "target_checkpoint_changed": True,
    }
    write_json(out / "upstream_138r_manifest.json", manifest)
    return manifest


def build_rollout_alignment_failure_summary(ga: dict[str, Any], g: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_138h_rollout_alignment_failure_summary_v1",
        "primary_bottleneck": tagged("train_namespace_rollout_alignment_failure", "computed_from_artifact", "138GA decision + classification reports"),
        "near_match_disambiguation": tagged(ga["primary_label_counts"], "artifact_observed", "138GA decision.json.primary_label_counts"),
        "meaningful_near_match_rate": tagged(ga["meaningful_near_match_rate"], "artifact_observed", "138GA decision.json.meaningful_near_match_rate"),
        "rollout_accuracy": tagged(r["mean_real_raw_reasoning_accuracy"], "artifact_observed", "138R aggregate_metrics.json.mean_real_raw_reasoning_accuracy"),
        "expected_token_inclusion_rate": tagged(r["expected_token_inclusion_rate"], "artifact_observed", "138R aggregate_metrics.json.expected_token_inclusion_rate"),
        "teacher_forced_loss_fields": tagged("diagnostic_gap", "artifact_observed", "138G teacher_forcing_vs_rollout_report.json"),
        "train_loss_decreased": tagged(g.get("train_loss_decreased"), "artifact_observed", "138G teacher_forcing_vs_rollout_report.json.computed.train_loss_decreased"),
        "interpretation": tagged(
            "target checkpoint changed, but helper-only rollout emitted train namespace answer patterns instead of eval namespace answers",
            "computed_from_artifact",
            "138R/138G/138GA artifacts",
        ),
    }


def objective_redesign_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_138h_objective_redesign_requirements_v1",
        "evidence_type": "computed_from_artifact",
        "source": "138GA train_namespace_overlap + 138R zero rollout accuracy",
        "must_target": [
            "train namespace replay",
            "helper-only autoregressive free rollout",
            "answer prefix accuracy",
            "answer value accuracy",
            "stale chat rollout",
            "off-prompt continuation",
            "loss-only success rejection",
            "teacher-forcing-only success rejection",
        ],
        "must_not_optimize_for": [
            "teacher-forcing-only success",
            "loss-only success",
            "threshold weakening",
            "expected-output construction",
            "helper/backend modification to improve score",
            "post-generation repair",
        ],
    }


def namespace_policy() -> dict[str, Any]:
    return {
        "schema_version": "phase_138h_train_eval_namespace_policy_v1",
        "train_namespace": "ANSWER=T...",
        "eval_namespace": "ANSWER=E...",
        "forbidden_namespace_leakage": "ANSWER=T... on eval rows",
        "eval_requirement": "ANSWER=E... required on eval rows where applicable",
        "hard_failure_route": "138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS",
        "required_metrics_for_138i": [
            "train_namespace_leak_rate",
            "eval_namespace_emission_accuracy",
            "answer_prefix_accuracy",
            "answer_value_accuracy",
            "stale_chat_fragment_rate",
            "off_prompt_output_rate",
        ],
        "namespace_leak_hard_gate": {
            "train_namespace_leak_rate_must_be_zero_or_below_declared_floor": True,
            "ANSWER_T_eval_leak_is_not_progress": True,
        },
    }


def rollout_aligned_training_design() -> dict[str, Any]:
    return {
        "schema_version": "phase_138h_rollout_aligned_training_design_v1",
        "target_milestone": "138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE",
        "purpose": "repair/probe whether the checkpoint can learn eval-namespace task-bound answers under helper-only free rollout",
        "allowed_checkpoint_target": "target/pilot_wave/stable_loop_phase_lock_138i_real_raw_reasoning_rollout_aligned_repair_probe/smoke/checkpoints/target_138i_rollout_aligned_reasoning/model.pt",
        "source_checkpoint_immutable": True,
        "target_checkpoint_under_target_only": True,
        "objective_components": [
            {
                "name": "output_namespace_alignment",
                "punishes": ["generated ANSWER=T... when eval requires ANSWER=E..."],
                "metrics": ["train_namespace_leak_rate", "eval_namespace_emission_accuracy"],
            },
            {
                "name": "free_rollout_alignment",
                "punishes": ["train loss improves but helper-only final eval remains unchanged", "teacher-forcing-only success"],
                "metrics": ["helper_only_rollout_accuracy_delta", "generated_text_before_scoring"],
            },
            {
                "name": "scoring_format_discipline",
                "punishes": ["missing ANSWER prefix", "wrong answer value", "stale chat fragment", "off-prompt continuation"],
                "metrics": ["answer_prefix_accuracy", "answer_value_accuracy", "stale_chat_fragment_rate", "off_prompt_output_rate"],
            },
        ],
        "clean_negative_accepted": True,
    }


def final_eval_gate_design() -> dict[str, Any]:
    return {
        "schema_version": "phase_138h_final_eval_gate_design_v1",
        "shared_raw_generation_helper_only": True,
        "generated_text_before_scoring": True,
        "helper_request_contains_no_expected_or_scorer_metadata": True,
        "expected_output_canary_required": True,
        "ast_shortcut_scan_required": True,
        "deterministic_replay_required": True,
        "controls_must_fail": True,
        "leakage_rejected_required": True,
        "source_checkpoint_unchanged_required": True,
        "target_checkpoint_under_target_only": True,
        "clean_negative_routes": {
            "no_rollout_improvement": {
                "verdict": "REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_FAILS",
                "next": "138I_FAILURE_ANALYSIS",
            },
            "determinism_mismatch": {"next": "138N_DETERMINISM_FAILURE_ANALYSIS"},
            "namespace_leak_persists": {"next": "138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS"},
            "helper_integrity_fails": {"next": "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"},
        },
    }


def anti_shortcut_requirements() -> dict[str, Any]:
    return {
        "schema_version": "phase_138h_anti_shortcut_requirements_v1",
        "required": [
            "shared_raw_generation_helper.py only",
            "generated_text before scoring",
            "expected-output canary",
            "AST shortcut scan",
            "deterministic replay",
            "controls fail",
            "leakage rejected",
            "clean negative accepted",
        ],
        "explicitly_reject": [
            "teacher-forcing-only success",
            "loss-only success",
            "threshold weakening",
            "expected-output construction",
            "old runner imports",
            "helper/backend modification to improve score",
            "oracle/rerank/verifier/LLM judge",
            "constrained decoding",
            "JSON mode",
            "regex fixer",
            "post-generation repair",
            "retry loop",
            "best-of-n",
        ],
    }


def next_138i_plan(namespace: dict[str, Any], design: dict[str, Any], gates: dict[str, Any], anti_shortcut: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase_138h_next_138i_milestone_plan_v1",
        "milestone": "138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE",
        "type": "targeted repair/probe",
        "primary_bottleneck": "train_namespace_rollout_alignment_failure",
        "namespace_policy": namespace,
        "rollout_aligned_training_design": design,
        "final_eval_gate_design": gates,
        "anti_shortcut_requirements": anti_shortcut,
        "success_requires": [
            "train_namespace_leak_rate controlled by hard gate",
            "eval_namespace_emission_accuracy improves",
            "answer_prefix_accuracy improves",
            "answer_value_accuracy improves",
            "helper-only rollout accuracy improves",
            "canary/AST/determinism/controls/leakage gates pass",
        ],
        "no_positive_from": [
            "train loss alone",
            "teacher-forced loss alone",
            "ANSWER=T... eval leakage",
            "post-generation repair",
            "threshold weakening",
        ],
    }


def risk_register() -> dict[str, Any]:
    return {
        "schema_version": "phase_138h_risk_register_v1",
        "risks": [
            {
                "risk": "138I repeats 138R as loss-only training",
                "mitigation": "138I plan rejects train-loss-only and teacher-forcing-only success",
            },
            {
                "risk": "target checkpoint emits ANSWER=T... on eval rows",
                "mitigation": "namespace policy adds train_namespace_leak_rate and 138S failure route",
            },
            {
                "risk": "helper or scoring shortcuts re-enter",
                "mitigation": "shared helper only, canary, AST scan, generated-before-scoring proof",
            },
        ],
    }


def decide(summary: dict[str, Any], anti_shortcut: dict[str, Any]) -> dict[str, Any]:
    required_gates = [
        "shared_raw_generation_helper.py only",
        "generated_text before scoring",
        "expected-output canary",
        "AST shortcut scan",
        "deterministic replay",
        "controls fail",
        "leakage rejected",
        "source checkpoint unchanged",
        "target checkpoint under target/ only",
        "train namespace leak gate",
        "clean negative accepted",
    ]
    return {
        "schema_version": "phase_138h_decision_v1",
        "verdict": "ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN_COMPLETE",
        "decision": "rollout_aligned_objective_redesign_plan_complete",
        "next": "138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE",
        "primary_bottleneck": "train_namespace_rollout_alignment_failure",
        "evidence_summary": {
            "138ga_primary_label_counts": summary["near_match_disambiguation"]["value"],
            "138ga_meaningful_near_match_rate": summary["meaningful_near_match_rate"]["value"],
            "138r_rollout_accuracy": summary["rollout_accuracy"]["value"],
            "138r_expected_token_inclusion_rate": summary["expected_token_inclusion_rate"]["value"],
            "138g_teacher_forced_loss_fields": summary["teacher_forced_loss_fields"]["value"],
        },
        "rejected_alternatives": anti_shortcut["explicitly_reject"],
        "required_138i_gates": required_gates,
        "no_capability_restored": True,
        "artifact_only_planning": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "runtime_surface_mutated": False,
        "root_license_changed": False,
        **FALSE_FLAGS,
    }


def write_failure(args: argparse.Namespace, error: GateError) -> None:
    try:
        out = resolve_target_out(args.out)
    except GateError:
        out = (REPO_ROOT / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if error.verdict == "UPSTREAM_138GA_ARTIFACT_MISSING":
        decision_name = "upstream_138ga_artifact_missing"
        next_name = "138H_UPSTREAM_138GA_ARTIFACT_MISSING"
    elif error.verdict in {"UPSTREAM_138G_ARTIFACT_MISSING", "UPSTREAM_138R_ARTIFACT_MISSING"}:
        decision_name = "rollout_objective_redesign_blocked"
        next_name = "138HA_ROLLOUT_OBJECTIVE_EVIDENCE_RECHECK"
    else:
        decision_name = "raw_helper_integrity_failure"
        next_name = "135G_RAW_HELPER_ORACLE_SHORTCUT_REMOVAL"
    decision = {
        "schema_version": "phase_138h_failure_decision_v1",
        "verdict": error.verdict,
        "decision": decision_name,
        "next": next_name,
        "failure_message": error.message,
        "failure_details": error.details,
        "artifact_only_planning": True,
        "training_performed": False,
        "new_model_inference_run": False,
        "shared_helper_called": False,
        "torch_forward_pass_run": False,
        "checkpoint_mutated": False,
        "no_capability_restored": True,
        **FALSE_FLAGS,
    }
    write_json(out / "decision.json", decision)
    append_progress(out, "failure", status="failed", verdict=error.verdict, message=error.message)
    write_summary(out, "failure", [error.verdict], decision, error.message)
    write_report(out, [error.verdict], decision)


def run(args: argparse.Namespace) -> None:
    out = resolve_target_out(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "progress.jsonl", "")
    root_138ga = resolve_path(args.upstream_138ga_root)
    root_138g = resolve_path(args.upstream_138g_root)
    root_138r = resolve_path(args.upstream_138r_root)
    write_json(
        out / "queue.json",
        {
            "schema_version": "phase_138h_queue_v1",
            "milestone": MILESTONE,
            "status": "started",
            "started_at": utc_now(),
            "heartbeat_sec": args.heartbeat_sec,
        },
    )
    append_progress(out, "startup", heartbeat_sec=args.heartbeat_sec)
    refresh_status(out, "running", ["138H_RUNNING"], {"decision": "pending", "next": "pending"})

    ga = verify_138ga(out, root_138ga)
    g = verify_138g(out, root_138g)
    r = verify_138r(out, root_138r)
    append_progress(out, "upstream verification", upstream_138ga_verified=True, upstream_138g_verified=True, upstream_138r_verified=True)
    refresh_status(out, "running", ["UPSTREAMS_VERIFIED"], {"decision": "pending", "next": "pending"})

    append_progress(out, "artifact loading", artifact_only=True)
    summary = build_rollout_alignment_failure_summary(ga, g, r)
    write_json(out / "rollout_alignment_failure_summary.json", summary)
    append_progress(out, "diagnosis", primary_bottleneck=summary["primary_bottleneck"]["value"])
    refresh_status(out, "running", ["ROLLOUT_ALIGNMENT_FAILURE_SUMMARIZED"], {"decision": "pending", "next": "pending"})

    requirements = objective_redesign_requirements()
    namespace = namespace_policy()
    write_json(out / "objective_redesign_requirements.json", requirements)
    write_json(out / "train_eval_namespace_policy.json", namespace)
    append_progress(out, "namespace policy drafting", metrics=namespace["required_metrics_for_138i"])

    design = rollout_aligned_training_design()
    gates = final_eval_gate_design()
    anti_shortcut = anti_shortcut_requirements()
    plan_138i = next_138i_plan(namespace, design, gates, anti_shortcut)
    write_json(out / "rollout_aligned_training_design.json", design)
    write_json(out / "final_eval_gate_design.json", gates)
    write_json(out / "anti_shortcut_requirements.json", anti_shortcut)
    write_json(out / "next_138i_milestone_plan.json", plan_138i)
    write_json(out / "risk_register.json", risk_register())
    append_progress(out, "138I plan drafting", next=plan_138i["milestone"])

    decision = decide(summary, anti_shortcut)
    write_json(out / "decision.json", decision)
    append_progress(out, "decision writing", decision=decision["decision"], next=decision["next"])
    verdicts = [
        "ROLLOUT_ALIGNED_OBJECTIVE_REDESIGN_PLAN_COMPLETE",
        "ARTIFACT_ONLY_PLANNING",
        "NO_TRAINING",
        "NO_INFERENCE",
        "CAPABILITY_FLAGS_FALSE",
    ]
    refresh_status(out, "complete", verdicts, decision)
    append_progress(out, "final verdict", verdicts=verdicts)
    write_json(
        out / "queue.json",
        {
            "schema_version": "phase_138h_queue_v1",
            "milestone": MILESTONE,
            "status": "completed",
            "completed_at": utc_now(),
            "heartbeat_sec": args.heartbeat_sec,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-138ga-root", default=str(DEFAULT_UPSTREAM_138GA_ROOT))
    parser.add_argument("--upstream-138g-root", default=str(DEFAULT_UPSTREAM_138G_ROOT))
    parser.add_argument("--upstream-138r-root", default=str(DEFAULT_UPSTREAM_138R_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
        return 0
    except GateError as exc:
        write_failure(args, exc)
        print(f"138H failed closed: {exc.verdict}: {exc.message}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
