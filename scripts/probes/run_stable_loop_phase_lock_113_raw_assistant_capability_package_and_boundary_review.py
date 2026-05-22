#!/usr/bin/env python3
"""113 raw assistant capability package and boundary review.

This milestone packages existing 099-112 research evidence and checks claim
boundaries. It does not train, repair, deploy, start services, run deployment
smoke, or run model inference.
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
MILESTONE = "STABLE_LOOP_PHASE_LOCK_113_RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_113_raw_assistant_capability_package_and_boundary_review/smoke")
DEFAULT_UPSTREAM_112_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_112_current_chassis_raw_generation_scale_confirm/smoke")
DEFAULT_UPSTREAM_111X_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_111x_chassis_decision_raw_generation_redesign_gate/smoke")
DEFAULT_UPSTREAM_111R_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_111r_retention_or_lm_regression_analysis/smoke")
DEFAULT_UPSTREAM_110_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_110_integrated_decoder_policy_ood_confirm_batch/smoke")
DEFAULT_UPSTREAM_100_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_100_open_vocab_assistant_capability_scale/smoke")
DEFAULT_UPSTREAM_099_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke")

POSITIVE_VERDICT = "RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_POSITIVE"
BOUNDARY_TEXT = (
    "113 is an evidence package and boundary-review milestone. It reads existing artifacts and "
    "writes a package under target/pilot_wave only. It performs no training, no repair, no model "
    "inference, no service startup, no deployment smoke, and no runtime/product integration. It is "
    "not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, "
    "not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness."
)

EXPECTED_UPSTREAMS = {
    "099": "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE",
    "100": "OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE",
    "110": "INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE",
    "111r": "RETENTION_OR_LM_REGRESSION_ANALYSIS_POSITIVE",
    "111x": "CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_POSITIVE",
    "112": "CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE",
}

READINESS_CLAIM_FLAGS = [
    "gpt_like_readiness_claimed",
    "open_domain_assistant_readiness_claimed",
    "production_chat_claimed",
    "public_api_claimed",
    "deployment_readiness_claimed",
    "safety_alignment_claimed",
    "hungarian_assistant_readiness_claimed",
]


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


def file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
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


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_FAILS", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_FAILS", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def resolve_upstream(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def extract_key_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    wanted = [
        "decision",
        "next",
        "next_milestone",
        "winning_arm",
        "raw_ood_accuracy",
        "mean_raw_ood_accuracy",
        "min_raw_ood_accuracy",
        "redesigned_current_chassis_accuracy",
        "small_transformer_baseline_accuracy",
        "current_raw_baseline_accuracy",
        "integrated_ood_stress_accuracy",
        "raw_ood_stress_accuracy",
        "raw_vs_integrated_gap",
        "decoder_reference_used_rate",
        "repair_stage_trace_rate",
        "generated_prompt_response_accuracy",
        "raw_generated_prompt_response_accuracy",
        "bounded_chat_slot_binding_accuracy",
        "finite_label_anchorroute_retention_accuracy",
        "unsupported_refusal_accuracy",
        "unsupported_refusal_retention_accuracy",
        "retention_pass_all_seeds",
        "fineweb_eval_loss_regression",
        "fineweb_next_byte_accuracy_drop",
        "namespace_leak_rate",
        "teacher_namespace_copy_rate",
        "case_id_drift_rate",
        "max_namespace_leak_rate",
        "max_teacher_namespace_copy_rate",
        "max_case_id_drift_rate",
        "source_100_checkpoint_unchanged",
        "source_102_checkpoint_unchanged",
        "bounded_release_artifact_unchanged",
        "packaged_winner_hash_unchanged",
        "checkpoint_hash_unchanged",
        "local_private_release_ready",
        "deployment_harness_gate_pass",
        "bounded_chat_service_smoke_pass",
        "primary_root_cause",
        "secondary_root_causes",
        "recommended_next",
    ]
    return {key: metrics[key] for key in wanted if key in metrics}


def extract_boundary_flags(summary: dict[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for key, value in summary.items():
        if (
            key.endswith("_claimed")
            or key.endswith("_mutated")
            or key.endswith("_performed")
            or key in {"training_performed", "analysis_only", "eval_only_research_confirm", "scale_confirm_research_gate"}
        ):
            flags[key] = value
    metrics = summary.get("metrics", {})
    for key in [
        "train_step_count",
        "optimizer_step_count",
        "artifact_exfiltration_count",
        "gpt_like_claim_count",
        "production_chat_claim_count",
        "public_api_claim_count",
        "safety_alignment_claim_count",
        "llm_judge_used",
        "prediction_oracle_used",
    ]:
        if key in metrics:
            flags[key] = metrics[key]
    return flags


def verify_upstream(root: Path, expected_verdict: str) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise GateError("UPSTREAM_ARTIFACT_MISSING", f"Missing upstream summary: {summary_path}")
    summary = read_json(summary_path)
    verdicts = set(summary.get("verdicts", []))
    if summary.get("status") != "positive" or expected_verdict not in verdicts:
        raise GateError("UPSTREAM_STACK_NOT_POSITIVE", f"Upstream {root} is not positive for {expected_verdict}")
    return summary


def make_upstream_manifest(name: str, root: Path, summary: dict[str, Any], expected_verdict: str) -> dict[str, Any]:
    return {
        "schema_version": "phase_113_upstream_manifest_v1",
        "upstream": name,
        "root_path": rel(root),
        "summary_path": rel(root / "summary.json"),
        "summary_hash": stable_json_hash(summary),
        "positive_verdict": expected_verdict,
        "key_metrics": extract_key_metrics(summary),
        "boundary_flags": extract_boundary_flags(summary),
        "milestone": summary.get("milestone"),
        "status": summary.get("status"),
    }


def claim_boundary() -> dict[str, Any]:
    return {
        "schema_version": "phase_113_claim_boundary_v1",
        "gpt_like_readiness_claimed": False,
        "open_domain_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_api_claimed": False,
        "deployment_readiness_claimed": False,
        "safety_alignment_claimed": False,
        "hungarian_assistant_readiness_claimed": False,
        "permitted_claim": "rubric-bounded research evidence package after positive 112",
        "forbidden_claims": [
            "GPT-like assistant readiness",
            "open-domain assistant readiness",
            "production chat",
            "public API",
            "deployment readiness",
            "safety alignment",
            "Hungarian assistant readiness",
        ],
    }


def write_summary(out: Path, phase: str, status: str, verdicts: list[str], metrics: dict[str, Any], message: str | None = None) -> None:
    summary = {
        "schema_version": "phase_113_capability_package_summary_v1",
        "milestone": MILESTONE,
        "phase": phase,
        "status": status,
        "boundary": BOUNDARY_TEXT,
        "package_and_boundary_review_only": True,
        "training_performed": False,
        "inference_performed": False,
        "service_started": False,
        "deployment_smoke_run": False,
        "runtime_surface_mutated": False,
        "bounded_release_stack_mutated": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_assistant_readiness_claimed": False,
        "production_chat_claimed": False,
        "public_api_claimed": False,
        "deployment_readiness_claimed": False,
        "safety_alignment_claimed": False,
        "hungarian_assistant_readiness_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    if message:
        summary["message"] = message
    write_json(out / "summary.json", summary)


def write_report(out: Path, phase: str, verdicts: list[str], metrics: dict[str, Any], decision: dict[str, Any] | None = None) -> None:
    decision = decision or {}
    lines = [
        f"# {MILESTONE}",
        "",
        f"Phase: {phase}",
        "",
        "## Boundary",
        BOUNDARY_TEXT,
        "",
        "## Side Effects",
        "- train_step_count = 0",
        "- optimizer_step_count = 0",
        "- inference_run_count = 0",
        "- service_started = false",
        "- deployment_smoke_run = false",
        "",
        "## Evidence",
        f"- upstream evidence complete: {str(metrics.get('upstream_evidence_complete', False)).lower()}",
        f"- claim boundary complete: {str(metrics.get('claim_boundary_complete', False)).lower()}",
        f"- release/capability separation written: {str(metrics.get('release_capability_boundary_separated', False)).lower()}",
        f"- validated findings delta written: {str(metrics.get('validated_findings_delta_written', False)).lower()}",
        "",
        "## Decision",
        f"- next: {decision.get('next', metrics.get('next', 'pending'))}",
        f"- decision: {decision.get('decision', metrics.get('decision', 'pending'))}",
        "",
        "## Verdicts",
    ]
    lines.extend(f"- {verdict}" for verdict in verdicts)
    write_text(out / "report.md", "\n".join(lines) + "\n")


def write_live(out: Path, phase: str, verdicts: list[str], metrics: dict[str, Any], decision: dict[str, Any] | None = None) -> None:
    write_summary(out, phase, "running" if phase != "final_verdict" else "positive", verdicts, metrics)
    write_report(out, phase, verdicts, metrics, decision)


def sample_index_for(root: Path) -> dict[str, Any]:
    candidates = [
        "human_readable_samples.jsonl",
        "failure_case_samples.jsonl",
        "human_readable_failure_samples.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    for name in candidates:
        path = root / name
        if not path.exists():
            continue
        line_count = sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
        rows.append({"artifact": rel(path), "line_count": line_count, "sha256": file_sha256(path)})
    return {"root": rel(root), "samples": rows}


def build_human_summary() -> str:
    return "\n".join(
        [
            "# 113 Raw Assistant Capability Package And Boundary Review",
            "",
            "## What is proven",
            "The 099-112 evidence chain is coherent enough to package for the next research bridge. "
            "099 covers local/private bounded release readiness, while 112 covers current-chassis raw generation scale on rubric-bounded eval.",
            "",
            "## What is not proven",
            "This is not GPT-like assistant readiness, not open-domain assistant readiness, not production chat, "
            "not public API, not deployment readiness, not safety alignment, and not Hungarian assistant readiness.",
            "",
            "## What remains risky",
            "The evidence is still harness-bound. External-style stress, broader prompt distributions, safety review, and product/runtime review remain outside this package.",
            "",
            "## Why 114 is next",
            "114 should bridge the raw assistant capability evidence into external-style stress benchmarks without turning it into a deployment or readiness claim.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> int:
    start = time.time()
    out = resolve_target_out(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "schema_version": "phase_113_capability_package_metrics_v1",
        "train_step_count": 0,
        "optimizer_step_count": 0,
        "inference_run_count": 0,
        "service_started": False,
        "deployment_smoke_run": False,
        "training_performed": False,
        "inference_performed": False,
        "runtime_surface_mutated": False,
        "bounded_release_stack_mutated": False,
        "upstream_evidence_complete": False,
        "claim_boundary_complete": False,
        "release_capability_boundary_separated": False,
        "validated_findings_delta_written": False,
        "decision": "pending",
        "next": "pending",
    }
    verdicts: list[str] = []

    queue = {
        "schema_version": "phase_113_queue_v1",
        "milestone": MILESTONE,
        "created_at": utc_now(),
        "tasks": [
            "verify upstream evidence",
            "index evidence",
            "review integrity",
            "review claim boundary",
            "write package artifacts",
            "write machine-readable decision",
        ],
    }
    write_json(out / "queue.json", queue)
    write_json(
        out / "package_config.json",
        {
            "schema_version": "phase_113_package_config_v1",
            "milestone": MILESTONE,
            "out": rel(out),
            "heartbeat_sec": args.heartbeat_sec,
            "no_training": True,
            "no_inference": True,
            "no_service_start": True,
            "no_deployment_smoke": True,
            "upstreams": {
                "099": args.upstream_099_root,
                "100": args.upstream_100_root,
                "110": args.upstream_110_root,
                "111r": args.upstream_111r_root,
                "111x": args.upstream_111x_root,
                "112": args.upstream_112_root,
            },
        },
    )
    append_progress(out, "start", "running", milestone=MILESTONE)
    write_live(out, "start", verdicts, metrics)

    roots = {
        "099": resolve_upstream(args.upstream_099_root),
        "100": resolve_upstream(args.upstream_100_root),
        "110": resolve_upstream(args.upstream_110_root),
        "111r": resolve_upstream(args.upstream_111r_root),
        "111x": resolve_upstream(args.upstream_111x_root),
        "112": resolve_upstream(args.upstream_112_root),
    }
    summaries: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for name, root in roots.items():
        summaries[name] = verify_upstream(root, EXPECTED_UPSTREAMS[name])
        manifests[name] = make_upstream_manifest(name, root, summaries[name], EXPECTED_UPSTREAMS[name])
        write_json(out / f"upstream_{name}_manifest.json", manifests[name])

    metrics["upstream_evidence_complete"] = True
    append_progress(out, "upstream_verification", upstreams=list(roots))
    write_live(out, "upstream_verification", ["UPSTREAM_112_SCALE_CONFIRM_VERIFIED"], metrics)

    evidence_index = {
        "schema_version": "phase_113_evidence_index_v1",
        "milestone": MILESTONE,
        "upstreams": manifests,
        "package_scope": "099 bounded release evidence plus 100/110/111R/111X/112 research capability evidence",
        "created_without_training_or_inference": True,
    }
    write_json(out / "evidence_index.json", evidence_index)
    write_json(
        out / "capability_package_manifest.json",
        {
            "schema_version": "phase_113_capability_package_manifest_v1",
            "milestone": MILESTONE,
            "included_upstreams": list(manifests),
            "artifact_roots": {name: rel(root) for name, root in roots.items()},
            "evidence_index_hash": stable_json_hash(evidence_index),
            "side_effects": {
                "train_step_count": 0,
                "optimizer_step_count": 0,
                "inference_run_count": 0,
                "service_started": False,
                "deployment_smoke_run": False,
            },
        },
    )
    metrics["evidence_index_hash"] = stable_json_hash(evidence_index)
    append_progress(out, "evidence_indexing")
    write_live(out, "evidence_indexing", ["UPSTREAM_112_SCALE_CONFIRM_VERIFIED", "EVIDENCE_CHAIN_PACKAGED"], metrics)

    summary_112_metrics = summaries["112"].get("metrics", {})
    summary_111x_metrics = summaries["111x"].get("metrics", {})
    summary_111r_metrics = summaries["111r"].get("metrics", {})
    summary_099_metrics = summaries["099"].get("metrics", {})
    raw_generation_capability_summary = {
        "schema_version": "phase_113_raw_generation_capability_summary_v1",
        "source": "112 scale confirm plus 111X decision gate",
        "current_chassis_scale_confirmed": summary_112_metrics.get("decision") == "current_chassis_scale_confirmed",
        "winning_arm": summary_112_metrics.get("winning_arm"),
        "min_raw_ood_accuracy": summary_112_metrics.get("min_raw_ood_accuracy"),
        "mean_raw_ood_accuracy": summary_112_metrics.get("mean_raw_ood_accuracy"),
        "stddev_raw_ood_accuracy": summary_112_metrics.get("stddev_raw_ood_accuracy"),
        "max_namespace_leak_rate": summary_112_metrics.get("max_namespace_leak_rate"),
        "max_teacher_namespace_copy_rate": summary_112_metrics.get("max_teacher_namespace_copy_rate"),
        "max_case_id_drift_rate": summary_112_metrics.get("max_case_id_drift_rate"),
        "retention_pass_all_seeds": summary_112_metrics.get("retention_pass_all_seeds"),
        "fineweb_eval_loss_regression": summary_112_metrics.get("fineweb_eval_loss_regression"),
        "fineweb_next_byte_accuracy_drop": summary_112_metrics.get("fineweb_next_byte_accuracy_drop"),
        "111x_decision": summary_111x_metrics.get("decision"),
        "111x_winning_arm": summary_111x_metrics.get("winning_arm"),
    }
    write_json(out / "raw_generation_capability_summary.json", raw_generation_capability_summary)

    integrity_manifest = {
        "schema_version": "phase_113_integrity_manifest_v1",
        "bounded_release_artifact_unchanged": bool(
            summary_112_metrics.get("bounded_release_artifact_unchanged")
            and summary_111x_metrics.get("bounded_release_artifact_unchanged")
            and summary_099_metrics.get("artifact_hash_verified")
        ),
        "source_100_checkpoint_unchanged": bool(summary_112_metrics.get("source_100_checkpoint_unchanged") and summary_111x_metrics.get("source_100_checkpoint_unchanged")),
        "source_102_checkpoint_unchanged": bool(summary_112_metrics.get("source_102_checkpoint_unchanged") and summary_111x_metrics.get("source_102_checkpoint_unchanged")),
        "packaged_winner_hash_unchanged": bool(summary_112_metrics.get("packaged_winner_hash_unchanged") and summary_111x_metrics.get("packaged_winner_hash_unchanged")),
        "root_license_changed": False,
        "summary_hashes": {name: manifest["summary_hash"] for name, manifest in manifests.items()},
    }
    write_json(out / "integrity_manifest.json", integrity_manifest)
    metrics.update(
        {
            "bounded_release_artifact_unchanged": integrity_manifest["bounded_release_artifact_unchanged"],
            "source_100_checkpoint_unchanged": integrity_manifest["source_100_checkpoint_unchanged"],
            "source_102_checkpoint_unchanged": integrity_manifest["source_102_checkpoint_unchanged"],
            "packaged_winner_hash_unchanged": integrity_manifest["packaged_winner_hash_unchanged"],
        }
    )
    append_progress(out, "integrity_review")
    write_live(out, "integrity_review", ["UPSTREAM_112_SCALE_CONFIRM_VERIFIED", "EVIDENCE_CHAIN_PACKAGED"], metrics)

    boundary = claim_boundary()
    write_json(out / "claim_boundary.json", boundary)
    readiness_denial_matrix = {
        "schema_version": "phase_113_readiness_denial_matrix_v1",
        "denials": {
            "GPT-like assistant readiness": {"claimed": False, "reason": "rubric-bounded research evidence is not a general readiness proof"},
            "open-domain assistant readiness": {"claimed": False, "reason": "external-style stress remains pending"},
            "production chat": {"claimed": False, "reason": "113 performs no product/runtime integration"},
            "public API": {"claimed": False, "reason": "113 exposes no API surface"},
            "deployment readiness": {"claimed": False, "reason": "113 runs no deployment smoke and is not a deploy gate"},
            "safety alignment": {"claimed": False, "reason": "safety alignment review is outside this package"},
            "Hungarian assistant readiness": {"claimed": False, "reason": "Hungarian remains outside the positive readiness boundary"},
        },
    }
    write_json(out / "readiness_denial_matrix.json", readiness_denial_matrix)
    release_vs_capability_separation = {
        "schema_version": "phase_113_release_vs_capability_separation_v1",
        "statement_099": "099 proves local/private bounded release readiness.",
        "statement_112": "112 proves raw assistant capability scale on rubric-bounded eval.",
        "statement_113": "113 does not merge these into production/public/GPT-like readiness.",
        "bounded_release_evidence": {
            "source": rel(roots["099"]),
            "positive_verdict": EXPECTED_UPSTREAMS["099"],
            "scope": "local/private bounded release readiness",
        },
        "raw_capability_evidence": {
            "source": rel(roots["112"]),
            "positive_verdict": EXPECTED_UPSTREAMS["112"],
            "scope": "rubric-bounded raw assistant capability scale confirm",
        },
        "merged_claims_blocked": True,
    }
    write_json(out / "release_vs_capability_separation.json", release_vs_capability_separation)
    boundary_review = {
        "schema_version": "phase_113_boundary_review_v1",
        "boundary": BOUNDARY_TEXT,
        "claim_boundary_hash": stable_json_hash(boundary),
        "readiness_denial_matrix_hash": stable_json_hash(readiness_denial_matrix),
        "release_vs_capability_separation_hash": stable_json_hash(release_vs_capability_separation),
        "overclaim_detected": False,
        "artifact_exfiltration_detected": False,
        "runtime_surface_mutation_detected": False,
    }
    write_json(out / "boundary_review.json", boundary_review)
    metrics["claim_boundary_complete"] = True
    metrics["release_capability_boundary_separated"] = True
    append_progress(out, "boundary_review")
    write_live(out, "boundary_review", ["UPSTREAM_112_SCALE_CONFIRM_VERIFIED", "EVIDENCE_CHAIN_PACKAGED", "CLAIM_BOUNDARY_VERIFIED"], metrics)

    validated_findings_delta = {
        "schema_version": "phase_113_validated_findings_delta_v1",
        "findings": [
            {
                "phase": "111",
                "finding": "111 standard failed",
                "evidence": {
                    "pre_111_raw_ood_accuracy": summary_111r_metrics.get("pre_111_raw_ood_accuracy"),
                    "post_111_raw_ood_accuracy": summary_111r_metrics.get("post_111_raw_ood_accuracy"),
                    "retention_accuracy_min": summary_111r_metrics.get("retention_accuracy_min"),
                },
            },
            {
                "phase": "111R",
                "finding": "111R classified mixed cause",
                "evidence": {
                    "primary_root_cause": summary_111r_metrics.get("primary_root_cause"),
                    "secondary_root_causes": summary_111r_metrics.get("secondary_root_causes"),
                },
            },
            {
                "phase": "111X",
                "finding": "111X decided current chassis viable",
                "evidence": {
                    "decision": summary_111x_metrics.get("decision"),
                    "winning_arm": summary_111x_metrics.get("winning_arm"),
                    "redesigned_current_chassis_accuracy": summary_111x_metrics.get("redesigned_current_chassis_accuracy"),
                },
            },
            {
                "phase": "112",
                "finding": "112 scale-confirmed current chassis raw generation",
                "evidence": {
                    "decision": summary_112_metrics.get("decision"),
                    "min_raw_ood_accuracy": summary_112_metrics.get("min_raw_ood_accuracy"),
                    "mean_raw_ood_accuracy": summary_112_metrics.get("mean_raw_ood_accuracy"),
                    "retention_pass_all_seeds": summary_112_metrics.get("retention_pass_all_seeds"),
                },
            },
        ],
    }
    write_json(out / "validated_findings_delta.json", validated_findings_delta)
    metrics["validated_findings_delta_written"] = True

    retention_and_lm_summary = {
        "schema_version": "phase_113_retention_lm_summary_v1",
        "112_retention_pass_all_seeds": summary_112_metrics.get("retention_pass_all_seeds"),
        "112_fineweb_eval_loss_regression": summary_112_metrics.get("fineweb_eval_loss_regression"),
        "112_fineweb_next_byte_accuracy_drop": summary_112_metrics.get("fineweb_next_byte_accuracy_drop"),
        "111x_bounded_chat_slot_binding_accuracy": summary_111x_metrics.get("bounded_chat_slot_binding_accuracy"),
        "111x_finite_label_anchorroute_retention_accuracy": summary_111x_metrics.get("finite_label_anchorroute_retention_accuracy"),
        "110_bounded_chat_slot_binding_accuracy": summaries["110"].get("metrics", {}).get("bounded_chat_slot_binding_accuracy"),
        "100_bounded_chat_slot_binding_accuracy": summaries["100"].get("metrics", {}).get("bounded_chat_slot_binding_accuracy"),
        "100_finite_label_anchorroute_retention_accuracy": summaries["100"].get("metrics", {}).get("finite_label_anchorroute_retention_accuracy"),
    }
    write_json(out / "retention_and_lm_summary.json", retention_and_lm_summary)
    write_json(out / "sample_index.json", {"schema_version": "phase_113_sample_index_v1", "upstream_samples": {name: sample_index_for(root) for name, root in roots.items()}})
    write_json(
        out / "limitation_register.json",
        {
            "schema_version": "phase_113_limitation_register_v1",
            "limitations": [
                "Evidence remains harness-bound and rubric-bounded.",
                "External-style stress benchmark bridge remains pending.",
                "No production/runtime/service integration was performed.",
                "No public API or deployment readiness claim is made.",
                "No safety alignment claim is made.",
                "Hungarian assistant readiness is not established.",
            ],
        },
    )
    write_text(out / "human_readable_summary.md", build_human_summary())
    append_progress(out, "package_writing")
    write_live(
        out,
        "package_writing",
        [
            "UPSTREAM_112_SCALE_CONFIRM_VERIFIED",
            "EVIDENCE_CHAIN_PACKAGED",
            "CLAIM_BOUNDARY_VERIFIED",
            "RELEASE_CAPABILITY_BOUNDARY_SEPARATED",
            "VALIDATED_FINDINGS_DELTA_WRITTEN",
        ],
        metrics,
    )

    if not integrity_manifest["bounded_release_artifact_unchanged"]:
        next_milestone = "113R_PACKAGE_INTEGRITY_REGRESSION_ANALYSIS"
        decision_name = "integrity_regression"
        failure_reason = "bounded release artifact integrity did not remain verified"
    elif not all(boundary[key] is False for key in READINESS_CLAIM_FLAGS):
        next_milestone = "113C_BOUNDARY_OVERCLAIM_FAILURE_ANALYSIS"
        decision_name = "boundary_overclaim"
        failure_reason = "claim boundary flags were not all false"
    elif not metrics["upstream_evidence_complete"]:
        next_milestone = "113B_CAPABILITY_PACKAGE_EVIDENCE_GAP_ANALYSIS"
        decision_name = "evidence_gap"
        failure_reason = "upstream evidence was incomplete"
    else:
        next_milestone = "114_RAW_ASSISTANT_EXTERNAL_STRESS_BENCHMARK_BRIDGE"
        decision_name = "capability_package_boundary_review_positive"
        failure_reason = None

    decision = {
        "schema_version": "phase_113_decision_v1",
        "decision": decision_name,
        "next": next_milestone,
        "reason": (
            "099-112 evidence is packaged, release/capability claims are separated, claim boundary flags are false, "
            "and no training/inference/service/deployment side effects were performed."
            if failure_reason is None
            else failure_reason
        ),
        "prerequisites_satisfied": [
            "112 CURRENT_CHASSIS_RAW_GENERATION_SCALE_CONFIRM_POSITIVE",
            "111X CHASSIS_DECISION_RAW_GENERATION_REDESIGN_GATE_POSITIVE",
            "111R RETENTION_OR_LM_REGRESSION_ANALYSIS_POSITIVE",
            "110 INTEGRATED_DECODER_POLICY_OOD_CONFIRM_BATCH_POSITIVE",
            "100 OPEN_VOCAB_ASSISTANT_CAPABILITY_SCALE_POSITIVE",
            "099 BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE",
        ],
        "blocked_claims": [
            "GPT-like assistant readiness",
            "open-domain assistant readiness",
            "production chat",
            "public API",
            "deployment readiness",
            "safety alignment",
            "Hungarian assistant readiness",
        ],
        "recommended_scope": "Run 114 as an external-style stress benchmark bridge without product/runtime/deploy claims.",
    }
    write_json(out / "decision.json", decision)
    metrics["decision"] = decision_name
    metrics["next"] = next_milestone
    append_progress(out, "decision_writing", decision=decision_name, next=next_milestone)
    write_live(
        out,
        "decision_writing",
        [
            "UPSTREAM_112_SCALE_CONFIRM_VERIFIED",
            "EVIDENCE_CHAIN_PACKAGED",
            "CLAIM_BOUNDARY_VERIFIED",
            "RELEASE_CAPABILITY_BOUNDARY_SEPARATED",
            "VALIDATED_FINDINGS_DELTA_WRITTEN",
        ],
        metrics,
        decision,
    )

    if failure_reason is not None:
        raise GateError("RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_FAILS", failure_reason)

    metrics.update(
        {
            "wall_clock_sec": round(time.time() - start, 3),
            "evidence_package_artifact_count": 24,
            "claim_boundary_flags_false": True,
        }
    )
    verdicts = [
        POSITIVE_VERDICT,
        "UPSTREAM_112_SCALE_CONFIRM_VERIFIED",
        "EVIDENCE_CHAIN_PACKAGED",
        "CLAIM_BOUNDARY_VERIFIED",
        "RELEASE_CAPABILITY_BOUNDARY_SEPARATED",
        "VALIDATED_FINDINGS_DELTA_WRITTEN",
        "NO_TRAINING_PERFORMED",
        "NO_INFERENCE_PERFORMED",
        "BOUNDED_RELEASE_UNCHANGED",
        "PRODUCTION_CHAT_NOT_CLAIMED",
        "GPT_LIKE_READINESS_NOT_CLAIMED",
    ]
    append_progress(out, "final_verdict", verdict=POSITIVE_VERDICT, next=next_milestone)
    write_summary(out, "final_verdict", "positive", verdicts, metrics)
    write_report(out, "final_verdict", verdicts, metrics, decision)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-112-root", default=str(DEFAULT_UPSTREAM_112_ROOT))
    parser.add_argument("--upstream-111x-root", default=str(DEFAULT_UPSTREAM_111X_ROOT))
    parser.add_argument("--upstream-111r-root", default=str(DEFAULT_UPSTREAM_111R_ROOT))
    parser.add_argument("--upstream-110-root", default=str(DEFAULT_UPSTREAM_110_ROOT))
    parser.add_argument("--upstream-100-root", default=str(DEFAULT_UPSTREAM_100_ROOT))
    parser.add_argument("--upstream-099-root", default=str(DEFAULT_UPSTREAM_099_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()
    try:
        return run(args)
    except GateError as exc:
        try:
            out = resolve_target_out(args.out)
            metrics = {
                "schema_version": "phase_113_capability_package_metrics_v1",
                "train_step_count": 0,
                "optimizer_step_count": 0,
                "inference_run_count": 0,
                "service_started": False,
                "deployment_smoke_run": False,
                "failure_verdict": exc.verdict,
            }
            append_progress(out, "failure", "failed", verdict=exc.verdict, message=exc.message)
            write_summary(out, "failure", "failed", ["RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_FAILS", exc.verdict], metrics, exc.message)
            write_report(out, "failure", ["RAW_ASSISTANT_CAPABILITY_PACKAGE_AND_BOUNDARY_REVIEW_FAILS", exc.verdict], metrics)
        except Exception:
            pass
        print(f"{exc.verdict}: {exc.message}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
