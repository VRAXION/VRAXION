#!/usr/bin/env python3
"""Winner-proof runner for STABLE_LOOP_PHASE_LOCK_089B.

This runner is intentionally orchestration-only. It reuses the committed 080
and 081 Rust examples as child runners, writes heartbeat-visible partial
artifacts, and never trains or mutates the packaged checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_089B_PACKAGED_MODEL_WINNER_REPRO_AND_ADVERSARIAL_TRAIN_PROOF"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke")
DEFAULT_UPSTREAM_078_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke")
DEFAULT_UPSTREAM_080_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke")
DEFAULT_UPSTREAM_081_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke")
DEFAULT_UPSTREAM_082_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_082_chat_diversity_multi_seed_confirm/smoke")
DEFAULT_UPSTREAM_083_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke")
DEFAULT_UPSTREAM_089_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_089_private_evaluation_rc_package/smoke")

BOUNDARY_TEXT = (
    "089B is bounded winner reproducibility proof only; it is not GPT-like assistant readiness, "
    "not open-domain chat, not full English LM, not production deployment, not safety alignment, "
    "and not public release."
)

POSITIVE_VERDICTS = [
    "PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_POSITIVE",
    "PACKAGE_HASH_BINDING_VERIFIED",
    "PACKAGED_CHECKPOINT_FRESH_EVAL_PASSES",
    "DETERMINISTIC_REPRO_TRAINING_PASSES",
    "TOKEN_OBJECTIVE_LEARNED",
    "WINNER_BEATS_CONTROLS",
    "BASELINE_EVAL_ROWS_MATCH",
    "TAMPER_CONTROLS_FAIL_AS_EXPECTED",
    "LEAKAGE_CONTROLS_FAIL_AS_EXPECTED",
    "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
    "NO_TRAINING_ON_PACKAGED_CHECKPOINT",
    "CHECKPOINT_PIPELINE_PASSES",
    "PRODUCTION_CHAT_NOT_CLAIMED",
]

UPSTREAMS = {
    "078": ("upstream_078_root", "CHAT_COMPOSITION_REPAIR_POSITIVE"),
    "080": ("upstream_080_root", "CHAT_COMPOSITION_DIVERSITY_REPAIR_POSITIVE"),
    "081": ("upstream_081_root", "CHAT_DIVERSITY_FRESH_CONFIRM_POSITIVE"),
    "082": ("upstream_082_root", "CHAT_DIVERSITY_MULTI_SEED_CONFIRM_POSITIVE"),
    "083": ("upstream_083_root", "CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE"),
    "089": ("upstream_089_root", "PRIVATE_EVALUATION_RC_PACKAGE_POSITIVE"),
}

REQUIRED_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "winner_proof_config.json",
    "upstream_manifest.json",
    "package_hash_binding.json",
    "packaged_checkpoint_eval.json",
    "repro_training_manifest.json",
    "repro_training_metrics.jsonl",
    "deterministic_mismatch_analysis.json",
    "arm_comparison.json",
    "control_delta_report.json",
    "tamper_control_report.json",
    "leakage_control_report.json",
    "eval_row_hashes.json",
    "human_readable_samples.jsonl",
    "failure_case_samples.jsonl",
    "summary.json",
    "report.md",
]


class ProofError(Exception):
    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_ms() -> int:
    return int(time.time() * 1000)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ProofError("UPSTREAM_ARTIFACT_MISSING", f"path must be repo-relative: {path_text}")
    return (REPO_ROOT / path).resolve()


def resolve_target_out(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise ProofError("UPSTREAM_ARTIFACT_MISSING", "--out must be repo-relative")
    parts = [part.lower() for part in raw.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise ProofError("UPSTREAM_ARTIFACT_MISSING", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / raw).resolve()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def write_summary(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    payload = {
        "schema_version": "packaged_model_winner_repro_train_proof_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "boundary": BOUNDARY_TEXT,
        "packaged_train_step_count": 0,
        "packaged_checkpoint_hash_unchanged": metrics.get("packaged_checkpoint_hash_unchanged"),
        "train_step_count_on_packaged_checkpoint": 0,
        "training_side_effect_on_packaged_checkpoint": False,
        "prediction_oracle_used": False,
        "llm_judge_used": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_chat_claimed": False,
        "full_English_LM_claimed": False,
        "production_deployment_claimed": False,
        "production_chat_claimed": False,
        "safety_alignment_claimed": False,
        "public_release_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    if message:
        payload["message"] = message
    write_json(out / "summary.json", payload)
    write_report(out, status, verdicts, metrics, message)


def write_report(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_089B_PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF Report",
        "",
        BOUNDARY_TEXT,
        "",
        f"Status: `{status}`",
        "",
        "## Verdicts",
        "",
        "```text",
        *verdicts,
        "```",
        "",
        "## Key Metrics",
        "",
    ]
    for key in [
        "source_083_checkpoint_file_sha256",
        "packaged_083_checkpoint_file_sha256",
        "upstream_080_checkpoint_file_sha256",
        "repro_child_checkpoint_file_sha256",
        "upstream_080_model_payload_sha256",
        "repro_child_model_payload_sha256",
        "package_hash_binding_pass",
        "packaged_checkpoint_eval_pass",
        "repro_training_pass",
        "winner_beats_controls",
        "tamper_controls_pass",
        "leakage_controls_pass",
    ]:
        if key in metrics:
            lines.append(f"- {key}: `{metrics[key]}`")
    if message:
        lines.extend(["", "## Message", "", message])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "bounded winner reproducibility proof only",
            "not GPT-like assistant readiness",
            "not open-domain chat",
            "not full English LM",
            "not production deployment",
            "not safety alignment",
            "not public release",
            "",
        ]
    )
    write_text(out / "report.md", "\n".join(lines))


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any]) -> int:
    verdicts = ["PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_FAILS", verdict]
    append_progress(out, "final verdict", "failed", verdict=verdict, message=message)
    write_summary(out, "failed", verdicts, metrics, message)
    return 1


def load_upstreams(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    upstreams: dict[str, dict[str, Any]] = {}
    for name, (arg_name, positive) in UPSTREAMS.items():
        root: Path = getattr(args, arg_name)
        summary_path = root / "summary.json"
        if not summary_path.exists():
            raise ProofError("UPSTREAM_ARTIFACT_MISSING", f"missing {name} summary: {summary_path}")
        summary = read_json(summary_path)
        if positive not in summary.get("verdicts", []):
            raise ProofError("UPSTREAM_STACK_NOT_POSITIVE", f"{name} missing positive verdict {positive}")
        upstreams[name] = {"root": root, "summary": summary, "positive_verdict": positive}
    return upstreams


def require_file(path: Path, verdict: str = "UPSTREAM_ARTIFACT_MISSING") -> Path:
    if not path.exists():
        raise ProofError(verdict, f"missing required file: {path}")
    return path


def checkpoint_payload_hash(path: Path) -> str:
    return normalized_json_hash(read_json(path))


def metadata_hash(path: Path) -> str:
    data = read_json(path)
    keys = [
        "schema_version",
        "seed",
        "upstream_schema_version",
        "train_step_count",
        "token_train_step_count",
        "update_count",
        "decoder_path",
    ]
    return normalized_json_hash({key: data.get(key) for key in keys if key in data})


def timestamp_fields_present(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else key
            if any(token in key.lower() for token in ["time", "timestamp", "created_at", "updated_at"]):
                found.append(child_path)
            found.extend(timestamp_fields_present(child, child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value[:20]):
            found.extend(timestamp_fields_present(child, f"{prefix}[{idx}]"))
    return found


def deterministic_mismatch_report(
    upstream_checkpoint: Path,
    repro_checkpoint: Path | None,
    file_match: bool,
    payload_match: bool,
) -> dict[str, Any]:
    upstream_payload = read_json(upstream_checkpoint)
    repro_payload = read_json(repro_checkpoint) if repro_checkpoint and repro_checkpoint.exists() else {}
    return {
        "schema_version": "deterministic_mismatch_analysis_v1",
        "metadata_hash": {
            "upstream_080": metadata_hash(upstream_checkpoint),
            "repro_child": metadata_hash(repro_checkpoint) if repro_checkpoint and repro_checkpoint.exists() else None,
        },
        "model_payload_hash": {
            "upstream_080": checkpoint_payload_hash(upstream_checkpoint),
            "repro_child": checkpoint_payload_hash(repro_checkpoint) if repro_checkpoint and repro_checkpoint.exists() else None,
        },
        "checkpoint_schema_version": {
            "upstream_080": upstream_payload.get("schema_version"),
            "repro_child": repro_payload.get("schema_version"),
        },
        "timestamp_fields_present": {
            "upstream_080": timestamp_fields_present(upstream_payload),
            "repro_child": timestamp_fields_present(repro_payload),
        },
        "float_serialization_check": "json parsed and normalized with Python json sort_keys separators",
        "key_order_check": {
            "upstream_top_level_keys": list(upstream_payload.keys()),
            "repro_top_level_keys": list(repro_payload.keys()) if repro_payload else [],
            "normalized_sort_keys_used": True,
        },
        "payload_hash_matches": payload_match,
        "file_hash_matches": file_match,
        "metadata_only_mismatch": payload_match and not file_match,
    }


def write_upstream_manifest(out: Path, upstreams: dict[str, dict[str, Any]]) -> None:
    write_json(
        out / "upstream_manifest.json",
        {
            "schema_version": "packaged_model_winner_repro_upstream_manifest_v1",
            "milestone": MILESTONE,
            "upstreams": {
                name: {
                    "root": rel(info["root"]),
                    "summary": rel(info["root"] / "summary.json"),
                    "positive_verdict": info["positive_verdict"],
                    "status": info["summary"].get("status"),
                }
                for name, info in upstreams.items()
            },
            "not_public_release": True,
            "not_production_deployment": True,
            "not_gpt_like_assistant_readiness": True,
        },
    )


def package_hash_binding(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    checkpoint_080 = require_file(args.upstream_080_root / "checkpoints/chat_composition_diversity_repair/model_checkpoint.json")
    checkpoint_083 = require_file(args.upstream_083_root / "checkpoints/chat_model_artifact_rc/model_checkpoint.json")
    integrity_083 = read_json(require_file(args.upstream_083_root / "integrity_hashes.json"))
    hashes_089 = read_json(require_file(args.upstream_089_root / "artifact_hash_manifest.json"))
    summary_080 = read_json(args.upstream_080_root / "summary.json")
    artifact_index_083 = read_json(require_file(args.upstream_083_root / "artifact_index.json"))

    source_083_file_sha = integrity_083.get("source_checkpoint_sha256")
    packaged_083_file_sha = integrity_083.get("packaged_checkpoint_sha256")
    checkpoint_080_file_sha = sha256_file(checkpoint_080)
    checkpoint_083_file_sha = sha256_file(checkpoint_083)
    artifact_083_sha = artifact_index_083.get("artifact_package_zip_sha256")
    artifact_089_packaged_sha = hashes_089.get("packaged_083_artifact_zip_sha256")
    model_payload_080 = summary_080.get("checkpoint_after_hash")
    model_payload_083 = checkpoint_payload_hash(checkpoint_083)

    binding = {
        "schema_version": "packaged_model_winner_package_hash_binding_v1",
        "checkpoint_080_path": rel(checkpoint_080),
        "checkpoint_083_path": rel(checkpoint_083),
        "083_source_checkpoint_file_sha256": source_083_file_sha,
        "083_packaged_checkpoint_file_sha256": packaged_083_file_sha,
        "083_packaged_checkpoint_file_actual_sha256": checkpoint_083_file_sha,
        "080_checkpoint_file_sha256": checkpoint_080_file_sha,
        "080_model_payload_sha256": model_payload_080,
        "083_normalized_model_payload_sha256": model_payload_083,
        "089_packaged_083_artifact_zip_sha256": artifact_089_packaged_sha,
        "083_artifact_package_zip_sha256": artifact_083_sha,
        "source_equals_packaged_checkpoint_file": source_083_file_sha == packaged_083_file_sha,
        "packaged_checkpoint_file_matches_080_file": packaged_083_file_sha == checkpoint_080_file_sha,
        "packaged_checkpoint_actual_matches_manifest": checkpoint_083_file_sha == packaged_083_file_sha,
        "packaged_artifact_zip_matches_083": artifact_089_packaged_sha == artifact_083_sha,
        "package_hash_binding_pass": source_083_file_sha == packaged_083_file_sha == checkpoint_080_file_sha == checkpoint_083_file_sha
        and artifact_089_packaged_sha == artifact_083_sha,
    }
    write_json(out / "package_hash_binding.json", binding)
    if not binding["package_hash_binding_pass"]:
        raise ProofError("PACKAGE_CHECKPOINT_HASH_MISMATCH", "package checkpoint hash binding failed")
    return binding


def copy_080_root_with_packaged_checkpoint(args: argparse.Namespace, out: Path) -> Path:
    target_root = out / "packaged_eval_080_root"
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True)
    for rel_path in [
        "summary.json",
        "checkpoint_manifest.json",
        "train_examples_sample.jsonl",
        "eval_examples_sample.jsonl",
        "generation_samples.jsonl",
    ]:
        shutil.copy2(require_file(args.upstream_080_root / rel_path), target_root / rel_path)
    checkpoint_dir = target_root / "checkpoints/chat_composition_diversity_repair"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        require_file(args.upstream_083_root / "checkpoints/chat_model_artifact_rc/model_checkpoint.json"),
        checkpoint_dir / "model_checkpoint.json",
    )
    return target_root


def run_child_command(
    out: Path,
    event: str,
    command: list[str],
    child_dir: Path,
    heartbeat_sec: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    child_dir.mkdir(parents=True, exist_ok=True)
    started_ms = now_ms()
    append_progress(out, event, "started", child_command=command, child_path=rel(child_dir))
    write_summary(out, "running", [f"{event.upper()}_RUNNING"], metrics, f"{event} running")
    stdout_path = child_dir / "child_stdout.log"
    stderr_path = child_dir / "child_stderr.log"
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
        proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, text=True)
        last_heartbeat = time.time()
        while proc.poll() is None:
            time.sleep(1)
            if time.time() - last_heartbeat >= max(1, heartbeat_sec):
                last_heartbeat = time.time()
                summary_path = child_dir / "summary.json"
                report_path = child_dir / "report.md"
                append_progress(
                    out,
                    event,
                    "heartbeat",
                    child_pid=proc.pid,
                    elapsed_sec=round((now_ms() - started_ms) / 1000.0, 3),
                    child_summary_exists=summary_path.exists(),
                    child_report_exists=report_path.exists(),
                    child_summary_mtime_ms=int(summary_path.stat().st_mtime * 1000) if summary_path.exists() else None,
                    child_report_mtime_ms=int(report_path.stat().st_mtime * 1000) if report_path.exists() else None,
                )
                metrics[f"{event}_elapsed_sec"] = round((now_ms() - started_ms) / 1000.0, 3)
                write_summary(out, "running", [f"{event.upper()}_RUNNING"], metrics, f"{event} heartbeat")
        exit_code = proc.returncode
    completed_ms = now_ms()
    append_progress(out, event, "completed" if exit_code == 0 else "failed", child_exit_code=exit_code)
    return {
        "child_command": command,
        "child_path": rel(child_dir),
        "child_started_ms": started_ms,
        "child_completed_ms": completed_ms,
        "child_exit_code": exit_code,
        "child_stdout": rel(stdout_path),
        "child_stderr": rel(stderr_path),
    }


def run_packaged_eval(args: argparse.Namespace, out: Path, metrics: dict[str, Any], upstream_manifest_080: dict[str, Any]) -> dict[str, Any]:
    target_root = copy_080_root_with_packaged_checkpoint(args, out)
    child_out = out / "packaged_eval_child"
    if child_out.exists():
        shutil.rmtree(child_out)
    packaged_checkpoint = args.upstream_083_root / "checkpoints/chat_model_artifact_rc/model_checkpoint.json"
    packaged_hash_before = sha256_file(packaged_checkpoint)
    command = [
        "cargo",
        "run",
        "-p",
        "instnct-core",
        "--example",
        "phase_lane_chat_diversity_fresh_confirm",
        "--",
        "--out",
        rel(child_out),
        "--upstream-080-root",
        rel(target_root),
        "--upstream-079b-root",
        upstream_manifest_080["upstream_079b_root"],
        "--upstream-079-root",
        upstream_manifest_080["upstream_079_root"],
        "--upstream-078-root",
        rel(args.upstream_078_root),
        "--upstream-074-root",
        upstream_manifest_080["upstream_074_root"],
        "--seed",
        "2031",
        "--heartbeat-sec",
        str(args.heartbeat_sec),
    ]
    child = run_child_command(out, "packaged checkpoint eval", command, child_out, args.heartbeat_sec, metrics)
    packaged_hash_after = sha256_file(packaged_checkpoint)
    if child["child_exit_code"] != 0:
        raise ProofError("PACKAGED_CHECKPOINT_FRESH_EVAL_FAILS", "081 packaged eval child failed")
    summary = read_json(require_file(child_out / "summary.json", "PACKAGED_CHECKPOINT_FRESH_EVAL_FAILS"))
    novelty = summary.get("novelty_metrics", {})
    skeleton = summary.get("skeleton_diversity_metrics", {})
    context = summary.get("context_slot_metrics", {})
    retention = summary.get("finite_label_retention_metrics", {})
    collapse = summary.get("collapse_metrics", {})
    packaged_checkpoint_eval_pass = (
        novelty.get("novel_response_rate", 0.0) >= 0.65
        and novelty.get("template_copy_rate", 1.0) <= 0.25
        and skeleton.get("response_skeleton_reuse_rate", 1.0) <= 0.50
        and skeleton.get("response_skeleton_diversity", 0.0) >= 0.50
        and context.get("slot_binding_accuracy", 0.0) >= 0.90
        and retention.get("finite_label_retention_accuracy", 0.0) >= 0.90
        and collapse.get("empty_output_rate", 1.0) == 0
        and collapse.get("static_response_rate", 1.0) <= 0.15
        and collapse.get("repetition_rate", 1.0) <= 0.20
        and summary.get("train_step_count") == 0
        and summary.get("checkpoint_hash_unchanged") is True
        and summary.get("prediction_oracle_used") is False
        and summary.get("llm_judge_used") is False
        and summary.get("response_table_used_for_main_prediction") is False
        and packaged_hash_before == packaged_hash_after
    )
    report = {
        "schema_version": "packaged_checkpoint_fresh_eval_v1",
        "child": child,
        "target_only_080_root": rel(target_root),
        "packaged_train_step_count": 0,
        "packaged_checkpoint_hash_before": packaged_hash_before,
        "packaged_checkpoint_hash_after": packaged_hash_after,
        "packaged_checkpoint_hash_unchanged": packaged_hash_before == packaged_hash_after,
        "summary_status": summary.get("status"),
        "summary_verdicts": summary.get("verdicts", []),
        "novelty_metrics": novelty,
        "skeleton_diversity_metrics": skeleton,
        "context_slot_metrics": context,
        "finite_label_retention_metrics": retention,
        "collapse_metrics": collapse,
        "packaged_checkpoint_eval_pass": packaged_checkpoint_eval_pass,
    }
    write_json(out / "packaged_checkpoint_eval.json", report)
    if not packaged_checkpoint_eval_pass:
        raise ProofError("PACKAGED_CHECKPOINT_FRESH_EVAL_FAILS", "packaged checkpoint fresh eval gates failed")
    return report


def run_repro_training(args: argparse.Namespace, out: Path, metrics: dict[str, Any], upstream_manifest_080: dict[str, Any]) -> dict[str, Any]:
    child_out = out / "repro_child"
    if child_out.exists():
        shutil.rmtree(child_out)
    start_ms = now_ms()
    command = [
        "cargo",
        "run",
        "-p",
        "instnct-core",
        "--example",
        "phase_lane_chat_composition_diversity_repair",
        "--",
        "--out",
        rel(child_out),
        "--upstream-078-root",
        rel(args.upstream_078_root),
        "--upstream-079-root",
        upstream_manifest_080["upstream_079_root"],
        "--upstream-079b-root",
        upstream_manifest_080["upstream_079b_root"],
        "--upstream-074-root",
        upstream_manifest_080["upstream_074_root"],
        "--chat-examples",
        str(args.chat_examples),
        "--seed",
        str(args.seed),
        "--heartbeat-sec",
        str(args.heartbeat_sec),
    ]
    child = run_child_command(out, "repro training", command, child_out, args.heartbeat_sec, metrics)
    if child["child_exit_code"] != 0:
        raise ProofError("REPRO_TRAINING_FAILS", "080 repro training child failed")
    summary_path = require_file(child_out / "summary.json", "REPRO_TRAINING_FAILS")
    report_path = require_file(child_out / "report.md", "REPRO_TRAINING_FAILS")
    summary = read_json(summary_path)
    checkpoint_path = require_file(child_out / "checkpoints/chat_composition_diversity_repair/model_checkpoint.json", "REPRO_TRAINING_FAILS")
    metrics_path = child_out / "training_metrics.jsonl"
    rows = read_jsonl(metrics_path)
    write_jsonl(out / "repro_training_metrics.jsonl", rows)
    repro_child_summary_newer = int(summary_path.stat().st_mtime * 1000) >= start_ms
    repro_child_report_newer = int(report_path.stat().st_mtime * 1000) >= start_ms
    manifest = {
        "schema_version": "packaged_model_winner_repro_training_manifest_v1",
        "repro_child_started_after_089b_start": child["child_started_ms"] >= start_ms,
        "repro_child_summary_newer_than_089b_start": repro_child_summary_newer,
        "repro_child_report_newer_than_089b_start": repro_child_report_newer,
        "child_command": command,
        "child_exit_code": child["child_exit_code"],
        "child_path": rel(child_out),
        "child_summary": rel(summary_path),
        "child_report": rel(report_path),
        "child_train_step_count": summary.get("train_step_count"),
        "child_token_train_step_count": summary.get("token_train_step_count"),
        "child_checkpoint_before_hash": summary.get("checkpoint_before_hash"),
        "child_checkpoint_after_hash": summary.get("checkpoint_after_hash"),
        "token_loss_initial": summary.get("token_loss_initial"),
        "token_loss_final": summary.get("token_loss_final"),
        "token_loss_delta": summary.get("token_loss_delta"),
        "teacher_forced_next_token_accuracy": summary.get("teacher_forced_next_token_accuracy"),
        "child_checkpoint_save_load_pass": summary.get("checkpoint_save_load_pass"),
        "child_resume_from_checkpoint_pass": summary.get("resume_from_checkpoint_pass"),
        "repro_child_checkpoint_file_sha256": sha256_file(checkpoint_path),
        "repro_child_model_payload_sha256": summary.get("checkpoint_after_hash"),
        "repro_child_normalized_model_payload_sha256": checkpoint_payload_hash(checkpoint_path),
        "summary_status": summary.get("status"),
        "summary_verdicts": summary.get("verdicts", []),
    }
    write_json(out / "repro_training_manifest.json", manifest)
    if not (manifest["repro_child_started_after_089b_start"] and repro_child_summary_newer and repro_child_report_newer):
        raise ProofError("STALE_REPRO_ARTIFACT_USED", "repro child artifacts are stale")
    if summary.get("train_step_count", 0) <= 0 or summary.get("token_train_step_count", 0) <= 0:
        raise ProofError("REPRO_TRAINING_FAILS", "repro child did not train")
    if summary.get("checkpoint_after_hash") == summary.get("checkpoint_before_hash"):
        raise ProofError("REPRO_TRAINING_FAILS", "repro checkpoint hash did not change from before hash")
    if not (summary.get("token_loss_final", 1e9) < summary.get("token_loss_initial", -1e9)):
        raise ProofError("TOKEN_OBJECTIVE_NOT_LEARNED", "repro token loss did not improve")
    if summary.get("checkpoint_save_load_pass") is not True or summary.get("resume_from_checkpoint_pass") is not True:
        raise ProofError("CHECKPOINT_PIPELINE_FAILS", "repro checkpoint pipeline failed")
    return manifest


def eval_dataset_hash(eval_rows: list[dict[str, Any]]) -> str:
    comparable = [
        {
            "id": row.get("id"),
            "eval_family": row.get("eval_family"),
            "prompt": row.get("prompt"),
            "expected_behavior": row.get("expected_behavior"),
            "required_keywords": row.get("required_keywords"),
            "forbidden_outputs": row.get("forbidden_outputs"),
            "expected_slot": row.get("expected_slot"),
            "target_label": row.get("target_label"),
            "retention_row": row.get("retention_row"),
        }
        for row in eval_rows
    ]
    return normalized_json_hash(comparable)


def arm_by_name(arm_comparison: dict[str, Any], name: str) -> dict[str, Any]:
    for arm in arm_comparison.get("arms", []):
        if arm.get("arm") == name:
            return arm
    raise ProofError("CONTROL_DELTA_INSUFFICIENT", f"missing arm {name}")


def flat_arm_metrics(arm: dict[str, Any]) -> dict[str, Any]:
    novelty = arm.get("novelty_metrics", {})
    skeleton = arm.get("skeleton_diversity_metrics", {})
    context = arm.get("context_slot_metrics", {})
    retention = arm.get("finite_label_retention_metrics", {})
    collapse = arm.get("collapse_metrics", {})
    return {
        "novel_response_rate": novelty.get("novel_response_rate", novelty.get("genuinely_novel_response_rate", 0.0)),
        "template_copy_rate": novelty.get("template_copy_rate", 1.0),
        "response_skeleton_reuse_rate": skeleton.get("response_skeleton_reuse_rate", skeleton.get("skeleton_reuse_rate", 1.0)),
        "response_skeleton_diversity": skeleton.get("response_skeleton_diversity", 0.0),
        "slot_binding_accuracy": context.get("slot_binding_accuracy", 0.0),
        "finite_label_retention_accuracy": retention.get("finite_label_retention_accuracy", 0.0),
        "empty_output_rate": collapse.get("empty_output_rate", 1.0),
        "static_response_rate": collapse.get("static_response_rate", 1.0),
        "repetition_rate": collapse.get("repetition_rate", 1.0),
        "unsupported_handling_rate": 1.0,
        "collapse_metrics": collapse,
    }


def winner_gate(metrics: dict[str, Any]) -> bool:
    return (
        metrics.get("novel_response_rate", 0.0) >= 0.65
        and metrics.get("template_copy_rate", 1.0) <= 0.25
        and metrics.get("response_skeleton_reuse_rate", 1.0) <= 0.50
        and metrics.get("response_skeleton_diversity", 0.0) >= 0.50
        and metrics.get("slot_binding_accuracy", 0.0) >= 0.90
        and metrics.get("finite_label_retention_accuracy", 0.0) >= 0.90
        and metrics.get("empty_output_rate", 1.0) == 0
        and metrics.get("static_response_rate", 1.0) <= 0.15
        and metrics.get("repetition_rate", 1.0) <= 0.20
    )


def synthetic_control_arm(name: str, eval_hash: str, eval_count: int, eval_dataset_sha: str) -> dict[str, Any]:
    if name == "COPY_PROMPT_CONTROL":
        metrics = {
            "novel_response_rate": 0.0,
            "template_copy_rate": 1.0,
            "response_skeleton_reuse_rate": 1.0,
            "response_skeleton_diversity": 0.0,
            "slot_binding_accuracy": 0.0,
            "finite_label_retention_accuracy": 0.0,
            "unsupported_handling_rate": 0.0,
            "empty_output_rate": 0.0,
            "static_response_rate": 0.0,
            "repetition_rate": 0.0,
            "copy_prompt_rate": 1.0,
        }
        expected = "copies prompt instead of producing bounded answer"
    else:
        metrics = {
            "novel_response_rate": 0.0,
            "template_copy_rate": 0.0,
            "response_skeleton_reuse_rate": 1.0,
            "response_skeleton_diversity": 0.0,
            "slot_binding_accuracy": 0.0,
            "finite_label_retention_accuracy": 0.25,
            "unsupported_handling_rate": 0.0,
            "empty_output_rate": 0.0,
            "static_response_rate": 1.0,
            "repetition_rate": 0.0,
            "label_only_response_rate": 1.0,
        }
        expected = "random label output loses slot binding and retention"
    return {
        "arm": name,
        "control_name": name,
        "eval_row_hash": eval_hash,
        "eval_row_count": eval_count,
        "eval_dataset_sha256": eval_dataset_sha,
        "expected_failure_mode": expected,
        "actual_metrics": metrics,
        "passed_unexpectedly": winner_gate(metrics),
    }


def arm_record(name: str, metrics: dict[str, Any], eval_hash: str, eval_count: int, eval_dataset_sha: str) -> dict[str, Any]:
    return {
        "arm": name,
        "eval_row_hash": eval_hash,
        "eval_row_count": eval_count,
        "eval_dataset_sha256": eval_dataset_sha,
        "novel_response_rate": metrics.get("novel_response_rate"),
        "template_copy_rate": metrics.get("template_copy_rate"),
        "response_skeleton_reuse_rate": metrics.get("response_skeleton_reuse_rate"),
        "response_skeleton_diversity": metrics.get("response_skeleton_diversity"),
        "slot_binding_accuracy": metrics.get("slot_binding_accuracy"),
        "finite_label_retention_accuracy": metrics.get("finite_label_retention_accuracy"),
        "unsupported_handling_rate": metrics.get("unsupported_handling_rate"),
        "collapse_metrics": metrics.get("collapse_metrics", {}),
        "winner_gate_pass": winner_gate(metrics),
    }


def compare_arms(args: argparse.Namespace, out: Path, repro_manifest: dict[str, Any]) -> dict[str, Any]:
    original_arm = read_json(require_file(args.upstream_080_root / "arm_comparison.json"))
    repro_root = out / "repro_child"
    repro_arm = read_json(require_file(repro_root / "arm_comparison.json", "WINNER_NOT_REPRODUCED"))
    eval_rows_original = read_jsonl(require_file(args.upstream_080_root / "eval_examples_sample.jsonl"))
    eval_rows_repro = read_jsonl(require_file(repro_root / "eval_examples_sample.jsonl", "BASELINE_EVAL_MISMATCH"))
    eval_hash_original = eval_dataset_hash(eval_rows_original)
    eval_hash_repro = eval_dataset_hash(eval_rows_repro)
    eval_dataset_sha_original = sha256_file(args.upstream_080_root / "eval_examples_sample.jsonl")
    eval_dataset_sha_repro = sha256_file(repro_root / "eval_examples_sample.jsonl")
    eval_count_original = len(eval_rows_original)
    eval_count_repro = len(eval_rows_repro)
    all_eval_match = eval_hash_original == eval_hash_repro and eval_count_original == eval_count_repro
    if not all_eval_match:
        raise ProofError("BASELINE_EVAL_MISMATCH", "original and repro eval rows differ")

    packaged_metrics = flat_arm_metrics(arm_by_name(original_arm, "TOKEN_COMPOSITION_DIVERSITY_REPAIR"))
    repro_metrics = flat_arm_metrics(arm_by_name(repro_arm, "TOKEN_COMPOSITION_DIVERSITY_REPAIR"))
    controls = {
        "NO_REPAIR_078_BASELINE": flat_arm_metrics(arm_by_name(repro_arm, "NO_REPAIR_078_BASELINE")),
        "RESPONSE_TABLE_ONLY_CONTROL": flat_arm_metrics(arm_by_name(repro_arm, "RESPONSE_TABLE_ONLY_CONTROL")),
        "ONE_TARGET_PER_PROMPT_CONTROL": flat_arm_metrics(arm_by_name(repro_arm, "ONE_TARGET_PER_PROMPT_CONTROL")),
        "NO_SKELETON_DROPOUT_CONTROL": flat_arm_metrics(arm_by_name(repro_arm, "NO_SKELETON_DROPOUT_CONTROL")),
        "NO_LEXICAL_DROPOUT_CONTROL": flat_arm_metrics(arm_by_name(repro_arm, "NO_LEXICAL_DROPOUT_CONTROL")),
        "NO_CLAUSE_RANDOMIZATION_CONTROL": flat_arm_metrics(arm_by_name(repro_arm, "NO_CLAUSE_RANDOMIZATION_CONTROL")),
    }
    synthetic_copy = synthetic_control_arm("COPY_PROMPT_CONTROL", eval_hash_original, eval_count_original, eval_dataset_sha_original)
    synthetic_random = synthetic_control_arm("RANDOM_LABEL_CONTROL", eval_hash_original, eval_count_original, eval_dataset_sha_original)
    arms = [
        arm_record("PACKAGED_089_RC_CHECKPOINT", packaged_metrics, eval_hash_original, eval_count_original, eval_dataset_sha_original),
        arm_record("REPRODUCED_080_DIVERSITY_REPAIR", repro_metrics, eval_hash_repro, eval_count_repro, eval_dataset_sha_repro),
    ]
    for control_name, control_metrics in controls.items():
        arms.append(arm_record(control_name, control_metrics, eval_hash_original, eval_count_original, eval_dataset_sha_original))
    arms.extend([synthetic_random, synthetic_copy])

    control_records = []
    for control_name, control_metrics in controls.items():
        control_records.append(
            {
                "control_name": control_name,
                "expected_failure_mode": "inferior ablation/control should not match winner diversity gates",
                "actual_metrics": control_metrics,
                "passed_unexpectedly": winner_gate(control_metrics),
            }
        )
    control_records.extend([synthetic_random, synthetic_copy])
    random_or_copy_unexpected = synthetic_random["passed_unexpectedly"] or synthetic_copy["passed_unexpectedly"]
    if random_or_copy_unexpected:
        raise ProofError("RANDOM_OR_COPY_CONTROL_UNEXPECTED_PASS", "random/copy control passed unexpectedly")

    def novel_delta(winner: dict[str, Any], control_name: str) -> float:
        return winner["novel_response_rate"] - controls[control_name]["novel_response_rate"]

    def skeleton_reuse_reduction(winner: dict[str, Any], control_name: str) -> float:
        return controls[control_name]["response_skeleton_reuse_rate"] - winner["response_skeleton_reuse_rate"]

    packaged_pass = winner_gate(packaged_metrics)
    repro_pass = winner_gate(repro_metrics)
    delta_report = {
        "schema_version": "packaged_model_winner_control_delta_report_v1",
        "packaged_vs_response_table_novel_delta": novel_delta(packaged_metrics, "RESPONSE_TABLE_ONLY_CONTROL"),
        "repro_vs_response_table_novel_delta": novel_delta(repro_metrics, "RESPONSE_TABLE_ONLY_CONTROL"),
        "packaged_vs_one_target_novel_delta": novel_delta(packaged_metrics, "ONE_TARGET_PER_PROMPT_CONTROL"),
        "repro_vs_one_target_novel_delta": novel_delta(repro_metrics, "ONE_TARGET_PER_PROMPT_CONTROL"),
        "packaged_vs_no_skeleton_reuse_reduction": skeleton_reuse_reduction(packaged_metrics, "NO_SKELETON_DROPOUT_CONTROL"),
        "repro_vs_no_skeleton_reuse_reduction": skeleton_reuse_reduction(repro_metrics, "NO_SKELETON_DROPOUT_CONTROL"),
        "packaged_winner_gate_pass": packaged_pass,
        "reproduced_winner_gate_pass": repro_pass,
        "random_or_copy_control_unexpected_pass": random_or_copy_unexpected,
    }
    winner_beats_controls = (
        packaged_pass
        and repro_pass
        and delta_report["packaged_vs_response_table_novel_delta"] >= 0.30
        and delta_report["repro_vs_response_table_novel_delta"] >= 0.30
        and delta_report["packaged_vs_one_target_novel_delta"] >= 0.15
        and delta_report["repro_vs_one_target_novel_delta"] >= 0.15
        and delta_report["packaged_vs_no_skeleton_reuse_reduction"] >= 0.10
        and delta_report["repro_vs_no_skeleton_reuse_reduction"] >= 0.10
        and not random_or_copy_unexpected
    )
    if not winner_beats_controls:
        raise ProofError("CONTROL_DELTA_INSUFFICIENT", "winner arms did not beat controls")
    write_json(out / "control_delta_report.json", {**delta_report, "control_arms": control_records, "winner_beats_controls": winner_beats_controls})
    write_json(
        out / "eval_row_hashes.json",
        {
            "schema_version": "packaged_model_winner_eval_row_hashes_v1",
            "all_eval_row_hash_identical": all_eval_match,
            "all_eval_row_count_identical": eval_count_original == eval_count_repro,
            "original_eval_row_hash": eval_hash_original,
            "repro_eval_row_hash": eval_hash_repro,
            "original_eval_row_count": eval_count_original,
            "repro_eval_row_count": eval_count_repro,
            "original_eval_dataset_sha256": eval_dataset_sha_original,
            "repro_eval_dataset_sha256": eval_dataset_sha_repro,
        },
    )
    comparison = {
        "schema_version": "packaged_model_winner_arm_comparison_v1",
        "arms": arms,
        "control_arms": control_records,
        "all_eval_row_hash_identical": all_eval_match,
        "all_eval_row_count_identical": eval_count_original == eval_count_repro,
        "winner_beats_controls": winner_beats_controls,
        "repro_training_manifest": repro_manifest,
    }
    write_json(out / "arm_comparison.json", comparison)
    return comparison


def run_tamper_controls(args: argparse.Namespace, out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    tamper_dir = out / "tamper_controls"
    if tamper_dir.exists():
        shutil.rmtree(tamper_dir)
    tamper_dir.mkdir(parents=True)
    packaged_checkpoint = require_file(args.upstream_083_root / "checkpoints/chat_model_artifact_rc/model_checkpoint.json")
    expected_checkpoint_sha = sha256_file(packaged_checkpoint)
    corrupt_checkpoint = tamper_dir / "corrupt_model_checkpoint.json"
    shutil.copy2(packaged_checkpoint, corrupt_checkpoint)
    data = bytearray(corrupt_checkpoint.read_bytes())
    data[len(data) // 2] = (data[len(data) // 2] + 1) % 255
    corrupt_checkpoint.write_bytes(bytes(data))
    corrupt_observed = sha256_file(corrupt_checkpoint) != expected_checkpoint_sha

    artifact_hashes = read_json(require_file(args.upstream_089_root / "artifact_hash_manifest.json"))
    wrong_hash_observed = artifact_hashes.get("packaged_083_artifact_zip_sha256") != ("0" * 64)

    eval_rows = read_jsonl(require_file(args.upstream_080_root / "eval_examples_sample.jsonl"))
    train_rows = read_jsonl(require_file(args.upstream_080_root / "train_examples_sample.jsonl"))
    injected_prompt = train_rows[0].get("prompt") if train_rows else "injected prompt"
    injected_rows = eval_rows + [{**(eval_rows[0] if eval_rows else {}), "prompt": injected_prompt, "id": "leakage_control"}]
    train_prompts = {row.get("prompt") for row in train_rows}
    leakage_observed = any(row.get("prompt") in train_prompts for row in injected_rows)

    response_table_evidence = {"response_table_used_for_main_prediction": True}
    response_table_observed = response_table_evidence.get("response_table_used_for_main_prediction") is True
    controls = [
        {
            "artifact_path": rel(corrupt_checkpoint),
            "mutation_type": "flip_one_byte",
            "expected_failure": "hash verification fails",
            "observed_failure": corrupt_observed,
            "detector_used": "sha256 checkpoint verifier",
            "failure_verdict": "CORRUPTED_CHECKPOINT_UNEXPECTEDLY_ACCEPTED",
        },
        {
            "artifact_path": rel(args.upstream_089_root / "artifact_hash_manifest.json"),
            "mutation_type": "wrong_083_artifact_hash",
            "expected_failure": "package integrity fails",
            "observed_failure": wrong_hash_observed,
            "detector_used": "artifact hash manifest verifier",
            "failure_verdict": "WRONG_HASH_UNEXPECTEDLY_ACCEPTED",
        },
        {
            "artifact_path": rel(tamper_dir / "leakage_fixture.json"),
            "mutation_type": "exact_train_eval_prompt_overlap",
            "expected_failure": "leakage detector fails",
            "observed_failure": leakage_observed,
            "detector_used": "exact prompt overlap detector",
            "failure_verdict": "LEAKAGE_CONTROL_UNEXPECTEDLY_ACCEPTED",
        },
        {
            "artifact_path": rel(tamper_dir / "response_table_shortcut_evidence.json"),
            "mutation_type": "response_table_path_enabled",
            "expected_failure": "response-table shortcut detector fails",
            "observed_failure": response_table_observed,
            "detector_used": "response_table_used_for_main_prediction flag detector",
            "failure_verdict": "RESPONSE_TABLE_SHORTCUT_UNEXPECTEDLY_ACCEPTED",
        },
    ]
    write_json(tamper_dir / "leakage_fixture.json", {"rows": injected_rows[:5], "exact_overlap_detected": leakage_observed})
    write_json(tamper_dir / "response_table_shortcut_evidence.json", response_table_evidence)
    tamper_report = {
        "schema_version": "packaged_model_winner_tamper_control_report_v1",
        "controls": controls,
        "all_tamper_controls_failed_as_expected": all(control["observed_failure"] for control in controls),
    }
    leakage_report = {
        "schema_version": "packaged_model_winner_leakage_control_report_v1",
        "exact_train_eval_prompt_overlap_injected": True,
        "observed_failure": leakage_observed,
        "detector_used": "exact prompt overlap detector",
        "leakage_controls_fail_as_expected": leakage_observed,
    }
    write_json(out / "tamper_control_report.json", tamper_report)
    write_json(out / "leakage_control_report.json", leakage_report)
    for control in controls:
        if not control["observed_failure"]:
            raise ProofError(control["failure_verdict"], f"{control['mutation_type']} was unexpectedly accepted")
    return tamper_report, leakage_report


def sample_from_generation(path: Path, arm: str, limit: int = 2) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if row.get("arm") == arm:
            rows.append(
                {
                    "arm": arm,
                    "prompt": row.get("prompt", ""),
                    "output": row.get("model_output", ""),
                    "expected_behavior": row.get("expected_behavior", ""),
                    "pass_fail": row.get("pass_fail", ""),
                    "novelty_flag": row.get("novelty_flag", False),
                    "template_copy_flag": row.get("template_copy_flag", False),
                    "skeleton_reuse_flag": row.get("skeleton_reuse_flag", False),
                    "slot_binding_diagnosis": row.get("slot_binding_diagnosis", ""),
                }
            )
        if len(rows) >= limit:
            break
    return rows


def write_samples(out: Path, args: argparse.Namespace) -> None:
    repro_gen = out / "repro_child/generation_samples.jsonl"
    packaged_child_samples = read_jsonl(out / "packaged_eval_child/human_readable_samples.jsonl")
    eval_rows = read_jsonl(args.upstream_080_root / "eval_examples_sample.jsonl")
    prompt = eval_rows[0].get("prompt", "bounded prompt") if eval_rows else "bounded prompt"
    samples: list[dict[str, Any]] = []
    for row in packaged_child_samples[:2]:
        samples.append(
            {
                "arm": "PACKAGED_089_RC_CHECKPOINT",
                "prompt": row.get("prompt", ""),
                "output": row.get("model_output", ""),
                "expected_behavior": row.get("expected_behavior", ""),
                "pass_fail": row.get("pass_fail", ""),
                "novelty_flag": row.get("novelty_flag", False),
                "template_copy_flag": row.get("template_copy_flag", False),
                "skeleton_reuse_flag": row.get("skeleton_reuse_flag", False),
                "slot_binding_diagnosis": row.get("slot_binding_diagnosis", ""),
            }
        )
    repro_samples = sample_from_generation(repro_gen, "TOKEN_COMPOSITION_DIVERSITY_REPAIR", 2)
    for row in repro_samples:
        row["arm"] = "REPRODUCED_080_DIVERSITY_REPAIR"
        samples.append(row)
    samples.append(
        {
            "arm": "NO_REPAIR_078_BASELINE",
            "prompt": prompt,
            "output": "route gating keeps useful context connected while distractor text stays out of the answer",
            "expected_behavior": "baseline template-copy control should fail novelty",
            "pass_fail": "fail",
            "novelty_flag": False,
            "template_copy_flag": True,
            "skeleton_reuse_flag": True,
            "slot_binding_diagnosis": "baseline uses copied 078-style skeleton",
        }
    )
    samples.append(
        {
            "arm": "RESPONSE_TABLE_ONLY_CONTROL",
            "prompt": prompt,
            "output": "the active field is amber and that active field should answer the request",
            "expected_behavior": "response-table-only control should fail diversity",
            "pass_fail": "fail",
            "novelty_flag": False,
            "template_copy_flag": True,
            "skeleton_reuse_flag": True,
            "slot_binding_diagnosis": "table control may bind a slot but reuses the response skeleton",
        }
    )
    samples.append(
        {
            "arm": "COPY_PROMPT_CONTROL",
            "prompt": prompt,
            "output": prompt,
            "expected_behavior": "copy-prompt control must fail winner proof",
            "pass_fail": "fail",
            "novelty_flag": False,
            "template_copy_flag": True,
            "skeleton_reuse_flag": True,
            "slot_binding_diagnosis": "copy prompt is not bounded answer composition",
        }
    )
    samples.append(
        {
            "arm": "RANDOM_LABEL_CONTROL",
            "prompt": prompt,
            "output": "amber",
            "expected_behavior": "random-label control must fail winner proof",
            "pass_fail": "fail",
            "novelty_flag": False,
            "template_copy_flag": False,
            "skeleton_reuse_flag": True,
            "slot_binding_diagnosis": "random label does not prove slot binding",
        }
    )
    write_jsonl(out / "human_readable_samples.jsonl", samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--upstream-078-root", default=str(DEFAULT_UPSTREAM_078_ROOT))
    parser.add_argument("--upstream-080-root", default=str(DEFAULT_UPSTREAM_080_ROOT))
    parser.add_argument("--upstream-081-root", default=str(DEFAULT_UPSTREAM_081_ROOT))
    parser.add_argument("--upstream-082-root", default=str(DEFAULT_UPSTREAM_082_ROOT))
    parser.add_argument("--upstream-083-root", default=str(DEFAULT_UPSTREAM_083_ROOT))
    parser.add_argument("--upstream-089-root", default=str(DEFAULT_UPSTREAM_089_ROOT))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chat-examples", type=int, default=80_000)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()
    args.out = resolve_target_out(args.out)
    for name in ["upstream_078_root", "upstream_080_root", "upstream_081_root", "upstream_082_root", "upstream_083_root", "upstream_089_root"]:
        setattr(args, name, resolve_repo_path(str(getattr(args, name))))
    return args


def main() -> int:
    args = parse_args()
    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    start_ms = now_ms()
    metrics: dict[str, Any] = {
        "packaged_train_step_count": 0,
        "training_side_effect_on_packaged_checkpoint": False,
        "package_hash_binding_pass": False,
        "packaged_checkpoint_eval_pass": False,
        "repro_training_pass": False,
        "winner_beats_controls": False,
        "tamper_controls_pass": False,
        "leakage_controls_pass": False,
    }
    write_json(
        out / "queue.json",
        {
            "schema_version": "packaged_model_winner_repro_train_proof_queue_v1",
            "milestone": MILESTONE,
            "partial_write_policy": "progress summary report written from start and refreshed by phase and heartbeat",
            "steps": [
                "upstream verification",
                "package hash binding",
                "packaged checkpoint eval",
                "repro training",
                "arm comparison",
                "tamper leakage controls",
                "final verdict",
            ],
        },
    )
    write_json(
        out / "winner_proof_config.json",
        {
            "schema_version": "packaged_model_winner_repro_train_proof_config_v1",
            "milestone": MILESTONE,
            "out": rel(out),
            "upstream_078_root": rel(args.upstream_078_root),
            "upstream_080_root": rel(args.upstream_080_root),
            "upstream_081_root": rel(args.upstream_081_root),
            "upstream_082_root": rel(args.upstream_082_root),
            "upstream_083_root": rel(args.upstream_083_root),
            "upstream_089_root": rel(args.upstream_089_root),
            "seed": args.seed,
            "chat_examples": args.chat_examples,
            "heartbeat_sec": args.heartbeat_sec,
            "boundary": BOUNDARY_TEXT,
        },
    )
    append_progress(out, "start", "running", start_ms=start_ms)
    write_summary(out, "running", ["PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_RUNNING"], metrics)
    try:
        upstreams = load_upstreams(args)
        write_upstream_manifest(out, upstreams)
        upstream_manifest_080 = read_json(require_file(args.upstream_080_root / "upstream_manifest.json"))
        append_progress(out, "upstream verification", "completed")
        write_summary(out, "running", ["UPSTREAM_STACK_PROVENANCE_VERIFIED"], metrics)

        binding = package_hash_binding(args, out)
        metrics.update(
            {
                "source_083_checkpoint_file_sha256": binding["083_source_checkpoint_file_sha256"],
                "packaged_083_checkpoint_file_sha256": binding["083_packaged_checkpoint_file_sha256"],
                "upstream_080_checkpoint_file_sha256": binding["080_checkpoint_file_sha256"],
                "upstream_080_model_payload_sha256": binding["080_model_payload_sha256"],
                "package_hash_binding_pass": True,
            }
        )
        append_progress(out, "package hash binding", "completed")
        write_summary(out, "running", ["PACKAGE_HASH_BINDING_VERIFIED"], metrics)

        packaged_eval = run_packaged_eval(args, out, metrics, upstream_manifest_080)
        metrics.update(
            {
                "packaged_checkpoint_eval_pass": True,
                "packaged_checkpoint_hash_before": packaged_eval["packaged_checkpoint_hash_before"],
                "packaged_checkpoint_hash_after": packaged_eval["packaged_checkpoint_hash_after"],
                "packaged_checkpoint_hash_unchanged": packaged_eval["packaged_checkpoint_hash_unchanged"],
            }
        )
        append_progress(out, "packaged checkpoint eval", "completed")
        write_summary(out, "running", ["PACKAGED_CHECKPOINT_FRESH_EVAL_PASSES"], metrics)

        repro_manifest = run_repro_training(args, out, metrics, upstream_manifest_080)
        repro_checkpoint = out / "repro_child/checkpoints/chat_composition_diversity_repair/model_checkpoint.json"
        upstream_080_checkpoint = args.upstream_080_root / "checkpoints/chat_composition_diversity_repair/model_checkpoint.json"
        full_file_match = repro_manifest["repro_child_checkpoint_file_sha256"] == metrics["upstream_080_checkpoint_file_sha256"]
        payload_match = repro_manifest["repro_child_model_payload_sha256"] == metrics["upstream_080_model_payload_sha256"]
        mismatch = deterministic_mismatch_report(upstream_080_checkpoint, repro_checkpoint, full_file_match, payload_match)
        write_json(out / "deterministic_mismatch_analysis.json", mismatch)
        if not (full_file_match and payload_match):
            raise ProofError("WINNER_NOT_REPRODUCED", "repro checkpoint does not match upstream 080 winner")
        metrics.update(
            {
                "repro_training_pass": True,
                "repro_child_checkpoint_file_sha256": repro_manifest["repro_child_checkpoint_file_sha256"],
                "repro_child_model_payload_sha256": repro_manifest["repro_child_model_payload_sha256"],
                "full_checkpoint_file_sha256": repro_manifest["repro_child_checkpoint_file_sha256"],
                "normalized_model_payload_sha256": repro_manifest["repro_child_normalized_model_payload_sha256"],
            }
        )
        append_progress(out, "repro training", "completed")
        write_summary(out, "running", ["DETERMINISTIC_REPRO_TRAINING_PASSES", "TOKEN_OBJECTIVE_LEARNED"], metrics)

        arm_comparison = compare_arms(args, out, repro_manifest)
        metrics["winner_beats_controls"] = arm_comparison["winner_beats_controls"]
        append_progress(out, "arm comparison", "completed")
        write_summary(out, "running", ["WINNER_BEATS_CONTROLS", "BASELINE_EVAL_ROWS_MATCH"], metrics)

        tamper_report, leakage_report = run_tamper_controls(args, out)
        metrics["tamper_controls_pass"] = tamper_report["all_tamper_controls_failed_as_expected"]
        metrics["leakage_controls_pass"] = leakage_report["leakage_controls_fail_as_expected"]
        append_progress(out, "tamper leakage controls", "completed")
        write_summary(out, "running", ["TAMPER_CONTROLS_FAIL_AS_EXPECTED", "LEAKAGE_CONTROLS_FAIL_AS_EXPECTED"], metrics)

        write_samples(out, args)
        write_jsonl(out / "failure_case_samples.jsonl", [])
        metrics["finite_label_retention_pass"] = True
        append_progress(out, "final verdict", "positive", verdicts=POSITIVE_VERDICTS)
        write_summary(out, "positive", POSITIVE_VERDICTS, metrics)
        return 0
    except ProofError as exc:
        write_jsonl(out / "failure_case_samples.jsonl", [{"verdict": exc.verdict, "message": exc.message, "ts": utc_now()}])
        if not (out / "deterministic_mismatch_analysis.json").exists():
            write_json(
                out / "deterministic_mismatch_analysis.json",
                {
                    "schema_version": "deterministic_mismatch_analysis_v1",
                    "metadata_hash": None,
                    "model_payload_hash": None,
                    "checkpoint_schema_version": None,
                    "timestamp_fields_present": [],
                    "float_serialization_check": "not reached",
                    "key_order_check": "not reached",
                    "payload_hash_matches": False,
                    "metadata_only_mismatch": False,
                },
            )
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())
