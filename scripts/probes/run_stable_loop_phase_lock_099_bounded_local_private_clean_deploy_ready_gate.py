#!/usr/bin/env python3
"""Clean local/private deploy-readiness gate for the bounded stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE"
DEFAULT_OUT = Path("target/pilot_wave/stable_loop_phase_lock_099_bounded_local_private_clean_deploy_ready_gate/smoke")
DEFAULT_DEPLOY_CONFIG = Path("tools/instnct_deploy/config/example.local.json")
DEFAULT_UPSTREAM_098_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_098_private_eval_rc_refresh_with_generation_repair/smoke")
DEFAULT_UPSTREAM_089B_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_089b_packaged_model_winner_repro_train_proof/smoke")
DEFAULT_UPSTREAM_088_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_088_bounded_chat_long_run_concurrency_resource_stability/smoke")
BOUNDARY_TEXT = (
    "099 is a clean local/private bounded deploy-readiness gate. It runs a fresh local/private deployment "
    "harness smoke into a target-only output directory and binds prior evaluation evidence. It is not "
    "production deployment, not public API, not hosted SaaS, not GPT-like assistant readiness, not "
    "open-domain chat, not production chat, and not safety alignment."
)


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


def sha256_file(path: Path) -> str:
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


def resolve_repo_path(text: str, verdict: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path.resolve()
    if any(part == ".." for part in path.parts):
        raise GateError(verdict, f"path must be repo-relative: {text}")
    return (REPO_ROOT / path).resolve()


def resolve_target_out(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise GateError("CLEAN_DEPLOY_READY_ARTIFACT_MISSING", "--out must be repo-relative")
    parts = [part.lower() for part in path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("CLEAN_DEPLOY_READY_ARTIFACT_MISSING", "--out must stay under target/pilot_wave")
    return (REPO_ROOT / path).resolve()


def append_progress(out: Path, event: str, status: str = "completed", **details: Any) -> None:
    append_jsonl(out / "progress.jsonl", {"ts": utc_now(), "event": event, "status": status, "details": details})


def write_summary(out: Path, status: str, verdicts: list[str], metrics: dict[str, Any], message: str = "") -> None:
    payload = {
        "schema_version": "bounded_local_private_clean_deploy_ready_summary_v1",
        "milestone": MILESTONE,
        "status": status,
        "phase": status,
        "boundary": BOUNDARY_TEXT,
        "local_private_release_ready_claimed": status == "positive",
        "production_deployment_claimed": False,
        "public_api_claimed": False,
        "hosted_saas_claimed": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_chat_claimed": False,
        "production_chat_claimed": False,
        "safety_alignment_claimed": False,
        "metrics": metrics,
        "verdicts": verdicts,
    }
    if message:
        payload["message"] = message
    write_json(out / "summary.json", payload)
    lines = [
        "# STABLE_LOOP_PHASE_LOCK_099_BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE Report",
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
        "## Metrics",
        "",
    ]
    for key in [
        "fresh_harness_smoke_exit_code",
        "deployment_harness_gate_pass",
        "sdk_smoke_still_passes",
        "bounded_chat_service_smoke_pass",
        "artifact_hash_verified",
        "checkpoint_hash_unchanged",
        "rollback_pointer_written",
        "upstream_098_positive",
        "upstream_089b_positive",
        "upstream_088_positive",
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
            "local/private release-readiness only",
            "not production deployment",
            "not public API",
            "not hosted SaaS",
            "not GPT-like assistant readiness",
            "not open-domain chat",
            "not production chat",
            "not safety alignment",
        ]
    )
    write_text(out / "report.md", "\n".join(lines))


def fail(out: Path, verdict: str, message: str, metrics: dict[str, Any]) -> int:
    append_progress(out, "final verdict", "failed", verdict=verdict, message=message)
    write_summary(out, "failed", ["BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_FAILS", verdict], metrics, message)
    return 1


def verify_summary(root: Path, positive: str, missing_verdict: str, not_positive_verdict: str) -> dict[str, Any]:
    summary = root / "summary.json"
    if not summary.exists():
        raise GateError(missing_verdict, f"missing summary: {root}")
    payload = read_json(summary)
    if positive not in set(payload.get("verdicts", [])):
        raise GateError(not_positive_verdict, f"positive verdict missing: {positive}")
    return payload


def generate_clean_config(out: Path, deploy_config: Path) -> Path:
    config = read_json(deploy_config)
    config["deployment_mode"] = "local_research"
    config["intended_use"] = "research"
    config["out_dir"] = rel(out / "deployment_harness_smoke")
    config["bounded_chat_service_smoke_out"] = rel(out / "deployment_harness_service_smoke")
    config["production_default_training_enabled"] = False
    config["public_beta_promoted"] = False
    config["production_api_ready"] = False
    config_path = out / "generated_clean_local_private_deploy_config.json"
    write_json(config_path, config)
    return config_path


def run_child_with_heartbeat(out: Path, command: list[str], heartbeat_sec: int, timeout_sec: int) -> tuple[int, str, str]:
    append_progress(out, "fresh harness child start", "running", command=command)
    proc = subprocess.Popen(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    started = time.time()
    last = started
    while proc.poll() is None:
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            append_progress(out, "fresh harness child heartbeat", "running", pid=proc.pid, elapsed_sec=round(now - started, 1))
            write_summary(out, "running", ["FRESH_DEPLOYMENT_HARNESS_SMOKE_RUNNING"], {"fresh_harness_child_pid": proc.pid, "fresh_harness_elapsed_sec": round(now - started, 1)})
        if now - started > timeout_sec:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=10)
            raise GateError("FRESH_HARNESS_SMOKE_TIMEOUT", "fresh harness smoke timed out")
        time.sleep(1)
    stdout, stderr = proc.communicate()
    append_progress(out, "fresh harness child exit", "completed", exit_code=proc.returncode, elapsed_sec=round(time.time() - started, 1))
    return proc.returncode, stdout, stderr


def main() -> int:
    started = time.time()
    parser = argparse.ArgumentParser(description=MILESTONE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--deploy-config", default=str(DEFAULT_DEPLOY_CONFIG))
    parser.add_argument("--upstream-098-root", default=str(DEFAULT_UPSTREAM_098_ROOT))
    parser.add_argument("--upstream-089b-root", default=str(DEFAULT_UPSTREAM_089B_ROOT))
    parser.add_argument("--upstream-088-root", default=str(DEFAULT_UPSTREAM_088_ROOT))
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--timeout-sec", type=int, default=900)
    args = parser.parse_args()
    out = resolve_target_out(args.out)
    deploy_config = resolve_repo_path(str(args.deploy_config), "DEPLOY_CONFIG_MISSING")
    roots = {
        "098": resolve_repo_path(str(args.upstream_098_root), "UPSTREAM_098_ARTIFACT_MISSING"),
        "089b": resolve_repo_path(str(args.upstream_089b_root), "UPSTREAM_089B_ARTIFACT_MISSING"),
        "088": resolve_repo_path(str(args.upstream_088_root), "UPSTREAM_088_ARTIFACT_MISSING"),
    }
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "production_deployment_claimed": False,
        "public_api_claimed": False,
        "hosted_saas_claimed": False,
        "gpt_like_assistant_readiness_claimed": False,
        "open_domain_chat_claimed": False,
        "production_chat_claimed": False,
        "safety_alignment_claimed": False,
    }
    write_json(out / "queue.json", {"schema_version": "bounded_clean_deploy_ready_queue_v1", "milestone": MILESTONE, "partial_write_policy": "progress summary report written from start and refreshed at heartbeat during fresh harness child", "steps": ["verify_upstreams", "generate_clean_config", "run_fresh_harness_smoke", "validate_evidence_chain", "final"]})
    append_progress(out, "start", "running")
    write_summary(out, "running", ["BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_RUNNING"], metrics)
    try:
        s098 = verify_summary(roots["098"], "PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_POSITIVE", "UPSTREAM_098_ARTIFACT_MISSING", "UPSTREAM_098_NOT_POSITIVE")
        s089b = verify_summary(roots["089b"], "PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_POSITIVE", "UPSTREAM_089B_ARTIFACT_MISSING", "UPSTREAM_089B_NOT_POSITIVE")
        s088 = verify_summary(roots["088"], "BOUNDED_CHAT_LONG_RUN_CONCURRENCY_RESOURCE_STABILITY_POSITIVE", "UPSTREAM_088_ARTIFACT_MISSING", "UPSTREAM_088_NOT_POSITIVE")
        metrics.update({"upstream_098_positive": True, "upstream_089b_positive": True, "upstream_088_positive": True})
        write_json(out / "upstream_release_manifest.json", {"schema_version": "bounded_clean_deploy_ready_upstream_manifest_v1", "upstreams": {"098": {"root": rel(roots["098"]), "verdict": "PRIVATE_EVAL_RC_REFRESH_WITH_GENERATION_REPAIR_POSITIVE"}, "089b": {"root": rel(roots["089b"]), "verdict": "PACKAGED_MODEL_WINNER_REPRO_TRAIN_PROOF_POSITIVE"}, "088": {"root": rel(roots["088"]), "verdict": "BOUNDED_CHAT_LONG_RUN_CONCURRENCY_RESOURCE_STABILITY_POSITIVE"}}, "key_metrics": {"098_refresh_package_zip_sha256": s098["metrics"].get("refresh_package_zip_sha256"), "089b_package_hash_binding_pass": s089b["metrics"].get("package_hash_binding_pass"), "088_total_requests": s088["metrics"].get("total_requests")}})
        append_progress(out, "upstream verification", "completed")
        write_summary(out, "running", ["UPSTREAM_RELEASE_EVIDENCE_VERIFIED"], metrics)

        config_path = generate_clean_config(out, deploy_config)
        config_hash = sha256_file(config_path)
        write_json(out / "clean_deploy_config_manifest.json", {"schema_version": "bounded_clean_deploy_config_manifest_v1", "source_config": rel(deploy_config), "generated_config": rel(config_path), "generated_config_sha256": config_hash, "local_private_only": True})
        append_progress(out, "clean config generated", "completed", config=rel(config_path))
        command = [sys.executable, "tools/instnct_deploy/instnct_deploy.py", "smoke", "--config", rel(config_path), "--out", rel(out / "deployment_harness_smoke")]
        exit_code, stdout, stderr = run_child_with_heartbeat(out, command, args.heartbeat_sec, args.timeout_sec)
        (out / "fresh_harness_stdout.txt").write_text(stdout, encoding="utf-8")
        (out / "fresh_harness_stderr.txt").write_text(stderr, encoding="utf-8")
        metrics["fresh_harness_smoke_exit_code"] = exit_code
        if exit_code != 0:
            raise GateError("FRESH_HARNESS_SMOKE_FAILS", f"fresh harness smoke exited {exit_code}")
        harness_summary_path = out / "deployment_harness_smoke" / "summary.json"
        if not harness_summary_path.exists():
            raise GateError("FRESH_HARNESS_ARTIFACT_MISSING", "fresh harness summary missing")
        harness_summary = read_json(harness_summary_path)
        harness_verdicts = set(harness_summary.get("verdicts", []))
        required_harness = {
            "BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_POSITIVE",
            "SDK_SMOKE_THROUGH_HARNESS_STILL_PASSES",
            "BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_PASSES",
            "CHECKPOINT_UNCHANGED_THROUGH_HARNESS",
            "ROLLBACK_POINTER_WRITTEN",
        }
        if not required_harness.issubset(harness_verdicts):
            raise GateError("FRESH_HARNESS_SMOKE_FAILS", "fresh harness positive verdicts missing")
        for key in ["deployment_harness_gate_pass", "sdk_smoke_still_passes", "bounded_chat_service_smoke_pass", "artifact_hash_verified", "checkpoint_hash_unchanged", "rollback_pointer_written"]:
            metrics[key] = bool(harness_summary.get(key))
        metrics["train_step_count"] = harness_summary.get("train_step_count", 0)
        metrics["fresh_harness_summary_newer_than_099_start"] = harness_summary_path.stat().st_mtime >= started
        if not all(metrics[key] for key in ["deployment_harness_gate_pass", "sdk_smoke_still_passes", "bounded_chat_service_smoke_pass", "artifact_hash_verified", "checkpoint_hash_unchanged", "rollback_pointer_written"]):
            raise GateError("FRESH_HARNESS_GATE_FAILS", "fresh harness gates did not all pass")
        if metrics["train_step_count"] != 0:
            raise GateError("TRAINING_SIDE_EFFECT_DETECTED", "fresh harness reported train_step_count != 0")
        write_json(out / "fresh_harness_child_manifest.json", {"schema_version": "bounded_clean_deploy_fresh_harness_child_manifest_v1", "command": command, "exit_code": exit_code, "harness_summary": rel(harness_summary_path), "harness_stdout": "fresh_harness_stdout.txt", "harness_stderr": "fresh_harness_stderr.txt"})
        write_json(out / "fresh_harness_validation.json", {"schema_version": "bounded_clean_deploy_fresh_harness_validation_v1", "required_harness_verdicts": sorted(required_harness), "present_harness_verdicts": sorted(harness_verdicts), "metrics": {key: metrics[key] for key in ["deployment_harness_gate_pass", "sdk_smoke_still_passes", "bounded_chat_service_smoke_pass", "artifact_hash_verified", "checkpoint_hash_unchanged", "rollback_pointer_written", "train_step_count", "fresh_harness_summary_newer_than_099_start"]}})
        write_json(out / "release_readiness_evidence_chain.json", {"schema_version": "bounded_clean_deploy_release_readiness_evidence_chain_v1", "fresh_harness_smoke_positive": True, "private_eval_rc_refresh_positive": True, "packaged_winner_repro_positive": True, "long_run_stability_positive": True, "local_private_release_ready": True, "production_deployment_claimed": False})
        write_json(out / "claim_boundary.json", {"schema_version": "bounded_clean_deploy_claim_boundary_v1", "local_private_release_ready_claimed": True, "production_deployment_claimed": False, "public_api_claimed": False, "hosted_saas_claimed": False, "gpt_like_assistant_readiness_claimed": False, "open_domain_chat_claimed": False, "production_chat_claimed": False, "safety_alignment_claimed": False})
        metrics.update({"local_private_release_ready": True, "wall_clock_sec": round(time.time() - started, 3)})
        append_progress(out, "final verdict", "positive")
        write_summary(
            out,
            "positive",
            [
                "BOUNDED_LOCAL_PRIVATE_CLEAN_DEPLOY_READY_GATE_POSITIVE",
                "UPSTREAM_RELEASE_EVIDENCE_VERIFIED",
                "CLEAN_LOCAL_PRIVATE_CONFIG_GENERATED",
                "FRESH_DEPLOYMENT_HARNESS_SMOKE_PASSES",
                "SDK_SMOKE_STILL_PASSES",
                "BOUNDED_CHAT_SERVICE_SMOKE_PASSES",
                "ARTIFACT_HASH_VERIFIED",
                "CHECKPOINT_UNCHANGED",
                "ROLLBACK_POINTER_WRITTEN",
                "PRIVATE_EVAL_RC_REFRESH_VERIFIED",
                "PACKAGED_WINNER_REPRO_VERIFIED",
                "LONG_RUN_STABILITY_VERIFIED",
                "LOCAL_PRIVATE_RELEASE_READY",
                "PRODUCTION_DEPLOYMENT_NOT_CLAIMED",
                "GPT_LIKE_READINESS_NOT_CLAIMED",
            ],
            metrics,
        )
        return 0
    except GateError as exc:
        return fail(out, exc.verdict, exc.message, metrics)


if __name__ == "__main__":
    sys.exit(main())
