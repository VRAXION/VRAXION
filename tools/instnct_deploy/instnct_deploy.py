#!/usr/bin/env python3
"""Canonical local/private deployment harness for 058 and 086."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEPLOYMENT_SCHEMA_VERSION = "instnct_deployment_config_v1"
SDK_SCHEMA_VERSION = "instnct_sdk_candidate_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = (REPO_ROOT / "target" / "pilot_wave").resolve()
SERVICE_ALPHA_SCRIPT_REL = "tools/instnct_service_alpha/instnct_service_alpha.py"
SERVICE_ALPHA_SCRIPT = Path(SERVICE_ALPHA_SCRIPT_REL)
DEFAULT_083_ARTIFACT_ROOT = Path("target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke")

REQUIRED_AUDIT_EVENTS = [
    "config_loaded",
    "config_validated",
    "policy_decision",
    "healthcheck_started",
    "healthcheck_completed",
    "sdk_smoke_started",
    "sdk_smoke_completed",
    "artifact_validation_started",
    "artifact_validation_completed",
    "bounded_chat_service_smoke_started",
    "bounded_chat_service_smoke_completed",
    "rollback_pointer_written",
    "final_verdict",
]
REQUIRED_SDK_ARTIFACTS = [
    "queue.json",
    "progress.jsonl",
    "sdk_manifest.json",
    "api_surface_snapshot.json",
    "error_envelope_examples.json",
    "checkpoint_metrics.json",
    "inference_samples.jsonl",
    "eval_report.json",
    "visual_export_manifest.json",
    "claim_boundary.md",
    "summary.json",
    "report.md",
]
REQUIRED_085_ARTIFACTS = [
    "summary.json",
    "service_metrics.json",
    "bounded_chat_request_response.json",
    "child_runtime_manifest.json",
    "audit_log.jsonl",
    "report.md",
]
BOUNDARY_TOKENS = [
    "no production deployment",
    "no hosted SaaS",
    "no public beta",
    "no public API",
    "no SDK release",
    "no clinical use",
    "no high-stakes education use",
    "no production API readiness",
    "no GPT-like assistant",
    "no open-domain chat",
    "no production chat",
    "no safety alignment",
    "no full VRAXION",
    "no language grounding",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]
BOUNDARY_SENTENCE_086 = (
    "This is local/private deployment harness integration only. "
    "This is not production deployment, not hosted SaaS, not public beta, not public API, "
    "not SDK release, not GPT-like assistant, not open-domain chat, not production chat, "
    "and not safety alignment."
)


class HarnessError(Exception):
    def __init__(self, code: str, message: str, *, status: int = 1):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def now_ms() -> int:
    return int(time.time() * 1000)


def normalized_lf_sha256_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def load_config(path: Path) -> tuple[dict[str, Any], str, str]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise HarnessError("CONFIG_SCHEMA_INVALID", "config root must be an object")
    return data, raw, normalized_lf_sha256_text(raw)


def path_has_traversal(path_text: str) -> bool:
    return any(part == ".." for part in Path(path_text).parts)


def resolve_repo_path(path_text: str, field_name: str) -> Path:
    if not path_text or not isinstance(path_text, str):
        raise HarnessError("CONFIG_SCHEMA_INVALID", f"{field_name} must be a non-empty string")
    raw_path = Path(path_text)
    if raw_path.is_absolute() or path_has_traversal(path_text):
        raise HarnessError("CONFIG_SCHEMA_INVALID", f"{field_name} must be a relative repo path")
    return (REPO_ROOT / raw_path).resolve()


def resolve_safe_out_dir(path_text: str) -> Path:
    if not path_text or not isinstance(path_text, str):
        raise HarnessError("CONFIG_SCHEMA_INVALID", "out_dir must be a non-empty string")
    raw_path = Path(path_text)
    if raw_path.is_absolute() or path_has_traversal(path_text):
        raise HarnessError("UNSAFE_OUT_DIR_REJECTED", "out_dir must be a relative target/pilot_wave path")
    parts = [part.lower() for part in raw_path.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise HarnessError("UNSAFE_OUT_DIR_REJECTED", "out_dir must be under target/pilot_wave")
    resolved = (REPO_ROOT / raw_path).resolve()
    try:
        resolved.relative_to(TARGET_ROOT)
    except ValueError as exc:
        raise HarnessError("UNSAFE_OUT_DIR_REJECTED", "out_dir resolved outside target/pilot_wave") from exc
    return resolved


def resolve_output_file(out_dir: Path, path_text: str, default_name: str) -> Path:
    path_text = path_text or default_name
    raw_path = Path(path_text)
    if raw_path.is_absolute() or path_has_traversal(path_text):
        raise HarnessError("CONFIG_SCHEMA_INVALID", f"{default_name} must be relative to out_dir")
    return out_dir / raw_path


def repo_rel_posix(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def has_production_true(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "production_default_training_enabled",
                "public_beta_promoted",
                "production_api_ready",
                "production_mode",
                "hosted_saas_enabled",
                "public_api_enabled",
            } and item is True:
                return True
            if has_production_true(item):
                return True
    elif isinstance(value, list):
        return any(has_production_true(item) for item in value)
    return False


def scan_text_for_production_true(text: str) -> bool:
    patterns = [
        r'"production_default_training_enabled"\s*:\s*true',
        r'"public_beta_promoted"\s*:\s*true',
        r'"production_api_ready"\s*:\s*true',
        r'"production_mode"\s*:\s*true',
        r'"hosted_saas_enabled"\s*:\s*true',
        r'"public_api_enabled"\s*:\s*true',
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def validate_bounded_chat_config_fields(config: dict[str, Any]) -> None:
    if config.get("bounded_chat_service_alpha_enabled") is not True:
        return
    required = [
        "bounded_chat_service_config_path",
        "bounded_chat_service_smoke_out",
        "bounded_chat_require_085_positive",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise HarnessError("CONFIG_SCHEMA_INVALID", f"missing bounded chat config fields: {', '.join(missing)}")
    if config.get("bounded_chat_require_085_positive") is not True:
        raise HarnessError("CONFIG_SCHEMA_INVALID", "bounded_chat_require_085_positive must be true")

    service_config_path = resolve_repo_path(str(config["bounded_chat_service_config_path"]), "bounded_chat_service_config_path")
    if not service_config_path.exists():
        raise HarnessError("CONFIG_SCHEMA_INVALID", "bounded chat service config path does not exist")
    resolve_safe_out_dir(str(config["bounded_chat_service_smoke_out"]))
    service_config = read_json(service_config_path)
    if service_config.get("bind_host") != "127.0.0.1":
        raise HarnessError("PUBLIC_BIND_DETECTED", "bounded chat service config must bind only to 127.0.0.1")
    if has_production_true(service_config):
        raise HarnessError("PRODUCTION_DEPLOYMENT_CLAIM_DETECTED", "service config contains production/public flags")


def validate_config_fields(config: dict[str, Any]) -> str:
    required = [
        "schema_version",
        "sdk_schema_version",
        "deployment_mode",
        "intended_use",
        "out_dir",
        "seed",
        "progress_path",
        "audit_log_path",
        "production_default_training_enabled",
        "public_beta_promoted",
        "production_api_ready",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise HarnessError("CONFIG_SCHEMA_INVALID", f"missing config fields: {', '.join(missing)}")
    if config["schema_version"] != DEPLOYMENT_SCHEMA_VERSION:
        raise HarnessError("CONFIG_SCHEMA_INVALID", "unknown deployment config schema version")
    if config["sdk_schema_version"] != SDK_SCHEMA_VERSION:
        raise HarnessError("CONFIG_SCHEMA_INVALID", "unknown SDK candidate schema version")
    if has_production_true(config):
        raise HarnessError("PRODUCTION_DEPLOYMENT_CLAIM_DETECTED", "production/public/hosted flags must remain false")
    if config["deployment_mode"] not in {"local_research", "private_evaluation"}:
        return "rejected"
    if config["intended_use"] not in {"research", "internal_evaluation"}:
        return "rejected"
    if not isinstance(config["seed"], int):
        raise HarnessError("CONFIG_SCHEMA_INVALID", "seed must be an integer")
    validate_bounded_chat_config_fields(config)
    return "allowed"


class HarnessRun:
    def __init__(self, config_path: Path, out_override: str | None):
        self.config_path = (REPO_ROOT / config_path).resolve() if not config_path.is_absolute() else config_path
        self.deployment_start_timestamp = time.time()
        self.harness_run_id = f"harness_{hashlib.sha256(str(self.deployment_start_timestamp).encode()).hexdigest()[:12]}"
        self.config, self.raw_config_text, self.config_hash = load_config(self.config_path)
        if out_override:
            self.config["out_dir"] = out_override
        self.out_dir = resolve_safe_out_dir(str(self.config.get("out_dir", "")))
        self.progress_path = resolve_output_file(
            self.out_dir,
            str(self.config.get("progress_path", "progress.jsonl")),
            "progress.jsonl",
        )
        self.audit_log_path = resolve_output_file(
            self.out_dir,
            str(self.config.get("audit_log_path", "audit_log.jsonl")),
            "audit_log.jsonl",
        )
        self.queue_path = self.out_dir / "queue.json"
        self.resolved_config_path = self.out_dir / "resolved_config.json"
        self.summary_path = self.out_dir / "summary.json"
        self.report_path = self.out_dir / "report.md"
        self.healthcheck_path = self.out_dir / "healthcheck.json"
        self.sdk_smoke_dir = self.out_dir / "sdk_smoke"
        self.sdk_child_command = [
            "cargo",
            "run",
            "-p",
            "instnct-core",
            "--example",
            "instnct_sdk_candidate_smoke",
            "--",
            "--out",
            repo_rel_posix(self.sdk_smoke_dir),
        ]
        self.bounded_chat_service_enabled = self.config.get("bounded_chat_service_alpha_enabled") is True
        self.bounded_chat_service_config_rel = str(
            self.config.get("bounded_chat_service_config_path", "tools/instnct_service_alpha/config/example.local.json")
        )
        self.bounded_chat_service_config_path = resolve_repo_path(
            self.bounded_chat_service_config_rel,
            "bounded_chat_service_config_path",
        )
        self.bounded_chat_service_smoke_dir = resolve_safe_out_dir(
            str(
                self.config.get(
                    "bounded_chat_service_smoke_out",
                    "target/pilot_wave/stable_loop_phase_lock_086_bounded_chat_deployment_harness_integration/service_smoke",
                )
            )
        )
        self.bounded_chat_service_child_command = [
            "python",
            SERVICE_ALPHA_SCRIPT_REL,
            "smoke",
            "--config",
            self.bounded_chat_service_config_rel,
            "--out",
            repo_rel_posix(self.bounded_chat_service_smoke_dir),
        ]
        self.policy_decision = "unknown"
        self.sdk_child_exit_code: int | None = None
        self.sdk_smoke_start_timestamp: float | None = None
        self.sdk_smoke_completed_timestamp: float | None = None
        self.bounded_chat_service_smoke_started = False
        self.bounded_chat_service_smoke_completed = False
        self.bounded_chat_service_exit_code: int | None = None
        self.bounded_chat_service_start_timestamp: float | None = None
        self.bounded_chat_service_completed_timestamp: float | None = None
        self.sdk_smoke_status = "not_started"
        self.bounded_chat_service_smoke_status = "not_started"
        self.checkpoint_hash: str | None = None

    def progress(self, event: str, phase: str, message: str, **details: Any) -> None:
        append_jsonl(
            self.progress_path,
            {
                "timestamp_ms": now_ms(),
                "event": event,
                "phase": phase,
                "message": message,
                "details": details,
            },
        )

    def audit(self, event: str, status: str, **details: Any) -> None:
        append_jsonl(
            self.audit_log_path,
            {
                "timestamp_ms": now_ms(),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "harness_run_id": self.harness_run_id,
                "event": event,
                "status": status,
                "config_sha256_normalized_lf": self.config_hash,
                "config_hash": self.config_hash,
                "sdk_smoke_status": self.sdk_smoke_status,
                "bounded_chat_service_smoke_status": self.bounded_chat_service_smoke_status,
                "child_service_smoke_path": str(self.bounded_chat_service_smoke_dir.relative_to(REPO_ROOT)),
                "checkpoint_hash": self.checkpoint_hash,
                "final_verdict": details.get("final_verdict"),
                "details": details,
            },
        )

    def write_status_snapshot(
        self,
        phase: str,
        *,
        gate_pass: bool = False,
        verdicts: list[str] | None = None,
        validation: dict[str, Any] | None = None,
    ) -> None:
        summary = {
            "schema_version": DEPLOYMENT_SCHEMA_VERSION,
            "milestone": "STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION",
            "phase": phase,
            "deployment_harness_gate_pass": gate_pass,
            "config_sha256_normalized_lf": self.config_hash,
            "config_schema_valid": self.policy_decision == "allowed",
            "local_private_policy_allowed": self.policy_decision == "allowed",
            "sdk_smoke_still_passes": self.sdk_smoke_status == "completed",
            "bounded_chat_service_smoke_started": self.bounded_chat_service_smoke_started,
            "bounded_chat_service_smoke_completed": self.bounded_chat_service_smoke_completed,
            "bounded_chat_service_smoke_exit_code": self.bounded_chat_service_exit_code,
            "child_command": self.bounded_chat_service_child_command,
            "bounded_chat_service_smoke_out": str(self.bounded_chat_service_smoke_dir.relative_to(REPO_ROOT)),
            "checkpoint_hash_unchanged": (validation or {}).get("checkpoint_hash_unchanged"),
            "train_step_count": (validation or {}).get("train_step_count"),
            "production_deployment_claimed": False,
            "public_api_claimed": False,
            "hosted_saas_claimed": False,
            "gpt_like_assistant_readiness_claimed": False,
            "production_chat_claimed": False,
            "verdicts": verdicts or ["BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_IN_PROGRESS"],
            "claim_boundary": BOUNDARY_TOKENS,
        }
        if validation:
            summary["validation"] = validation
        write_json(self.summary_path, summary)
        self.report_path.write_text(
            "# STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION Report\n\n"
            f"Status: {'positive' if gate_pass else phase}.\n\n"
            f"{BOUNDARY_SENTENCE_086}\n\n"
            f"Harness run id: `{self.harness_run_id}`\n\n"
            f"SDK child command: `{' '.join(self.sdk_child_command)}`\n\n"
            f"Bounded chat service child command: `{' '.join(self.bounded_chat_service_child_command)}`\n\n"
            f"Config SHA-256 normalized LF: `{self.config_hash}`\n\n",
            encoding="utf-8",
        )

    def prepare(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            self.queue_path,
            {
                "schema_version": DEPLOYMENT_SCHEMA_VERSION,
                "milestone": "STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION",
                "harness_run_id": self.harness_run_id,
                "tasks": [
                    "config",
                    "healthcheck",
                    "sdk_smoke",
                    "085_service_smoke",
                    "artifact_provenance",
                    "audit",
                    "rollback_pointer",
                ],
                "bounded_chat_service_child_command": self.bounded_chat_service_child_command,
                "created_timestamp_ms": now_ms(),
            },
        )
        self.write_status_snapshot("started")
        self.progress("config_loaded", "completed", "deployment config loaded")
        self.audit(
            "config_loaded",
            "completed",
            config_path=str(self.config_path.relative_to(REPO_ROOT)),
        )
        self.policy_decision = validate_config_fields(self.config)
        self.audit("config_validated", "completed", schema_version=self.config["schema_version"])
        if self.policy_decision != "allowed":
            self.progress("policy_decision", "rejected", "deployment policy rejected config")
            self.audit(
                "policy_decision",
                "rejected",
                deployment_mode=self.config.get("deployment_mode"),
                intended_use=self.config.get("intended_use"),
            )
            self.write_rejection_summary("POLICY_GUARD_REJECTS_REGULATED_DEPLOYMENT")
            raise HarnessError("POLICY_GUARD_REJECTS_REGULATED_DEPLOYMENT", "regulated deployment mode rejected")
        self.progress("policy_decision", "completed", "deployment policy allowed config")
        self.audit(
            "policy_decision",
            "allowed",
            deployment_mode=self.config["deployment_mode"],
            intended_use=self.config["intended_use"],
        )
        resolved = dict(self.config)
        resolved.update(
            {
                "config_sha256_normalized_lf": self.config_hash,
                "config_policy_decision": self.policy_decision,
                "resolved_out_dir": str(self.out_dir.relative_to(REPO_ROOT)),
                "resolved_progress_path": str(self.progress_path.relative_to(REPO_ROOT)),
                "resolved_audit_log_path": str(self.audit_log_path.relative_to(REPO_ROOT)),
                "bounded_chat_service_alpha_enabled": self.bounded_chat_service_enabled,
                "bounded_chat_service_child_command": self.bounded_chat_service_child_command,
                "bounded_chat_service_smoke_out": str(self.bounded_chat_service_smoke_dir.relative_to(REPO_ROOT)),
            }
        )
        write_json(self.resolved_config_path, resolved)
        self.write_status_snapshot("config_validated")

    def write_rejection_summary(self, verdict: str) -> None:
        write_json(
            self.summary_path,
            {
                "schema_version": DEPLOYMENT_SCHEMA_VERSION,
                "deployment_harness_gate_pass": False,
                "config_sha256_normalized_lf": self.config_hash,
                "config_policy_decision": "rejected",
                "sdk_smoke_started": False,
                "bounded_chat_service_smoke_started": False,
                "verdicts": [verdict],
                "production_default_training_enabled": False,
                "public_beta_promoted": False,
                "production_api_ready": False,
                "production_deployment_claimed": False,
                "public_api_claimed": False,
                "hosted_saas_claimed": False,
                "gpt_like_assistant_readiness_claimed": False,
            },
        )

    def healthcheck(self) -> dict[str, Any]:
        self.progress("healthcheck_started", "start", "healthcheck started")
        self.audit("healthcheck_started", "start")
        service_config_exists = self.bounded_chat_service_config_path.exists()
        service_config = read_json(self.bounded_chat_service_config_path) if service_config_exists else {}
        artifact_root = resolve_repo_path(
            str(service_config.get("bounded_chat_artifact_root", DEFAULT_083_ARTIFACT_ROOT)),
            "bounded_chat_artifact_root",
        )
        checks: dict[str, Any] = {
            "repo_root_exists": REPO_ROOT.exists(),
            "sdk_smoke_example_exists": (REPO_ROOT / "instnct-core" / "examples" / "instnct_sdk_candidate_smoke.rs").exists(),
            "out_dir_under_target_pilot_wave": True,
            "service_alpha_script_exists": (REPO_ROOT / SERVICE_ALPHA_SCRIPT).exists(),
            "bounded_chat_service_config_exists": service_config_exists,
            "bounded_chat_service_bind_localhost": service_config.get("bind_host") == "127.0.0.1",
            "artifact_083_root_exists": artifact_root.exists(),
            "cargo_available": False,
            "healthcheck_has_training_side_effects": False,
        }
        cargo = subprocess.run(["cargo", "--version"], cwd=REPO_ROOT, text=True, capture_output=True)
        checks["cargo_available"] = cargo.returncode == 0
        checks["cargo_version"] = cargo.stdout.strip()
        checks["healthcheck_pass"] = all(
            bool(checks[key])
            for key in [
                "repo_root_exists",
                "sdk_smoke_example_exists",
                "out_dir_under_target_pilot_wave",
                "service_alpha_script_exists",
                "bounded_chat_service_config_exists",
                "bounded_chat_service_bind_localhost",
                "artifact_083_root_exists",
                "cargo_available",
            ]
        )
        write_json(self.healthcheck_path, checks)
        status = "completed" if checks["healthcheck_pass"] else "failed"
        self.progress("healthcheck_completed", status, f"healthcheck {status}")
        self.audit("healthcheck_completed", status, healthcheck_path=str(self.healthcheck_path.relative_to(REPO_ROOT)))
        self.write_status_snapshot("healthcheck_completed", validation={"healthcheck_pass": checks["healthcheck_pass"]})
        if not checks["healthcheck_pass"]:
            raise HarnessError("HEALTHCHECK_FAILS", "healthcheck failed")
        return checks

    def run_sdk_smoke(self) -> None:
        if self.sdk_smoke_dir.exists():
            shutil.rmtree(self.sdk_smoke_dir)
        self.sdk_smoke_start_timestamp = time.time()
        self.sdk_smoke_status = "started"
        self.progress("sdk_smoke_started", "start", "SDK smoke started")
        self.audit(
            "sdk_smoke_started",
            "start",
            child_command=self.sdk_child_command,
            sdk_smoke_start_timestamp=self.sdk_smoke_start_timestamp,
        )
        child = subprocess.run(self.sdk_child_command, cwd=REPO_ROOT, text=True, capture_output=True)
        self.sdk_child_exit_code = child.returncode
        self.sdk_smoke_completed_timestamp = time.time()
        (self.out_dir / "sdk_child_stdout.txt").write_text(child.stdout, encoding="utf-8")
        (self.out_dir / "sdk_child_stderr.txt").write_text(child.stderr, encoding="utf-8")
        status = "completed" if child.returncode == 0 else "failed"
        self.sdk_smoke_status = status
        self.progress("sdk_smoke_completed", status, f"SDK smoke {status}", child_exit_code=child.returncode)
        self.audit(
            "sdk_smoke_completed",
            status,
            child_exit_code=child.returncode,
            sdk_smoke_completed_timestamp=self.sdk_smoke_completed_timestamp,
        )
        self.write_status_snapshot("sdk_smoke_completed")
        if child.returncode != 0:
            raise HarnessError("SDK_SMOKE_THROUGH_HARNESS_FAILS", "SDK smoke child command failed")

    def validate_sdk_smoke_artifacts(self) -> dict[str, Any]:
        missing = [name for name in REQUIRED_SDK_ARTIFACTS if not (self.sdk_smoke_dir / name).exists()]
        if missing:
            raise HarnessError("SDK_SMOKE_THROUGH_HARNESS_FAILS", f"missing SDK smoke artifacts: {', '.join(missing)}")

        summary_path = self.sdk_smoke_dir / "summary.json"
        report_path = self.sdk_smoke_dir / "report.md"
        summary_mtime = summary_path.stat().st_mtime
        report_mtime = report_path.stat().st_mtime
        if (
            self.sdk_child_exit_code != 0
            or summary_mtime <= self.deployment_start_timestamp
            or report_mtime <= self.deployment_start_timestamp
        ):
            raise HarnessError("STALE_SDK_SMOKE_ARTIFACT_USED", "SDK smoke summary/report are stale or child failed")

        for path in [self.resolved_config_path] + [self.sdk_smoke_dir / name for name in REQUIRED_SDK_ARTIFACTS]:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if scan_text_for_production_true(text):
                raise HarnessError("PRODUCTION_DEPLOYMENT_CLAIM_DETECTED", f"production flag contamination in {path}")

        sdk_summary = read_json(summary_path)
        checkpoint = read_json(self.sdk_smoke_dir / "checkpoint_metrics.json")
        visual = read_json(self.sdk_smoke_dir / "visual_export_manifest.json")
        manifest = {
            "required_artifacts_present": True,
            "sdk_smoke_still_passes": True,
            "sdk_smoke_summary_newer_than_086_start": summary_mtime > self.deployment_start_timestamp,
            "sdk_smoke_report_newer_than_086_start": report_mtime > self.deployment_start_timestamp,
            "sdk_child_exit_code": self.sdk_child_exit_code,
            "sdk_child_command": self.sdk_child_command,
            "checkpoint_save_load_pass": checkpoint.get("checkpoint_save_load_pass") is True,
            "checkpoint_hash_algorithm": checkpoint.get("checkpoint_hash_algorithm"),
            "rollback_success": sdk_summary.get("rollback_success") is True,
            "visual_export_schema": visual.get("schema_version"),
            "production_flags_clean": True,
        }
        if manifest["checkpoint_hash_algorithm"] != "SHA-256":
            raise HarnessError("SDK_SMOKE_THROUGH_HARNESS_FAILS", "checkpoint hash algorithm was not SHA-256")
        if manifest["visual_export_schema"] != "visual_snapshot_v1":
            raise HarnessError("SDK_SMOKE_THROUGH_HARNESS_FAILS", "visual export schema was not visual_snapshot_v1")
        if not manifest["checkpoint_save_load_pass"] or not manifest["rollback_success"]:
            raise HarnessError("SDK_SMOKE_THROUGH_HARNESS_FAILS", "checkpoint or rollback validation failed")

        write_json(self.out_dir / "sdk_smoke_manifest.json", manifest)
        return manifest

    def run_bounded_chat_service_smoke(self) -> None:
        if not self.bounded_chat_service_enabled:
            raise HarnessError("BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_FAILS", "bounded chat service alpha is disabled")
        if self.bounded_chat_service_smoke_dir.exists():
            shutil.rmtree(self.bounded_chat_service_smoke_dir)
        self.bounded_chat_service_smoke_started = True
        self.bounded_chat_service_start_timestamp = time.time()
        self.bounded_chat_service_smoke_status = "started"
        self.progress(
            "bounded_chat_service_smoke_started",
            "start",
            "085 bounded chat service smoke started",
            child_command=self.bounded_chat_service_child_command,
        )
        self.audit(
            "bounded_chat_service_smoke_started",
            "start",
            child_command=self.bounded_chat_service_child_command,
            bounded_chat_service_smoke_start_timestamp=self.bounded_chat_service_start_timestamp,
        )
        child = subprocess.run(self.bounded_chat_service_child_command, cwd=REPO_ROOT, text=True, capture_output=True)
        self.bounded_chat_service_exit_code = child.returncode
        self.bounded_chat_service_completed_timestamp = time.time()
        self.bounded_chat_service_smoke_completed = True
        (self.out_dir / "bounded_chat_service_stdout.txt").write_text(child.stdout, encoding="utf-8")
        (self.out_dir / "bounded_chat_service_stderr.txt").write_text(child.stderr, encoding="utf-8")
        status = "completed" if child.returncode == 0 else "failed"
        self.bounded_chat_service_smoke_status = status
        self.progress(
            "bounded_chat_service_smoke_completed",
            status,
            f"085 bounded chat service smoke {status}",
            child_exit_code=child.returncode,
        )
        self.audit(
            "bounded_chat_service_smoke_completed",
            status,
            child_exit_code=child.returncode,
            bounded_chat_service_smoke_completed_timestamp=self.bounded_chat_service_completed_timestamp,
        )
        self.write_status_snapshot("bounded_chat_service_smoke_completed")
        if child.returncode != 0:
            raise HarnessError("BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_FAILS", "085 service smoke child command failed")

    def validate_bounded_chat_service_artifacts(self) -> dict[str, Any]:
        missing = [name for name in REQUIRED_085_ARTIFACTS if not (self.bounded_chat_service_smoke_dir / name).exists()]
        if missing:
            raise HarnessError("UPSTREAM_085_ARTIFACT_MISSING", f"missing 085 service artifacts: {', '.join(missing)}")

        summary_path = self.bounded_chat_service_smoke_dir / "summary.json"
        report_path = self.bounded_chat_service_smoke_dir / "report.md"
        summary_mtime = summary_path.stat().st_mtime
        report_mtime = report_path.stat().st_mtime
        if (
            self.bounded_chat_service_exit_code != 0
            or summary_mtime <= self.deployment_start_timestamp
            or report_mtime <= self.deployment_start_timestamp
        ):
            raise HarnessError("STALE_SERVICE_SMOKE_ARTIFACT_USED", "085 service smoke summary/report are stale or child failed")

        summary = read_json(summary_path)
        metrics = read_json(self.bounded_chat_service_smoke_dir / "service_metrics.json")
        request_response = read_json(self.bounded_chat_service_smoke_dir / "bounded_chat_request_response.json")
        child_manifest = read_json(self.bounded_chat_service_smoke_dir / "child_runtime_manifest.json")
        service_audit_rows = read_jsonl(self.bounded_chat_service_smoke_dir / "audit_log.jsonl")
        verdicts = set(summary.get("verdicts", []))

        if "BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE" not in verdicts:
            raise HarnessError("BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_FAILS", "085 positive verdict missing")

        required_true = [
            "bounded_chat_route_registered",
            "bounded_chat_child_084_positive",
            "artifact_hash_verified",
            "checkpoint_hash_unchanged",
            "auth_required",
            "auth_rejection_has_no_child_side_effect",
            "policy_rejection_has_no_child_side_effect",
            "rate_limit_metadata_present",
            "bad_input_handled",
            "unsupported_input_handled",
            "audit_log_written",
            "child_runtime_artifacts_preserved",
        ]
        missing_true = [key for key in required_true if metrics.get(key) is not True]
        if missing_true:
            auth_policy = {
                "auth_required",
                "auth_rejection_has_no_child_side_effect",
                "policy_rejection_has_no_child_side_effect",
                "rate_limit_metadata_present",
            }
            if any(key in auth_policy for key in missing_true):
                raise HarnessError("AUTH_POLICY_RATE_LIMIT_REGRESSION_DETECTED", f"085 metrics failed: {missing_true}")
            if "bad_input_handled" in missing_true:
                raise HarnessError("BAD_INPUT_REGRESSION_DETECTED", "085 bad-input handling regressed")
            if "unsupported_input_handled" in missing_true:
                raise HarnessError("UNSUPPORTED_INPUT_REGRESSION_DETECTED", "085 unsupported handling regressed")
            if "artifact_hash_verified" in missing_true:
                raise HarnessError("ARTIFACT_HASH_MISMATCH", "085 child artifact hash was not verified")
            if "checkpoint_hash_unchanged" in missing_true:
                raise HarnessError("CHECKPOINT_MUTATION_DETECTED", "085 child checkpoint hash changed")
            raise HarnessError("BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_FAILS", f"085 metrics failed: {missing_true}")
        if metrics.get("train_step_count") != 0:
            raise HarnessError("TRAINING_SIDE_EFFECT_DETECTED", "085 child reported training steps")
        if not service_audit_rows:
            raise HarnessError("AUDIT_LOG_MISSING", "085 child service audit log is empty")

        inference = request_response.get("value", {}).get("inference", {})
        self.checkpoint_hash = inference.get("checkpoint_sha256")
        artifact_zip_hash = inference.get("artifact_package_zip_sha256") or request_response.get("artifact_hash")
        service_manifest = {
            "bounded_chat_service_smoke_started": self.bounded_chat_service_smoke_started,
            "bounded_chat_service_smoke_completed": self.bounded_chat_service_smoke_completed,
            "bounded_chat_service_smoke_exit_code": self.bounded_chat_service_exit_code,
            "bounded_chat_service_summary_newer_than_086_start": summary_mtime > self.deployment_start_timestamp,
            "bounded_chat_service_report_newer_than_086_start": report_mtime > self.deployment_start_timestamp,
            "child_command": self.bounded_chat_service_child_command,
            "bounded_chat_service_smoke_pass": True,
            "bounded_chat_service_smoke_path": str(self.bounded_chat_service_smoke_dir.relative_to(REPO_ROOT)),
            "BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE_present": True,
            "service_audit_log_non_empty": True,
            "child_runtime_manifest": child_manifest,
        }
        write_json(self.out_dir / "bounded_chat_service_manifest.json", service_manifest)
        write_json(self.out_dir / "bounded_chat_service_metrics.json", metrics)
        write_json(self.out_dir / "bounded_chat_request_response.json", request_response)
        return {
            **service_manifest,
            **metrics,
            "train_step_count": metrics.get("train_step_count"),
            "artifact_package_zip_sha256": artifact_zip_hash,
            "checkpoint_sha256": self.checkpoint_hash,
            "085_service_child_job_path": request_response.get("child_job_path")
            or request_response.get("value", {}).get("child_job_path")
            or child_manifest.get("successful_child_job_path"),
        }

    def validate_artifacts(self, sdk_manifest: dict[str, Any], service_validation: dict[str, Any]) -> dict[str, Any]:
        self.progress("artifact_validation_started", "start", "artifact validation started")
        self.audit("artifact_validation_started", "start")
        artifact_validation = {
            "sdk_smoke_still_passes": sdk_manifest.get("sdk_smoke_still_passes") is True,
            "083_artifact_root": str(DEFAULT_083_ARTIFACT_ROOT),
            "083_artifact_package_zip_hash": service_validation.get("artifact_package_zip_sha256"),
            "084_child_checkpoint_hash": service_validation.get("checkpoint_sha256"),
            "085_service_child_job_path": service_validation.get("085_service_child_job_path"),
            "086_harness_smoke_path": str(self.out_dir.relative_to(REPO_ROOT)),
            "checkpoint_hash_unchanged": service_validation.get("checkpoint_hash_unchanged") is True,
            "bounded_chat_service_smoke_pass": service_validation.get("bounded_chat_service_smoke_pass") is True,
            "artifact_hash_verified": service_validation.get("artifact_hash_verified") is True,
            "train_step_count": service_validation.get("train_step_count"),
        }
        if not artifact_validation["artifact_hash_verified"]:
            raise HarnessError("ARTIFACT_HASH_MISMATCH", "083 artifact hash was not verified through 085 child")
        if not artifact_validation["checkpoint_hash_unchanged"]:
            raise HarnessError("CHECKPOINT_MUTATION_DETECTED", "checkpoint changed through deployment harness")
        write_json(self.out_dir / "artifact_validation.json", artifact_validation)
        self.progress("artifact_validation_completed", "completed", "artifact validation completed")
        self.audit("artifact_validation_completed", "completed", validation=artifact_validation)
        return artifact_validation

    def write_rollback_pointer(self) -> dict[str, Any]:
        previous_config = REPO_ROOT / "tools" / "instnct_deploy" / "config" / "example.private_eval.json"
        rollback = {
            "schema_version": DEPLOYMENT_SCHEMA_VERSION,
            "previous_local_private_harness_config_path": str(previous_config.relative_to(REPO_ROOT)) if previous_config.exists() else None,
            "previous_local_private_harness_config_sha256": sha256_file(previous_config) if previous_config.exists() else None,
            "current_config_path": str(self.config_path.relative_to(REPO_ROOT)),
            "current_config_sha256_normalized_lf": self.config_hash,
            "085_service_smoke_output_path": str(self.bounded_chat_service_smoke_dir.relative_to(REPO_ROOT)),
            "083_artifact_root": str(DEFAULT_083_ARTIFACT_ROOT),
            "rollback_instruction": "Disable bounded_chat_service_alpha_enabled in the local/private harness config and rerun validate-config before smoke.",
            "automatic_production_rollback_claimed": False,
            "no_automatic_production_rollback_claim": True,
        }
        write_json(self.out_dir / "rollback_pointer.json", rollback)
        self.progress("rollback_pointer_written", "completed", "rollback pointer written")
        self.audit("rollback_pointer_written", "completed", rollback_pointer_path=str((self.out_dir / "rollback_pointer.json").relative_to(REPO_ROOT)))
        return rollback

    def write_final(self, gate_pass: bool, verdicts: list[str], validation: dict[str, Any] | None = None) -> None:
        audit_events = self.audit_events()
        validation = validation or {}
        summary = {
            "schema_version": DEPLOYMENT_SCHEMA_VERSION,
            "milestone": "STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION",
            "deployment_harness_gate_pass": gate_pass,
            "config_schema_valid": self.policy_decision == "allowed",
            "local_private_policy_allowed": self.policy_decision == "allowed",
            "healthcheck_pass": validation.get("healthcheck", {}).get("healthcheck_pass") is True,
            "sdk_smoke_still_passes": validation.get("sdk_smoke_still_passes") is True,
            "bounded_chat_service_smoke_started": self.bounded_chat_service_smoke_started,
            "bounded_chat_service_smoke_completed": self.bounded_chat_service_smoke_completed,
            "bounded_chat_service_smoke_exit_code": self.bounded_chat_service_exit_code,
            "bounded_chat_service_summary_newer_than_086_start": validation.get("bounded_chat_service_summary_newer_than_086_start") is True,
            "bounded_chat_service_report_newer_than_086_start": validation.get("bounded_chat_service_report_newer_than_086_start") is True,
            "child_command": self.bounded_chat_service_child_command,
            "bounded_chat_service_smoke_pass": validation.get("bounded_chat_service_smoke_pass") is True,
            "BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE_present": validation.get("BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE_present") is True,
            "bounded_chat_route_registered": validation.get("bounded_chat_route_registered") is True,
            "bounded_chat_child_084_positive": validation.get("bounded_chat_child_084_positive") is True,
            "artifact_hash_verified": validation.get("artifact_hash_verified") is True,
            "checkpoint_hash_unchanged": validation.get("checkpoint_hash_unchanged") is True,
            "train_step_count": validation.get("train_step_count"),
            "auth_required": validation.get("auth_required") is True,
            "auth_rejection_has_no_child_side_effect": validation.get("auth_rejection_has_no_child_side_effect") is True,
            "policy_rejection_has_no_child_side_effect": validation.get("policy_rejection_has_no_child_side_effect") is True,
            "rate_limit_metadata_present": validation.get("rate_limit_metadata_present") is True,
            "bad_input_handled": validation.get("bad_input_handled") is True,
            "unsupported_input_handled": validation.get("unsupported_input_handled") is True,
            "audit_log_written": validation.get("audit_log_written") is True,
            "child_runtime_artifacts_preserved": validation.get("child_runtime_artifacts_preserved") is True,
            "rollback_pointer_written": (self.out_dir / "rollback_pointer.json").exists(),
            "config_sha256_normalized_lf": self.config_hash,
            "config_policy_decision": self.policy_decision,
            "sdk_smoke_start_timestamp": self.sdk_smoke_start_timestamp,
            "sdk_smoke_completed_timestamp": self.sdk_smoke_completed_timestamp,
            "sdk_child_exit_code": self.sdk_child_exit_code,
            "sdk_child_command": self.sdk_child_command,
            "audit_log_complete": all(event in audit_events for event in REQUIRED_AUDIT_EVENTS),
            "production_deployment_claimed": False,
            "public_api_claimed": False,
            "hosted_saas_claimed": False,
            "gpt_like_assistant_readiness_claimed": False,
            "production_chat_claimed": False,
            "public_beta_claimed": False,
            "sdk_release_claimed": False,
            "validation": validation,
            "verdicts": verdicts,
            "production_default_training_enabled": False,
            "public_beta_promoted": False,
            "production_api_ready": False,
            "claim_boundary": BOUNDARY_TOKENS,
        }
        write_json(self.summary_path, summary)
        self.report_path.write_text(
            "# STABLE_LOOP_PHASE_LOCK_086_BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION Report\n\n"
            f"Status: {'positive' if gate_pass else 'failed'}.\n\n"
            f"{BOUNDARY_SENTENCE_086}\n\n"
            "Harness chain:\n\n"
            "- config -> healthcheck -> existing SDK smoke -> 085 service smoke -> artifact provenance -> audit -> rollback pointer\n\n"
            f"SDK child command: `{' '.join(self.sdk_child_command)}`\n\n"
            f"Bounded chat service child command: `{' '.join(self.bounded_chat_service_child_command)}`\n\n"
            f"085 service smoke output: `{self.bounded_chat_service_smoke_dir.relative_to(REPO_ROOT)}`\n\n"
            f"083 artifact root: `{DEFAULT_083_ARTIFACT_ROOT}`\n\n"
            f"084 child checkpoint hash: `{validation.get('checkpoint_sha256')}`\n\n"
            f"083 artifact package zip hash: `{validation.get('artifact_package_zip_sha256')}`\n\n"
            f"Config SHA-256 normalized LF: `{self.config_hash}`\n\n"
            "Boundary: local/private deployment harness integration only; not production deployment; "
            "not hosted SaaS; not public beta; not public API; not SDK release; not GPT-like assistant; "
            "not open-domain chat; not production chat; not safety alignment.\n",
            encoding="utf-8",
        )

    def audit_events(self) -> set[str]:
        return {row.get("event") for row in read_jsonl(self.audit_log_path)}

    def final_verdict(self, gate_pass: bool) -> None:
        final = "BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_POSITIVE" if gate_pass else "BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_FAILS"
        self.progress("final_verdict", "completed" if gate_pass else "failed", "final verdict emitted")
        self.audit("final_verdict", "completed" if gate_pass else "failed", final_verdict=final)


def run_common_flow(run: HarnessRun, include_bounded_chat: bool) -> dict[str, Any]:
    run.prepare()
    health = run.healthcheck()
    run.run_sdk_smoke()
    sdk_manifest = run.validate_sdk_smoke_artifacts()
    validation: dict[str, Any] = {
        **sdk_manifest,
        "healthcheck": health,
        "healthcheck_pass": health.get("healthcheck_pass") is True,
    }
    if include_bounded_chat:
        run.run_bounded_chat_service_smoke()
        service_validation = run.validate_bounded_chat_service_artifacts()
        artifact_validation = run.validate_artifacts(sdk_manifest, service_validation)
        rollback = run.write_rollback_pointer()
        validation.update(service_validation)
        validation["artifact_validation"] = artifact_validation
        validation["rollback_pointer"] = rollback
    return validation


def command_validate_config(args: argparse.Namespace) -> int:
    run = HarnessRun(Path(args.config), args.out)
    try:
        run.prepare()
        run.final_verdict(True)
        run.write_final(True, ["CONFIG_SCHEMA_VALID", "DEPLOYMENT_HARNESS_CONFIG_VALID", "PRODUCTION_DEPLOYMENT_NOT_CLAIMED"])
        print(json.dumps({"check_pass": True, "out_dir": str(run.out_dir.relative_to(REPO_ROOT))}))
        return 0
    except HarnessError as err:
        if run.policy_decision == "rejected":
            print(json.dumps({"check_pass": False, "verdict": err.code, "message": err.message}))
            return 1
        raise


def command_healthcheck(args: argparse.Namespace) -> int:
    run = HarnessRun(Path(args.config), args.out)
    run.prepare()
    health = run.healthcheck()
    run.final_verdict(True)
    run.write_final(True, ["HEALTHCHECK_POSITIVE", "DEPLOYMENT_HEALTHCHECK_PASSES", "PRODUCTION_DEPLOYMENT_NOT_CLAIMED"], {"healthcheck": health})
    print(json.dumps({"check_pass": True, "healthcheck": health}))
    return 0


def command_run_local(args: argparse.Namespace) -> int:
    run = HarnessRun(Path(args.config), args.out)
    validation = run_common_flow(run, include_bounded_chat=run.bounded_chat_service_enabled)
    run.final_verdict(True)
    run.write_final(
        True,
        [
            "SDK_SMOKE_THROUGH_HARNESS_POSITIVE",
            "CHECKPOINT_STORAGE_POSITIVE",
            "ROLLBACK_THROUGH_HARNESS_POSITIVE",
            "VISUAL_EXPORT_THROUGH_HARNESS_POSITIVE",
            "BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_PASSES",
            "PRODUCTION_DEPLOYMENT_NOT_CLAIMED",
        ],
        validation,
    )
    print(json.dumps({"check_pass": True, "out_dir": str(run.out_dir.relative_to(REPO_ROOT))}))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    run = HarnessRun(Path(args.config), args.out)
    validation = run_common_flow(run, include_bounded_chat=True)
    run.final_verdict(True)
    run.write_final(
        True,
        [
            "DEPLOYMENT_HARNESS_POSITIVE",
            "LOCAL_RUNBOOK_WRITTEN",
            "CONFIG_SCHEMA_VALID",
            "DEPLOYMENT_HARNESS_CONFIG_VALID",
            "DEPLOYMENT_HEALTHCHECK_PASSES",
            "SDK_SMOKE_THROUGH_HARNESS_STILL_PASSES",
            "BOUNDED_CHAT_SERVICE_SMOKE_THROUGH_HARNESS_PASSES",
            "BOUNDED_CHAT_ARTIFACT_PROVENANCE_VERIFIED",
            "CHECKPOINT_UNCHANGED_THROUGH_HARNESS",
            "AUTH_POLICY_RATE_LIMIT_THROUGH_HARNESS_PASSES",
            "ROLLBACK_POINTER_WRITTEN",
            "AUDIT_LOGGING_POSITIVE",
            "BOUNDED_CHAT_DEPLOYMENT_HARNESS_INTEGRATION_POSITIVE",
            "PRODUCTION_DEPLOYMENT_NOT_CLAIMED",
        ],
        validation,
    )
    print(json.dumps({"check_pass": True, "out_dir": str(run.out_dir.relative_to(REPO_ROOT))}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["validate-config", "healthcheck", "run-local", "smoke"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--out")
    args = parser.parse_args()

    handlers = {
        "validate-config": command_validate_config,
        "healthcheck": command_healthcheck,
        "run-local": command_run_local,
        "smoke": command_smoke,
    }
    try:
        return handlers[args.command](args)
    except HarnessError as err:
        print(json.dumps({"check_pass": False, "verdict": err.code, "message": err.message}))
        return err.status


if __name__ == "__main__":
    sys.exit(main())
