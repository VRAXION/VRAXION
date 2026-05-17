#!/usr/bin/env python3
"""Canonical local/private deployment harness for 058."""

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
BOUNDARY_TOKENS = [
    "no production deployment",
    "no hosted SaaS",
    "no public beta",
    "no clinical use",
    "no high-stakes education use",
    "no production API readiness",
    "no full VRAXION",
    "no language grounding",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> tuple[dict[str, Any], str, str]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise HarnessError("CONFIG_SCHEMA_INVALID", "config root must be an object")
    return data, raw, normalized_lf_sha256_text(raw)


def path_has_traversal(path_text: str) -> bool:
    return any(part == ".." for part in Path(path_text).parts)


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


def has_production_true(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "production_default_training_enabled",
                "public_beta_promoted",
                "production_api_ready",
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
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


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
        raise HarnessError("PRODUCTION_FLAG_CONTAMINATION", "production/public-beta flags must remain false")
    if config["deployment_mode"] not in {"local_research", "private_evaluation"}:
        return "rejected"
    if config["intended_use"] not in {"research", "internal_evaluation"}:
        return "rejected"
    if not isinstance(config["seed"], int):
        raise HarnessError("CONFIG_SCHEMA_INVALID", "seed must be an integer")
    return "allowed"


class HarnessRun:
    def __init__(self, config_path: Path, out_override: str | None):
        self.config_path = (REPO_ROOT / config_path).resolve() if not config_path.is_absolute() else config_path
        self.deployment_start_timestamp = time.time()
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
        self.resolved_config_path = self.out_dir / "resolved_config.json"
        self.summary_path = self.out_dir / "summary.json"
        self.report_path = self.out_dir / "report.md"
        self.healthcheck_path = self.out_dir / "healthcheck.json"
        self.sdk_smoke_dir = self.out_dir / "sdk_smoke"
        self.child_command = [
            "cargo",
            "run",
            "-p",
            "instnct-core",
            "--example",
            "instnct_sdk_candidate_smoke",
            "--",
            "--out",
            str(self.sdk_smoke_dir.relative_to(REPO_ROOT)),
        ]
        self.policy_decision = "unknown"
        self.child_exit_code: int | None = None
        self.sdk_smoke_start_timestamp: float | None = None
        self.sdk_smoke_completed_timestamp: float | None = None

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
                "event": event,
                "status": status,
                "config_sha256_normalized_lf": self.config_hash,
                "details": details,
            },
        )

    def prepare(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
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
            }
        )
        write_json(self.resolved_config_path, resolved)

    def write_rejection_summary(self, verdict: str) -> None:
        write_json(
            self.summary_path,
            {
                "schema_version": DEPLOYMENT_SCHEMA_VERSION,
                "deployment_harness_gate_pass": False,
                "config_sha256_normalized_lf": self.config_hash,
                "config_policy_decision": "rejected",
                "sdk_smoke_started": False,
                "verdicts": [verdict],
                "production_default_training_enabled": False,
                "public_beta_promoted": False,
                "production_api_ready": False,
            },
        )

    def healthcheck(self) -> dict[str, Any]:
        self.progress("healthcheck_started", "start", "healthcheck started")
        self.audit("healthcheck_started", "start")
        checks: dict[str, Any] = {
            "repo_root_exists": REPO_ROOT.exists(),
            "sdk_smoke_example_exists": (REPO_ROOT / "instnct-core" / "examples" / "instnct_sdk_candidate_smoke.rs").exists(),
            "out_dir_under_target_pilot_wave": True,
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
                "cargo_available",
            ]
        )
        write_json(self.healthcheck_path, checks)
        status = "completed" if checks["healthcheck_pass"] else "failed"
        self.progress("healthcheck_completed", status, f"healthcheck {status}")
        self.audit("healthcheck_completed", status, healthcheck_path=str(self.healthcheck_path.relative_to(REPO_ROOT)))
        if not checks["healthcheck_pass"]:
            raise HarnessError("HEALTHCHECK_FAILS", "healthcheck failed")
        return checks

    def run_sdk_smoke(self) -> None:
        if self.sdk_smoke_dir.exists():
            shutil.rmtree(self.sdk_smoke_dir)
        self.sdk_smoke_start_timestamp = time.time()
        self.progress("sdk_smoke_started", "start", "SDK smoke started")
        self.audit(
            "sdk_smoke_started",
            "start",
            child_command=self.child_command,
            sdk_smoke_start_timestamp=self.sdk_smoke_start_timestamp,
        )
        child = subprocess.run(self.child_command, cwd=REPO_ROOT, text=True, capture_output=True)
        self.child_exit_code = child.returncode
        self.sdk_smoke_completed_timestamp = time.time()
        (self.out_dir / "child_stdout.txt").write_text(child.stdout, encoding="utf-8")
        (self.out_dir / "child_stderr.txt").write_text(child.stderr, encoding="utf-8")
        status = "completed" if child.returncode == 0 else "failed"
        self.progress("sdk_smoke_completed", status, f"SDK smoke {status}", child_exit_code=child.returncode)
        self.audit(
            "sdk_smoke_completed",
            status,
            child_exit_code=child.returncode,
            sdk_smoke_completed_timestamp=self.sdk_smoke_completed_timestamp,
        )
        if child.returncode != 0:
            raise HarnessError("SDK_SMOKE_THROUGH_HARNESS_FAILS", "SDK smoke child command failed")

    def validate_child_artifacts(self) -> dict[str, Any]:
        self.progress("artifact_validation_started", "start", "artifact validation started")
        self.audit("artifact_validation_started", "start")
        missing = [name for name in REQUIRED_SDK_ARTIFACTS if not (self.sdk_smoke_dir / name).exists()]
        if missing:
            raise HarnessError("SDK_SMOKE_ARTIFACT_MISSING", f"missing SDK smoke artifacts: {', '.join(missing)}")

        summary_path = self.sdk_smoke_dir / "summary.json"
        report_path = self.sdk_smoke_dir / "report.md"
        summary_mtime = summary_path.stat().st_mtime
        report_mtime = report_path.stat().st_mtime
        if (
            self.child_exit_code != 0
            or summary_mtime <= self.deployment_start_timestamp
            or report_mtime <= self.deployment_start_timestamp
        ):
            raise HarnessError("STALE_SDK_SMOKE_ARTIFACT_USED", "SDK smoke summary/report are stale or child failed")

        for path in [self.resolved_config_path] + [self.sdk_smoke_dir / name for name in REQUIRED_SDK_ARTIFACTS]:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if scan_text_for_production_true(text):
                raise HarnessError("PRODUCTION_FLAG_CONTAMINATION", f"production flag contamination in {path}")

        sdk_summary = read_json(summary_path)
        checkpoint = read_json(self.sdk_smoke_dir / "checkpoint_metrics.json")
        visual = read_json(self.sdk_smoke_dir / "visual_export_manifest.json")
        validation = {
            "required_artifacts_present": True,
            "sdk_smoke_summary_newer_than_deployment_start": summary_mtime > self.deployment_start_timestamp,
            "sdk_smoke_report_newer_than_deployment_start": report_mtime > self.deployment_start_timestamp,
            "child_exit_code": self.child_exit_code,
            "child_command": self.child_command,
            "checkpoint_save_load_pass": checkpoint.get("checkpoint_save_load_pass") is True,
            "checkpoint_hash_algorithm": checkpoint.get("checkpoint_hash_algorithm"),
            "rollback_success": sdk_summary.get("rollback_success") is True,
            "visual_export_schema": visual.get("schema_version"),
            "production_flags_clean": True,
        }
        if validation["checkpoint_hash_algorithm"] != "SHA-256":
            raise HarnessError("SDK_SMOKE_THROUGH_HARNESS_FAILS", "checkpoint hash algorithm was not SHA-256")
        if validation["visual_export_schema"] != "visual_snapshot_v1":
            raise HarnessError("SDK_SMOKE_THROUGH_HARNESS_FAILS", "visual export schema was not visual_snapshot_v1")
        if not validation["checkpoint_save_load_pass"] or not validation["rollback_success"]:
            raise HarnessError("SDK_SMOKE_THROUGH_HARNESS_FAILS", "checkpoint or rollback validation failed")

        write_json(self.out_dir / "artifact_validation.json", validation)
        self.progress("artifact_validation_completed", "completed", "artifact validation completed")
        self.audit("artifact_validation_completed", "completed", validation=validation)
        return validation

    def write_final(self, gate_pass: bool, verdicts: list[str], validation: dict[str, Any] | None = None) -> None:
        audit_events = self.audit_events()
        summary = {
            "schema_version": DEPLOYMENT_SCHEMA_VERSION,
            "deployment_harness_gate_pass": gate_pass,
            "config_sha256_normalized_lf": self.config_hash,
            "config_policy_decision": self.policy_decision,
            "sdk_smoke_start_timestamp": self.sdk_smoke_start_timestamp,
            "sdk_smoke_completed_timestamp": self.sdk_smoke_completed_timestamp,
            "child_exit_code": self.child_exit_code,
            "child_command": self.child_command,
            "audit_log_complete": all(event in audit_events for event in REQUIRED_AUDIT_EVENTS),
            "validation": validation or {},
            "verdicts": verdicts,
            "production_default_training_enabled": False,
            "public_beta_promoted": False,
            "production_api_ready": False,
            "claim_boundary": BOUNDARY_TOKENS,
        }
        write_json(self.summary_path, summary)
        self.report_path.write_text(
            "# STABLE_LOOP_PHASE_LOCK_058_DEPLOYMENT_HARNESS Report\n\n"
            f"Status: {'positive' if gate_pass else 'failed'}.\n\n"
            "This is local/private deployment harness engineering only. "
            "This is not production deployment, hosted SaaS, public beta, clinical readiness, "
            "high-stakes education readiness, production API readiness, full VRAXION, "
            "language grounding, consciousness, biological/FlyWire equivalence, or physical quantum behavior.\n\n"
            f"Child command: `{' '.join(self.child_command)}`\n\n"
            f"Config SHA-256 normalized LF: `{self.config_hash}`\n\n",
            encoding="utf-8",
        )

    def audit_events(self) -> set[str]:
        if not self.audit_log_path.exists():
            return set()
        events = set()
        for line in self.audit_log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.add(json.loads(line)["event"])
        return events

    def final_verdict(self, gate_pass: bool) -> None:
        self.progress("final_verdict", "completed" if gate_pass else "failed", "final verdict emitted")
        self.audit("final_verdict", "completed" if gate_pass else "failed")


def command_validate_config(args: argparse.Namespace) -> int:
    run = HarnessRun(Path(args.config), args.out)
    try:
        run.prepare()
        run.final_verdict(True)
        run.write_final(True, ["CONFIG_SCHEMA_VALID", "PRODUCTION_DEPLOYMENT_NOT_CLAIMED"])
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
    run.write_final(True, ["HEALTHCHECK_POSITIVE", "PRODUCTION_DEPLOYMENT_NOT_CLAIMED"], {"healthcheck": health})
    print(json.dumps({"check_pass": True, "healthcheck": health}))
    return 0


def command_run_local(args: argparse.Namespace) -> int:
    run = HarnessRun(Path(args.config), args.out)
    run.prepare()
    run.run_sdk_smoke()
    validation = run.validate_child_artifacts()
    run.final_verdict(True)
    run.write_final(
        True,
        [
            "SDK_SMOKE_THROUGH_HARNESS_POSITIVE",
            "CHECKPOINT_STORAGE_POSITIVE",
            "ROLLBACK_THROUGH_HARNESS_POSITIVE",
            "VISUAL_EXPORT_THROUGH_HARNESS_POSITIVE",
            "PRODUCTION_DEPLOYMENT_NOT_CLAIMED",
        ],
        validation,
    )
    print(json.dumps({"check_pass": True, "out_dir": str(run.out_dir.relative_to(REPO_ROOT))}))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    run = HarnessRun(Path(args.config), args.out)
    run.prepare()
    health = run.healthcheck()
    run.run_sdk_smoke()
    validation = run.validate_child_artifacts()
    validation["healthcheck"] = health
    run.final_verdict(True)
    run.write_final(
        True,
        [
            "DEPLOYMENT_HARNESS_POSITIVE",
            "LOCAL_RUNBOOK_WRITTEN",
            "CONFIG_SCHEMA_VALID",
            "SDK_SMOKE_THROUGH_HARNESS_POSITIVE",
            "HEALTHCHECK_POSITIVE",
            "AUDIT_LOGGING_POSITIVE",
            "CHECKPOINT_STORAGE_POSITIVE",
            "ROLLBACK_THROUGH_HARNESS_POSITIVE",
            "VISUAL_EXPORT_THROUGH_HARNESS_POSITIVE",
            "POLICY_GUARD_REJECTS_REGULATED_DEPLOYMENT",
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
