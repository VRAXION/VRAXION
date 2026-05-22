#!/usr/bin/env python3
"""Localhost-only INSTNCT service/API alpha for 062."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import unquote, urlparse


SERVICE_SCHEMA_VERSION = "instnct_service_api_alpha_v1"
CONFIG_SCHEMA_VERSION = "instnct_service_alpha_config_v1"
RATE_LIMIT_POLICY = "alpha_static_local"
REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = (REPO_ROOT / "target" / "pilot_wave").resolve()

CLAIM_BOUNDARY = [
    "no production API readiness",
    "no production deployment",
    "no hosted SaaS",
    "no public beta",
    "no multi-tenant IAM",
    "no clinical use",
    "no high-stakes education use",
    "no commercial launch",
    "no full VRAXION",
    "no language grounding",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]

ROUTES = [
    "GET /v1/health",
    "POST /v1/policy/check",
    "POST /v1/jobs",
    "GET /v1/jobs/{job_id}",
    "POST /v1/infer",
    "POST /v1/bounded-chat/infer",
    "POST /v1/evaluate",
    "POST /v1/visual-export",
    "GET /v1/artifacts/{job_id}/{artifact_name}",
]

ALLOWED_OPERATIONS = {"healthcheck", "infer", "bounded_chat_infer", "evaluate", "visual_export"}
REGULATED_TOKENS = [
    "clinical",
    "diagnosis",
    "treatment",
    "triage",
    "medication",
    "clinical decision support",
    "cds",
    "grading",
    "admissions",
    "student ranking",
    "placement",
    "proctoring",
    "high-stakes",
    "high_stakes",
    "high_stakes_education",
    "production",
    "production mode",
    "production_mode",
    "hosted saas",
    "hosted_saas",
    "public mode",
    "public_mode",
    "production deployment",
    "production_deployment",
    "public beta",
    "public_beta",
]


class ServiceError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 400,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable
        self.details = details or {}


def now_ms() -> int:
    return int(time.time() * 1000)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def has_path_traversal(path_text: str) -> bool:
    return any(part == ".." for part in Path(path_text).parts)


def resolve_repo_file(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute() or has_path_traversal(path_text):
        raise ServiceError("INVALID_PATH", "path must be relative to the repository")
    resolved = (REPO_ROOT / raw).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ServiceError("INVALID_PATH", "path escaped repository root") from exc
    return resolved


def resolve_out_dir(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute() or has_path_traversal(path_text):
        raise ServiceError("UNSAFE_OUT_DIR", "out_dir must be a relative target/pilot_wave path")
    parts = [part.lower() for part in raw.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise ServiceError("UNSAFE_OUT_DIR", "out_dir must be under target/pilot_wave")
    resolved = (REPO_ROOT / raw).resolve()
    try:
        resolved.relative_to(TARGET_ROOT)
    except ValueError as exc:
        raise ServiceError("UNSAFE_OUT_DIR", "out_dir escaped target/pilot_wave") from exc
    return resolved


def production_flag_contaminated(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {
                "production_default_training_enabled",
                "public_beta_promoted",
                "production_api_ready",
            } and item is True:
                return True
            if production_flag_contaminated(item):
                return True
    if isinstance(value, list):
        return any(production_flag_contaminated(item) for item in value)
    return False


def validate_bind_host(bind_host: str) -> None:
    if bind_host != "127.0.0.1":
        raise ServiceError(
            "PUBLIC_BIND_DETECTED",
            "service alpha bind_host must be exactly 127.0.0.1",
            status=400,
            details={"bind_host": bind_host},
        )


def load_config(path: Path, out_override: str | None = None) -> dict[str, Any]:
    config_path = resolve_repo_file(str(path)) if not path.is_absolute() else path
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ServiceError("CONFIG_INVALID", "config root must be an object")
    if out_override:
        config["out_dir"] = out_override
    required = [
        "schema_version",
        "service_schema_version",
        "bind_host",
        "port",
        "bearer_token",
        "deployment_config_path",
        "out_dir",
        "rate_limit_max_requests",
        "rate_limit_window_sec",
        "bounded_chat_artifact_root",
        "bounded_chat_runtime_out_root",
        "bounded_chat_max_input_chars",
        "bounded_chat_max_response_tokens",
        "bounded_chat_timeout_ms",
        "production_default_training_enabled",
        "public_beta_promoted",
        "production_api_ready",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ServiceError("CONFIG_INVALID", "missing config fields", details={"missing": missing})
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ServiceError("CONFIG_INVALID", "unknown service config schema version")
    if config["service_schema_version"] != SERVICE_SCHEMA_VERSION:
        raise ServiceError("CONFIG_INVALID", "unknown service API alpha schema version")
    if production_flag_contaminated(config):
        raise ServiceError("PRODUCTION_FLAG_CONTAMINATION", "production flags must remain false")
    validate_bind_host(str(config["bind_host"]))
    if not isinstance(config["port"], int) or not 0 <= config["port"] <= 65535:
        raise ServiceError("CONFIG_INVALID", "port must be an integer from 0 to 65535")
    if not isinstance(config["rate_limit_max_requests"], int) or config["rate_limit_max_requests"] < 1:
        raise ServiceError("CONFIG_INVALID", "rate_limit_max_requests must be positive")
    if not isinstance(config["rate_limit_window_sec"], int) or config["rate_limit_window_sec"] < 1:
        raise ServiceError("CONFIG_INVALID", "rate_limit_window_sec must be positive")
    artifact_root = resolve_out_dir(str(config["bounded_chat_artifact_root"]))
    if not artifact_root.exists():
        raise ServiceError("UPSTREAM_083_ARTIFACT_MISSING", "bounded_chat_artifact_root does not exist")
    runtime_out_root = resolve_out_dir(str(config["bounded_chat_runtime_out_root"]))
    for key in ["bounded_chat_max_input_chars", "bounded_chat_max_response_tokens", "bounded_chat_timeout_ms"]:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ServiceError("CONFIG_INVALID", f"{key} must be a positive integer")
    deployment_config = resolve_repo_file(str(config["deployment_config_path"]))
    if not deployment_config.exists():
        raise ServiceError("CONFIG_INVALID", "deployment_config_path does not exist")
    config["_config_path"] = str(config_path.relative_to(REPO_ROOT))
    config["_deployment_config_resolved"] = str(deployment_config.relative_to(REPO_ROOT))
    config["_out_dir_resolved"] = str(resolve_out_dir(str(config["out_dir"])).relative_to(REPO_ROOT))
    config["_bounded_chat_artifact_root_resolved"] = str(artifact_root.relative_to(REPO_ROOT))
    config["_bounded_chat_runtime_out_root_resolved"] = str(runtime_out_root.relative_to(REPO_ROOT))
    return config


def policy_decision(body: Any) -> dict[str, Any]:
    text = canonical_json(body).lower()
    matched = [token for token in REGULATED_TOKENS if token in text]
    if matched:
        return {
            "allowed": False,
            "reason": "regulated_or_unsupported_alpha_request",
            "matched_tokens": matched,
        }
    return {"allowed": True, "reason": "local_private_alpha_scope"}


@dataclass
class JobRecord:
    job_id: str
    operation: str
    request_id: str
    request_body_hash: str
    idempotency_key: str | None
    job_dir: Path
    job_created_at: float
    child_command: list[str]
    child_exit_code: int | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "job_id": self.job_id,
            "operation": self.operation,
            "request_id": self.request_id,
            "request_body_hash": self.request_body_hash,
            "idempotency_key": self.idempotency_key,
            "job_created_at": self.job_created_at,
            "child_command": self.child_command,
            "child_exit_code": self.child_exit_code,
            "artifacts": sorted(self.artifacts),
            "claim_boundary": CLAIM_BOUNDARY,
            "production_default_training_enabled": False,
            "public_beta_promoted": False,
            "production_api_ready": False,
        }


class ServiceState:
    def __init__(self, config: dict[str, Any], service_smoke_start: float | None = None):
        self.config = config
        self.out_dir = resolve_out_dir(str(config["out_dir"]))
        self.deployment_config_path = str(config["_deployment_config_resolved"])
        self.bounded_chat_artifact_root = resolve_out_dir(str(config["bounded_chat_artifact_root"]))
        self.bounded_chat_runtime_out_root = resolve_out_dir(str(config["bounded_chat_runtime_out_root"]))
        self.bounded_chat_max_input_chars = int(config["bounded_chat_max_input_chars"])
        self.bounded_chat_max_response_tokens = int(config["bounded_chat_max_response_tokens"])
        self.bounded_chat_timeout_ms = int(config["bounded_chat_timeout_ms"])
        self.bearer_token = str(config["bearer_token"])
        self.rate_limit_max_requests = int(config["rate_limit_max_requests"])
        self.rate_limit_window_sec = int(config["rate_limit_window_sec"])
        self.request_times: list[float] = []
        self.jobs: dict[str, JobRecord] = {}
        self.idempotency: dict[str, tuple[str, str]] = {}
        self.service_smoke_start = service_smoke_start
        self.lock = threading.Lock()

    def service_progress(self, event: str, status: str, **details: Any) -> None:
        append_jsonl(
            self.out_dir / "progress.jsonl",
            {
                "timestamp_ms": now_ms(),
                "event": event,
                "status": status,
                "details": details,
            },
        )

    def service_audit(
        self,
        *,
        request_id: str,
        route: str,
        auth_result: str,
        policy_result: str,
        status: str,
        prompt_sha256: str | None = None,
        child_job_path: str | None = None,
        checkpoint_sha256: str | None = None,
        artifact_package_zip_sha256: str | None = None,
    ) -> None:
        append_jsonl(
            self.out_dir / "audit_log.jsonl",
            {
                "request_id": request_id,
                "timestamp": now_ms(),
                "route": route,
                "auth_result": auth_result,
                "policy_result": policy_result,
                "status": status,
                "prompt_sha256": prompt_sha256,
                "child_job_path": child_job_path,
                "checkpoint_sha256": checkpoint_sha256,
                "artifact_package_zip_sha256": artifact_package_zip_sha256,
            },
        )

    def check_rate_limit(self) -> tuple[int, int | None]:
        now = time.time()
        window_start = now - self.rate_limit_window_sec
        self.request_times = [item for item in self.request_times if item >= window_start]
        remaining = self.rate_limit_max_requests - len(self.request_times)
        if remaining <= 0:
            oldest = min(self.request_times) if self.request_times else now
            retry_after = max(1, int((oldest + self.rate_limit_window_sec) - now) + 1)
            raise ServiceError(
                "RATE_LIMIT_EXCEEDED",
                "alpha static local rate limit exceeded",
                status=429,
                retryable=True,
                details={
                    "rate_limit_policy": RATE_LIMIT_POLICY,
                    "limit": self.rate_limit_max_requests,
                    "rate_limit_remaining": 0,
                    "reset_after": self.rate_limit_window_sec,
                    "retry_after": retry_after,
                },
            )
        self.request_times.append(now)
        return remaining - 1, None

    def authenticate(self, headers: Any) -> None:
        auth = headers.get("Authorization", "")
        expected = f"Bearer {self.bearer_token}"
        if auth != expected:
            raise ServiceError("AUTH_REQUIRED", "valid bearer token required", status=401)

    def progress(self, job: JobRecord, event: str, status: str, **details: Any) -> None:
        append_jsonl(
            job.job_dir / "progress.jsonl",
            {
                "timestamp_ms": now_ms(),
                "event": event,
                "status": status,
                "details": details,
            },
        )

    def audit(self, job: JobRecord, event: str, status: str, **details: Any) -> None:
        append_jsonl(
            job.job_dir / "audit_log.jsonl",
            {
                "timestamp_ms": now_ms(),
                "event": event,
                "status": status,
                "request_id": job.request_id,
                "job_id": job.job_id,
                "details": details,
            },
        )

    def create_job(self, operation: str, body: dict[str, Any], request_id: str) -> JobRecord:
        if operation not in ALLOWED_OPERATIONS:
            raise ServiceError("INVALID_OPERATION", "unsupported alpha operation", details={"operation": operation})
        decision = policy_decision(body)
        if not decision["allowed"]:
            raise ServiceError("POLICY_GUARD_REJECTED", "policy guard rejected request", status=403, details=decision)

        body_hash = sha256_text(canonical_json(body))
        idempotency_key = body.get("idempotency_key")
        if idempotency_key is not None and not isinstance(idempotency_key, str):
            raise ServiceError("INVALID_INPUT", "idempotency_key must be a string")

        with self.lock:
            if idempotency_key and idempotency_key in self.idempotency:
                old_hash, old_job_id = self.idempotency[idempotency_key]
                if old_hash != body_hash:
                    raise ServiceError(
                        "IDEMPOTENCY_CONFLICT",
                        "same idempotency_key used with a different request body",
                        status=409,
                        details={"idempotency_key": idempotency_key},
                    )
                return self.jobs[old_job_id]

            seed = idempotency_key or f"{request_id}:{time.time_ns()}"
            job_id = "job_" + hashlib.sha256(f"{seed}:{body_hash}".encode("utf-8")).hexdigest()[:16]
            job_dir = self.out_dir / "jobs" / job_id
            if job_dir.exists() and not idempotency_key:
                shutil.rmtree(job_dir)
            job_dir.mkdir(parents=True, exist_ok=True)
            harness_out = job_dir / "harness_healthcheck"
            harness_out_rel = str(harness_out.relative_to(REPO_ROOT))
            child_command = [
                "python",
                "tools/instnct_deploy/instnct_deploy.py",
                "healthcheck",
                "--config",
                self.deployment_config_path,
                "--out",
                harness_out_rel,
            ]
            job = JobRecord(
                job_id=job_id,
                operation=operation,
                request_id=request_id,
                request_body_hash=body_hash,
                idempotency_key=idempotency_key,
                job_dir=job_dir,
                job_created_at=time.time(),
                child_command=child_command,
            )
            self.jobs[job_id] = job
            if idempotency_key:
                self.idempotency[idempotency_key] = (body_hash, job_id)

        self.run_job(job, body)
        return job

    def validate_bounded_chat_body(self, body: dict[str, Any]) -> tuple[str, int]:
        if "prompt" not in body:
            raise ServiceError("INVALID_INPUT", "missing prompt")
        prompt = body["prompt"]
        if not isinstance(prompt, str):
            raise ServiceError("INVALID_INPUT", "prompt must be a string")
        if not prompt:
            raise ServiceError("INVALID_INPUT", "prompt must not be empty")
        if not prompt.strip():
            raise ServiceError("INVALID_INPUT", "prompt must not be whitespace")
        if len(prompt) > self.bounded_chat_max_input_chars:
            raise ServiceError(
                "INVALID_INPUT",
                "prompt exceeds bounded_chat_max_input_chars",
                details={"max_input_chars": self.bounded_chat_max_input_chars},
            )
        max_tokens = body.get("max_response_tokens", self.bounded_chat_max_response_tokens)
        if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > self.bounded_chat_max_response_tokens:
            raise ServiceError(
                "INVALID_INPUT",
                "max_response_tokens must be a positive integer within the configured bound",
                details={"max_response_tokens_limit": self.bounded_chat_max_response_tokens},
            )
        return prompt, max_tokens

    def create_bounded_chat_job(self, body: dict[str, Any], request_id: str) -> JobRecord:
        prompt, max_tokens = self.validate_bounded_chat_body(body)
        decision = policy_decision(body)
        if not decision["allowed"]:
            raise ServiceError("POLICY_GUARD_REJECTED", "policy guard rejected request", status=403, details=decision)

        body_hash = sha256_text(canonical_json(body))
        idempotency_key = body.get("idempotency_key")
        if idempotency_key is not None and not isinstance(idempotency_key, str):
            raise ServiceError("INVALID_INPUT", "idempotency_key must be a string")

        with self.lock:
            if idempotency_key and idempotency_key in self.idempotency:
                old_hash, old_job_id = self.idempotency[idempotency_key]
                if old_hash != body_hash:
                    raise ServiceError(
                        "IDEMPOTENCY_CONFLICT",
                        "same idempotency_key used with a different request body",
                        status=409,
                        details={"idempotency_key": idempotency_key},
                    )
                return self.jobs[old_job_id]

            seed = idempotency_key or f"{request_id}:{time.time_ns()}"
            job_id = "job_" + hashlib.sha256(f"bounded-chat:{seed}:{body_hash}".encode("utf-8")).hexdigest()[:16]
            job_dir = self.out_dir / "jobs" / job_id
            if job_dir.exists() and not idempotency_key:
                shutil.rmtree(job_dir)
            job_dir.mkdir(parents=True, exist_ok=True)
            child_out = self.bounded_chat_runtime_out_root / job_id
            if child_out.exists() and not idempotency_key:
                shutil.rmtree(child_out)
            child_command = [
                "cargo",
                "run",
                "-p",
                "instnct-core",
                "--example",
                "phase_lane_bounded_chat_inference_runtime",
                "--",
                "--out",
                str(child_out.relative_to(REPO_ROOT)),
                "--artifact-root",
                str(self.bounded_chat_artifact_root.relative_to(REPO_ROOT)),
                "--prompt",
                prompt,
                "--max-input-chars",
                str(self.bounded_chat_max_input_chars),
                "--max-response-tokens",
                str(max_tokens),
                "--timeout-ms",
                str(self.bounded_chat_timeout_ms),
                "--json",
                "--heartbeat-sec",
                "20",
            ]
            job = JobRecord(
                job_id=job_id,
                operation="bounded_chat_infer",
                request_id=request_id,
                request_body_hash=body_hash,
                idempotency_key=idempotency_key,
                job_dir=job_dir,
                job_created_at=time.time(),
                child_command=child_command,
            )
            self.jobs[job_id] = job
            if idempotency_key:
                self.idempotency[idempotency_key] = (body_hash, job_id)

        if not (job.job_dir / "bounded_chat_response.json").exists():
            self.run_bounded_chat_job(job, prompt)
        return job

    def run_bounded_chat_job(self, job: JobRecord, prompt: str) -> None:
        self.progress(job, "job_created", "completed", operation=job.operation)
        self.audit(job, "job_created", "completed", operation=job.operation)
        self.progress(job, "child_started", "start", child_command=job.child_command)
        self.audit(job, "child_started", "start", child_command=job.child_command)
        process_timeout_sec = max(30.0, self.bounded_chat_timeout_ms / 1000.0 + 10.0)
        try:
            child = subprocess.run(
                job.child_command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                timeout=process_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            job.child_exit_code = None
            (job.job_dir / "child_stdout.txt").write_text(str(exc.stdout or ""), encoding="utf-8")
            (job.job_dir / "child_stderr.txt").write_text(str(exc.stderr or ""), encoding="utf-8")
            self.progress(job, "child_timeout", "failed", timeout_sec=process_timeout_sec)
            self.audit(job, "child_timeout", "failed", timeout_sec=process_timeout_sec)
            raise ServiceError(
                "TIMEOUT_GUARD_FAILS",
                "bounded chat child runtime exceeded service child timeout",
                status=504,
                retryable=True,
                details={"child_job_path": str(job.job_dir.relative_to(REPO_ROOT))},
            ) from exc

        job.child_exit_code = child.returncode
        (job.job_dir / "child_stdout.txt").write_text(child.stdout, encoding="utf-8")
        (job.job_dir / "child_stderr.txt").write_text(child.stderr, encoding="utf-8")
        self.progress(
            job,
            "child_completed",
            "completed" if child.returncode == 0 else "failed",
            child_exit_code=child.returncode,
        )
        self.audit(
            job,
            "child_completed",
            "completed" if child.returncode == 0 else "failed",
            child_exit_code=child.returncode,
        )

        child_out = self.bounded_chat_runtime_out_root / job.job_id
        required = ["single_inference.json", "runtime_metrics.json", "summary.json", "report.md", "audit_log.jsonl"]
        missing = [name for name in required if not (child_out / name).exists()]
        if missing:
            raise ServiceError(
                "BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_FAILS",
                "bounded chat child artifacts missing",
                status=500,
                details={"missing": missing},
            )
        inference = read_json(child_out / "single_inference.json")
        metrics = read_json(child_out / "runtime_metrics.json")
        summary = read_json(child_out / "summary.json")
        verdicts = summary.get("verdicts", []) if isinstance(summary, dict) else []
        child_positive = "BOUNDED_CHAT_INFERENCE_RUNTIME_POSITIVE" in verdicts
        unsupported_child = inference.get("status") == "unsupported"
        if child.returncode != 0 and not unsupported_child:
            raise ServiceError(
                "BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_FAILS",
                "bounded chat child runtime failed",
                status=500,
                details={"child_exit_code": child.returncode, "child_job_path": str(job.job_dir.relative_to(REPO_ROOT))},
            )
        if not child_positive and not unsupported_child:
            raise ServiceError("BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_FAILS", "child 084 summary was not positive", status=500)
        if metrics.get("artifact_hash_verified") is not True:
            raise ServiceError("ARTIFACT_HASH_MISMATCH", "child artifact hash verification failed", status=500)
        if metrics.get("checkpoint_hash_unchanged") is not True:
            raise ServiceError("CHECKPOINT_MUTATION_DETECTED", "child checkpoint hash changed", status=500)
        if metrics.get("train_step_count") != 0:
            raise ServiceError("TRAINING_SIDE_EFFECT_DETECTED", "child runtime reported training steps", status=500)

        artifact_hash = inference.get("artifact_package_zip_sha256")
        preserved_child_out = job.job_dir / "bounded_chat_runtime"
        if preserved_child_out.exists():
            shutil.rmtree(preserved_child_out)
        preserved_child_out.mkdir(parents=True, exist_ok=True)
        for name in required:
            shutil.copyfile(child_out / name, preserved_child_out / name)

        response = {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "job_id": job.job_id,
            "operation": job.operation,
            "request_id": job.request_id,
            "idempotency_key": job.idempotency_key,
            "route": "/v1/bounded-chat/infer",
            "artifact_hash": artifact_hash,
            "child_job_path": str(child_out.relative_to(REPO_ROOT)),
            "child_exit_code": job.child_exit_code,
            "inference": inference,
            "runtime_metrics": metrics,
            "bounded_chat_child_084_positive": child_positive,
            "production_default_training_enabled": False,
            "public_beta_promoted": False,
            "production_api_ready": False,
        }
        write_json(job.job_dir / "bounded_chat_response.json", response)
        job.artifacts["bounded_chat_response.json"] = "bounded_chat_response.json"
        job.artifacts["child_stdout.txt"] = "child_stdout.txt"
        job.artifacts["child_stderr.txt"] = "child_stderr.txt"
        for name in required:
            job.artifacts[f"bounded_chat_runtime/{name}"] = f"bounded_chat_runtime/{name}"
        self.add_artifact(job, "progress.jsonl", (job.job_dir / "progress.jsonl").read_text(encoding="utf-8"))
        self.add_artifact(job, "audit_log.jsonl", (job.job_dir / "audit_log.jsonl").read_text(encoding="utf-8"))
        self.add_artifact(job, "job_manifest.json", job.to_manifest())
        self.progress(job, "artifacts_written", "completed", artifacts=sorted(job.artifacts))
        self.audit(job, "artifacts_written", "completed", artifacts=sorted(job.artifacts))
        self.add_artifact(job, "job_manifest.json", job.to_manifest())

    def bounded_chat_value(self, job: JobRecord) -> dict[str, Any]:
        path = job.job_dir / "bounded_chat_response.json"
        if not path.exists():
            raise ServiceError("BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_FAILS", "bounded chat response artifact missing", status=500)
        return read_json(path)

    def run_job(self, job: JobRecord, body: dict[str, Any]) -> None:
        self.progress(job, "job_created", "completed", operation=job.operation)
        self.audit(job, "job_created", "completed", operation=job.operation)
        self.progress(job, "child_started", "start", child_command=job.child_command)
        self.audit(job, "child_started", "start", child_command=job.child_command)
        child = subprocess.run(job.child_command, cwd=REPO_ROOT, text=True, capture_output=True)
        job.child_exit_code = child.returncode
        (job.job_dir / "child_stdout.txt").write_text(child.stdout, encoding="utf-8")
        (job.job_dir / "child_stderr.txt").write_text(child.stderr, encoding="utf-8")
        self.progress(job, "child_completed", "completed" if child.returncode == 0 else "failed", child_exit_code=child.returncode)
        self.audit(job, "child_completed", "completed" if child.returncode == 0 else "failed", child_exit_code=child.returncode)
        if child.returncode != 0:
            raise ServiceError("CHILD_COMMAND_FAILED", "alpha child command failed", status=500)
        self.write_operation_artifacts(job, body)

    def add_artifact(self, job: JobRecord, artifact_name: str, value: Any) -> None:
        path = job.job_dir / artifact_name
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            write_json(path, value)
        job.artifacts[artifact_name] = str(path.relative_to(job.job_dir))

    def write_operation_artifacts(self, job: JobRecord, body: dict[str, Any]) -> None:
        common = {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "job_id": job.job_id,
            "operation": job.operation,
            "request_body_hash": job.request_body_hash,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        if job.operation == "infer":
            self.add_artifact(
                job,
                "inference_result.json",
                {
                    **common,
                    "input_count": len(body.get("inputs", [])) if isinstance(body.get("inputs", []), list) else 0,
                    "result_kind": "alpha_inference_envelope",
                    "production_default_training_enabled": False,
                    "public_beta_promoted": False,
                    "production_api_ready": False,
                },
            )
        elif job.operation == "evaluate":
            self.add_artifact(
                job,
                "eval_report.json",
                {
                    **common,
                    "eval_suite_ref": body.get("eval_suite_ref", "alpha_local_eval"),
                    "result_kind": "alpha_eval_envelope",
                    "production_default_training_enabled": False,
                    "public_beta_promoted": False,
                    "production_api_ready": False,
                },
            )
        elif job.operation == "visual_export":
            self.add_artifact(
                job,
                "visual_export_manifest.json",
                {
                    **common,
                    "visual_schema_version": "visual_snapshot_v1",
                    "result_kind": "alpha_visual_export_envelope",
                    "production_default_training_enabled": False,
                    "public_beta_promoted": False,
                    "production_api_ready": False,
                },
            )
        else:
            self.add_artifact(
                job,
                "healthcheck_result.json",
                {
                    **common,
                    "result_kind": "alpha_healthcheck_job",
                    "production_default_training_enabled": False,
                    "public_beta_promoted": False,
                    "production_api_ready": False,
                },
            )
        self.add_artifact(job, "progress.jsonl", (job.job_dir / "progress.jsonl").read_text(encoding="utf-8"))
        self.add_artifact(job, "audit_log.jsonl", (job.job_dir / "audit_log.jsonl").read_text(encoding="utf-8"))
        job.artifacts["child_stdout.txt"] = "child_stdout.txt"
        job.artifacts["child_stderr.txt"] = "child_stderr.txt"
        self.add_artifact(job, "job_manifest.json", job.to_manifest())
        self.progress(job, "artifacts_written", "completed", artifacts=sorted(job.artifacts))
        self.audit(job, "artifacts_written", "completed", artifacts=sorted(job.artifacts))
        self.add_artifact(job, "job_manifest.json", job.to_manifest())

    def artifact_value(self, job_id: str, artifact_name: str) -> dict[str, Any]:
        if "/" in artifact_name or "\\" in artifact_name or artifact_name.startswith("."):
            raise ServiceError("ARTIFACT_PATH_ESCAPE", "artifact name is not allowlisted", status=403)
        job = self.jobs.get(job_id)
        if not job:
            raise ServiceError("JOB_NOT_FOUND", "job not found", status=404)
        rel = job.artifacts.get(artifact_name)
        if not rel:
            raise ServiceError("ARTIFACT_PATH_ESCAPE", "artifact name is not registered for this job", status=403)
        path = (job.job_dir / rel).resolve()
        try:
            path.relative_to(job.job_dir.resolve())
        except ValueError as exc:
            raise ServiceError("ARTIFACT_PATH_ESCAPE", "artifact path escaped job directory", status=403) from exc
        text = path.read_text(encoding="utf-8")
        parsed: Any | None = None
        if artifact_name.endswith(".json"):
            parsed = json.loads(text)
        return {
            "job_id": job_id,
            "artifact_name": artifact_name,
            "content_text": text,
            "content_json": parsed,
        }


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "INSTNCTServiceAlpha/0.1"

    @property
    def state(self) -> ServiceState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def request_id(self) -> str:
        return self.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex[:12]}"

    def read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ServiceError("INVALID_JSON", "malformed JSON request body") from exc
        if not isinstance(body, dict):
            raise ServiceError("INVALID_INPUT", "request JSON body must be an object")
        return body

    def send_envelope(
        self,
        status: int,
        request_id: str,
        *,
        ok: bool,
        value: Any | None = None,
        error: dict[str, Any] | None = None,
        rate_limit_remaining: int | None = None,
        retry_after: int | None = None,
    ) -> None:
        route = urlparse(self.path).path
        idempotency_key = None
        artifact_hash = None
        child_job_path = None
        if isinstance(value, dict):
            idempotency_key = value.get("idempotency_key")
            artifact_hash = value.get("artifact_hash")
            child_job_path = value.get("child_job_path")
        envelope: dict[str, Any] = {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "ok": ok,
            "value": value if ok else None,
            "error": None if ok else error,
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "route": route,
            "claim_boundary": CLAIM_BOUNDARY,
            "rate_limit_policy": RATE_LIMIT_POLICY,
            "rate_limit_remaining": rate_limit_remaining,
            "rate_limit": {
                "policy": RATE_LIMIT_POLICY,
                "limit": self.state.rate_limit_max_requests,
                "remaining": rate_limit_remaining,
                "reset_after": self.state.rate_limit_window_sec,
                "retry_after": retry_after,
            },
            "artifact_hash": artifact_hash,
            "child_job_path": child_job_path,
        }
        if retry_after is not None:
            envelope["retry_after"] = retry_after
        encoded = json.dumps(envelope, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def handle_error(self, err: ServiceError, request_id: str, remaining: int | None = None) -> None:
        retry_after = None
        if err.code == "RATE_LIMIT_EXCEEDED":
            remaining = 0
            retry_after = int(err.details.get("retry_after", 1))
        self.send_envelope(
            err.status,
            request_id,
            ok=False,
            error={
                "code": err.code,
                "message": err.message,
                "retryable": err.retryable,
                "details": err.details,
            },
            rate_limit_remaining=remaining,
            retry_after=retry_after,
        )

    def guarded(self, *, auth_required: bool, handler: Any) -> None:
        request_id = self.request_id()
        remaining: int | None = None
        route = urlparse(self.path).path
        try:
            remaining, _ = self.state.check_rate_limit()
            if auth_required:
                self.state.authenticate(self.headers)
            value = handler(request_id)
            self.send_envelope(200, request_id, ok=True, value=value, rate_limit_remaining=remaining)
        except ServiceError as err:
            if route == "/v1/bounded-chat/infer":
                auth_result = "failed" if err.code == "AUTH_REQUIRED" else "passed"
                policy_result = "failed" if err.code == "POLICY_GUARD_REJECTED" else "not_evaluated"
                self.state.service_progress("bounded_chat_request_rejected", "failed", error_code=err.code)
                self.state.service_audit(
                    request_id=request_id,
                    route=route,
                    auth_result=auth_result,
                    policy_result=policy_result,
                    status=err.code,
                )
            self.handle_error(err, request_id, remaining)
        except Exception as exc:  # pragma: no cover - defensive envelope guard.
            if route == "/v1/bounded-chat/infer":
                self.state.service_progress(
                    "bounded_chat_request_rejected",
                    "failed",
                    error_code="INTERNAL_ERROR",
                    error_message=str(exc),
                )
                self.state.service_audit(
                    request_id=request_id,
                    route=route,
                    auth_result="passed" if auth_required else "not_required",
                    policy_result="unknown",
                    status=f"INTERNAL_ERROR:{exc}",
                )
            self.handle_error(
                ServiceError("INTERNAL_ERROR", str(exc), status=500, retryable=False),
                request_id,
                remaining,
            )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/v1/health":
            self.guarded(auth_required=False, handler=lambda request_id: self.route_health(request_id))
            return
        if path.startswith("/v1/jobs/"):
            self.guarded(auth_required=True, handler=lambda request_id: self.route_get_job(path))
            return
        if path.startswith("/v1/artifacts/"):
            self.guarded(auth_required=True, handler=lambda request_id: self.route_get_artifact(path))
            return
        self.guarded(
            auth_required=False,
            handler=lambda request_id: (_ for _ in ()).throw(ServiceError("NOT_FOUND", "route not found", status=404)),
        )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/v1/policy/check":
            self.guarded(auth_required=True, handler=lambda request_id: self.route_policy_check())
            return
        if path == "/v1/jobs":
            self.guarded(auth_required=True, handler=lambda request_id: self.route_create_job(request_id, "healthcheck"))
            return
        if path == "/v1/infer":
            self.guarded(auth_required=True, handler=lambda request_id: self.route_create_job(request_id, "infer"))
            return
        if path == "/v1/bounded-chat/infer":
            self.guarded(auth_required=True, handler=lambda request_id: self.route_bounded_chat_infer(request_id))
            return
        if path == "/v1/evaluate":
            self.guarded(auth_required=True, handler=lambda request_id: self.route_create_job(request_id, "evaluate"))
            return
        if path == "/v1/visual-export":
            self.guarded(auth_required=True, handler=lambda request_id: self.route_create_job(request_id, "visual_export"))
            return
        self.guarded(
            auth_required=False,
            handler=lambda request_id: (_ for _ in ()).throw(ServiceError("NOT_FOUND", "route not found", status=404)),
        )

    def route_health(self, request_id: str) -> dict[str, Any]:
        return {
            "service": "instnct_service_alpha",
            "schema_version": SERVICE_SCHEMA_VERSION,
            "bind_host": self.state.config["bind_host"],
            "routes": ROUTES,
            "production_default_training_enabled": False,
            "public_beta_promoted": False,
            "production_api_ready": False,
        }

    def route_policy_check(self) -> dict[str, Any]:
        body = self.read_body()
        decision = policy_decision(body)
        return {"policy_decision": decision, "side_effects": "none"}

    def route_create_job(self, request_id: str, operation: str) -> dict[str, Any]:
        body = self.read_body()
        if self.path == "/v1/jobs":
            operation = str(body.get("operation", operation)).replace("-", "_")
        job = self.state.create_job(operation, body, request_id)
        return job.to_manifest()

    def route_bounded_chat_infer(self, request_id: str) -> dict[str, Any]:
        body = self.read_body()
        prompt = body.get("prompt") if isinstance(body, dict) else None
        prompt_sha = sha256_text(prompt) if isinstance(prompt, str) else None
        job = self.state.create_bounded_chat_job(body, request_id)
        value = self.state.bounded_chat_value(job)
        inference = value.get("inference", {})
        self.state.service_progress(
            "bounded_chat_inference_completed",
            "completed",
            job_id=job.job_id,
            child_job_path=value.get("child_job_path"),
            inference_status=inference.get("status"),
        )
        self.state.service_audit(
            request_id=request_id,
            route="/v1/bounded-chat/infer",
            auth_result="passed",
            policy_result="passed",
            status=str(inference.get("status", "ok")),
            prompt_sha256=prompt_sha,
            child_job_path=value.get("child_job_path"),
            checkpoint_sha256=inference.get("checkpoint_sha256"),
            artifact_package_zip_sha256=inference.get("artifact_package_zip_sha256"),
        )
        return value

    def route_get_job(self, path: str) -> dict[str, Any]:
        parts = path.rstrip("/").split("/")
        if len(parts) != 4:
            raise ServiceError("NOT_FOUND", "job route not found", status=404)
        job_id = unquote(parts[3])
        job = self.state.jobs.get(job_id)
        if not job:
            raise ServiceError("JOB_NOT_FOUND", "job not found", status=404)
        return job.to_manifest()

    def route_get_artifact(self, path: str) -> dict[str, Any]:
        parts = path.rstrip("/").split("/")
        if len(parts) != 5:
            raise ServiceError("ARTIFACT_PATH_ESCAPE", "artifact route requires job_id and artifact_name", status=403)
        job_id = unquote(parts[3])
        artifact_name = unquote(parts[4])
        return self.state.artifact_value(job_id, artifact_name)


def make_server(config: dict[str, Any], service_smoke_start: float | None = None) -> ThreadingHTTPServer:
    state = ServiceState(config, service_smoke_start)
    server = ThreadingHTTPServer((str(config["bind_host"]), int(config["port"])), RequestHandler)
    server.state = state  # type: ignore[attr-defined]
    return server


def http_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    body: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None
    headers = {"Content-Type": "application/json", "X-Request-Id": f"req_{uuid.uuid4().hex[:10]}"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
        if body is None:
            body = {}
        body["idempotency_key"] = idempotency_key
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers=headers, method=method)
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
    except urlerror.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload


def http_raw_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    raw_body: str = "",
) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": "application/json", "X-Request-Id": f"req_{uuid.uuid4().hex[:10]}"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = urlrequest.Request(url, data=raw_body.encode("utf-8"), headers=headers, method=method)
    try:
        with urlrequest.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
    except urlerror.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload


def assert_envelope(payload: dict[str, Any], *, expect_ok: bool) -> None:
    required = [
        "schema_version",
        "ok",
        "value",
        "error",
        "request_id",
        "idempotency_key",
        "route",
        "rate_limit",
        "artifact_hash",
        "child_job_path",
        "claim_boundary",
        "rate_limit_policy",
        "rate_limit_remaining",
    ]
    for key in required:
        if key not in payload:
            raise ServiceError("API_ERROR_ENVELOPE_INCONSISTENT", f"missing envelope key {key}")
    if payload["schema_version"] != SERVICE_SCHEMA_VERSION:
        raise ServiceError("API_ERROR_ENVELOPE_INCONSISTENT", "wrong schema_version")
    if payload["ok"] is not expect_ok:
        raise ServiceError("API_ERROR_ENVELOPE_INCONSISTENT", "unexpected ok value")
    if expect_ok and "value" not in payload:
        raise ServiceError("API_ERROR_ENVELOPE_INCONSISTENT", "success envelope missing value")
    if not expect_ok:
        error = payload.get("error")
        if not isinstance(error, dict):
            raise ServiceError("API_ERROR_ENVELOPE_INCONSISTENT", "error envelope missing error object")
        for key in ["code", "message", "retryable", "details"]:
            if key not in error:
                raise ServiceError("API_ERROR_ENVELOPE_INCONSISTENT", f"error envelope missing {key}")
    rate_limit = payload.get("rate_limit")
    if not isinstance(rate_limit, dict):
        raise ServiceError("RATE_LIMIT_BOUNDARY_MISSING", "rate_limit metadata object missing")
    for key in ["limit", "remaining", "reset_after"]:
        if key not in rate_limit:
            raise ServiceError("RATE_LIMIT_BOUNDARY_MISSING", f"rate_limit metadata missing {key}")


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_smoke_progress(out_dir: Path, event: str, status: str, **details: Any) -> None:
    append_jsonl(
        out_dir / "progress.jsonl",
        {
            "timestamp_ms": now_ms(),
            "event": event,
            "status": status,
            "details": details,
        },
    )


def write_085_summary_report(out_dir: Path, status: str, metrics: dict[str, Any], verdicts: list[str]) -> None:
    summary = {
        "schema_version": SERVICE_SCHEMA_VERSION,
        "milestone": "STABLE_LOOP_PHASE_LOCK_085_BOUNDED_CHAT_SERVICE_API_ALPHA",
        "status": status,
        "service_api_alpha_only": True,
        "localhost_private_only": True,
        "deploy_ready_service": False,
        "public_api_claimed": False,
        "sdk_surface_exposed": False,
        "GPT_like_assistant_readiness_claimed": False,
        "open_domain_chat_supported": False,
        "production_chat_claimed": False,
        "safety_alignment_claimed": False,
        "public_beta_claimed": False,
        "GA_claimed": False,
        "hosted_SaaS_claimed": False,
        "service_metrics": metrics,
        "verdicts": verdicts,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(
        "# STABLE_LOOP_PHASE_LOCK_085_BOUNDED_CHAT_SERVICE_API_ALPHA Report\n\n"
        f"Status: `{status}`\n\n"
        "085 is service API alpha only, localhost/private only. It is not deploy-ready service, "
        "not public API, not SDK surface, not GPT-like assistant, not open-domain chat, "
        "not production chat, not safety alignment, not public beta / GA / hosted SaaS.\n\n"
        "The bounded chat API route delegates inference to the 084 Rust CLI child runtime and "
        "parses `single_inference.json`, `runtime_metrics.json`, `summary.json`, `report.md`, "
        "and `audit_log.jsonl`.\n",
        encoding="utf-8",
    )


def run_smoke(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    initial_metrics = {
        "localhost_bind_only": False,
        "public_bind_rejected": False,
        "auth_required": False,
        "auth_rejection_has_no_child_side_effect": False,
        "policy_rejection_has_no_child_side_effect": False,
        "rate_limit_metadata_present": False,
        "bounded_chat_route_registered": False,
        "bounded_chat_single_prompt_pass": False,
        "bounded_chat_json_envelope_pass": False,
        "bounded_chat_child_084_positive": False,
        "artifact_hash_verified": False,
        "checkpoint_hash_unchanged": False,
        "train_step_count": 0,
        "unsupported_input_handled": False,
        "bad_input_handled": False,
        "timeout_guard_pass": False,
        "idempotency_reuse_pass": False,
        "idempotency_conflict_pass": False,
        "audit_log_written": False,
        "child_runtime_artifacts_preserved": False,
        "existing_062_routes_preserved": False,
        "service_api_alpha_only": True,
        "production_chat_claimed": False,
        "gpt_like_assistant_readiness_claimed": False,
        "sdk_surface_exposed": False,
        "deployment_harness_mutated": False,
    }
    write_json(
        out_dir / "queue.json",
        {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "milestone": "STABLE_LOOP_PHASE_LOCK_085_BOUNDED_CHAT_SERVICE_API_ALPHA",
            "steps": [
                "start_localhost_service",
                "verify_auth_policy_guards",
                "run_bounded_chat_child_runtime",
                "verify_bad_and_unsupported_inputs",
                "verify_idempotency",
                "verify_rate_limit",
                "write_final_summary",
            ],
            "service_api_alpha_only": True,
            "localhost_private_only": True,
        },
    )
    write_smoke_progress(out_dir, "start", "running", milestone="STABLE_LOOP_PHASE_LOCK_085_BOUNDED_CHAT_SERVICE_API_ALPHA")
    write_085_summary_report(out_dir, "running", initial_metrics, ["BOUNDED_CHAT_SERVICE_API_ALPHA_RUNNING"])
    service_smoke_start = time.time()
    config = dict(config)
    config["out_dir"] = str(out_dir.relative_to(REPO_ROOT))
    config["bounded_chat_runtime_out_root"] = str((out_dir / "runtime_children").relative_to(REPO_ROOT))
    config["port"] = 0
    resolved_config = dict(config)
    resolved_config["bearer_token"] = "<redacted>"
    write_json(out_dir / "service_config_resolved.json", resolved_config)
    write_json(out_dir / "route_manifest.json", {"schema_version": SERVICE_SCHEMA_VERSION, "routes": ROUTES})
    server = make_server(config, service_smoke_start)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    token = str(config["bearer_token"])
    observations: dict[str, Any] = {"base_url": base_url}
    metrics = dict(initial_metrics)
    metrics["localhost_bind_only"] = host == "127.0.0.1"
    try:
        status, health = http_json("GET", f"{base_url}/v1/health")
        assert status == 200
        assert_envelope(health, expect_ok=True)
        if "POST /v1/bounded-chat/infer" not in health["value"]["routes"]:
            raise ServiceError("BOUNDED_CHAT_ROUTE_MISSING", "bounded chat route missing from health metadata")
        metrics["bounded_chat_route_registered"] = True
        observations["health_ok"] = True
        write_smoke_progress(out_dir, "health_checked", "completed", routes=health["value"]["routes"])

        jobs_before = len(list((out_dir / "jobs").glob("*"))) if (out_dir / "jobs").exists() else 0
        status, auth_error = http_json("POST", f"{base_url}/v1/infer", body={"inputs": ["alpha"]})
        assert status == 401
        assert_envelope(auth_error, expect_ok=False)
        jobs_after_auth = len(list((out_dir / "jobs").glob("*"))) if (out_dir / "jobs").exists() else 0
        if jobs_after_auth != jobs_before:
            raise ServiceError("AUTHZ_SIDE_EFFECT_LEAK", "auth rejection created job side effects")
        observations["auth_side_effect_guard"] = True

        bounded_children_before = len(list((out_dir / "runtime_children").glob("*"))) if (out_dir / "runtime_children").exists() else 0
        status, bounded_auth_error = http_json(
            "POST",
            f"{base_url}/v1/bounded-chat/infer",
            body={"prompt": "active code silver, distractor pocket teal; produce active answer"},
        )
        assert status == 401
        assert_envelope(bounded_auth_error, expect_ok=False)
        bounded_children_after_auth = len(list((out_dir / "runtime_children").glob("*"))) if (out_dir / "runtime_children").exists() else 0
        if bounded_children_after_auth != bounded_children_before:
            raise ServiceError("AUTHZ_SIDE_EFFECT_LEAK", "auth rejection created bounded chat child side effects")
        metrics["auth_required"] = True
        metrics["auth_rejection_has_no_child_side_effect"] = True

        status, policy_error = http_json(
            "POST",
            f"{base_url}/v1/infer",
            token=token,
            body={"intended_use": "clinical diagnosis", "inputs": ["alpha"]},
        )
        assert status == 403
        assert_envelope(policy_error, expect_ok=False)
        jobs_after_policy = len(list((out_dir / "jobs").glob("*"))) if (out_dir / "jobs").exists() else 0
        if jobs_after_policy != jobs_before:
            raise ServiceError("POLICY_REJECTION_SIDE_EFFECT_LEAK", "policy rejection created job side effects")
        observations["policy_side_effect_guard"] = True

        status, bounded_policy_error = http_json(
            "POST",
            f"{base_url}/v1/bounded-chat/infer",
            token=token,
            body={"intended_use": "clinical high_stakes_education production public_beta hosted_saas", "prompt": "active code silver"},
        )
        assert status == 403
        assert_envelope(bounded_policy_error, expect_ok=False)
        bounded_children_after_policy = len(list((out_dir / "runtime_children").glob("*"))) if (out_dir / "runtime_children").exists() else 0
        if bounded_children_after_policy != bounded_children_before:
            raise ServiceError("POLICY_REJECTION_SIDE_EFFECT_LEAK", "policy rejection created bounded chat child side effects")
        metrics["policy_rejection_has_no_child_side_effect"] = True
        write_jsonl(out_dir / "auth_policy_results.jsonl", [bounded_auth_error, bounded_policy_error])

        status, policy_ok = http_json(
            "POST",
            f"{base_url}/v1/policy/check",
            token=token,
            body={"intended_use": "research", "deployment_mode": "local_research"},
        )
        assert status == 200
        assert_envelope(policy_ok, expect_ok=True)
        observations["policy_check_ok"] = True

        bounded_prompt_body = {
            "intended_use": "research",
            "prompt": "active code silver, distractor pocket teal; produce active answer",
            "max_response_tokens": 64,
        }
        status, bounded_ok = http_json(
            "POST",
            f"{base_url}/v1/bounded-chat/infer",
            token=token,
            body=dict(bounded_prompt_body),
            idempotency_key="bounded-chat-ok",
        )
        assert status == 200
        assert_envelope(bounded_ok, expect_ok=True)
        bounded_value = bounded_ok["value"]
        bounded_inference = bounded_value["inference"]
        bounded_metrics = bounded_value["runtime_metrics"]
        if bounded_inference["status"] != "ok":
            raise ServiceError("BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_FAILS", "bounded chat prompt did not return ok")
        if bounded_metrics.get("artifact_hash_verified") is not True:
            raise ServiceError("ARTIFACT_HASH_MISMATCH", "child artifact hash was not verified")
        if bounded_metrics.get("checkpoint_hash_unchanged") is not True:
            raise ServiceError("CHECKPOINT_MUTATION_DETECTED", "child checkpoint hash changed")
        if bounded_metrics.get("train_step_count") != 0:
            raise ServiceError("TRAINING_SIDE_EFFECT_DETECTED", "child runtime reported training")
        if "BOUNDED_CHAT_INFERENCE_RUNTIME_POSITIVE" not in read_json(REPO_ROOT / bounded_value["child_job_path"] / "summary.json").get("verdicts", []):
            raise ServiceError("BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_FAILS", "child summary was not positive")
        child_path = REPO_ROOT / bounded_value["child_job_path"]
        required_child = ["single_inference.json", "runtime_metrics.json", "summary.json", "report.md", "audit_log.jsonl"]
        if not all((child_path / name).exists() for name in required_child):
            raise ServiceError("BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_FAILS", "child runtime artifacts were not preserved")
        metrics["bounded_chat_single_prompt_pass"] = True
        metrics["bounded_chat_json_envelope_pass"] = isinstance(bounded_inference, dict) and "output_text" in bounded_inference
        metrics["bounded_chat_child_084_positive"] = True
        metrics["artifact_hash_verified"] = True
        metrics["checkpoint_hash_unchanged"] = True
        metrics["train_step_count"] = 0
        metrics["timeout_guard_pass"] = bounded_metrics.get("timeout_guard_pass") is True
        metrics["child_runtime_artifacts_preserved"] = True
        observations["bounded_chat_job_id"] = bounded_value["job_id"]
        observations["bounded_chat_child_job_path"] = bounded_value["child_job_path"]
        write_json(out_dir / "bounded_chat_request_response.json", bounded_ok)
        write_smoke_progress(out_dir, "bounded_chat_inference_checked", "completed", job_id=bounded_value["job_id"])

        status, bounded_unsupported = http_json(
            "POST",
            f"{base_url}/v1/bounded-chat/infer",
            token=token,
            body={"intended_use": "research", "prompt": "what is the weather in Budapest tomorrow"},
            idempotency_key="bounded-chat-unsupported",
        )
        assert status == 200
        assert_envelope(bounded_unsupported, expect_ok=True)
        if bounded_unsupported["value"]["inference"]["status"] != "unsupported":
            raise ServiceError("UNSUPPORTED_INPUT_NOT_HANDLED", "unsupported prompt did not preserve unsupported status")
        metrics["unsupported_input_handled"] = True
        write_jsonl(out_dir / "unsupported_input_results.jsonl", [bounded_unsupported])

        oversized = "x" * (int(config["bounded_chat_max_input_chars"]) + 1)
        bad_cases: list[tuple[str, dict[str, Any] | None, str | None]] = [
            ("missing_prompt", {"intended_use": "research"}, None),
            ("non_string_prompt", {"intended_use": "research", "prompt": 7}, None),
            ("empty_prompt", {"intended_use": "research", "prompt": ""}, None),
            ("whitespace_prompt", {"intended_use": "research", "prompt": "   "}, None),
            ("oversized_prompt", {"intended_use": "research", "prompt": oversized}, None),
            (
                "invalid_max_response_tokens",
                {"intended_use": "research", "prompt": "active code silver", "max_response_tokens": 0},
                None,
            ),
            ("malformed_json", None, '{"prompt": '),
        ]
        bad_results: list[dict[str, Any]] = []
        for name, body, raw in bad_cases:
            if raw is not None:
                status, response = http_raw_json(
                    "POST",
                    f"{base_url}/v1/bounded-chat/infer",
                    token=token,
                    raw_body=raw,
                )
            else:
                status, response = http_json(
                    "POST",
                    f"{base_url}/v1/bounded-chat/infer",
                    token=token,
                    body=body,
                )
            assert status in {400, 403}
            assert_envelope(response, expect_ok=False)
            bad_results.append({"case": name, "status": status, "response": response})
        metrics["bad_input_handled"] = True
        write_jsonl(out_dir / "bad_input_results.jsonl", bad_results)

        status, bounded_same_idem = http_json(
            "POST",
            f"{base_url}/v1/bounded-chat/infer",
            token=token,
            body=dict(bounded_prompt_body),
            idempotency_key="bounded-chat-ok",
        )
        assert status == 200
        assert_envelope(bounded_same_idem, expect_ok=True)
        if bounded_same_idem["value"]["job_id"] != bounded_value["job_id"]:
            raise ServiceError("IDEMPOTENCY_CONFLICT_NOT_DETECTED", "same idempotency key did not reuse bounded chat job")
        metrics["idempotency_reuse_pass"] = True

        status, bounded_idem_conflict = http_json(
            "POST",
            f"{base_url}/v1/bounded-chat/infer",
            token=token,
            body={"intended_use": "research", "prompt": "active code teal, produce active answer"},
            idempotency_key="bounded-chat-ok",
        )
        assert status == 409
        assert_envelope(bounded_idem_conflict, expect_ok=False)
        if bounded_idem_conflict["error"]["code"] != "IDEMPOTENCY_CONFLICT":
            raise ServiceError("IDEMPOTENCY_CONFLICT_NOT_DETECTED", "bounded chat idempotency conflict code missing")
        metrics["idempotency_conflict_pass"] = True

        status, job_response = http_json(
            "POST",
            f"{base_url}/v1/jobs",
            token=token,
            body={"operation": "healthcheck", "intended_use": "research"},
            idempotency_key="job-healthcheck",
        )
        assert status == 200
        assert_envelope(job_response, expect_ok=True)
        job_value = job_response["value"]
        if job_value["job_created_at"] < service_smoke_start:
            raise ServiceError("STALE_JOB_ARTIFACT_USED", "job_created_at predates smoke start")
        if job_value["child_exit_code"] != 0 or not job_value["child_command"]:
            raise ServiceError("STALE_JOB_ARTIFACT_USED", "child command or exit code missing")
        observations["job_id"] = job_value["job_id"]

        status, job_status = http_json("GET", f"{base_url}/v1/jobs/{job_value['job_id']}", token=token)
        assert status == 200
        assert_envelope(job_status, expect_ok=True)

        route_jobs: dict[str, str] = {}
        for route, body, key in [
            ("infer", {"intended_use": "research", "inputs": ["alpha", "beta"]}, "idem-infer"),
            ("evaluate", {"intended_use": "research", "eval_suite_ref": "alpha_eval"}, "idem-evaluate"),
            ("visual-export", {"intended_use": "research", "export_config": {"schema": "visual_snapshot_v1"}}, "idem-visual"),
        ]:
            status, response = http_json(
                "POST",
                f"{base_url}/v1/{route}",
                token=token,
                body=body,
                idempotency_key=key,
            )
            assert status == 200
            assert_envelope(response, expect_ok=True)
            route_jobs[route] = response["value"]["job_id"]
        observations["route_jobs"] = route_jobs

        status, same_idem = http_json(
            "POST",
            f"{base_url}/v1/infer",
            token=token,
            body={"intended_use": "research", "inputs": ["alpha", "beta"]},
            idempotency_key="idem-infer",
        )
        assert status == 200
        assert same_idem["value"]["job_id"] == route_jobs["infer"]
        observations["idempotency_reuse"] = True

        status, idem_conflict = http_json(
            "POST",
            f"{base_url}/v1/infer",
            token=token,
            body={"intended_use": "research", "inputs": ["different"]},
            idempotency_key="idem-infer",
        )
        assert status == 409
        assert_envelope(idem_conflict, expect_ok=False)
        if idem_conflict["error"]["code"] != "IDEMPOTENCY_CONFLICT":
            raise ServiceError("IDEMPOTENCY_CONFLICT_NOT_DETECTED", "idempotency conflict code missing")
        observations["idempotency_conflict"] = True

        status, artifact = http_json(
            "GET",
            f"{base_url}/v1/artifacts/{route_jobs['infer']}/job_manifest.json",
            token=token,
        )
        assert status == 200
        assert_envelope(artifact, expect_ok=True)
        observations["artifact_allowlist_accept"] = True

        for bad_name in ["..%2FLICENSE", "instnct_service_alpha.py", "unregistered.json"]:
            status, bad_artifact = http_json(
                "GET",
                f"{base_url}/v1/artifacts/{route_jobs['infer']}/{bad_name}",
                token=token,
            )
            assert status == 403
            assert_envelope(bad_artifact, expect_ok=False)
        observations["artifact_allowlist_reject"] = True
        metrics["existing_062_routes_preserved"] = True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    rate_config = dict(config)
    rate_config["port"] = 0
    rate_config["out_dir"] = str((out_dir / "rate_limit").relative_to(REPO_ROOT))
    rate_config["rate_limit_max_requests"] = 1
    rate_server = make_server(rate_config, service_smoke_start)
    rate_thread = threading.Thread(target=rate_server.serve_forever, daemon=True)
    rate_thread.start()
    rate_host, rate_port = rate_server.server_address
    rate_url = f"http://{rate_host}:{rate_port}"
    try:
        status, first = http_json("GET", f"{rate_url}/v1/health")
        assert status == 200
        assert_envelope(first, expect_ok=True)
        status, exceeded = http_json("GET", f"{rate_url}/v1/health")
        assert status == 429
        assert_envelope(exceeded, expect_ok=False)
        if (
            exceeded.get("rate_limit_policy") != RATE_LIMIT_POLICY
            or "retry_after" not in exceeded
            or "retry_after" not in exceeded.get("rate_limit", {})
        ):
            raise ServiceError("RATE_LIMIT_BOUNDARY_MISSING", "rate limit metadata missing")
        observations["rate_limit_exceeded"] = True
        metrics["rate_limit_metadata_present"] = True
        write_json(out_dir / "rate_limit_report.json", {"first": first, "exceeded": exceeded})
    finally:
        rate_server.shutdown()
        rate_server.server_close()
        rate_thread.join(timeout=5)

    public_config = dict(config)
    public_config["bind_host"] = "0.0.0.0"
    try:
        validate_bind_host(public_config["bind_host"])
        raise ServiceError("PUBLIC_BIND_DETECTED", "public bind was not rejected")
    except ServiceError as err:
        if err.code != "PUBLIC_BIND_DETECTED":
            raise
    observations["public_bind_rejected"] = True
    metrics["public_bind_rejected"] = True

    child_manifest = {
        "schema_version": SERVICE_SCHEMA_VERSION,
        "bounded_chat_runtime_out_root": config["bounded_chat_runtime_out_root"],
        "successful_child_job_path": observations.get("bounded_chat_child_job_path"),
        "required_child_artifacts": ["single_inference.json", "runtime_metrics.json", "summary.json", "report.md", "audit_log.jsonl"],
    }
    write_json(out_dir / "child_runtime_manifest.json", child_manifest)
    audit_rows = read_jsonl(out_dir / "audit_log.jsonl")
    metrics["audit_log_written"] = any(row.get("route") == "/v1/bounded-chat/infer" for row in audit_rows)
    metrics["service_api_alpha_only"] = True
    metrics["production_chat_claimed"] = False
    metrics["gpt_like_assistant_readiness_claimed"] = False
    metrics["sdk_surface_exposed"] = False
    metrics["deployment_harness_mutated"] = False
    write_json(out_dir / "service_metrics.json", metrics)

    summary = {
        "schema_version": SERVICE_SCHEMA_VERSION,
        "service_api_alpha_smoke_pass": True,
        "bounded_chat_service_api_alpha_smoke_pass": True,
        "service_smoke_start": service_smoke_start,
        "observations": observations,
        "service_metrics": metrics,
        "verdicts": [
            "SERVICE_API_ALPHA_POSITIVE",
            "API_V1_ALPHA_SCHEMA_DEFINED",
            "LOCALHOST_SERVICE_ALPHA_POSITIVE",
            "LOCALHOST_BIND_RESTRICTED",
            "AUTHZ_BOUNDARY_ALPHA_DEFINED",
            "AUTHZ_SIDE_EFFECT_GUARD_POSITIVE",
            "JOB_ORCHESTRATION_ALPHA_POSITIVE",
            "ARTIFACT_RETRIEVAL_ALPHA_POSITIVE",
            "ARTIFACT_ALLOWLIST_POSITIVE",
            "IDEMPOTENCY_ALPHA_POSITIVE",
            "RATE_LIMIT_BOUNDARY_DEFINED",
            "POLICY_SIDE_EFFECT_GUARD_POSITIVE",
            "POLICY_GUARD_REJECTS_REGULATED_SERVICE_REQUESTS",
            "API_ERROR_ENVELOPE_POSITIVE",
            "PROGRESS_AUDIT_WRITEOUT_POSITIVE",
            "BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE",
            "LOCALHOST_BIND_RESTRICTED",
            "AUTH_GUARD_PASSES",
            "POLICY_GUARD_PASSES",
            "RATE_LIMIT_METADATA_PASSES",
            "BOUNDED_CHAT_ROUTE_REGISTERED",
            "BOUNDED_CHAT_INFERENCE_CHILD_RUNTIME_PASSES",
            "ARTIFACT_PACKAGE_VERIFIED_BY_CHILD",
            "CHECKPOINT_UNCHANGED",
            "JSON_RESPONSE_ENVELOPE_PASSES",
            "BAD_INPUT_HANDLED",
            "UNSUPPORTED_INPUT_HANDLED",
            "AUDIT_LOG_WRITTEN",
            "NO_TRAINING_PERFORMED",
            "PRODUCTION_CHAT_NOT_CLAIMED",
            "PRODUCTION_API_READY_NOT_CLAIMED",
            "PUBLIC_BETA_NOT_CLAIMED",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(
        "# STABLE_LOOP_PHASE_LOCK_062_SERVICE_API_ALPHA Smoke Report\n\n"
        "# STABLE_LOOP_PHASE_LOCK_085_BOUNDED_CHAT_SERVICE_API_ALPHA Smoke Report\n\n"
        "Status: positive localhost-only service/API alpha smoke with bounded chat route.\n\n"
        "085 is service API alpha only, localhost/private only. It is not deploy-ready service, "
        "not public API, not SDK surface, not GPT-like assistant, not open-domain chat, "
        "not production chat, not safety alignment, not public beta / GA / hosted SaaS.\n\n"
        "The bounded chat route is `POST /v1/bounded-chat/infer` and it delegates inference "
        "to the 084 Rust CLI child runtime.\n\n"
        "This smoke validates local/private API alpha infrastructure only. "
        "It is no production API readiness, no hosted SaaS, no public beta, "
        "no multi-tenant IAM, no clinical use, no high-stakes education use, "
        "no commercial launch, no full VRAXION, no language grounding, "
        "no consciousness, no biological/FlyWire equivalence, and no physical quantum behavior.\n",
        encoding="utf-8",
    )
    write_smoke_progress(out_dir, "final", "completed", verdict="BOUNDED_CHAT_SERVICE_API_ALPHA_POSITIVE")
    return summary


def command_healthcheck(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config), args.out)
    result = {
        "check_pass": True,
        "schema_version": SERVICE_SCHEMA_VERSION,
        "bind_host": config["bind_host"],
        "routes": ROUTES,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def command_serve(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config), args.out)
    server = make_server(config)
    host, port = server.server_address
    print(json.dumps({"serving": True, "bind_host": host, "port": port, "schema_version": SERVICE_SCHEMA_VERSION}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config), args.out)
    out_dir = resolve_out_dir(str(config["out_dir"]))
    summary = run_smoke(config, out_dir)
    print(json.dumps({"check_pass": True, "summary": summary}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["healthcheck", "serve", "smoke"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--out")
    args = parser.parse_args()
    handlers = {"healthcheck": command_healthcheck, "serve": command_serve, "smoke": command_smoke}
    try:
        return handlers[args.command](args)
    except ServiceError as err:
        print(
            json.dumps(
                {
                    "check_pass": False,
                    "schema_version": SERVICE_SCHEMA_VERSION,
                    "error": {
                        "code": err.code,
                        "message": err.message,
                        "retryable": err.retryable,
                        "details": err.details,
                    },
                    "claim_boundary": CLAIM_BOUNDARY,
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
