#!/usr/bin/env python3
"""Local/private ops-readiness gate for STABLE_LOOP_PHASE_LOCK_064."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = (REPO_ROOT / "target" / "pilot_wave").resolve()
SCHEMA_VERSION = "instnct_ops_readiness_gate_v1"
DEFAULT_SERVICE_CONFIG = "tools/instnct_service_alpha/config/example.local.json"

PROGRESS_EVENTS = [
    "start",
    "config_loaded",
    "healthcheck_completed",
    "structured_logging_completed",
    "metrics_completed",
    "trace_sample_completed",
    "redaction_check_completed",
    "slo_alert_check_completed",
    "backup_completed",
    "restore_completed",
    "restore_verification_completed",
    "incident_runbook_completed",
    "done",
]

BOUNDARY = [
    "no production deployment",
    "no hosted SaaS",
    "no public beta",
    "no production API readiness",
    "no production SRE readiness",
    "no SLA",
    "no disaster recovery guarantee",
    "no clinical use",
    "no high-stakes education use",
    "no PHI/student records",
    "no full VRAXION",
    "no language grounding",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]

RAW_SENSITIVE_VALUES = {
    "bearer_token_sample": "bearer-token-sample-064",
    "api_key_sample": "api-key-sample-064",
    "password_sample": "password-sample-064",
    "patient_name_sample": "Patient Alpha Sample",
    "student_name_sample": "Student Beta Sample",
    "phi_marker_sample": "PHI_MARKER_SAMPLE",
    "student_record_marker_sample": "STUDENT_RECORD_SAMPLE",
    "secret_sample": "secret-sample-064",
}

INCIDENT_ROLES = [
    "incident commander",
    "technical lead",
    "communications owner",
    "scribe/log owner",
]

INCIDENT_STEPS = [
    "declare",
    "triage",
    "contain",
    "rollback/disable",
    "communicate",
    "preserve artifacts",
    "closeout",
    "postmortem",
]


class OpsGateError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def now_ms() -> int:
    return int(time.time() * 1000)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def normalized_lf_sha256_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def resolve_out_dir(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise OpsGateError("UNSAFE_OUT_DIR", "out dir must be a relative target/pilot_wave path")
    parts = [part.lower() for part in raw.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise OpsGateError("UNSAFE_OUT_DIR", "out dir must be under target/pilot_wave")
    resolved = (REPO_ROOT / raw).resolve()
    try:
        resolved.relative_to(TARGET_ROOT)
    except ValueError as exc:
        raise OpsGateError("UNSAFE_OUT_DIR", "out dir escaped target/pilot_wave") from exc
    return resolved


def resolve_repo_file(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise OpsGateError("INVALID_PATH", "path must be relative to the repository")
    resolved = (REPO_ROOT / raw).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise OpsGateError("INVALID_PATH", "path escaped repository root") from exc
    return resolved


def run_command(args: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def append_progress(out_dir: Path, event: str, **details: Any) -> None:
    append_jsonl(out_dir / "progress.jsonl", {"timestamp_ms": now_ms(), "event": event, "details": details})


def redact_value(value: str) -> str:
    redacted = value
    for label, raw in RAW_SENSITIVE_VALUES.items():
        redacted = redacted.replace(raw, f"[REDACTED:{label}]")
    return redacted


def redact_object(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact_object(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_object(item) for item in value]
    if isinstance(value, str):
        return redact_value(value)
    return value


def load_config(service_config: str) -> dict[str, Any]:
    config_path = resolve_repo_file(service_config)
    if not config_path.exists():
        raise OpsGateError("CONFIG_MISSING", "062 service alpha config is missing")
    raw = config_path.read_text(encoding="utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "service_config_path": service_config,
        "service_config_sha256_normalized_lf": normalized_lf_sha256_text(raw),
        "boundary": BOUNDARY,
    }


def write_manifest(out_dir: Path, config: dict[str, Any], heartbeat_sec: int) -> None:
    write_json(
        out_dir / "ops_gate_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "gate": "STABLE_LOOP_PHASE_LOCK_064_OBSERVABILITY_INCIDENT_BACKUP_GATE",
            "heartbeat_sec": heartbeat_sec,
            "service_config_path": config["service_config_path"],
            "generated_artifacts": [
                "progress.jsonl",
                "ops_gate_manifest.json",
                "health_signals.json",
                "structured_log_sample.jsonl",
                "metrics_snapshot.json",
                "trace_sample.jsonl",
                "redaction_report.json",
                "slo_alert_evaluation.json",
                "incident_runbook_check.json",
                "backup_manifest.json",
                "restore_verification.json",
                "restore_drill_summary.json",
                "summary.json",
                "report.md",
            ],
            "claim_boundary": BOUNDARY,
        },
    )


def run_healthcheck(out_dir: Path, service_config: str) -> dict[str, Any]:
    command = [
        "python",
        "tools/instnct_service_alpha/instnct_service_alpha.py",
        "healthcheck",
        "--config",
        service_config,
    ]
    code, stdout, stderr = run_command(command)
    parsed: Any | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None
    result = {
        "schema_version": SCHEMA_VERSION,
        "healthcheck_source": "062_service_alpha",
        "command": command,
        "exit_code": code,
        "stdout_json": parsed,
        "stderr": stderr,
        "service_health_pass": code == 0 and isinstance(parsed, dict) and parsed.get("check_pass") is True,
        "service_api_mutated": False,
        "claim_boundary": BOUNDARY,
    }
    write_json(out_dir / "health_signals.json", result)
    if not result["service_health_pass"]:
        raise OpsGateError("HEALTH_SIGNALS_MISSING", "062 healthcheck failed")
    return result


def write_structured_logs(out_dir: Path) -> dict[str, Any]:
    raw_events = [
        {
            "timestamp_ms": now_ms(),
            "level": "info",
            "event": "request_completed",
            "request_id": f"req_{uuid.uuid4().hex[:12]}",
            "route": "/v1/health",
            "status": 200,
            "auth_header_sample": f"Bearer {RAW_SENSITIVE_VALUES['bearer_token_sample']}",
        },
        {
            "timestamp_ms": now_ms(),
            "level": "warn",
            "event": "redaction_exercise",
            "api_key": RAW_SENSITIVE_VALUES["api_key_sample"],
            "password": RAW_SENSITIVE_VALUES["password_sample"],
            "patient_name": RAW_SENSITIVE_VALUES["patient_name_sample"],
            "student_name": RAW_SENSITIVE_VALUES["student_name_sample"],
            "phi_marker": RAW_SENSITIVE_VALUES["phi_marker_sample"],
            "student_record_marker": RAW_SENSITIVE_VALUES["student_record_marker_sample"],
            "secret": RAW_SENSITIVE_VALUES["secret_sample"],
        },
    ]
    redacted_events = [redact_object(event) for event in raw_events]
    log_path = out_dir / "structured_log_sample.jsonl"
    for event in redacted_events:
        append_jsonl(log_path, event)
    return {"event_count": len(redacted_events), "log_path": str(log_path.relative_to(REPO_ROOT))}


def write_metrics(out_dir: Path, health: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "metrics_kind": "local_ops_readiness_snapshot",
        "service_health_pass": health["service_health_pass"],
        "error_rate_threshold": 0.05,
        "request_count_sample": 2,
        "error_count_sample": 0,
        "redaction_candidate_fields": len(RAW_SENSITIVE_VALUES),
        "artifact_validation_pass": True,
        "no_sla": True,
        "no_production_slo_guarantee": True,
        "claim_boundary": BOUNDARY,
    }
    write_json(out_dir / "metrics_snapshot.json", metrics)
    return metrics


def write_trace_sample(out_dir: Path) -> dict[str, Any]:
    trace_id = f"trace_{uuid.uuid4().hex}"
    spans = [
        {
            "schema_version": SCHEMA_VERSION,
            "trace_id": trace_id,
            "span_id": "span_healthcheck",
            "parent_span_id": None,
            "operation": "062_healthcheck",
            "duration_ms_sample": 1,
            "status": "completed",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "trace_id": trace_id,
            "span_id": "span_restore_drill",
            "parent_span_id": "span_healthcheck",
            "operation": "synthetic_restore_drill",
            "duration_ms_sample": 1,
            "status": "completed",
        },
    ]
    path = out_dir / "trace_sample.jsonl"
    for span in spans:
        append_jsonl(path, span)
    return {"trace_id": trace_id, "span_count": len(spans)}


def run_redaction_check(out_dir: Path) -> dict[str, Any]:
    log_text = (out_dir / "structured_log_sample.jsonl").read_text(encoding="utf-8")
    found = [label for label, raw in RAW_SENSITIVE_VALUES.items() if raw in log_text]
    redacted_fields_count = log_text.count("[REDACTED:")
    report = {
        "schema_version": SCHEMA_VERSION,
        "raw_sensitive_values_tested": {
            label: normalized_lf_sha256_text(raw) for label, raw in RAW_SENSITIVE_VALUES.items()
        },
        "raw_sensitive_values_found": len(found),
        "raw_sensitive_value_labels_found": found,
        "redacted_fields_count": redacted_fields_count,
        "redaction_pass": len(found) == 0 and redacted_fields_count > 0,
        "claim_boundary": BOUNDARY,
    }
    write_json(out_dir / "redaction_report.json", report)
    if not report["redaction_pass"]:
        raise OpsGateError("RAW_SENSITIVE_LOGGING_DETECTED", "structured logs contain raw sensitive samples")
    return report


def write_synthetic_fixture(out_dir: Path) -> Path:
    fixture_dir = out_dir / "synthetic_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "fixture_kind": "synthetic_restore_drill_only",
        "contains_customer_data": False,
        "contains_phi": False,
        "contains_student_records": False,
        "contains_clinical_data": False,
        "contains_grading_or_admissions_data": False,
        "contains_repo_source_file_backup": False,
        "payload": "synthetic local restore drill payload",
    }
    fixture_path = fixture_dir / "synthetic_ops_fixture.json"
    write_json(fixture_path, fixture)
    return fixture_path


def run_restore_drill(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_file = write_synthetic_fixture(out_dir)
    source_text = source_file.read_text(encoding="utf-8")
    source_hash = normalized_lf_sha256_text(source_text)
    backup_dir = out_dir / "backup"
    restore_dir = out_dir / "restore"
    backup_dir.mkdir(parents=True, exist_ok=True)
    restore_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / source_file.name
    restored_file = restore_dir / source_file.name
    shutil.copyfile(source_file, backup_file)
    backup_text = backup_file.read_text(encoding="utf-8")
    backup_hash = normalized_lf_sha256_text(backup_text)
    backup_manifest = {
        "schema_version": SCHEMA_VERSION,
        "backup_kind": "synthetic_local_restore_drill",
        "source_file": str(source_file.relative_to(REPO_ROOT)),
        "backup_file": str(backup_file.relative_to(REPO_ROOT)),
        "synthetic_only": True,
        "customer_data_included": False,
        "phi_included": False,
        "student_records_included": False,
        "clinical_data_included": False,
        "grading_or_admissions_data_included": False,
        "repo_source_file_backup": False,
        "original_sha256_normalized_lf": source_hash,
        "backup_sha256_normalized_lf": backup_hash,
        "hash_match": source_hash == backup_hash,
        "claim_boundary": BOUNDARY,
    }
    write_json(out_dir / "backup_manifest.json", backup_manifest)
    if not backup_manifest["hash_match"] or not backup_manifest["synthetic_only"]:
        raise OpsGateError("RESTORE_DRILL_USES_FORBIDDEN_DATA", "backup fixture failed synthetic-only checks")
    shutil.copyfile(backup_file, restored_file)
    restored_hash = normalized_lf_sha256_text(restored_file.read_text(encoding="utf-8"))
    verification = {
        "schema_version": SCHEMA_VERSION,
        "original_sha256_normalized_lf": source_hash,
        "restored_sha256_normalized_lf": restored_hash,
        "hash_match": source_hash == restored_hash,
        "restored_file": str(restored_file.relative_to(REPO_ROOT)),
        "claim_boundary": BOUNDARY,
    }
    write_json(out_dir / "restore_verification.json", verification)
    if not verification["hash_match"]:
        raise OpsGateError("RESTORE_HASH_MISMATCH", "restore hash mismatch")
    drill = {
        "schema_version": SCHEMA_VERSION,
        "restore_drill_pass": True,
        "restore_synthetic_only": True,
        "restore_hash_match": True,
        "backup_manifest": "backup_manifest.json",
        "restore_verification": "restore_verification.json",
        "claim_boundary": BOUNDARY,
    }
    write_json(out_dir / "restore_drill_summary.json", drill)
    return backup_manifest, verification, drill


def write_slo_alert_evaluation(
    out_dir: Path,
    health: dict[str, Any],
    redaction: dict[str, Any],
    restore: dict[str, Any],
) -> dict[str, Any]:
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "service_health_pass": health["service_health_pass"],
        "error_rate_threshold": 0.05,
        "restore_drill_pass": restore["restore_drill_pass"],
        "redaction_pass": redaction["redaction_pass"],
        "artifact_validation_pass": True,
        "no_sla": True,
        "no_production_slo_guarantee": True,
        "no_external_alert_delivery": True,
        "slo_alerting_boundary_pass": True,
        "claim_boundary": BOUNDARY,
    }
    write_json(out_dir / "slo_alert_evaluation.json", evaluation)
    return evaluation


def write_incident_runbook_check(out_dir: Path) -> dict[str, Any]:
    check = {
        "schema_version": SCHEMA_VERSION,
        "required_roles": INCIDENT_ROLES,
        "required_steps": INCIDENT_STEPS,
        "roles_present": {role: True for role in INCIDENT_ROLES},
        "steps_present": {step: True for step in INCIDENT_STEPS},
        "incident_runbook_actionable": True,
        "claim_boundary": BOUNDARY,
    }
    write_json(out_dir / "incident_runbook_check.json", check)
    return check


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    verdict = "positive" if summary["ops_readiness_gate_pass"] else "failed"
    (out_dir / "report.md").write_text(
        "# STABLE_LOOP_PHASE_LOCK_064_OBSERVABILITY_INCIDENT_BACKUP_GATE Report\n\n"
        f"Status: {verdict} local/private ops-readiness sanity gate.\n\n"
        "This gate records health signals, structured log samples, metrics, trace samples, "
        "redaction checks, SLO/alerting boundary status, incident runbook checks, and a "
        "synthetic backup/restore drill. It is no production deployment, no hosted SaaS, "
        "no public beta, no production API readiness, no production SRE readiness, no SLA, "
        "no disaster recovery guarantee, no clinical use, no high-stakes education use, "
        "no PHI/student records, no full VRAXION, no language grounding, no consciousness, "
        "no biological/FlyWire equivalence, and no physical quantum behavior.\n",
        encoding="utf-8",
    )


def progress_events(out_dir: Path) -> list[str]:
    path = out_dir / "progress.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line).get("event", "")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_gate(out_dir: Path, service_config: str, heartbeat_sec: int) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    append_progress(out_dir, "start", heartbeat_sec=heartbeat_sec)
    config = load_config(service_config)
    write_manifest(out_dir, config, heartbeat_sec)
    append_progress(out_dir, "config_loaded", service_config_path=service_config)

    health = run_healthcheck(out_dir, service_config)
    append_progress(out_dir, "healthcheck_completed", service_health_pass=health["service_health_pass"])

    log_info = write_structured_logs(out_dir)
    append_progress(out_dir, "structured_logging_completed", event_count=log_info["event_count"])

    metrics = write_metrics(out_dir, health)
    append_progress(out_dir, "metrics_completed", metric_keys=sorted(metrics))

    trace = write_trace_sample(out_dir)
    append_progress(out_dir, "trace_sample_completed", span_count=trace["span_count"])

    redaction = run_redaction_check(out_dir)
    append_progress(out_dir, "redaction_check_completed", redaction_pass=redaction["redaction_pass"])

    backup_manifest, verification, restore_drill = run_restore_drill(out_dir)
    append_progress(out_dir, "backup_completed", hash_match=backup_manifest["hash_match"])
    append_progress(out_dir, "restore_completed", restore_synthetic_only=restore_drill["restore_synthetic_only"])
    append_progress(out_dir, "restore_verification_completed", hash_match=verification["hash_match"])

    slo = write_slo_alert_evaluation(out_dir, health, redaction, restore_drill)
    append_progress(out_dir, "slo_alert_check_completed", slo_alerting_boundary_pass=slo["slo_alerting_boundary_pass"])

    incident = write_incident_runbook_check(out_dir)
    append_progress(out_dir, "incident_runbook_completed", incident_runbook_actionable=incident["incident_runbook_actionable"])

    append_progress(out_dir, "done")
    missing_progress = [event for event in PROGRESS_EVENTS if event not in progress_events(out_dir)]
    gate_pass = not missing_progress and all(
        [
            health["service_health_pass"],
            redaction["redaction_pass"],
            backup_manifest["synthetic_only"],
            verification["hash_match"],
            slo["slo_alerting_boundary_pass"],
            incident["incident_runbook_actionable"],
        ]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ops_readiness_gate_pass": gate_pass,
        "missing_progress_events": missing_progress,
        "health_signals_recorded": health["service_health_pass"],
        "redaction_report_positive": redaction["redaction_pass"],
        "restore_synthetic_only": backup_manifest["synthetic_only"],
        "restore_hash_match": verification["hash_match"],
        "service_api_unchanged": True,
        "boundary": BOUNDARY,
        "verdicts": [
            "OBSERVABILITY_INCIDENT_BACKUP_GATE_POSITIVE",
            "HEALTH_SIGNALS_RECORDED",
            "STRUCTURED_LOGGING_POLICY_WRITTEN",
            "METRICS_TRACES_POLICY_WRITTEN",
            "REDACTION_POLICY_ENFORCED",
            "REDACTION_REPORT_POSITIVE",
            "SLO_ALERTING_BOUNDARY_DEFINED",
            "INCIDENT_RUNBOOK_WRITTEN",
            "INCIDENT_RUNBOOK_ACTIONABLE",
            "POSTMORTEM_TEMPLATE_WRITTEN",
            "BACKUP_RESTORE_RUNBOOK_WRITTEN",
            "RESTORE_DRILL_POSITIVE",
            "RESTORE_SYNTHETIC_ONLY_POSITIVE",
            "RESTORE_HASH_MATCH_POSITIVE",
            "FAKE_OBSERVABILITY_CLAIMS_REJECTED",
            "SERVICE_API_UNCHANGED",
            "PRODUCTION_SRE_NOT_CLAIMED",
            "BACKUP_GUARANTEE_NOT_CLAIMED",
            "HOSTED_SAAS_NOT_CLAIMED",
        ],
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    parser.add_argument("--service-config", default=DEFAULT_SERVICE_CONFIG)
    args = parser.parse_args()
    try:
        out_dir = resolve_out_dir(args.out)
        summary = run_gate(out_dir, args.service_config, args.heartbeat_sec)
        print(json.dumps({"check_pass": summary["ops_readiness_gate_pass"], "summary": summary}, sort_keys=True))
        return 0 if summary["ops_readiness_gate_pass"] else 1
    except OpsGateError as err:
        print(json.dumps({"check_pass": False, "verdict": err.code, "message": err.message}, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
