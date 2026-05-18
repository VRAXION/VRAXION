#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_064 ops-readiness gate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = [
    "docs/product/INSTNCT_OBSERVABILITY_INCIDENT_BACKUP_GATE.md",
    "docs/product/INSTNCT_OBSERVABILITY_POLICY.md",
    "docs/product/INSTNCT_STRUCTURED_LOGGING_POLICY.md",
    "docs/product/INSTNCT_METRICS_TRACES_POLICY.md",
    "docs/product/INSTNCT_REDACTION_POLICY.md",
    "docs/product/INSTNCT_SLO_ALERTING_POLICY.md",
    "docs/product/INSTNCT_INCIDENT_RESPONSE_RUNBOOK.md",
    "docs/product/INSTNCT_POSTMORTEM_TEMPLATE.md",
    "docs/product/INSTNCT_BACKUP_RESTORE_RUNBOOK.md",
    "docs/product/INSTNCT_RESTORE_DRILL_POLICY.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_064_OBSERVABILITY_INCIDENT_BACKUP_GATE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_064_OBSERVABILITY_INCIDENT_BACKUP_GATE_RESULT.md",
    "tools/ops_readiness/README.md",
]

REQUIRED_SOURCE = [
    "tools/ops_readiness/instnct_ops_gate.py",
    "tools/ops_readiness/ops_gate.ps1",
    "tools/ops_readiness/config/example.local.json",
]

EXACT_COMMANDS = [
    "python -m py_compile tools/ops_readiness/instnct_ops_gate.py",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_064_observability_incident_backup_check.py",
    "python tools/ops_readiness/instnct_ops_gate.py --out target/pilot_wave/stable_loop_phase_lock_064_observability_incident_backup_gate/smoke --heartbeat-sec 20",
    "python scripts/probes/run_stable_loop_phase_lock_064_observability_incident_backup_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_063_security_supply_chain_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
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

REQUIRED_TERMS = [
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
    "raw_sensitive_values_tested",
    "raw_sensitive_values_found = 0",
    "redacted_fields_count > 0",
    "original_sha256_normalized_lf",
    "restored_sha256_normalized_lf",
    "hash_match = true",
    "service_health_pass",
    "error_rate_threshold",
    "restore_drill_pass",
    "redaction_pass",
    "artifact_validation_pass",
    "no production SLO guarantee",
    "no external alert delivery",
]

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

SOURCE_TERMS = [
    "RAW_SENSITIVE_VALUES",
    "raw_sensitive_values_found",
    "redacted_fields_count",
    "original_sha256_normalized_lf",
    "restored_sha256_normalized_lf",
    "synthetic_only",
    "service_api_unchanged",
    "instnct_service_alpha.py",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "FAKE_OBSERVABILITY_CLAIM_DETECTED": [
        "OpenTelemetry compliance",
        "OpenTelemetry compliant",
        "SOC2 readiness",
        "HIPAA readiness",
        "FERPA readiness",
        "production monitoring",
    ],
    "PRODUCTION_SRE_CLAIM_DETECTED": [
        "production SRE readiness",
        "production SRE ready",
    ],
    "BACKUP_GUARANTEE_CLAIM_DETECTED": [
        "disaster recovery guarantee",
        "backup guarantee",
    ],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "draft"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAMES = [
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
]

POSITIVE_VERDICTS = [
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
]


def read_files() -> tuple[list[str], dict[str, str]]:
    missing: list[str] = []
    docs: dict[str, str] = {}
    for rel in REQUIRED_DOCS + REQUIRED_SOURCE:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if rel in REQUIRED_DOCS and len(text.strip()) < 200:
            missing.append(f"{rel}:too_short")
        docs[rel] = text
    return missing, docs


def git_status(paths: list[str]) -> str:
    result = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def root_license_changed() -> bool:
    return bool(git_status(["LICENSE"]))


def service_api_mutation_detected() -> bool:
    return bool(git_status(["tools/instnct_service_alpha", "instnct-core"]))


def placeholder_hits(docs: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in docs.items():
        if rel.endswith(".py") or rel.endswith(".ps1") or rel.endswith(".json"):
            continue
        for marker in PLACEHOLDERS:
            if re.search(rf"\b{re.escape(marker)}\b", text, flags=re.IGNORECASE):
                hits.append({"file": rel, "marker": marker})
    return hits


def line_is_negated(line: str, phrase: str) -> bool:
    phrase_start = line.find(phrase.lower())
    if phrase_start < 0:
        return False
    prefix = line[max(0, phrase_start - 80) : phrase_start]
    return any(marker in prefix for marker in NEGATION_MARKERS)


def forbidden_claim_hits(docs: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in docs.items():
        if rel.endswith(".py") or rel.endswith(".ps1") or rel.endswith(".json"):
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            lower = line.lower()
            for verdict, phrases in FORBIDDEN_CLAIMS.items():
                for phrase in phrases:
                    if phrase.lower() in lower and not line_is_negated(lower, phrase):
                        hits.append({"file": rel, "line": idx, "phrase": phrase, "verdict": verdict})
    return hits


def missing_commands(docs: dict[str, str]) -> list[str]:
    text = "\n".join(docs.values())
    return [command for command in EXACT_COMMANDS if command not in text]


def missing_boundary_tokens(docs: dict[str, str]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for rel in REQUIRED_DOCS:
        text = docs.get(rel, "")
        lower = text.lower()
        for token in BOUNDARY_TOKENS:
            if token.lower() not in lower:
                missing.append({"file": rel, "token": token})
    return missing


def missing_required_terms(docs: dict[str, str]) -> list[str]:
    text = "\n".join(docs.values())
    lower = text.lower()
    missing = [term for term in REQUIRED_TERMS if term.lower() not in lower]
    for event in PROGRESS_EVENTS:
        if event.lower() not in lower:
            missing.append(f"progress:{event}")
    for role in INCIDENT_ROLES:
        if role.lower() not in lower:
            missing.append(f"role:{role}")
    for step in INCIDENT_STEPS:
        if step.lower() not in lower:
            missing.append(f"step:{step}")
    source = docs.get("tools/ops_readiness/instnct_ops_gate.py", "")
    for term in SOURCE_TERMS:
        if term not in source:
            missing.append(f"source:{term}")
    return missing


def generated_artifact_staged() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    hits: list[str] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        path = raw_line[3:].replace("\\", "/")
        lower = path.lower()
        if any(part in lower for part in GENERATED_PATH_PARTS):
            hits.append(path)
            continue
        if any(lower.endswith(suffix) for suffix in GENERATED_SUFFIXES):
            hits.append(path)
            continue
        if any(name in path for name in GENERATED_NAMES):
            hits.append(path)
            continue
        if "service job artifacts" in lower or "restore drill outputs" in lower or "smoke" in lower and lower.startswith("target/"):
            hits.append(path)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        print("Only --check-only is supported.", file=sys.stderr)
        return 2

    missing_docs, docs = read_files()
    placeholders = placeholder_hits(docs)
    commands = missing_commands(docs)
    boundary = missing_boundary_tokens(docs)
    claims = forbidden_claim_hits(docs)
    terms = missing_required_terms(docs)
    generated = generated_artifact_staged()
    root_changed = root_license_changed()
    service_mutation = service_api_mutation_detected()

    check_pass = not any(
        [
            missing_docs,
            placeholders,
            commands,
            boundary,
            claims,
            terms,
            generated,
            root_changed,
            service_mutation,
        ]
    )
    verdicts = POSITIVE_VERDICTS if check_pass else ["OBSERVABILITY_INCIDENT_BACKUP_GATE_FAILS"]
    if service_mutation:
        verdicts.append("SERVICE_API_MUTATION_DETECTED")
    if generated:
        verdicts.append("GENERATED_ARTIFACT_STAGED")
    if root_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    result: dict[str, Any] = {
        "check_pass": check_pass,
        "missing_docs": missing_docs,
        "placeholder_hits": placeholders,
        "missing_commands": commands,
        "missing_boundary_tokens": boundary,
        "forbidden_claim_hits": claims,
        "generated_artifact_staged": generated,
        "root_license_changed": root_changed,
        "missing_required_terms": terms,
        "service_api_mutation_detected": service_mutation,
        "verdicts": verdicts,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
