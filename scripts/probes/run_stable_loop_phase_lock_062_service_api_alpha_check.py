#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_062 service/API alpha."""

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
    "docs/product/INSTNCT_SERVICE_API_ALPHA.md",
    "docs/product/INSTNCT_SERVICE_API_ALPHA_SCHEMA.md",
    "docs/product/INSTNCT_SERVICE_ALPHA_AUTHZ_BOUNDARY.md",
    "docs/product/INSTNCT_SERVICE_ALPHA_JOB_ORCHESTRATION.md",
    "docs/product/INSTNCT_SERVICE_ALPHA_ARTIFACT_RETRIEVAL.md",
    "docs/product/INSTNCT_SERVICE_ALPHA_IDEMPOTENCY_RATE_LIMIT.md",
    "docs/product/INSTNCT_SERVICE_ALPHA_RUNBOOK.md",
    "docs/product/INSTNCT_SERVICE_ALPHA_CLAIM_BOUNDARY.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_062_SERVICE_API_ALPHA_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_062_SERVICE_API_ALPHA_RESULT.md",
    "tools/instnct_service_alpha/README.md",
]

REQUIRED_SOURCE = [
    "tools/instnct_service_alpha/instnct_service_alpha.py",
    "tools/instnct_service_alpha/config/example.local.json",
    "tools/instnct_service_alpha/healthcheck.ps1",
    "tools/instnct_service_alpha/smoke.ps1",
    "tools/instnct_service_alpha/run_service.ps1",
]

EXACT_COMMANDS = [
    "python -m py_compile tools/instnct_service_alpha/instnct_service_alpha.py",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py --check-only",
    "python tools/instnct_service_alpha/instnct_service_alpha.py smoke --config tools/instnct_service_alpha/config/example.local.json --out target/pilot_wave/stable_loop_phase_lock_062_service_api_alpha/smoke",
    "cargo check -p instnct-core --example instnct_sdk_candidate_smoke",
    "cargo test -p instnct-core sdk_candidate",
    "git diff --check",
]

BOUNDARY_TOKENS = [
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

REQUIRED_TERMS = [
    "instnct_service_api_alpha_v1",
    "GET /v1/health",
    "POST /v1/policy/check",
    "POST /v1/jobs",
    "GET /v1/jobs/{job_id}",
    "POST /v1/infer",
    "POST /v1/evaluate",
    "POST /v1/visual-export",
    "GET /v1/artifacts/{job_id}/{artifact_name}",
    "alpha_static_local",
    "idempotency_key",
    "POLICY_GUARD_REJECTED",
    "ARTIFACT_PATH_ESCAPE",
    "IDEMPOTENCY_CONFLICT",
    "rate_limit_remaining",
    "retry_after",
]

REQUIRED_SOURCE_TERMS = [
    'SERVICE_SCHEMA_VERSION = "instnct_service_api_alpha_v1"',
    "ThreadingHTTPServer",
    "subprocess.run(job.child_command",
    "Authorization",
    "POLICY_GUARD_REJECTED",
    "ARTIFACT_PATH_ESCAPE",
    "IDEMPOTENCY_CONFLICT",
    'RATE_LIMIT_POLICY = "alpha_static_local"',
    "PUBLIC_BIND_DETECTED",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "PRODUCTION_API_READY_CLAIM_DETECTED": [
        "production API ready",
        "production API readiness",
        "production ready",
        "production deployment",
    ],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "SERVICE_API_ALPHA_FAILS": [
        "hosted SaaS launch",
        "commercial launch",
        "clinical ready",
        "high-stakes education ready",
    ],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "non-"]

GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["service_smoke", "checkpoint", "smoke/"]

POSITIVE_VERDICTS = [
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
    "PRODUCTION_API_READY_NOT_CLAIMED",
    "PUBLIC_BETA_NOT_CLAIMED",
]


def read_required_files() -> tuple[list[str], dict[str, str]]:
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


def root_license_changed() -> bool:
    status = subprocess.run(
        ["git", "status", "--short", "--", "LICENSE"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(status.stdout.strip())


def placeholder_hits(docs: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in docs.items():
        if rel.endswith(".json") or rel.endswith(".py") or rel.endswith(".ps1"):
            continue
        for marker in PLACEHOLDERS:
            if re.search(rf"\b{re.escape(marker)}\b", text, flags=re.IGNORECASE):
                hits.append({"file": rel, "marker": marker})
    return hits


def line_is_negated(line: str, phrase: str) -> bool:
    phrase_start = line.find(phrase.lower())
    if phrase_start < 0:
        return False
    prefix = line[max(0, phrase_start - 40) : phrase_start]
    return any(marker in prefix for marker in NEGATION_MARKERS) or (
        phrase_start > 0 and line[phrase_start - 1] == "_"
    )


def forbidden_claim_hits(docs: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in docs.items():
        if rel.endswith(".json") or rel.endswith(".py") or rel.endswith(".ps1"):
            continue
        for raw_line in text.splitlines():
            line = raw_line.lower()
            if not line.strip():
                continue
            for verdict, phrases in FORBIDDEN_CLAIMS.items():
                for phrase in phrases:
                    lowered = phrase.lower()
                    if lowered in line and not line_is_negated(line, lowered):
                        hits.append({"file": rel, "claim": raw_line.strip(), "verdict": verdict})
    return hits


def missing_boundary_tokens(docs: dict[str, str]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for rel in REQUIRED_DOCS:
        text = docs.get(rel, "")
        for token in BOUNDARY_TOKENS:
            if token not in text:
                missing.append({"file": rel, "term": token})
    return missing


def missing_exact_commands(docs: dict[str, str]) -> list[str]:
    lines = {line.strip() for text in docs.values() for line in text.splitlines()}
    return [command for command in EXACT_COMMANDS if command not in lines]


def missing_required_terms(docs: dict[str, str]) -> list[dict[str, str]]:
    bundle = "\n".join(docs.values())
    missing = [{"term": term} for term in REQUIRED_TERMS if term not in bundle]
    source = docs.get("tools/instnct_service_alpha/instnct_service_alpha.py", "")
    for term in REQUIRED_SOURCE_TERMS:
        if term not in source:
            missing.append({"file": "tools/instnct_service_alpha/instnct_service_alpha.py", "term": term})
    if "shell=True" in source or "shell = True" in source:
        missing.append({"file": "tools/instnct_service_alpha/instnct_service_alpha.py", "term": "COMMAND_ARGUMENT_UNSAFE"})
    missing.extend(missing_boundary_tokens(docs))
    return missing


def unsafe_defaults(docs: dict[str, str]) -> list[str]:
    config_text = docs.get("tools/instnct_service_alpha/config/example.local.json", "{}")
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError:
        return ["config_json_invalid"]
    issues: list[str] = []
    if config.get("bind_host") != "127.0.0.1":
        issues.append("PUBLIC_BIND_DETECTED")
    if config.get("service_schema_version") != "instnct_service_api_alpha_v1":
        issues.append("API_SCHEMA_MISSING")
    if config.get("production_default_training_enabled") is not False:
        issues.append("production_default_training_enabled_not_false")
    if config.get("public_beta_promoted") is not False:
        issues.append("public_beta_promoted_not_false")
    if config.get("production_api_ready") is not False:
        issues.append("production_api_ready_not_false")
    out_dir = str(config.get("out_dir", ""))
    if not out_dir.startswith("target/pilot_wave/") or ".." in out_dir:
        issues.append("unsafe_out_dir")
    return issues


def generated_artifact_staged() -> list[str]:
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    paths: set[str] = set()
    for line in status.stdout.splitlines():
        if len(line) >= 4:
            paths.add(line[3:].replace("\\", "/"))
    generated: list[str] = []
    for rel in sorted(paths):
        lower = rel.lower()
        if any(part in lower for part in GENERATED_PATH_PARTS):
            generated.append(rel)
        elif any(lower.endswith(suffix) for suffix in GENERATED_SUFFIXES):
            generated.append(rel)
        elif any(part in lower for part in GENERATED_NAME_PARTS) and not rel.startswith("tools/"):
            generated.append(rel)
    return generated


def derive_verdicts(
    check_pass: bool,
    missing_terms: list[dict[str, str]],
    unsafe: list[str],
    claims: list[dict[str, str]],
    generated: list[str],
    root_changed: bool,
) -> list[str]:
    if check_pass:
        return POSITIVE_VERDICTS
    verdicts = ["SERVICE_API_ALPHA_FAILS"]
    if any(item.get("term") == "COMMAND_ARGUMENT_UNSAFE" for item in missing_terms):
        verdicts.append("COMMAND_ARGUMENT_UNSAFE")
    if "PUBLIC_BIND_DETECTED" in unsafe:
        verdicts.append("PUBLIC_BIND_DETECTED")
    if any(item.get("term") == "POLICY_GUARD_REJECTED" for item in missing_terms):
        verdicts.append("POLICY_GUARD_FAILS")
    if any(item.get("term") == "ARTIFACT_PATH_ESCAPE" for item in missing_terms):
        verdicts.append("ARTIFACT_RETRIEVAL_FAILS")
    if any(item.get("term") == "IDEMPOTENCY_CONFLICT" for item in missing_terms):
        verdicts.append("IDEMPOTENCY_FAILS")
    if any(item.get("term") == "alpha_static_local" for item in missing_terms):
        verdicts.append("RATE_LIMIT_BOUNDARY_MISSING")
    if generated:
        verdicts.append("GENERATED_ARTIFACT_STAGED")
    if root_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    for hit in claims:
        if hit["verdict"] not in verdicts:
            verdicts.append(hit["verdict"])
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        parser.error("only --check-only is supported")

    missing_docs, docs = read_required_files()
    placeholders = placeholder_hits(docs)
    claims = forbidden_claim_hits(docs)
    commands = missing_exact_commands(docs)
    terms = missing_required_terms(docs)
    unsafe = unsafe_defaults(docs)
    generated = generated_artifact_staged()
    root_changed = root_license_changed()

    check_pass = not any([missing_docs, placeholders, claims, commands, terms, unsafe, generated, root_changed])
    result: dict[str, Any] = {
        "check_pass": check_pass,
        "missing_docs": missing_docs,
        "placeholder_hits": placeholders,
        "missing_commands": commands,
        "missing_required_terms": terms,
        "unsafe_defaults": unsafe,
        "forbidden_claim_hits": claims,
        "root_license_changed": root_changed,
        "generated_artifact_staged": generated,
        "verdicts": derive_verdicts(check_pass, terms, unsafe, claims, generated, root_changed),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
