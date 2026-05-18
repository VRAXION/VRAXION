#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_065 private pilot readiness."""

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
    "docs/product/INSTNCT_PRIVATE_PILOT_READINESS.md",
    "docs/product/INSTNCT_PRIVATE_PILOT_AGREEMENT_CHECKLIST.md",
    "docs/product/INSTNCT_PRIVATE_PILOT_ONBOARDING_GUIDE.md",
    "docs/product/INSTNCT_PRIVATE_PILOT_OPERATOR_RUNBOOK.md",
    "docs/product/INSTNCT_PRIVATE_PILOT_SUPPORT_CHANNEL_POLICY.md",
    "docs/product/INSTNCT_PRIVATE_PILOT_ISSUE_TRIAGE_POLICY.md",
    "docs/product/INSTNCT_PRIVATE_PILOT_SUCCESS_FAILURE_CRITERIA.md",
    "docs/product/INSTNCT_PRIVATE_PILOT_GO_NO_GO_GATE.md",
    "docs/product/INSTNCT_PRIVATE_PILOT_DATA_OPS_HANDOFF_CHECKLIST.md",
    "docs/product/INSTNCT_PRIVATE_PILOT_POST_PILOT_REPORT_TEMPLATE.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_065_PRIVATE_PILOT_RELEASE_READINESS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_065_PRIVATE_PILOT_RELEASE_READINESS_RESULT.md",
]

EXACT_COMMANDS = [
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_065_private_pilot_readiness_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_065_private_pilot_readiness_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_064_observability_incident_backup_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_063_security_supply_chain_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_061_release_candidate_check.py --check-only",
    "git diff --check",
]

DOC_REFERENCES = [
    "059 pilot boundary",
    "060 license package drafts",
    "061 RC_001 package",
    "062 service/API alpha",
    "063 security/supply-chain gate",
    "064 ops readiness gate",
]

BOUNDARY_TOKENS = [
    "no pilot launched",
    "no partner approved",
    "no external use authorized",
    "no production deployment",
    "no hosted SaaS",
    "no public beta",
    "no GA",
    "no production API readiness",
    "no production SRE readiness",
    "no SLA",
    "no clinical use",
    "no high-stakes education use",
    "no PHI/student records without separate written agreement",
    "no final legal terms",
    "no commercial launch",
    "no full VRAXION",
    "no language grounding",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]

REQUIRED_TERMS = [
    "best-effort evaluation support only",
    "no production support",
    "no clinical/high-stakes support",
    "support channel must be named before pilot start",
    "external support requires separate written agreement",
    "GO only if every required prior gate passes",
    "NO-GO if any regulated use appears",
    "NO-GO if PHI/student records appear without separate written agreement",
    "NO-GO if rollback/disable path is missing",
    "NO-GO if a critical issue is unresolved",
    "NO-GO if support owner/channel is missing",
    "NO-GO if legal/counsel review required but missing",
    "allowed data class",
    "forbidden data class",
    "retention plan",
    "deletion path",
    "audit log owner",
    "incident contact",
    "rollback/disable owner",
    "closeout owner",
    "post-pilot report owner",
    "Critical:",
    "High:",
    "Medium:",
    "Low:",
    "suspected sensitive data exposure",
    "regulated use request",
    "production/public-beta misuse",
    "checkpoint/artifact integrity failure",
    "rollback failure",
    "policy guard bypass",
    "install/check smoke completes",
    "deployment harness smoke completes",
    "service alpha healthcheck passes",
    "063 static checker passes",
    "064 static checker passes",
    "no unresolved critical issue",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "PILOT_LAUNCH_CLAIM_DETECTED": [
        "pilot started",
        "pilot launched",
        "pilot is live",
        "pilot start approved",
    ],
    "PARTNER_APPROVAL_CLAIM_DETECTED": [
        "pilot approved",
        "partner approved",
        "customer approved",
        "external use authorized",
        "production use approved",
    ],
    "PRODUCTION_DEPLOYMENT_CLAIM_DETECTED": [
        "production deployment",
        "production use",
    ],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "CLINICAL_READY_CLAIM_DETECTED": ["clinical readiness", "clinical ready"],
    "HIGH_STAKES_EDUCATION_READY_CLAIM_DETECTED": [
        "high-stakes education readiness",
        "high-stakes education ready",
    ],
    "FINAL_LEGAL_TERMS_CLAIM_DETECTED": ["final legal terms"],
    "COMMERCIAL_LAUNCH_CLAIM_DETECTED": ["commercial launch"],
    "PRODUCTION_API_READY_CLAIM_DETECTED": ["production API readiness"],
    "PRODUCTION_SRE_CLAIM_DETECTED": ["production SRE readiness"],
    "SUPPORT_OVERCLAIM_DETECTED": ["SLA"],
}

NEGATION_MARKERS = ["not ", "no ", "no-go ", "does not ", "do not ", "without ", "false", "requires "]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "sdk_smoke/", "smoke/", "service job artifacts", "pilot-generated"]
MUTATION_PATHS = ["instnct-core", "tools/instnct_service_alpha", "tools/instnct_deploy"]

POSITIVE_VERDICTS = [
    "PRIVATE_PILOT_RELEASE_READINESS_POSITIVE",
    "PILOT_READINESS_PACKAGE_WRITTEN",
    "PILOT_AGREEMENT_CHECKLIST_WRITTEN",
    "PILOT_ONBOARDING_GUIDE_WRITTEN",
    "PILOT_OPERATOR_RUNBOOK_WRITTEN",
    "SUPPORT_CHANNEL_POLICY_WRITTEN",
    "SUPPORT_BOUNDARY_RESTRICTED",
    "ISSUE_TRIAGE_POLICY_WRITTEN",
    "ISSUE_TRIAGE_POLICY_COMPLETE",
    "PILOT_SUCCESS_FAILURE_CRITERIA_DEFINED",
    "PILOT_GO_NO_GO_GATE_DEFINED",
    "GO_NO_GO_BINARY_GATE_DEFINED",
    "DATA_OPS_HANDOFF_CHECKLIST_WRITTEN",
    "POST_PILOT_REPORT_TEMPLATE_WRITTEN",
    "PRIOR_GATES_REFERENCED",
    "NO_PILOT_LAUNCHED",
    "PARTNER_APPROVAL_NOT_CLAIMED",
    "PRODUCTION_DEPLOYMENT_NOT_CLAIMED",
    "PUBLIC_BETA_NOT_CLAIMED",
]


def read_docs() -> tuple[list[str], dict[str, str]]:
    missing: list[str] = []
    docs: dict[str, str] = {}
    for rel in REQUIRED_DOCS:
        path = REPO_ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        text = path.read_text(encoding="utf-8")
        if len(text.strip()) < 200:
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
    return bool(git_status(MUTATION_PATHS))


def placeholder_hits(docs: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in docs.items():
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
        for idx, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.lower()
            if not line.strip():
                continue
            for verdict, phrases in FORBIDDEN_CLAIMS.items():
                for phrase in phrases:
                    lowered = phrase.lower()
                    if lowered in line and not line_is_negated(line, lowered):
                        hits.append({"file": rel, "line": idx, "phrase": phrase, "verdict": verdict})
    return hits


def missing_commands(docs: dict[str, str]) -> list[str]:
    text = "\n".join(docs.values())
    return [command for command in EXACT_COMMANDS if command not in text]


def missing_doc_references(docs: dict[str, str]) -> list[str]:
    text = "\n".join(docs.values()).lower()
    return [ref for ref in DOC_REFERENCES if ref.lower() not in text]


def missing_boundary_tokens(docs: dict[str, str]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for rel in REQUIRED_DOCS:
        text = docs.get(rel, "").lower()
        for token in BOUNDARY_TOKENS:
            if token.lower() not in text:
                missing.append({"file": rel, "token": token})
    return missing


def missing_required_terms(docs: dict[str, str]) -> list[str]:
    text = "\n".join(docs.values()).lower()
    return [term for term in REQUIRED_TERMS if term.lower() not in text]


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
        if any(part in lower for part in GENERATED_NAME_PARTS):
            hits.append(path)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        print("Only --check-only is supported.", file=sys.stderr)
        return 2

    missing_docs, docs = read_docs()
    placeholders = placeholder_hits(docs)
    commands = missing_commands(docs)
    references = missing_doc_references(docs)
    boundary = missing_boundary_tokens(docs)
    claims = forbidden_claim_hits(docs)
    generated = generated_artifact_staged()
    root_changed = root_license_changed()
    service_mutation = service_api_mutation_detected()
    terms = missing_required_terms(docs)

    check_pass = not any(
        [
            missing_docs,
            placeholders,
            commands,
            references,
            boundary,
            claims,
            generated,
            root_changed,
            service_mutation,
            terms,
        ]
    )
    verdicts = POSITIVE_VERDICTS if check_pass else ["PRIVATE_PILOT_RELEASE_READINESS_FAILS"]
    if generated:
        verdicts.append("GENERATED_ARTIFACT_STAGED")
    if root_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    if service_mutation:
        verdicts.append("SERVICE_API_MUTATION_DETECTED")
    result: dict[str, Any] = {
        "check_pass": check_pass,
        "missing_docs": missing_docs,
        "placeholder_hits": placeholders,
        "missing_commands": commands,
        "missing_doc_references": references,
        "missing_boundary_tokens": boundary,
        "forbidden_claim_hits": claims,
        "generated_artifact_staged": generated,
        "root_license_changed": root_changed,
        "service_api_mutation_detected": service_mutation,
        "missing_required_terms": terms,
        "verdicts": verdicts,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
