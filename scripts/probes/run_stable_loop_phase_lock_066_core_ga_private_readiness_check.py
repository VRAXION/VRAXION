#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_066 Core readiness."""

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
    "docs/product/INSTNCT_CORE_PRIVATE_SELF_HOSTED_READINESS.md",
    "docs/product/INSTNCT_CORE_RELEASE_READINESS_MATRIX.md",
    "docs/product/INSTNCT_CORE_APPROVED_USE_BOUNDARY.md",
    "docs/product/INSTNCT_CORE_INSTALL_SMOKE_SECURITY_OPS_SUPPORT_CHECKLIST.md",
    "docs/product/INSTNCT_CORE_GO_NO_GO_GATE.md",
    "docs/product/INSTNCT_CORE_RESIDUAL_RISKS.md",
    "docs/product/INSTNCT_CORE_GO_NO_GO_DECISION_RECORD_TEMPLATE.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_066_CORE_GA_PRIVATE_RELEASE_READINESS_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_066_CORE_GA_PRIVATE_RELEASE_READINESS_RESULT.md",
]

EXACT_COMMANDS = [
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_066_core_ga_private_readiness_check.py",
    "python scripts/probes/run_stable_loop_phase_lock_066_core_ga_private_readiness_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_065_private_pilot_readiness_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_064_observability_incident_backup_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_063_security_supply_chain_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_061_release_candidate_check.py --check-only",
    "git diff --check",
]

DOC_REFERENCES = [
    "056 productization boundary",
    "057 SDK candidate",
    "058 deployment harness",
    "059 pilot boundary",
    "060 license drafts",
    "061 RC_001 package",
    "062 service/API alpha",
    "063 security/supply-chain gate",
    "064 ops-readiness gate",
    "065 private pilot readiness",
]

MATRIX_FIELDS = [
    "component",
    "source gate",
    "status",
    "evidence doc",
    "checker command",
    "residual risk",
]

BOUNDARY_TOKENS = [
    "no GA launched",
    "no production release",
    "no production deployment",
    "no hosted SaaS",
    "no public beta",
    "no public launch",
    "no partner approved",
    "no pilot launched",
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
    "Core private/self-hosted release readiness candidate",
    "SDK candidate",
    "deployment harness",
    "service/API alpha",
    "visual audit package",
    "security/supply-chain gate",
    "ops-readiness gate",
    "release docs",
    "license/support/pilot boundaries",
    "private/self-hosted",
    "approved non-regulated use",
    "local/private evaluation or controlled private deployment",
    "written approval required",
    "GO only if:",
    "all 056-065 gates positive",
    "all 061-065 static checkers pass",
    "root LICENSE unchanged",
    "no runtime/API mutation",
    "no generated artifacts staged",
    "no forbidden claims",
    "residual risks documented",
    "NO-GO if any prior checker fails",
    "NO-GO if root `LICENSE` changed unexpectedly",
    "NO-GO if runtime/API surfaces changed unexpectedly",
    "NO-GO if generated artifacts are staged",
    "NO-GO if residual risks are missing",
    "no signed release yet",
    "no counsel-approved license",
    "no external pilot completed",
    "no regulated-use compliance",
    "no customer data approval",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "GA_LAUNCH_CLAIM_DETECTED": [
        "GA launch",
        "GA launched",
        "GA release",
        "generally available",
    ],
    "PRODUCTION_RELEASE_CLAIM_DETECTED": [
        "production release",
        "release launched",
        "production deployment",
    ],
    "HOSTED_SAAS_CLAIM_DETECTED": ["hosted SaaS"],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "PARTNER_APPROVAL_CLAIM_DETECTED": [
        "partner approved",
        "pilot launched",
        "external use authorized",
    ],
    "PRODUCTION_API_READY_CLAIM_DETECTED": ["production API readiness"],
    "PRODUCTION_SRE_CLAIM_DETECTED": ["production SRE readiness"],
    "SUPPORT_OVERCLAIM_DETECTED": ["SLA"],
    "CLINICAL_READY_CLAIM_DETECTED": ["clinical readiness", "clinical ready"],
    "HIGH_STAKES_EDUCATION_READY_CLAIM_DETECTED": [
        "high-stakes education readiness",
        "high-stakes education ready",
    ],
    "FINAL_LEGAL_TERMS_CLAIM_DETECTED": ["final legal terms"],
    "COMMERCIAL_LAUNCH_CLAIM_DETECTED": ["commercial launch"],
}

NEGATION_MARKERS = ["not ", "no ", "no-go ", "does not ", "do not ", "without ", "false", "avoid "]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAME_PARTS = ["checkpoint", "sdk_smoke/", "smoke/", "service job artifacts", "release artifacts"]
MUTATION_PATHS = ["instnct-core", "tools/instnct_service_alpha", "tools/instnct_deploy"]

POSITIVE_VERDICTS = [
    "CORE_GA_PRIVATE_RELEASE_READINESS_POSITIVE",
    "CORE_RELEASE_READINESS_MATRIX_WRITTEN",
    "CORE_RELEASE_MATRIX_TRACEABLE",
    "PRIVATE_SELF_HOSTED_BOUNDARY_DEFINED",
    "APPROVED_USE_BOUNDARY_DEFINED",
    "CORE_COMPONENTS_DEFINED",
    "INSTALL_SMOKE_SECURITY_OPS_SUPPORT_CHECKLIST_WRITTEN",
    "PRIOR_GATES_056_065_REFERENCED",
    "CORE_GO_NO_GO_GATE_DEFINED",
    "GO_NO_GO_BINARY_GATE_DEFINED",
    "RESIDUAL_RISKS_DOCUMENTED",
    "RUNTIME_SURFACE_UNCHANGED",
    "NO_GA_LAUNCHED",
    "NO_PRODUCTION_RELEASE_CLAIMED",
    "NO_PARTNER_APPROVED",
    "NO_PUBLIC_BETA_CLAIMED",
    "NO_HOSTED_SAAS_CLAIMED",
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


def runtime_mutation_detected() -> bool:
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
    missing = [ref for ref in DOC_REFERENCES if ref.lower() not in text]
    matrix = docs.get("docs/product/INSTNCT_CORE_RELEASE_READINESS_MATRIX.md", "").lower()
    for ref in DOC_REFERENCES:
        if ref.lower() not in matrix:
            missing.append(f"matrix:{ref}")
    return missing


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
    missing = [term for term in REQUIRED_TERMS if term.lower() not in text]
    matrix = docs.get("docs/product/INSTNCT_CORE_RELEASE_READINESS_MATRIX.md", "").lower()
    for field in MATRIX_FIELDS:
        if field.lower() not in matrix:
            missing.append(f"matrix_field:{field}")
    for ref in DOC_REFERENCES:
        if ref.lower() not in matrix:
            missing.append(f"matrix_row:{ref}")
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
    runtime_mutation = runtime_mutation_detected()
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
            runtime_mutation,
            terms,
        ]
    )
    verdicts = POSITIVE_VERDICTS if check_pass else ["CORE_GA_PRIVATE_RELEASE_READINESS_FAILS"]
    if generated:
        verdicts.append("GENERATED_ARTIFACT_STAGED")
    if root_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    if runtime_mutation:
        verdicts.append("RUNTIME_MUTATION_DETECTED")
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
        "runtime_mutation_detected": runtime_mutation,
        "missing_required_terms": terms,
        "verdicts": verdicts,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
