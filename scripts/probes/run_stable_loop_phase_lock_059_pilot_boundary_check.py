#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_059 pilot boundary docs."""

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
    "docs/product/INSTNCT_HOSPITAL_PILOT_BOUNDARY.md",
    "docs/product/INSTNCT_SCHOOL_PILOT_BOUNDARY.md",
    "docs/product/INSTNCT_PUBLIC_BENEFIT_PILOT_REQUEST_TEMPLATE.md",
    "docs/product/INSTNCT_PILOT_DATA_HANDLING_CHECKLIST.md",
    "docs/product/INSTNCT_HUMAN_OVERSIGHT_CHECKLIST.md",
    "docs/product/INSTNCT_PILOT_APPROVAL_WORKFLOW.md",
    "docs/product/INSTNCT_PILOT_REJECTION_REASONS.md",
    "docs/product/INSTNCT_PUBLIC_BENEFIT_TERMS_DRAFT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_059_HOSPITAL_SCHOOL_PILOT_BOUNDARY_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_059_HOSPITAL_SCHOOL_PILOT_BOUNDARY_RESULT.md",
]

PLACEHOLDERS = [
    "TODO",
    "TBD",
    "placeholder",
    "pending measured",
    "coming soon",
]

FORBIDDEN_CLAIMS = [
    "approved for hospitals",
    "approved for schools",
    "safe for clinical use",
    "safe for grading",
    "ready for student data",
    "ready for PHI",
]

BOUNDARY_TOKENS = [
    "no production deployment",
    "no hosted SaaS",
    "no public beta",
    "no production API readiness",
    "no clinical use",
    "no high-stakes education use",
    "no PHI/student records without separate written agreement",
    "no full VRAXION",
    "no language grounding",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]

REQUIRED_TERMS = {
    "allowed_wording": [
        "may request review",
        "pilot candidate",
        "requires written approval",
        "non-clinical only",
        "non-high-stakes only",
    ],
    "hospital_scope": [
        "non-clinical administration exploration",
        "research support",
        "internal document-routing experiments",
        "model behavior visualization",
        "non-clinical education or simulation",
        "diagnosis",
        "treatment recommendation",
        "triage",
        "medication decisions",
        "clinical decision support",
        "emergency prioritization",
        "patient-specific risk scoring",
        "direct patient-care decision",
    ],
    "school_scope": [
        "tutoring and explanation support",
        "practice generation",
        "classroom demonstration",
        "teacher-support planning",
        "non-high-stakes learning aid",
        "grading",
        "admissions",
        "student ranking",
        "placement decisions",
        "proctoring",
        "prohibited-behavior detection",
        "high-stakes profiling",
    ],
    "data_matrix": [
        "allowed by default",
        "requires separate written agreement",
        "forbidden before compliance review",
        "synthetic data",
        "public non-sensitive data",
        "internal admin data",
        "PHI",
        "student education records",
        "minors sensitive data",
        "biometric data",
        "financial data",
        "live clinical data",
        "live grading/admissions data",
    ],
    "human_oversight": [
        "named human owner",
        "human reviews outputs before use",
        "no autonomous decision authority",
        "disable/rollback path",
        "audit log review",
        "incident contact",
        "closeout review",
    ],
    "request_template": [
        "organization type",
        "hospital",
        "school",
        "nonprofit",
        "research organization",
        "intended use",
        "user population",
        "data categories",
        "minors are involved",
        "patients are involved",
        "outputs affect rights, access, health, or education outcomes",
        "requested deployment mode",
        "human owner",
        "rollback or disable plan",
        "data retention plan",
    ],
    "deployment_scope": [
        "local_research",
        "private_evaluation",
        "hosted SaaS",
        "production deployment",
        "public beta",
        "customer-facing unsupervised use",
    ],
    "counsel_gate": [
        "draft only",
        "not final legal terms",
        "counsel review is required before external use",
        "no external pilot may start from this draft alone",
    ],
    "rejection_reasons": [
        "clinical diagnosis",
        "treatment recommendation",
        "triage",
        "medication decision",
        "clinical decision support or CDS",
        "grading",
        "admissions",
        "student ranking",
        "placement",
        "proctoring",
        "PHI or student records without agreement",
        "production automation",
        "Hosted/SaaS request",
        "missing human owner",
        "unclear data retention",
        "claim-boundary conflict",
    ],
}

POSITIVE_VERDICTS = [
    "HOSPITAL_SCHOOL_PILOT_BOUNDARY_POSITIVE",
    "HOSPITAL_NON_CLINICAL_SCOPE_DEFINED",
    "SCHOOL_NON_HIGH_STAKES_SCOPE_DEFINED",
    "PILOT_REQUEST_TEMPLATE_WRITTEN",
    "PILOT_DATA_HANDLING_CHECKLIST_WRITTEN",
    "HUMAN_OVERSIGHT_CHECKLIST_WRITTEN",
    "PILOT_APPROVAL_WORKFLOW_DEFINED",
    "PILOT_REJECTION_REASONS_DEFINED",
    "PUBLIC_BENEFIT_TERMS_DRAFT_WRITTEN",
    "DATA_CATEGORY_MATRIX_DEFINED",
    "COUNSEL_REVIEW_GATE_PRESENT",
    "PILOT_REQUEST_CLASSIFICATION_COMPLETE",
    "DEPLOYMENT_SCOPE_RESTRICTED",
    "NO_IMPLIED_AUTHORIZATION",
    "CLINICAL_READY_NOT_CLAIMED",
    "HIGH_STAKES_EDUCATION_READY_NOT_CLAIMED",
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


def find_placeholder_hits(docs: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in docs.items():
        for marker in PLACEHOLDERS:
            if re.search(rf"\b{re.escape(marker)}\b", text, flags=re.IGNORECASE):
                hits.append({"file": rel, "marker": marker})
    return hits


def find_forbidden_claims(docs: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in docs.items():
        lower = text.lower()
        for claim in FORBIDDEN_CLAIMS:
            if claim.lower() in lower:
                hits.append({"file": rel, "claim": claim})
    return hits


def missing_boundary_tokens(docs: dict[str, str]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for rel, text in docs.items():
        for token in BOUNDARY_TOKENS:
            if token not in text:
                missing.append({"file": rel, "token": token})
    return missing


def missing_required_terms(bundle: str) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    haystack = re.sub(r"\s+", " ", bundle.lower())
    for group, terms in REQUIRED_TERMS.items():
        for term in terms:
            needle = re.sub(r"\s+", " ", term.lower())
            if needle not in haystack:
                missing.append({"group": group, "term": term})
    return missing


def docs_only_changed() -> list[str]:
    try:
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return []
    bad: list[str] = []
    for line in status.stdout.splitlines():
        path = line[3:].replace("\\", "/")
        if path.startswith("instnct-core/") or path.startswith("tools/instnct_deploy/"):
            bad.append(path)
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        parser.error("only --check-only is supported")

    missing_docs, docs = read_docs()
    placeholder_hits = find_placeholder_hits(docs)
    forbidden_claim_hits = find_forbidden_claims(docs)
    boundary_missing = missing_boundary_tokens(docs)
    bundle = "\n".join(docs.values())
    required_missing = missing_required_terms(bundle)
    disallowed_code_changes = docs_only_changed()

    check_pass = not any(
        [
            missing_docs,
            placeholder_hits,
            forbidden_claim_hits,
            boundary_missing,
            required_missing,
            disallowed_code_changes,
        ]
    )

    verdicts = POSITIVE_VERDICTS if check_pass else ["HOSPITAL_SCHOOL_PILOT_BOUNDARY_FAILS"]
    if forbidden_claim_hits:
        verdicts.append("IMPLIED_AUTHORIZATION_DETECTED")
    if any(item["group"] == "counsel_gate" for item in required_missing):
        verdicts.append("COUNSEL_REVIEW_GATE_MISSING")
    if any(item["group"] == "data_matrix" for item in required_missing):
        verdicts.append("DATA_CATEGORY_MATRIX_MISSING")
    if any(item["group"] == "human_oversight" for item in required_missing):
        verdicts.append("HUMAN_OVERSIGHT_TOO_VAGUE")
    if any(item["group"] == "deployment_scope" for item in required_missing):
        verdicts.append("DEPLOYMENT_SCOPE_TOO_BROAD")
    if any(item["group"] == "request_template" for item in required_missing):
        verdicts.append("PILOT_REQUEST_UNCLASSIFIABLE")
    if any(item["group"] == "rejection_reasons" for item in required_missing):
        verdicts.append("REJECTION_REASONS_INCOMPLETE")

    result: dict[str, Any] = {
        "check_pass": check_pass,
        "missing_docs": missing_docs,
        "placeholder_hits": placeholder_hits,
        "missing_boundary_tokens": boundary_missing,
        "forbidden_claim_hits": forbidden_claim_hits,
        "missing_required_terms": required_missing,
        "disallowed_code_changes": disallowed_code_changes,
        "verdicts": verdicts,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
