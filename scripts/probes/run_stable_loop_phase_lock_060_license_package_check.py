#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_060 license package docs."""

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
    "docs/product/INSTNCT_SOURCE_AVAILABLE_NONCOMMERCIAL_LICENSE_DRAFT.md",
    "docs/product/INSTNCT_COMMERCIAL_LICENSE_TEMPLATE_DRAFT.md",
    "docs/product/INSTNCT_PUBLIC_BENEFIT_RIDER_DRAFT.md",
    "docs/product/INSTNCT_DCO_CONTRIBUTOR_POLICY_DRAFT.md",
    "docs/product/INSTNCT_TRADEMARK_POLICY_DRAFT.md",
    "docs/product/INSTNCT_CLAIM_BOUNDARY_POLICY.md",
    "docs/product/INSTNCT_ACCEPTABLE_USE_POLICY_FINAL_DRAFT.md",
    "docs/product/INSTNCT_LICENSE_PACKAGE_COUNSEL_REVIEW_GATE.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_060_LICENSE_PACKAGE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_060_LICENSE_PACKAGE_RESULT.md",
]

PRODUCT_DRAFT_DOCS = REQUIRED_DOCS[:8]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

OPEN_SOURCE_FORBIDDEN = [
    "open-source license",
    "open source license",
    "free open source",
    "commercial use prohibited open source",
]

PRODUCTION_CLAIM_FORBIDDEN = [
    "production ready",
    "production-ready",
    "ready for production",
    "clinical ready",
    "ready for clinical",
    "high-stakes education ready",
]

BOUNDARY_TOKENS = [
    "no production deployment",
    "no hosted SaaS",
    "no public beta",
    "no production API readiness",
    "no production readiness",
    "no clinical use",
    "no high-stakes education use",
    "no PHI/student records without separate written agreement",
    "no commercial license starts from this draft alone",
    "no public-benefit permission starts from this draft alone",
    "no contributor permission starts from this draft alone",
    "no trademark permission starts from this draft alone",
    "no full VRAXION",
    "no language grounding",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]

COMMON_GATE = [
    "draft only",
    "not final legal terms",
    "counsel review required before external use",
    "no agreement or permission starts from this draft alone",
]

REQUIRED_TERMS = {
    "source_available": [
        "source-available/noncommercial",
        "not OSI open source",
        "not open source",
        "commercial use requires a separate written commercial license",
    ],
    "commercial_triggers": [
        "selling",
        "hosted access",
        "embedding",
        "paid consulting delivery",
        "production customer workflows",
        "redistribution outside noncommercial terms",
        "internal business production use",
        "training/evaluating customer systems as a paid service",
    ],
    "regulated_carveout": [
        "commercial license alone does not authorize",
        "diagnosis",
        "treatment",
        "triage",
        "medication decisions",
        "clinical decision support or CDS",
        "grading",
        "admissions",
        "student ranking",
        "placement",
        "proctoring",
        "high-stakes profiling",
    ],
    "public_benefit": [
        "organizations may request review",
        "written approval required",
        "no external pilot starts from rider alone",
        "no PHI/student records without separate written agreement",
        "no clinical/high-stakes use",
    ],
    "dco_policy": [
        "Signed-off-by",
        "contributor certifies right to submit contribution",
        "contribution can be used under project license",
        "CLA/copyright assignment not adopted yet",
        "counsel review required before changing contributor model",
    ],
    "trademark_policy": [
        "endorsement",
        "official partnership",
        "production readiness",
        "clinical readiness",
        "high-stakes education readiness",
        "full VRAXION",
        "language grounding",
        "consciousness",
    ],
}

POSITIVE_VERDICTS = [
    "LICENSE_PACKAGE_POSITIVE",
    "SOURCE_AVAILABLE_NONCOMMERCIAL_DRAFT_WRITTEN",
    "COMMERCIAL_LICENSE_TEMPLATE_DRAFT_WRITTEN",
    "PUBLIC_BENEFIT_RIDER_DRAFT_WRITTEN",
    "DCO_CONTRIBUTOR_POLICY_DRAFT_WRITTEN",
    "TRADEMARK_POLICY_DRAFT_WRITTEN",
    "CLAIM_BOUNDARY_POLICY_WRITTEN",
    "ACCEPTABLE_USE_POLICY_FINAL_DRAFT_WRITTEN",
    "COUNSEL_REVIEW_GATE_PRESENT",
    "ROOT_LICENSE_UNCHANGED",
    "NOT_OPEN_SOURCE_CLAIM_CONTROLLED",
    "PRODUCTION_READY_NOT_CLAIMED",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text)


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
        for marker in PLACEHOLDERS:
            if re.search(rf"\b{re.escape(marker)}\b", text, flags=re.IGNORECASE):
                hits.append({"file": rel, "marker": marker})
    return hits


def open_source_hits(docs: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    allowed_phrases = [
        "not osi open source",
        "not open source",
        "source-available, not open source",
    ]
    for rel, text in docs.items():
        for raw_line in text.splitlines():
            line = raw_line.lower()
            if "open source" not in line and "open-source" not in line:
                continue
            if any(forbidden in line for forbidden in OPEN_SOURCE_FORBIDDEN):
                hits.append({"file": rel, "claim": raw_line.strip()})
                continue
            if not any(allowed in line for allowed in allowed_phrases):
                hits.append({"file": rel, "claim": raw_line.strip()})
    return hits


def production_claim_hits(docs: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in docs.items():
        lower = text.lower()
        for claim in PRODUCTION_CLAIM_FORBIDDEN:
            if claim in lower:
                hits.append({"file": rel, "claim": claim})
    return hits


def missing_boundary_tokens(docs: dict[str, str]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for rel, text in docs.items():
        for token in BOUNDARY_TOKENS:
            if token not in text:
                missing.append({"file": rel, "token": token})
    return missing


def missing_common_gate(docs: dict[str, str]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for rel in PRODUCT_DRAFT_DOCS:
        haystack = normalize(docs.get(rel, "").lower())
        for term in COMMON_GATE:
            if normalize(term.lower()) not in haystack:
                missing.append({"group": "counsel_gate", "file": rel, "term": term})
    return missing


def missing_required_terms(docs: dict[str, str]) -> list[dict[str, str]]:
    bundle = normalize("\n".join(docs.values()).lower())
    missing: list[dict[str, str]] = []
    for group, terms in REQUIRED_TERMS.items():
        for term in terms:
            if normalize(term.lower()) not in bundle:
                missing.append({"group": group, "term": term})
    missing.extend(missing_common_gate(docs))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        parser.error("only --check-only is supported")

    missing_docs, docs = read_docs()
    root_changed = root_license_changed()
    placeholders = placeholder_hits(docs)
    forbidden_claims = open_source_hits(docs) + production_claim_hits(docs)
    missing_boundary = missing_boundary_tokens(docs)
    missing_terms = missing_required_terms(docs)
    if missing_boundary:
        missing_terms.extend(
            {"group": "claim_boundary", "file": item["file"], "term": item["token"]}
            for item in missing_boundary
        )

    check_pass = not any([root_changed, missing_docs, placeholders, forbidden_claims, missing_terms])
    verdicts = POSITIVE_VERDICTS if check_pass else ["LICENSE_PACKAGE_FAILS"]
    if root_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    if open_source_hits(docs):
        verdicts.append("OPEN_SOURCE_CLAIM_DETECTED")
    if any(item["group"] == "counsel_gate" for item in missing_terms):
        verdicts.append("COUNSEL_REVIEW_GATE_MISSING")
    if any(item["group"] == "commercial_triggers" for item in missing_terms):
        verdicts.append("COMMERCIAL_TRIGGER_MISSING")
    if any(item["group"] == "regulated_carveout" for item in missing_terms):
        verdicts.append("REGULATED_USE_CARVEOUT_MISSING")
    if any(item["group"] == "public_benefit" for item in missing_terms):
        verdicts.append("PUBLIC_BENEFIT_RESTRICTION_MISSING")
    if any(item["group"] == "dco_policy" for item in missing_terms):
        verdicts.append("DCO_POLICY_MISSING")
    if any(item["group"] == "trademark_policy" for item in missing_terms):
        verdicts.append("TRADEMARK_POLICY_MISSING")
    if any(item["group"] == "claim_boundary" for item in missing_terms):
        verdicts.append("CLAIM_BOUNDARY_MISSING")
    if production_claim_hits(docs):
        verdicts.append("PRODUCTION_READY_CLAIM_DETECTED")

    result: dict[str, Any] = {
        "check_pass": check_pass,
        "root_license_changed": root_changed,
        "missing_docs": missing_docs,
        "placeholder_hits": placeholders,
        "forbidden_claim_hits": forbidden_claims,
        "missing_required_terms": missing_terms,
        "verdicts": verdicts,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
