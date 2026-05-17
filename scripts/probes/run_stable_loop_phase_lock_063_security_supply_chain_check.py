#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_063 security supply-chain gate."""

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
    "docs/product/INSTNCT_SECURITY_SUPPLY_CHAIN_GATE.md",
    "docs/product/INSTNCT_SBOM_POLICY.md",
    "docs/product/INSTNCT_RELEASE_PROVENANCE_POLICY.md",
    "docs/product/INSTNCT_SECRET_SCAN_POLICY.md",
    "docs/product/INSTNCT_DEPENDENCY_AUDIT_POLICY.md",
    "docs/product/INSTNCT_RELEASE_INTEGRITY_POLICY.md",
    "docs/product/INSTNCT_THREAT_MODEL_ALPHA.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_063_SECURITY_SUPPLY_CHAIN_GATE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_063_SECURITY_SUPPLY_CHAIN_GATE_RESULT.md",
    "tools/security_supply_chain/README.md",
]

REQUIRED_SOURCE = [
    "tools/security_supply_chain/instnct_security_gate.py",
    "tools/security_supply_chain/security_gate.ps1",
]

EXACT_COMMANDS = [
    "python -m py_compile tools/security_supply_chain/instnct_security_gate.py",
    "python -m py_compile scripts/probes/run_stable_loop_phase_lock_063_security_supply_chain_check.py",
    "python tools/security_supply_chain/instnct_security_gate.py --out target/pilot_wave/stable_loop_phase_lock_063_security_supply_chain_gate/smoke --heartbeat-sec 20",
    "python scripts/probes/run_stable_loop_phase_lock_063_security_supply_chain_check.py --check-only",
    "python scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py --check-only",
    "git diff --check",
]

BOUNDARY_TOKENS = [
    "no signed release",
    "no CycloneDX compliance",
    "no SPDX compliance",
    "no SLSA compliance",
    "no vulnerability-clean status",
    "no production-ready security",
    "no production readiness",
    "no hosted SaaS readiness",
    "no public beta",
    "no clinical use",
    "no high-stakes education use",
    "no commercial launch",
    "no final legal terms",
    "no full VRAXION",
    "no language grounding",
    "no consciousness",
    "no biological/FlyWire equivalence",
    "no physical quantum behavior",
]

REQUIRED_TERMS = [
    "instnct_sbom_v1",
    "progress.jsonl",
    "security_gate_manifest.json",
    "sbom.instnct.json",
    "checksums.sha256.json",
    "dependency_inventory.json",
    "secret_scan.json",
    "provenance.json",
    "threat_model_summary.json",
    "release_integrity.json",
    "summary.json",
    "report.md",
    "vulnerability database scan not performed in 063",
    "checksums_recorded = true",
    "signing_policy_documented = true",
    "signed_release_artifacts = false",
    "release_archive_created = false",
    "production_release = false",
]

PROGRESS_EVENTS = [
    "start",
    "sbom_completed",
    "checksums_completed",
    "secret_scan_completed",
    "dependency_inventory_completed",
    "provenance_completed",
    "threat_model_completed",
    "release_integrity_completed",
    "done",
]

THREATS = [
    "stale artifact reuse",
    "command injection",
    "path traversal",
    "artifact retrieval escape",
    "policy guard bypass",
    "token/auth bypass",
    "checkpoint tampering",
    "secret leakage",
    "dependency compromise",
    "prompt/data leakage through logs",
    "claim-boundary drift",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "FAKE_COMPLIANCE_CLAIM_DETECTED": [
        "CycloneDX compliant",
        "SPDX compliant",
        "SLSA compliant",
        "CycloneDX compliance",
        "SPDX compliance",
        "SLSA compliance",
    ],
    "SIGNED_RELEASE_CLAIM_DETECTED": [
        "signed release artifacts",
        "signed release",
        "release signature present",
    ],
    "VULNERABILITY_CLEAN_FALSE_CLAIM": [
        "vulnerability-clean status",
        "vulnerability clean",
        "no vulnerabilities",
    ],
    "PRODUCTION_READY_CLAIM_DETECTED": [
        "production-ready security",
        "production ready",
        "production readiness",
    ],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
}

NEGATION_MARKERS = ["not ", "no ", "does not ", "do not ", "without ", "false", "deferred"]
GENERATED_PATH_PARTS = ["target/", "node_modules/", ".svelte-kit/"]
GENERATED_SUFFIXES = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
GENERATED_NAMES = [
    "security_gate_manifest.json",
    "sbom.instnct.json",
    "checksums.sha256.json",
    "dependency_inventory.json",
    "secret_scan.json",
    "provenance.json",
    "threat_model_summary.json",
    "release_integrity.json",
]

POSITIVE_VERDICTS = [
    "SECURITY_SUPPLY_CHAIN_GATE_POSITIVE",
    "SBOM_INVENTORY_WRITTEN",
    "CHECKSUMS_RECORDED",
    "SECRET_SCAN_RECORDED",
    "SECRET_SCAN_POLICY_ENFORCED",
    "DEPENDENCY_INVENTORY_RECORDED",
    "VULNERABILITY_SCAN_DEFERRED_EXPLICITLY",
    "PROVENANCE_RECORDED",
    "PROVENANCE_DIRTY_STATE_RECORDED",
    "THREAT_MODEL_WRITTEN",
    "RELEASE_INTEGRITY_GATE_POSITIVE",
    "SIGNING_POLICY_DOCUMENTED",
    "SIGNED_RELEASE_NOT_CLAIMED",
    "RELEASE_SIGNATURE_NOT_CLAIMED",
    "FAKE_COMPLIANCE_CLAIMS_REJECTED",
    "PRODUCTION_READY_NOT_CLAIMED",
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
        if rel.endswith(".py") or rel.endswith(".ps1"):
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
        if rel.endswith(".py") or rel.endswith(".ps1"):
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
                missing.append({"file": rel, "token": token})
    return missing


def missing_commands(docs: dict[str, str]) -> list[str]:
    lines = {line.strip() for text in docs.values() for line in text.splitlines()}
    return [command for command in EXACT_COMMANDS if command not in lines]


def missing_required_terms(docs: dict[str, str]) -> list[dict[str, str]]:
    bundle = "\n".join(docs.values())
    missing = [{"term": term} for term in REQUIRED_TERMS if term not in bundle]
    for event in PROGRESS_EVENTS:
        if event not in bundle:
            missing.append({"group": "progress", "term": event})
    for threat in THREATS:
        if threat not in bundle:
            missing.append({"group": "threat_model", "term": threat})
    source = docs.get("tools/security_supply_chain/instnct_security_gate.py", "")
    for source_term in [
        "SECRET_PATTERNS",
        "SUPPRESSIONS",
        "instnct_sbom_v1",
        "vulnerability_database_scan_performed",
        "signed_release_artifacts",
        "release_archive_created",
        "production_release",
    ]:
        if source_term not in source:
            missing.append({"file": "tools/security_supply_chain/instnct_security_gate.py", "term": source_term})
    return missing


def generated_artifact_staged() -> list[str]:
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    generated: list[str] = []
    for line in status.stdout.splitlines():
        if len(line) < 4:
            continue
        rel = line[3:].replace("\\", "/")
        lower = rel.lower()
        if any(part in lower for part in GENERATED_PATH_PARTS):
            generated.append(rel)
        elif any(lower.endswith(suffix) for suffix in GENERATED_SUFFIXES):
            generated.append(rel)
        elif any(name in lower for name in GENERATED_NAMES):
            generated.append(rel)
    return generated


def derive_verdicts(
    check_pass: bool,
    claims: list[dict[str, str]],
    generated: list[str],
    root_changed: bool,
    missing_terms: list[dict[str, str]],
) -> list[str]:
    if check_pass:
        return POSITIVE_VERDICTS
    verdicts = ["SECURITY_SUPPLY_CHAIN_GATE_FAILS"]
    if root_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    if generated:
        verdicts.append("GENERATED_ARTIFACT_STAGED")
    if any(item.get("group") == "threat_model" for item in missing_terms):
        verdicts.append("THREAT_MODEL_INCOMPLETE")
    if any(item.get("group") == "progress" for item in missing_terms):
        verdicts.append("PROGRESS_WRITEOUT_INCOMPLETE")
    if any(item.get("term") == "vulnerability database scan not performed in 063" for item in missing_terms):
        verdicts.append("DEPENDENCY_INVENTORY_MISSING")
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

    missing_docs, docs = read_files()
    placeholders = placeholder_hits(docs)
    commands = missing_commands(docs)
    boundary = missing_boundary_tokens(docs)
    claims = forbidden_claim_hits(docs)
    terms = missing_required_terms(docs)
    generated = generated_artifact_staged()
    root_changed = root_license_changed()
    check_pass = not any([missing_docs, placeholders, commands, boundary, claims, terms, generated, root_changed])
    result: dict[str, Any] = {
        "check_pass": check_pass,
        "missing_docs": missing_docs,
        "placeholder_hits": placeholders,
        "missing_commands": commands,
        "missing_boundary_tokens": boundary,
        "forbidden_claim_hits": claims,
        "missing_required_terms": terms,
        "generated_artifact_staged": generated,
        "root_license_changed": root_changed,
        "verdicts": derive_verdicts(check_pass, claims, generated, root_changed, terms),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
