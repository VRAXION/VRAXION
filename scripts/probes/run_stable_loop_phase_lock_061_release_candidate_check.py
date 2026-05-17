#!/usr/bin/env python3
"""Static checker for STABLE_LOOP_PHASE_LOCK_061 release candidate package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKSUM_FILE = "docs/releases/INSTNCT_RC_001_CHECKSUMS.json"

REQUIRED_DOCS = [
    "docs/releases/INSTNCT_RC_001_RELEASE_MANIFEST.md",
    "docs/releases/INSTNCT_RC_001_INSTALL_GUIDE.md",
    "docs/releases/INSTNCT_RC_001_SMOKE_TEST_GUIDE.md",
    "docs/releases/INSTNCT_RC_001_KNOWN_LIMITATIONS.md",
    "docs/releases/INSTNCT_RC_001_SUPPORT_BOUNDARY.md",
    CHECKSUM_FILE,
    "docs/releases/INSTNCT_RC_001_DOC_INDEX.md",
    "docs/releases/INSTNCT_RC_001_CLAIM_BOUNDARY.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_061_RELEASE_CANDIDATE_PACKAGE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_061_RELEASE_CANDIDATE_PACKAGE_RESULT.md",
]

REQUIRED_HASH_TARGETS = [
    "docs/releases/INSTNCT_RC_001_RELEASE_MANIFEST.md",
    "docs/releases/INSTNCT_RC_001_INSTALL_GUIDE.md",
    "docs/releases/INSTNCT_RC_001_SMOKE_TEST_GUIDE.md",
    "docs/releases/INSTNCT_RC_001_KNOWN_LIMITATIONS.md",
    "docs/releases/INSTNCT_RC_001_SUPPORT_BOUNDARY.md",
    "docs/releases/INSTNCT_RC_001_DOC_INDEX.md",
    "docs/releases/INSTNCT_RC_001_CLAIM_BOUNDARY.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_061_RELEASE_CANDIDATE_PACKAGE_CONTRACT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_061_RELEASE_CANDIDATE_PACKAGE_RESULT.md",
    "scripts/probes/run_stable_loop_phase_lock_061_release_candidate_check.py",
    "docs/research/STABLE_LOOP_PHASE_LOCK_050_EXPECTED_HASHES.json",
    "docs/research/STABLE_LOOP_PHASE_LOCK_050_REPRODUCIBILITY_PACKAGE_AND_PAPER_AUDIT_RESULT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_055_VISUAL_SECTION_CLOSURE_REAL_RUN_REPLAY_RESULT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_056_PRODUCTIZATION_ARCHITECTURE_AND_LICENSE_BOUNDARY_RESULT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_057_INSTNCT_SDK_RELEASE_CANDIDATE_RESULT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_058_DEPLOYMENT_HARNESS_RESULT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_059_HOSPITAL_SCHOOL_PILOT_BOUNDARY_RESULT.md",
    "docs/research/STABLE_LOOP_PHASE_LOCK_060_LICENSE_PACKAGE_RESULT.md",
    "tools/instnct_deploy/instnct_deploy.py",
    "instnct-core/examples/instnct_sdk_candidate_smoke.rs",
]

DOC_INDEX_TOKENS = [
    "050 repro/audit package",
    "055 visual closure",
    "056 productization boundary",
    "057 SDK release candidate",
    "058 deployment harness",
    "059 pilot boundary",
    "060 license package drafts",
]

EXACT_COMMANDS = [
    "cargo check -p instnct-core --example instnct_sdk_candidate_smoke",
    "cargo test -p instnct-core sdk_candidate",
    "python tools/instnct_deploy/instnct_deploy.py validate-config --config tools/instnct_deploy/config/example.local.json",
    "python tools/instnct_deploy/instnct_deploy.py healthcheck --config tools/instnct_deploy/config/example.local.json",
    "powershell -ExecutionPolicy Bypass -File tools/instnct_deploy/smoke.ps1 -Config tools/instnct_deploy/config/example.local.json -Out target/pilot_wave/stable_loop_phase_lock_061_release_candidate_package/smoke",
]

PLACEHOLDERS = ["TODO", "TBD", "placeholder", "pending measured", "coming soon"]

FORBIDDEN_CLAIMS = {
    "PRODUCTION_READY_CLAIM_DETECTED": [
        "generally available",
        "GA release",
        "production ready",
        "production-ready",
        "ready for production",
        "production deployment",
        "hosted SaaS launch",
        "clinical ready",
        "high-stakes education ready",
    ],
    "PUBLIC_BETA_CLAIM_DETECTED": ["public beta"],
    "FINAL_LEGAL_TERMS_CLAIM_DETECTED": ["final legal terms"],
}

NEGATION_MARKERS = [
    "not ",
    "no ",
    "does not ",
    "do not ",
    "without ",
    "unsupported",
    "non-",
]

GENERATED_PATH_PARTS = [
    "target/",
    "node_modules/",
    ".svelte-kit/",
]

GENERATED_SUFFIXES = [
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".7z",
    ".ckpt",
    ".bin",
]

GENERATED_NAME_PARTS = [
    "checkpoint",
    "sdk_smoke/",
    "smoke/",
]

POSITIVE_VERDICTS = [
    "RELEASE_CANDIDATE_PACKAGE_POSITIVE",
    "RELEASE_MANIFEST_WRITTEN",
    "INSTALL_GUIDE_WRITTEN",
    "SMOKE_TEST_GUIDE_WRITTEN",
    "CHECKSUMS_WRITTEN",
    "DOC_INDEX_WRITTEN",
    "KNOWN_LIMITATIONS_WRITTEN",
    "SUPPORT_BOUNDARY_WRITTEN",
    "CLAIM_BOUNDARY_WRITTEN",
    "ROOT_LICENSE_UNCHANGED",
    "PRODUCTION_READY_NOT_CLAIMED",
    "PUBLIC_BETA_NOT_CLAIMED",
    "FINAL_LEGAL_TERMS_NOT_CLAIMED",
]


def normalize_lf_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
        if rel.endswith(".json"):
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
    if any(marker in prefix for marker in NEGATION_MARKERS):
        return True
    return phrase_start > 0 and line[phrase_start - 1] == "_"


def forbidden_claim_hits(docs: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in docs.items():
        if rel.endswith(".json"):
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


def missing_commands(docs: dict[str, str]) -> list[str]:
    smoke_doc = docs.get("docs/releases/INSTNCT_RC_001_SMOKE_TEST_GUIDE.md", "")
    lines = {line.strip() for line in smoke_doc.splitlines()}
    return [command for command in EXACT_COMMANDS if command not in lines]


def missing_doc_references(docs: dict[str, str]) -> list[str]:
    doc_index = docs.get("docs/releases/INSTNCT_RC_001_DOC_INDEX.md", "")
    missing = [token for token in DOC_INDEX_TOKENS if token not in doc_index]
    for rel in REQUIRED_HASH_TARGETS:
        if rel not in "\n".join(docs.values()):
            missing.append(rel)
    return missing


def checksum_mismatches() -> list[dict[str, str]]:
    path = REPO_ROOT / CHECKSUM_FILE
    if not path.exists():
        return [{"path": CHECKSUM_FILE, "error": "missing"}]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [{"path": CHECKSUM_FILE, "error": f"json_decode:{exc}"}]

    mismatches: list[dict[str, str]] = []
    if payload.get("hash_mode") != "sha256_normalized_lf":
        mismatches.append({"path": CHECKSUM_FILE, "error": "hash_mode"})
    if payload.get("self_excluded") is not True:
        mismatches.append({"path": CHECKSUM_FILE, "error": "self_excluded"})

    entries = payload.get("files", [])
    if not isinstance(entries, list):
        return [{"path": CHECKSUM_FILE, "error": "files_not_list"}]

    seen: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            mismatches.append({"path": CHECKSUM_FILE, "error": "entry_not_object"})
            continue
        rel = str(entry.get("path", ""))
        expected = str(entry.get("sha256_normalized_lf", ""))
        if not rel or not expected:
            mismatches.append({"path": rel or CHECKSUM_FILE, "error": "missing_path_or_hash"})
            continue
        if rel == CHECKSUM_FILE:
            mismatches.append({"path": rel, "error": "checksum_file_must_be_self_excluded"})
            continue
        seen[rel] = expected
        target = REPO_ROOT / rel
        if not target.exists():
            mismatches.append({"path": rel, "error": "listed_file_missing"})
            continue
        observed = normalize_lf_sha256(target)
        if observed != expected:
            mismatches.append({"path": rel, "expected": expected, "observed": observed})

    for rel in REQUIRED_HASH_TARGETS:
        if rel not in seen:
            mismatches.append({"path": rel, "error": "required_hash_target_missing"})
    return mismatches


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
        if len(line) < 4:
            continue
        paths.add(line[3:].replace("\\", "/"))

    generated: list[str] = []
    for rel in sorted(paths):
        lower = rel.lower()
        if any(part in lower for part in GENERATED_PATH_PARTS):
            generated.append(rel)
            continue
        if any(lower.endswith(suffix) for suffix in GENERATED_SUFFIXES):
            generated.append(rel)
            continue
        if rel.startswith("docs/releases/INSTNCT_RC_001_") and any(
            part in lower for part in GENERATED_NAME_PARTS
        ):
            generated.append(rel)
    return generated


def support_boundary_missing(docs: dict[str, str]) -> list[str]:
    support = docs.get("docs/releases/INSTNCT_RC_001_SUPPORT_BOUNDARY.md", "")
    required = [
        "No SLA",
        "No production support",
        "No clinical/high-stakes support",
        "RC support is best-effort / evaluation only",
        "External support requires separate written agreement",
    ]
    return [term for term in required if term not in support]


def derive_verdicts(
    check_pass: bool,
    root_changed: bool,
    command_missing: list[str],
    doc_refs_missing: list[str],
    checksum_errors: list[dict[str, str]],
    claim_hits: list[dict[str, str]],
    generated: list[str],
) -> list[str]:
    if check_pass:
        return POSITIVE_VERDICTS
    verdicts = ["RELEASE_CANDIDATE_PACKAGE_FAILS"]
    if root_changed:
        verdicts.append("ROOT_LICENSE_CHANGED")
    if command_missing:
        verdicts.append("RELEASE_COMMAND_MISSING")
    if doc_refs_missing:
        verdicts.append("RELEASE_DOC_INDEX_INCOMPLETE")
    if checksum_errors:
        verdicts.append("RELEASE_CHECKSUM_MISMATCH")
    if generated:
        verdicts.append("GENERATED_ARTIFACT_STAGED")
    for hit in claim_hits:
        verdict = hit["verdict"]
        if verdict not in verdicts:
            verdicts.append(verdict)
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only:
        parser.error("only --check-only is supported")

    missing_docs, docs = read_docs()
    root_changed = root_license_changed()
    placeholders = placeholder_hits(docs)
    command_missing = missing_commands(docs)
    doc_refs_missing = missing_doc_references(docs)
    checksum_errors = checksum_mismatches()
    claim_hits = forbidden_claim_hits(docs)
    generated = generated_artifact_staged()
    support_missing = support_boundary_missing(docs)
    if support_missing:
        doc_refs_missing.extend(f"support_boundary:{term}" for term in support_missing)

    check_pass = not any(
        [
            root_changed,
            missing_docs,
            placeholders,
            command_missing,
            doc_refs_missing,
            checksum_errors,
            claim_hits,
            generated,
        ]
    )

    result: dict[str, Any] = {
        "check_pass": check_pass,
        "root_license_changed": root_changed,
        "missing_docs": missing_docs,
        "placeholder_hits": placeholders,
        "missing_commands": command_missing,
        "missing_doc_references": doc_refs_missing,
        "checksum_mismatches": checksum_errors,
        "forbidden_claim_hits": claim_hits,
        "generated_artifact_staged": generated,
        "verdicts": derive_verdicts(
            check_pass,
            root_changed,
            command_missing,
            doc_refs_missing,
            checksum_errors,
            claim_hits,
            generated,
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if check_pass else 1


if __name__ == "__main__":
    sys.exit(main())
