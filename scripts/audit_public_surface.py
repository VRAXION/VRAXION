#!/usr/bin/env python3
"""Public surface audit for VRAXION.

This script scans tracked files only. It reports paths and fixed reason codes,
not file contents. It is intended to prevent private artifacts from entering the
public repository.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

MAX_TEXT_BYTES = 2_000_000
PUBLIC_STUB_MARKER = "PUBLIC_SAFE_STUB"


@dataclass(frozen=True)
class Finding:
    severity: str
    path: str
    reason: str


TEXT_SCAN_EXEMPT_PATHS = {
    "scripts/audit_public_surface.py",
}

FORBIDDEN_GLOBS = [
    "docs/research/artifact_samples/**",
    "frontier_runs/**",
    "frontier_experiments/**",
    "**/*.jsonl",
    "**/*.parquet",
    "**/*.arrow",
    "**/*.sqlite",
    "**/*.db",
    "**/.env",
    "**/.env.*",
]

OPERATIONAL_STUB_OR_FAIL_GLOBS = [
    "scripts/probes/run_e18*",
    "scripts/probes/run_e19*",
    "scripts/probes/run_e20*",
    "scripts/probes/run_e21*",
    "scripts/probes/run_e22*",
    "scripts/probes/run_e23*",
    "scripts/probes/run_e113_fineweb*",
    "scripts/probes/run_e114_fineweb*",
    "scripts/probes/run_e119_fineweb*",
    "scripts/probes/run_e120_fineweb*",
    "scripts/probes/run_e123_orange_baseline_fineweb*",
    "scripts/tools/prepare_fineweb*",
]

FORBIDDEN_PATH_SUBSTRINGS = [
    "frontier_runs",
    "frontier_experiments",
    "operator_candidate_manifest",
    "operator_activation_ledger",
    "dataset_manifest.json",
    "reader_manifest.json",
    "resume_state.json",
]

FORBIDDEN_TEXT_PATTERNS = [
    ("windows_dataset_path", re.compile(r"S:\\", re.IGNORECASE)),
    ("messy_training_data_path", re.compile(r"MESSY\s+TRAINING\s+DATA", re.IGNORECASE)),
    ("private_frontier_name", re.compile(r"VRAXION[_-]frontier", re.IGNORECASE)),
    ("dataset_manifest_reference", re.compile(r"\bdataset_manifest\.json\b", re.IGNORECASE)),
    ("reader_manifest_reference", re.compile(r"\breader_manifest\.json\b", re.IGNORECASE)),
    ("operator_candidate_manifest_reference", re.compile(r"\boperator_candidate_manifest\b", re.IGNORECASE)),
    ("operator_activation_ledger_reference", re.compile(r"\boperator_activation_ledger\b", re.IGNORECASE)),
    ("resume_state_reference", re.compile(r"\bresume_state\.json\b", re.IGNORECASE)),
    ("generic_credential_assignment", re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\s*[:=]")),
]

WARN_PATH_PATTERNS = [
    ("probe_script", "scripts/probes/**"),
    ("training_script", "**/*training*"),
    ("research_result_doc", "docs/research/*RESULT*.md"),
    ("implementation_heavy_runtime", "vraxion-runtime/src/bin/**"),
]

WARN_TEXT_PATTERNS = [
    ("fineweb_reference", re.compile(r"\bFineWeb\b|\bfineweb\b")),
    ("large_exact_metric_block", re.compile(r"(?m)^\s*[a-zA-Z0-9_ -]{3,80}\s*=\s*[0-9][0-9,_.]*(\.[0-9]+)?\s*$")),
]


def run_git_ls_files() -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def posix(path: str) -> str:
    return str(PurePosixPath(path.replace(os.sep, "/")))


def matches_glob(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern)


def read_text_sample(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_TEXT_BYTES + 1)
    except OSError:
        return None

    if b"\x00" in data:
        return None
    if len(data) > MAX_TEXT_BYTES:
        data = data[:MAX_TEXT_BYTES]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def has_public_stub_marker(path: str) -> bool:
    text = read_text_sample(path)
    return text is not None and PUBLIC_STUB_MARKER in text


def audit_path(path: str) -> list[Finding]:
    findings: list[Finding] = []
    p = posix(path)

    for pattern in FORBIDDEN_GLOBS:
        if matches_glob(p, pattern):
            findings.append(Finding("FAIL", p, f"forbidden_glob:{pattern}"))

    for pattern in OPERATIONAL_STUB_OR_FAIL_GLOBS:
        if matches_glob(p, pattern):
            if has_public_stub_marker(path):
                findings.append(Finding("WARN", p, f"public_safe_stub:{pattern}"))
            else:
                findings.append(Finding("FAIL", p, f"operational_script_not_stubbed:{pattern}"))

    for substring in FORBIDDEN_PATH_SUBSTRINGS:
        if substring.lower() in p.lower():
            findings.append(Finding("FAIL", p, f"forbidden_path:{substring}"))

    for reason, pattern in WARN_PATH_PATTERNS:
        if matches_glob(p, pattern):
            findings.append(Finding("WARN", p, reason))

    return findings


def audit_text(path: str) -> list[Finding]:
    findings: list[Finding] = []
    p = posix(path)
    if p in TEXT_SCAN_EXEMPT_PATHS:
        return findings

    text = read_text_sample(path)
    if text is None:
        return findings

    for reason, regex in FORBIDDEN_TEXT_PATTERNS:
        if regex.search(text):
            findings.append(Finding("FAIL", p, reason))

    for reason, regex in WARN_TEXT_PATTERNS:
        if regex.search(text):
            findings.append(Finding("WARN", p, reason))

    return findings


def main() -> int:
    try:
        paths = run_git_ls_files()
    except Exception as exc:
        print(f"FAIL audit_runtime git_ls_files_failed {type(exc).__name__}", file=sys.stderr)
        return 2

    findings: list[Finding] = []
    for path in paths:
        findings.extend(audit_path(path))
        findings.extend(audit_text(path))

    fail_count = sum(1 for item in findings if item.severity == "FAIL")
    warn_count = sum(1 for item in findings if item.severity == "WARN")

    print("PUBLIC_SURFACE_AUDIT")
    print(f"tracked_files={len(paths)}")
    print(f"failure_count={fail_count}")
    print(f"warning_count={warn_count}")

    for item in findings:
        print(f"{item.severity} {item.reason} {item.path}")

    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
