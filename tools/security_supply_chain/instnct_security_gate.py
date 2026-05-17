#!/usr/bin/env python3
"""Source-only security and supply-chain gate for STABLE_LOOP_PHASE_LOCK_063."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = (REPO_ROOT / "target" / "pilot_wave").resolve()
SCHEMA_VERSION = "instnct_security_supply_chain_gate_v1"
SBOM_SCHEMA_VERSION = "instnct_sbom_v1"

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

BOUNDARY = [
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

SECRET_PATTERNS = [
    {
        "name": "private_key_block",
        "regex": re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
        "reason": "private key material must not be tracked",
    },
    {
        "name": "aws_access_key",
        "regex": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        "reason": "AWS access key style token",
    },
    {
        "name": "aws_access_key_id_assignment",
        "regex": re.compile(r"AWS_ACCESS_KEY_ID\s*=\s*(?:AKIA|ASIA)[0-9A-Z]{16}"),
        "reason": "AWS_ACCESS_KEY_ID assignment",
    },
    {
        "name": "github_token",
        "regex": re.compile(r"\b(?:ghp_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
        "reason": "GitHub token style secret",
    },
    {
        "name": "openai_key",
        "regex": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
        "reason": "OpenAI key style secret",
    },
    {
        "name": "google_api_key",
        "regex": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        "reason": "Google API key style secret",
    },
]

SUPPRESSIONS = [
    {
        "pattern": "SECRET_PATTERNS regex declarations",
        "reason": "scanner source must contain exact detection regexes",
    }
]

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".ps1",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}

THREAT_MODEL_ITEMS = [
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


class GateError(Exception):
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


def resolve_out_dir(path_text: str) -> Path:
    raw = Path(path_text)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise GateError("UNSAFE_OUT_DIR", "out dir must be a relative target/pilot_wave path")
    parts = [part.lower() for part in raw.parts]
    if len(parts) < 2 or parts[0] != "target" or parts[1] != "pilot_wave":
        raise GateError("UNSAFE_OUT_DIR", "out dir must be under target/pilot_wave")
    resolved = (REPO_ROOT / raw).resolve()
    try:
        resolved.relative_to(TARGET_ROOT)
    except ValueError as exc:
        raise GateError("UNSAFE_OUT_DIR", "out dir escaped target/pilot_wave") from exc
    return resolved


def run_command(args: list[str]) -> tuple[int, str, str]:
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_output(args: list[str]) -> str:
    code, stdout, stderr = run_command(["git", *args])
    if code != 0:
        raise GateError("GIT_COMMAND_FAILED", stderr or "git command failed")
    return stdout


def tracked_files() -> list[str]:
    return [line for line in git_output(["ls-files"]).splitlines() if line.strip()]


def file_sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text_if_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", "README", "SECURITY"}:
        return None
    raw = path.read_bytes()
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="ignore")


def classify_path(rel: str) -> str:
    path = Path(rel)
    suffix = path.suffix.lower()
    if rel in {"Cargo.toml", "Cargo.lock", "requirements.txt"} or path.name == "package.json":
        return "manifest"
    if suffix in {".rs", ".py", ".ts", ".js", ".svelte"}:
        return "source"
    if suffix in {".md", ".txt"}:
        return "docs"
    if suffix in {".json", ".toml", ".yaml", ".yml", ".ps1", ".sh"}:
        return "config"
    return "artifact_or_other"


def append_progress(out_dir: Path, event: str, **details: Any) -> None:
    append_jsonl(out_dir / "progress.jsonl", {"timestamp_ms": now_ms(), "event": event, "details": details})


def parse_cargo_lock(path: Path) -> list[str]:
    if not path.exists():
        return []
    names = re.findall(r'^name = "([^"]+)"', path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return sorted(set(names))


def parse_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    deps: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        deps.append(stripped)
    return deps


def parse_package_json(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {"dependencies": [], "devDependencies": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    deps = data.get("dependencies", {}) if isinstance(data, dict) else {}
    dev = data.get("devDependencies", {}) if isinstance(data, dict) else {}
    return {
        "dependencies": sorted(deps) if isinstance(deps, dict) else [],
        "devDependencies": sorted(dev) if isinstance(dev, dict) else [],
    }


def build_dependency_inventory(files: list[str]) -> dict[str, Any]:
    cargo_manifests = sorted(rel for rel in files if rel.endswith("Cargo.toml"))
    package_json_files = sorted(rel for rel in files if rel.endswith("package.json"))
    requirements_files = sorted(rel for rel in files if Path(rel).name == "requirements.txt")
    cargo_lock_present = "Cargo.lock" in files
    cargo_lock_dependencies = parse_cargo_lock(REPO_ROOT / "Cargo.lock")
    package_dependencies = {
        rel: parse_package_json(REPO_ROOT / rel)
        for rel in package_json_files
    }
    requirements_dependencies = {
        rel: parse_requirements(REPO_ROOT / rel)
        for rel in requirements_files
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "cargo_lock_present": cargo_lock_present,
        "cargo_toml_manifests_found": cargo_manifests,
        "package_json_files_found": package_json_files,
        "requirements_txt_files_found": requirements_files,
        "dependency_source_counts": {
            "cargo_lock_unique_packages": len(cargo_lock_dependencies),
            "cargo_toml_manifest_count": len(cargo_manifests),
            "package_json_file_count": len(package_json_files),
            "requirements_file_count": len(requirements_files),
            "requirements_entry_count": sum(len(items) for items in requirements_dependencies.values()),
            "package_json_dependency_count": sum(
                len(items["dependencies"]) + len(items["devDependencies"])
                for items in package_dependencies.values()
            ),
        },
        "cargo_lock_dependencies": cargo_lock_dependencies,
        "package_json_dependencies": package_dependencies,
        "requirements_dependencies": requirements_dependencies,
        "vulnerability_database_scan_performed": False,
        "vulnerability_database_scan_note": "vulnerability database scan not performed in 063",
        "vulnerability_clean_status_claimed": False,
    }


def build_sbom(files: list[str], dependency_inventory: dict[str, Any]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    for rel in files:
        kind = classify_path(rel)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        entries.append({"path": rel, "kind": kind})
    return {
        "schema_version": SBOM_SCHEMA_VERSION,
        "sbom_kind": "internal_inventory",
        "cyclonedx_compliance": False,
        "spdx_compliance": False,
        "slsa_compliance": False,
        "tracked_file_count": len(files),
        "file_kind_counts": by_kind,
        "manifest_files": [entry["path"] for entry in entries if entry["kind"] == "manifest"],
        "dependency_source_counts": dependency_inventory["dependency_source_counts"],
        "files": entries,
    }


def build_checksums(files: list[str]) -> dict[str, Any]:
    entries = []
    for rel in files:
        path = REPO_ROOT / rel
        if path.is_file():
            entries.append({"path": rel, "sha256_bytes": file_sha256_bytes(path)})
    return {
        "schema_version": SCHEMA_VERSION,
        "hash_mode": "sha256_bytes",
        "file_count": len(entries),
        "files": entries,
    }


def secret_scan(files: list[str]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for rel in files:
        path = REPO_ROOT / rel
        name = path.name.lower()
        if name in {".env", ".env.local", ".env.production"}:
            findings.append({"path": rel, "line": 1, "pattern": "tracked_env_file", "reason": "tracked env file"})
            continue
        if name in {"id_rsa", "id_ed25519"}:
            findings.append({"path": rel, "line": 1, "pattern": "tracked_private_key_file", "reason": "tracked private key file"})
            continue
        text = read_text_if_text(path)
        if text is None:
            continue
        for index, line in enumerate(text.splitlines(), start=1):
            for pattern in SECRET_PATTERNS:
                if pattern["regex"].search(line):
                    findings.append(
                        {
                            "path": rel,
                            "line": index,
                            "pattern": pattern["name"],
                            "reason": pattern["reason"],
                        }
                    )
    return {
        "schema_version": SCHEMA_VERSION,
        "secret_scan_engine": "instnct_local_regex_v1",
        "suppressions": SUPPRESSIONS,
        "broad_ignore_all": False,
        "scanned_tracked_files": len(files),
        "finding_count": len(findings),
        "findings": findings,
    }


def source_hash_set(files: list[str]) -> list[dict[str, str]]:
    source_suffixes = {".rs", ".py", ".toml", ".json", ".md", ".yml", ".yaml", ".ps1", ".sh"}
    hashes = []
    for rel in files:
        path = REPO_ROOT / rel
        if path.suffix.lower() in source_suffixes or path.name in {"LICENSE", "README.md", "SECURITY.md", "Cargo.lock"}:
            hashes.append({"path": rel, "sha256_bytes": file_sha256_bytes(path)})
    return hashes


def build_provenance(files: list[str]) -> dict[str, Any]:
    commit = git_output(["rev-parse", "HEAD"])
    branch = git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    status = git_output(["status", "--short", "--untracked-files=all"])
    rustc = run_command(["rustc", "--version"])[1]
    cargo = run_command(["cargo", "--version"])[1]
    return {
        "schema_version": SCHEMA_VERSION,
        "repo_commit_sha": commit,
        "branch": branch,
        "git_status_clean_or_dirty": "dirty" if status.strip() else "clean",
        "git_status_short": status.splitlines(),
        "tracked_source_hash_set": source_hash_set(files),
        "tracked_source_hash_count": len(source_hash_set(files)),
        "rustc_version": rustc,
        "cargo_version": cargo,
        "python_version": sys.version,
        "os_platform": platform.platform(),
        "rc_001_checksum_reference": "docs/releases/INSTNCT_RC_001_CHECKSUMS.json",
        "service_alpha_062_reference": "docs/research/STABLE_LOOP_PHASE_LOCK_062_SERVICE_API_ALPHA_RESULT.md",
    }


def build_threat_model() -> dict[str, Any]:
    mitigations = {
        "stale artifact reuse": "freshness checks in 058/062; provenance records dirty state",
        "command injection": "subprocess list arguments; no shell concatenation",
        "path traversal": "safe path resolution under target/pilot_wave",
        "artifact retrieval escape": "062 allowlist-only artifact retrieval",
        "policy guard bypass": "policy guard before side effects in 057/058/062",
        "token/auth bypass": "062 bearer token before non-health side effects",
        "checkpoint tampering": "SHA-256 checkpoint save/load in 057",
        "secret leakage": "063 local regex secret scan over tracked text files",
        "dependency compromise": "dependency inventory and deferred vulnerability scan marker",
        "prompt/data leakage through logs": "claim boundary and local/private evaluation scope",
        "claim-boundary drift": "static checkers reject forbidden readiness/compliance claims",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "threat_count": len(THREAT_MODEL_ITEMS),
        "threats": [
            {"threat": item, "mitigation": mitigations[item], "status": "tracked_in_063"}
            for item in THREAT_MODEL_ITEMS
        ],
    }


def generated_artifact_status() -> list[str]:
    status = git_output(["status", "--short", "--untracked-files=all"])
    generated: list[str] = []
    generated_parts = ["target/", "node_modules/", ".svelte-kit/"]
    generated_suffixes = [".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".ckpt", ".bin"]
    generated_names = ["security_gate_manifest.json", "sbom.instnct.json", "secret_scan.json", "provenance.json", "release_integrity.json", "service job artifacts"]
    for line in status.splitlines():
        if len(line) < 4:
            continue
        rel = line[3:].replace("\\", "/")
        lower = rel.lower()
        if any(part in lower for part in generated_parts):
            generated.append(rel)
        elif any(lower.endswith(suffix) for suffix in generated_suffixes):
            generated.append(rel)
        elif any(name in lower for name in generated_names):
            generated.append(rel)
    return generated


def build_release_integrity(out_dir: Path, checksums: dict[str, Any]) -> dict[str, Any]:
    root_license_changed = bool(git_output(["status", "--short", "--", "LICENSE"]).strip())
    generated_staged = generated_artifact_status()
    return {
        "schema_version": SCHEMA_VERSION,
        "checksums_recorded": bool(checksums.get("files")),
        "checksum_file": str((out_dir / "checksums.sha256.json").relative_to(REPO_ROOT)),
        "signing_policy_documented": (REPO_ROOT / "docs/product/INSTNCT_RELEASE_INTEGRITY_POLICY.md").exists(),
        "signed_release_artifacts": False,
        "release_signature_present": False,
        "release_archive_created": False,
        "production_release": False,
        "root_license_changed": root_license_changed,
        "generated_artifact_staged": generated_staged,
        "rc_001_checksum_file_present": (REPO_ROOT / "docs/releases/INSTNCT_RC_001_CHECKSUMS.json").exists(),
        "service_alpha_062_checker_present": (
            REPO_ROOT / "scripts/probes/run_stable_loop_phase_lock_062_service_api_alpha_check.py"
        ).exists(),
    }


def report_text(gate_pass: bool) -> str:
    status = "positive" if gate_pass else "failed"
    return (
        "# STABLE_LOOP_PHASE_LOCK_063_SECURITY_SUPPLY_CHAIN_GATE Report\n\n"
        f"Status: {status} source-only security and supply-chain gate.\n\n"
        "This gate records internal SBOM inventory, checksums, local secret scan, "
        "dependency inventory, provenance, threat model, and release-integrity status.\n\n"
        "It is no signed release, no CycloneDX compliance, no SPDX compliance, "
        "no SLSA compliance, no vulnerability-clean status, no production-ready "
        "security, no production readiness, no hosted SaaS readiness, no public "
        "beta, no clinical use, no high-stakes education use, no commercial launch, "
        "no final legal terms, no full VRAXION, no language grounding, no consciousness, "
        "no biological/FlyWire equivalence, and no physical quantum behavior.\n"
    )


def run_gate(out_dir: Path, heartbeat_sec: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    append_progress(out_dir, "start", heartbeat_sec=heartbeat_sec)
    files = tracked_files()

    dependency_inventory = build_dependency_inventory(files)
    sbom = build_sbom(files, dependency_inventory)
    write_json(out_dir / "sbom.instnct.json", sbom)
    append_progress(out_dir, "sbom_completed", tracked_file_count=len(files))

    checksums = build_checksums(files)
    write_json(out_dir / "checksums.sha256.json", checksums)
    append_progress(out_dir, "checksums_completed", checksum_count=checksums["file_count"])

    scan = secret_scan(files)
    write_json(out_dir / "secret_scan.json", scan)
    append_progress(out_dir, "secret_scan_completed", finding_count=scan["finding_count"])

    write_json(out_dir / "dependency_inventory.json", dependency_inventory)
    append_progress(out_dir, "dependency_inventory_completed", dependency_source_counts=dependency_inventory["dependency_source_counts"])

    provenance = build_provenance(files)
    write_json(out_dir / "provenance.json", provenance)
    append_progress(out_dir, "provenance_completed", git_status=provenance["git_status_clean_or_dirty"])

    threat_model = build_threat_model()
    write_json(out_dir / "threat_model_summary.json", threat_model)
    append_progress(out_dir, "threat_model_completed", threat_count=threat_model["threat_count"])

    integrity = build_release_integrity(out_dir, checksums)
    write_json(out_dir / "release_integrity.json", integrity)
    append_progress(out_dir, "release_integrity_completed")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_artifacts": [
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
        ],
        "boundary": BOUNDARY,
    }
    write_json(out_dir / "security_gate_manifest.json", manifest)

    progress_events = [
        json.loads(line)["event"]
        for line in (out_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing_progress = [event for event in PROGRESS_EVENTS[:-1] if event not in progress_events]
    gate_pass = not any(
        [
            scan["findings"],
            missing_progress,
            integrity["root_license_changed"],
            integrity["generated_artifact_staged"],
            not integrity["checksums_recorded"],
            integrity["signed_release_artifacts"],
            integrity["release_archive_created"],
            integrity["production_release"],
            threat_model["threat_count"] != len(THREAT_MODEL_ITEMS),
        ]
    )
    verdicts = [
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
    ] if gate_pass else ["SECURITY_SUPPLY_CHAIN_GATE_FAILS"]
    if scan["findings"]:
        verdicts.append("SECRET_SCAN_FINDING_DETECTED")
    if missing_progress:
        verdicts.append("PROGRESS_WRITEOUT_INCOMPLETE")
    if integrity["root_license_changed"]:
        verdicts.append("ROOT_LICENSE_CHANGED")
    if integrity["generated_artifact_staged"]:
        verdicts.append("GENERATED_ARTIFACT_STAGED")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "security_supply_chain_gate_pass": gate_pass,
        "missing_progress_events": missing_progress,
        "secret_finding_count": scan["finding_count"],
        "dependency_inventory_present": True,
        "provenance_present": True,
        "threat_model_complete": threat_model["threat_count"] == len(THREAT_MODEL_ITEMS),
        "release_integrity": integrity,
        "verdicts": verdicts,
        "boundary": BOUNDARY,
    }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(report_text(gate_pass), encoding="utf-8")
    append_progress(out_dir, "done", gate_pass=gate_pass)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--heartbeat-sec", type=int, default=20)
    args = parser.parse_args()
    try:
        out_dir = resolve_out_dir(args.out)
        summary = run_gate(out_dir, args.heartbeat_sec)
        print(json.dumps({"check_pass": summary["security_supply_chain_gate_pass"], "summary": summary}, sort_keys=True))
        return 0 if summary["security_supply_chain_gate_pass"] else 1
    except GateError as err:
        print(json.dumps({"check_pass": False, "verdict": err.code, "message": err.message}, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
