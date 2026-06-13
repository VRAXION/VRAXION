#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


MILESTONE = "E60_RUST_CORE_RUNTIME_PARITY_BOOTSTRAP"
DECISIONS = {
    "e60_rust_core_runtime_ready_for_full_bake",
    "e60_rust_core_runtime_not_ready",
    "e60_locked_probe_regression_detected",
    "e60_invalid_artifact_detected",
}
REQ_TARGET = [
    "backend_manifest.json",
    "command_results.json",
    "rust_probe_result.json",
    "locked_probe_sample_checks.json",
    "final_readiness_report.json",
    "decision.json",
    "summary.json",
    "deterministic_replay.json",
    "progress.jsonl",
    "hardware_heartbeat.jsonl",
    "report.md",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def static_check() -> list[str]:
    failures: list[str] = []
    lib = Path("vraxion-runtime/src/lib.rs")
    probe = Path("vraxion-runtime/src/bin/adversarial_probe.rs")
    for path in [lib, probe]:
        if not path.exists():
            failures.append(f"missing Rust source {path}")
            continue
        source = path.read_text(encoding="utf-8")
        if "unsafe" in source:
            failures.append(f"Rust source contains unsafe token: {path}")
    runner = Path("scripts/probes/run_e60_rust_core_runtime_parity_bootstrap.py")
    ast.parse(runner.read_text(encoding="utf-8"))
    cargo = Path("vraxion-runtime/Cargo.toml").read_text(encoding="utf-8")
    if "[dependencies]" in cargo:
        failures.append("vraxion-runtime should remain dependency-free for the minimal core")
    return failures


def validate(out: Path, write_summary: bool) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQ_TARGET:
        if not (out / name).exists():
            failures.append(f"missing artifact {name}")
    if failures:
        result = {"passed": False, "failure_count": len(failures), "failures": failures}
        if write_summary:
            write_json(out / "checker_summary.json", result)
        return result

    failures.extend(static_check())
    manifest = read_json(out / "backend_manifest.json")
    report = read_json(out / "final_readiness_report.json")
    decision = read_json(out / "decision.json")
    summary = read_json(out / "summary.json")
    rust_probe = read_json(out / "rust_probe_result.json")
    commands = read_json(out / "command_results.json")
    sample_checks = read_json(out / "locked_probe_sample_checks.json")
    replay = read_json(out / "deterministic_replay.json")

    if manifest.get("milestone") != MILESTONE:
        failures.append("manifest milestone mismatch")
    if decision.get("decision") != report.get("decision") or summary.get("decision") != report.get("decision"):
        failures.append("decision mismatch")
    if report.get("decision") not in DECISIONS:
        failures.append("invalid decision label")
    if report.get("decision") == "e60_rust_core_runtime_ready_for_full_bake":
        if any(command.get("returncode") != 0 for command in commands):
            failures.append("ready decision but a command failed")
        if any((not check.get("passed")) or check.get("failure_count") != 0 for check in sample_checks):
            failures.append("ready decision but a locked sample check failed")
        if rust_probe.get("passed") is not True:
            failures.append("ready decision but rust probe did not pass")
        for key in ["false_commit", "false_frame", "wrong_feature"]:
            if rust_probe.get(key) != 0:
                failures.append(f"ready decision but rust probe {key}={rust_probe.get(key)}")
        if int(rust_probe.get("cases", 0)) < 100_000:
            failures.append("ready decision but rust adversarial probe case count below 100k")
    if len(sample_checks) != 4:
        failures.append("expected four locked sample checks: E56C/E57/E58/E59")
    for expected in ["e56c_sample_only", "e57_sample_only", "e58_sample_only", "e59_sample_only"]:
        if expected not in {check.get("name") for check in sample_checks}:
            failures.append(f"missing sample check {expected}")

    for name, expected_hash in replay.get("artifact_hashes", {}).items():
        path = out / name
        if not path.exists():
            failures.append(f"replay hash references missing artifact {name}")
        elif file_sha256(path) != expected_hash:
            failures.append(f"deterministic replay hash mismatch for {name}")
    if replay.get("passed") is not True or replay.get("deterministic_replay_match_rate") != 1.0:
        failures.append("deterministic replay did not pass")

    result = {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "decision": report.get("decision"),
        "run_id": report.get("run_id"),
    }
    if write_summary:
        write_json(out / "checker_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e60_rust_core_runtime_parity_bootstrap")
    parser.add_argument("--write-summary", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = validate(Path(args.out), args.write_summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
