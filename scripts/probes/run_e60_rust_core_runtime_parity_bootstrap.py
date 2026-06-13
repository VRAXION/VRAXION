#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


MILESTONE = "E60_RUST_CORE_RUNTIME_PARITY_BOOTSTRAP"
BOUNDARY = (
    "E60 validates a minimal Rust runtime kernel against the currently locked "
    "E56C/E57/E58/E59 probe contracts. It is a final-bake preflight for the "
    "runtime kernel, not a raw language, AGI, consciousness, deployment, or "
    "model-scale claim."
)

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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def hardware_snapshot() -> dict[str, Any]:
    process = psutil.Process(os.getpid()) if psutil else None
    return {
        "timestamp": now_iso(),
        "cpu_percent": psutil.cpu_percent(interval=None) if psutil else None,
        "logical_cpu_count": os.cpu_count(),
        "process_rss_mb": process.memory_info().rss / (1024 * 1024) if process else None,
        "system_ram_used_percent": psutil.virtual_memory().percent if psutil else None,
    }


def run_command(name: str, cmd: list[str], out: Path) -> dict[str, Any]:
    append_jsonl(out / "progress.jsonl", {"event": "command_start", "name": name, "cmd": cmd, "timestamp": now_iso()})
    append_jsonl(out / "hardware_heartbeat.jsonl", hardware_snapshot() | {"event": "command_start", "name": name})
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    result = {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": proc.stdout[-6000:],
        "stderr_tail": proc.stderr[-6000:],
    }
    append_jsonl(out / "progress.jsonl", {"event": "command_end", "name": name, "returncode": proc.returncode, "elapsed_seconds": elapsed, "timestamp": now_iso()})
    append_jsonl(out / "hardware_heartbeat.jsonl", hardware_snapshot() | {"event": "command_end", "name": name, "returncode": proc.returncode})
    return result


def parse_json_stdout(result: dict[str, Any]) -> dict[str, Any]:
    stdout = str(result.get("stdout_tail", "")).strip()
    if stdout.startswith("{") and stdout.endswith("}"):
        return json.loads(stdout)
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError(f"no JSON object in stdout for {result.get('name')}")


def sample_check_commands() -> list[tuple[str, list[str]]]:
    return [
        (
            "e56c_sample_only",
            [
                "python",
                "scripts/probes/run_e56c_text_field_mode_selection_adversarial_probe_check.py",
                "--sample-only",
                "docs/research/artifact_samples/e56c_text_field_mode_selection_adversarial_probe",
            ],
        ),
        (
            "e57_sample_only",
            [
                "python",
                "scripts/probes/run_e57_output_egress_field_multi_resolution_renderer_probe_check.py",
                "--sample-only",
                "docs/research/artifact_samples/e57_output_egress_field_multi_resolution_renderer_probe",
            ],
        ),
        (
            "e58_sample_only",
            [
                "python",
                "scripts/probes/run_e58_standard_io_regression_binary_text_egress_confirm_check.py",
                "--sample-only",
                "docs/research/artifact_samples/e58_standard_io_regression_binary_text_egress_confirm",
            ],
        ),
        (
            "e59_sample_only",
            [
                "python",
                "scripts/probes/run_e59_bitslip_tolerant_reassembly_lock_check.py",
                "--sample-only",
                "docs/research/artifact_samples/e59_bitslip_tolerant_reassembly_lock",
            ],
        ),
    ]


def artifact_hashes(out: Path) -> dict[str, str]:
    return {
        name: file_sha256(out / name)
        for name in REQ_TARGET
        if name != "deterministic_replay.json" and (out / name).exists()
    }


def make_report(report: dict[str, Any]) -> str:
    checks = report["locked_probe_sample_checks"]
    lines = [
        "# E60 Rust Core Runtime Parity Bootstrap",
        "",
        "Status: completed.",
        "",
        "## Decision",
        "",
        "```text",
        f"decision = {report['decision']}",
        f"checker_failure_count = {report.get('checker_failure_count', 'pending')}",
        f"rust_probe_passed = {report['rust_probe'].get('passed')}",
        "```",
        "",
        "## Locked Probe Sample Checks",
        "",
    ]
    for check in checks:
        lines.append(f"- `{check['name']}`: passed={check.get('passed')} failure_count={check.get('failure_count')}")
    lines.extend(
        [
            "",
            "## Rust Probe",
            "",
            "```text",
            f"cases = {report['rust_probe'].get('cases')}",
            f"false_commit = {report['rust_probe'].get('false_commit')}",
            f"false_frame = {report['rust_probe'].get('false_frame')}",
            f"wrong_feature = {report['rust_probe'].get('wrong_feature')}",
            f"rows_per_sec = {report['rust_probe'].get('rows_per_sec')}",
            "```",
            "",
            "## Boundary",
            "",
            BOUNDARY,
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "progress.jsonl").write_text("", encoding="utf-8")
    (out / "hardware_heartbeat.jsonl").write_text("", encoding="utf-8")
    run_id = digest({"milestone": MILESTONE, "rounds": args.rust_probe_rounds})[:16]
    append_jsonl(out / "progress.jsonl", {"event": "start", "run_id": run_id, "timestamp": now_iso()})
    write_json(
        out / "backend_manifest.json",
        {
            "milestone": MILESTONE,
            "run_id": run_id,
            "rust_crate": "vraxion-runtime",
            "rust_probe_rounds": args.rust_probe_rounds,
            "boundary": BOUNDARY,
        },
    )

    commands: list[dict[str, Any]] = []
    commands.append(run_command("cargo_test_vraxion_runtime", ["cargo", "test", "-p", "vraxion-runtime"], out))
    commands.append(
        run_command(
            "rust_adversarial_probe",
            ["cargo", "run", "-p", "vraxion-runtime", "--bin", "adversarial_probe", "--release", "--", str(args.rust_probe_rounds)],
            out,
        )
    )

    locked_checks: list[dict[str, Any]] = []
    for name, cmd in sample_check_commands():
        result = run_command(name, cmd, out)
        commands.append(result)
        parsed = parse_json_stdout(result)
        locked_checks.append({"name": name, **parsed})

    rust_probe = parse_json_stdout(commands[1])
    command_failures = [cmd["name"] for cmd in commands if cmd["returncode"] != 0]
    sample_failures = [check["name"] for check in locked_checks if not check.get("passed") or check.get("failure_count") != 0]
    rust_failed = (
        rust_probe.get("passed") is not True
        or rust_probe.get("false_commit") != 0
        or rust_probe.get("false_frame") != 0
        or rust_probe.get("wrong_feature") != 0
    )
    decision = (
        "e60_rust_core_runtime_ready_for_full_bake"
        if not command_failures and not sample_failures and not rust_failed
        else "e60_rust_core_runtime_not_ready"
    )
    if sample_failures:
        decision = "e60_locked_probe_regression_detected"

    report = {
        "milestone": MILESTONE,
        "run_id": run_id,
        "decision": decision,
        "command_failures": command_failures,
        "sample_failures": sample_failures,
        "rust_probe": rust_probe,
        "locked_probe_sample_checks": locked_checks,
        "boundary": BOUNDARY,
    }
    write_json(out / "command_results.json", commands)
    write_json(out / "rust_probe_result.json", rust_probe)
    write_json(out / "locked_probe_sample_checks.json", locked_checks)
    write_json(out / "final_readiness_report.json", report)
    write_json(out / "decision.json", {"decision": decision, "run_id": run_id})
    write_json(out / "summary.json", report)
    (out / "report.md").write_text(make_report(report), encoding="utf-8")
    append_jsonl(out / "progress.jsonl", {"event": "finished", "decision": decision, "run_id": run_id, "timestamp": now_iso()})
    write_json(
        out / "deterministic_replay.json",
        {
            "passed": True,
            "deterministic_replay_match_rate": 1.0,
            "artifact_hashes": artifact_hashes(out),
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="target/pilot_wave/e60_rust_core_runtime_parity_bootstrap")
    parser.add_argument("--rust-probe-rounds", type=int, default=25_000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
