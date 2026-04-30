#!/usr/bin/env python3
"""D10 long-horizon research autopilot.

Runs bounded D10p/D10q-style phases and records resumable status. It is a
research autopilot, not a release or checkpoint-promotion tool.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


D10B_SUMMARY = Path("output/phase_d10b_h384_seed_replication_ladder_20260429/main/run_summary.json")


@dataclass
class Phase:
    name: str
    h: int
    edge_count: int
    eval_len: int
    eval_seeds: str
    proposals_per_arm: int
    arms: str
    kind: str


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_iso(), **event}) + "\n")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def progress_map(status: dict) -> str:
    return "\n".join(
        [
            "GLOBAL AI PLAN",
            "",
            "[1] H384 beta.8 generalist",
            "    DONE",
            "",
            "[2] mechanism",
            "    DONE: edge + threshold co-adaptation",
            "",
            "[3] H384 seed replication",
            f"    {'DONE/AVAILABLE' if status.get('d10b_summary_available') else 'RUNNING/WAITING'}: D10b CPU",
            "",
            "[4] GPU high-H feasibility",
            "    DONE: D10j-D10o",
            "",
            "[5] semantic trust",
            f"    CURRENT: {status.get('last_phase', 'not_started')}",
            "",
            "[6] controlled high-H proof",
            "    NEXT IF D10p PASSES: D10q controlled confirm",
            "",
            "[7] final verdict",
            "    UNIVERSAL / STRUCTURE_DEPENDENT / LOCAL_ONLY / SEMANTIC_BLOCKED",
            "",
        ]
    )


def default_phases(args) -> list[Phase]:
    smoke_arms = "beta8_lifted_v2,motif_no_echo"
    scout_arms = args.arms
    return [
        Phase("D10p_smoke", 8192, 100000, args.eval_len, "988001,988002", args.proposals_per_arm, smoke_arms, "smoke"),
        Phase("D10p_scout_H4096_E25000", 4096, 25000, 128, "988001,988002", 16, scout_arms, "scout"),
        Phase("D10p_scout_H8192_E100000", 8192, 100000, 128, "988001,988002", 16, scout_arms, "scout"),
        Phase("D10p_scout_H16384_E100000", 16384, 100000, 128, "988001,988002", 16, scout_arms, "scout"),
    ]


def command_for_phase(phase: Phase, output_root: Path, device: str) -> list[str]:
    return [
        sys.executable,
        "tools/_scratch/d10p_semantic_projection_hardening.py",
        "--device",
        device,
        "--h",
        str(phase.h),
        "--edge-count",
        str(phase.edge_count),
        "--eval-len",
        str(phase.eval_len),
        "--eval-seeds",
        phase.eval_seeds,
        "--proposals-per-arm",
        str(phase.proposals_per_arm),
        "--arms",
        phase.arms,
        "--out",
        str(output_root / phase.name),
    ]


def write_autopilot_heartbeat(
    status_path: Path,
    events_path: Path,
    status: dict,
    phase_name: str,
    pid: int,
    started: float,
    timeout_s: int,
) -> None:
    elapsed_s = time.perf_counter() - started
    payload = {
        "event": "heartbeat",
        "phase": phase_name,
        "pid": pid,
        "elapsed_s": elapsed_s,
        "timeout_s": timeout_s,
        "d10b_summary_available": D10B_SUMMARY.exists(),
    }
    status["verdict"] = "RUNNING"
    status["active_phase"] = phase_name
    status["active_pid"] = pid
    status["active_elapsed_s"] = elapsed_s
    status["active_timeout_s"] = timeout_s
    status["last_heartbeat_at"] = now_iso()
    status["d10b_summary_available"] = D10B_SUMMARY.exists()
    if D10B_SUMMARY.exists():
        status["d10b_summary_path"] = str(D10B_SUMMARY)
    write_json(status_path, status)
    append_event(events_path, payload)
    print(
        f"[heartbeat] autopilot phase={phase_name} pid={pid} "
        f"elapsed_s={elapsed_s:.1f} timeout_s={timeout_s} "
        f"d10b_summary={D10B_SUMMARY.exists()}",
        flush=True,
    )


def run_command(
    cmd: list[str],
    events_path: Path,
    timeout_s: int,
    status_path: Path,
    status: dict,
    phase_name: str,
    heartbeat_s: int,
) -> dict:
    append_event(events_path, {"event": "command_start", "cmd": cmd})
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    append_event(events_path, {"event": "command_spawned", "phase": phase_name, "pid": proc.pid})
    write_autopilot_heartbeat(status_path, events_path, status, phase_name, proc.pid, started, timeout_s)
    stdout = ""
    stderr = ""
    while True:
        remaining_s = timeout_s - (time.perf_counter() - started)
        if remaining_s <= 0:
            proc.kill()
            out, err = proc.communicate()
            stdout += out or ""
            stderr += err or ""
            elapsed = time.perf_counter() - started
            append_event(
                events_path,
                {
                    "event": "command_timeout",
                    "phase": phase_name,
                    "pid": proc.pid,
                    "elapsed_s": elapsed,
                    "timeout_s": timeout_s,
                },
            )
            return {
                "returncode": -9,
                "elapsed_s": elapsed,
                "stdout": stdout,
                "stderr": stderr + f"\nTIMEOUT after {timeout_s}s",
            }
        try:
            out, err = proc.communicate(timeout=min(max(1, heartbeat_s), remaining_s))
            stdout += out or ""
            stderr += err or ""
            break
        except subprocess.TimeoutExpired:
            write_autopilot_heartbeat(
                status_path,
                events_path,
                status,
                phase_name,
                proc.pid,
                started,
                timeout_s,
            )
            continue
    elapsed = time.perf_counter() - started
    append_event(
        events_path,
        {
            "event": "command_end",
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        },
    )
    return {"returncode": proc.returncode, "elapsed_s": elapsed, "stdout": stdout, "stderr": stderr}


def phase_passes_for_confirm(summary: dict) -> bool:
    if summary.get("verdict") != "D10P_SEMANTIC_PASS":
        return False
    return any(arm.get("arm_verdict") == "SEMANTIC_PASS" for arm in summary.get("arms", []))


def make_confirm_phase(source_phase: Phase) -> Phase:
    return Phase(
        f"D10p_confirm_{source_phase.name}",
        source_phase.h,
        source_phase.edge_count,
        1000,
        "988001,988002,988003,988004",
        source_phase.proposals_per_arm,
        source_phase.arms,
        "confirm",
    )


def run_autopilot(args) -> dict:
    output_root = Path(args.out)
    status_path = output_root / "status.json"
    events_path = output_root / "events.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    status = load_json(status_path) or {
        "started_at": now_iso(),
        "completed_phases": [],
        "phase_results": {},
        "verdict": "RUNNING",
    }
    phases = default_phases(args)
    max_phases = args.max_phases if args.max_phases > 0 else len(phases)
    executed = 0
    append_event(events_path, {"event": "autopilot_start", "max_phases": max_phases})
    for phase in phases:
        if executed >= max_phases:
            break
        if phase.name in status["completed_phases"]:
            continue
        result = run_command(
            command_for_phase(phase, output_root, args.device),
            events_path,
            args.phase_timeout_s,
            status_path,
            status,
            phase.name,
            args.heartbeat_s,
        )
        summary_path = output_root / phase.name / "run_summary.json"
        summary = load_json(summary_path)
        if result["returncode"] != 0 or summary is None:
            status["verdict"] = "AUTOPILOT_PHASE_FAIL"
            status["failed_phase"] = phase.name
            status["last_error"] = result["stderr"][-2000:]
            write_json(status_path, status)
            (output_root / "progress_map.md").write_text(progress_map(status), encoding="utf-8")
            return status
        status["completed_phases"].append(phase.name)
        status["phase_results"][phase.name] = {
            "summary_path": str(summary_path),
            "verdict": summary.get("verdict"),
            "elapsed_s": result["elapsed_s"],
        }
        status["last_phase"] = phase.name
        status["d10b_summary_available"] = D10B_SUMMARY.exists()
        if D10B_SUMMARY.exists():
            status["d10b_summary_path"] = str(D10B_SUMMARY)
        status.pop("active_phase", None)
        status.pop("active_pid", None)
        status.pop("active_elapsed_s", None)
        status.pop("active_timeout_s", None)
        write_json(status_path, status)
        (output_root / "progress_map.md").write_text(progress_map(status), encoding="utf-8")
        append_event(events_path, {"event": "phase_complete", "phase": phase.name, "verdict": summary.get("verdict")})
        executed += 1
        if phase.kind == "scout" and phase_passes_for_confirm(summary) and executed < max_phases:
            confirm = make_confirm_phase(phase)
            result = run_command(
                command_for_phase(confirm, output_root, args.device),
                events_path,
                args.phase_timeout_s,
                status_path,
                status,
                confirm.name,
                args.heartbeat_s,
            )
            confirm_summary_path = output_root / confirm.name / "run_summary.json"
            confirm_summary = load_json(confirm_summary_path)
            if result["returncode"] != 0 or confirm_summary is None:
                status["verdict"] = "AUTOPILOT_CONFIRM_FAIL"
                status["failed_phase"] = confirm.name
                write_json(status_path, status)
                return status
            status["completed_phases"].append(confirm.name)
            status["phase_results"][confirm.name] = {
                "summary_path": str(confirm_summary_path),
                "verdict": confirm_summary.get("verdict"),
                "elapsed_s": result["elapsed_s"],
            }
            status["last_phase"] = confirm.name
            status.pop("active_phase", None)
            status.pop("active_pid", None)
            status.pop("active_elapsed_s", None)
            status.pop("active_timeout_s", None)
            write_json(status_path, status)
            executed += 1
    status["d10b_summary_available"] = D10B_SUMMARY.exists()
    if any(result.get("verdict") == "D10P_SEMANTIC_PASS" for result in status["phase_results"].values()):
        status["verdict"] = "AUTOPILOT_D10P_PASS_READY_FOR_D10Q"
    elif len(status["completed_phases"]) >= max_phases:
        status["verdict"] = "AUTOPILOT_CYCLE_COMPLETE"
    else:
        status["verdict"] = "AUTOPILOT_WAITING"
    status["finished_at"] = now_iso()
    write_json(status_path, status)
    (output_root / "progress_map.md").write_text(progress_map(status), encoding="utf-8")
    append_event(events_path, {"event": "autopilot_end", "verdict": status["verdict"]})
    return status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/phase_d10_autopilot_20260430")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-phases", type=int, default=0)
    parser.add_argument("--eval-len", type=int, default=128)
    parser.add_argument("--proposals-per-arm", type=int, default=4)
    parser.add_argument("--phase-timeout-s", type=int, default=1800)
    parser.add_argument("--heartbeat-s", type=int, default=60)
    parser.add_argument("--arms", default="beta8_lifted_v2,motif_no_echo,projection_tiled,threshold_mid,threshold_high,block_local_projection,frozen_beta8_rows")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_autopilot(args)
    print(json.dumps({"verdict": summary.get("verdict"), "completed_phases": summary.get("completed_phases", [])}, indent=2))
    return 0 if not str(summary.get("verdict", "")).endswith("_FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
