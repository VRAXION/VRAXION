#!/usr/bin/env python3
"""D21 medium-horizon A-block memory autopilot.

Runs a bounded, gate-controlled D21E queue. It writes resumable status,
append-only events, a progress map, and a wake trigger. It never commits
generated output and never promotes checkpoints.
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


@dataclass
class Phase:
    name: str
    kind: str
    command: list[str] | None = None
    timeout_s: int = 1800


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": now_iso(), **event}) + "\n")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def progress_map(status: dict) -> str:
    active = status.get("active_phase") or status.get("last_phase") or "not_started"
    return "\n".join(
        [
            "GLOBAL D21 A-BLOCK MEMORY PLAN",
            "",
            "[1] D21A byte IO",
            "    DONE",
            "",
            "[2] D21B context write lane",
            "    DONE",
            "",
            "[3] D21C one-step recurrent state",
            "    DONE",
            "",
            "[4] D21D marker-memory recall",
            "    DONE",
            "",
            "[5] D21E multi-slot / key-value memory",
            f"    CURRENT: {active}",
            "",
            "[6] D21F mini sequence reasoning",
            "    NEXT IF D21E PASSES",
            "",
            f"verdict: {status.get('verdict', 'RUNNING')}",
            "",
        ]
    )


def write_wake(output_root: Path, status: dict, reason: str) -> None:
    write_json(
        output_root / "wake_trigger.json",
        {
            "ts": now_iso(),
            "reason": reason,
            "verdict": status.get("verdict"),
            "active_phase": status.get("active_phase"),
            "last_phase": status.get("last_phase"),
            "status_path": str(output_root / "status.json"),
            "progress_map": str(output_root / "progress_map.md"),
        },
    )


def persist(output_root: Path, status: dict, reason: str | None = None) -> None:
    write_json(output_root / "status.json", status)
    (output_root / "progress_map.md").write_text(progress_map(status), encoding="utf-8")
    if reason:
        write_wake(output_root, status, reason)


def d21e_command(args, mode: str, out_dir: Path, extra: list[str]) -> list[str]:
    return [
        sys.executable,
        "tools/_scratch/d21e_multislot_memory_ablock_core.py",
        "--mode",
        mode,
        "--out",
        str(out_dir),
        *extra,
    ]


def build_phases(args, output_root: Path) -> list[Phase]:
    d21e_root = output_root / "d21e"
    eval_sequences = str(args.eval_sequences)
    smoke_eval = str(max(1024, min(args.eval_sequences, 4096)))
    atlas_samples = str(args.samples)
    atlas_eval = str(max(1024, min(args.eval_sequences, 8192)))
    crystallize_eval = str(max(2048, min(args.eval_sequences, 32768)))
    confirm_eval = str(max(4096, min(args.confirm_eval_sequences, 131072)))
    return [
        Phase("preflight", "preflight"),
        Phase(
            "D21E_baseline",
            "d21e",
            d21e_command(args, "baseline-check", d21e_root / "baseline", []),
            args.phase_timeout_s,
        ),
        Phase(
            "D21E_oracle",
            "d21e",
            d21e_command(
                args,
                "multislot-oracle",
                d21e_root / "oracle",
                [
                    "--slot-counts",
                    args.slot_counts,
                    "--distractor-lengths",
                    args.distractor_lengths,
                    "--eval-sequences",
                    eval_sequences,
                    "--state-dim",
                    args.oracle_state_dim,
                    "--memory-edge-budget",
                    args.oracle_edge_budget,
                ],
            ),
            args.oracle_timeout_s,
        ),
        Phase(
            "D21E_atlas",
            "d21e",
            d21e_command(
                args,
                "memory-atlas",
                d21e_root / "atlas",
                [
                    "--state-dims",
                    args.state_dims,
                    "--slot-counts",
                    args.slot_counts,
                    "--memory-edge-budgets",
                    args.memory_edge_budgets,
                    "--distractor-lengths",
                    args.distractor_lengths,
                    "--samples",
                    atlas_samples,
                    "--eval-sequences",
                    atlas_eval,
                    "--workers",
                    str(args.workers),
                ],
            ),
            args.atlas_timeout_s,
        ),
        Phase(
            "D21E_crystallize",
            "d21e",
            d21e_command(
                args,
                "crystallize-memory",
                d21e_root / "crystallize",
                [
                    "--state-dim",
                    args.crystallize_state_dim,
                    "--slot-counts",
                    args.slot_counts,
                    "--memory-edge-budget",
                    args.crystallize_edge_budget,
                    "--distractor-lengths",
                    args.distractor_lengths,
                    "--max-steps",
                    str(args.max_steps),
                    "--eval-sequences",
                    crystallize_eval,
                ],
            ),
            args.crystallize_timeout_s,
        ),
        Phase(
            "D21E_confirm",
            "d21e",
            d21e_command(
                args,
                "confirm",
                d21e_root / "confirm",
                [
                    "--candidates",
                    str(d21e_root / "atlas" / "memory_candidates.csv"),
                    "--top-k",
                    str(args.top_k),
                    "--slot-counts",
                    args.slot_counts,
                    "--distractor-lengths",
                    args.confirm_distractor_lengths,
                    "--eval-sequences",
                    confirm_eval,
                ],
            ),
            args.confirm_timeout_s,
        ),
    ]


def run_subprocess(phase: Phase, output_root: Path, status: dict, args) -> dict:
    events_path = output_root / "events.jsonl"
    append_event(events_path, {"event": "phase_start", "phase": phase.name, "cmd": phase.command})
    started = time.perf_counter()
    proc = subprocess.Popen(phase.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    status["active_phase"] = phase.name
    status["active_pid"] = proc.pid
    status["verdict"] = "RUNNING"
    persist(output_root, status, "phase_start")
    stdout = ""
    stderr = ""
    while True:
        elapsed = time.perf_counter() - started
        remaining = phase.timeout_s - elapsed
        if remaining <= 0:
            proc.kill()
            out, err = proc.communicate()
            stdout += out or ""
            stderr += err or ""
            result = {"returncode": -9, "elapsed_s": time.perf_counter() - started, "stdout": stdout, "stderr": stderr}
            append_event(events_path, {"event": "phase_timeout", "phase": phase.name, "elapsed_s": result["elapsed_s"]})
            return result
        try:
            out, err = proc.communicate(timeout=min(args.heartbeat_s, remaining))
            stdout += out or ""
            stderr += err or ""
            break
        except subprocess.TimeoutExpired:
            status["active_elapsed_s"] = time.perf_counter() - started
            status["last_heartbeat_at"] = now_iso()
            persist(output_root, status, "heartbeat")
            append_event(
                events_path,
                {
                    "event": "heartbeat",
                    "phase": phase.name,
                    "pid": proc.pid,
                    "elapsed_s": status["active_elapsed_s"],
                    "timeout_s": phase.timeout_s,
                },
            )
    result = {"returncode": proc.returncode, "elapsed_s": time.perf_counter() - started, "stdout": stdout, "stderr": stderr}
    append_event(
        events_path,
        {
            "event": "phase_end",
            "phase": phase.name,
            "returncode": result["returncode"],
            "elapsed_s": result["elapsed_s"],
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        },
    )
    return result


def run_preflight(output_root: Path, status: dict, args) -> dict:
    checks = [
        [sys.executable, "-m", "py_compile", "tools/_scratch/d21d_marker_memory_ablock_core.py"],
        [sys.executable, "-m", "py_compile", "tools/_scratch/d21e_multislot_memory_ablock_core.py"],
        [sys.executable, "-m", "py_compile", "tools/_scratch/d21_medium_horizon_autopilot.py"],
        [sys.executable, "tools/check_public_surface.py"],
    ]
    required = [
        Path("tools/_scratch/d21a_reciprocal_byte_ablock.py"),
        Path("tools/_scratch/d21b_context_ablock.py"),
        Path("tools/_scratch/d21c_tiny_recurrent_ablock_core.py"),
        Path("tools/_scratch/d21d_marker_memory_ablock_core.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    status["active_phase"] = "preflight"
    status["verdict"] = "RUNNING"
    persist(output_root, status, "phase_start")
    if missing:
        return {"ok": False, "error": f"missing required paths: {missing}"}
    for cmd in checks:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=args.preflight_timeout_s)
        append_event(
            output_root / "events.jsonl",
            {
                "event": "preflight_check",
                "cmd": cmd,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1000:],
                "stderr_tail": result.stderr[-1000:],
            },
        )
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr[-2000:] or result.stdout[-2000:]}
    return {"ok": True}


def phase_summary_path(output_root: Path, phase: Phase) -> Path:
    if phase.name == "preflight":
        return output_root / "status.json"
    return output_root / "d21e" / phase.name.replace("D21E_", "") / "memory_top.json"


def verdict_is_pass(verdict: str) -> bool:
    return verdict in {"D21E_BASELINE_REPRODUCED", "D21E_MULTISLOT_MEMORY_PASS"}


def run_autopilot(args) -> dict:
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    status = load_json(output_root / "status.json") or {
        "started_at": now_iso(),
        "completed_phases": [],
        "phase_results": {},
        "verdict": "RUNNING",
    }
    phases = build_phases(args, output_root)
    max_phases = args.max_phases if args.max_phases > 0 else len(phases)
    started = time.perf_counter()
    append_event(output_root / "events.jsonl", {"event": "autopilot_start", "max_phases": max_phases, "budget_s": args.budget_s})
    for phase in phases[:max_phases]:
        if time.perf_counter() - started > args.budget_s:
            status["verdict"] = "D21_AUTOPILOT_BUDGET_EXHAUSTED"
            persist(output_root, status, "budget_exhausted")
            return status
        if phase.name in status["completed_phases"]:
            continue
        if phase.kind == "preflight":
            result = run_preflight(output_root, status, args)
            if not result["ok"]:
                status["verdict"] = "D21_AUTOPILOT_PREFLIGHT_FAIL"
                status["failed_phase"] = phase.name
                status["last_error"] = result["error"]
                persist(output_root, status, "fail")
                return status
            phase_verdict = "PASS"
        else:
            result = run_subprocess(phase, output_root, status, args)
            summary = load_json(phase_summary_path(output_root, phase))
            if result["returncode"] != 0 or summary is None:
                status["verdict"] = "D21_AUTOPILOT_PHASE_FAIL"
                status["failed_phase"] = phase.name
                status["last_error"] = result["stderr"][-2000:] or result["stdout"][-2000:]
                persist(output_root, status, "fail")
                return status
            phase_verdict = summary.get("verdict", "UNKNOWN")
            if phase.name == "D21E_oracle" and phase_verdict != "D21E_MULTISLOT_MEMORY_PASS":
                status["verdict"] = "D21E_ORACLE_FAIL"
                status["failed_phase"] = phase.name
                persist(output_root, status, "decision")
                return status
            if phase.name == "D21E_atlas" and phase_verdict not in {"D21E_MULTISLOT_MEMORY_PASS", "D21E_2SLOT_ONLY", "D21E_WEAK_PASS"}:
                status["verdict"] = phase_verdict
                status["failed_phase"] = phase.name
                persist(output_root, status, "decision")
                return status
            if phase.name == "D21E_confirm" and phase_verdict != "D21E_MULTISLOT_MEMORY_PASS":
                status["verdict"] = phase_verdict
                status["failed_phase"] = phase.name
                persist(output_root, status, "decision")
                return status
        status["completed_phases"].append(phase.name)
        status["phase_results"][phase.name] = {"verdict": phase_verdict}
        status["last_phase"] = phase.name
        status.pop("active_phase", None)
        status.pop("active_pid", None)
        persist(output_root, status, "phase_complete")
        append_event(output_root / "events.jsonl", {"event": "phase_complete", "phase": phase.name, "verdict": phase_verdict})
    status["verdict"] = "D21E_AUTOPILOT_PASS" if "D21E_confirm" in status["completed_phases"] else "D21_AUTOPILOT_CYCLE_COMPLETE"
    status["finished_at"] = now_iso()
    persist(output_root, status, "done")
    append_event(output_root / "events.jsonl", {"event": "autopilot_end", "verdict": status["verdict"]})
    return status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/phase_d21_medium_horizon_autopilot_20260502")
    parser.add_argument("--max-phases", type=int, default=0)
    parser.add_argument("--budget-s", type=int, default=21_600)
    parser.add_argument("--heartbeat-s", type=int, default=300)
    parser.add_argument("--phase-timeout-s", type=int, default=900)
    parser.add_argument("--oracle-timeout-s", type=int, default=1800)
    parser.add_argument("--atlas-timeout-s", type=int, default=7200)
    parser.add_argument("--crystallize-timeout-s", type=int, default=7200)
    parser.add_argument("--confirm-timeout-s", type=int, default=7200)
    parser.add_argument("--preflight-timeout-s", type=int, default=300)
    parser.add_argument("--slot-counts", default="2,4")
    parser.add_argument("--distractor-lengths", default="1,2,4,8,16")
    parser.add_argument("--confirm-distractor-lengths", default="1,2,4,8,16,32")
    parser.add_argument("--eval-sequences", type=int, default=65_536)
    parser.add_argument("--confirm-eval-sequences", type=int, default=131_072)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--state-dims", default="32,64,128")
    parser.add_argument("--memory-edge-budgets", default="16,32,64,96")
    parser.add_argument("--oracle-state-dim", default="64")
    parser.add_argument("--oracle-edge-budget", default="64")
    parser.add_argument("--crystallize-state-dim", default="64")
    parser.add_argument("--crystallize-edge-budget", default="64")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=16)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_autopilot(args)
    print(json.dumps({"verdict": summary.get("verdict"), "completed_phases": summary.get("completed_phases", [])}, indent=2), flush=True)
    verdict = str(summary.get("verdict", ""))
    return 2 if verdict.endswith("_FAIL") or verdict.endswith("_EXHAUSTED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
