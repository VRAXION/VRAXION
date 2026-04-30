#!/usr/bin/env python3
"""D10 release-ready research autopilot.

Runs a bounded, gate-controlled D10r-v2 -> D10s queue. It writes resumable
status, append-only events, a progress map, and a wake trigger. It never commits
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


REQUIRED_PATHS = [
    Path("output/releases/v5.0.0-beta.8/seed2042_improved_generalist_v1.ckpt"),
    Path("output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_2042/final.ckpt"),
    Path("output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_42/final.ckpt"),
    Path("output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_1042/final.ckpt"),
    Path("output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_3042/final.ckpt"),
    Path("output/phase_d7_operator_bandit_20260427/H_384/D7_BASELINE/seed_4042/final.ckpt"),
]
FULL_CONTROLS = (
    "random_label,random_bigram,unigram_decoy,projection_shuffle,"
    "projection_reinit,time_shuffle,state_shuffle,no_network_random_state"
)


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
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_iso(), **event}) + "\n")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def progress_map(status: dict) -> str:
    active = status.get("active_phase") or status.get("last_phase") or "not_started"
    return "\n".join(
        [
            "GLOBAL RELEASE-READY AI PLAN",
            "",
            "[1] beta.8 H384 generalist",
            "    DONE",
            "",
            "[2] causal mechanism",
            "    DONE: edge + threshold co-adaptation",
            "",
            "[3] seed replication",
            "    DONE: D10b no broad replication",
            "",
            "[4] evaluator hardening",
            f"    CURRENT: {active}",
            "        |-- D10r-v2 pass -> D10s wiring-prior sweep",
            "        '-- D10r-v2 fail -> projection/readout redesign; high-H blocked",
            "",
            "[5] D10s wiring-prior sweep",
            "    GATED behind D10r-v2",
            "",
            "[6] H512 / H8192",
            "    BLOCKED until trusted non-seed2042 D10s signal",
            "",
            f"verdict: {status.get('verdict', 'RUNNING')}",
            "",
        ]
    )


def write_wake(output_root: Path, status: dict, reason: str) -> None:
    payload = {
        "ts": now_iso(),
        "reason": reason,
        "verdict": status.get("verdict"),
        "active_phase": status.get("active_phase"),
        "last_phase": status.get("last_phase"),
        "status_path": str(output_root / "status.json"),
        "progress_map": str(output_root / "progress_map.md"),
    }
    write_json(output_root / "wake_trigger.json", payload)


def persist(output_root: Path, status: dict, reason: str | None = None) -> None:
    write_json(output_root / "status.json", status)
    (output_root / "progress_map.md").write_text(progress_map(status), encoding="utf-8")
    if reason:
        write_wake(output_root, status, reason)


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
        [sys.executable, "-m", "py_compile", "tools/_scratch/d10_release_ready_autopilot.py"],
        [sys.executable, "-m", "py_compile", "tools/_scratch/d10r_hardened_eval.py"],
        [sys.executable, "-m", "py_compile", "tools/_scratch/d10s_wiring_prior_sweep.py"],
        [sys.executable, "tools/check_public_surface.py"],
    ]
    missing = [str(path) for path in REQUIRED_PATHS if not path.exists()]
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


def d10r_command(args, phase_name: str, eval_len: int, eval_seeds: str, control_repeats: int, output_root: Path) -> list[str]:
    return [
        sys.executable,
        "tools/_scratch/d10r_hardened_eval.py",
        "--device",
        args.device,
        "--eval-len",
        str(eval_len),
        "--eval-seeds",
        eval_seeds,
        "--controls",
        FULL_CONTROLS,
        "--control-repeats",
        str(control_repeats),
        "--max-charge",
        "7",
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--permutation-samples",
        str(args.permutation_samples),
        "--out",
        str(output_root / phase_name),
    ]


def d10s_command(args, phase_name: str, eval_len: int, eval_seeds: str, proposals: int, output_root: Path) -> list[str]:
    return [
        sys.executable,
        "tools/_scratch/d10s_wiring_prior_sweep.py",
        "--device",
        args.device,
        "--eval-len",
        str(eval_len),
        "--eval-seeds",
        eval_seeds,
        "--controls",
        "random_label,unigram_decoy,state_shuffle,no_network_random_state",
        "--control-repeats",
        str(args.d10s_control_repeats),
        "--proposals-per-arm",
        str(proposals),
        "--max-charge",
        "7",
        "--out",
        str(output_root / phase_name),
    ]


def build_phases(args, output_root: Path) -> list[Phase]:
    return [
        Phase("preflight", "preflight"),
        Phase(
            "D10r_v2_state_shuffle_smoke",
            "d10r",
            d10r_command(args, "D10r_v2_state_shuffle_smoke", args.eval_len, "970001,970002,970003,970004", args.control_repeats, output_root),
            args.phase_timeout_s,
        ),
        Phase(
            "D10r_v2_state_shuffle_main",
            "d10r",
            d10r_command(
                args,
                "D10r_v2_state_shuffle_main",
                1000,
                "970001,970002,970003,970004,970005,970006,970007,970008,970009,970010,970011,970012,970013,970014,970015,970016",
                8,
                output_root,
            ),
            args.d10r_main_timeout_s,
        ),
        Phase(
            "D10s_wiring_prior_smoke",
            "d10s",
            d10s_command(args, "D10s_wiring_prior_smoke", 1000, "970001,970002,970003,970004,970005,970006,970007,970008", 32, output_root),
            args.d10s_timeout_s,
        ),
    ]


def phase_summary_path(output_root: Path, phase: Phase) -> Path:
    if phase.kind == "d10r":
        return output_root / phase.name / "d10r_run_summary.json"
    return output_root / phase.name / "run_summary.json"


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
            status["verdict"] = "AUTOPILOT_BUDGET_EXHAUSTED"
            persist(output_root, status, "budget_exhausted")
            return status
        if phase.name in status["completed_phases"]:
            continue
        if phase.kind == "preflight":
            result = run_preflight(output_root, status, args)
            if not result["ok"]:
                status["verdict"] = "AUTOPILOT_PREFLIGHT_FAIL"
                status["failed_phase"] = phase.name
                status["last_error"] = result["error"]
                persist(output_root, status, "fail")
                return status
            phase_verdict = "PASS"
        else:
            result = run_subprocess(phase, output_root, status, args)
            summary = load_json(phase_summary_path(output_root, phase))
            if result["returncode"] != 0 or summary is None:
                status["verdict"] = "AUTOPILOT_PHASE_FAIL"
                status["failed_phase"] = phase.name
                status["last_error"] = result["stderr"][-2000:]
                persist(output_root, status, "fail")
                return status
            phase_verdict = summary.get("verdict", "UNKNOWN")
            if phase.name == "D10r_v2_state_shuffle_smoke" and summary.get("checkpoint_summaries", [{}])[0].get("real_mo_delta_mean", 0.0) <= 0.0:
                status["verdict"] = "D10R_V2_SMOKE_REAL_SIGNAL_FAIL"
                status["failed_phase"] = phase.name
                persist(output_root, status, "decision")
                return status
            if phase.name == "D10r_v2_state_shuffle_main" and phase_verdict != "D10R_TRUST_PASS":
                status["verdict"] = "D10R_V2_PROJECTION_READOUT_BLOCKED"
                status["failed_phase"] = phase.name
                persist(output_root, status, "decision")
                return status
            if phase.name == "D10s_wiring_prior_smoke" and phase_verdict != "D10S_REPLICABLE_WIRING_PRIOR_SIGNAL":
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
    status["verdict"] = "AUTOPILOT_READY_FOR_H512_PLANNING" if "D10s_wiring_prior_smoke" in status["completed_phases"] else "AUTOPILOT_CYCLE_COMPLETE"
    status["finished_at"] = now_iso()
    persist(output_root, status, "done")
    append_event(output_root / "events.jsonl", {"event": "autopilot_end", "verdict": status["verdict"]})
    return status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output/phase_d10_release_ready_autopilot_20260430")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-phases", type=int, default=0)
    parser.add_argument("--budget-s", type=int, default=14_400)
    parser.add_argument("--heartbeat-s", type=int, default=300)
    parser.add_argument("--phase-timeout-s", type=int, default=1_800)
    parser.add_argument("--d10r-main-timeout-s", type=int, default=9_000)
    parser.add_argument("--d10s-timeout-s", type=int, default=7_200)
    parser.add_argument("--preflight-timeout-s", type=int, default=300)
    parser.add_argument("--eval-len", type=int, default=256)
    parser.add_argument("--control-repeats", type=int, default=4)
    parser.add_argument("--d10s-control-repeats", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--permutation-samples", type=int, default=5000)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = run_autopilot(args)
    print(json.dumps({"verdict": summary.get("verdict"), "completed_phases": summary.get("completed_phases", [])}, indent=2), flush=True)
    return 0 if not str(summary.get("verdict", "")).endswith("_FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
