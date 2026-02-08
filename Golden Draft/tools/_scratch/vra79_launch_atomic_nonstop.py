#!/usr/bin/env python3
"""Launch the full VRA-79 atomic nonstop pipeline.

Starts detached lanes on a shared run_root:
- GPU trainer under supervisor
- CPU catch-up evaluator
- Plateau stop watcher
- CI gate stop watcher (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
START_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_start_saturation_supervised.py"
CATCHUP_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_cpu_eval_catchup.py"
PLATEAU_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_plateau_stop.py"
CI_STOP_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_ci_gate_stop.py"
DASHBOARD_TOOL = REPO_ROOT / r"Golden Draft\tools\live_dashboard.py"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Launch GPU+CPU+plateau lanes for atomic nonstop training.")
    ap.add_argument("--label", default="atomic_nonstop")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--ring-len", type=int, default=64)
    ap.add_argument("--slot-dim", type=int, default=16)
    ap.add_argument("--expert-heads", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--synth-len", type=int, default=256)
    ap.add_argument("--assoc-keys", type=int, default=64)
    ap.add_argument("--assoc-pairs", type=int, default=4)
    ap.add_argument("--assoc-val-range", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--save-every-steps", type=int, default=10)
    ap.add_argument("--watchdog-no-output-s", type=int, default=900)
    ap.add_argument("--max-restarts", type=int, default=1)

    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=2048)
    ap.add_argument("--anchor-steps", default="")
    ap.add_argument("--anchor-eval-n", type=int, default=0)
    ap.add_argument("--adaptive-mode", type=int, default=1, choices=[0, 1])
    ap.add_argument("--queue-max-depth", type=int, default=64)
    ap.add_argument("--backlog-soft-s", type=int, default=1200)
    ap.add_argument("--backlog-hard-s", type=int, default=2400)
    ap.add_argument("--soft-stride-mult", type=int, default=2)
    ap.add_argument("--hard-stride-mult", type=int, default=4)
    ap.add_argument("--recover-low-ratio", type=float, default=0.50)
    ap.add_argument("--recover-ticks", type=int, default=3)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--eval-heartbeat-s", type=int, default=20)
    ap.add_argument("--eval-poll-s", type=float, default=2.0)
    ap.add_argument("--plateau-poll-s", type=float, default=15.0)
    ap.add_argument("--plateau-warmup-step", type=int, default=2000)
    ap.add_argument("--plateau-window-checkpoints", type=int, default=20)
    ap.add_argument("--plateau-window-gain-threshold", type=float, default=0.0005)
    ap.add_argument("--plateau-medium-window", type=int, default=4)
    ap.add_argument("--plateau-medium-slope-threshold", type=float, default=0.0)
    ap.add_argument("--ci-stop", type=int, default=1, choices=[0, 1])
    ap.add_argument("--ci", type=int, default=95, choices=[95, 99])
    ap.add_argument("--ci-consecutive", type=int, default=3)
    ap.add_argument("--ci-poll-s", type=float, default=10.0)
    ap.add_argument("--dashboard", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dashboard-refresh", type=int, default=2)
    ap.add_argument("--dashboard-max-rows", type=int, default=5000)
    return ap.parse_args()


def _spawn_detached(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        )
    else:
        start_new_session = True
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
    return int(proc.pid)


def _pick_dashboard_port(start: int = 8501, end: int = 8510) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free dashboard port in 8501..8510")


def _run_start(cmd: list[str]) -> dict[str, Any]:
    cp = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if int(cp.returncode) != 0:
        raise RuntimeError(
            "start tool failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"rc: {cp.returncode}\n"
            f"stdout:\n{cp.stdout}\n"
            f"stderr:\n{cp.stderr}"
        )
    out = (cp.stdout or "").strip()
    if not out:
        raise RuntimeError("start tool returned empty stdout")
    return json.loads(out)


def main() -> int:
    args = _parse_args()
    for tool in (START_TOOL, CATCHUP_TOOL, PLATEAU_TOOL):
        if not tool.exists():
            raise FileNotFoundError(f"missing tool: {tool}")
    if int(args.ci_stop) and (not CI_STOP_TOOL.exists()):
        raise FileNotFoundError(f"missing tool: {CI_STOP_TOOL}")

    stamp = _utc_stamp()
    run_root = REPO_ROOT / "bench_vault" / "_tmp" / "vra79_atomic_nonstop" / f"{stamp}-{args.label}"
    run_root.mkdir(parents=True, exist_ok=True)
    stop_file = run_root / "stop_now.flag"

    start_cmd = [
        sys.executable,
        str(START_TOOL),
        "--label",
        str(args.label),
        "--run-root",
        str(run_root),
        "--device",
        str(args.device),
        "--ring-len",
        str(int(args.ring_len)),
        "--slot-dim",
        str(int(args.slot_dim)),
        "--expert-heads",
        str(int(args.expert_heads)),
        "--batch-size",
        str(int(args.batch_size)),
        "--synth-len",
        str(int(args.synth_len)),
        "--assoc-keys",
        str(int(args.assoc_keys)),
        "--assoc-pairs",
        str(int(args.assoc_pairs)),
        "--assoc-val-range",
        str(int(args.assoc_val_range)),
        "--max-steps",
        str(int(args.max_steps)),
        "--ignore-max-steps",
        "0",
        "--ignore-wall-clock",
        "1",
        "--save-every-steps",
        str(int(args.save_every_steps)),
        "--eval-every-steps",
        "0",
        "--eval-at-checkpoint",
        "0",
        "--eval-samples",
        str(int(args.eval_n)),
        "--offline-only",
        "1",
        "--watchdog-no-output-s",
        str(int(args.watchdog_no_output_s)),
        "--max-restarts",
        str(int(args.max_restarts)),
        "--stop-on-file",
        str(stop_file),
        "--detach",
        "1",
    ]
    trainer_info = _run_start(start_cmd)

    catchup_cmd = [
        sys.executable,
        "-u",
        str(CATCHUP_TOOL),
        "--run-root",
        str(run_root),
        "--poll-s",
        str(float(args.eval_poll_s)),
        "--eval-device",
        "cpu",
        "--eval-batch-size",
        str(int(args.eval_batch_size)),
        "--heartbeat-s",
        str(int(args.eval_heartbeat_s)),
        "--force-eval-disjoint",
        "1",
        "--force-eval-subset",
        "0",
        "--stop-on-file",
        str(stop_file),
        "--max-step",
        str(int(args.max_steps)),
        "--eval-every",
        str(int(args.eval_every)),
        "--eval-n",
        str(int(args.eval_n)),
        "--anchor-steps",
        str(args.anchor_steps),
        "--anchor-eval-n",
        str(int(args.anchor_eval_n)),
        "--adaptive-mode",
        str(int(args.adaptive_mode)),
        "--queue-max-depth",
        str(int(args.queue_max_depth)),
        "--backlog-soft-s",
        str(int(args.backlog_soft_s)),
        "--backlog-hard-s",
        str(int(args.backlog_hard_s)),
        "--soft-stride-mult",
        str(int(args.soft_stride_mult)),
        "--hard-stride-mult",
        str(int(args.hard_stride_mult)),
        "--recover-low-ratio",
        str(float(args.recover_low_ratio)),
        "--recover-ticks",
        str(int(args.recover_ticks)),
    ]
    catchup_log = run_root / "catchup.log"
    catchup_pid = _spawn_detached(catchup_cmd, catchup_log)

    plateau_cmd = [
        sys.executable,
        "-u",
        str(PLATEAU_TOOL),
        "--run-root",
        str(run_root),
        "--stop-file",
        str(stop_file),
        "--poll-s",
        str(float(args.plateau_poll_s)),
        "--split",
        "disjoint",
        "--micro-n",
        str(int(args.eval_n)),
        "--medium-n",
        str(int(args.eval_n)),
        "--hard-max-step",
        str(int(args.max_steps)),
        "--warmup-step",
        str(int(args.plateau_warmup_step)),
        "--window-checkpoints",
        str(int(args.plateau_window_checkpoints)),
        "--window-gain-threshold",
        str(float(args.plateau_window_gain_threshold)),
        "--medium-window",
        str(int(args.plateau_medium_window)),
        "--medium-slope-threshold",
        str(float(args.plateau_medium_slope_threshold)),
    ]
    plateau_log = run_root / "plateau.log"
    plateau_pid = _spawn_detached(plateau_cmd, plateau_log)

    ci_info: dict[str, Any] | None = None
    if int(args.ci_stop):
        ci_cmd = [
            sys.executable,
            "-u",
            str(CI_STOP_TOOL),
            "--run-root",
            str(run_root),
            "--stop-file",
            str(stop_file),
            "--poll-s",
            str(float(args.ci_poll_s)),
            "--split",
            "disjoint",
            "--eval-n",
            str(int(args.eval_n)),
            "--ci",
            str(int(args.ci)),
            "--consecutive",
            str(int(args.ci_consecutive)),
            "--hard-max-step",
            str(int(args.max_steps)),
        ]
        ci_log = run_root / "ci_stop.log"
        ci_pid = _spawn_detached(ci_cmd, ci_log)
        ci_info = {
            "pid": int(ci_pid),
            "log": str(ci_log),
            "cmd": ci_cmd,
        }

    dashboard_info: dict[str, Any] | None = None
    if int(args.dashboard):
        if not DASHBOARD_TOOL.exists():
            raise FileNotFoundError(f"missing dashboard tool: {DASHBOARD_TOOL}")
        dash_port = _pick_dashboard_port()
        dash_url = f"http://localhost:{dash_port}"
        dash_log = run_root / "dashboard.log"
        train_log = run_root / r"train\vraxion.log"
        eval_stream_path = run_root / "eval_stream.jsonl"
        eval_status_path = run_root / "eval_catchup_status.json"
        dashboard_cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(DASHBOARD_TOOL),
            "--server.port",
            str(int(dash_port)),
            "--",
            "--log",
            str(train_log),
            "--eval-stream",
            str(eval_stream_path),
            "--eval-status",
            str(eval_status_path),
            "--refresh",
            str(int(args.dashboard_refresh)),
            "--max-rows",
            str(int(args.dashboard_max_rows)),
        ]
        dashboard_pid = _spawn_detached(dashboard_cmd, dash_log)
        dashboard_info = {
            "dashboard_pid": int(dashboard_pid),
            "url": str(dash_url),
            "port": int(dash_port),
            "log": str(dash_log),
            "cmd": dashboard_cmd,
        }
        try:
            webbrowser.open(dash_url)
        except Exception:
            pass

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_root": str(run_root),
        "stop_file": str(stop_file),
        "trainer": trainer_info,
        "catchup_pid": int(catchup_pid),
        "catchup_log": str(catchup_log),
        "catchup_cmd": catchup_cmd,
        "plateau_pid": int(plateau_pid),
        "plateau_log": str(plateau_log),
        "plateau_cmd": plateau_cmd,
        "ci_stop": ci_info,
        "dashboard": dashboard_info,
    }
    manifest_path = run_root / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
