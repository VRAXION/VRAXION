#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
CATCHUP_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_cpu_eval_catchup.py"
PLATEAU_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_plateau_stop.py"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Hot-swap VRA-79 CPU eval lane without touching trainer.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--max-step", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=10)
    # Default to a sub-minute "heartbeat" eval for large hallway models on CPU.
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--anchor-steps", default="")
    ap.add_argument("--anchor-eval-n", type=int, default=0)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--heartbeat-s", type=int, default=20)
    ap.add_argument("--poll-s", type=float, default=2.0)
    ap.add_argument("--queue-max-depth", type=int, default=64)
    ap.add_argument("--backlog-soft-s", type=int, default=1200)
    ap.add_argument("--backlog-hard-s", type=int, default=2400)
    ap.add_argument("--soft-stride-mult", type=int, default=2)
    ap.add_argument("--hard-stride-mult", type=int, default=4)
    ap.add_argument("--recover-low-ratio", type=float, default=0.5)
    ap.add_argument("--recover-ticks", type=int, default=3)
    ap.add_argument("--plateau-poll-s", type=float, default=15.0)
    ap.add_argument("--plateau-warmup-step", type=int, default=2000)
    ap.add_argument("--plateau-window-checkpoints", type=int, default=20)
    ap.add_argument("--plateau-window-gain-threshold", type=float, default=0.0005)
    ap.add_argument("--plateau-medium-window", type=int, default=4)
    ap.add_argument("--plateau-medium-slope-threshold", type=float, default=0.0)
    ap.add_argument("--kill-existing", type=int, default=1, choices=[0, 1])
    return ap.parse_args()


def _spawn_detached(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    start_new_session = False
    if os.name == "nt":
        flags = (
            int(getattr(subprocess, "DETACHED_PROCESS", 0))
            | int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
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
            creationflags=flags,
            start_new_session=start_new_session,
        )
    return int(proc.pid)


def _list_python_rows() -> list[tuple[int, str]]:
    cmd = ['wmic', 'process', 'where', "name='python.exe'", 'get', 'ProcessId,CommandLine']
    cp = subprocess.run(cmd, capture_output=True, text=True)
    rows: list[tuple[int, str]] = []
    for line in (cp.stdout or "").splitlines():
        raw = line.strip()
        if not raw or raw.lower().startswith("commandline"):
            continue
        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            continue
        cmdline = parts[0].strip()
        try:
            pid = int(parts[1].strip())
        except Exception:
            continue
        if pid > 0 and cmdline:
            rows.append((pid, cmdline))
    return rows


def _kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    subprocess.run(["taskkill", "/PID", str(int(pid)), "/F"], capture_output=True, text=True)


def main() -> int:
    args = _parse_args()
    run_root = Path(args.run_root).resolve()
    stop_file = run_root / "stop_now.flag"
    catchup_log = run_root / "catchup.log"
    plateau_log = run_root / "plateau.log"
    if not CATCHUP_TOOL.exists():
        raise FileNotFoundError(str(CATCHUP_TOOL))
    if not PLATEAU_TOOL.exists():
        raise FileNotFoundError(str(PLATEAU_TOOL))

    killed: list[dict[str, object]] = []
    if int(args.kill_existing):
        needles = (
            str(run_root),
            "vra79_cpu_eval_catchup.py",
            "vra79_plateau_stop.py",
            "eval_ckpt_assoc_byte.py",
        )
        for pid, cmdline in _list_python_rows():
            cl = cmdline.replace("/", "\\")
            if needles[0] not in cl:
                continue
            if not any(n in cl for n in needles[1:]):
                continue
            _kill_pid(pid)
            killed.append({"pid": int(pid), "cmd": cmdline})

    catchup_cmd = [
        sys.executable,
        "-u",
        str(CATCHUP_TOOL),
        "--run-root",
        str(run_root),
        "--poll-s",
        str(float(args.poll_s)),
        "--eval-device",
        "cpu",
        "--eval-batch-size",
        str(int(args.eval_batch_size)),
        "--heartbeat-s",
        str(int(args.heartbeat_s)),
        "--force-eval-disjoint",
        "1",
        "--force-eval-subset",
        "0",
        "--stop-on-file",
        str(stop_file),
        "--max-step",
        str(int(args.max_step)),
        "--eval-every",
        str(int(args.eval_every)),
        "--eval-n",
        str(int(args.eval_n)),
        "--anchor-steps",
        str(args.anchor_steps),
        "--anchor-eval-n",
        str(int(args.anchor_eval_n)),
        "--adaptive-mode",
        "1",
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
        str(int(args.max_step)),
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

    catchup_pid = _spawn_detached(catchup_cmd, catchup_log)
    plateau_pid = _spawn_detached(plateau_cmd, plateau_log)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_root": str(run_root),
        "killed": killed,
        "catchup_pid": int(catchup_pid),
        "catchup_log": str(catchup_log),
        "catchup_cmd": catchup_cmd,
        "plateau_pid": int(plateau_pid),
        "plateau_log": str(plateau_log),
        "plateau_cmd": plateau_cmd,
    }
    out_path = run_root / "lane_rewire.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
