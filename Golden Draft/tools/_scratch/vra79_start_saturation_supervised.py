#!/usr/bin/env python3
"""Start one supervised VRA-79 saturation run (parameterized).

This is the safe entrypoint for "uninterrupted" runs:
- Always launches under `vraxion_lab_supervisor.py` (watchdog, logs, restarts).
- Writes only to `bench_vault/_tmp/...` (gitignored) by default.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
SUPERVISOR_TOOL = REPO_ROOT / r"Golden Draft\tools\vraxion_lab_supervisor.py"
ENTRY_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_saturation_entry.py"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Start a supervised VRA-79 saturation run.")
    ap.add_argument("--label", required=True, help="Run label suffix (used in folder name).")
    ap.add_argument("--run-root", default="", help="Optional explicit run root path.")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--ring-len", type=int, required=True)
    ap.add_argument("--slot-dim", type=int, required=True)
    ap.add_argument("--expert-heads", type=int, default=1)
    ap.add_argument(
        "--expert-capacity-split",
        default="",
        help="Optional comma split for non-equal expert capacity (e.g. 0.55,0.34,0.11).",
    )
    ap.add_argument("--expert-capacity-total-mult", type=float, default=1.0)
    ap.add_argument("--expert-capacity-min-hidden", type=int, default=8)
    ap.add_argument("--batch-size", type=int, required=True)

    # Synthetic task knobs (atomic-friendly tests depend on dialing these down).
    ap.add_argument("--synth-len", type=int, default=256)
    ap.add_argument("--assoc-keys", type=int, default=64)
    ap.add_argument("--assoc-pairs", type=int, default=4)
    ap.add_argument("--assoc-val-range", type=int, default=256)

    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--ignore-max-steps", type=int, default=0, choices=[0, 1])
    ap.add_argument("--ignore-wall-clock", type=int, default=1, choices=[0, 1])
    ap.add_argument("--resume", type=int, default=0, choices=[0, 1], help="Resume from train/checkpoint.pt if present.")
    ap.add_argument("--save-every-steps", type=int, default=100)
    ap.add_argument("--eval-every-steps", type=int, default=0)
    ap.add_argument("--eval-at-checkpoint", type=int, default=0, choices=[0, 1])
    ap.add_argument("--eval-samples", type=int, default=512)
    ap.add_argument("--offline-only", type=int, default=1, choices=[0, 1])

    # Supervisor safety knobs.
    ap.add_argument("--watchdog-no-output-s", type=int, default=900)
    ap.add_argument("--poll-interval-s", type=int, default=10)
    ap.add_argument("--max-restarts", type=int, default=1)
    ap.add_argument("--stop-on-file", default="", help="Optional stop sentinel path.")
    ap.add_argument("--detach", type=int, default=1, choices=[0, 1])
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if not SUPERVISOR_TOOL.exists():
        raise FileNotFoundError(f"Missing supervisor tool: {SUPERVISOR_TOOL}")
    if not ENTRY_TOOL.exists():
        raise FileNotFoundError(f"Missing entry tool: {ENTRY_TOOL}")

    if str(args.run_root).strip():
        run_root = Path(str(args.run_root)).resolve()
    else:
        stamp = _utc_stamp()
        run_root = REPO_ROOT / "bench_vault" / "_tmp" / "vra79_saturation_ab" / f"{stamp}-{args.label}"
    run_root.mkdir(parents=True, exist_ok=True)
    job_root = run_root / "supervisor_job"
    launch_log = run_root / "launcher_boot.log"

    child_cmd = [
        sys.executable,
        "-u",
        str(ENTRY_TOOL),
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
        "--expert-capacity-total-mult",
        str(float(args.expert_capacity_total_mult)),
        "--expert-capacity-min-hidden",
        str(int(args.expert_capacity_min_hidden)),
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
        str(int(args.ignore_max_steps)),
        "--ignore-wall-clock",
        str(int(args.ignore_wall_clock)),
        "--resume",
        str(int(args.resume)),
        "--save-every-steps",
        str(int(args.save_every_steps)),
        "--eval-every-steps",
        str(int(args.eval_every_steps)),
        "--eval-at-checkpoint",
        str(int(args.eval_at_checkpoint)),
        "--eval-samples",
        str(int(args.eval_samples)),
        "--offline-only",
        str(int(args.offline_only)),
    ]
    if str(args.expert_capacity_split).strip():
        child_cmd += [
            "--expert-capacity-split",
            str(args.expert_capacity_split).strip(),
        ]

    sup_cmd = [
        sys.executable,
        "-u",
        str(SUPERVISOR_TOOL),
        "--job-name",
        "vra79_saturation",
        "--job-root",
        str(job_root),
        "--wake-window-title",
        "VRAXION_CLI",
        "--wake-after-s",
        "120",
        "--watchdog-no-output-s",
        str(int(args.watchdog_no_output_s)),
        "--watchdog-surge-s",
        "180",
        "--watchdog-spill-s",
        "420",
        "--watchdog-abort-after-kills",
        "3",
        "--poll-interval-s",
        str(int(args.poll_interval_s)),
        "--max-restarts",
        str(int(args.max_restarts)),
    ]
    if str(args.stop_on_file).strip():
        sup_cmd += [
            "--stop-on-file",
            str(args.stop_on_file),
        ]
    sup_cmd += ["--"] + child_cmd

    creationflags = 0
    if os.name == "nt" and int(args.detach):
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        )

    stdout = None
    stderr = None
    start_new_session = False
    if int(args.detach):
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
        start_new_session = (os.name != "nt")

    with launch_log.open("a", encoding="utf-8") as logh:
        proc = subprocess.Popen(
            sup_cmd,
            cwd=str(REPO_ROOT),
            stdout=logh if int(args.detach) else None,
            stderr=subprocess.STDOUT if int(args.detach) else None,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "supervisor_pid": int(proc.pid),
        "run_root": str(run_root),
        "job_root": str(job_root),
        "launch_log": str(launch_log),
        "run_log": str(run_root / r"train\vraxion.log"),
        "child_stdout_log": str(job_root / "child_stdout.log"),
        "child_stderr_log": str(job_root / "child_stderr.log"),
        "supervisor_log": str(job_root / "supervisor.log"),
        "cmd": sup_cmd,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
