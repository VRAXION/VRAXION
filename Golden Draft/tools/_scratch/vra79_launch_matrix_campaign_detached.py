#!/usr/bin/env python3
"""Launch matrix campaign detached (no popup console windows)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
CAMPAIGN_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_hashmap_matrix_campaign.py"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detached launcher for VRA-79 matrix campaign.")
    ap.add_argument("--label", default="fibo_vs_homo_det600")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--seeds", default="123,231,312")
    ap.add_argument("--eval-n-realtime", type=int, default=64)
    ap.add_argument("--eval-n-anchor", type=int, default=256)
    ap.add_argument("--anchor-steps", default="200,400,600")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    if not CAMPAIGN_TOOL.exists():
        raise FileNotFoundError(str(CAMPAIGN_TOOL))

    launch_root = REPO_ROOT / "bench_vault" / "_tmp" / "vra79_hashmap_campaign"
    launch_root.mkdir(parents=True, exist_ok=True)
    stdout_log = launch_root / "campaign_stdout.log"
    stderr_log = launch_root / "campaign_stderr.log"

    cmd = [
        sys.executable,
        str(CAMPAIGN_TOOL),
        "--label",
        str(args.label),
        "--device",
        str(args.device),
        "--steps",
        str(int(args.steps)),
        "--checkpoint-every",
        str(int(args.checkpoint_every)),
        "--seeds",
        str(args.seeds),
        "--eval-n-realtime",
        str(int(args.eval_n_realtime)),
        "--eval-n-anchor",
        str(int(args.eval_n_anchor)),
        "--anchor-steps",
        str(args.anchor_steps),
    ]

    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = (
            int(getattr(subprocess, "DETACHED_PROCESS", 0))
            | int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            | int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        )
    else:
        start_new_session = True

    with stdout_log.open("a", encoding="utf-8") as out_h, stderr_log.open("a", encoding="utf-8") as err_h:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=out_h,
            stderr=err_h,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pid": int(proc.pid),
        "cmd": cmd,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    out_path = launch_root / "campaign_launch.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
