#!/usr/bin/env python3
"""Launch the VRA-79 Fibonacci frequency colony pipeline (GPU nonstop + CPU eval catch-up).

This launcher supports two Fibonacci modes:
- Frequency-only: equal experts + non-uniform router_map.
- Capacity+frequency: non-equal expert capacities + non-uniform router_map.

Pipeline:
1) Init a 1-step checkpoint (no supervisor) to generate checkpoint.pt.
2) Rewrite router_map in checkpoint.pt to a halving bucket distribution.
3) Launch the main GPU trainer under supervisor (resume=1).
4) Launch CPU eval catch-up lane (detached, no popups).
5) Launch Streamlit dashboard (detached, no popups) and open browser tab.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(r"S:\AI\work\VRAXION_DEV")
ENTRY_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_saturation_entry.py"
START_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_start_saturation_supervised.py"
ROUTER_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_router_map_fibo.py"
CATCHUP_TOOL = REPO_ROOT / r"Golden Draft\tools\_scratch\vra79_cpu_eval_catchup.py"
DASHBOARD_TOOL = REPO_ROOT / r"Golden Draft\tools\live_dashboard.py"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Launch the Fibonacci frequency nonstop pipeline.")
    ap.add_argument("--label", default="fibo_freq", help="Run label (folder suffix).")
    ap.add_argument("--seed", type=int, default=123, help="Run seed tag (encoded as seed### in folder name).")
    ap.add_argument("--run-root", default="", help="Optional explicit run root (must include seed### for stable eval).")

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--ring-len", type=int, required=True)
    ap.add_argument("--slot-dim", type=int, required=True)
    ap.add_argument("--expert-heads", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)

    # Task knobs.
    ap.add_argument("--synth-len", type=int, default=256)
    ap.add_argument("--assoc-keys", type=int, default=64)
    ap.add_argument("--assoc-pairs", type=int, default=4)
    ap.add_argument("--assoc-val-range", type=int, default=256)

    # Init stage.
    # NOTE: instnct_train_wallclock increments `step` before the MAX_STEPS break,
    # and checkpoint saves happen after that break. With MAX_STEPS=1, no ckpt is
    # written. We need at least 2 to guarantee a saved checkpoint.
    ap.add_argument("--init-steps", type=int, default=2)
    ap.add_argument("--router-buckets", type=int, default=7)
    ap.add_argument(
        "--router-ratio",
        default="",
        help="Optional comma ratio for router weights (e.g. 0.55,0.34,0.11).",
    )
    ap.add_argument("--router-permute-seed", type=int, default=12345)
    ap.add_argument(
        "--capacity-split",
        default="",
        help="Optional comma split for non-equal expert capacity (e.g. 0.55,0.34,0.11).",
    )
    ap.add_argument("--capacity-total-mult", type=float, default=1.0)
    ap.add_argument("--capacity-min-hidden", type=int, default=8)

    # Main stage.
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--save-every-steps", type=int, default=10)
    ap.add_argument("--watchdog-no-output-s", type=int, default=900)
    ap.add_argument("--max-restarts", type=int, default=0)

    # CPU eval lane.
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--anchor-steps", default="200,400,600")
    ap.add_argument("--anchor-eval-n", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--eval-heartbeat-s", type=int, default=20)
    ap.add_argument("--eval-poll-s", type=float, default=2.0)
    ap.add_argument("--adaptive-mode", type=int, default=1, choices=[0, 1])

    # Dashboard.
    ap.add_argument("--dashboard", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dashboard-refresh", type=int, default=2)
    ap.add_argument("--dashboard-max-rows", type=int, default=8000)
    return ap.parse_args()


def _ratio_len(raw: str) -> int:
    return len([p for p in str(raw).split(",") if p.strip()])


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


def _run_json(cmd: list[str]) -> Dict[str, Any]:
    cp = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if int(cp.returncode) != 0:
        raise RuntimeError(
            "command failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"rc: {cp.returncode}\n"
            f"stdout:\n{cp.stdout}\n"
            f"stderr:\n{cp.stderr}"
        )
    out = (cp.stdout or "").strip()
    if not out:
        raise RuntimeError("command returned empty stdout (expected JSON)")
    return json.loads(out)


def _wait_for_checkpoint(ckpt_path: Path, timeout_s: int = 300) -> None:
    t0 = time.time()
    while True:
        if ckpt_path.exists():
            return
        if (time.time() - t0) > float(timeout_s):
            raise TimeoutError(f"Timed out waiting for checkpoint: {ckpt_path}")
        time.sleep(0.25)


def main() -> int:
    args = _parse_args()
    for tool in (ENTRY_TOOL, START_TOOL, ROUTER_TOOL, CATCHUP_TOOL):
        if not tool.exists():
            raise FileNotFoundError(f"Missing tool: {tool}")

    if str(args.run_root).strip():
        run_root = Path(str(args.run_root)).resolve()
    else:
        stamp = _utc_stamp()
        # IMPORTANT: include seed### in folder name so eval uses a stable dataset.
        run_root = (
            REPO_ROOT
            / "bench_vault"
            / "_tmp"
            / "vra79_fibo_freq"
            / f"{stamp}-{args.label}_seed{int(args.seed)}"
        )
    run_root.mkdir(parents=True, exist_ok=True)

    # Separate stop controls:
    # - stop_gpu.flag stops the supervisor (GPU trainer).
    # - stop_eval.flag stops the CPU eval lane after it drains its queue.
    stop_gpu_file = run_root / "stop_gpu.flag"
    stop_eval_file = run_root / "stop_eval.flag"
    train_root = run_root / "train"
    ckpt_path = train_root / "checkpoint.pt"
    router_buckets = int(args.router_buckets)
    if str(args.router_ratio).strip():
        ratio_count = _ratio_len(args.router_ratio)
        if ratio_count <= 0:
            raise ValueError("--router-ratio provided but empty after parsing")
        router_buckets = int(ratio_count)

    # Stage 1: init checkpoint (foreground, 1 step).
    init_max_steps = max(2, int(args.init_steps))
    init_cmd = [
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
        str(float(args.capacity_total_mult)),
        "--expert-capacity-min-hidden",
        str(int(args.capacity_min_hidden)),
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
        str(int(init_max_steps)),
        "--ignore-max-steps",
        "0",
        "--ignore-wall-clock",
        "1",
        "--resume",
        "0",
        "--save-every-steps",
        "1",
        "--eval-every-steps",
        "0",
        "--eval-at-checkpoint",
        "0",
        "--eval-samples",
        "0",
        "--offline-only",
        "1",
    ]
    if str(args.capacity_split).strip():
        init_cmd += [
            "--expert-capacity-split",
            str(args.capacity_split).strip(),
        ]
    init_log = run_root / "init_stage.log"
    init_log.parent.mkdir(parents=True, exist_ok=True)
    init_env = dict(os.environ)
    # Critical: otherwise INSTNCT synth mode loops forever (intended for nonstop runs).
    init_env["VRX_SYNTH_ONCE"] = "1"
    with init_log.open("a", encoding="utf-8") as handle:
        cp = subprocess.run(init_cmd, cwd=str(REPO_ROOT), env=init_env, stdout=handle, stderr=subprocess.STDOUT)
    if int(cp.returncode) != 0:
        raise RuntimeError(f"Init stage failed (rc={cp.returncode}). See: {init_log}")
    _wait_for_checkpoint(ckpt_path, timeout_s=300)

    # Stage 2: rewire router_map (in-place).
    rewire_cmd = [
        sys.executable,
        str(ROUTER_TOOL),
        "--in-ckpt",
        str(ckpt_path),
        "--buckets",
        str(int(router_buckets)),
        "--permute-seed",
        str(int(args.router_permute_seed)),
    ]
    if str(args.router_ratio).strip():
        rewire_cmd += [
            "--ratio",
            str(args.router_ratio).strip(),
        ]
    rewire_info = _run_json(rewire_cmd)

    # Stage 3: launch supervised main training (detached).
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
        "--expert-capacity-total-mult",
        str(float(args.capacity_total_mult)),
        "--expert-capacity-min-hidden",
        str(int(args.capacity_min_hidden)),
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
        "--resume",
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
        str(stop_gpu_file),
        "--detach",
        "1",
    ]
    if str(args.capacity_split).strip():
        start_cmd += [
            "--expert-capacity-split",
            str(args.capacity_split).strip(),
        ]
    trainer_info = _run_json(start_cmd)

    # Stage 4: CPU eval catch-up lane (detached).
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
        str(stop_eval_file),
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
    ]
    catchup_log = run_root / "catchup.log"
    catchup_pid = _spawn_detached(catchup_cmd, catchup_log)

    dashboard_info: Dict[str, Any] | None = None
    if int(args.dashboard):
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
        "stop_gpu_file": str(stop_gpu_file),
        "stop_eval_file": str(stop_eval_file),
        "init_log": str(init_log),
        "rewire": rewire_info,
        "trainer": trainer_info,
        "catchup_pid": int(catchup_pid),
        "catchup_log": str(catchup_log),
        "catchup_cmd": catchup_cmd,
        "dashboard": dashboard_info,
    }
    manifest_path = run_root / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
