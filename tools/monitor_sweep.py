"""Live progress monitor for a Block C sweep running on Modal.

Polls the Modal volume every N seconds and fetches each seed's
progress_seed{S}.json (written after every epoch by the torch script,
committed to the volume by the Modal wrapper's commit-hook watcher).

Run:
    python3 tools/monitor_sweep.py \\
        --remote runs/big_100mb_E96_16ep \\
        --seeds 1,3,7 --poll 30
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import modal


def read_progress(vol, sweep_dir: str, E_guess: int, seed: int):
    """Return (epochs_done, last_curve_row) or (None, None) if not yet written."""
    # We don't know the E dir name, so listdir the sweep dir and pick the
    # first subdir matching 'E*_seed<seed>'. For single-E sweeps this is
    # deterministic.
    target_sub = None
    for e in vol.listdir(sweep_dir):
        if e.type.name == "DIRECTORY" and e.path.endswith(f"_seed{seed}"):
            target_sub = e.path
            break
    if target_sub is None:
        return None, None
    prog_path = f"{target_sub}/progress_seed{seed}.json"
    try:
        buf = io.BytesIO()
        for chunk in vol.read_file(prog_path):
            buf.write(chunk)
        s = json.loads(buf.getvalue().decode())
    except Exception:
        return None, None
    return s.get("epochs_done"), (s["curve"][-1] if s.get("curve") else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="vraxion-block-c")
    ap.add_argument("--remote", required=True,
                    help="Sweep dir under /runs, e.g. 'big_100mb_E96_16ep' "
                         "(without the 'runs/' prefix)")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--E", type=int, default=96)
    ap.add_argument("--poll", type=int, default=30,
                    help="Seconds between polls")
    ap.add_argument("--max-iters", type=int, default=0,
                    help="Stop after N polls; 0 = run forever")
    args = ap.parse_args()

    vol = modal.Volume.from_name(args.volume)
    sweep_dir = f"runs/{args.remote}"
    seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]

    it = 0
    t0 = time.time()
    print(f"Monitoring {sweep_dir} seeds={seed_list} every {args.poll}s")
    while True:
        it += 1
        elapsed = int(time.time() - t0)
        print(f"\n[t+{elapsed:>4}s  poll #{it}]  {time.strftime('%H:%M:%S')}")
        any_done = False
        for seed in seed_list:
            eps_done, last = read_progress(vol, sweep_dir, args.E, seed)
            if eps_done is None:
                print(f"  seed={seed}  (no data yet)")
            else:
                any_done = True
                print(f"  seed={seed}  ep {eps_done:>2d}  "
                      f"train_ce={last['train_ce']:.4f}  "
                      f"test_ce={last['test_ce']:.4f}  "
                      f"acc1={last['acc_top1']:5.2f}%  "
                      f"pair={last['min_pair']:.3f}")
        if args.max_iters and it >= args.max_iters:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
