#!/usr/bin/env python3
"""c19_copy_multihead_run.py - run c19_grower on all 9 grid3_copy_bit_N heads
and aggregate the per-head + full-grid exact-match results.

Uses a pre-built c19_grower binary (snapshot so concurrent Rust rebuilds in
other agents do not interfere). Writes per-head logs under
target/c19_grid3_copy/logs/ and an overall summary.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "target" / "release" / "examples" / "c19_grower_snapshot.exe"
OUT_ROOT = ROOT / "target" / "c19_grid3_copy"


def run_one_head(bit_idx: int) -> dict:
    task = f"grid3_copy_bit_{bit_idx}"
    out_dir = OUT_ROOT / f"bit_{bit_idx}"
    log_file = OUT_ROOT / "logs" / f"bit_{bit_idx}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(BIN_PATH),
        "--task", task,
        "--search-seed", "0",
        "--out-dir", str(out_dir),
        "--max-neurons", "4",
    ]
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(args, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"bit": bit_idx, "status": "FAILED", "returncode": proc.returncode, "elapsed_s": elapsed}
    final_path = out_dir / "final.json"
    if not final_path.exists():
        return {"bit": bit_idx, "status": "NO_FINAL", "elapsed_s": elapsed}
    d = json.loads(final_path.read_text())
    neurons = d.get("neurons", [])
    depth = max((n.get("tick", 0) for n in neurons), default=0)
    return {
        "bit": bit_idx,
        "status": "OK",
        "val": d.get("ensemble_val", 0.0),
        "test": d.get("ensemble_test", 0.0),
        "train": d.get("ensemble_train", 0.0),
        "neurons": len(neurons),
        "depth": depth,
        "elapsed_s": elapsed,
    }


def main():
    if not BIN_PATH.exists():
        print(f"ERROR: snapshot binary not found: {BIN_PATH}")
        return 1
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("c19_copy_multihead_run - 9 bit heads on grid3_copy_bit_[0..8]")
    print("=" * 72)
    t0 = time.time()
    results = []
    for i in range(9):
        print(f"[{time.strftime('%H:%M:%S')}] starting bit {i}", flush=True)
        r = run_one_head(i)
        results.append(r)
        if r["status"] == "OK":
            print(f"  bit {i}: val={r['val']:6.2f}%  test={r['test']:6.2f}%  "
                  f"train={r['train']:6.2f}%  N={r['neurons']:<2} depth={r['depth']:<2}  "
                  f"{r['elapsed_s']:5.1f}s", flush=True)
        else:
            print(f"  bit {i}: {r['status']}", flush=True)
    total_elapsed = time.time() - t0
    ok = [r for r in results if r["status"] == "OK"]
    print()
    print(f"=== DONE in {total_elapsed:.1f}s ===")
    print()
    print("bit  val     test    train   N    depth")
    for r in results:
        if r["status"] == "OK":
            print(f"  {r['bit']}  {r['val']:6.2f}  {r['test']:6.2f}  {r['train']:6.2f}  {r['neurons']:<3}  {r['depth']}")
        else:
            print(f"  {r['bit']}  {r['status']}")
    if ok:
        total_n = sum(r["neurons"] for r in ok)
        max_d = max(r["depth"] for r in ok)
        mean_val = sum(r["val"] for r in ok) / len(ok)
        mean_test = sum(r["test"] for r in ok) / len(ok)
        heads_100 = sum(1 for r in ok if r["val"] >= 99.999)
        print()
        print(f"Total neurons (all 9 heads): {total_n}")
        print(f"Max depth (any head):        {max_d}")
        print(f"Mean val  (per head):        {mean_val:.2f}%")
        print(f"Mean test (per head):        {mean_test:.2f}%")
        print(f"Heads at 100% val:           {heads_100}/9")
    summary = {
        "task_family": "grid3_copy_v1",
        "n_heads": 9,
        "results": results,
        "elapsed_total_s": total_elapsed,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"Summary: {OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
