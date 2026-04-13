#!/usr/bin/env python3
"""Retry the 3 stalled balanced bit heads (3, 5, 6) with multiple search seeds."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_PATH = ROOT / "target" / "release" / "examples" / "c19_grower_snapshot.exe"
OUT_ROOT = ROOT / "target" / "c19_grid3_copy"

STALLED_BITS = [3, 5, 6]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def run_one(bit_idx: int, seed: int) -> dict:
    out_dir = OUT_ROOT / f"bit_{bit_idx}_seed{seed}"
    log_file = OUT_ROOT / "logs" / f"bit_{bit_idx}_seed{seed}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(BIN_PATH),
        "--task", f"grid3_copy_bit_{bit_idx}",
        "--search-seed", str(seed),
        "--out-dir", str(out_dir),
        "--max-neurons", "6",
    ]
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(args, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"bit": bit_idx, "seed": seed, "status": "FAILED", "elapsed_s": elapsed}
    final_path = out_dir / "final.json"
    if not final_path.exists():
        return {"bit": bit_idx, "seed": seed, "status": "NO_FINAL", "elapsed_s": elapsed}
    d = json.loads(final_path.read_text())
    neurons = d.get("neurons", [])
    return {
        "bit": bit_idx,
        "seed": seed,
        "status": "OK",
        "val": d.get("ensemble_val", 0.0),
        "test": d.get("ensemble_test", 0.0),
        "neurons": len(neurons),
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 72)
    print(f"Retrying balanced bits {STALLED_BITS} with seeds {SEEDS}")
    print("=" * 72)
    results = {b: [] for b in STALLED_BITS}
    t0 = time.time()
    for bit in STALLED_BITS:
        for seed in SEEDS:
            print(f"[{time.strftime('%H:%M:%S')}] bit {bit} seed {seed} ...", flush=True)
            r = run_one(bit, seed)
            results[bit].append(r)
            if r["status"] == "OK":
                print(f"  val={r['val']:5.2f}%  test={r['test']:5.2f}%  N={r['neurons']}  {r['elapsed_s']:.1f}s", flush=True)
            else:
                print(f"  {r['status']}", flush=True)
    total_elapsed = time.time() - t0
    print()
    print(f"=== DONE in {total_elapsed:.1f}s ===")
    print()
    for bit in STALLED_BITS:
        best = max((r for r in results[bit] if r["status"] == "OK"), key=lambda r: r["val"], default=None)
        if best:
            print(f"bit {bit} best: val={best['val']:.2f}%  test={best['test']:.2f}%  seed={best['seed']}  N={best['neurons']}")
        else:
            print(f"bit {bit}: no successful runs")
    print()
    # Save all results for analysis
    (OUT_ROOT / "balanced_retry.json").write_text(json.dumps(results, indent=2))
    print(f"Details: {OUT_ROOT / 'balanced_retry.json'}")


if __name__ == "__main__":
    main()
