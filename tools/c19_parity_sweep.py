#!/usr/bin/env python3
"""c19_parity_sweep.py — 20-seed c19_grower sweep on grid3_full_parity.

Calls the release binary directly (avoids cargo lock contention).
Writes per-seed logs to target/c19_parity_sweep/logs/ and a final summary.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "target" / "c19_parity_sweep"
BIN_PATH = ROOT / "target" / "release" / "examples" / "c19_grower.exe"

BASELINE_VAL = 76.20  # grid3_curriculum best (seed 4, 28 neurons, depth 12)


def run_one(seed: int) -> dict:
    out_dir = SWEEP_DIR / f"seed_{seed:02d}"
    log_file = SWEEP_DIR / "logs" / f"seed_{seed:02d}.log"
    args = [
        str(BIN_PATH),
        "--task", "grid3_full_parity",
        "--search-seed", str(seed),
        "--out-dir", str(out_dir),
        "--max-neurons", "32",
    ]
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(args, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"seed": seed, "status": "FAILED", "returncode": proc.returncode, "elapsed_s": elapsed}
    final_path = out_dir / "final.json"
    if not final_path.exists():
        return {"seed": seed, "status": "NO_FINAL", "elapsed_s": elapsed}
    try:
        d = json.loads(final_path.read_text())
    except Exception as e:
        return {"seed": seed, "status": "PARSE_ERR", "error": str(e), "elapsed_s": elapsed}
    neurons = d.get("neurons", [])
    depth = max((n.get("tick", 0) for n in neurons), default=0)
    return {
        "seed": seed,
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
        print(f"ERROR: release binary not found: {BIN_PATH}", flush=True)
        print("       Run: cargo build --release --example c19_grower -p instnct-core", flush=True)
        sys.exit(1)

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    (SWEEP_DIR / "logs").mkdir(parents=True, exist_ok=True)
    progress_file = SWEEP_DIR / "progress.log"
    if progress_file.exists():
        progress_file.unlink()

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(progress_file, "a") as f:
            f.write(line + "\n")

    log(f"c19_parity_sweep start - 20 seeds, grid3_full_parity, max_neurons=32")
    log(f"baseline (grid3_curriculum seed 4): val={BASELINE_VAL:.2f}% (28 neurons, depth 12)")
    log(f"binary: {BIN_PATH}")
    log("")

    t0 = time.time()
    results = []
    for seed in range(20):
        log(f"starting seed {seed}")
        r = run_one(seed)
        results.append(r)
        if r["status"] == "OK":
            log(f"  seed {seed:<2}: val={r['val']:5.2f}%  test={r['test']:5.2f}%  "
                f"train={r['train']:5.2f}%  N={r['neurons']:<3} depth={r['depth']:<2}  {r['elapsed_s']:5.1f}s")
        else:
            log(f"  seed {seed:<2}: {r['status']}")
    total_elapsed = time.time() - t0

    log("")
    log(f"=== DONE in {total_elapsed/60:.1f} min ({total_elapsed:.0f}s) ===")
    log("")
    log("seed  val     test    train   N    depth  elapsed")
    for r in results:
        if r["status"] == "OK":
            log(f"  {r['seed']:<2}  {r['val']:5.2f}   {r['test']:5.2f}   "
                f"{r['train']:5.2f}   {r['neurons']:<3}  {r['depth']:<4}  {r['elapsed_s']:5.1f}s")
        else:
            log(f"  {r['seed']:<2}  {r['status']}")

    ok = [r for r in results if r["status"] == "OK"]
    if ok:
        vals = sorted([r["val"] for r in ok], reverse=True)
        best = max(ok, key=lambda r: r["val"])
        mean_val = sum(r["val"] for r in ok) / len(ok)
        median_val = sorted(r["val"] for r in ok)[len(ok) // 2]
        log("")
        log(f"Best  val: {best['val']:.2f}%  (seed {best['seed']}, test={best['test']:.2f}%, "
            f"N={best['neurons']}, depth={best['depth']})")
        log(f"Mean  val: {mean_val:.2f}%")
        log(f"Median val: {median_val:.2f}%")
        log(f"Top-5 vals: {[f'{v:.2f}' for v in vals[:5]]}")
        log(f"Worst val: {min(r['val'] for r in ok):.2f}%")
        log("")
        delta = best["val"] - BASELINE_VAL
        verdict = "c19 BEATS baseline" if delta > 0 else ("TIED" if delta == 0 else "c19 BELOW baseline")
        log(f"c19 best vs baseline {BASELINE_VAL:.2f}%: {delta:+.2f}pp  ->  {verdict}")

    summary = {
        "task": "grid3_full_parity",
        "n_seeds": 20,
        "max_neurons": 32,
        "baseline_val": BASELINE_VAL,
        "results": results,
        "elapsed_total_s": total_elapsed,
    }
    (SWEEP_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"Summary: {SWEEP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
