#!/usr/bin/env python3
"""Run baseline grid3_curriculum (ternary threshold) on all 9 grid3_copy_bit_N
heads — the 'hybrid' test comparing to the c19 stall on bits 3, 5, 6.
"""

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "examples" / "grid3_curriculum_snapshot.exe"
OUT_ROOT = ROOT / "target" / "baseline_grid3_copy"


def run_one(bit_idx: int) -> dict:
    out_dir = OUT_ROOT / f"bit_{bit_idx}"
    log_file = OUT_ROOT / "logs" / f"bit_{bit_idx}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    args = [
        str(BIN),
        "--task", f"grid3_copy_bit_{bit_idx}",
        "--search-seed", "0",
        "--out-dir", str(out_dir),
        "--max-neurons", "4",
    ]
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(args, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return {"bit": bit_idx, "status": "FAILED", "elapsed_s": elapsed}
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
    if not BIN.exists():
        print(f"ERROR: snapshot binary not found: {BIN}")
        return 1
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("BASELINE grid3_curriculum (ternary threshold) on all 9 grid3_copy_bit_N")
    print("=" * 72)
    t0 = time.time()
    results = []
    for i in range(9):
        print(f"[{time.strftime('%H:%M:%S')}] starting bit {i}", flush=True)
        r = run_one(i)
        results.append(r)
        if r["status"] == "OK":
            print(f"  bit {i}: val={r['val']:5.2f}%  test={r['test']:5.2f}%  "
                  f"train={r['train']:5.2f}%  N={r['neurons']:<2} depth={r['depth']:<2}  "
                  f"{r['elapsed_s']:5.1f}s", flush=True)
        else:
            print(f"  bit {i}: {r['status']}", flush=True)
    total = time.time() - t0
    ok = [r for r in results if r["status"] == "OK"]
    print()
    print(f"=== DONE in {total:.1f}s ===")
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
        heads_100 = sum(1 for r in ok if r["val"] >= 99.999)
        print()
        print(f"Total baseline neurons across 9 heads: {total_n}")
        print(f"Max depth (any head):                  {max_d}")
        print(f"Heads at 100% val:                     {heads_100}/9")
    (OUT_ROOT / "summary.json").write_text(json.dumps({"task_family": "grid3_copy_v1", "grower": "baseline_threshold", "results": results}, indent=2))
    print()
    print(f"Summary: {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
