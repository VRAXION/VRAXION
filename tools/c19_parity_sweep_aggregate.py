#!/usr/bin/env python3
"""c19_parity_sweep_aggregate.py — Post-process existing seed_XX/final.json files
into a unified summary.json (runs after a failed sweep finalize step)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "target" / "c19_parity_sweep"
BASELINE_VAL = 76.20


def main():
    if not SWEEP_DIR.exists():
        print(f"ERROR: sweep dir not found: {SWEEP_DIR}")
        return 1

    results = []
    for seed in range(20):
        out_dir = SWEEP_DIR / f"seed_{seed:02d}"
        final_path = out_dir / "final.json"
        if not final_path.exists():
            results.append({"seed": seed, "status": "NO_FINAL"})
            continue
        try:
            d = json.loads(final_path.read_text())
        except Exception as e:
            results.append({"seed": seed, "status": "PARSE_ERR", "error": str(e)})
            continue
        neurons = d.get("neurons", [])
        depth = max((n.get("tick", 0) for n in neurons), default=0)
        results.append({
            "seed": seed,
            "status": "OK",
            "val": d.get("ensemble_val", 0.0),
            "test": d.get("ensemble_test", 0.0),
            "train": d.get("ensemble_train", 0.0),
            "neurons": len(neurons),
            "depth": depth,
        })

    ok = [r for r in results if r["status"] == "OK"]
    print(f"aggregated {len(ok)}/20 successful seeds")
    print()
    print("seed  val     test    train   N    depth")
    for r in results:
        if r["status"] == "OK":
            print(f"  {r['seed']:<2}  {r['val']:5.2f}   {r['test']:5.2f}   "
                  f"{r['train']:5.2f}   {r['neurons']:<3}  {r['depth']}")
        else:
            print(f"  {r['seed']:<2}  {r['status']}")
    print()

    if ok:
        vals = sorted([r["val"] for r in ok], reverse=True)
        best = max(ok, key=lambda r: r["val"])
        mean_val = sum(r["val"] for r in ok) / len(ok)
        median_val = sorted(r["val"] for r in ok)[len(ok) // 2]
        above_baseline = sum(1 for r in ok if r["val"] > BASELINE_VAL)
        print(f"Best val: {best['val']:.2f}% (seed {best['seed']}, test={best['test']:.2f}%, N={best['neurons']}, depth={best['depth']})")
        print(f"Mean val: {mean_val:.2f}%")
        print(f"Median val: {median_val:.2f}%")
        print(f"Seeds above baseline {BASELINE_VAL:.2f}%: {above_baseline}/20")
        delta = best["val"] - BASELINE_VAL
        verdict = "c19 BEATS baseline" if delta > 0 else "c19 TIED OR BELOW"
        print(f"c19 best delta: {delta:+.2f}pp  ({verdict})")

    summary = {
        "task": "grid3_full_parity",
        "n_seeds": 20,
        "max_neurons": 32,
        "baseline_val": BASELINE_VAL,
        "results": results,
        "mean_val": mean_val if ok else None,
        "median_val": median_val if ok else None,
        "best": {
            "seed": best["seed"],
            "val": best["val"],
            "test": best["test"],
            "neurons": best["neurons"],
            "depth": best["depth"],
        } if ok else None,
        "seeds_above_baseline": above_baseline if ok else 0,
    }
    out_path = SWEEP_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Summary written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
