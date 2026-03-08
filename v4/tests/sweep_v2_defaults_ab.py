"""A/B sweep: v1 defaults vs v2 defaults (pointer interp + seam fix + R=2).

Compares old config (R=1, mod seam, no interp) against new config
(R=2, shortest_arc seam, linear pointer interp) on all nightly surfaces.

Usage:
    python sweep_v2_defaults_ab.py                          # all surfaces, CPU
    python sweep_v2_defaults_ab.py --surface wikitext_sequential_carry --steps 500
    python sweep_v2_defaults_ab.py --device cuda --steps 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for subdir in ("model", "training"):
    p = str(ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

from nightly_research_runner import SURFACES, VARIANTS, run_surface  # type: ignore[import-not-found]

# Surfaces to compare: v1 (old) vs v2 (new defaults)
AB_PAIRS = [
    ("wikitext_sequential_carry", "wikitext_sequential_carry_v2"),
    ("fast_memory_carry", "fast_memory_carry_v2"),
]

# Variants to test on each surface pair
AB_VARIANTS = ["LL", "LLT", "LLT7"]


def _fmt(val, digits=4):
    if val is None:
        return "n/a"
    return f"{float(val):.{digits}f}"


def run_ab_sweep(
    surface_filter: str | None = None,
    variant_filter: str | None = None,
    steps: int | None = None,
    device: str | None = None,
    seed: int = 42,
    out_dir: str | None = None,
):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(out_dir) if out_dir else ROOT / "sweep_results" / f"v2_defaults_ab_{stamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for v1_surface, v2_surface in AB_PAIRS:
        if surface_filter and surface_filter not in (v1_surface, v2_surface):
            continue

        variants_to_test = [variant_filter] if variant_filter else AB_VARIANTS
        for variant in variants_to_test:
            if variant not in VARIANTS:
                print(f"  [SKIP] Unknown variant: {variant}")
                continue

            for surface, label in [(v1_surface, "v1_baseline"), (v2_surface, "v2_upgraded")]:
                if surface not in SURFACES:
                    print(f"  [SKIP] Surface {surface} not found")
                    continue

                tag = f"{surface}_{variant}"
                print(f"\n{'='*80}")
                print(f"  Running: {label} | surface={surface} | variant={variant}")
                print(f"{'='*80}")

                t0 = time.time()
                try:
                    payload = run_surface(
                        surface=surface,
                        variant=variant,
                        steps_override=steps,
                        device_override=device,
                        seed_override=seed,
                    )
                    elapsed = time.time() - t0
                    result = payload["result"]
                    meta = payload["meta"]

                    row = {
                        "label": label,
                        "surface": surface,
                        "variant": variant,
                        "pointer_interp_mode": meta.get("pointer_interp_mode", "off"),
                        "pointer_seam_mode": meta.get("pointer_seam_mode", "mod"),
                        "R": SURFACES[surface].get("R", 1),
                        "steps": SURFACES[surface]["steps"] if steps is None else steps,
                        "seed": seed,
                        "final_loss": result.get("final_loss"),
                        "final_bpc": result.get("final_bpc"),
                        "final_acc": result.get("final_acc"),
                        "best_acc": result.get("best_acc"),
                        "time_s": result.get("time_s") or result.get("wall_time"),
                        "s_per_step": result.get("s_per_step"),
                        "max_grad": result.get("max_grad"),
                        "carry_eval_acc": result.get("carry_eval_acc"),
                        "fresh_eval_acc": result.get("fresh_eval_acc"),
                        "carry_minus_reset_pp": result.get("carry_minus_reset_pp"),
                    }
                    all_results.append(row)

                    print(
                        f"  => loss={_fmt(row['final_loss'])} bpc={_fmt(row['final_bpc'])} "
                        f"acc={_fmt(row['final_acc'])} best={_fmt(row['best_acc'])} "
                        f"time={_fmt(row['time_s'], 1)}s"
                    )

                    # Save per-run JSON
                    json_path = results_dir / f"{label}_{variant}_{surface}.json"
                    with open(json_path, "w") as f:
                        json.dump(payload, f, indent=2)

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  => FAILED after {elapsed:.1f}s: {e}")
                    all_results.append({
                        "label": label,
                        "surface": surface,
                        "variant": variant,
                        "error": str(e),
                    })

    # Write summary CSV
    if all_results:
        csv_path = results_dir / "summary.csv"
        keys = list(all_results[0].keys())
        for r in all_results[1:]:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n{'='*80}")
        print(f"Summary CSV: {csv_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("A/B COMPARISON: v1 (baseline) vs v2 (upgraded defaults)")
    print(f"{'='*80}")
    print(f"{'Variant':<12} {'Surface':<35} {'Loss':>8} {'BPC':>8} {'Acc':>8} {'Best':>8} {'Time':>8}")
    print("-" * 95)
    for row in all_results:
        if "error" in row:
            print(f"{row.get('variant','?'):<12} {row.get('surface','?'):<35} ERROR: {row['error']}")
        else:
            print(
                f"{row.get('variant','?'):<12} {row.get('surface','?'):<35} "
                f"{_fmt(row.get('final_loss')):>8} {_fmt(row.get('final_bpc')):>8} "
                f"{_fmt(row.get('final_acc')):>8} {_fmt(row.get('best_acc')):>8} "
                f"{_fmt(row.get('time_s'), 1):>8}"
            )

    # Print deltas
    print(f"\n{'='*80}")
    print("DELTAS (v2 - v1):")
    print(f"{'='*80}")
    v1_results = {(r["variant"], r["surface"]): r for r in all_results if r.get("label") == "v1_baseline" and "error" not in r}
    v2_results = {(r["variant"], r["surface"].replace("_v2", "")): r for r in all_results if r.get("label") == "v2_upgraded" and "error" not in r}
    for key in v1_results:
        v1 = v1_results[key]
        v2 = v2_results.get(key)
        if v2 is None:
            continue
        variant, surface = key
        loss_delta = (v2.get("final_loss") or 0) - (v1.get("final_loss") or 0)
        acc_delta = ((v2.get("final_acc") or 0) - (v1.get("final_acc") or 0)) * 100
        bpc_delta = (v2.get("final_bpc") or 0) - (v1.get("final_bpc") or 0)
        time_v1 = v1.get("time_s") or 1
        time_v2 = v2.get("time_s") or 1
        speed_pct = (time_v1 - time_v2) / time_v1 * 100
        print(
            f"  {variant:<12} loss: {loss_delta:+.4f}  bpc: {bpc_delta:+.4f}  "
            f"acc: {acc_delta:+.2f}pp  speed: {speed_pct:+.1f}%"
        )

    results_json = results_dir / "results.json"
    with open(results_json, "w") as f:
        json.dump({"timestamp": stamp, "results": all_results}, f, indent=2)
    print(f"\nResults: {results_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="A/B sweep: v1 vs v2 defaults")
    parser.add_argument("--surface", type=str, default=None, help="Filter to specific surface")
    parser.add_argument("--variant", type=str, default=None, help="Filter to specific variant")
    parser.add_argument("--steps", type=int, default=None, help="Override step count")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    run_ab_sweep(
        surface_filter=args.surface,
        variant_filter=args.variant,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
