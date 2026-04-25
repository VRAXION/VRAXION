"""Phase D0.6: Offline minimum-useful threshold (+δ) calibration.

CALIBRATION ONLY — DOES NOT PREDICT peak_acc OR final_acc.

Per GPT review (2026-04-25): a +δ minimum-useful threshold cannot be
back-tested for outcome on existing logs, because rejecting an
originally-accepted move makes the entire downstream trajectory diverge.
The log only contains candidates from the original parent state.

What this analyzer DOES:
  - For each (arm × δ in grid), count fraction of original positive
    best-of-K ΔU values that survive threshold δ
  - Distribution of positive ΔU per arm — quantiles tell us where
    δ_small / δ_med should be set for Phase E
  - C_K_at_δ_synthetic = E[max(0, best - δ)] / E[cost_K] — what the
    per-step productivity formula WOULD evaluate to under threshold δ,
    GIVEN the trajectory unchanged. Trajectory invariance is impossible
    without re-running the experiment; this is a CALIBRATION metric,
    not a prediction.

What this analyzer DOES NOT compute:
  - peak_acc(δ): would require new run
  - final_acc(δ): would require new run
  - any trajectory-dependent outcome under threshold δ

Run (GPT local):
  python tools/diag_phase_d0_6_minimum_useful.py \\
      --root output/phase_b1_horizon_ties_20260425 \\
      --out  output/phase_d0_6_minimum_useful
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# δ grid for offline +δ calibration. Spaced log-uniformly over the range
# of typical positive ΔU values.
DELTA_GRID = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3]


def load_candidate_logs(root: Path) -> dict[tuple[str, int], list[dict]]:
    bundles: dict[tuple[str, int], list[dict]] = {}
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        arm = arm_dir.name
        for seed_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            seed = int(seed_dir.name.split("_", 1)[1])
            csv_path = seed_dir / "candidates.csv"
            if not csv_path.exists():
                continue
            rows = []
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            bundles[(arm, seed)] = rows
            print(f"  loaded {arm}/seed={seed}: {len(rows)} rows", file=sys.stderr)
    return bundles


def positive_delta_distribution(rows: list[dict]) -> np.ndarray:
    """Extract the positive-ΔU subset from rows, ignoring K-grouping (this is
    a per-candidate metric, not a per-step best-of-K)."""
    deltas = []
    for r in rows:
        try:
            d = float(r.get("delta_U", "nan"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(d) and d > 0:
            deltas.append(d)
    return np.array(deltas, dtype=float)


def best_per_step(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Group rows by step, take best ΔU per step. Also return total per-step
    eval cost. Returns (best_deltas, total_costs_per_step)."""
    grouped_d = defaultdict(list)
    grouped_c = defaultdict(list)
    for r in rows:
        try:
            step = int(r["step"])
            d = float(r.get("delta_U", "nan"))
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(d):
            continue
        # Try multiple cost field names (per GPT note about candidate_eval_ms)
        c = float("nan")
        for key in ("candidate_eval_ms", "eval_ms", "wall_ms"):
            v = r.get(key, None)
            if v is None:
                continue
            try:
                c = float(v)
                break
            except (TypeError, ValueError):
                continue
        grouped_d[step].append(d)
        if np.isfinite(c):
            grouped_c[step].append(c)
    bests = []
    costs = []
    for step in sorted(grouped_d.keys()):
        bests.append(max(grouped_d[step]))
        costs.append(sum(grouped_c.get(step, [0.0])))
    return np.array(bests, dtype=float), np.array(costs, dtype=float)


def delta_calibration_arm(rows: list[dict]) -> dict:
    """Per-arm calibration: how does threshold δ filter the best-of-K positives?"""
    bests, costs = best_per_step(rows)
    n_steps = len(bests)
    if n_steps == 0:
        return {"n_steps": 0}
    pos_dist = bests[bests > 0]
    n_pos = len(pos_dist)
    out = {
        "n_steps": int(n_steps),
        "n_positive_best": int(n_pos),
        "positive_best_rate": float(n_pos / n_steps) if n_steps else 0.0,
    }
    # Quantiles of positive distribution
    if n_pos > 0:
        for q in (10, 25, 50, 75, 90, 95):
            out[f"pos_q{q}"] = float(np.percentile(pos_dist, q))
        out["pos_mean"] = float(np.mean(pos_dist))
        out["pos_max"] = float(np.max(pos_dist))
    # Per-δ calibration
    delta_curve = []
    mean_cost = float(np.mean(costs)) if len(costs) and np.mean(costs) > 0 else float("nan")
    for d in DELTA_GRID:
        survives = bests > d  # bests strictly above threshold
        n_survive = int(np.sum(survives))
        # synthetic C_K — calibration only, NOT a prediction
        useful_at_d = np.where(bests > d, bests - d, 0.0)
        if not np.isnan(mean_cost) and mean_cost > 0:
            ck_at_d = float(np.mean(useful_at_d) / mean_cost)
        else:
            ck_at_d = float("nan")
        delta_curve.append({
            "delta": d,
            "n_survive_above_delta": n_survive,
            "fraction_survive": float(n_survive / n_steps) if n_steps else 0.0,
            "fraction_of_positives_filtered": float(1 - n_survive / n_pos) if n_pos > 0 else 0.0,
            "C_K_synthetic": ck_at_d,
        })
    out["delta_calibration_curve"] = delta_curve
    return out


def recommend_delta(per_arm: dict[str, dict]) -> dict:
    """Recommend δ_small (filters bottom 25% of positive-bests) and δ_med
    (filters bottom 50%) per arm — calibration ONLY."""
    suggestions = {}
    for arm, agg in per_arm.items():
        if agg.get("n_positive_best", 0) == 0:
            suggestions[arm] = {"valid": False, "reason": "no_positive_bests"}
            continue
        delta_small = agg.get("pos_q25", float("nan"))
        delta_med = agg.get("pos_q50", float("nan"))
        suggestions[arm] = {
            "valid": True,
            "delta_small_q25": delta_small,
            "delta_med_q50": delta_med,
            "delta_large_q75": agg.get("pos_q75", float("nan")),
            "rationale": "Phase E δ candidates: small filters bottom 25% of original positive-bests, med filters bottom 50%. ZERO trajectory-outcome implication — these need new runs for outcome.",
        }
    return suggestions


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Phase B.1 output root")
    p.add_argument("--out", required=True, help="output dir")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2

    print(f"Loading candidate logs from {root} ...", file=sys.stderr)
    bundles = load_candidate_logs(root)
    if not bundles:
        print(f"ERROR: no candidate.csv files found", file=sys.stderr)
        return 2
    print(f"Loaded {len(bundles)} (arm, seed) bundles", file=sys.stderr)

    # Per-arm pooled across seeds
    arm_groups = defaultdict(list)
    for (arm, _seed), rows in bundles.items():
        arm_groups[arm].extend(rows)
    per_arm = {}
    for arm, rows in sorted(arm_groups.items()):
        print(f"  calibrating {arm} ({len(rows)} candidate-rows) ...", file=sys.stderr)
        per_arm[arm] = delta_calibration_arm(rows)

    suggestions = recommend_delta(per_arm)

    summary = {
        "n_arms": len(per_arm),
        "delta_grid": DELTA_GRID,
        "per_arm": per_arm,
        "delta_recommendations": suggestions,
        "DISCLAIMER": "This is a CALIBRATION analysis only. No outcome (peak_acc, final_acc, trajectory) is predicted from existing logs. Phase E (live runs) required for outcome claims.",
    }
    (out_dir / "d0_6_summary.json").write_text(json.dumps(summary, indent=2))

    # CSV: per-arm × δ table
    with (out_dir / "d0_6_per_arm_delta.csv").open("w", newline="") as f:
        cols = ["arm", "delta", "n_survive_above_delta", "fraction_survive",
                "fraction_of_positives_filtered", "C_K_synthetic"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, agg in sorted(per_arm.items()):
            for row in agg.get("delta_calibration_curve", []):
                w.writerow({"arm": arm, **row})

    # Markdown
    lines = [
        "# Phase D0.6 Minimum-Useful Threshold (+δ) Calibration",
        "",
        "**Calibration only.** This analysis does NOT predict peak_acc or final_acc under threshold δ. Trajectory invariance under counter-factual rejection is not a property the existing logs preserve. Phase E (live runs) is required for outcome claims.",
        "",
        "## Per-arm positive ΔU distribution (best-of-K)",
        "",
        "| arm | n_pos | mean | q10 | q25 | q50 (med) | q75 | q90 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, agg in sorted(per_arm.items()):
        if agg.get("n_positive_best", 0) == 0:
            continue
        lines.append(
            f"| {arm} | {agg['n_positive_best']:,} | {agg.get('pos_mean', float('nan')):.3e} | "
            f"{agg.get('pos_q10', float('nan')):.3e} | {agg.get('pos_q25', float('nan')):.3e} | "
            f"{agg.get('pos_q50', float('nan')):.3e} | {agg.get('pos_q75', float('nan')):.3e} | "
            f"{agg.get('pos_q90', float('nan')):.3e} |"
        )
    lines.extend([
        "",
        "## Phase E δ recommendation (calibration only)",
        "",
        "| arm | δ_small (q25) | δ_med (q50) | δ_large (q75) |",
        "|---|---|---|---|",
    ])
    for arm, s in sorted(suggestions.items()):
        if not s.get("valid"):
            continue
        lines.append(
            f"| {arm} | {s['delta_small_q25']:.3e} | {s['delta_med_q50']:.3e} | {s['delta_large_q75']:.3e} |"
        )
    lines.extend([
        "",
        "These δ values are the **quartile-cuts** of the original positive best-of-K distribution. They calibrate where in the +δ axis Phase E should place its sweep points; they do NOT predict outcomes.",
        "",
        "## What Phase E (live) would test",
        "",
        "Phase E hypothesis: there exists δ* > 0 such that the substrate trained with `accept iff ΔU > δ*` reaches better peak_acc than strict (δ = 0). Mechanism candidate: filtering trivially-small positive ΔU prevents drift on noise.",
        "",
        "Falsifying outcome: every δ > 0 produces lower or equal peak_acc to strict (δ = 0) — the substrate genuinely benefits from accumulating small positive moves.",
    ])
    (out_dir / "d0_6_recommendation.md").write_text("\n".join(lines))

    print(f"\nWrote: {out_dir}/d0_6_summary.json", file=sys.stderr)
    print(f"Wrote: {out_dir}/d0_6_per_arm_delta.csv", file=sys.stderr)
    print(f"Wrote: {out_dir}/d0_6_recommendation.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
