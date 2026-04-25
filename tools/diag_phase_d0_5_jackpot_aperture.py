"""Phase D0.5: Offline K-resampling on Phase B.1 candidate logs.

Insight (per GPT/user discussion 2026-04-25):
  The "Acceptance Aperture" is two-tier:
    K (jackpot size) = sampling aperture (upstream)
    acceptance policy = valve  (downstream)
  D0 showed best_negative_rate ≈ 0.06% under K=9 — but this is K-induced
  zero-saturation, not substrate property. With K small, the picture
  changes substantially.

This analyzer simulates what would have happened with K ∈ {1, 2, 3, 5, 9}
by taking the first k candidates per step (the existing logs have all 9
candidates per step, sequentially generated).

Reports per arm × K:
  - best_negative_rate(K), best_exact_zero_rate(K), best_positive_rate(K)
  - strict_accept_rate(K), ties_accept_rate(K)
  - C_K_window_ratio(K) — per-step useful improvement / per-step cost
  - selection_pressure(K) — E[positive | best > 0]

Proposes D1 K choice based on which K still leaves selection pressure
meaningful (zero_best_rate < 0.5, say) AND maintains discovery
(positive_best_rate > 0.05).

Run (GPT local):
  python tools/diag_phase_d0_5_jackpot_aperture.py \\
      --root output/phase_b1_horizon_ties_20260425 \\
      --out  output/phase_d0_5_jackpot_aperture
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


K_GRID = [1, 2, 3, 5, 9]
EPS_DETECT = 1e-9


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


def group_by_step(rows: list[dict]) -> list[list[dict]]:
    """Group rows by (run_id, step). Returns list of step-groups, each containing
    K candidates in order."""
    grouped = defaultdict(list)
    for r in rows:
        try:
            step = int(r["step"])
        except (KeyError, ValueError):
            continue
        grouped[step].append(r)
    return [grouped[k] for k in sorted(grouped.keys())]


def k_resample_arm(rows: list[dict], k_grid: list[int]) -> dict[int, dict]:
    """For each K in k_grid, simulate the first-K-candidates jackpot and
    compute best-of-K statistics."""
    step_groups = group_by_step(rows)
    out = {}
    for k in k_grid:
        bests = []
        all_evals_per_step = []
        for group in step_groups:
            if len(group) < k:
                # Not enough candidates for this step, skip
                continue
            deltas = []
            costs = []
            for cand in group[:k]:
                try:
                    d = float(cand.get("delta_U", "nan"))
                    c = float(cand.get("eval_ms", "nan"))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(d):
                    deltas.append(d)
                if np.isfinite(c):
                    costs.append(c)
            if not deltas:
                continue
            bests.append(max(deltas))
            all_evals_per_step.append(sum(costs))
        if not bests:
            out[k] = {"n_steps": 0}
            continue
        bests = np.array(bests, dtype=float)
        eval_costs = np.array(all_evals_per_step, dtype=float)
        useful = np.where(bests > EPS_DETECT, bests, 0.0)
        # C_K = E[max(0, best - eps)] / E[cost_K]
        if np.mean(eval_costs) > 0:
            ck_window_ratio = float(np.mean(useful) / np.mean(eval_costs))
        else:
            ck_window_ratio = float("nan")
        out[k] = {
            "n_steps": int(len(bests)),
            "best_negative_rate": float(np.mean(bests < -EPS_DETECT)),
            "best_exact_zero_rate": float(np.mean(np.abs(bests) <= EPS_DETECT)),
            "best_positive_rate": float(np.mean(bests > EPS_DETECT)),
            "strict_accept_rate": float(np.mean(bests > EPS_DETECT)),
            "ties_accept_rate": float(np.mean(bests >= -EPS_DETECT)),
            "selection_pressure_pos": float(np.mean(bests[bests > EPS_DETECT])) if np.any(bests > EPS_DETECT) else 0.0,
            "C_K_window_ratio": ck_window_ratio,
            "mean_cost_K": float(np.mean(eval_costs)),
        }
    return out


def fit_geometric_model(per_arm_per_K: dict[str, dict[int, dict]]) -> dict:
    """From the K-grid resampling, estimate p_pos, p_zero, p_neg per arm
    using K=1 statistics. Then validate the geometric prediction at K=3, 5, 9."""
    fits = {}
    for arm, by_k in per_arm_per_K.items():
        if 1 not in by_k or by_k[1].get("n_steps", 0) == 0:
            fits[arm] = {"valid": False, "reason": "no_K=1_data"}
            continue
        k1 = by_k[1]
        p_pos = k1.get("best_positive_rate", float("nan"))
        p_zero = k1.get("best_exact_zero_rate", float("nan"))
        p_neg = k1.get("best_negative_rate", float("nan"))
        # geometric prediction at each K
        predictions = {}
        for k in K_GRID:
            if k not in by_k:
                continue
            pred_strict = 1 - (1 - p_pos) ** k if not np.isnan(p_pos) else float("nan")
            pred_ties = 1 - p_neg ** k if not np.isnan(p_neg) else float("nan")
            obs = by_k[k]
            predictions[k] = {
                "K": k,
                "predicted_strict_accept": pred_strict,
                "observed_strict_accept": obs.get("strict_accept_rate", float("nan")),
                "predicted_ties_accept": pred_ties,
                "observed_ties_accept": obs.get("ties_accept_rate", float("nan")),
            }
        fits[arm] = {
            "valid": True,
            "p_pos_K1": p_pos,
            "p_zero_K1": p_zero,
            "p_neg_K1": p_neg,
            "predictions_vs_observations": predictions,
        }
    return fits


def recommend_k(per_arm_per_K: dict[str, dict[int, dict]]) -> dict:
    """Recommend K* per arm: largest K where ties_accept_rate < 0.95 AND
    positive_best_rate > 0.05.

    Rationale: this is the K-range where the policy axis (strict / zero_p)
    can still differentiate — i.e., where ties does NOT autosaturate."""
    suggestions = {}
    for arm, by_k in per_arm_per_K.items():
        candidates = []
        for k in sorted(by_k.keys()):
            row = by_k[k]
            if row.get("n_steps", 0) == 0:
                continue
            ties_rate = row.get("ties_accept_rate", float("nan"))
            pos_rate = row.get("best_positive_rate", float("nan"))
            if not np.isnan(ties_rate) and ties_rate < 0.95 and pos_rate > 0.05:
                candidates.append(k)
        suggestions[arm] = {
            "k_grid_useful": candidates,
            "k_max_useful": max(candidates) if candidates else None,
            "k_recommended_for_D1": max(candidates) if candidates else min(by_k.keys()),
            "rationale": "largest K where ties_accept < 0.95 (selection pressure preserved) AND positive_best > 0.05 (discovery preserved)",
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

    # Group by arm (pool seeds), then K-resample
    print("\nK-resampling per arm × K ∈ {1, 2, 3, 5, 9} ...", file=sys.stderr)
    per_arm_per_K: dict[str, dict[int, dict]] = {}
    arm_groups: dict[str, list[dict]] = defaultdict(list)
    for (arm, _seed), rows in bundles.items():
        arm_groups[arm].extend(rows)
    for arm, rows in sorted(arm_groups.items()):
        print(f"  resampling {arm} ({len(rows)} candidate-rows) ...", file=sys.stderr)
        per_arm_per_K[arm] = k_resample_arm(rows, K_GRID)

    geom_fit = fit_geometric_model(per_arm_per_K)
    suggestions = recommend_k(per_arm_per_K)

    summary = {
        "n_arms": len(per_arm_per_K),
        "k_grid": K_GRID,
        "per_arm_per_K": per_arm_per_K,
        "geometric_fit_validation": geom_fit,
        "k_recommendations": suggestions,
    }
    (out_dir / "d0_5_summary.json").write_text(json.dumps(summary, indent=2))

    # CSV: per-arm-per-K table
    with (out_dir / "d0_5_per_arm_per_K.csv").open("w", newline="") as f:
        cols = ["arm", "K", "n_steps", "best_negative_rate", "best_exact_zero_rate",
                "best_positive_rate", "strict_accept_rate", "ties_accept_rate",
                "selection_pressure_pos", "C_K_window_ratio", "mean_cost_K"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, by_k in sorted(per_arm_per_K.items()):
            for k, row in sorted(by_k.items()):
                if row.get("n_steps", 0) == 0:
                    continue
                w.writerow({"arm": arm, "K": k, **{c: row.get(c) for c in cols if c not in ("arm", "K")}})

    # Markdown report
    lines = [
        "# Phase D0.5 Jackpot-Aperture Resampling Report",
        "",
        f"K-resampling on {len(bundles)} (arm, seed) bundles from `{root}`.",
        "",
        "## Per-arm K-resampled accept rates",
        "",
        "| arm | K | best_neg | best_zero | best_pos | strict_accept | ties_accept | C_K |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, by_k in sorted(per_arm_per_K.items()):
        for k, row in sorted(by_k.items()):
            if row.get("n_steps", 0) == 0:
                continue
            lines.append(
                f"| {arm} | {k} | {row['best_negative_rate']:.4f} | {row['best_exact_zero_rate']:.4f} | "
                f"{row['best_positive_rate']:.4f} | {row['strict_accept_rate']:.4f} | "
                f"{row['ties_accept_rate']:.4f} | {row['C_K_window_ratio']:.3e} |"
            )
    lines.extend([
        "",
        "## Geometric prediction validation",
        "",
        "If the per-candidate p_pos / p_zero / p_neg are stable, the prediction",
        "P(strict accepts at K) = 1 - (1 - p_pos)^K  and  P(ties accepts at K) = 1 - p_neg^K",
        "should match observation across K.",
        "",
        "| arm | p_pos (K=1) | p_zero (K=1) | p_neg (K=1) | observed strict @ K=9 | predicted strict @ K=9 |",
        "|---|---|---|---|---|---|",
    ])
    for arm, fit in sorted(geom_fit.items()):
        if not fit.get("valid"):
            continue
        pred9 = fit.get("predictions_vs_observations", {}).get(9, {})
        lines.append(
            f"| {arm} | {fit['p_pos_K1']:.4f} | {fit['p_zero_K1']:.4f} | {fit['p_neg_K1']:.4f} | "
            f"{pred9.get('observed_strict_accept', float('nan')):.4f} | "
            f"{pred9.get('predicted_strict_accept', float('nan')):.4f} |"
        )
    lines.extend([
        "",
        "## D1 K* recommendation",
        "",
        "Recommended K for D1 is the largest K where ties_accept < 0.95 (selection pressure preserved) AND positive_best > 0.05 (discovery preserved). This is the regime where the zero_p axis still has meaningful range to test.",
        "",
        "| arm | K_max_useful | K_recommended_for_D1 |",
        "|---|---|---|",
    ])
    for arm, s in sorted(suggestions.items()):
        lines.append(
            f"| {arm} | {s['k_max_useful']} | {s['k_recommended_for_D1']} |"
        )
    lines.extend([
        "",
        "## Implications for D1",
        "",
        "If the recommended K* is significantly lower than the current K=9 used in B.1,",
        "the D1 design should either:",
        "  (a) reduce K to K* (e.g. K=3) and sweep zero_p as planned, OR",
        "  (b) factorial K × zero_p sweep — e.g. K ∈ {3, 5, 9} × zero_p ∈ {strict, 0.3, 1.0}",
        "      = 9 cells × 5 seeds = 45 cells, only modestly larger than the planned 30-cell D1.",
        "",
        "If K* ≈ K=9 (no change), the original D1 design stands.",
    ])
    (out_dir / "d0_5_recommendation.md").write_text("\n".join(lines))

    print(f"\nWrote: {out_dir}/d0_5_summary.json", file=sys.stderr)
    print(f"Wrote: {out_dir}/d0_5_per_arm_per_K.csv", file=sys.stderr)
    print(f"Wrote: {out_dir}/d0_5_recommendation.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
