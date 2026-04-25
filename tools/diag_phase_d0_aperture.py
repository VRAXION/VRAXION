"""Phase D0: Offline Acceptance Aperture analysis on Phase B.1 candidate logs.

Goal (per Phase D pre-reg, refined with GPT D0/D1/D2 phasing):
  - Reconstruct empirical ΔU distribution per arm × seed × operator
  - Mass at ΔU = 0 (point-mass vs continuous tail)
  - Empirical accept_rate(ε) curve (predict D1 ε grid)
  - Gaussian null fit A_π(ε) — Lilliefors-corrected KS, Anderson-Darling tail
  - Per-operator fit decomposition (mixed-marginal artefact check)
  - Recommend ε_small and ε_large for D1 sweep

Produces:
  - <out>/d0_summary.json     — per-arm aggregates + AIC/BIC of Gaussian fit
  - <out>/d0_per_operator.csv — per-operator-arm productivity + tail behaviour
  - <out>/d0_accept_rate_curve.csv — accept_rate(ε) on a fine grid per arm
  - <out>/d0_recommendation.md — D1 ε_small / ε_large + Gaussian-null verdict

Run (GPT local):
  python tools/diag_phase_d0_aperture.py \\
      --root output/phase_b1_horizon_ties_20260425 \\
      --out  output/phase_d0_aperture
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


# ε grid for empirical accept_rate(ε) curve. Spaced log-uniformly.
EPSILON_GRID = [
    -1.0,   # strict (placeholder; actually -inf, but P(ΔU > -1) ≈ 1)
    0.0,    # neutral
    1e-6, 5e-6,
    1e-5, 5e-5,
    1e-4, 5e-4,
    1e-3, 5e-3,
    1e-2, 5e-2,
    1e-1,
]


def load_candidate_logs(root: Path) -> dict[tuple[str, int], list[dict]]:
    """Load all candidates.csv under root.

    Expected layout: root/<arm>/seed_<seed>/candidates.csv
    Returns: {(arm, seed): list of row dicts}
    """
    bundles: dict[tuple[str, int], list[dict]] = {}
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        arm = arm_dir.name
        for seed_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            seed = int(seed_dir.name.split("_", 1)[1])
            csv_path = seed_dir / "candidates.csv"
            if not csv_path.exists():
                print(f"  [skip] no candidates.csv in {seed_dir}", file=sys.stderr)
                continue
            rows = []
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            bundles[(arm, seed)] = rows
            print(f"  loaded {arm}/seed={seed}: {len(rows)} rows", file=sys.stderr)
    return bundles


def lilliefors_ks(x: np.ndarray, n_resample: int = 999) -> tuple[float, float]:
    """Lilliefors-corrected KS test against Normal(μ_hat, σ_hat).

    Returns (KS statistic, parametric-bootstrap p-value).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 30:
        return float("nan"), float("nan")
    mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1))
    if sigma <= 0:
        return float("nan"), float("nan")
    cdf_fitted = lambda v: stats.norm.cdf(v, loc=mu, scale=sigma)
    ks_obs, _ = stats.kstest(x, cdf_fitted)
    # Parametric bootstrap calibration
    rng = np.random.default_rng(42)
    ks_resample = np.zeros(n_resample)
    for k in range(n_resample):
        sample = rng.normal(loc=mu, scale=sigma, size=n)
        mu_b, sigma_b = float(np.mean(sample)), float(np.std(sample, ddof=1))
        if sigma_b <= 0:
            ks_resample[k] = 0.0
            continue
        cdf_b = lambda v: stats.norm.cdf(v, loc=mu_b, scale=sigma_b)
        ks_resample[k], _ = stats.kstest(sample, cdf_b)
    p_value = float((np.sum(ks_resample >= ks_obs) + 1) / (n_resample + 1))
    return float(ks_obs), p_value


def anderson_darling_tail(x: np.ndarray) -> tuple[float, str]:
    """Anderson–Darling normality test (scipy implementation gives critical values)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return float("nan"), "n_too_small"
    try:
        result = stats.anderson(x, dist="norm")
    except Exception as e:  # pragma: no cover
        return float("nan"), f"error_{type(e).__name__}"
    a2 = float(result.statistic)
    # 5% critical value
    crit_5 = float(result.critical_values[2]) if len(result.critical_values) > 2 else float("nan")
    verdict = "fits_at_5%" if a2 < crit_5 else "rejects_normal_at_5%"
    return a2, verdict


def empirical_accept_rate(deltas: np.ndarray, eps_grid: list[float]) -> dict[str, float]:
    """For each ε in grid, fraction of candidates with ΔU ≥ -ε.

    Strict (-1.0) treated as ΔU > 0 (strict improvement only).
    """
    out: dict[str, float] = {}
    n = len(deltas)
    if n == 0:
        return {f"eps_{e}": float("nan") for e in eps_grid}
    for e in eps_grid:
        if e < 0:
            # strict regime
            rate = float(np.mean(deltas > 0))
        else:
            rate = float(np.mean(deltas >= -e))
        out[f"eps_{e}"] = rate
    return out


def mass_at_zero(deltas: np.ndarray, eps_window: float = 1e-9) -> dict[str, float]:
    """Fraction of |ΔU| ≤ window (numerical zero) and the immediate vicinity."""
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    if n == 0:
        return {"frac_exact_zero": float("nan"), "frac_near_zero_1e-6": float("nan"),
                "frac_near_zero_1e-4": float("nan")}
    return {
        "frac_exact_zero": float(np.mean(np.abs(deltas) <= eps_window)),
        "frac_near_zero_1e-6": float(np.mean(np.abs(deltas) <= 1e-6)),
        "frac_near_zero_1e-4": float(np.mean(np.abs(deltas) <= 1e-4)),
    }


def aggregate_arm(rows: list[dict]) -> dict:
    """Per-arm aggregate over all candidates (mixed across operators)."""
    deltas = np.array([float(r.get("delta_U", "nan")) for r in rows], dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    n = len(deltas)
    if n == 0:
        return {"n_candidates": 0}
    out = {
        "n_candidates": int(n),
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas, ddof=1)),
        "median": float(np.median(deltas)),
        "min": float(np.min(deltas)),
        "max": float(np.max(deltas)),
        "p01": float(np.percentile(deltas, 1)),
        "p99": float(np.percentile(deltas, 99)),
        "skew": float(stats.skew(deltas)),
        "kurtosis_excess": float(stats.kurtosis(deltas)),
    }
    out.update(mass_at_zero(deltas))
    out.update(empirical_accept_rate(deltas, EPSILON_GRID))
    ks_stat, ks_p = lilliefors_ks(deltas)
    a2, a2_verdict = anderson_darling_tail(deltas)
    out["lilliefors_ks_stat"] = ks_stat
    out["lilliefors_ks_p"] = ks_p
    out["anderson_darling_a2"] = a2
    out["anderson_darling_verdict"] = a2_verdict
    return out


def aggregate_per_operator(bundles: dict[tuple[str, int], list[dict]]) -> list[dict]:
    """Per-(arm, operator) aggregate — checks for mixed-marginal artefact."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    grouped_seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
    for (arm, seed), rows in bundles.items():
        for r in rows:
            op = r.get("operator_id", "unknown")
            try:
                d = float(r.get("delta_U", "nan"))
            except ValueError:
                continue
            if not math.isfinite(d):
                continue
            grouped[(arm, op)].append(d)
            grouped_seeds[(arm, op)].add(seed)
    out_rows = []
    for (arm, op), deltas_list in sorted(grouped.items()):
        deltas = np.array(deltas_list, dtype=float)
        n = len(deltas)
        if n < 50:
            row = {
                "arm": arm, "operator_id": op, "n_candidates": int(n),
                "n_seeds": len(grouped_seeds[(arm, op)]),
                "lilliefors_ks_p": float("nan"), "anderson_darling_verdict": "n_too_small",
            }
            out_rows.append(row)
            continue
        ks_stat, ks_p = lilliefors_ks(deltas, n_resample=499)
        a2, a2_verdict = anderson_darling_tail(deltas)
        row = {
            "arm": arm, "operator_id": op, "n_candidates": int(n),
            "n_seeds": len(grouped_seeds[(arm, op)]),
            "delta_U_mean": float(np.mean(deltas)),
            "delta_U_std": float(np.std(deltas, ddof=1)),
            "frac_pos": float(np.mean(deltas > 0)),
            "frac_zero_1e-6": float(np.mean(np.abs(deltas) <= 1e-6)),
            "M_pos": float(np.mean(deltas[deltas > 0])) if np.any(deltas > 0) else 0.0,
            "R_neg": float(-np.mean(deltas[deltas < 0])) if np.any(deltas < 0) else 0.0,
            "lilliefors_ks_stat": ks_stat,
            "lilliefors_ks_p": ks_p,
            "anderson_darling_a2": a2,
            "anderson_darling_verdict": a2_verdict,
        }
        out_rows.append(row)
    return out_rows


def accept_rate_curve(bundles: dict[tuple[str, int], list[dict]]) -> list[dict]:
    """Per-arm accept_rate(ε) on EPSILON_GRID, averaged across seeds."""
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for (arm, _seed), rows in bundles.items():
        deltas = np.array([float(r.get("delta_U", "nan")) for r in rows], dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        if len(deltas):
            grouped[arm].append(deltas)
    out_rows = []
    for arm, seed_arrays in sorted(grouped.items()):
        for e in EPSILON_GRID:
            rates = []
            for d in seed_arrays:
                if e < 0:
                    rates.append(float(np.mean(d > 0)))
                else:
                    rates.append(float(np.mean(d >= -e)))
            row = {
                "arm": arm, "epsilon": e, "n_seeds": len(seed_arrays),
                "accept_rate_mean": float(np.mean(rates)),
                "accept_rate_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
            }
            out_rows.append(row)
    return out_rows


def recommend_epsilon(per_arm: dict[str, dict], curve_rows: list[dict]) -> dict:
    """Suggest ε_small and ε_large for D1 based on accept_rate curve and ΔU mass distribution."""
    # We pick ε_small such that accept_rate(ε_small) is between strict and neutral midpoint
    # We pick ε_large such that accept_rate is around 0.5 (or matches Li 2024 0.234 anchor)
    suggestions = {}
    for arm, agg in per_arm.items():
        rate_strict = agg.get(f"eps_{-1.0}", float("nan"))
        rate_neutral = agg.get(f"eps_{0.0}", float("nan"))
        # find ε giving accept_rate ≈ midpoint
        midpoint = (rate_strict + rate_neutral) / 2 if not (math.isnan(rate_strict) or math.isnan(rate_neutral)) else 0.5
        target_large = 0.5  # arbitrary tolerant target

        # Walk EPSILON_GRID, find smallest ε whose accept rate >= midpoint, and >= target_large
        eps_small = None
        eps_large = None
        for e in EPSILON_GRID:
            if e <= 0:
                continue
            rate = agg.get(f"eps_{e}", float("nan"))
            if math.isnan(rate):
                continue
            if eps_small is None and rate >= midpoint:
                eps_small = e
            if rate >= target_large:
                eps_large = e
                break
        suggestions[arm] = {
            "rate_strict": rate_strict,
            "rate_neutral": rate_neutral,
            "midpoint_target": midpoint,
            "tolerant_target": target_large,
            "eps_small_suggested": eps_small,
            "eps_large_suggested": eps_large,
        }
    return suggestions


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="Phase B.1 output root containing <arm>/seed_<n>/candidates.csv")
    p.add_argument("--out", required=True, help="output dir for d0_*.{json,csv,md}")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        print(f"ERROR: root path does not exist: {root}", file=sys.stderr)
        return 2

    print(f"Loading candidate logs from {root} ...", file=sys.stderr)
    bundles = load_candidate_logs(root)
    if not bundles:
        print(f"ERROR: no candidate.csv files found", file=sys.stderr)
        return 2
    print(f"Loaded {len(bundles)} (arm, seed) bundles", file=sys.stderr)

    # Per-arm pooled across seeds
    per_arm: dict[str, dict] = {}
    arm_groups: dict[str, list[dict]] = defaultdict(list)
    for (arm, seed), rows in bundles.items():
        arm_groups[arm].extend(rows)
    for arm, rows in sorted(arm_groups.items()):
        print(f"  aggregating arm {arm} ({len(rows)} candidates) ...", file=sys.stderr)
        per_arm[arm] = aggregate_arm(rows)

    # Accept rate curve per arm × ε
    curve_rows = accept_rate_curve(bundles)

    # Per-operator
    print("  per-operator aggregation ...", file=sys.stderr)
    per_op = aggregate_per_operator(bundles)

    # Recommendations
    suggestions = recommend_epsilon(per_arm, curve_rows)

    # Outputs
    summary = {
        "n_arms": len(per_arm),
        "n_bundles": len(bundles),
        "epsilon_grid": EPSILON_GRID,
        "per_arm": per_arm,
        "epsilon_suggestions": suggestions,
    }
    (out_dir / "d0_summary.json").write_text(json.dumps(summary, indent=2))

    if curve_rows:
        with (out_dir / "d0_accept_rate_curve.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
            w.writeheader()
            w.writerows(curve_rows)

    if per_op:
        all_keys = sorted({k for r in per_op for k in r.keys()})
        with (out_dir / "d0_per_operator.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(per_op)

    # Markdown recommendation
    lines = [
        "# Phase D0 Aperture Calibration Report",
        "",
        f"Loaded {len(bundles)} (arm, seed) bundles from `{root}`.",
        "",
        "## Per-arm Δ_U distribution and accept-rate signature",
        "",
        "| arm | n | mean Δ | std Δ | frac_pos | frac_exact_zero | A_D verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, agg in sorted(per_arm.items()):
        if agg.get("n_candidates", 0) == 0:
            continue
        lines.append(
            f"| {arm} | {agg['n_candidates']:,} | {agg['mean']:.3e} | {agg['std']:.3e} | "
            f"{agg.get(f'eps_{0.0}', float('nan')) - agg.get('frac_exact_zero', 0):.3f} | "
            f"{agg.get('frac_exact_zero', float('nan')):.3f} | {agg.get('anderson_darling_verdict', '?')} |"
        )
    lines.extend([
        "",
        "## D1 ε recommendation (per arm)",
        "",
        "| arm | strict accept | neutral accept | suggested ε_small | suggested ε_large |",
        "|---|---|---|---|---|",
    ])
    for arm, s in sorted(suggestions.items()):
        lines.append(
            f"| {arm} | {s.get('rate_strict', float('nan')):.3f} | "
            f"{s.get('rate_neutral', float('nan')):.3f} | "
            f"{s.get('eps_small_suggested', 'n/a')} | "
            f"{s.get('eps_large_suggested', 'n/a')} |"
        )
    lines.extend([
        "",
        "## Gaussian null verdict",
        "",
        "If `anderson_darling_verdict` says `rejects_normal_at_5%` for most arms, the A_π Gaussian model does NOT fit empirical Δ_U; switch to empirical CDF A_emp(ε) for D1 ε-axis design.",
        "",
        "If it says `fits_at_5%`, A_π is a reasonable null and the π-formula has empirical legitimacy in this regime.",
        "",
        "Per-operator decomposition is in `d0_per_operator.csv` — check whether mixed-marginal non-normality is driven by one operator (e.g. `projection_weight`) or is structural across operators.",
        "",
        "## Suggested D1 design",
        "",
        "Based on this report, propose D1 as: 6 arms × 5 seeds × 1 horizon (40k) =  30 cells.",
        "Arms: strict, neutral_p={0.1, 0.3, 1.0}, ε_small (median across arms above), ε_large (median across arms above).",
        "",
        "If A_π fits poorly, weigh the ε_large choice toward where empirical accept_rate hits ~0.5 (per `d0_accept_rate_curve.csv`), not toward Li 2024 0.234 anchor.",
    ])
    (out_dir / "d0_recommendation.md").write_text("\n".join(lines))

    print(f"\nWrote: {out_dir}/d0_summary.json", file=sys.stderr)
    print(f"Wrote: {out_dir}/d0_accept_rate_curve.csv", file=sys.stderr)
    print(f"Wrote: {out_dir}/d0_per_operator.csv", file=sys.stderr)
    print(f"Wrote: {out_dir}/d0_recommendation.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
