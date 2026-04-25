"""Offline jackpot-aperture resampling for Phase D0.5.

Reads existing K=9 candidate logs and estimates what the best-of-K aperture
would have looked like for smaller prefix K values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("output/phase_b1_horizon_ties_20260425")
DEFAULT_OUT = Path("output/phase_d0_5_jackpot_aperture")
K_VALUES = (1, 2, 3, 5, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-runs", type=int, default=30)
    parser.add_argument("--expected-rows", type=int, default=12_600_000)
    return parser.parse_args()


def load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def finish_step(rows: list[tuple[float, bool, float]], per_k: dict[int, dict], step_wall_ms: float) -> None:
    if not rows:
        return
    for k in K_VALUES:
        subset = rows[:k]
        eligible = [delta for delta, within_cap, _ in subset if within_cap]
        best = max(eligible) if eligible else float("nan")
        stats = per_k[k]
        stats["steps"] += 1
        stats["step_cost_ms_sum"] += step_wall_ms * min(k, len(rows)) / max(1, len(rows))
        if math.isnan(best):
            stats["ineligible"] += 1
            continue
        stats["useful_delta_sum"] += max(0.0, best)
        if best > 0.0:
            stats["best_positive"] += 1
        elif best == 0.0:
            stats["best_zero"] += 1
        else:
            stats["best_negative"] += 1
        for eps in (0.0, 1e-6, 1e-4, 1e-3):
            if best >= -eps:
                stats[f"accept_eps_{eps:g}"] += 1


def new_k_stats() -> dict:
    return {
        "steps": 0,
        "ineligible": 0,
        "best_positive": 0,
        "best_zero": 0,
        "best_negative": 0,
        "useful_delta_sum": 0.0,
        "step_cost_ms_sum": 0.0,
        "accept_eps_0": 0,
        "accept_eps_1e-06": 0,
        "accept_eps_0.0001": 0,
        "accept_eps_0.001": 0,
    }


def analyze_run(csv_path: Path) -> tuple[list[dict], int]:
    meta = load_meta(csv_path.parent)
    arm = csv_path.parent.parent.name
    seed = int(csv_path.parent.name.split("_", 1)[1])
    horizon_steps = int(meta.get("horizon_steps", meta.get("steps", 0)))
    accept_ties = bool(meta.get("accept_ties", False))
    jackpot = int(meta.get("jackpot", 9))
    if jackpot < max(K_VALUES):
        raise ValueError(f"{csv_path}: jackpot={jackpot}, cannot resample K={max(K_VALUES)}")

    per_k = {k: new_k_stats() for k in K_VALUES}
    raw_rows = 0
    raw_positive = 0
    raw_zero = 0
    raw_negative = 0
    current_step: int | None = None
    step_rows: list[tuple[float, bool, float]] = []
    step_wall_ms = 0.0

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows += 1
            step = int(row["step"])
            if current_step is None:
                current_step = step
            if step != current_step:
                finish_step(step_rows, per_k, step_wall_ms)
                current_step = step
                step_rows = []
                step_wall_ms = 0.0

            before = float(row["before_U"])
            after = float(row["after_U"])
            delta = float(row["delta_U"])
            if abs(delta - (after - before)) > 1e-12:
                raise ValueError(f"{csv_path}: delta mismatch at row {raw_rows}")
            if delta > 0.0:
                raw_positive += 1
            elif delta == 0.0:
                raw_zero += 1
            else:
                raw_negative += 1
            step_wall_ms = max(step_wall_ms, float(row["step_wall_ms"]))
            step_rows.append((delta, row["within_cap"].lower() == "true", float(row["candidate_eval_ms"])))

    finish_step(step_rows, per_k, step_wall_ms)
    raw_total = max(1, raw_rows)
    p_pos = raw_positive / raw_total
    p_zero = raw_zero / raw_total
    p_neg = raw_negative / raw_total
    out = []
    for k, stats in per_k.items():
        steps = max(1, stats["steps"])
        strict_pred = 1.0 - (1.0 - p_pos) ** k
        ties_pred = 1.0 - p_neg**k
        strict_emp = stats["best_positive"] / steps
        ties_emp = (stats["best_positive"] + stats["best_zero"]) / steps
        out.append({
            "arm": arm,
            "seed": seed,
            "horizon_steps": horizon_steps,
            "accept_ties": accept_ties,
            "K": k,
            "steps": stats["steps"],
            "raw_rows": raw_rows,
            "raw_p_pos": p_pos,
            "raw_p_zero": p_zero,
            "raw_p_neg": p_neg,
            "best_positive_rate": strict_emp,
            "best_zero_rate": stats["best_zero"] / steps,
            "best_negative_rate": stats["best_negative"] / steps,
            "ties_accept_rate": ties_emp,
            "strict_accept_rate": strict_emp,
            "strict_accept_pred_independent": strict_pred,
            "ties_accept_pred_independent": ties_pred,
            "strict_pred_error": strict_emp - strict_pred,
            "ties_pred_error": ties_emp - ties_pred,
            "C_K_window_ratio": stats["useful_delta_sum"] / max(stats["step_cost_ms_sum"], 1e-12),
            "ineligible_steps": stats["ineligible"],
        })
    return out, raw_rows


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metrics = [
        "best_positive_rate",
        "best_zero_rate",
        "best_negative_rate",
        "ties_accept_rate",
        "strict_accept_rate",
        "strict_accept_pred_independent",
        "ties_accept_pred_independent",
        "strict_pred_error",
        "ties_pred_error",
        "C_K_window_ratio",
        "raw_p_pos",
        "raw_p_zero",
        "raw_p_neg",
    ]
    grouped = []
    for key, group in df.groupby(["horizon_steps", "accept_ties", "arm", "K"], dropna=False):
        row = {col: value for col, value in zip(["horizon_steps", "accept_ties", "arm", "K"], key)}
        row["n"] = int(len(group))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        grouped.append(row)
    return pd.DataFrame(grouped).sort_values(["horizon_steps", "accept_ties", "arm", "K"])


def recommend(agg: pd.DataFrame) -> dict:
    # Focus on the B.1 40k strict operating point because D1's planned horizon is 40k.
    subset = agg[(agg["horizon_steps"] == 40_000) & (agg["accept_ties"] == False)]
    if subset.empty:
        subset = agg[agg["accept_ties"] == False]
    candidates = subset[
        (subset["ties_accept_rate_mean"] < 0.95)
        & (subset["best_positive_rate_mean"] >= 0.05)
    ]
    if candidates.empty:
        chosen = subset.sort_values("K").iloc[-1]
        reason = "No K preserved ties_accept<0.95 and positive_best>=0.05; using largest observed K."
    else:
        chosen = candidates.sort_values("K").iloc[-1]
        reason = "Largest K preserving neutral saturation below 95% while keeping positive discovery >=5%."
    return {
        "recommended_K": int(chosen["K"]),
        "basis_arm": str(chosen["arm"]),
        "basis_horizon_steps": int(chosen["horizon_steps"]),
        "ties_accept_rate": float(chosen["ties_accept_rate_mean"]),
        "strict_accept_rate": float(chosen["strict_accept_rate_mean"]),
        "best_zero_rate": float(chosen["best_zero_rate_mean"]),
        "reason": reason,
    }


def make_plots(out_dir: Path, agg: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    focus = agg[(agg["horizon_steps"] == 40_000) & (agg["accept_ties"] == False)]
    if focus.empty:
        focus = agg[agg["accept_ties"] == False]
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label in [
        ("best_positive_rate_mean", "positive best"),
        ("best_zero_rate_mean", "zero best"),
        ("ties_accept_rate_mean", "ties accept"),
    ]:
        ax.plot(focus["K"], focus[metric], marker="o", label=label)
    ax.axhline(0.95, color="tab:red", linestyle="--", linewidth=1, label="95% saturation")
    ax.set_title("D0.5 jackpot aperture at 40k strict reference")
    ax.set_xlabel("prefix K")
    ax.set_ylabel("rate")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "d0_5_k_aperture_40k_strict.png", dpi=160)
    plt.close(fig)


def render_report(out_dir: Path, agg: pd.DataFrame, rec: dict, total_rows: int) -> None:
    def md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for col in cols:
                value = row[col]
                values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    focus_cols = [
        "horizon_steps",
        "accept_ties",
        "arm",
        "K",
        "best_positive_rate_mean",
        "best_zero_rate_mean",
        "best_negative_rate_mean",
        "ties_accept_rate_mean",
        "strict_accept_rate_mean",
        "strict_accept_pred_independent_mean",
        "C_K_window_ratio_mean",
    ]
    focus = agg[(agg["horizon_steps"] == 40_000) & (agg["accept_ties"] == False)]
    report = [
        "# Phase D0.5 Jackpot Aperture",
        "",
        "## Verdict",
        "",
        f"- Candidate rows analyzed: `{total_rows:,}`.",
        f"- Recommended K for D1: `{rec['recommended_K']}`.",
        f"- Reason: {rec['reason']}",
        f"- Reference rates: ties_accept `{rec['ties_accept_rate']:.3f}`, strict_accept `{rec['strict_accept_rate']:.3f}`, zero_best `{rec['best_zero_rate']:.3f}`.",
        "",
        "## 40k Strict Reference",
        "",
        md(focus[focus_cols]),
        "",
        "## Interpretation",
        "",
        "- Jackpot K is the sampling funnel before acceptance policy; zero-p is the valve after best-of-K selection.",
        "- If K saturates ties acceptance near 1.0, a fixed-K zero-p sweep measures valve behavior in an already saturated funnel.",
        "- The independent-binomial prediction is a diagnostic only; candidate effects are not assumed iid.",
    ]
    (out_dir / "PHASE_D0_5_JACKPOT_APERTURE.md").write_text("\n".join(report) + "\n")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(args.root.glob("B1_*/seed_*/candidates.csv"))
    if len(csv_paths) != args.expected_runs:
        raise SystemExit(f"expected {args.expected_runs} candidate logs, found {len(csv_paths)}")
    rows = []
    total_rows = 0
    for path in csv_paths:
        run_rows, raw_rows = analyze_run(path)
        rows.extend(run_rows)
        total_rows += raw_rows
    if total_rows != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} rows, found {total_rows}")
    run_df = pd.DataFrame(rows)
    agg = aggregate(rows)
    rec = recommend(agg)
    run_df.to_csv(args.out / "jackpot_aperture_run_summary.csv", index=False)
    agg.to_csv(args.out / "jackpot_aperture_summary.csv", index=False)
    payload = {
        "status": "PASS",
        "root": str(args.root),
        "candidate_rows": total_rows,
        "recommendation": rec,
        "rows": agg.to_dict(orient="records"),
    }
    (args.out / "jackpot_aperture_summary.json").write_text(json.dumps(payload, indent=2))
    make_plots(args.out, agg)
    render_report(args.out, agg, rec, total_rows)
    print(json.dumps({
        "status": "PASS",
        "candidate_rows": total_rows,
        "recommended_K": rec["recommended_K"],
        "report": str(args.out / "PHASE_D0_5_JACKPOT_APERTURE.md"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
