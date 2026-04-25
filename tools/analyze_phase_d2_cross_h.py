"""Analyze Phase D2 cross-H search-activation validation artifacts.

D2 asks whether the D1 H=384 result generalizes across smaller H:
  - H in {128, 256}
  - K in {1, 3, 9}
  - policy in {strict, zero_p=1.0}
  - mutual_inhibition, 40k steps, 5 seeds
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_ROOT = Path("output/phase_d2_cross_h_activation_20260426")
DEFAULT_REPORT = Path("docs/research/PHASE_D2_CROSS_H_VERDICT.md")
EXPECTED_ARMS = [
    "D2_K1_STRICT",
    "D2_K1_ZERO_P10",
    "D2_K3_STRICT",
    "D2_K3_ZERO_P10",
    "D2_K9_STRICT",
    "D2_K9_ZERO_P10",
]
POLICY_ORDER = ["strict", "zero_p_1.0"]


@dataclass
class Paths:
    root: Path
    analysis_dir: Path
    figures_dir: Path
    report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--expected-steps", type=int, default=40_000)
    parser.add_argument("--expected-H", default="128,256")
    return parser.parse_args()


def ensure_dirs(root: Path, report: Path) -> Paths:
    analysis_dir = root / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return Paths(root, analysis_dir, figures_dir, report)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"required artifact missing: {path}")
    return pd.read_csv(path)


def finite_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def policy_label(row: pd.Series) -> str:
    policy = str(row.get("accept_policy", "")).strip().lower()
    neutral_p = finite_float(row.get("neutral_p", float("nan")))
    arm = str(row.get("arm", ""))
    if policy == "strict" or arm.endswith("_STRICT"):
        return "strict"
    if policy == "zero-p" or "ZERO_P" in arm:
        if math.isfinite(neutral_p):
            return f"zero_p_{neutral_p:.1f}"
        return "zero_p_1.0"
    return policy or arm


def normalize(results: pd.DataFrame, construct: pd.DataFrame, operators: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for frame in [results, construct, operators]:
        if frame.empty:
            continue
        frame["H"] = pd.to_numeric(frame["H"], errors="coerce").astype("Int64")
        frame["jackpot"] = pd.to_numeric(frame["jackpot"], errors="coerce").astype("Int64")
        if "neutral_p" in frame.columns:
            frame["neutral_p"] = pd.to_numeric(frame["neutral_p"], errors="coerce")
        else:
            frame["neutral_p"] = np.nan
        if "accept_policy" not in frame.columns:
            frame["accept_policy"] = ""
        frame["policy_label"] = frame.apply(policy_label, axis=1)
    for metric in ["peak_acc", "final_acc"]:
        values = pd.to_numeric(results[metric], errors="coerce")
        results[f"{metric}_pct"] = values * 100.0 if values.max(skipna=True) <= 1.0 else values
    results["peak_final_gap_pp"] = results["peak_acc_pct"] - results["final_acc_pct"]
    return results, construct, operators


def validate(paths: Paths, results: pd.DataFrame, construct: pd.DataFrame, expected_h: list[int], expected_seeds: int) -> dict:
    expected_runs = len(expected_h) * len(EXPECTED_ARMS) * expected_seeds
    failures = []
    candidates = list(paths.root.glob("H_*/D2_*/seed_*/candidates.csv"))
    checkpoints = list(paths.root.glob("H_*/D2_*/seed_*/final.ckpt"))
    panel_summaries = list(paths.root.glob("H_*/D2_*/seed_*/panel_summary.json"))
    panel_timeseries = list(paths.root.glob("H_*/D2_*/seed_*/panel_timeseries.csv"))
    if len(results) != expected_runs:
        failures.append(f"expected {expected_runs} result rows, got {len(results)}")
    if len(construct) != expected_runs:
        failures.append(f"expected {expected_runs} constructability rows, got {len(construct)}")
    for label, files in [
        ("candidate logs", candidates),
        ("checkpoints", checkpoints),
        ("panel summaries", panel_summaries),
        ("panel timeseries", panel_timeseries),
    ]:
        if len(files) != expected_runs:
            failures.append(f"expected {expected_runs} {label}, got {len(files)}")
    missing_arms = sorted(set(EXPECTED_ARMS) - set(results["arm"]))
    extra_arms = sorted(set(results["arm"]) - set(EXPECTED_ARMS))
    missing_h = sorted(set(expected_h) - {int(x) for x in results["H"].dropna().unique()})
    if missing_arms:
        failures.append(f"missing arms: {missing_arms}")
    if extra_arms:
        failures.append(f"unexpected arms: {extra_arms}")
    if missing_h:
        failures.append(f"missing H values: {missing_h}")
    expected_candidate_rows = int(pd.to_numeric(results["expected_candidate_rows"], errors="coerce").sum())
    actual_candidate_rows = int(pd.to_numeric(construct["candidate_rows"], errors="coerce").sum())
    if expected_candidate_rows != actual_candidate_rows:
        failures.append(f"candidate rows mismatch expected={expected_candidate_rows} actual={actual_candidate_rows}")
    if failures:
        raise SystemExit(f"Phase D2 artifact validation failed: {failures}")
    return {
        "passed": True,
        "runs": int(len(results)),
        "candidate_rows": actual_candidate_rows,
        "checkpoints": len(checkpoints),
        "panel_summaries": len(panel_summaries),
        "panel_timeseries_files": len(panel_timeseries),
    }


def summarize(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row["n"] = int(len(group))
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols) if rows else pd.DataFrame()


def welch(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    if len(left) < 2 or len(right) < 2:
        return float("nan"), float("nan"), float("nan")
    test = stats.ttest_ind(left, right, equal_var=False)
    pooled = math.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2.0)
    d = float((np.mean(left) - np.mean(right)) / pooled) if pooled > 0 else 0.0
    return float(test.statistic), float(test.pvalue), d


def comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric = "peak_acc_pct"
    for h in sorted(results["H"].dropna().unique()):
        h_subset = results[results["H"] == h]
        for jackpot in [1, 3, 9]:
            subset = h_subset[h_subset["jackpot"] == jackpot]
            ties = subset[subset["policy_label"] == "zero_p_1.0"][metric].dropna().to_numpy(dtype=float)
            strict = subset[subset["policy_label"] == "strict"][metric].dropna().to_numpy(dtype=float)
            if len(ties) and len(strict):
                t, p, d = welch(ties, strict)
                rows.append({
                    "axis": "policy_within_H_K",
                    "comparison": f"H{int(h)}_K{jackpot}:zero_p_1.0_vs_strict",
                    "H": int(h),
                    "left_K": jackpot,
                    "right_K": jackpot,
                    "left_policy": "zero_p_1.0",
                    "right_policy": "strict",
                    "left_mean_peak": float(np.mean(ties)),
                    "right_mean_peak": float(np.mean(strict)),
                    "delta_pp": float(np.mean(ties) - np.mean(strict)),
                    "welch_t": t,
                    "welch_p": p,
                    "cohen_d": d,
                })
        for policy in POLICY_ORDER:
            subset = h_subset[h_subset["policy_label"] == policy]
            for left, right in [(3, 1), (9, 3), (9, 1)]:
                a = subset[subset["jackpot"] == left][metric].dropna().to_numpy(dtype=float)
                b = subset[subset["jackpot"] == right][metric].dropna().to_numpy(dtype=float)
                if len(a) and len(b):
                    t, p, d = welch(a, b)
                    rows.append({
                        "axis": "K_within_H_policy",
                        "comparison": f"H{int(h)}_{policy}:K{left}_vs_K{right}",
                        "H": int(h),
                        "left_K": left,
                        "right_K": right,
                        "left_policy": policy,
                        "right_policy": policy,
                        "left_mean_peak": float(np.mean(a)),
                        "right_mean_peak": float(np.mean(b)),
                        "delta_pp": float(np.mean(a) - np.mean(b)),
                        "welch_t": t,
                        "welch_p": p,
                        "cohen_d": d,
                    })
    return pd.DataFrame(rows)


def plot_heatmap(grouped: pd.DataFrame, h: int, value_col: str, title: str, path: Path, fmt: str = ".2f") -> None:
    sub = grouped[grouped["H"] == h]
    if sub.empty or value_col not in sub.columns:
        return
    pivot = sub.pivot(index="jackpot", columns="policy_label", values=value_col).reindex(index=[1, 3, 9], columns=POLICY_ORDER)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(POLICY_ORDER)), POLICY_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(3), ["K=1", "K=3", "K=9"])
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if math.isfinite(data[i, j]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def winner_table(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, sub in grouped.groupby("H"):
        best = sub.sort_values("peak_acc_pct_mean", ascending=False).iloc[0]
        rows.append({
            "H": int(h),
            "winner_K": int(best["jackpot"]),
            "winner_policy": str(best["policy_label"]),
            "winner_peak_mean": float(best["peak_acc_pct_mean"]),
            "winner_peak_std": float(best["peak_acc_pct_std"]),
        })
    return pd.DataFrame(rows).sort_values("H")


def write_report(paths: Paths, validation: dict, grouped: pd.DataFrame, construct_grouped: pd.DataFrame, comp: pd.DataFrame, winners: pd.DataFrame) -> None:
    merged = grouped.merge(
        construct_grouped[["H", "jackpot", "policy_label", "C_K_window_ratio_mean"]],
        on=["H", "jackpot", "policy_label"],
        how="left",
    )
    lines = [
        "# Phase D2 Verdict: Cross-H Search Activation",
        "",
        "D2 tests whether the D1 H=384 activation result generalizes to smaller H.",
        "",
        "## Integrity",
        "",
        f"- Runs: `{validation['runs']}`",
        f"- Candidate rows: `{validation['candidate_rows']}`",
        f"- Checkpoints: `{validation['checkpoints']}`",
        f"- Panel summaries: `{validation['panel_summaries']}`",
        f"- Panel timeseries files: `{validation['panel_timeseries_files']}`",
        "",
        "## Winners",
        "",
        "| H | winner K | winner policy | peak mean | peak std |",
        "|---:|---:|---|---:|---:|",
    ]
    for _, row in winners.iterrows():
        lines.append(f"| {int(row['H'])} | {int(row['winner_K'])} | {row['winner_policy']} | {row['winner_peak_mean']:.2f} | {row['winner_peak_std']:.2f} |")
    lines.extend([
        "",
        "## Arm Stats",
        "",
        "| H | K | policy | n | peak mean | peak std | final mean | accept mean | alive mean | C_K mean |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in merged.sort_values(["H", "jackpot", "policy_label"]).iterrows():
        lines.append(
            f"| {int(row['H'])} | {int(row['jackpot'])} | {row['policy_label']} | {int(row['n'])} | "
            f"{row.get('peak_acc_pct_mean', float('nan')):.2f} | "
            f"{row.get('peak_acc_pct_std', float('nan')):.2f} | "
            f"{row.get('final_acc_pct_mean', float('nan')):.2f} | "
            f"{row.get('accept_rate_pct_mean', float('nan')):.2f} | "
            f"{row.get('alive_frac_mean_mean', float('nan')):.3f} | "
            f"{row.get('C_K_window_ratio_mean', float('nan')):.3e} |"
        )
    lines.extend([
        "",
        "## Comparisons",
        "",
        "Welch tests are diagnostic at n=5 and not used alone as the decision criterion.",
        "",
        "| axis | comparison | delta pp | p | d |",
        "|---|---|---:|---:|---:|",
    ])
    for _, row in comp.iterrows():
        lines.append(f"| {row['axis']} | {row['comparison']} | {row['delta_pp']:.2f} | {row['welch_p']:.4g} | {row['cohen_d']:.2f} |")
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Machine summary: `{paths.analysis_dir / 'phase_d2_cross_h_verdict.json'}`",
        f"- Group stats: `{paths.analysis_dir / 'phase_d2_group_stats.csv'}`",
        f"- Comparisons: `{paths.analysis_dir / 'phase_d2_comparisons.csv'}`",
        f"- Figures: `{paths.figures_dir}`",
    ])
    paths.report_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    paths = ensure_dirs(args.root, args.report)
    expected_h = [int(x) for x in args.expected_H.split(",") if x.strip()]
    results = read_csv_required(paths.root / "results.csv")
    construct = read_csv_required(paths.root / "constructability_summary.csv")
    operators = read_csv_required(paths.root / "constructability_operator_summary.csv")
    results, construct, operators = normalize(results, construct, operators)
    validation = validate(paths, results, construct, expected_h, args.expected_seeds)

    result_metrics = ["peak_acc_pct", "final_acc_pct", "peak_final_gap_pp", "accept_rate_pct", "alive_frac_mean", "wall_clock_s"]
    construct_metrics = ["C_K_window_ratio", "V_raw", "V_sel", "M_pos", "R_neg", "cost_eval_ms", "accepted_positive_steps", "accepted_nonpositive_steps"]
    grouped = summarize(results, ["H", "jackpot", "policy_label"], result_metrics)
    construct_grouped = summarize(construct, ["H", "jackpot", "policy_label"], construct_metrics)
    operator_grouped = summarize(operators, ["H", "jackpot", "policy_label", "operator_id"], ["V_raw", "M_pos", "R_neg"])
    comp = comparisons(results)
    winners = winner_table(grouped)

    grouped.to_csv(paths.analysis_dir / "phase_d2_group_stats.csv", index=False)
    construct_grouped.to_csv(paths.analysis_dir / "phase_d2_constructability_stats.csv", index=False)
    operator_grouped.to_csv(paths.analysis_dir / "phase_d2_operator_stats.csv", index=False)
    comp.to_csv(paths.analysis_dir / "phase_d2_comparisons.csv", index=False)
    winners.to_csv(paths.analysis_dir / "phase_d2_winners.csv", index=False)
    merged_for_plot = grouped.merge(construct_grouped, on=["H", "jackpot", "policy_label"], how="left")
    for h in expected_h:
        plot_heatmap(merged_for_plot, h, "peak_acc_pct_mean", f"Phase D2 H={h} mean peak accuracy (%)", paths.figures_dir / f"phase_d2_H{h}_peak_heatmap.png")
        plot_heatmap(merged_for_plot, h, "C_K_window_ratio_mean", f"Phase D2 H={h} mean C_K", paths.figures_dir / f"phase_d2_H{h}_ck_heatmap.png", ".2e")

    payload = {
        "root": str(paths.root),
        "validation": validation,
        "winners": json.loads(winners.to_json(orient="records")),
        "group_stats": json.loads(grouped.to_json(orient="records")),
        "constructability_stats": json.loads(construct_grouped.to_json(orient="records")),
        "comparisons": json.loads(comp.to_json(orient="records")),
    }
    (paths.analysis_dir / "phase_d2_cross_h_verdict.json").write_text(json.dumps(payload, indent=2))
    write_report(paths, validation, grouped, construct_grouped, comp, winners)

    print(json.dumps({
        "status": "PASS",
        "runs": validation["runs"],
        "candidate_rows": validation["candidate_rows"],
        "winners": json.loads(winners.to_json(orient="records")),
        "report": str(paths.report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
