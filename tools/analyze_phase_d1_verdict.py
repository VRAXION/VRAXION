"""Analyze Phase D1 K x zero_p search-activation sweep artifacts.

Primary question:
  What is the best measured search activation setting (K*, zero_p*) for
  mutual_inhibition, H=384, 40k-step runs?

Inputs are the normal sweep artifacts produced by diag_dimensionality_sweep.py:
  results.csv
  constructability_summary.csv
  constructability_operator_summary.csv
  D1_*/seed_*/candidates.csv
  D1_*/seed_*/final.ckpt
  D1_*/seed_*/panel_summary.json
  D1_*/seed_*/panel_timeseries.csv
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


DEFAULT_ROOT = Path("output/phase_d1_activation_20260425")
DEFAULT_REPORT = Path("docs/research/PHASE_D1_VERDICT.md")
EXPECTED_ARMS = [
    "D1_K1_STRICT",
    "D1_K1_ZERO_P03",
    "D1_K1_ZERO_P10",
    "D1_K3_STRICT",
    "D1_K3_ZERO_P03",
    "D1_K3_ZERO_P10",
    "D1_K9_STRICT",
    "D1_K9_ZERO_P03",
    "D1_K9_ZERO_P10",
]
POLICY_ORDER = ["strict", "zero_p_0.3", "zero_p_1.0"]


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
        if arm.endswith("ZERO_P03"):
            return "zero_p_0.3"
        if arm.endswith("ZERO_P10"):
            return "zero_p_1.0"
    if str(row.get("accept_ties", "")).strip().lower() in {"true", "1", "yes"}:
        return "zero_p_1.0"
    return policy or arm


def normalize(results: pd.DataFrame, construct: pd.DataFrame, operators: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for frame in [results, construct, operators]:
        if frame.empty:
            continue
        if "jackpot" in frame.columns:
            frame["jackpot"] = pd.to_numeric(frame["jackpot"], errors="coerce").astype("Int64")
        if "neutral_p" in frame.columns:
            frame["neutral_p"] = pd.to_numeric(frame["neutral_p"], errors="coerce")
        else:
            frame["neutral_p"] = np.nan
        if "accept_policy" not in frame.columns:
            frame["accept_policy"] = ""
        frame["policy_label"] = frame.apply(policy_label, axis=1)
        if "phase" not in frame.columns:
            frame["phase"] = ""
    for metric in ["peak_acc", "final_acc"]:
        if metric in results.columns:
            values = pd.to_numeric(results[metric], errors="coerce")
            results[f"{metric}_pct"] = values * 100.0 if values.max(skipna=True) <= 1.0 else values
    if "peak_acc_pct" in results.columns and "final_acc_pct" in results.columns:
        results["peak_final_gap_pp"] = results["peak_acc_pct"] - results["final_acc_pct"]
    return results, construct, operators


def validate(paths: Paths, results: pd.DataFrame, construct: pd.DataFrame, expected_seeds: int, expected_steps: int) -> dict:
    expected_runs = len(EXPECTED_ARMS) * expected_seeds
    candidates = list(paths.root.glob("D1_*/seed_*/candidates.csv"))
    checkpoints = list(paths.root.glob("D1_*/seed_*/final.ckpt"))
    panel_summaries = list(paths.root.glob("D1_*/seed_*/panel_summary.json"))
    panel_timeseries = list(paths.root.glob("D1_*/seed_*/panel_timeseries.csv"))
    failures = []
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
    if missing_arms:
        failures.append(f"missing arms: {missing_arms}")
    if extra_arms:
        failures.append(f"unexpected arms: {extra_arms}")
    if "expected_candidate_rows" in results.columns and "candidate_rows" in construct.columns:
        expected_candidate_rows = int(pd.to_numeric(results["expected_candidate_rows"], errors="coerce").sum())
        actual_candidate_rows = int(pd.to_numeric(construct["candidate_rows"], errors="coerce").sum())
        if expected_candidate_rows != actual_candidate_rows:
            failures.append(f"candidate rows mismatch expected={expected_candidate_rows} actual={actual_candidate_rows}")
    else:
        expected_candidate_rows = int(expected_steps * expected_seeds * (1 + 1 + 1 + 3 + 3 + 3 + 9 + 9 + 9))
        actual_candidate_rows = int(pd.to_numeric(construct.get("candidate_rows", pd.Series(dtype=float)), errors="coerce").sum())
    if failures:
        raise SystemExit(f"Phase D1 artifact validation failed: {failures}")
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
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(group_cols)
    return out


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
    for jackpot in [1, 3, 9]:
        subset = results[results["jackpot"] == jackpot]
        for left, right in [("zero_p_0.3", "strict"), ("zero_p_1.0", "strict"), ("zero_p_1.0", "zero_p_0.3")]:
            a = subset[subset["policy_label"] == left][metric].dropna().to_numpy(dtype=float)
            b = subset[subset["policy_label"] == right][metric].dropna().to_numpy(dtype=float)
            if len(a) and len(b):
                t, p, d = welch(a, b)
                rows.append({
                    "axis": "policy_within_K",
                    "comparison": f"K{jackpot}:{left}_vs_{right}",
                    "left_K": jackpot,
                    "right_K": jackpot,
                    "left_policy": left,
                    "right_policy": right,
                    "left_mean_peak": float(np.mean(a)),
                    "right_mean_peak": float(np.mean(b)),
                    "delta_pp": float(np.mean(a) - np.mean(b)),
                    "welch_t": t,
                    "welch_p": p,
                    "cohen_d": d,
                })
    for policy in POLICY_ORDER:
        subset = results[results["policy_label"] == policy]
        for left, right in [(3, 1), (9, 3), (9, 1)]:
            a = subset[subset["jackpot"] == left][metric].dropna().to_numpy(dtype=float)
            b = subset[subset["jackpot"] == right][metric].dropna().to_numpy(dtype=float)
            if len(a) and len(b):
                t, p, d = welch(a, b)
                rows.append({
                    "axis": "K_within_policy",
                    "comparison": f"{policy}:K{left}_vs_K{right}",
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


def add_policy_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["policy_label"] = pd.Categorical(df["policy_label"], categories=POLICY_ORDER, ordered=True)
    return df.sort_values(["jackpot", "policy_label"])


def plot_heatmap(stats_df: pd.DataFrame, value_col: str, title: str, path: Path, fmt: str = ".2f") -> None:
    if stats_df.empty or value_col not in stats_df.columns:
        return
    pivot = stats_df.pivot(index="jackpot", columns="policy_label", values=value_col).reindex(index=[1, 3, 9], columns=POLICY_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(POLICY_ORDER)), POLICY_ORDER, rotation=25, ha="right")
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


def plot_figures(grouped: pd.DataFrame, figures_dir: Path) -> None:
    plot_heatmap(grouped, "peak_acc_pct_mean", "Phase D1 mean peak accuracy (%)", figures_dir / "phase_d1_peak_heatmap.png")
    plot_heatmap(grouped, "final_acc_pct_mean", "Phase D1 mean final accuracy (%)", figures_dir / "phase_d1_final_heatmap.png")
    plot_heatmap(grouped, "accept_rate_pct_mean", "Phase D1 mean accept rate (%)", figures_dir / "phase_d1_accept_heatmap.png")
    if "C_K_window_ratio_mean" in grouped.columns:
        plot_heatmap(grouped, "C_K_window_ratio_mean", "Phase D1 mean C_K window ratio", figures_dir / "phase_d1_ck_heatmap.png", ".2e")


def decision_lines(best: dict, grouped: pd.DataFrame) -> list[str]:
    lines = []
    k = int(best.get("jackpot", -1))
    policy = str(best.get("policy_label", ""))
    lines.append(f"Measured winner by mean `peak_acc`: `K={k}`, `{policy}`.")
    if k == 9 and policy == "zero_p_1.0":
        lines.append("Decision rule: full neutral walk with saturated funnel is supported.")
    elif k == 3 and policy in {"zero_p_0.3", "zero_p_1.0"}:
        lines.append("Decision rule: K=9 is likely over-saturated; smaller funnel plus neutral valve is supported.")
    elif k == 1:
        lines.append("Decision rule: jackpot discovery may be less important than policy softness in this regime, but check variance before adopting.")
    elif policy == "strict":
        lines.append("Decision rule: neutral drift is harmful or unnecessary in the winning aperture regime.")
    elif policy == "zero_p_0.3":
        lines.append("Decision rule: controlled Zero-Drive is sufficient; full ties are looser than needed.")
    if "peak_acc_pct_mean" in grouped.columns:
        k1 = grouped[grouped["jackpot"] == 1]["peak_acc_pct_mean"]
        if len(k1) and float(k1.max()) < float(grouped["peak_acc_pct_mean"].max()) - 0.25:
            lines.append("K=1 is weaker than the best grid point, so some jackpot discovery remains useful.")
    return lines


def write_report(paths: Paths, validation: dict, grouped: pd.DataFrame, construct_grouped: pd.DataFrame, comp: pd.DataFrame, best: dict, decisions: list[str], expected_seeds: int) -> None:
    lines = [
        "# Phase D1 Verdict: Search-Activation Sweep",
        "",
        "D1 tests the aperture as a two-axis search activation: jackpot pool size `K` and neutral-step valve `zero_p`.",
        "",
        "## Integrity",
        "",
        f"- Runs: `{validation['runs']}`",
        f"- Candidate rows: `{validation['candidate_rows']}`",
        f"- Checkpoints: `{validation['checkpoints']}`",
        f"- Panel summaries: `{validation['panel_summaries']}`",
        f"- Panel timeseries files: `{validation['panel_timeseries_files']}`",
        f"- Expected seeds per arm: `{expected_seeds}`",
        "",
        "## Winner",
        "",
        f"- Best mean peak: `K={int(best['jackpot'])}`, `{best['policy_label']}` = `{best['peak_acc_pct_mean']:.2f}%`",
        f"- Mean final: `{best.get('final_acc_pct_mean', float('nan')):.2f}%`",
        f"- Mean accept rate: `{best.get('accept_rate_pct_mean', float('nan')):.2f}%`",
        "",
        "## Decision Notes",
        "",
    ]
    lines.extend(f"- {line}" for line in decisions)
    lines.extend([
        "",
        "## Arm Stats",
        "",
        "| K | policy | n | peak mean | peak std | final mean | accept mean | alive mean | C_K mean |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    merged = grouped.merge(
        construct_grouped[["jackpot", "policy_label", "C_K_window_ratio_mean"]] if "C_K_window_ratio_mean" in construct_grouped.columns else construct_grouped[["jackpot", "policy_label"]],
        on=["jackpot", "policy_label"],
        how="left",
    )
    merged = add_policy_order(merged)
    for _, row in merged.iterrows():
        lines.append(
            f"| {int(row['jackpot'])} | {row['policy_label']} | {int(row['n'])} | "
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
        lines.append(
            f"| {row['axis']} | {row['comparison']} | {row['delta_pp']:.2f} | "
            f"{row['welch_p']:.4g} | {row['cohen_d']:.2f} |"
        )
    lines.extend([
        "",
        "## Outputs",
        "",
        f"- Machine summary: `{paths.analysis_dir / 'phase_d1_verdict.json'}`",
        f"- Group stats: `{paths.analysis_dir / 'phase_d1_group_stats.csv'}`",
        f"- Comparisons: `{paths.analysis_dir / 'phase_d1_comparisons.csv'}`",
        f"- Figures: `{paths.figures_dir}`",
    ])
    paths.report_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    paths = ensure_dirs(args.root, args.report)

    results = read_csv_required(paths.root / "results.csv")
    construct = read_csv_required(paths.root / "constructability_summary.csv")
    operators = read_csv_required(paths.root / "constructability_operator_summary.csv")
    results, construct, operators = normalize(results, construct, operators)
    validation = validate(paths, results, construct, args.expected_seeds, args.expected_steps)

    result_metrics = [
        "peak_acc_pct",
        "final_acc_pct",
        "peak_final_gap_pp",
        "accept_rate_pct",
        "alive_frac_mean",
        "wall_clock_s",
    ]
    construct_metrics = [
        "C_K_window_ratio",
        "V_raw",
        "V_sel",
        "M_pos",
        "R_neg",
        "cost_eval_ms",
        "accepted_positive_steps",
        "accepted_nonpositive_steps",
    ]
    grouped = add_policy_order(summarize(results, ["jackpot", "policy_label"], result_metrics))
    construct_grouped = add_policy_order(summarize(construct, ["jackpot", "policy_label"], construct_metrics))
    operator_grouped = add_policy_order(summarize(operators, ["jackpot", "policy_label", "operator_id"], ["V_raw", "M_pos", "R_neg"]))
    comp = comparisons(results)
    best_row = grouped.sort_values("peak_acc_pct_mean", ascending=False).iloc[0].to_dict()
    decisions = decision_lines(best_row, grouped)

    grouped.to_csv(paths.analysis_dir / "phase_d1_group_stats.csv", index=False)
    construct_grouped.to_csv(paths.analysis_dir / "phase_d1_constructability_stats.csv", index=False)
    operator_grouped.to_csv(paths.analysis_dir / "phase_d1_operator_stats.csv", index=False)
    comp.to_csv(paths.analysis_dir / "phase_d1_comparisons.csv", index=False)
    plot_figures(grouped.merge(construct_grouped, on=["jackpot", "policy_label"], how="left"), paths.figures_dir)

    payload = {
        "root": str(paths.root),
        "validation": validation,
        "best": best_row,
        "decision_notes": decisions,
        "group_stats": json.loads(grouped.to_json(orient="records")),
        "constructability_stats": json.loads(construct_grouped.to_json(orient="records")),
        "comparisons": json.loads(comp.to_json(orient="records")),
    }
    (paths.analysis_dir / "phase_d1_verdict.json").write_text(json.dumps(payload, indent=2))
    write_report(paths, validation, grouped, construct_grouped, comp, best_row, decisions, args.expected_seeds)

    print(json.dumps({
        "status": "PASS",
        "runs": validation["runs"],
        "candidate_rows": validation["candidate_rows"],
        "best": {
            "jackpot": int(best_row["jackpot"]),
            "policy_label": str(best_row["policy_label"]),
            "peak_acc_pct_mean": best_row["peak_acc_pct_mean"],
        },
        "report": str(paths.report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
