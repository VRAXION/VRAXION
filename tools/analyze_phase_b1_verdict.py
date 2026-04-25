"""Analyze Phase B.1 horizon x accept_ties ablation artifacts."""
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


DEFAULT_ROOT = Path("output/phase_b1_horizon_ties_20260425")
DEFAULT_REPORT = Path("docs/research/PHASE_B1_VERDICT.md")
H256_REFERENCE_MEAN = 5.28
EXPECTED_ARMS = [
    "B1_S20_STRICT",
    "B1_S20_TIES",
    "B1_S40_STRICT",
    "B1_S40_TIES",
    "B1_S80_STRICT",
    "B1_S80_TIES",
]


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
    parser.add_argument("--base-steps", type=int, default=20_000)
    parser.add_argument("--expected-seeds", type=int, default=5)
    return parser.parse_args()


def ensure_dirs(root: Path, report: Path) -> Paths:
    analysis_dir = root / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    return Paths(root, analysis_dir, figures_dir, report)


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(root / "results.csv")
    construct = pd.read_csv(root / "constructability_summary.csv")
    operators = pd.read_csv(root / "constructability_operator_summary.csv")
    panel_rows = []
    for panel_path in sorted(root.glob("B1_*/seed_*/panel_timeseries.csv")):
        panel = pd.read_csv(panel_path)
        run_dir = panel_path.parent
        panel["arm"] = run_dir.parent.name
        panel["seed"] = int(run_dir.name.split("_", 1)[1])
        panel_rows.append(panel)
    panels = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame()
    return results, construct, operators, panels


def normalize(results: pd.DataFrame, construct: pd.DataFrame, operators: pd.DataFrame, panels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for frame in [results, construct, operators, panels]:
        if "horizon_steps" not in frame.columns and "configured_steps" in frame.columns:
            frame["horizon_steps"] = frame["configured_steps"]
        if "accept_ties" in frame.columns:
            frame["accept_ties"] = frame["accept_ties"].astype(str).str.lower().isin(["true", "1", "yes"])
    return results, construct, operators, panels


def validate(root: Path, results: pd.DataFrame, construct: pd.DataFrame, panels: pd.DataFrame, expected_seeds: int) -> dict:
    candidates = list(root.glob("B1_*/seed_*/candidates.csv"))
    checkpoints = list(root.glob("B1_*/seed_*/final.ckpt"))
    panel_summaries = list(root.glob("B1_*/seed_*/panel_summary.json"))
    panel_timeseries = list(root.glob("B1_*/seed_*/panel_timeseries.csv"))
    expected_runs = len(EXPECTED_ARMS) * expected_seeds
    expected_candidate_rows = int(results["expected_candidate_rows"].sum())
    actual_candidate_rows = int(construct["candidate_rows"].sum())
    failures = []
    if len(results) != expected_runs:
        failures.append(f"expected {expected_runs} result rows, got {len(results)}")
    if len(construct) != expected_runs:
        failures.append(f"expected {expected_runs} constructability rows, got {len(construct)}")
    if len(candidates) != expected_runs:
        failures.append(f"expected {expected_runs} candidate logs, got {len(candidates)}")
    if len(checkpoints) != expected_runs:
        failures.append(f"expected {expected_runs} checkpoints, got {len(checkpoints)}")
    if len(panel_summaries) != expected_runs:
        failures.append(f"expected {expected_runs} panel summaries, got {len(panel_summaries)}")
    if len(panel_timeseries) != expected_runs:
        failures.append(f"expected {expected_runs} panel timeseries, got {len(panel_timeseries)}")
    if expected_candidate_rows != actual_candidate_rows:
        failures.append("expected candidate rows do not match constructability rows")
    missing_arms = sorted(set(EXPECTED_ARMS) - set(results["arm"]))
    if missing_arms:
        failures.append(f"missing arms: {missing_arms}")
    if failures:
        raise SystemExit(f"Phase B.1 artifact validation failed: {failures}")
    return {
        "runs": int(len(results)),
        "candidate_rows": actual_candidate_rows,
        "checkpoints": len(checkpoints),
        "panel_timeseries_files": len(panel_timeseries),
        "panel_rows": int(len(panels)),
        "passed": True,
    }


def summarize(df: pd.DataFrame, metrics: list[str], group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row["n"] = int(len(group))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    test = stats.ttest_ind(a, b, equal_var=False)
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    d = float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0
    return float(test.statistic), float(test.pvalue), d


def comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for accept_ties in [False, True]:
        subset = results[results["accept_ties"] == accept_ties]
        for left, right in [(40_000, 20_000), (80_000, 40_000), (80_000, 20_000)]:
            a = subset[subset["horizon_steps"] == left]["peak_acc"].to_numpy(dtype=float) * 100.0
            b = subset[subset["horizon_steps"] == right]["peak_acc"].to_numpy(dtype=float) * 100.0
            if len(a) and len(b):
                t, p, d = welch(a, b)
                rows.append({
                    "accept_ties": accept_ties,
                    "comparison": f"{left}_vs_{right}",
                    "left_steps": left,
                    "right_steps": right,
                    "left_mean_peak": float(np.mean(a)),
                    "right_mean_peak": float(np.mean(b)),
                    "delta_pp": float(np.mean(a) - np.mean(b)),
                    "welch_t": t,
                    "welch_p": p,
                    "cohen_d": d,
                })
    for steps in sorted(results["horizon_steps"].unique()):
        ties = results[(results["horizon_steps"] == steps) & (results["accept_ties"])]["peak_acc"].to_numpy(dtype=float) * 100.0
        strict = results[(results["horizon_steps"] == steps) & (~results["accept_ties"])]["peak_acc"].to_numpy(dtype=float) * 100.0
        if len(ties) and len(strict):
            t, p, d = welch(ties, strict)
            rows.append({
                "accept_ties": "ties_vs_strict",
                "comparison": f"{steps}_ties_vs_strict",
                "left_steps": int(steps),
                "right_steps": int(steps),
                "left_mean_peak": float(np.mean(ties)),
                "right_mean_peak": float(np.mean(strict)),
                "delta_pp": float(np.mean(ties) - np.mean(strict)),
                "welch_t": t,
                "welch_p": p,
                "cohen_d": d,
            })
    return pd.DataFrame(rows)


def plot_results(results: pd.DataFrame, grouped: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for accept_ties, label in [(False, "strict"), (True, "ties")]:
        subset = grouped[grouped["accept_ties"] == accept_ties].sort_values("horizon_steps")
        ax.errorbar(
            subset["horizon_steps"],
            subset["peak_acc_mean"],
            yerr=subset["peak_acc_std"],
            marker="o",
            capsize=4,
            label=label,
        )
    ax.axhline(H256_REFERENCE_MEAN, color="tab:green", linestyle="--", linewidth=1.5, label="H=256 Phase A ref")
    ax.set_title("Phase B.1 peak accuracy by horizon and tie policy")
    ax.set_xlabel("horizon steps")
    ax.set_ylabel("peak accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_b1_peak_by_horizon.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for accept_ties, label in [(False, "strict"), (True, "ties")]:
        data = [
            results[(results["horizon_steps"] == steps) & (results["accept_ties"] == accept_ties)]["peak_acc"].to_numpy() * 100.0
            for steps in sorted(results["horizon_steps"].unique())
        ]
        positions = np.array(range(len(data)), dtype=float) + (-0.18 if not accept_ties else 0.18)
        ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, boxprops={"alpha": 0.6}, tick_labels=[""] * len(data))
    ax.set_xticks(range(len(sorted(results["horizon_steps"].unique()))), [str(x) for x in sorted(results["horizon_steps"].unique())])
    ax.set_title("Phase B.1 peak distribution")
    ax.set_xlabel("horizon steps")
    ax.set_ylabel("peak accuracy (%)")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_b1_peak_boxplot.png", dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in df.columns) + " |")
    return "\n".join(lines)


def write_report(path: Path, artifact_status: dict, grouped: pd.DataFrame, comp: pd.DataFrame, construct_grouped: pd.DataFrame) -> None:
    display = grouped.copy()
    for col in display.columns:
        if col not in ["horizon_steps", "accept_ties", "n"]:
            display[col] = display[col].map(lambda x: f"{float(x):.4f}")
    comp_display = comp.copy()
    for col in ["left_mean_peak", "right_mean_peak", "delta_pp", "welch_t", "welch_p", "cohen_d"]:
        comp_display[col] = comp_display[col].map(lambda x: f"{float(x):.6g}")
    construct_display = construct_grouped.copy()
    for col in construct_display.columns:
        if col not in ["horizon_steps", "accept_ties", "n"]:
            construct_display[col] = construct_display[col].map(lambda x: f"{float(x):.6g}")

    best = grouped.sort_values("peak_acc_mean", ascending=False).iloc[0]
    lines = [
        "# Phase B.1 Verdict",
        "",
        "## Verdict",
        "",
        f"Best mean peak arm: `steps={int(best['horizon_steps'])}`, `accept_ties={bool(best['accept_ties'])}`, mean peak `{best['peak_acc_mean']:.2f}%`.",
        "Interpretation must follow the comparison table below; this report is generated from the completed artifact bundle.",
        "",
        "## Artifact Integrity",
        "",
        f"- Runs: `{artifact_status['runs']}`",
        f"- Candidate rows: `{artifact_status['candidate_rows']:,}`",
        f"- Checkpoints: `{artifact_status['checkpoints']}`",
        f"- Panel timeseries files: `{artifact_status['panel_timeseries_files']}`",
        f"- Panel rows: `{artifact_status['panel_rows']}`",
        "",
        "## Peak/Final Summary",
        "",
        markdown_table(display),
        "",
        "## Comparisons",
        "",
        markdown_table(comp_display),
        "",
        "## Constructability Summary",
        "",
        markdown_table(construct_display),
        "",
        "## Decision Boundary",
        "",
        "- Horizon is strengthened if 40k or 80k strict recovers or exceeds the 5.28% H=256 reference.",
        "- A tie-policy effect is active only if same-horizon ties-vs-strict deltas are consistent, not just one outlier.",
        "- If 80k does not exceed 40k, treat H=384 as plateau/drift-limited after rescue.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = ensure_dirs(args.root, args.report)
    results, construct, operators, panels = load_inputs(paths.root)
    results, construct, operators, panels = normalize(results, construct, operators, panels)
    artifact_status = validate(paths.root, results, construct, panels, args.expected_seeds)

    results_pct = results.copy()
    for col in ["peak_acc", "final_acc", "alive_frac_mean"]:
        results_pct[col] = results_pct[col] * 100.0
    grouped = summarize(
        results_pct,
        ["peak_acc", "final_acc", "accept_rate_pct", "alive_frac_mean", "wall_clock_s"],
        ["horizon_steps", "accept_ties"],
    )
    construct_grouped = summarize(
        construct,
        ["C_K_window_ratio", "V_raw", "V_sel", "M_pos", "R_neg", "cost_eval_ms", "accepted_nonpositive_steps"],
        ["horizon_steps", "accept_ties"],
    )
    comp = comparisons(results)

    payload = {
        "artifact_status": artifact_status,
        "h256_reference_mean": H256_REFERENCE_MEAN,
        "summary": grouped.to_dict(orient="records"),
        "comparisons": comp.to_dict(orient="records"),
        "constructability": construct_grouped.to_dict(orient="records"),
    }
    (paths.analysis_dir / "phase_b1_verdict.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    grouped.to_csv(paths.analysis_dir / "phase_b1_horizon_tie_stats.csv", index=False)
    comp.to_csv(paths.analysis_dir / "phase_b1_comparisons.csv", index=False)
    construct_grouped.to_csv(paths.analysis_dir / "phase_b1_constructability_stats.csv", index=False)
    plot_results(results, grouped, paths.figures_dir)
    write_report(paths.report_path, artifact_status, grouped, comp, construct_grouped)

    print(json.dumps({
        "status": "PASS",
        "analysis_dir": str(paths.analysis_dir),
        "report": str(paths.report_path),
        "best": grouped.sort_values("peak_acc_mean", ascending=False).iloc[0].to_dict(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
