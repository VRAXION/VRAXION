"""Generate the Phase B verdict bundle from completed run artifacts.

This script does not launch experiments. It reads the Phase B output directory,
validates artifact completeness, computes the preregistered statistics and
constructability diagnostics, writes machine-readable summaries, figures, and
the human report under docs/research.
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


DEFAULT_ROOT = Path("output/phase_b_full_20260424")
DEFAULT_REPORT = Path("docs/research/PHASE_B_VERDICT.md")
H256_REFERENCE_MEAN = 5.28
H256_REFERENCE_STD = 1.79
BONFERRONI_ALPHA = 0.0125
RAW_ALPHA = 0.05
BOOTSTRAP_N = 20_000
BOOTSTRAP_SEED = 20260425


@dataclass
class ArtifactPaths:
    root: Path
    analysis_dir: Path
    figures_dir: Path
    report_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_N)
    return parser.parse_args()


def ensure_dirs(root: Path, report_path: Path) -> ArtifactPaths:
    analysis_dir = root / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(root=root, analysis_dir=analysis_dir, figures_dir=figures_dir, report_path=report_path)


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(root / "results.csv")
    construct = pd.read_csv(root / "constructability_summary.csv")
    operators = pd.read_csv(root / "constructability_operator_summary.csv")
    panel_rows = []
    for panel_path in sorted(root.glob("B*/seed_*/panel_timeseries.csv")):
        panel = pd.read_csv(panel_path)
        run_dir = panel_path.parent
        arm = run_dir.parent.name
        seed = int(run_dir.name.split("_", 1)[1])
        panel["arm"] = arm
        panel["seed"] = seed
        panel["run_dir"] = str(run_dir)
        panel_rows.append(panel)
    panels = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame()
    return results, construct, operators, panels


def validate_artifacts(root: Path, results: pd.DataFrame, construct: pd.DataFrame, panels: pd.DataFrame) -> dict:
    candidates = list(root.glob("B*/seed_*/candidates.csv"))
    checkpoints = list(root.glob("B*/seed_*/final.ckpt"))
    panel_summaries = list(root.glob("B*/seed_*/panel_summary.json"))
    panel_timeseries = list(root.glob("B*/seed_*/panel_timeseries.csv"))
    expected_rows = int(results["expected_candidate_rows"].sum())
    actual_candidate_rows = int(construct["candidate_rows"].sum())
    panel_rows = int(len(panels))
    status = {
        "runs": int(len(results)),
        "constructability_runs": int(len(construct)),
        "candidate_files": len(candidates),
        "checkpoint_files": len(checkpoints),
        "panel_summary_files": len(panel_summaries),
        "panel_timeseries_files": len(panel_timeseries),
        "expected_candidate_rows": expected_rows,
        "actual_candidate_rows": actual_candidate_rows,
        "panel_rows": panel_rows,
        "passed": True,
        "failures": [],
    }
    checks = [
        (len(results) == 25, "expected 25 result rows"),
        (len(construct) == 25, "expected 25 constructability rows"),
        (len(candidates) == 25, "expected 25 candidate logs"),
        (len(checkpoints) == 25, "expected 25 final checkpoints"),
        (len(panel_summaries) == 25, "expected 25 panel summaries"),
        (len(panel_timeseries) == 25, "expected 25 panel timeseries files"),
        (actual_candidate_rows == 6_300_000, "expected 6,300,000 candidate rows"),
        (expected_rows == actual_candidate_rows, "expected rows should match candidate rows"),
        (panel_rows == 300, "expected 300 panel rows"),
    ]
    for ok, message in checks:
        if not ok:
            status["passed"] = False
            status["failures"].append(message)
    if not status["passed"]:
        raise SystemExit(f"artifact integrity failed: {status['failures']}")
    return status


def summarize_by_arm(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for arm, group in sorted(df.groupby("arm")):
        row = {"arm": arm, "n": int(len(group))}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_min"] = float(np.min(values))
            row[f"{metric}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci_delta(a: np.ndarray, b: np.ndarray, n: int) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(n):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        deltas.append(float(np.mean(a_s) - np.mean(b_s)))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = math.sqrt((var_a + var_b) / 2.0)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0


def welch_tests(results: pd.DataFrame, bootstrap_n: int) -> pd.DataFrame:
    baseline = results[results["arm"] == "B0"]["peak_acc"].to_numpy(dtype=float) * 100.0
    rows = []
    for arm in ["B1", "B2", "B3", "B4"]:
        values = results[results["arm"] == arm]["peak_acc"].to_numpy(dtype=float) * 100.0
        test = stats.ttest_ind(values, baseline, equal_var=False)
        delta = float(np.mean(values) - np.mean(baseline))
        ci_lo, ci_hi = bootstrap_ci_delta(values, baseline, bootstrap_n)
        if test.pvalue < BONFERRONI_ALPHA:
            verdict = "PASS_PREREG_SIGNIFICANT"
        elif test.pvalue < RAW_ALPHA:
            verdict = "DIRECTIONAL_UNDERPOWERED_AFTER_BONFERRONI"
        else:
            verdict = "FAIL_OR_INCONCLUSIVE"
        rows.append({
            "arm": arm,
            "baseline": "B0",
            "mean_peak": float(np.mean(values)),
            "baseline_mean_peak": float(np.mean(baseline)),
            "delta_mean_peak": delta,
            "bootstrap_ci95_low": ci_lo,
            "bootstrap_ci95_high": ci_hi,
            "welch_t": float(test.statistic),
            "welch_p": float(test.pvalue),
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "cohen_d": cohen_d(values, baseline),
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


def operator_summary(operators: pd.DataFrame) -> pd.DataFrame:
    grouped = operators.groupby(["arm", "operator_id"], as_index=False).agg(
        candidate_rows=("candidate_rows", "sum"),
        evaluated_rows=("evaluated_rows", "sum"),
        V_raw_mean=("V_raw", "mean"),
        M_pos_mean=("M_pos", "mean"),
        R_neg_mean=("R_neg", "mean"),
    )
    grouped["usefulness_proxy"] = grouped["V_raw_mean"] * grouped["M_pos_mean"]
    return grouped.sort_values(["arm", "usefulness_proxy"], ascending=[True, False])


def panel_summary(panels: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "main_peak_acc",
        "panel_probe_acc",
        "accept_rate_window",
        "edges",
        "f_active",
        "stable_rank",
        "kernel_rank",
        "separation_sp",
    ]
    rows = []
    for arm, group in sorted(panels.groupby("arm")):
        final = group.sort_values("step").groupby("seed").tail(1)
        row = {"arm": arm, "n": int(final["seed"].nunique())}
        for metric in metrics:
            values = final[metric].to_numpy(dtype=float)
            row[f"final_{metric}_mean"] = float(np.mean(values))
            row[f"final_{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def neutral_policy_summary(results: pd.DataFrame, construct: pd.DataFrame) -> dict:
    merged = results[["run_id", "arm", "seed", "peak_acc"]].merge(
        construct[["run_id", "accepted_positive_steps", "accepted_nonpositive_steps", "accepted_steps"]],
        on="run_id",
        how="inner",
    )
    merged["neutral_fraction"] = merged["accepted_nonpositive_steps"] / merged["accepted_steps"].replace(0, np.nan)
    corr = merged[["peak_acc", "neutral_fraction"]].corr(method="pearson").iloc[0, 1]
    by_arm = merged.groupby("arm", as_index=False).agg(
        mean_accepted_positive_steps=("accepted_positive_steps", "mean"),
        mean_accepted_nonpositive_steps=("accepted_nonpositive_steps", "mean"),
        mean_neutral_fraction=("neutral_fraction", "mean"),
    )
    return {
        "pearson_corr_peak_vs_neutral_fraction": None if pd.isna(corr) else float(corr),
        "by_arm": by_arm.to_dict(orient="records"),
    }


def derive_operator_findings(raw_operators: pd.DataFrame, ops: pd.DataFrame, results: pd.DataFrame) -> dict:
    b0 = ops[ops["arm"] == "B0"].copy()
    b0["schedule_share"] = b0["candidate_rows"] / b0["candidate_rows"].sum()
    b0_ranked = b0.sort_values("usefulness_proxy", ascending=False)

    def operator_record(arm_df: pd.DataFrame, operator_id: str) -> dict:
        rows = arm_df[arm_df["operator_id"] == operator_id]
        if rows.empty:
            return {}
        row = rows.iloc[0]
        return {
            "operator_id": operator_id,
            "candidate_rows": int(row["candidate_rows"]),
            "schedule_share": float(row.get("schedule_share", np.nan)),
            "V_raw_mean": float(row["V_raw_mean"]),
            "M_pos_mean": float(row["M_pos_mean"]),
            "R_neg_mean": float(row["R_neg_mean"]),
            "usefulness_proxy": float(row["usefulness_proxy"]),
            "productivity_share": float(row["usefulness_proxy"] / b0["usefulness_proxy"].sum()) if b0["usefulness_proxy"].sum() > 0 else 0.0,
        }

    top_b0 = [operator_record(b0, op) for op in b0_ranked["operator_id"].head(5)]
    projection_b0 = operator_record(b0, "projection_weight")
    theta_b0 = operator_record(b0, "theta")
    channel_b0 = operator_record(b0, "channel")

    b0_r = ops[ops["arm"] == "B0"][["operator_id", "R_neg_mean"]].rename(columns={"R_neg_mean": "B0_R_neg"})
    b3_r = ops[ops["arm"] == "B3"][["operator_id", "R_neg_mean"]].rename(columns={"R_neg_mean": "B3_R_neg"})
    b3_ratio = b3_r.merge(b0_r, on="operator_id", how="inner")
    b3_ratio = b3_ratio[b3_ratio["B0_R_neg"] > 0].copy()
    b3_ratio["R_neg_ratio"] = b3_ratio["B3_R_neg"] / b3_ratio["B0_R_neg"]
    b3_ratio = b3_ratio.sort_values("R_neg_ratio", ascending=False)

    b3_seed3042 = raw_operators[(raw_operators["arm"] == "B3") & (raw_operators["seed"] == 3042)].copy()
    b3_seed3042_nonzero = b3_seed3042[b3_seed3042["R_neg"] > 0]

    b2_peaks = results[results["arm"] == "B2"][["seed", "peak_acc", "final_acc", "accept_rate_pct"]].copy()
    b2_peaks["peak_acc_pct"] = b2_peaks["peak_acc"] * 100.0
    b2_peaks["final_acc_pct"] = b2_peaks["final_acc"] * 100.0

    return {
        "b0_top_productivity_operators": top_b0,
        "b0_projection_weight": projection_b0,
        "b0_theta": theta_b0,
        "b0_channel": channel_b0,
        "b3_vs_b0_r_neg_ratio": {
            "mean": float(b3_ratio["R_neg_ratio"].mean()),
            "min": float(b3_ratio["R_neg_ratio"].min()),
            "max": float(b3_ratio["R_neg_ratio"].max()),
            "by_operator": b3_ratio.to_dict(orient="records"),
        },
        "b3_seed3042_r_neg": {
            "mean_nonzero": float(b3_seed3042_nonzero["R_neg"].mean()) if not b3_seed3042_nonzero.empty else 0.0,
            "max_nonzero": float(b3_seed3042_nonzero["R_neg"].max()) if not b3_seed3042_nonzero.empty else 0.0,
            "operators": b3_seed3042.sort_values("R_neg", ascending=False).to_dict(orient="records"),
        },
        "b2_peak_distribution_pct": {
            "mean": float(b2_peaks["peak_acc_pct"].mean()),
            "std": float(b2_peaks["peak_acc_pct"].std(ddof=1)),
            "median": float(b2_peaks["peak_acc_pct"].median()),
            "min": float(b2_peaks["peak_acc_pct"].min()),
            "max": float(b2_peaks["peak_acc_pct"].max()),
            "by_seed": b2_peaks.sort_values("seed").to_dict(orient="records"),
        },
    }


def plot_arm_boxplots(results: pd.DataFrame, figures_dir: Path) -> None:
    data = [results[results["arm"] == arm]["peak_acc"].to_numpy() * 100.0 for arm in ["B0", "B1", "B2", "B3", "B4"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=["B0", "B1", "B2", "B3", "B4"], showmeans=True)
    ax.axhline(H256_REFERENCE_MEAN, color="tab:green", linestyle="--", linewidth=1.5, label="H=256 reference mean")
    ax.set_title("Phase B peak accuracy by arm")
    ax.set_ylabel("peak accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_b_peak_boxplot.png", dpi=160)
    plt.close(fig)

    data_final = [results[results["arm"] == arm]["final_acc"].to_numpy() * 100.0 for arm in ["B0", "B1", "B2", "B3", "B4"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data_final, tick_labels=["B0", "B1", "B2", "B3", "B4"], showmeans=True)
    ax.set_title("Phase B final accuracy by arm")
    ax.set_ylabel("final accuracy (%)")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_b_final_boxplot.png", dpi=160)
    plt.close(fig)


def plot_delta_and_ck(welch: pd.DataFrame, construct_arm: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(welch["arm"], welch["delta_mean_peak"], color="tab:blue")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Mean peak accuracy delta vs B0")
    ax.set_ylabel("delta peak accuracy (percentage points)")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_b_delta_vs_b0.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(construct_arm["arm"], construct_arm["C_K_window_ratio_mean"], color="tab:orange")
    ax.set_title("Constructability C_K window ratio by arm")
    ax.set_ylabel("mean C_K_window_ratio")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_b_ck_bar.png", dpi=160)
    plt.close(fig)


def plot_operator_heatmap(ops: pd.DataFrame, figures_dir: Path) -> None:
    pivot = ops.pivot(index="operator_id", columns="arm", values="usefulness_proxy").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Operator usefulness proxy: V_raw * M_pos")
    fig.colorbar(im, ax=ax, label="V_raw * M_pos")
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_b_operator_heatmap.png", dpi=160)
    plt.close(fig)


def plot_panel_timeseries(panels: pd.DataFrame, figures_dir: Path) -> None:
    for metric, ylabel in [
        ("main_peak_acc", "main peak accuracy"),
        ("accept_rate_window", "accept rate window"),
        ("stable_rank", "stable rank"),
        ("f_active", "f_active"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for arm, group in sorted(panels.groupby("arm")):
            arm_step = group.groupby("step")[metric].agg(["mean", "std"]).reset_index()
            x = arm_step["step"].to_numpy()
            mean = arm_step["mean"].to_numpy()
            std = arm_step["std"].fillna(0.0).to_numpy()
            ax.plot(x, mean, label=arm)
            ax.fill_between(x, mean - std, mean + std, alpha=0.12)
        ax.set_title(f"Panel timeseries: {metric}")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / f"phase_b_panel_{metric}.png", dpi=160)
        plt.close(fig)


def write_markdown_report(
    path: Path,
    artifact_status: dict,
    arm_stats: pd.DataFrame,
    welch: pd.DataFrame,
    construct_arm: pd.DataFrame,
    ops: pd.DataFrame,
    operator_findings: dict,
    neutral: dict,
) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        columns = [str(col) for col in df.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for _, row in df.iterrows():
            values = [str(row[col]).replace("|", "\\|") for col in df.columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    peak_cols = ["arm", "n", "peak_acc_mean", "peak_acc_std", "peak_acc_median", "peak_acc_min", "peak_acc_max", "final_acc_mean", "accept_rate_pct_mean"]
    peak_table = arm_stats[peak_cols].copy()
    for col in peak_table.columns:
        if col not in ["arm", "n"]:
            peak_table[col] = peak_table[col].map(lambda x: f"{x:.4f}")

    welch_table = welch.copy()
    for col in ["mean_peak", "baseline_mean_peak", "delta_mean_peak", "bootstrap_ci95_low", "bootstrap_ci95_high", "welch_t", "welch_p", "cohen_d"]:
        welch_table[col] = welch_table[col].map(lambda x: f"{x:.6g}")

    construct_cols = ["arm", "n", "C_K_window_ratio_mean", "V_raw_mean", "V_sel_mean", "M_pos_mean", "R_neg_mean", "cost_eval_ms_mean", "accepted_nonpositive_steps_mean"]
    construct_table = construct_arm[construct_cols].copy()
    for col in construct_table.columns:
        if col not in ["arm", "n"]:
            construct_table[col] = construct_table[col].map(lambda x: f"{x:.6g}")

    top_ops = ops.sort_values("usefulness_proxy", ascending=False).head(12).copy()
    for col in ["V_raw_mean", "M_pos_mean", "R_neg_mean", "usefulness_proxy"]:
        top_ops[col] = top_ops[col].map(lambda x: f"{x:.6g}")

    def pct(x: float) -> str:
        return f"{100.0 * x:.2f}%"

    projection = operator_findings["b0_projection_weight"]
    theta = operator_findings["b0_theta"]
    channel = operator_findings["b0_channel"]
    b3_ratio = operator_findings["b3_vs_b0_r_neg_ratio"]
    b3_seed3042 = operator_findings["b3_seed3042_r_neg"]
    b2_dist = operator_findings["b2_peak_distribution_pct"]
    top_b0 = pd.DataFrame(operator_findings["b0_top_productivity_operators"])
    top_b0_table = top_b0[["operator_id", "schedule_share", "productivity_share", "V_raw_mean", "M_pos_mean", "R_neg_mean", "usefulness_proxy"]].copy()
    for col in ["schedule_share", "productivity_share"]:
        top_b0_table[col] = top_b0_table[col].map(lambda x: pct(float(x)))
    for col in ["V_raw_mean", "M_pos_mean", "R_neg_mean", "usefulness_proxy"]:
        top_b0_table[col] = top_b0_table[col].map(lambda x: f"{float(x):.6g}")

    b1_row = arm_stats[arm_stats["arm"] == "B1"].iloc[0]
    b0_row = arm_stats[arm_stats["arm"] == "B0"].iloc[0]
    b1_test = welch[welch["arm"] == "B1"].iloc[0]
    verdict = (
        "Phase B supports a training-horizon confound: B1 (2x steps) has the highest mean peak "
        f"accuracy ({b1_row['peak_acc_mean']:.2f}% vs B0 {b0_row['peak_acc_mean']:.2f}%) and recovers "
        f"to the H=256 reference band ({H256_REFERENCE_MEAN:.2f} +/- {H256_REFERENCE_STD:.2f}%). "
        f"The Welch p-value is {b1_test['welch_p']:.4f}, directional at raw alpha=0.05 but not "
        "Bonferroni-significant at alpha=0.0125."
    )

    lines = [
        "# Phase B Verdict",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "Do not claim formal preregistered significance. Claim a strong directional finding and require an n>=10 replication for the strict Bonferroni gate.",
        "",
        "## Artifact Integrity",
        "",
        f"- Runs: `{artifact_status['runs']}`",
        f"- Candidate rows: `{artifact_status['actual_candidate_rows']:,}`",
        f"- Checkpoints: `{artifact_status['checkpoint_files']}`",
        f"- Panel timeseries files: `{artifact_status['panel_timeseries_files']}`",
        f"- Panel rows: `{artifact_status['panel_rows']}`",
        "",
        "## Arm Statistics",
        "",
        markdown_table(peak_table),
        "",
        "## Pre-Registered Tests",
        "",
        markdown_table(welch_table),
        "",
        "## Constructability",
        "",
        markdown_table(construct_table),
        "",
        "C_K agrees only weakly with peak accuracy: B1 is slightly above B0, while B3/B4 collapse clearly. Treat C_K as a diagnostic panel for now, not as a frozen scalar objective.",
        "",
        "## Operator Findings",
        "",
        markdown_table(top_ops[["arm", "operator_id", "candidate_rows", "V_raw_mean", "M_pos_mean", "R_neg_mean", "usefulness_proxy"]]),
        "",
        "### Operator Interpretation",
        "",
        markdown_table(top_b0_table),
        "",
        f"- `projection_weight` is effectively inert in B0: it used {pct(projection['schedule_share'])} of candidate draws but only reached `V_raw={projection['V_raw_mean']:.6g}` and `usefulness_proxy={projection['usefulness_proxy']:.6g}`.",
        f"- `theta` is the strongest B0 operator by `V_raw*M_pos`: current draw share {pct(theta['schedule_share'])}, productivity share {pct(theta['productivity_share'])}, `V_raw={theta['V_raw_mean']:.6g}`.",
        f"- `channel` is the second strongest B0 operator by the same proxy: current draw share {pct(channel['schedule_share'])}, productivity share {pct(channel['productivity_share'])}.",
        "- The current mutation schedule is plausibly misallocated: high-draw operators like `add_edge` are not the most productive, while `theta` is under-sampled. A theta/channel-heavy retuned schedule is a Phase C hypothesis, not a Phase B result.",
        f"- B2 shows heavy-tail behavior rather than reliable improvement: peak accuracy mean `{b2_dist['mean']:.2f}%`, median `{b2_dist['median']:.2f}%`, std `{b2_dist['std']:.2f}%`, range `{b2_dist['min']:.2f}%..{b2_dist['max']:.2f}%`.",
        f"- B3 increases destructive risk: arm-level `R_neg` is `{b3_ratio['mean']:.2f}x` B0 on average across operators (range `{b3_ratio['min']:.2f}x..{b3_ratio['max']:.2f}x`). The worst seed-level B3 case has nonzero `R_neg` mean `{b3_seed3042['mean_nonzero']:.3f}` and max `{b3_seed3042['max_nonzero']:.3f}`.",
        "- Do not label B3 as resonance or chaos from this analysis alone. The data supports higher destructive mutation sensitivity at 12 ticks; the mechanism needs a separate perturbation/Derrida-style test.",
        "",
        "## Neutral Accept Policy",
        "",
        f"- Pearson correlation between peak accuracy and neutral-accept fraction: `{neutral['pearson_corr_peak_vs_neutral_fraction']}`",
        "- Neutral accepted mutations are valid under `accept_ties=true`; this is policy behavior, not run corruption.",
        "- Follow-up should test `accept_ties=true` vs `accept_ties=false` on the winning B1 horizon condition.",
        "",
        "## Next Step",
        "",
        "Run Phase B.1: H=384 with horizon scaling and tie-policy ablation. Minimum matrix: `accept_ties=true/false` x `20k/40k/80k` on the same seeds, with the same candidate/panel logging.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = ensure_dirs(args.root, args.report)
    results, construct, operators, panels = load_inputs(paths.root)

    # Convert accuracy metrics to percentage points for all human-facing tables.
    results_pct = results.copy()
    for col in ["peak_acc", "final_acc", "alive_frac_mean"]:
        results_pct[col] = results_pct[col] * 100.0

    artifact_status = validate_artifacts(paths.root, results, construct, panels)
    arm_stats = summarize_by_arm(
        results_pct,
        ["peak_acc", "final_acc", "accept_rate_pct", "alive_frac_mean", "wall_clock_s"],
    )
    construct_arm = summarize_by_arm(
        construct,
        ["C_K_window_ratio", "V_raw", "V_sel", "M_pos", "R_neg", "cost_eval_ms", "accepted_nonpositive_steps"],
    )
    welch = welch_tests(results, args.bootstrap_n)
    ops = operator_summary(operators)
    panel_arm = panel_summary(panels)
    neutral = neutral_policy_summary(results, construct)
    operator_findings = derive_operator_findings(operators, ops, results)

    phase_b_verdict = {
        "artifact_status": artifact_status,
        "h256_reference": {"mean_peak": H256_REFERENCE_MEAN, "std_peak": H256_REFERENCE_STD},
        "headline": {
            "winner_by_mean_peak": str(arm_stats.sort_values("peak_acc_mean", ascending=False).iloc[0]["arm"]),
            "interpretation": "training_horizon_confound_supported_directional_underpowered",
            "bonferroni_alpha": BONFERRONI_ALPHA,
        },
        "arm_stats": arm_stats.to_dict(orient="records"),
        "welch_tests": welch.to_dict(orient="records"),
        "constructability_by_arm": construct_arm.to_dict(orient="records"),
        "panel_by_arm": panel_arm.to_dict(orient="records"),
        "operator_findings": operator_findings,
        "neutral_accept_policy": neutral,
    }

    (paths.analysis_dir / "phase_b_verdict.json").write_text(json.dumps(phase_b_verdict, indent=2), encoding="utf-8")
    arm_stats.to_csv(paths.analysis_dir / "phase_b_arm_stats.csv", index=False)
    welch.to_csv(paths.analysis_dir / "phase_b_welch_tests.csv", index=False)
    ops.to_csv(paths.analysis_dir / "phase_b_operator_summary.csv", index=False)
    panel_arm.to_csv(paths.analysis_dir / "phase_b_panel_summary.csv", index=False)

    plot_arm_boxplots(results, paths.figures_dir)
    plot_delta_and_ck(welch, construct_arm, paths.figures_dir)
    plot_operator_heatmap(ops, paths.figures_dir)
    plot_panel_timeseries(panels, paths.figures_dir)

    write_markdown_report(paths.report_path, artifact_status, arm_stats, welch, construct_arm, ops, operator_findings, neutral)

    print(json.dumps({
        "status": "PASS",
        "analysis_dir": str(paths.analysis_dir),
        "report": str(paths.report_path),
        "winner_by_mean_peak": phase_b_verdict["headline"]["winner_by_mean_peak"],
        "interpretation": phase_b_verdict["headline"]["interpretation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
