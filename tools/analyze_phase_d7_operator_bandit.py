"""Analyze Phase D7.1 Safe Operator Bandit runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("output/phase_d7_operator_bandit_20260427")
DEFAULT_REPORT = Path("docs/research/PHASE_D7_OPERATOR_BANDIT_AUDIT.md")
ENTROPY_FLOOR = 0.60
ENTROPY_RATIO_FLOOR = 0.75
MATERIAL_REGRESSION_PP = -0.50


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_results(root: Path) -> pd.DataFrame:
    path = root / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing results.csv: {path}")
    df = pd.read_csv(path)
    df["peak_acc_pct"] = df["peak_acc"].astype(float) * 100.0
    df["final_acc_pct"] = df["final_acc"].astype(float) * 100.0
    df["H"] = df["H"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["jackpot"] = df["jackpot"].astype(int)
    return df


def load_operator_policy(root: Path, results: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in results.to_dict("records"):
        path = Path(str(row.get("operator_policy_timeseries", "")))
        if not path.exists():
            run_dir = Path(str(row["run_dir"]))
            path = run_dir / "operator_policy_timeseries.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["H"] = int(row["H"])
        df["seed"] = int(row["seed"])
        df["arm"] = str(row["arm"])
        df["run_id"] = str(row["run_id"])
        df["jackpot"] = int(row["jackpot"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_peak_timing(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in results.to_dict("records"):
        run_dir = Path(str(row["run_dir"]))
        panel = read_csv(run_dir / "panel_timeseries.csv")
        candidate_rows_to_peak = math.nan
        step_to_peak = math.nan
        if not panel.empty and "main_peak_acc" in panel.columns:
            peak = float(panel["main_peak_acc"].max())
            hit = panel[panel["main_peak_acc"] >= peak - 1e-12]
            if not hit.empty:
                step_to_peak = int(hit.iloc[0]["step"])
                candidate_rows_to_peak = step_to_peak * int(row["jackpot"])
        row["step_to_peak_panel"] = step_to_peak
        row["candidate_rows_to_peak"] = candidate_rows_to_peak
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, op_ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = (
        results.groupby(["H", "arm"], dropna=False)
        .agg(
            n=("peak_acc_pct", "size"),
            peak_mean_pct=("peak_acc_pct", "mean"),
            peak_median_pct=("peak_acc_pct", "median"),
            peak_std_pct=("peak_acc_pct", "std"),
            final_mean_pct=("final_acc_pct", "mean"),
            accept_mean_pct=("accept_rate_pct", "mean"),
            rows_to_peak_median=("candidate_rows_to_peak", "median"),
            wall_mean_s=("wall_clock_s", "mean"),
        )
        .reset_index()
    )
    entropy = pd.DataFrame()
    effectiveness = pd.DataFrame()
    if not op_ts.empty:
        entropy = (
            op_ts.groupby(["H", "arm", "run_id"], dropna=False)["normalized_entropy"]
            .median()
            .reset_index(name="entropy_median_run")
            .groupby(["H", "arm"], dropna=False)
            .agg(entropy_median=("entropy_median_run", "median"), entropy_min=("entropy_median_run", "min"))
            .reset_index()
        )
        effectiveness = (
            op_ts.groupby(["H", "arm", "operator_id"], dropna=False)
            .agg(
                probability_mean=("probability", "mean"),
                attempts=("attempts_window", "sum"),
                selected=("selected_window", "sum"),
                accepted=("accepted_window", "sum"),
                positive_delta=("positive_delta_window", "sum"),
            )
            .reset_index()
        )
        effectiveness["positive_delta_rate"] = effectiveness["positive_delta"] / effectiveness["attempts"].replace(0, np.nan)
        effectiveness["accepted_rate"] = effectiveness["accepted"] / effectiveness["attempts"].replace(0, np.nan)
    return grouped, entropy, effectiveness


def paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    base = results[results["arm"] == "D7_BASELINE"][
        ["H", "seed", "peak_acc_pct", "final_acc_pct", "candidate_rows_to_peak", "accept_rate_pct"]
    ].rename(columns={
        "peak_acc_pct": "baseline_peak_pct",
        "final_acc_pct": "baseline_final_pct",
        "candidate_rows_to_peak": "baseline_rows_to_peak",
        "accept_rate_pct": "baseline_accept_rate_pct",
    })
    rows = []
    for arm in ["D7_STATIC_PRIOR", "D7_PRIOR_EWMA"]:
        treatment = results[results["arm"] == arm][
            ["H", "seed", "peak_acc_pct", "final_acc_pct", "candidate_rows_to_peak", "accept_rate_pct"]
        ].rename(columns={
            "peak_acc_pct": "treatment_peak_pct",
            "final_acc_pct": "treatment_final_pct",
            "candidate_rows_to_peak": "treatment_rows_to_peak",
            "accept_rate_pct": "treatment_accept_rate_pct",
        })
        joined = base.merge(treatment, on=["H", "seed"], how="inner")
        joined["arm"] = arm
        joined["delta_peak_pct"] = joined["treatment_peak_pct"] - joined["baseline_peak_pct"]
        joined["delta_final_pct"] = joined["treatment_final_pct"] - joined["baseline_final_pct"]
        joined["delta_rows_to_peak"] = joined["treatment_rows_to_peak"] - joined["baseline_rows_to_peak"]
        joined["delta_accept_rate_pct"] = joined["treatment_accept_rate_pct"] - joined["baseline_accept_rate_pct"]
        rows.append(joined)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def leave_one_seed_medians(deltas: pd.DataFrame, arm: str) -> dict:
    sub = deltas[deltas["arm"] == arm]
    seeds = sorted(sub["seed"].unique())
    vals = []
    for seed in seeds:
        rem = sub[sub["seed"] != seed]
        if not rem.empty:
            vals.append(float(rem["delta_peak_pct"].median()))
    return {"min_leave_one_seed_median_delta": min(vals) if vals else math.nan}


def decide(deltas: pd.DataFrame, summary: pd.DataFrame, entropy: pd.DataFrame) -> tuple[str, dict]:
    candidates = {}
    for arm in ["D7_STATIC_PRIOR", "D7_PRIOR_EWMA"]:
        sub = deltas[deltas["arm"] == arm]
        if sub.empty:
            continue
        h_mean = sub.groupby("H")["delta_peak_pct"].mean().to_dict()
        h_improved = sum(1 for v in h_mean.values() if v > 0.0)
        material_regression = any(v < MATERIAL_REGRESSION_PP for v in h_mean.values())
        median_delta = float(sub["delta_peak_pct"].median())
        mean_delta = float(sub["delta_peak_pct"].mean())
        rows_delta_median = float(sub["delta_rows_to_peak"].median())
        accept_delta = float(sub["delta_accept_rate_pct"].median())
        loo = leave_one_seed_medians(deltas, arm)
        entropy_ok = True
        entropy_notes = []
        if not entropy.empty:
            for h in sorted(sub["H"].unique()):
                b = entropy[(entropy["H"] == h) & (entropy["arm"] == "D7_BASELINE")]
                t = entropy[(entropy["H"] == h) & (entropy["arm"] == arm)]
                if b.empty or t.empty:
                    continue
                t_med = float(t.iloc[0]["entropy_median"])
                b_med = float(b.iloc[0]["entropy_median"])
                ratio = t_med / b_med if b_med > 0 else 1.0
                entropy_notes.append({"H": int(h), "entropy_median": t_med, "entropy_ratio_vs_baseline": ratio})
                if t_med < ENTROPY_FLOOR or ratio < ENTROPY_RATIO_FLOOR:
                    entropy_ok = False
        peak_gate = median_delta > 0.0
        h_gate = h_improved >= 2 or not material_regression
        seed_gate = loo["min_leave_one_seed_median_delta"] > MATERIAL_REGRESSION_PP
        rows_gate = rows_delta_median <= 0.0 or median_delta > 0.0
        accept_only_fail = median_delta <= 0.0 and accept_delta > 0.0
        lock = peak_gate and h_gate and seed_gate and rows_gate and entropy_ok and not accept_only_fail
        candidates[arm] = {
            "median_delta_peak_pct": median_delta,
            "mean_delta_peak_pct": mean_delta,
            "h_mean_delta_peak_pct": {str(k): float(v) for k, v in h_mean.items()},
            "h_improved_count": h_improved,
            "material_regression": material_regression,
            "median_delta_rows_to_peak": rows_delta_median,
            "median_delta_accept_rate_pct": accept_delta,
            "entropy_ok": entropy_ok,
            "entropy_notes": entropy_notes,
            "accept_only_fail": accept_only_fail,
            **loo,
            "lock": lock,
        }
    if any(v["lock"] for v in candidates.values()):
        return "D7_OPERATOR_BANDIT_LOCK", candidates
    if any(v["median_delta_peak_pct"] > 0.0 for v in candidates.values()):
        return "D7_BANDIT_WEAK_SIGNAL", candidates
    if all(v["mean_delta_peak_pct"] < MATERIAL_REGRESSION_PP for v in candidates.values()):
        return "D7_BANDIT_REGRESSION", candidates
    return "D7_NEEDS_ARCHIVE_OR_FEATURE_POLICY", candidates


def plot_outputs(root: Path, summary: pd.DataFrame, deltas: pd.DataFrame, entropy: pd.DataFrame, op_eff: pd.DataFrame) -> None:
    fig_dir = root / "analysis" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        pivot = summary.pivot(index="H", columns="arm", values="peak_mean_pct")
        ax = pivot.plot(kind="bar", figsize=(9, 4))
        ax.set_ylabel("mean peak_acc (%)")
        ax.set_title("D7.1 peak by H and arm")
        plt.tight_layout()
        plt.savefig(fig_dir / "peak_by_arm_per_H.png", dpi=160)
        plt.close()
    if not deltas.empty:
        ax = deltas.boxplot(column="delta_peak_pct", by="arm", figsize=(8, 4))
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("paired delta peak_acc (pp)")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(fig_dir / "seed_paired_delta_peak.png", dpi=160)
        plt.close()
    if not entropy.empty:
        pivot = entropy.pivot(index="H", columns="arm", values="entropy_median")
        ax = pivot.plot(kind="bar", figsize=(9, 4))
        ax.axhline(ENTROPY_FLOOR, color="red", linestyle="--", linewidth=1)
        ax.set_ylabel("median normalized operator entropy")
        plt.tight_layout()
        plt.savefig(fig_dir / "operator_entropy_by_arm.png", dpi=160)
        plt.close()
    if not op_eff.empty:
        top = op_eff.groupby(["H", "arm"]).apply(lambda x: x.nlargest(5, "probability_mean")).reset_index(drop=True)
        top.to_csv(root / "analysis" / "operator_top_probabilities.csv", index=False)


def write_report(path: Path, verdict: str, decision: dict, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase D7.1 Operator Bandit Audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Summary",
        "",
        "- D7.1 tests only operator sampling weights over locked SAF v1.",
        "- Fixed: mutual_inhibition, strict gate, K(H), horizon, seeds, candidate budget.",
        "- Treatments are compared by paired H/seed deltas against D7_BASELINE.",
        "",
        "## Arm Stats",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Paired Delta Stats",
        "",
    ]
    if deltas.empty:
        lines.append("No paired deltas available.")
    else:
        stats = deltas.groupby("arm").agg(
            n=("delta_peak_pct", "size"),
            median_delta_peak_pct=("delta_peak_pct", "median"),
            mean_delta_peak_pct=("delta_peak_pct", "mean"),
            median_delta_rows_to_peak=("delta_rows_to_peak", "median"),
        ).reset_index()
        lines.extend(["```text", stats.to_string(index=False), "```"])
    lines.extend(["", "## Decision", "", "```json", json.dumps(decision, indent=2), "```", ""])
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    root = Path(args.root)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    results = add_peak_timing(load_results(root))
    op_ts = load_operator_policy(root, results)
    summary, entropy, op_eff = summarize(results, op_ts)
    deltas = paired_deltas(results)
    verdict, decision = decide(deltas, summary, entropy)

    summary.to_csv(analysis / "per_H_arm_summary.csv", index=False)
    deltas.to_csv(analysis / "seed_paired_deltas.csv", index=False)
    entropy.to_csv(analysis / "operator_entropy_summary.csv", index=False)
    op_eff.to_csv(analysis / "operator_effectiveness_by_arm.csv", index=False)
    op_ts.to_csv(analysis / "operator_probability_timeseries.csv", index=False)
    payload = {
        "verdict": verdict,
        "runs": int(len(results)),
        "operator_policy_rows": int(len(op_ts)),
        "decision": decision,
    }
    (analysis / "summary.json").write_text(json.dumps(payload, indent=2))
    plot_outputs(root, summary, deltas, entropy, op_eff)
    write_report(Path(args.report), verdict, decision, summary, deltas)
    print(f"Verdict: {verdict}")
    print(f"Runs: {len(results)}")
    print(f"Wrote: {analysis / 'summary.json'}")
    print(f"Wrote: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
