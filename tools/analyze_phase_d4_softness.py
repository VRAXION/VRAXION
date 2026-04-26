"""Analyze Phase D4 locked-K softness artifacts.

Phase D4 tests the SAF softness axis at the current K(H) lock:

  H=128 -> K=9
  H=256 -> K=18
  H=384 -> K=9

Tau is fixed at 0. The only policy axis is strict vs zero_p.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("output/phase_d4_softness_20260426")
DEFAULT_REPORT = Path("docs/research/PHASE_D4_SOFTNESS_VERDICT.md")
EXPECTED_ARMS = [
    "D4_H128_K9_STRICT",
    "D4_H128_K9_ZERO_P03",
    "D4_H128_K9_ZERO_P10",
    "D4_H256_K18_STRICT",
    "D4_H256_K18_ZERO_P03",
    "D4_H256_K18_ZERO_P10",
    "D4_H384_K9_STRICT",
    "D4_H384_K9_ZERO_P03",
    "D4_H384_K9_ZERO_P10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d4-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--expected-steps", type=int, default=40_000)
    parser.add_argument("--lock-margin-pp", type=float, default=0.5)
    parser.add_argument("--collapse-threshold-pp", type=float, default=1.0)
    parser.add_argument("--variance-penalty-pp", type=float, default=1.0)
    return parser.parse_args()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"required artifact missing: {path}")
    return pd.read_csv(path)


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["H"] = pd.to_numeric(out["H"], errors="coerce").astype("Int64")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["jackpot"] = pd.to_numeric(out["jackpot"], errors="coerce").astype("Int64")
    for metric in ["peak_acc", "final_acc"]:
        values = pd.to_numeric(out[metric], errors="coerce")
        out[f"{metric}_pct"] = values * 100.0 if values.max(skipna=True) <= 1.0 else values
    out["peak_final_gap_pp"] = out["peak_acc_pct"] - out["final_acc_pct"]
    out["policy"] = out.apply(policy_label, axis=1)
    return out


def normalize_construct(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["H"] = pd.to_numeric(out["H"], errors="coerce").astype("Int64")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["jackpot"] = pd.to_numeric(out["jackpot"], errors="coerce").astype("Int64")
    out["policy"] = out.apply(policy_label, axis=1)
    return out


def policy_label(row: pd.Series) -> str:
    arm = str(row.get("arm", ""))
    neutral_p = row.get("neutral_p", "")
    accept_policy = str(row.get("accept_policy", "")).lower()
    if arm.endswith("_STRICT") or accept_policy == "strict":
        return "strict"
    if "ZERO_P03" in arm or str(neutral_p) in {"0.3", "0.30"}:
        return "zero_p_0.3"
    if "ZERO_P10" in arm or str(neutral_p) in {"1", "1.0", "1.00"}:
        return "zero_p_1.0"
    return arm or accept_policy or "unknown"


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


def validate_artifacts(root: Path, results: pd.DataFrame, construct: pd.DataFrame, expected_seeds: int) -> dict:
    expected_runs = len(EXPECTED_ARMS) * expected_seeds
    failures = []
    if len(results) != expected_runs:
        failures.append(f"expected {expected_runs} result rows, got {len(results)}")
    if len(construct) != expected_runs:
        failures.append(f"expected {expected_runs} constructability rows, got {len(construct)}")
    missing_arms = sorted(set(EXPECTED_ARMS) - set(results["arm"]))
    if missing_arms and expected_seeds == 5:
        failures.append(f"missing D4 arms: {missing_arms}")

    candidates = list(root.glob("H_*/D4_*/seed_*/candidates.csv"))
    checkpoints = list(root.glob("H_*/D4_*/seed_*/final.ckpt"))
    panel_summaries = list(root.glob("H_*/D4_*/seed_*/panel_summary.json"))
    panel_timeseries = list(root.glob("H_*/D4_*/seed_*/panel_timeseries.csv"))
    for label, files in [
        ("candidate logs", candidates),
        ("checkpoints", checkpoints),
        ("panel summaries", panel_summaries),
        ("panel timeseries", panel_timeseries),
    ]:
        if len(files) != expected_runs:
            failures.append(f"expected {expected_runs} {label}, got {len(files)}")
    if not (root / "heartbeat.json").exists():
        failures.append("missing heartbeat.json")
    if not (root / "heartbeat.log").exists():
        failures.append("missing heartbeat.log")

    expected_candidate_rows = int(pd.to_numeric(results["expected_candidate_rows"], errors="coerce").sum())
    actual_candidate_rows = int(pd.to_numeric(construct["candidate_rows"], errors="coerce").sum())
    if expected_candidate_rows != actual_candidate_rows:
        failures.append(f"candidate rows mismatch expected={expected_candidate_rows} actual={actual_candidate_rows}")
    if failures:
        raise SystemExit(f"Phase D4 artifact validation failed: {failures}")
    return {
        "passed": True,
        "runs": int(len(results)),
        "candidate_rows": actual_candidate_rows,
        "checkpoints": len(checkpoints),
        "panel_summaries": len(panel_summaries),
        "panel_timeseries_files": len(panel_timeseries),
    }


def merge_stats(result_stats: pd.DataFrame, construct_stats: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "H", "jackpot", "policy", "C_K_window_ratio_mean", "cost_eval_ms_mean", "V_raw_mean",
        "M_pos_mean", "R_neg_mean", "accepted_positive_steps_mean", "accepted_nonpositive_steps_mean",
        "positive_delta_rows_mean",
    ]
    available = [col for col in keep if col in construct_stats.columns]
    return result_stats.merge(construct_stats[available], on=["H", "jackpot", "policy"], how="left")


def add_collapse_counts(stats: pd.DataFrame, raw: pd.DataFrame, threshold_pp: float) -> pd.DataFrame:
    rows = []
    for key, group in raw.groupby(["H", "jackpot", "policy"]):
        h, jackpot, policy = key
        peaks = pd.to_numeric(group["peak_acc_pct"], errors="coerce")
        rows.append({
            "H": h,
            "jackpot": jackpot,
            "policy": policy,
            "collapse_count": int((peaks < threshold_pp).sum()),
        })
    return stats.merge(pd.DataFrame(rows), on=["H", "jackpot", "policy"], how="left")


def classify(stats: pd.DataFrame, margin_pp: float, variance_penalty_pp: float) -> tuple[str, list[dict], list[str]]:
    notes = []
    h_rows = []
    clean_zero_winners = []
    unstable_zero_winners = []
    for h, group in stats.groupby("H"):
        strict = group[group["policy"] == "strict"]
        if strict.empty:
            notes.append(f"H={h}: strict row missing")
            continue
        strict = strict.iloc[0]
        strict_peak = float(strict["peak_acc_pct_mean"])
        strict_std = float(strict.get("peak_acc_pct_std", 0.0))
        strict_collapse = int(strict.get("collapse_count", 0))
        best_policy = "strict"
        best_delta = 0.0
        for _, row in group[group["policy"].astype(str).str.startswith("zero_p")].iterrows():
            delta = float(row["peak_acc_pct_mean"]) - strict_peak
            unstable = (
                int(row.get("collapse_count", 0)) > strict_collapse
                or float(row.get("peak_acc_pct_std", 0.0)) > strict_std + variance_penalty_pp
            )
            if delta > best_delta:
                best_delta = delta
                best_policy = str(row["policy"])
            if delta >= margin_pp:
                entry = {
                    "H": int(h),
                    "policy": str(row["policy"]),
                    "delta_vs_strict_pp": delta,
                    "unstable": unstable,
                }
                (unstable_zero_winners if unstable else clean_zero_winners).append(entry)
        h_rows.append({
            "H": int(h),
            "strict_peak_mean_pct": strict_peak,
            "best_policy": best_policy,
            "best_delta_vs_strict_pp": best_delta,
        })
    if clean_zero_winners:
        return "SAF_S_H_TABLE", h_rows, notes
    if unstable_zero_winners:
        notes.append("At least one zero_p arm beats strict by margin but increases collapse or variance.")
        return "SAF_UNSTABLE", h_rows, notes
    notes.append(f"No zero_p arm beats strict by >= {margin_pp:.2f}pp on any H.")
    return "SAF_STRICT_LOCK", h_rows, notes


def plot_peak(stats: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    policies = ["strict", "zero_p_0.3", "zero_p_1.0"]
    hs = sorted(int(x) for x in stats["H"].unique())
    x = np.arange(len(hs))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, policy in enumerate(policies):
        sub = stats[stats["policy"] == policy].set_index("H")
        means = [float(sub.loc[h, "peak_acc_pct_mean"]) if h in sub.index else np.nan for h in hs]
        errs = [float(sub.loc[h, "peak_acc_pct_std"]) if h in sub.index else 0.0 for h in hs]
        ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=3, label=policy)
    ax.set_title("Phase D4 locked-K softness sweep")
    ax.set_xlabel("H")
    ax.set_ylabel("peak accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hs])
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase_d4_softness_peak_by_h.png", dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(
    report: Path,
    root: Path,
    integrity: dict,
    verdict: str,
    h_rows: list[dict],
    notes: list[str],
    stats: pd.DataFrame,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    compact = stats[[
        "H", "jackpot", "policy", "n",
        "peak_acc_pct_mean", "peak_acc_pct_std", "final_acc_pct_mean",
        "peak_final_gap_pp_mean", "accept_rate_pct_mean", "alive_frac_mean_mean",
        "C_K_window_ratio_mean", "collapse_count",
    ]].copy()
    compact = compact.sort_values(["H", "policy"])
    lines = [
        "# Phase D4 Softness Verdict",
        "",
        f"Input root: `{root}`",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Artifact Integrity",
        "",
        f"- Runs: {integrity['runs']}",
        f"- Candidate rows: {integrity['candidate_rows']:,}",
        f"- Checkpoints: {integrity['checkpoints']}",
        f"- Panel summaries: {integrity['panel_summaries']}",
        f"- Panel timeseries files: {integrity['panel_timeseries_files']}",
        "",
        "## Decision Notes",
        "",
    ]
    lines.extend([f"- {note}" for note in notes] or ["- No additional notes."])
    lines.extend([
        "",
        "## H-Level Winners",
        "",
    ])
    lines.extend(markdown_table(pd.DataFrame(h_rows), ["H", "strict_peak_mean_pct", "best_policy", "best_delta_vs_strict_pp"]))
    lines.extend([
        "",
        "## Arm Statistics",
        "",
    ])
    lines.extend(markdown_table(compact, list(compact.columns)))
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `SAF_STRICT_LOCK` means SAF v1 can remain `SAF(K(H), tau=0, s=0)` for this substrate.",
        "- `SAF_S_H_TABLE` means softness is H-dependent and SAF v1 needs an `s(H)` table.",
        "- `SAF_UNSTABLE` means softness may improve mean peak but is not deployable without variance/collapse controls.",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.d4_root
    analysis_dir = root / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = normalize_results(read_csv_required(root / "results.csv"))
    construct = normalize_construct(read_csv_required(root / "constructability_summary.csv"))
    integrity = validate_artifacts(root, results, construct, args.expected_seeds)

    result_metrics = [
        "peak_acc_pct", "final_acc_pct", "peak_final_gap_pp", "accept_rate_pct",
        "alive_frac_mean", "wall_clock_s",
    ]
    construct_metrics = [
        "C_K_window_ratio", "cost_eval_ms", "V_raw", "M_pos", "R_neg",
        "accepted_positive_steps", "accepted_nonpositive_steps", "positive_delta_rows",
    ]
    result_stats = summarize(results, ["H", "jackpot", "policy"], result_metrics)
    result_stats = add_collapse_counts(result_stats, results, args.collapse_threshold_pp)
    construct_stats = summarize(construct, ["H", "jackpot", "policy"], construct_metrics)
    stats = merge_stats(result_stats, construct_stats)
    verdict, h_rows, notes = classify(stats, args.lock_margin_pp, args.variance_penalty_pp)

    plot_peak(stats, figures_dir)
    (analysis_dir / "phase_d4_softness_verdict.json").write_text(json.dumps({
        "verdict": verdict,
        "integrity": integrity,
        "h_level_winners": h_rows,
        "notes": notes,
    }, indent=2), encoding="utf-8")
    stats.to_csv(analysis_dir / "phase_d4_softness_stats.csv", index=False)
    write_report(args.report, root, integrity, verdict, h_rows, notes, stats)
    print(f"Verdict: {verdict}")
    print(f"Wrote: {analysis_dir / 'phase_d4_softness_verdict.json'}")
    print(f"Wrote: {analysis_dir / 'phase_d4_softness_stats.csv'}")
    print(f"Wrote: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
