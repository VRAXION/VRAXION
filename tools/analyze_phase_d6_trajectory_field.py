"""Phase D6: offline trajectory-field / pseudo-gradient audit.

This analyzer does not run new training. It asks whether existing
candidate/panel artifacts contain enough feature-space signal to justify a
learned proposal policy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_ROOTS = [
    Path("output/phase_b_full_20260424"),
    Path("output/phase_b1_horizon_ties_20260425"),
    Path("output/phase_d1_activation_20260425"),
    Path("output/phase_d2_cross_h_activation_20260426"),
    Path("output/phase_d3_klock_coarse_20260426"),
    Path("output/phase_d3_fine_k_20260426"),
    Path("output/phase_d4_softness_20260426"),
]
DEFAULT_OUT = Path("output/phase_d6_trajectory_field_20260427")
DEFAULT_REPORT = Path("docs/research/PHASE_D6_TRAJECTORY_FIELD_AUDIT.md")
FEATURES = [
    "edges",
    "unique_predictions",
    "collision_rate",
    "f_active",
    "stable_rank",
    "kernel_rank",
    "separation_sp",
    "accept_rate_window",
    "main_peak_acc",
    "panel_probe_acc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-runs", type=int, default=None, help="smoke-mode cap after sorting run rows")
    parser.add_argument("--top-bottom-lift-threshold", type=float, default=2.0)
    parser.add_argument("--spearman-threshold", type=float, default=0.40)
    parser.add_argument("--r2-threshold", type=float, default=0.25)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_results(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["source"] = source
    for col in ["H", "seed", "jackpot", "configured_steps", "horizon_steps"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for metric in ["peak_acc", "final_acc"]:
        if metric in out:
            values = pd.to_numeric(out[metric], errors="coerce")
            out[f"{metric}_pct"] = values * 100.0 if values.max(skipna=True) <= 1.0 else values
    return out


def load_all_roots(roots: list[Path], max_runs: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = []
    constructs = []
    operators = []
    for root in roots:
        if not root.exists():
            continue
        source = root.name
        res = normalize_results(read_csv(root / "results.csv"), source)
        con = read_csv(root / "constructability_summary.csv")
        op = read_csv(root / "constructability_operator_summary.csv")
        if not con.empty:
            con["source"] = source
        if not op.empty:
            op["source"] = source
        results.append(res)
        constructs.append(con)
        operators.append(op)
    if not results:
        raise SystemExit("no usable roots found")
    res_all = pd.concat(results, ignore_index=True)
    res_all = res_all.sort_values(["source", "H", "arm", "seed"], na_position="last")
    if max_runs is not None:
        keep = set(res_all.head(max_runs)["run_id"])
        res_all = res_all[res_all["run_id"].isin(keep)].copy()
    keep_ids = set(res_all["run_id"])
    con_all = pd.concat(constructs, ignore_index=True) if constructs else pd.DataFrame()
    op_all = pd.concat(operators, ignore_index=True) if operators else pd.DataFrame()
    if not con_all.empty and "run_id" in con_all:
        con_all = con_all[con_all["run_id"].isin(keep_ids)].copy()
    if not op_all.empty and "run_id" in op_all:
        op_all = op_all[op_all["run_id"].isin(keep_ids)].copy()
    return res_all, con_all, op_all


def load_panel_windows(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, run in results.iterrows():
        path = Path(str(run.get("panel_timeseries", "")))
        if not path.exists():
            continue
        panel = pd.read_csv(path)
        if panel.empty:
            continue
        panel = panel.copy()
        for col in ["run_id", "source", "arm", "fixture", "phase"]:
            panel[col] = run.get(col, "")
        for col in ["H", "seed", "jackpot", "horizon_steps"]:
            panel[col] = run.get(col, np.nan)
        panel["accept_policy"] = run.get("accept_policy", "")
        panel["neutral_p"] = run.get("neutral_p", "")
        panel["peak_acc_pct_final_run"] = run.get("peak_acc_pct", np.nan)
        panel["final_acc_pct_final_run"] = run.get("final_acc_pct", np.nan)
        panel["panel_index"] = np.arange(len(panel))
        for feature in FEATURES:
            if feature not in panel:
                panel[feature] = np.nan
        for feature in FEATURES:
            panel[f"delta_{feature}"] = panel[feature].diff()
        panel["delta_peak_window"] = panel["main_peak_acc"].diff()
        rows.append(panel)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def zscore_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    clean_cols = [c for c in columns if c in df and pd.to_numeric(df[c], errors="coerce").notna().sum() > 2]
    x = df[clean_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(med, inds[1])
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0.0] = 1.0
    return (x - mu) / sd, clean_cols


def ridge_loocv(df: pd.DataFrame, feature_cols: list[str], target_col: str, alpha: float) -> dict:
    work = df.dropna(subset=[target_col]).copy()
    if len(work) < 6:
        return {"valid": False, "reason": "not_enough_rows", "n": len(work)}
    x, used = zscore_matrix(work, feature_cols)
    y = pd.to_numeric(work[target_col], errors="coerce").to_numpy(dtype=float)
    y_mean = float(np.mean(y))
    preds = []
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        xt = x[mask]
        yt = y[mask]
        xt_aug = np.c_[np.ones(len(xt)), xt]
        xi_aug = np.r_[1.0, x[i]]
        reg = np.eye(xt_aug.shape[1]) * alpha
        reg[0, 0] = 0.0
        beta = np.linalg.pinv(xt_aug.T @ xt_aug + reg) @ xt_aug.T @ yt
        preds.append(float(xi_aug @ beta))
    preds_arr = np.array(preds)
    sse = float(np.sum((y - preds_arr) ** 2))
    sst = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    spearman = float(stats.spearmanr(y, preds_arr, nan_policy="omit").correlation)
    pearson = float(stats.pearsonr(y, preds_arr).statistic) if len(y) > 2 else float("nan")
    return {
        "valid": True,
        "n": int(len(y)),
        "features": used,
        "r2_loocv": r2,
        "spearman": spearman,
        "pearson": pearson,
    }


def ridge_group_cv(df: pd.DataFrame, feature_cols: list[str], target_col: str, group_col: str, alpha: float) -> dict:
    work = df.dropna(subset=[target_col, group_col]).copy()
    groups = sorted(work[group_col].dropna().unique().tolist())
    if len(work) < 8 or len(groups) < 3:
        return {"valid": False, "reason": "not_enough_groups", "n": len(work), "groups": len(groups)}
    x, used = zscore_matrix(work, feature_cols)
    y = pd.to_numeric(work[target_col], errors="coerce").to_numpy(dtype=float)
    group_vals = work[group_col].to_numpy()
    preds = np.full(len(y), np.nan)
    for group in groups:
        test = group_vals == group
        train = ~test
        if train.sum() < 4 or test.sum() == 0:
            continue
        xt = x[train]
        yt = y[train]
        xt_aug = np.c_[np.ones(len(xt)), xt]
        x_test_aug = np.c_[np.ones(test.sum()), x[test]]
        reg = np.eye(xt_aug.shape[1]) * alpha
        reg[0, 0] = 0.0
        beta = np.linalg.pinv(xt_aug.T @ xt_aug + reg) @ xt_aug.T @ yt
        preds[test] = x_test_aug @ beta
    ok = np.isfinite(preds)
    if ok.sum() < 4:
        return {"valid": False, "reason": "no_valid_predictions", "n": int(ok.sum()), "groups": len(groups)}
    y_ok = y[ok]
    p_ok = preds[ok]
    y_mean = float(np.mean(y_ok))
    sse = float(np.sum((y_ok - p_ok) ** 2))
    sst = float(np.sum((y_ok - y_mean) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return {
        "valid": True,
        "n": int(ok.sum()),
        "groups": int(len(groups)),
        "group_col": group_col,
        "features": used,
        "r2_group_cv": r2,
        "spearman_group_cv": float(stats.spearmanr(y_ok, p_ok, nan_policy="omit").correlation),
        "pearson_group_cv": float(stats.pearsonr(y_ok, p_ok).statistic) if len(y_ok) > 2 else float("nan"),
    }


def early_feature_prediction(panel: pd.DataFrame, alpha: float) -> tuple[dict, pd.DataFrame]:
    if panel.empty:
        return {"valid": False, "reason": "no_panel_rows"}, pd.DataFrame()
    early = panel.sort_values(["run_id", "step"]).groupby("run_id", as_index=False).first()
    features = [c for c in FEATURES if c in early]
    no_score_features = [c for c in features if c not in {"main_peak_acc", "panel_probe_acc"}]
    metrics = []
    for feature in features:
        x = pd.to_numeric(early[feature], errors="coerce")
        y = pd.to_numeric(early["peak_acc_pct_final_run"], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() > 3:
            metrics.append({
                "feature": feature,
                "pearson": float(stats.pearsonr(x[ok], y[ok]).statistic),
                "spearman": float(stats.spearmanr(x[ok], y[ok]).correlation),
            })
    model = {
        "loocv_all_features": ridge_loocv(early, features, "peak_acc_pct_final_run", alpha),
        "seed_group_cv_all_features": ridge_group_cv(early, features, "peak_acc_pct_final_run", "seed", alpha),
        "loocv_no_score_features": ridge_loocv(early, no_score_features, "peak_acc_pct_final_run", alpha),
        "seed_group_cv_no_score_features": ridge_group_cv(early, no_score_features, "peak_acc_pct_final_run", "seed", alpha),
    }
    model["valid"] = any(v.get("valid") for v in model.values() if isinstance(v, dict))
    return model, pd.DataFrame(metrics).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def trajectory_alignment(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    delta_cols = [f"delta_{c}" for c in FEATURES if f"delta_{c}" in panel]
    if panel.empty or not delta_cols:
        return {"valid": False, "reason": "no_delta_rows"}, pd.DataFrame()
    rows = []
    run_vectors = []
    for run_id, group in panel.groupby("run_id"):
        final_peak = float(group["peak_acc_pct_final_run"].iloc[0])
        total = []
        for col in delta_cols:
            total.append(pd.to_numeric(group[col], errors="coerce").fillna(0.0).sum())
        vec = np.array(total, dtype=float)
        norm = float(np.linalg.norm(vec))
        rows.append({
            "run_id": run_id,
            "source": group["source"].iloc[0],
            "H": group["H"].iloc[0],
            "arm": group["arm"].iloc[0],
            "seed": group["seed"].iloc[0],
            "peak_acc_pct": final_peak,
            "delta_norm": norm,
            **{col: vec[i] for i, col in enumerate(delta_cols)},
        })
        if norm > 0:
            run_vectors.append((run_id, final_peak, vec / norm))
    df = pd.DataFrame(rows)
    if len(run_vectors) < 4:
        return {"valid": False, "reason": "not_enough_vectors", "n": len(run_vectors)}, df
    peaks = np.array([v[1] for v in run_vectors])
    threshold = np.nanmedian(peaks)
    success = [v for v in run_vectors if v[1] >= threshold]
    fail = [v for v in run_vectors if v[1] < threshold]

    def mean_pair_cos(items: list[tuple[str, float, np.ndarray]]) -> float:
        vals = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                vals.append(float(np.dot(items[i][2], items[j][2])))
        return float(np.mean(vals)) if vals else 0.0

    success_cos = mean_pair_cos(success)
    fail_cos = mean_pair_cos(fail)
    shuffled = []
    rng = np.random.default_rng(42)
    labels = np.array([v[1] >= threshold for v in run_vectors])
    vectors = [v[2] for v in run_vectors]
    for _ in range(200):
        rng.shuffle(labels)
        items = [(str(i), 0.0, vectors[i]) for i, ok in enumerate(labels) if ok]
        shuffled.append(mean_pair_cos(items))
    shuffled_mean = float(np.mean(shuffled))
    return {
        "valid": True,
        "n_vectors": len(run_vectors),
        "success_threshold_peak_pct": float(threshold),
        "success_pairwise_cosine": success_cos,
        "failure_pairwise_cosine": fail_cos,
        "shuffle_success_cosine_mean": shuffled_mean,
        "success_lift_vs_shuffle": success_cos - shuffled_mean,
    }, df


def operator_field(operators: pd.DataFrame, lift_threshold: float) -> tuple[dict, pd.DataFrame]:
    if operators.empty:
        return {"valid": False, "reason": "no_operator_summary"}, pd.DataFrame()
    work = operators.copy()
    for col in ["H", "V_raw", "M_pos", "R_neg", "candidate_rows"]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["usefulness"] = work["V_raw"] * work["M_pos"]
    grouped = (
        work.groupby(["H", "operator_id"], dropna=False)
        .agg(
            n=("run_id", "nunique"),
            usefulness_mean=("usefulness", "mean"),
            usefulness_std=("usefulness", "std"),
            V_raw_mean=("V_raw", "mean"),
            M_pos_mean=("M_pos", "mean"),
            R_neg_mean=("R_neg", "mean"),
            candidate_rows_sum=("candidate_rows", "sum"),
        )
        .reset_index()
    )
    lifts = []
    for h, group in grouped.groupby("H"):
        ordered = group.sort_values("usefulness_mean", ascending=False)
        top = ordered.head(3)["usefulness_mean"].mean()
        bottom = ordered.tail(3)["usefulness_mean"].replace(0.0, np.nan).mean()
        lift = float(top / bottom) if bottom and not np.isnan(bottom) else float("inf")
        lifts.append({"H": int(h), "top3_bottom3_lift": lift, "passes": lift >= lift_threshold})
    pass_count = sum(1 for row in lifts if row["passes"])
    return {
        "valid": True,
        "n_rows": int(len(work)),
        "lift_threshold": lift_threshold,
        "h_lifts": lifts,
        "pass_h_count": pass_count,
    }, grouped.sort_values(["H", "usefulness_mean"], ascending=[True, False])


def plot_outputs(out_dir: Path, early_metrics: pd.DataFrame, op_summary: pd.DataFrame, align_df: pd.DataFrame) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not early_metrics.empty:
        top = early_metrics.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.barh(top["feature"], top["spearman"])
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title("Early panel feature Spearman vs final peak")
        fig.tight_layout()
        fig.savefig(fig_dir / "early_feature_spearman.png", dpi=160)
        plt.close(fig)
    if not op_summary.empty:
        pivot = op_summary.pivot(index="operator_id", columns="H", values="usefulness_mean").fillna(0.0)
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Operator usefulness V_raw * M_pos")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / "operator_usefulness_heatmap.png", dpi=160)
        plt.close(fig)
    if not align_df.empty and "delta_norm" in align_df:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(align_df["delta_norm"], align_df["peak_acc_pct"], alpha=0.75)
        ax.set_xlabel("trajectory delta norm")
        ax.set_ylabel("final run peak accuracy (%)")
        ax.set_title("Trajectory movement vs final peak")
        fig.tight_layout()
        fig.savefig(fig_dir / "trajectory_delta_norm_vs_peak.png", dpi=160)
        plt.close(fig)


def decide_verdict(model: dict, alignment: dict, op_signal: dict, args: argparse.Namespace) -> tuple[str, list[str]]:
    notes = []
    feature_signal = False
    if model.get("valid"):
        grouped = model.get("seed_group_cv_all_features", {})
        grouped_no_score = model.get("seed_group_cv_no_score_features", {})
        r2 = grouped.get("r2_group_cv", float("nan"))
        sp = grouped.get("spearman_group_cv", float("nan"))
        r2_ns = grouped_no_score.get("r2_group_cv", float("nan"))
        sp_ns = grouped_no_score.get("spearman_group_cv", float("nan"))
        feature_signal = (
            (not math.isnan(r2) and r2 >= args.r2_threshold)
            or (not math.isnan(sp) and sp >= args.spearman_threshold)
            or (not math.isnan(r2_ns) and r2_ns >= args.r2_threshold)
            or (not math.isnan(sp_ns) and sp_ns >= args.spearman_threshold)
        )
        notes.append(
            f"early feature model seed-held-out: R2={r2:.3f}, Spearman={sp:.3f}; "
            f"no-score R2={r2_ns:.3f}, no-score Spearman={sp_ns:.3f}"
        )
    else:
        notes.append(f"early feature model invalid: {model.get('reason')}")
    align_signal = False
    if alignment.get("valid"):
        lift = alignment.get("success_lift_vs_shuffle", 0.0)
        cos = alignment.get("success_pairwise_cosine", 0.0)
        align_signal = cos > 0.0 and lift > 0.02
        notes.append(f"trajectory alignment: success_cos={cos:.3f}, lift_vs_shuffle={lift:.3f}")
    else:
        notes.append(f"trajectory alignment invalid: {alignment.get('reason')}")
    op_signal_ok = False
    if op_signal.get("valid"):
        op_signal_ok = op_signal.get("pass_h_count", 0) >= 2
        notes.append(f"operator lift passes H-count={op_signal.get('pass_h_count', 0)}")
    else:
        notes.append(f"operator signal invalid: {op_signal.get('reason')}")
    if feature_signal or align_signal:
        return "D7_FEATURE_POLICY", notes
    if op_signal_ok:
        return "D7_OPERATOR_BANDIT", notes
    if model.get("valid") or alignment.get("valid") or op_signal.get("valid"):
        return "D6_NO_SIGNAL_INSTRUMENT_FIRST", notes
    return "D6_INCONCLUSIVE", notes


def write_report(path: Path, summary: dict, early_metrics: pd.DataFrame, op_summary: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase D6 Trajectory Field Audit",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "## Artifact Coverage",
        "",
        f"- Roots used: {summary['coverage']['roots_used']}",
        f"- Runs: {summary['coverage']['runs']}",
        f"- Candidate rows represented by constructability summaries: {summary['coverage']['candidate_rows']:,}",
        f"- Panel rows: {summary['coverage']['panel_rows']}",
        f"- Operator rows: {summary['coverage']['operator_rows']}",
        "",
        "## Decision Signals",
        "",
    ]
    lines.extend([f"- {note}" for note in summary["decision_notes"]])
    lines.extend([
        "",
        "## Early Feature Model",
        "",
        "```json",
        json.dumps(summary["early_feature_model"], indent=2),
        "```",
        "",
        "The decision uses seed-held-out validation. The no-score variant excludes `main_peak_acc` and `panel_probe_acc`.",
        "",
        "Top early feature correlations:",
        "",
        "| feature | Spearman | Pearson |",
        "|---|---:|---:|",
    ])
    for _, row in early_metrics.head(8).iterrows():
        lines.append(f"| {row['feature']} | {row['spearman']:.3f} | {row['pearson']:.3f} |")
    lines.extend([
        "",
        "## Trajectory Alignment",
        "",
        "```json",
        json.dumps(summary["trajectory_alignment"], indent=2),
        "```",
        "",
        "## Operator Field",
        "",
        "```json",
        json.dumps(summary["operator_signal"], indent=2),
        "```",
        "",
        "Top operator usefulness by H:",
        "",
        "| H | operator | usefulness | V_raw | M_pos | R_neg |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    if not op_summary.empty:
        for h, group in op_summary.groupby("H"):
            for _, row in group.head(5).iterrows():
                lines.append(
                    f"| {int(h)} | {row['operator_id']} | {row['usefulness_mean']:.4g} | "
                    f"{row['V_raw_mean']:.4g} | {row['M_pos_mean']:.4g} | {row['R_neg_mean']:.4g} |"
                )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This is not a raw graph-space gradient claim.",
        "- A feature-policy verdict means a learned proposal conditioned on panel-state is worth testing.",
        "- An operator-bandit verdict means adaptive operator weighting is the safer D7 first step.",
        "- A no-signal verdict means richer mutation-target instrumentation should precede adaptive search.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    analysis_dir = args.out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results, constructs, operators = load_all_roots(args.roots, args.max_runs)
    panel = load_panel_windows(results)

    early_model, early_metrics = early_feature_prediction(panel, args.ridge_alpha)
    alignment, align_df = trajectory_alignment(panel)
    op_signal, op_summary = operator_field(operators, args.top_bottom_lift_threshold)
    verdict, notes = decide_verdict(early_model, alignment, op_signal, args)

    candidate_rows = int(pd.to_numeric(constructs.get("candidate_rows", pd.Series(dtype=float)), errors="coerce").sum())
    roots_used = sorted(results["source"].dropna().unique().tolist())
    summary = {
        "verdict": verdict,
        "coverage": {
            "roots_used": roots_used,
            "runs": int(len(results)),
            "candidate_rows": candidate_rows,
            "panel_rows": int(len(panel)),
            "operator_rows": int(len(operators)),
        },
        "early_feature_model": early_model,
        "trajectory_alignment": alignment,
        "operator_signal": op_signal,
        "decision_notes": notes,
    }

    panel.to_csv(analysis_dir / "trajectory_panel_windows.csv", index=False)
    op_summary.to_csv(analysis_dir / "operator_field_summary.csv", index=False)
    align_df.to_csv(analysis_dir / "attractor_alignment.csv", index=False)
    early_metrics.to_csv(analysis_dir / "early_feature_correlations.csv", index=False)
    (analysis_dir / "trajectory_field_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_outputs(analysis_dir, early_metrics, op_summary, align_df)
    write_report(args.report, summary, early_metrics, op_summary)
    print(f"Verdict: {verdict}")
    print(f"Runs: {len(results)}  candidate_rows: {candidate_rows:,}  panel_rows: {len(panel)}")
    print(f"Wrote: {analysis_dir / 'trajectory_field_summary.json'}")
    print(f"Wrote: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
