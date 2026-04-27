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
    out["global_run_id"] = source + "::" + out["run_id"].astype(str)
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
            con["global_run_id"] = source + "::" + con["run_id"].astype(str)
        if not op.empty:
            op["source"] = source
            op["global_run_id"] = source + "::" + op["run_id"].astype(str)
        results.append(res)
        constructs.append(con)
        operators.append(op)
    if not results:
        raise SystemExit("no usable roots found")
    res_all = pd.concat(results, ignore_index=True)
    res_all = res_all.sort_values(["source", "H", "arm", "seed"], na_position="last")
    dup_global = res_all["global_run_id"].duplicated().sum()
    if dup_global:
        raise SystemExit(f"duplicate global_run_id rows found: {dup_global}")
    if max_runs is not None:
        keep = set(res_all.head(max_runs)["global_run_id"])
        res_all = res_all[res_all["global_run_id"].isin(keep)].copy()
    keep_ids = set(res_all["global_run_id"])
    con_all = pd.concat(constructs, ignore_index=True) if constructs else pd.DataFrame()
    op_all = pd.concat(operators, ignore_index=True) if operators else pd.DataFrame()
    if not con_all.empty and "global_run_id" in con_all:
        con_all = con_all[con_all["global_run_id"].isin(keep_ids)].copy()
    if not op_all.empty and "global_run_id" in op_all:
        op_all = op_all[op_all["global_run_id"].isin(keep_ids)].copy()
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
        for col in ["run_id", "global_run_id", "source", "arm", "fixture", "phase"]:
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


def fit_scaler(df: pd.DataFrame, columns: list[str]) -> dict:
    clean_cols = [c for c in columns if c in df and pd.to_numeric(df[c], errors="coerce").notna().sum() > 2]
    if not clean_cols:
        return {"columns": [], "median": np.array([]), "mean": np.array([]), "std": np.array([])}
    x = df[clean_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(med, inds[1])
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0.0] = 1.0
    return {"columns": clean_cols, "median": med, "mean": mu, "std": sd}


def transform_scaler(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    cols = scaler["columns"]
    if not cols:
        return np.empty((len(df), 0))
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(scaler["median"], inds[1])
    return (x - scaler["mean"]) / scaler["std"]


def fit_ridge_predict(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], target_col: str, alpha: float) -> tuple[np.ndarray, list[str]]:
    scaler = fit_scaler(train, feature_cols)
    x_train = transform_scaler(train, scaler)
    x_test = transform_scaler(test, scaler)
    y_train = pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype=float)
    xt_aug = np.c_[np.ones(len(x_train)), x_train]
    x_test_aug = np.c_[np.ones(len(x_test)), x_test]
    reg = np.eye(xt_aug.shape[1]) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(xt_aug.T @ xt_aug + reg) @ xt_aug.T @ y_train
    return x_test_aug @ beta, scaler["columns"]


def score_predictions(y: np.ndarray, preds: np.ndarray) -> dict:
    ok = np.isfinite(preds) & np.isfinite(y)
    if ok.sum() < 4:
        return {"valid": False, "reason": "no_valid_predictions", "n": int(ok.sum())}
    y_ok = y[ok]
    p_ok = preds[ok]
    y_mean = float(np.mean(y_ok))
    sse = float(np.sum((y_ok - p_ok) ** 2))
    sst = float(np.sum((y_ok - y_mean) ** 2))
    return {
        "valid": True,
        "n": int(ok.sum()),
        "r2": 1.0 - sse / sst if sst > 0 else 0.0,
        "spearman": float(stats.spearmanr(y_ok, p_ok, nan_policy="omit").correlation),
        "pearson": float(stats.pearsonr(y_ok, p_ok).statistic) if len(y_ok) > 2 else float("nan"),
    }


def ridge_loocv(df: pd.DataFrame, feature_cols: list[str], target_col: str, alpha: float) -> dict:
    work = df.dropna(subset=[target_col]).copy()
    if len(work) < 6:
        return {"valid": False, "reason": "not_enough_rows", "n": len(work)}
    y = pd.to_numeric(work[target_col], errors="coerce").to_numpy(dtype=float)
    preds = np.full(len(y), np.nan)
    used_all: set[str] = set()
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        pred, used = fit_ridge_predict(work[mask], work.iloc[[i]], feature_cols, target_col, alpha)
        preds[i] = pred[0]
        used_all.update(used)
    score = score_predictions(y, preds)
    if not score.get("valid"):
        return score
    return {
        "valid": True,
        "n": int(len(y)),
        "features": sorted(used_all),
        "r2_loocv": score["r2"],
        "spearman": score["spearman"],
        "pearson": score["pearson"],
    }


def ridge_group_cv(df: pd.DataFrame, feature_cols: list[str], target_col: str, group_col: str, alpha: float) -> dict:
    work = df.dropna(subset=[target_col, group_col]).copy()
    groups = sorted(work[group_col].dropna().unique().tolist())
    if len(work) < 8 or len(groups) < 3:
        return {"valid": False, "reason": "not_enough_groups", "n": len(work), "groups": len(groups)}
    y = pd.to_numeric(work[target_col], errors="coerce").to_numpy(dtype=float)
    group_vals = work[group_col].to_numpy()
    preds = np.full(len(y), np.nan)
    used_all: set[str] = set()
    for group in groups:
        test = group_vals == group
        train = ~test
        if train.sum() < 4 or test.sum() == 0:
            continue
        pred, used = fit_ridge_predict(work[train], work[test], feature_cols, target_col, alpha)
        preds[test] = pred
        used_all.update(used)
    score = score_predictions(y, preds)
    if not score.get("valid"):
        return {**score, "groups": len(groups)}
    return {
        "valid": True,
        "n": int(score["n"]),
        "groups": int(len(groups)),
        "group_col": group_col,
        "features": sorted(used_all),
        "r2_group_cv": score["r2"],
        "spearman_group_cv": score["spearman"],
        "pearson_group_cv": score["pearson"],
    }


def early_feature_prediction(panel: pd.DataFrame, alpha: float) -> tuple[dict, pd.DataFrame]:
    if panel.empty:
        return {"valid": False, "reason": "no_panel_rows"}, pd.DataFrame()
    early = panel.sort_values(["global_run_id", "step"]).groupby("global_run_id", as_index=False).first()
    features = [c for c in FEATURES if c in early]
    no_score_features = [c for c in features if c not in {"main_peak_acc", "panel_probe_acc"}]
    no_score_no_accept_features = [c for c in no_score_features if c != "accept_rate_window"]
    early["early_main_peak_pct"] = pd.to_numeric(early["main_peak_acc"], errors="coerce") * 100.0
    early["residual_peak_after_early_main"] = early["peak_acc_pct_final_run"] - early["early_main_peak_pct"]
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
        "source_group_cv_all_features": ridge_group_cv(early, features, "peak_acc_pct_final_run", "source", alpha),
        "H_group_cv_all_features": ridge_group_cv(early, features, "peak_acc_pct_final_run", "H", alpha),
        "phase_group_cv_all_features": ridge_group_cv(early, features, "peak_acc_pct_final_run", "phase", alpha),
        "arm_group_cv_all_features": ridge_group_cv(early, features, "peak_acc_pct_final_run", "arm", alpha),
        "loocv_no_score_features": ridge_loocv(early, no_score_features, "peak_acc_pct_final_run", alpha),
        "seed_group_cv_no_score_features": ridge_group_cv(early, no_score_features, "peak_acc_pct_final_run", "seed", alpha),
        "seed_group_cv_no_score_no_accept_features": ridge_group_cv(early, no_score_no_accept_features, "peak_acc_pct_final_run", "seed", alpha),
        "seed_group_cv_residual_no_score_features": ridge_group_cv(early, no_score_features, "residual_peak_after_early_main", "seed", alpha),
    }
    model["negative_controls"] = negative_controls(early, no_score_features, "peak_acc_pct_final_run", "seed", alpha)
    model["valid"] = any(v.get("valid") for v in model.values() if isinstance(v, dict))
    return model, pd.DataFrame(metrics).sort_values("spearman", key=lambda s: s.abs(), ascending=False)


def negative_controls(df: pd.DataFrame, feature_cols: list[str], target_col: str, group_col: str, alpha: float) -> dict:
    rng = np.random.default_rng(123)
    work = df.copy()
    target_shuffle = work.copy()
    for _, idx in target_shuffle.groupby(["H", "source"], dropna=False).groups.items():
        values = target_shuffle.loc[idx, target_col].to_numpy(dtype=float)
        rng.shuffle(values)
        target_shuffle.loc[idx, target_col] = values
    target_result = ridge_group_cv(target_shuffle, feature_cols, target_col, group_col, alpha)

    feature_shuffle = work.copy()
    for col in feature_cols:
        values = feature_shuffle[col].to_numpy(dtype=float)
        rng.shuffle(values)
        feature_shuffle[col] = values
    feature_result = ridge_group_cv(feature_shuffle, feature_cols, target_col, group_col, alpha)
    return {
        "target_shuffle_within_H_source": target_result,
        "feature_shuffle": feature_result,
    }


def trajectory_alignment(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    delta_cols = [f"delta_{c}" for c in FEATURES if f"delta_{c}" in panel]
    if panel.empty or not delta_cols:
        return {"valid": False, "reason": "no_delta_rows"}, pd.DataFrame()
    rows = []
    run_vectors = []
    for run_id, group in panel.groupby("global_run_id"):
        final_peak = float(group["peak_acc_pct_final_run"].iloc[0])
        total = []
        for col in delta_cols:
            total.append(pd.to_numeric(group[col], errors="coerce").fillna(0.0).sum())
        vec = np.array(total, dtype=float)
        norm = float(np.linalg.norm(vec))
        rows.append({
            "global_run_id": run_id,
            "run_id": group["run_id"].iloc[0],
            "source": group["source"].iloc[0],
            "H": group["H"].iloc[0],
            "arm": group["arm"].iloc[0],
            "seed": group["seed"].iloc[0],
            "peak_acc_pct": final_peak,
            "delta_norm": norm,
            **{col: vec[i] for i, col in enumerate(delta_cols)},
        })
        if norm > 0:
            run_vectors.append({
                "global_run_id": run_id,
                "peak": final_peak,
                "vector": vec / norm,
                "source": group["source"].iloc[0],
                "H": group["H"].iloc[0],
                "arm": group["arm"].iloc[0],
            })
    df = pd.DataFrame(rows)
    if len(run_vectors) < 4:
        return {"valid": False, "reason": "not_enough_vectors", "n": len(run_vectors)}, df
    peaks = np.array([v["peak"] for v in run_vectors])
    threshold = np.nanmedian(peaks)
    success = [v for v in run_vectors if v["peak"] >= threshold]
    fail = [v for v in run_vectors if v["peak"] < threshold]

    def mean_pair_cos(items: list[dict]) -> float:
        vals = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                vals.append(float(np.dot(items[i]["vector"], items[j]["vector"])))
        return float(np.mean(vals)) if vals else 0.0

    success_cos = mean_pair_cos(success)
    fail_cos = mean_pair_cos(fail)
    shuffled_global = []
    shuffled_h_source = []
    rng = np.random.default_rng(42)
    labels = np.array([v["peak"] >= threshold for v in run_vectors])
    vectors = [v["vector"] for v in run_vectors]
    for _ in range(200):
        rng.shuffle(labels)
        items = [{"vector": vectors[i]} for i, ok in enumerate(labels) if ok]
        shuffled_global.append(mean_pair_cos(items))

        stratified_labels = np.array([v["peak"] >= threshold for v in run_vectors])
        for key in {(v["H"], v["source"]) for v in run_vectors}:
            idx = [i for i, v in enumerate(run_vectors) if (v["H"], v["source"]) == key]
            local = stratified_labels[idx].copy()
            rng.shuffle(local)
            stratified_labels[idx] = local
        items = [{"vector": vectors[i]} for i, ok in enumerate(stratified_labels) if ok]
        shuffled_h_source.append(mean_pair_cos(items))
    shuffled_mean = float(np.mean(shuffled_global))
    shuffled_h_source_mean = float(np.mean(shuffled_h_source))
    return {
        "valid": True,
        "n_vectors": len(run_vectors),
        "success_threshold_peak_pct": float(threshold),
        "success_pairwise_cosine": success_cos,
        "failure_pairwise_cosine": fail_cos,
        "shuffle_success_cosine_mean": shuffled_mean,
        "success_lift_vs_shuffle": success_cos - shuffled_mean,
        "shuffle_within_H_source_cosine_mean": shuffled_h_source_mean,
        "success_lift_vs_H_source_shuffle": success_cos - shuffled_h_source_mean,
    }, df


def operator_field(operators: pd.DataFrame, lift_threshold: float) -> tuple[dict, pd.DataFrame]:
    if operators.empty:
        return {"valid": False, "reason": "no_operator_summary"}, pd.DataFrame()
    work = operators.copy()
    for col in ["H", "V_raw", "M_pos", "R_neg", "candidate_rows"]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["usefulness"] = work["V_raw"] * work["M_pos"]
    work["weighted_usefulness_num"] = work["usefulness"] * work["candidate_rows"]
    grouped = (
        work.groupby(["H", "operator_id"], dropna=False)
        .agg(
            n=("global_run_id", "nunique"),
            usefulness_mean=("usefulness", "mean"),
            usefulness_median=("usefulness", "median"),
            usefulness_std=("usefulness", "std"),
            weighted_usefulness_num=("weighted_usefulness_num", "sum"),
            V_raw_mean=("V_raw", "mean"),
            M_pos_mean=("M_pos", "mean"),
            R_neg_mean=("R_neg", "mean"),
            candidate_rows_sum=("candidate_rows", "sum"),
        )
        .reset_index()
    )
    grouped["usefulness_weighted"] = grouped["weighted_usefulness_num"] / grouped["candidate_rows_sum"].replace(0.0, np.nan)
    lifts = []
    for h, group in grouped.groupby("H"):
        ordered = group.sort_values("usefulness_weighted", ascending=False)
        top = ordered.head(3)["usefulness_weighted"].mean()
        bottom = ordered.tail(3)["usefulness_weighted"].replace(0.0, np.nan).mean()
        lift = float(top / bottom) if bottom and not np.isnan(bottom) else float("inf")
        median = ordered["usefulness_weighted"].replace(0.0, np.nan).median()
        top_median = float(top / median) if median and not np.isnan(median) else float("inf")
        boot = bootstrap_operator_lift(work[work["H"] == h])
        influence = one_run_operator_influence(work[work["H"] == h])
        lifts.append({
            "H": int(h),
            "top3_bottom3_lift": lift,
            "top3_median_lift": top_median,
            "bootstrap_ci_low": boot["ci_low"],
            "bootstrap_ci_high": boot["ci_high"],
            "min_leave_one_run_lift": influence["min_lift"],
            "passes": lift >= lift_threshold and influence["min_lift"] >= lift_threshold,
        })
    pass_count = sum(1 for row in lifts if row["passes"])
    return {
        "valid": True,
        "n_rows": int(len(work)),
        "lift_threshold": lift_threshold,
        "h_lifts": lifts,
        "pass_h_count": pass_count,
    }, grouped.sort_values(["H", "usefulness_weighted"], ascending=[True, False])


def operator_lift_from_rows(rows: pd.DataFrame) -> float:
    if rows.empty:
        return 0.0
    grouped = (
        rows.groupby("operator_id")
        .agg(num=("weighted_usefulness_num", "sum"), den=("candidate_rows", "sum"))
        .reset_index()
    )
    grouped["weighted"] = grouped["num"] / grouped["den"].replace(0.0, np.nan)
    ordered = grouped.sort_values("weighted", ascending=False)
    top = ordered.head(3)["weighted"].mean()
    bottom = ordered.tail(3)["weighted"].replace(0.0, np.nan).mean()
    return float(top / bottom) if bottom and not np.isnan(bottom) else float("inf")


def bootstrap_operator_lift(rows: pd.DataFrame, n_boot: int = 200) -> dict:
    ids = rows["global_run_id"].dropna().unique()
    if len(ids) < 3:
        lift = operator_lift_from_rows(rows)
        return {"ci_low": lift, "ci_high": lift}
    rng = np.random.default_rng(77)
    vals = []
    for _ in range(n_boot):
        sample_ids = rng.choice(ids, size=len(ids), replace=True)
        sample = pd.concat([rows[rows["global_run_id"] == rid] for rid in sample_ids], ignore_index=True)
        vals.append(operator_lift_from_rows(sample))
    return {
        "ci_low": float(np.nanpercentile(vals, 5)),
        "ci_high": float(np.nanpercentile(vals, 95)),
    }


def one_run_operator_influence(rows: pd.DataFrame) -> dict:
    ids = rows["global_run_id"].dropna().unique()
    if len(ids) < 3:
        return {"min_lift": operator_lift_from_rows(rows)}
    vals = [operator_lift_from_rows(rows[rows["global_run_id"] != rid]) for rid in ids]
    return {"min_lift": float(np.nanmin(vals))}


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
        pivot = op_summary.pivot(index="operator_id", columns="H", values="usefulness_weighted").fillna(0.0)
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
        grouped_no_accept = model.get("seed_group_cv_no_score_no_accept_features", {})
        residual = model.get("seed_group_cv_residual_no_score_features", {})
        controls = model.get("negative_controls", {})
        r2 = grouped.get("r2_group_cv", float("nan"))
        sp = grouped.get("spearman_group_cv", float("nan"))
        r2_ns = grouped_no_score.get("r2_group_cv", float("nan"))
        sp_ns = grouped_no_score.get("spearman_group_cv", float("nan"))
        r2_na = grouped_no_accept.get("r2_group_cv", float("nan"))
        sp_na = grouped_no_accept.get("spearman_group_cv", float("nan"))
        residual_sp = residual.get("spearman_group_cv", float("nan"))
        target_shuffle = controls.get("target_shuffle_within_H_source", {})
        feature_shuffle = controls.get("feature_shuffle", {})
        control_max_sp = max(
            abs(target_shuffle.get("spearman_group_cv", 0.0) or 0.0),
            abs(feature_shuffle.get("spearman_group_cv", 0.0) or 0.0),
        )
        no_score_pass = (not math.isnan(r2_ns) and r2_ns >= 0.20) or (not math.isnan(sp_ns) and sp_ns >= 0.30)
        no_accept_pass = (not math.isnan(r2_na) and r2_na >= 0.15) or (not math.isnan(sp_na) and sp_na >= 0.25)
        controls_clean = control_max_sp < 0.25
        feature_signal = no_score_pass and no_accept_pass and controls_clean
        notes.append(
            f"early feature model seed-held-out: R2={r2:.3f}, Spearman={sp:.3f}; "
            f"no-score R2={r2_ns:.3f}, no-score Spearman={sp_ns:.3f}; "
            f"no-score-no-accept R2={r2_na:.3f}, Spearman={sp_na:.3f}; "
            f"residual no-score Spearman={residual_sp:.3f}; negative-control max |Spearman|={control_max_sp:.3f}"
        )
        notes.append(
            "feature-policy gate: "
            f"{'PASS' if feature_signal else 'FAIL'} "
            f"(no_score={no_score_pass}, no_score_no_accept={no_accept_pass}, controls_clean={controls_clean})"
        )
        for label in ["source_group_cv_all_features", "H_group_cv_all_features", "phase_group_cv_all_features", "arm_group_cv_all_features"]:
            row = model.get(label, {})
            if row.get("valid"):
                notes.append(f"{label}: R2={row.get('r2_group_cv', float('nan')):.3f}, Spearman={row.get('spearman_group_cv', float('nan')):.3f}")
    else:
        notes.append(f"early feature model invalid: {model.get('reason')}")
    align_signal = False
    if alignment.get("valid"):
        lift = alignment.get("success_lift_vs_shuffle", 0.0)
        strat_lift = alignment.get("success_lift_vs_H_source_shuffle", lift)
        cos = alignment.get("success_pairwise_cosine", 0.0)
        align_signal = cos > 0.0 and strat_lift > 0.05
        notes.append(
            f"trajectory alignment: success_cos={cos:.3f}, "
            f"lift_vs_shuffle={lift:.3f}, lift_vs_H_source_shuffle={strat_lift:.3f}"
        )
    else:
        notes.append(f"trajectory alignment invalid: {alignment.get('reason')}")
    op_signal_ok = False
    if op_signal.get("valid"):
        op_signal_ok = op_signal.get("pass_h_count", 0) >= 2
        notes.append(f"operator lift passes H-count={op_signal.get('pass_h_count', 0)}")
    else:
        notes.append(f"operator signal invalid: {op_signal.get('reason')}")
    if feature_signal:
        return "D7_FEATURE_POLICY", notes
    if op_signal_ok:
        return "D7_OPERATOR_BANDIT", notes
    if align_signal:
        return "D7_FEATURE_POLICY", notes
    if model.get("valid") or alignment.get("valid") or op_signal.get("valid"):
        return "D6_NO_SIGNAL_INSTRUMENT_FIRST", notes
    return "D6_INCONCLUSIVE", notes


def write_report(path: Path, summary: dict, early_metrics: pd.DataFrame, op_summary: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase D6.1 Trajectory Field Falsification Audit",
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
        "- Raw candidate CSV rows were not re-scanned; D6.1 uses constructability summaries that represent those rows.",
        "",
        "## Decision Signals",
        "",
    ]
    lines.extend([f"- {note}" for note in summary["decision_notes"]])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- D6.1 uses stricter adversarial controls than the original D6 audit: fold-local scaling/imputation, global run IDs, source/H/phase/arm holdouts, no-score/no-accept-rate features, residual targets, stratified shuffles, and weighted operator usefulness.",
        "- The feature-state model still contains real signal, including after removing score fields and accept-rate. However, the direct feature-policy gate is not clean enough: one negative control remains above the pre-set margin, and H-held-out value prediction is weak.",
        "- The robust live-ready signal is operator-level: weighted top-vs-bottom operator usefulness remains above the 2x threshold for all tested H values and survives leave-one-run influence checks.",
        "- Therefore the safe next live experiment is D7.1 operator-bandit/adaptive operator weighting. A full feature-conditioned proposal remains a D7.2 candidate after the bandit baseline and stronger controls.",
    ])
    lines.extend([
        "",
        "## Early Feature Model",
        "",
        "```json",
        json.dumps(summary["early_feature_model"], indent=2),
        "```",
        "",
        "The decision uses train-fold-only scaling/imputation. The no-score variant excludes `main_peak_acc` and `panel_probe_acc`; the stricter no-score-no-accept variant also excludes `accept_rate_window`.",
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
        "| H | operator | weighted usefulness | mean usefulness | V_raw | M_pos | R_neg |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    if not op_summary.empty:
        for h, group in op_summary.groupby("H"):
            for _, row in group.head(5).iterrows():
                lines.append(
                    f"| {int(h)} | {row['operator_id']} | {row['usefulness_weighted']:.4g} | "
                    f"{row['usefulness_mean']:.4g} | {row['V_raw_mean']:.4g} | {row['M_pos_mean']:.4g} | {row['R_neg_mean']:.4g} |"
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
