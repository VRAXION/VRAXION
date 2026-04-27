"""Phase D8.0: Offline basin-potential (Psi) + archive replay audit.

This script is intentionally offline-only. It does not launch Rust runs and
does not scan raw candidates.csv logs. It asks whether existing panel states
contain enough signal to justify a future live archive layer over SAF v1.
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
    Path("output/phase_d7_operator_bandit_20260427"),
]
DEFAULT_OUT = Path("output/phase_d8_archive_psi_replay_20260427")
DEFAULT_REPORT = Path("docs/research/PHASE_D8_ARCHIVE_PSI_REPLAY_AUDIT.md")

PHI_FEATURES = [
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
    "H",
    "jackpot",
    "time_pct",
]
BUCKET_FEATURES = [
    "stable_rank",
    "separation_sp",
    "collision_rate",
    "unique_predictions",
    "f_active",
    "edges",
]
TARGETS = ["future_gain_local", "future_gain_mid", "future_gain_final"]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--archive-size", type=int, default=64)
    parser.add_argument("--bucket-bins", type=int, default=4)
    parser.add_argument("--basin-delta", type=float, default=0.005)
    parser.add_argument("--material-regression", type=float, default=-0.005)
    parser.add_argument("--control-spearman-margin", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def normalize_results(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["source"] = source
    if "run_id" not in out:
        out["run_id"] = [f"{source}_row_{i}" for i in range(len(out))]
    out["global_run_id"] = source + "::" + out["run_id"].astype(str)
    for col in ["H", "seed", "jackpot", "horizon_steps", "configured_steps"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for metric in ["peak_acc", "final_acc"]:
        if metric in out:
            val = pd.to_numeric(out[metric], errors="coerce")
            out[f"{metric}_pct"] = val * 100.0 if val.max(skipna=True) <= 1.0 else val
    return out


def load_results(roots: list[Path], max_runs: int | None) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    used_roots = []
    for root in roots:
        if not root.exists():
            continue
        df = normalize_results(read_csv(root / "results.csv"), root.name)
        if df.empty:
            continue
        frames.append(df)
        used_roots.append(root.name)
    if not frames:
        raise SystemExit("no usable results.csv files found")
    results = pd.concat(frames, ignore_index=True)
    results = results.sort_values(["source", "H", "arm", "seed"], na_position="last")
    dup = int(results["global_run_id"].duplicated().sum())
    if dup:
        raise SystemExit(f"duplicate global_run_id rows found: {dup}")
    if max_runs is not None:
        results = results.head(max_runs).copy()
    return results, used_roots


def resolve_path(value: object, root: Path | None = None) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    if root is not None:
        alt = root / path
        if alt.exists():
            return alt
    return path


def load_panel_dataset(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, run in results.iterrows():
        panel_path = resolve_path(run.get("panel_timeseries", ""))
        if not panel_path.exists():
            run_dir = resolve_path(run.get("run_dir", ""))
            panel_path = run_dir / "panel_timeseries.csv"
        if not panel_path.exists():
            continue
        panel = pd.read_csv(panel_path)
        if panel.empty or "main_peak_acc" not in panel:
            continue
        panel = panel.copy()
        panel["source"] = run.get("source", "")
        panel["run_id"] = str(run.get("run_id", ""))
        panel["global_run_id"] = str(run.get("global_run_id", ""))
        panel["phase"] = str(run.get("phase", ""))
        panel["arm"] = str(run.get("arm", ""))
        panel["fixture"] = str(run.get("fixture", ""))
        panel["H"] = int(run.get("H")) if pd.notna(run.get("H")) else -1
        panel["seed"] = int(run.get("seed")) if pd.notna(run.get("seed")) else -1
        panel["jackpot"] = int(run.get("jackpot")) if pd.notna(run.get("jackpot")) else -1
        panel["panel_index"] = np.arange(len(panel), dtype=int)
        panel["state_id"] = (
            panel["source"].astype(str)
            + "::"
            + panel["run_id"].astype(str)
            + "::"
            + panel["panel_index"].astype(str)
        )
        panel["parent_id"] = np.where(
            panel["panel_index"] > 0,
            panel["source"].astype(str)
            + "::"
            + panel["run_id"].astype(str)
            + "::"
            + (panel["panel_index"] - 1).astype(str),
            "",
        )
        if "step" in panel:
            step = pd.to_numeric(panel["step"], errors="coerce")
            denom = float(step.max()) if step.notna().any() and step.max() > 0 else float(len(panel))
            panel["time_pct"] = step / denom
        else:
            panel["time_pct"] = (panel["panel_index"] + 1) / max(len(panel), 1)
        for feature in PHI_FEATURES:
            if feature not in panel:
                panel[feature] = np.nan
        add_future_labels(panel)
        rows.append(panel)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_future_labels(panel: pd.DataFrame) -> None:
    peak = pd.to_numeric(panel["main_peak_acc"], errors="coerce").ffill().fillna(0.0).to_numpy(float)
    n = len(peak)
    local = np.zeros(n)
    mid = np.zeros(n)
    final = np.zeros(n)
    future_peak = np.zeros(n)
    for i in range(n):
        if i + 1 >= n:
            f_local = f_mid = f_final = peak[i]
        else:
            f_local = float(np.max(peak[i + 1 : min(n, i + 3)]))
            f_mid = float(np.max(peak[i + 1 : min(n, i + 6)]))
            f_final = float(np.max(peak[i + 1 :]))
        local[i] = max(0.0, f_local - peak[i])
        mid[i] = max(0.0, f_mid - peak[i])
        final[i] = max(0.0, f_final - peak[i])
        future_peak[i] = f_final
    panel["current_peak"] = peak
    panel["future_peak_final"] = future_peak
    panel["future_gain_local"] = local
    panel["future_gain_mid"] = mid
    panel["future_gain_final"] = final
    panel["basin_hit"] = (panel["future_gain_final"] >= 0.005).astype(int)


def fit_scaler(df: pd.DataFrame, cols: list[str]) -> dict:
    clean = [c for c in cols if c in df and pd.to_numeric(df[c], errors="coerce").notna().sum() > 2]
    if not clean:
        return {"columns": [], "median": np.array([]), "mean": np.array([]), "std": np.array([])}
    x = df[clean].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(med, inds[1])
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0.0] = 1.0
    return {"columns": clean, "median": med, "mean": mu, "std": sd}


def transform_scaler(df: pd.DataFrame, scaler: dict) -> np.ndarray:
    cols = scaler["columns"]
    if not cols:
        return np.empty((len(df), 0))
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(scaler["median"], inds[1])
    return (x - scaler["mean"]) / scaler["std"]


def fit_ridge(train: pd.DataFrame, feature_cols: list[str], target_col: str, alpha: float) -> tuple[np.ndarray, dict]:
    scaler = fit_scaler(train, feature_cols)
    x = transform_scaler(train, scaler)
    y = pd.to_numeric(train[target_col], errors="coerce").to_numpy(float)
    x_aug = np.c_[np.ones(len(x)), x]
    reg = np.eye(x_aug.shape[1]) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ y
    return beta, scaler


def predict_ridge(df: pd.DataFrame, beta: np.ndarray, scaler: dict) -> np.ndarray:
    x = transform_scaler(df, scaler)
    return np.c_[np.ones(len(x)), x] @ beta


def score_predictions(y: np.ndarray, pred: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(pred)
    if int(ok.sum()) < 5:
        return {"valid": False, "n": int(ok.sum()), "r2": math.nan, "spearman": math.nan, "pearson": math.nan}
    y_ok = y[ok]
    p_ok = pred[ok]
    sse = float(np.sum((y_ok - p_ok) ** 2))
    sst = float(np.sum((y_ok - float(np.mean(y_ok))) ** 2))
    return {
        "valid": True,
        "n": int(ok.sum()),
        "r2": 1.0 - sse / sst if sst > 0 else 0.0,
        "spearman": safe_spearman(y_ok, p_ok),
        "pearson": safe_pearson(y_ok, p_ok),
    }


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) <= EPS or np.nanstd(b) <= EPS:
        return 0.0
    val = stats.spearmanr(a, b, nan_policy="omit").correlation
    return float(0.0 if not np.isfinite(val) else val)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) <= EPS or np.nanstd(b) <= EPS:
        return 0.0
    val = stats.pearsonr(a, b).statistic
    return float(0.0 if not np.isfinite(val) else val)


def group_cv_predictions(df: pd.DataFrame, feature_cols: list[str], target_col: str, group_col: str, alpha: float) -> tuple[np.ndarray, dict]:
    work = df.dropna(subset=[target_col, group_col]).copy()
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    groups = sorted(work[group_col].dropna().unique().tolist())
    if len(groups) < 2 or len(work) < 10:
        y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)
        return pred.to_numpy(float), {"valid": False, "reason": "not_enough_groups", "groups": len(groups), "n": int(len(work))}
    for group in groups:
        test_idx = work.index[work[group_col] == group]
        train = work[work[group_col] != group]
        test = work.loc[test_idx]
        if len(train) < 5 or len(test) == 0:
            continue
        beta, scaler = fit_ridge(train, feature_cols, target_col, alpha)
        pred.loc[test_idx] = predict_ridge(test, beta, scaler)
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)
    score = score_predictions(y, pred.to_numpy(float))
    score["groups"] = len(groups)
    return pred.to_numpy(float), score


def validation_suite(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_frames = []
    feature_sets = {
        "psi_phi": PHI_FEATURES,
        "score_only": ["main_peak_acc"],
        "time_only": ["time_pct"],
    }
    for target in TARGETS:
        for model_name, cols in feature_sets.items():
            for group_col in ["seed", "source", "phase", "H"]:
                preds, score = group_cv_predictions(df, cols, target, group_col, args.ridge_alpha)
                rows.append({"target": target, "model": model_name, "cv": group_col, **score})
                if target == "future_gain_final" and model_name == "psi_phi" and group_col == "seed":
                    pred_frames.append(pd.DataFrame({"row_index": df.index, "psi_pred_seed_cv": preds}))
    controls = negative_controls(df, args)
    rows.extend(controls)
    pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return pd.DataFrame(rows), pred_df


def negative_controls(df: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(8008)
    rows = []
    work = df.copy()
    target_shuffle = work.copy()
    for _, idx in target_shuffle.groupby(["H", "source"], dropna=False).groups.items():
        vals = target_shuffle.loc[idx, "future_gain_final"].to_numpy(float)
        rng.shuffle(vals)
        target_shuffle.loc[idx, "future_gain_final"] = vals
    _, score = group_cv_predictions(target_shuffle, PHI_FEATURES, "future_gain_final", "seed", args.ridge_alpha)
    rows.append({"target": "future_gain_final", "model": "target_shuffle_H_source", "cv": "seed", **score})

    feature_shuffle = work.copy()
    for col in PHI_FEATURES:
        if col in feature_shuffle:
            vals = feature_shuffle[col].to_numpy(copy=True)
            rng.shuffle(vals)
            feature_shuffle[col] = vals
    _, score = group_cv_predictions(feature_shuffle, PHI_FEATURES, "future_gain_final", "seed", args.ridge_alpha)
    rows.append({"target": "future_gain_final", "model": "feature_shuffle", "cv": "seed", **score})
    return rows


def add_psi_predictions(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    out["psi_pred"] = np.nan
    for h, sub in out.groupby("H", dropna=False):
        if len(sub) < 8:
            continue
        beta, scaler = fit_ridge(sub, PHI_FEATURES, "future_gain_final", args.ridge_alpha)
        out.loc[sub.index, "psi_pred"] = predict_ridge(sub, beta, scaler)
    if out["psi_pred"].isna().all():
        out["psi_pred"] = pd.to_numeric(out["future_gain_final"], errors="coerce").fillna(0.0)
    out["psi_pred"] = out["psi_pred"].fillna(out["psi_pred"].median())
    return out


def quantile_bucket(sub: pd.DataFrame, feature: str, bins: int) -> pd.Series:
    vals = pd.to_numeric(sub[feature], errors="coerce")
    if vals.notna().sum() < 4 or vals.nunique(dropna=True) < 2:
        return pd.Series(["0"] * len(sub), index=sub.index)
    ranks = vals.rank(method="average", pct=True).fillna(0.5)
    q = np.floor(np.minimum(ranks.to_numpy() * bins, bins - 1)).astype(int)
    return pd.Series(q.astype(str), index=sub.index)


def build_buckets(sub: pd.DataFrame, bins: int) -> pd.Series:
    parts = []
    for feature in BUCKET_FEATURES:
        if feature in sub:
            parts.append(quantile_bucket(sub, feature, bins))
    if not parts:
        return pd.Series(["all"] * len(sub), index=sub.index)
    bucket = parts[0].copy()
    for part in parts[1:]:
        bucket = bucket + "_" + part
    return bucket


def zscore(vals: pd.Series) -> pd.Series:
    num = pd.to_numeric(vals, errors="coerce").fillna(0.0)
    sd = float(num.std())
    if sd <= EPS:
        return num * 0.0
    return (num - float(num.mean())) / sd


def novelty_scores(sub: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    scaler = fit_scaler(sub, feature_cols)
    x = transform_scaler(sub, scaler)
    n = len(x)
    if n < 3 or x.shape[1] == 0:
        return pd.Series(np.zeros(n), index=sub.index)
    scores = np.zeros(n)
    for i in range(n):
        dist = np.sqrt(np.sum((x - x[i]) ** 2, axis=1))
        dist[i] = np.inf
        k = min(5, n - 1)
        scores[i] = float(np.mean(np.partition(dist, k - 1)[:k]))
    return pd.Series(scores, index=sub.index)


def select_top_n(sub: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    return sub.sort_values(score_col, ascending=False).head(min(n, len(sub))).copy()


def replay_archives(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_frames = []
    for h, hsub in df.groupby("H", dropna=False):
        hsub = hsub.copy()
        if hsub.empty:
            continue
        n = min(args.archive_size, len(hsub))
        hsub["phi_bucket"] = build_buckets(hsub, args.bucket_bins)
        hsub["novelty"] = novelty_scores(hsub, [c for c in BUCKET_FEATURES if c in hsub])
        hsub["psi_novelty_score"] = zscore(hsub["psi_pred"]) + 0.25 * zscore(hsub["novelty"])
        selectors = {
            "S0_SCORE_TOPN": select_top_n(hsub, "current_peak", n),
            "A1_PHI_BUCKET": hsub.sort_values("current_peak", ascending=False).groupby("phi_bucket", as_index=False).head(1),
            "A2_BUCKET_PSI": hsub.sort_values("psi_pred", ascending=False).groupby("phi_bucket", as_index=False).head(1),
            "A3_BUCKET_PSI_NOVELTY": hsub.sort_values("psi_novelty_score", ascending=False).groupby("phi_bucket", as_index=False).head(1),
        }
        for selector, sel in selectors.items():
            if len(sel) > n:
                score_col = "current_peak" if selector in {"S0_SCORE_TOPN", "A1_PHI_BUCKET"} else "psi_pred"
                if selector == "A3_BUCKET_PSI_NOVELTY":
                    score_col = "psi_novelty_score"
                sel = select_top_n(sel, score_col, n)
            sel = sel.copy()
            sel["selector"] = selector
            selected_frames.append(sel)
            rows.append({
                "H": int(h) if pd.notna(h) else -1,
                "selector": selector,
                "n_selected": int(len(sel)),
                "mean_future_gain": float(sel["future_gain_final"].mean()) if len(sel) else math.nan,
                "median_future_gain": float(sel["future_gain_final"].median()) if len(sel) else math.nan,
                "topk_basin_precision": float(sel["basin_hit"].mean()) if len(sel) else math.nan,
                "mean_current_peak": float(sel["current_peak"].mean()) if len(sel) else math.nan,
                "coverage_buckets": int(sel["phi_bucket"].nunique()) if "phi_bucket" in sel else 1,
            })
    replay = pd.DataFrame(rows)
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    if not replay.empty:
        base = replay[replay["selector"] == "S0_SCORE_TOPN"][["H", "mean_future_gain", "topk_basin_precision"]]
        base = base.rename(columns={"mean_future_gain": "base_mean_future_gain", "topk_basin_precision": "base_basin_precision"})
        replay = replay.merge(base, on="H", how="left")
        replay["delta_mean_future_gain_vs_score"] = replay["mean_future_gain"] - replay["base_mean_future_gain"]
        replay["delta_basin_precision_vs_score"] = replay["topk_basin_precision"] - replay["base_basin_precision"]
    return replay, selected


def psi_deciles(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work["psi_pred"].nunique(dropna=True) < 2:
        work["psi_decile"] = 0
    else:
        work["psi_decile"] = pd.qcut(work["psi_pred"].rank(method="first"), q=10, labels=False, duplicates="drop")
    return (
        work.groupby(["H", "psi_decile"], dropna=False)
        .agg(
            n=("future_gain_final", "size"),
            mean_future_gain=("future_gain_final", "mean"),
            basin_rate=("basin_hit", "mean"),
            mean_psi=("psi_pred", "mean"),
        )
        .reset_index()
    )


def decide(validation: pd.DataFrame, replay: pd.DataFrame, args: argparse.Namespace) -> tuple[str, dict]:
    def metric(model: str, cv: str = "seed", target: str = "future_gain_final") -> dict:
        sub = validation[(validation["model"] == model) & (validation["cv"] == cv) & (validation["target"] == target)]
        return sub.iloc[0].to_dict() if not sub.empty else {}

    psi = metric("psi_phi")
    score = metric("score_only")
    time = metric("time_only")
    tshuf = metric("target_shuffle_H_source")
    fshuf = metric("feature_shuffle")
    psi_sp = float(psi.get("spearman", 0.0) or 0.0)
    score_sp = float(score.get("spearman", 0.0) or 0.0)
    time_sp = float(time.get("spearman", 0.0) or 0.0)
    ctrl_max = max(abs(float(tshuf.get("spearman", 0.0) or 0.0)), abs(float(fshuf.get("spearman", 0.0) or 0.0)))
    psi_beats = psi_sp > score_sp and psi_sp > time_sp
    controls_fail = ctrl_max < max(psi_sp - args.control_spearman_margin, 0.10)

    selector_stats = {}
    archive_pass_count = 0
    archive_material_regression = False
    archive_only_count = 0
    if not replay.empty:
        for selector in ["A1_PHI_BUCKET", "A2_BUCKET_PSI", "A3_BUCKET_PSI_NOVELTY"]:
            sub = replay[replay["selector"] == selector]
            h_deltas = {str(int(r["H"])): float(r["delta_mean_future_gain_vs_score"]) for _, r in sub.iterrows()}
            pass_h = sum(1 for v in h_deltas.values() if v > 0.0)
            material = any(v < args.material_regression for v in h_deltas.values())
            selector_stats[selector] = {"h_deltas": h_deltas, "pass_h_count": pass_h, "material_regression": material}
            if selector in {"A2_BUCKET_PSI", "A3_BUCKET_PSI_NOVELTY"}:
                archive_pass_count = max(archive_pass_count, pass_h)
                archive_material_regression = archive_material_regression or material
            if selector == "A1_PHI_BUCKET":
                archive_only_count = pass_h

    if len(replay) == 0 or len(validation) == 0:
        verdict = "D8_NEEDS_INSTRUMENTATION"
    elif psi_beats and controls_fail and archive_pass_count >= 2 and not archive_material_regression:
        verdict = "D8_PSI_ARCHIVE_OFFLINE_PASS"
    elif archive_only_count >= 2 and not selector_stats.get("A1_PHI_BUCKET", {}).get("material_regression", True):
        verdict = "D8_ARCHIVE_ONLY_PASS"
    elif any(s.get("pass_h_count", 0) >= 1 for s in selector_stats.values()) and not controls_fail:
        verdict = "D8_NEEDS_INSTRUMENTATION"
    else:
        verdict = "D8_REJECT_ARCHIVE_SIGNAL"

    return verdict, {
        "psi_seed_cv_spearman": psi_sp,
        "score_seed_cv_spearman": score_sp,
        "time_seed_cv_spearman": time_sp,
        "negative_control_max_abs_spearman": ctrl_max,
        "psi_beats_baselines": psi_beats,
        "controls_fail": controls_fail,
        "selector_stats": selector_stats,
    }


def write_plots(out: Path, replay: pd.DataFrame, deciles: pd.DataFrame, selected: pd.DataFrame) -> None:
    fig_dir = out / "analysis" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not deciles.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for h, sub in deciles.groupby("H"):
            ax.plot(sub["psi_decile"], sub["mean_future_gain"], marker="o", label=f"H={int(h)}")
        ax.set_xlabel("Psi decile")
        ax.set_ylabel("actual mean future_gain")
        ax.legend()
        plt.tight_layout()
        fig.savefig(fig_dir / "psi_decile_future_gain.png", dpi=160)
        plt.close(fig)
    if not replay.empty:
        pivot = replay.pivot(index="H", columns="selector", values="coverage_buckets")
        ax = pivot.plot(kind="bar", figsize=(9, 4))
        ax.set_ylabel("selected phi bucket coverage")
        plt.tight_layout()
        plt.savefig(fig_dir / "archive_coverage.png", dpi=160)
        plt.close()

        pivot = replay.pivot(index="H", columns="selector", values="mean_future_gain")
        ax = pivot.plot(kind="bar", figsize=(9, 4))
        ax.set_ylabel("mean selected future_gain")
        plt.tight_layout()
        plt.savefig(fig_dir / "selector_topk_future_gain.png", dpi=160)
        plt.close()


def write_report(path: Path, verdict: str, decision: dict, coverage: dict, validation: pd.DataFrame, replay: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase D8.0 Archive + Psi Replay Audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Summary",
        "",
        "- D8.0 is offline-only: no live search, no SAF changes, no raw candidate scan.",
        "- State identity is `source::run_id::panel_index`; parent identity is previous panel in the same run.",
        "- The audit tests whether panel fingerprints predict future gain and whether archive replay beats score-only retention.",
        "- Archive replay uses seed-held-out `Psi` predictions where available, with in-sample fallback only for missing predictions.",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(coverage, indent=2),
        "```",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(decision, indent=2),
        "```",
        "",
        "## Psi Validation",
        "",
        "```text",
        validation.to_string(index=False, max_rows=80),
        "```",
        "",
        "## Archive Replay",
        "",
        "```text",
        replay.to_string(index=False, max_rows=80),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "D8_PSI_ARCHIVE_OFFLINE_PASS":
        lines.append("- `Psi` and archive replay show enough offline signal to justify D8.1 instrumentation-only logging before any live archive steering.")
    elif verdict == "D8_ARCHIVE_ONLY_PASS":
        lines.append("- Behavior bucketing helps, but learned `Psi` is not clean enough yet. Next step should be archive instrumentation without `Psi` authority.")
    elif verdict == "D8_NEEDS_INSTRUMENTATION":
        lines.append("- Existing panel logs are not sufficient to lock the archive/Psi hypothesis. Add richer state/family instrumentation before live steering.")
    else:
        lines.append("- Existing logs do not support archive/Psi over score-only retention under the current controls.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    analysis_dir = args.out / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    results, used_roots = load_results(args.roots, args.max_runs)
    panel = load_panel_dataset(results)
    if panel.empty:
        raise SystemExit("no usable panel_timeseries rows found")
    panel["basin_hit"] = (panel["future_gain_final"] >= args.basin_delta).astype(int)

    validation, seed_preds = validation_suite(panel, args)
    panel_with_psi = add_psi_predictions(panel, args)
    panel_with_psi["psi_pred_source"] = "in_sample_per_H"
    if not seed_preds.empty:
        pred_map = seed_preds.set_index("row_index")["psi_pred_seed_cv"]
        panel_with_psi["psi_pred_seed_cv"] = panel_with_psi.index.map(pred_map)
        ok = panel_with_psi["psi_pred_seed_cv"].notna()
        if ok.any():
            panel_with_psi.loc[ok, "psi_pred"] = panel_with_psi.loc[ok, "psi_pred_seed_cv"]
            panel_with_psi.loc[ok, "psi_pred_source"] = "seed_held_out_cv"
    replay, selected = replay_archives(panel_with_psi, args)
    deciles = psi_deciles(panel_with_psi)
    verdict, decision = decide(validation, replay, args)

    coverage = {
        "used_roots": used_roots,
        "runs": int(results["global_run_id"].nunique()),
        "panel_rows": int(len(panel)),
        "H_values": sorted([int(x) for x in panel["H"].dropna().unique().tolist()]),
        "sources": sorted(panel["source"].dropna().unique().tolist()),
        "archive_size": args.archive_size,
        "bucket_bins": args.bucket_bins,
        "basin_delta": args.basin_delta,
        "psi_replay_prediction": "seed-held-out CV where available; in-sample per-H fallback only for missing predictions",
    }
    summary = {"verdict": verdict, "coverage": coverage, "decision": decision}

    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    panel_with_psi.to_csv(analysis_dir / "panel_state_dataset.csv", index=False)
    validation.to_csv(analysis_dir / "psi_validation.csv", index=False)
    replay.to_csv(analysis_dir / "archive_replay_summary.csv", index=False)
    if not replay.empty:
        replay[["H", "selector", "delta_mean_future_gain_vs_score", "delta_basin_precision_vs_score"]].to_csv(
            analysis_dir / "per_H_selector_deltas.csv", index=False
        )
    deciles.to_csv(analysis_dir / "psi_decile_future_gain.csv", index=False)
    selected.to_csv(analysis_dir / "selected_archive_states.csv", index=False)
    write_plots(args.out, replay, deciles, selected)
    write_report(args.report, verdict, decision, coverage, validation, replay)

    print(f"Verdict: {verdict}")
    print(f"Runs: {coverage['runs']}")
    print(f"Panel rows: {coverage['panel_rows']}")
    print(f"Wrote: {analysis_dir / 'summary.json'}")
    print(f"Wrote: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
