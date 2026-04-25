"""Analyze acceptance-aperture geometry from Phase B.1 candidate logs.

This is an offline D0 analysis: it reads existing per-candidate logs and
decides whether an epsilon-tolerant acceptance sweep is informative under the
current best-of-K selector.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_ROOT = Path("output/phase_b1_horizon_ties_20260425")
DEFAULT_REPORT = Path("docs/research/PHASE_D0_ACCEPTANCE_APERTURE.md")
ZERO_TOLS = (0.0, 1e-12, 1e-9, 1e-6)
EPS_GRID = (0.0, 1e-12, 1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2)
SAMPLE_CAP = 200_000
NEGATIVE_BEST_INFORMATIVE_RATE = 0.05


@dataclass
class SampleBuffer:
    values: list[float] = field(default_factory=list)
    seen: int = 0

    def add(self, value: float) -> None:
        """Deterministic down-sample: keep the first SAMPLE_CAP values."""
        self.seen += 1
        if len(self.values) < SAMPLE_CAP:
            self.values.append(value)


@dataclass
class Counter:
    rows: int = 0
    evaluated: int = 0
    exact_zero: int = 0
    near_1e12: int = 0
    near_1e9: int = 0
    near_1e6: int = 0
    positive: int = 0
    negative: int = 0
    neg_sum: float = 0.0
    pos_sum: float = 0.0
    sample: SampleBuffer = field(default_factory=SampleBuffer)

    def add(self, delta: float, evaluated: bool) -> None:
        self.rows += 1
        if evaluated:
            self.evaluated += 1
        if delta == 0.0:
            self.exact_zero += 1
        if abs(delta) <= 1e-12:
            self.near_1e12 += 1
        if abs(delta) <= 1e-9:
            self.near_1e9 += 1
        if abs(delta) <= 1e-6:
            self.near_1e6 += 1
        if delta > 0.0:
            self.positive += 1
            self.pos_sum += delta
        elif delta < 0.0:
            self.negative += 1
            self.neg_sum += -delta
        self.sample.add(delta)

    def as_row(self, extra: dict) -> dict:
        rows = max(1, self.rows)
        return {
            **extra,
            "rows": self.rows,
            "evaluated": self.evaluated,
            "exact_zero_rate": self.exact_zero / rows,
            "near_zero_1e12_rate": self.near_1e12 / rows,
            "near_zero_1e9_rate": self.near_1e9 / rows,
            "near_zero_1e6_rate": self.near_1e6 / rows,
            "positive_rate": self.positive / rows,
            "negative_rate": self.negative / rows,
            "mean_positive_delta": self.pos_sum / self.positive if self.positive else 0.0,
            "mean_negative_magnitude": self.neg_sum / self.negative if self.negative else 0.0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-runs", type=int, default=30)
    parser.add_argument("--expected-rows", type=int, default=12_600_000)
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return value.lower() in {"true", "1", "yes", "on"}


def load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def finish_step(best_deltas: list[float], best_delta: float | None) -> None:
    best_deltas.append(float("nan") if best_delta is None else best_delta)


def gaussian_diag(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return {
            "sample_n": int(len(arr)),
            "mean": 0.0,
            "std": 0.0,
            "ks_stat": float("nan"),
            "ks_p_naive": float("nan"),
            "anderson_stat": float("nan"),
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std <= 0.0:
        return {
            "sample_n": int(len(arr)),
            "mean": mean,
            "std": std,
            "ks_stat": float("nan"),
            "ks_p_naive": float("nan"),
            "anderson_stat": float("nan"),
        }
    z = (arr - mean) / std
    ks = stats.kstest(z, "norm")
    try:
        ad = stats.anderson(z, dist="norm")
        ad_stat = float(ad.statistic)
    except Exception:
        ad_stat = float("nan")
    return {
        "sample_n": int(len(arr)),
        "mean": mean,
        "std": std,
        "ks_stat": float(ks.statistic),
        "ks_p_naive": float(ks.pvalue),
        "anderson_stat": ad_stat,
    }


def summarize_best_deltas(best_deltas: list[float], extra: dict) -> dict:
    arr = np.asarray(best_deltas, dtype=float)
    finite = arr[np.isfinite(arr)]
    total = max(1, len(arr))
    row = {
        **extra,
        "steps": int(len(arr)),
        "ineligible_best_steps": int(np.sum(~np.isfinite(arr))) if len(arr) else 0,
        "best_negative_rate": float(np.mean(arr < 0.0)) if len(arr) else 0.0,
        "best_exact_zero_rate": float(np.mean(arr == 0.0)) if len(arr) else 0.0,
        "best_near_zero_1e12_rate": float(np.mean(np.abs(arr) <= 1e-12)) if len(arr) else 0.0,
        "best_positive_rate": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "best_mean": float(np.mean(finite)) if len(finite) else 0.0,
        "best_q50": float(np.quantile(finite, 0.50)) if len(finite) else 0.0,
        "best_q90": float(np.quantile(finite, 0.90)) if len(finite) else 0.0,
        "best_q95": float(np.quantile(finite, 0.95)) if len(finite) else 0.0,
        "best_q99": float(np.quantile(finite, 0.99)) if len(finite) else 0.0,
    }
    for eps in EPS_GRID:
        label = f"A_emp_eps_{eps:g}".replace("-", "m").replace(".", "p")
        row[label] = float(np.sum(arr >= -eps) / total)
    return row


def analyze_run(csv_path: Path) -> tuple[dict, list[dict], list[float], Counter]:
    run_dir = csv_path.parent
    meta = load_meta(run_dir)
    run_counter = Counter()
    op_counters: dict[str, Counter] = defaultdict(Counter)
    best_deltas: list[float] = []
    current_step: int | None = None
    current_count = 0
    current_best: float | None = None
    expected_jackpot = int(meta.get("jackpot", 0))

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, 1):
            step = int(row["step"])
            if current_step is None:
                current_step = step
            if step != current_step:
                if expected_jackpot and current_count != expected_jackpot:
                    raise ValueError(
                        f"{csv_path}: step {current_step} rows={current_count}, expected {expected_jackpot}"
                    )
                finish_step(best_deltas, current_best)
                current_step = step
                current_count = 0
                current_best = None

            before = float(row["before_U"])
            after = float(row["after_U"])
            delta = float(row["delta_U"])
            if abs(delta - (after - before)) > 1e-12:
                raise ValueError(f"{csv_path}: delta mismatch row={row_idx}")
            evaluated = as_bool(row["evaluated"])
            within_cap = as_bool(row["within_cap"])
            run_counter.add(delta, evaluated)
            op_counters[row["operator_id"]].add(delta, evaluated)
            if within_cap:
                current_best = delta if current_best is None else max(current_best, delta)
            current_count += 1

    if current_step is not None:
        if expected_jackpot and current_count != expected_jackpot:
            raise ValueError(
                f"{csv_path}: step {current_step} rows={current_count}, expected {expected_jackpot}"
            )
        finish_step(best_deltas, current_best)

    expected_steps = int(meta.get("steps", 0))
    if expected_steps and len(best_deltas) != expected_steps:
        raise ValueError(f"{csv_path}: steps={len(best_deltas)}, expected {expected_steps}")

    extra = {
        "arm": csv_path.parent.parent.name,
        "seed": int(csv_path.parent.name.split("_", 1)[1]),
        "phase": meta.get("phase", ""),
        "horizon_steps": int(meta.get("horizon_steps", meta.get("steps", 0))),
        "accept_ties": bool(meta.get("accept_ties", False)),
    }
    run_row = {
        **run_counter.as_row(extra),
        **summarize_best_deltas(best_deltas, {}),
        **{f"gaussian_delta_{k}": v for k, v in gaussian_diag(run_counter.sample.values).items()},
        **{f"gaussian_best_{k}": v for k, v in gaussian_diag(best_deltas[:SAMPLE_CAP]).items()},
        "candidate_log": str(csv_path),
    }
    op_rows = [
        {
            **counter.as_row({**extra, "operator_id": op}),
            **{f"gaussian_delta_{k}": v for k, v in gaussian_diag(counter.sample.values).items()},
        }
        for op, counter in sorted(op_counters.items())
    ]
    return run_row, op_rows, best_deltas, run_counter


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row["n"] = int(len(group))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else 0.0
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def make_plots(root: Path, figures_dir: Path, all_best: dict[str, list[float]], run_rows: list[dict]) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, values in sorted(all_best.items()):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            continue
        ax.step(EPS_GRID, [np.mean(arr >= -eps) for eps in EPS_GRID], where="post", label=arm)
    ax.set_xscale("symlog", linthresh=1e-12)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Empirical acceptance aperture A_emp(epsilon)")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("P(best_delta >= -epsilon)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "acceptance_aperture_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for arm, values in sorted(all_best.items()):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        nz = arr[np.abs(arr) > 1e-12]
        if len(nz):
            ax.hist(nz, bins=120, alpha=0.35, density=True, label=arm)
    ax.set_title("Non-zero best_delta distribution")
    ax.set_xlabel("best_delta")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "best_delta_hist_nonzero.png", dpi=160)
    plt.close(fig)

    best_all = np.concatenate([np.asarray(v, dtype=float) for v in all_best.values()])
    best_all = best_all[np.isfinite(best_all)]
    sample = best_all[np.abs(best_all) > 1e-12]
    if len(sample) >= 20:
        sample = sample[:SAMPLE_CAP]
        mean = float(np.mean(sample))
        std = float(np.std(sample, ddof=1))
        if std > 0.0:
            z = np.sort((sample - mean) / std)
            q = stats.norm.ppf((np.arange(len(z)) + 0.5) / len(z))
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(q, z, s=4, alpha=0.35)
            ax.plot([q.min(), q.max()], [q.min(), q.max()], color="black", linewidth=1)
            ax.set_title("QQ plot: non-zero best_delta vs Gaussian")
            ax.set_xlabel("Gaussian quantile")
            ax.set_ylabel("sample quantile")
            fig.tight_layout()
            fig.savefig(figures_dir / "best_delta_qq_nonzero.png", dpi=160)
            plt.close(fig)


def render_report(
    report: Path,
    root: Path,
    artifact: dict,
    arm_summary: pd.DataFrame,
    operator_summary: pd.DataFrame,
    recommendation: dict,
) -> None:
    def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
        if max_rows is not None:
            df = df.head(max_rows)
        if df.empty:
            return "_empty_"
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase D0 Acceptance Aperture",
        "",
        "## Verdict",
        "",
        f"- Status: `{artifact['status']}`",
        f"- Candidate rows: `{artifact['candidate_rows']:,}` across `{artifact['runs']}` runs.",
        f"- Epsilon informative under current best-of-K selector: `{recommendation['epsilon_informative']}`.",
        f"- Recommended D1 mode: `{recommendation['recommended_d1_mode']}`.",
        f"- Reason: {recommendation['reason']}",
        "",
        "## Arm Summary",
        "",
        markdown_table(arm_summary),
        "",
        "## Operator Summary",
        "",
        markdown_table(operator_summary, max_rows=40),
        "",
        "## Interpretation",
        "",
        "- `C_K` remains the empirical progress metric; `A_pi(epsilon)` is only a Gaussian/isotropic null model.",
        "- If `best_negative_rate` is near zero, nonzero epsilon does not open new selected moves because zero-delta candidates dominate the best-of-K aperture.",
        "- In that case D1 should test probabilistic Zero-Drive (`neutral_p`) before any larger epsilon sweep.",
        "",
        "## Artifacts",
        "",
        f"- Root: `{root}`",
        f"- Summary JSON: `{root / 'analysis' / 'acceptance_aperture_summary.json'}`",
        f"- Figures: `{root / 'analysis' / 'figures'}`",
    ]
    report.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    root = args.root
    analysis_dir = root / "analysis"
    figures_dir = analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(root.glob("B1_*/seed_*/candidates.csv"))
    if len(csv_paths) != args.expected_runs:
        raise SystemExit(f"expected {args.expected_runs} candidate logs, found {len(csv_paths)}")

    run_rows: list[dict] = []
    op_rows: list[dict] = []
    all_best_by_arm: dict[str, list[float]] = defaultdict(list)
    total_rows = 0
    for csv_path in csv_paths:
        run, ops, best_deltas, run_counter = analyze_run(csv_path)
        run_rows.append(run)
        op_rows.extend(ops)
        all_best_by_arm[run["arm"]].extend(best_deltas)
        total_rows += run_counter.rows

    if total_rows != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} candidate rows, found {total_rows}")

    run_df = pd.DataFrame(run_rows)
    op_df = pd.DataFrame(op_rows)
    arm_metrics = [
        "best_negative_rate",
        "best_exact_zero_rate",
        "best_near_zero_1e12_rate",
        "best_positive_rate",
        "positive_rate",
        "negative_rate",
        "exact_zero_rate",
        "near_zero_1e12_rate",
        "gaussian_best_ks_stat",
        "gaussian_delta_ks_stat",
    ]
    arm_summary = aggregate(run_df, ["horizon_steps", "accept_ties", "arm"], arm_metrics)
    operator_summary = aggregate(
        op_df,
        ["operator_id"],
        ["positive_rate", "negative_rate", "exact_zero_rate", "near_zero_1e12_rate", "gaussian_delta_ks_stat"],
    ).sort_values("exact_zero_rate_mean", ascending=False)

    global_best = np.concatenate([np.asarray(v, dtype=float) for v in all_best_by_arm.values()])
    global_best = global_best[np.isfinite(global_best)]
    best_negative_rate = float(np.mean(global_best < 0.0))
    epsilon_informative = best_negative_rate >= NEGATIVE_BEST_INFORMATIVE_RATE
    recommendation = {
        "epsilon_informative": epsilon_informative,
        "best_negative_rate": best_negative_rate,
        "negative_best_informative_threshold": NEGATIVE_BEST_INFORMATIVE_RATE,
        "recommended_d1_mode": "epsilon+zero-p" if epsilon_informative else "zero-p-only",
        "reason": (
            "Enough steps have negative best_delta that epsilon can open new selected moves."
            if epsilon_informative
            else "Current best-of-K selector is zero-dominated; epsilon would usually select the same zero-delta moves as ties."
        ),
    }
    if epsilon_informative:
        negatives = -global_best[global_best < 0.0]
        recommendation["epsilon_small"] = float(np.quantile(negatives, 0.25))
        recommendation["epsilon_large"] = float(np.quantile(negatives, 0.75))
    else:
        recommendation["epsilon_small"] = None
        recommendation["epsilon_large"] = None

    make_plots(root, figures_dir, all_best_by_arm, run_rows)
    write_csv(analysis_dir / "acceptance_aperture_run_summary.csv", run_rows)
    write_csv(analysis_dir / "acceptance_aperture_operator_summary.csv", op_rows)
    arm_summary.to_csv(analysis_dir / "acceptance_aperture_arm_summary.csv", index=False)
    operator_summary.to_csv(analysis_dir / "acceptance_aperture_operator_rollup.csv", index=False)

    artifact = {
        "status": "PASS",
        "root": str(root),
        "runs": len(csv_paths),
        "candidate_rows": total_rows,
        "recommendation": recommendation,
        "arm_summary": arm_summary.to_dict(orient="records"),
        "operator_summary": operator_summary.to_dict(orient="records"),
    }
    (analysis_dir / "acceptance_aperture_summary.json").write_text(json.dumps(artifact, indent=2))
    render_report(args.report, root, artifact, arm_summary, operator_summary, recommendation)
    print(json.dumps({
        "status": "PASS",
        "runs": len(csv_paths),
        "candidate_rows": total_rows,
        "epsilon_informative": epsilon_informative,
        "recommended_d1_mode": recommendation["recommended_d1_mode"],
        "report": str(args.report),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
