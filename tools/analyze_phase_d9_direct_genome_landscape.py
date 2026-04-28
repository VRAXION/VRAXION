"""Phase D9.0b: direct genome landscape analyzer.

Consumes samples emitted by `instnct-core/examples/d9_direct_landscape.rs`
and determines whether a direct core-genome neighborhood is smooth,
rugged-but-searchable, type-split, cliffy, or random.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_INPUT = Path("output/phase_d9_direct_genome_landscape_20260428/samples.csv")
DEFAULT_OUT = Path("output/phase_d9_direct_genome_landscape_20260428/analysis")
DEFAULT_REPORT = Path("docs/research/PHASE_D9_DIRECT_GENOME_LANDSCAPE_AUDIT.md")

VERDICT_SMOOTH = "D9_DIRECT_LANDSCAPE_SMOOTH"
VERDICT_RUGGED = "D9_DIRECT_LANDSCAPE_RUGGED_BUT_SEARCHABLE"
VERDICT_TYPE_SPLIT = "D9_DIRECT_LANDSCAPE_TYPE_SPLIT"
VERDICT_CLIFFY = "D9_DIRECT_LANDSCAPE_CLIFFY"
VERDICT_RANDOM = "D9_DIRECT_LANDSCAPE_RANDOM"
VERDICT_INFRA = "D9_DIRECT_LANDSCAPE_INFRA_FAIL"

CLIFF_DELTA = -0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cliff-delta", type=float, default=CLIFF_DELTA)
    parser.add_argument("--skip-html", action="store_true")
    return parser.parse_args()


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def safe_spearman(x: pd.Series, y: pd.Series) -> dict[str, float | None]:
    mask = x.notna() & y.notna()
    if mask.sum() < 4 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return {"rho": None, "p": None}
    rho, p = stats.spearmanr(x[mask], y[mask])
    return {
        "rho": float(rho) if math.isfinite(float(rho)) else None,
        "p": float(p) if math.isfinite(float(p)) else None,
    }


def chunk_best_of(values: pd.Series, k: int) -> float:
    arr = values.to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    chunks = [arr[i : i + k] for i in range(0, len(arr), k) if len(arr[i : i + k]) > 0]
    return float(np.mean([np.max(chunk) for chunk in chunks]))


def classify_type(group: pd.DataFrame, cliff_delta: float) -> str:
    low = group[group["requested_radius"] <= 4]
    low_cliff = float((low["delta_score"] <= cliff_delta).mean()) if len(low) else 1.0
    positive = float((low["delta_score"] > 0.0).mean()) if len(low) else 0.0
    best9 = chunk_best_of(low["delta_score"], 9) if len(low) else float("nan")
    rho = safe_spearman(group["requested_radius"], group["behavior_distance"])["rho"]
    rho = 0.0 if rho is None else rho

    if low_cliff >= 0.40:
        return "cliffy"
    if rho >= 0.45 and low_cliff <= 0.10 and (positive >= 0.02 or best9 > 0.0):
        return "smooth"
    if (rho >= 0.20 or best9 > 0.0 or positive >= 0.02) and low_cliff < 0.40:
        return "rugged"
    return "random"


def decide_verdict(type_classes: dict[str, str]) -> str:
    classes = set(type_classes.values())
    if not type_classes:
        return VERDICT_INFRA
    if classes == {"smooth"}:
        return VERDICT_SMOOTH
    if classes <= {"smooth", "rugged"}:
        return VERDICT_RUGGED
    if "smooth" in classes or "rugged" in classes:
        return VERDICT_TYPE_SPLIT
    if classes == {"cliffy"}:
        return VERDICT_CLIFFY
    return VERDICT_RANDOM


def summarize(df: pd.DataFrame, cliff_delta: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    df["cliff"] = df["delta_score"] <= cliff_delta
    df["positive"] = df["delta_score"] > 0.0

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(["base_index", "mutation_type", "requested_radius"], dropna=False):
        base_index, mutation_type, radius = keys
        rows.append(
            {
                "base_index": int(base_index),
                "mutation_type": mutation_type,
                "requested_radius": int(radius),
                "n": int(len(group)),
                "mean_delta_score": float(group["delta_score"].mean()),
                "median_delta_score": float(group["delta_score"].median()),
                "p10_delta_score": float(group["delta_score"].quantile(0.10)),
                "p90_delta_score": float(group["delta_score"].quantile(0.90)),
                "best_delta_score": float(group["delta_score"].max()),
                "positive_delta_rate": float(group["positive"].mean()),
                "cliff_rate": float(group["cliff"].mean()),
                "mean_behavior_distance": float(group["behavior_distance"].mean()),
                "best_of_9_gain": chunk_best_of(group["delta_score"], 9),
                "best_of_18_gain": chunk_best_of(group["delta_score"], 18),
            }
        )
    per_group = pd.DataFrame(rows).sort_values(["base_index", "mutation_type", "requested_radius"])

    type_rows: list[dict[str, Any]] = []
    type_classes: dict[str, str] = {}
    for mutation_type, group in df.groupby("mutation_type", dropna=False):
        c = classify_type(group, cliff_delta)
        type_classes[str(mutation_type)] = c
        low = group[group["requested_radius"] <= 4]
        radius_behavior = safe_spearman(group["requested_radius"], group["behavior_distance"])
        radius_abs_delta = safe_spearman(group["requested_radius"], group["delta_score"].abs())
        type_rows.append(
            {
                "mutation_type": mutation_type,
                "classification": c,
                "n": int(len(group)),
                "low_radius_n": int(len(low)),
                "low_radius_cliff_rate": float((low["delta_score"] <= cliff_delta).mean()) if len(low) else None,
                "low_radius_positive_rate": float((low["delta_score"] > 0.0).mean()) if len(low) else None,
                "best_of_9_low_radius": chunk_best_of(low["delta_score"], 9) if len(low) else None,
                "rho_radius_behavior": radius_behavior["rho"],
                "p_radius_behavior": radius_behavior["p"],
                "rho_radius_abs_delta": radius_abs_delta["rho"],
                "p_radius_abs_delta": radius_abs_delta["p"],
            }
        )
    per_type = pd.DataFrame(type_rows).sort_values("mutation_type")
    verdict = decide_verdict(type_classes)

    overall = {
        "verdict": verdict,
        "rows": int(len(df)),
        "base_count": int(df["base_index"].nunique()),
        "mutation_types": sorted(df["mutation_type"].dropna().astype(str).unique().tolist()),
        "radii": sorted(int(v) for v in df["requested_radius"].dropna().unique().tolist()),
        "type_classes": type_classes,
        "overall_delta_mean": float(df["delta_score"].mean()),
        "overall_delta_median": float(df["delta_score"].median()),
        "overall_cliff_rate": float(df["cliff"].mean()),
        "overall_positive_rate": float(df["positive"].mean()),
        "psi_available": "psi_pred" in df.columns or "delta_psi" in df.columns,
        "controls": {
            "radius_label_shuffle": "computed_by_verdict_as_orderless_group_summary",
            "random_archive_pair_baseline": "not_available_unless_extra_inputs_are_supplied",
            "repeated_base_eval_noise": "implicit_from_base_score_resampling_per_sample",
        },
    }
    return per_group, per_type, overall


def make_pca_projection(df: pd.DataFrame) -> np.ndarray:
    cols = [
        "direct_genome_distance",
        "edge_edits",
        "threshold_edits",
        "channel_edits",
        "polarity_edits",
        "behavior_distance",
        "edges",
        "panel_probe_acc",
        "unique_predictions",
        "collision_rate",
        "f_active",
        "stable_rank",
        "kernel_rank",
        "separation_sp",
    ]
    existing = [c for c in cols if c in df.columns]
    x = df[existing].astype(float).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).to_numpy(dtype=float)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def save_figures(df: pd.DataFrame, per_group: pd.DataFrame, out: Path) -> None:
    if len(df) == 0:
        return
    coords = make_pca_projection(df)
    df = df.copy()
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]
    df[["sample_id", "pc1", "pc2"]].to_csv(out / "local_zone_projection.csv", index=False)

    plt.figure(figsize=(9, 7))
    sc = plt.scatter(df["pc1"], df["pc2"], c=df["delta_score"], cmap="RdYlGn", s=28, alpha=0.85)
    plt.axhline(0, color="black", lw=0.4, alpha=0.4)
    plt.axvline(0, color="black", lw=0.4, alpha=0.4)
    plt.colorbar(sc, label="delta_score")
    plt.title("D9.0b local zone heatmap (PCA/SVD projection)")
    plt.xlabel("local zone PC1")
    plt.ylabel("local zone PC2")
    plt.tight_layout()
    plt.savefig(out / "local_zone_heatmap.png", dpi=160)
    plt.close()

    pivot = per_group.pivot_table(
        index="mutation_type",
        columns="requested_radius",
        values="median_delta_score",
        aggfunc="mean",
    )
    plt.figure(figsize=(10, 4.5))
    plt.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="median delta_score")
    plt.title("Radius x mutation type median score delta")
    plt.xlabel("requested radius")
    plt.ylabel("mutation type")
    plt.tight_layout()
    plt.savefig(out / "radius_score_delta_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for mutation_type, group in per_group.groupby("mutation_type"):
        g = group.groupby("requested_radius", as_index=False)["cliff_rate"].mean()
        plt.plot(g["requested_radius"], g["cliff_rate"], marker="o", label=str(mutation_type))
    plt.xscale("log", base=2)
    plt.ylim(-0.02, 1.02)
    plt.title("Cliff rate by radius")
    plt.xlabel("requested radius")
    plt.ylabel("cliff_rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "cliff_rate_by_radius.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    for mutation_type, group in per_group.groupby("mutation_type"):
        g = group.groupby("requested_radius", as_index=False)["positive_delta_rate"].mean()
        plt.plot(g["requested_radius"], g["positive_delta_rate"], marker="o", label=f"{mutation_type} positive")
    plt.xscale("log", base=2)
    plt.ylim(-0.02, 1.02)
    plt.title("Per-type positive neighbor rate")
    plt.xlabel("requested radius")
    plt.ylabel("positive_delta_rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "per_type_radius_profiles.png", dpi=160)
    plt.close()


def write_html(df: pd.DataFrame, per_group: pd.DataFrame, per_type: pd.DataFrame, overall: dict[str, Any], out: Path) -> None:
    top = df.sort_values("delta_score", ascending=False).head(200).copy()
    rows = []
    for row in top.to_dict(orient="records"):
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{row.get(col, ''):.6g}</td>" if isinstance(row.get(col), float) else f"<td>{row.get(col, '')}</td>"
                for col in [
                    "sample_id",
                    "base_index",
                    "mutation_type",
                    "requested_radius",
                    "direct_genome_distance",
                    "delta_score",
                    "behavior_distance",
                    "edge_edits",
                    "threshold_edits",
                    "channel_edits",
                    "polarity_edits",
                ]
            )
            + "</tr>"
        )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>D9.0b Direct Genome Landscape Atlas</title>
<style>
body {{ margin: 0; font-family: ui-sans-serif, Segoe UI, Arial; background: #10151d; color: #e8edf5; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
.card {{ background: #17202b; border: 1px solid #2a3a4c; border-radius: 14px; padding: 18px; margin: 16px 0; }}
.verdict {{ font-size: 28px; font-weight: 800; color: #8ddc9a; }}
img {{ width: 100%; max-width: 980px; background: #fff; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
td, th {{ border-bottom: 1px solid #2a3a4c; padding: 6px 8px; text-align: right; }}
th {{ color: #9fb1c7; }}
td:nth-child(3), th:nth-child(3) {{ text-align: left; }}
code {{ color: #ffc66d; }}
</style>
<main>
<h1>D9.0b Direct Genome Landscape Atlas</h1>
<div class="card"><div class="verdict">{overall["verdict"]}</div>
<p>Rows: {overall["rows"]} | Bases: {overall["base_count"]} | Radii: {overall["radii"]}</p>
<p>Type classes: <code>{json.dumps(overall["type_classes"], sort_keys=True)}</code></p></div>
<div class="card"><h2>Local Zone Heatmap</h2><img src="local_zone_heatmap.png"></div>
<div class="card"><h2>Radius Score Delta</h2><img src="radius_score_delta_heatmap.png"></div>
<div class="card"><h2>Cliff Rate</h2><img src="cliff_rate_by_radius.png"></div>
<div class="card"><h2>Per-Type Profiles</h2><img src="per_type_radius_profiles.png"></div>
<div class="card"><h2>Per-Type Summary</h2>{per_type.to_html(index=False, classes="data")}</div>
<div class="card"><h2>Best Neighbor Zones</h2><table><thead><tr>
<th>sample</th><th>base</th><th>type</th><th>radius</th><th>dist</th><th>delta</th><th>behavior</th><th>edge</th><th>thr</th><th>chan</th><th>pol</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
</main>"""
    (out / "direct_landscape_atlas.html").write_text(html, encoding="utf-8")


def write_report(
    overall: dict[str, Any],
    per_type: pd.DataFrame,
    out: Path,
    report: Path,
    run_meta: dict[str, Any],
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    type_cols = [
        "mutation_type",
        "classification",
        "n",
        "low_radius_cliff_rate",
        "low_radius_positive_rate",
        "best_of_9_low_radius",
        "rho_radius_behavior",
    ]
    type_table = markdown_table(per_type[type_cols])
    lines = [
        "# Phase D9.0b Direct Genome Landscape Audit",
        "",
        f"Verdict: **{overall['verdict']}**",
        "",
        "## Summary",
        "",
        f"- Rows analyzed: `{overall['rows']}`",
        f"- Base checkpoints: `{overall['base_count']}`",
        f"- Radii: `{overall['radii']}`",
        f"- Probe mode: `{run_meta.get('mode', 'unknown')}`",
        f"- Eval length: `{run_meta.get('eval_len', 'unknown')}`",
        f"- Samples per type/radius/base: `{run_meta.get('samples_per_type', 'unknown')}`",
        f"- Overall median delta: `{overall['overall_delta_median']:.6f}`",
        f"- Overall cliff rate: `{overall['overall_cliff_rate']:.3f}`",
        f"- Overall positive rate: `{overall['overall_positive_rate']:.3f}`",
        f"- Psi available: `{overall['psi_available']}`",
        "",
        "## Per-Type Classification",
        "",
        type_table,
        "",
        "## Visual Artifacts",
        "",
        f"- `{out / 'direct_landscape_atlas.html'}`",
        f"- `{out / 'local_zone_heatmap.png'}`",
        f"- `{out / 'radius_score_delta_heatmap.png'}`",
        f"- `{out / 'cliff_rate_by_radius.png'}`",
        f"- `{out / 'per_type_radius_profiles.png'}`",
        "",
        "## Caveats",
        "",
        "- D9.0b freezes projection and mutates only the persisted core genome.",
        "- A type-split verdict should not be collapsed into a global failure.",
        "- Short eval-length smoke/medium runs are diagnostic; use the full command for final evidence.",
        "- Random archive-pair baseline is marked unavailable unless supplied by a later extended run.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(c) for c in df.columns]
    rows = []
    for row in df.to_dict(orient="records"):
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append(values)
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    out.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        summary = {"verdict": VERDICT_INFRA, "reason": f"missing input {args.input}"}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise SystemExit(f"missing input: {args.input}")

    df = pd.read_csv(args.input)
    if len(df) == 0:
        summary = {"verdict": VERDICT_INFRA, "reason": "empty samples.csv"}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise SystemExit("empty samples.csv")

    run_meta_path = args.input.parent / "run_meta.json"
    run_meta: dict[str, Any] = {}
    if run_meta_path.exists():
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    per_group, per_type, overall = summarize(df, args.cliff_delta)
    per_group.to_csv(args.out / "per_radius_type_summary.csv", index=False)
    per_type.to_csv(args.out / "per_type_summary.csv", index=False)
    best = df.sort_values("delta_score", ascending=False).head(100)
    best.to_csv(args.out / "best_neighbor_zones.csv", index=False)
    save_figures(df, per_group, args.out)
    if not args.skip_html:
        write_html(df, per_group, per_type, overall, args.out)
    summary = {
        **overall,
        "run_meta": run_meta,
        "input": str(args.input),
        "analysis_out": str(args.out),
        "report": str(args.report),
    }
    (args.out / "summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    write_report(overall, per_type, args.out, args.report, run_meta)
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
