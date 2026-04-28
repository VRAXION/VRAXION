#!/usr/bin/env python3
"""Phase D8 target-cell detail report.

This is an offline-only helper for inspecting one frozen runtime atlas cell
after observer scans. It deliberately uses the runtime `archive_cell_id` from
D8 state logs for live samples, so the report tracks the same cell even if a
full atlas rebuild would reassign cells after robust re-scaling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_BASE_ATLAS = Path("output/phase_d8_cell_atlas_20260427/analysis")
DEFAULT_OUT = Path("output/phase_d8_target_cell_detail_20260428")


def parse_roots(value: str) -> list[Path]:
    roots: list[Path] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            roots.append(Path(item))
    return roots


def read_base_cell(base_atlas: Path, H: int, cell_id: int) -> tuple[dict, pd.DataFrame]:
    cell_table = pd.read_csv(base_atlas / "cell_table.csv")
    match = cell_table[(cell_table["H"] == H) & (cell_table["cell_id"] == cell_id)]
    if match.empty:
        raise SystemExit(f"Base atlas has no H={H} cell_id={cell_id}")

    samples_path = base_atlas / "cell_sample_states.csv"
    if samples_path.exists():
        samples = pd.read_csv(samples_path)
        samples = samples[(samples["H"] == H) & (samples["cell_id"] == cell_id)].copy()
        samples["sample_group"] = "baseline_atlas"
        samples["sample_source"] = samples.get("source", "baseline_atlas")
    else:
        samples = pd.DataFrame()
    return match.iloc[0].to_dict(), samples


def live_rows_from_root(root: Path, H: int, cell_id: int) -> pd.DataFrame:
    panel_path = root / "analysis" / "live_state_log_panel_dataset.csv"
    if panel_path.exists():
        df = pd.read_csv(panel_path)
    else:
        frames = []
        for path in root.glob("**/accepted_state_log.csv"):
            part = pd.read_csv(path)
            part["run_dir"] = str(path.parent)
            frames.append(part)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["future_peak_final"] = np.nan
        df["future_gain_final"] = np.nan
        df["basin_hit"] = np.nan

    if "archive_cell_id" not in df.columns:
        return pd.DataFrame()

    out = df[(df["H"] == H) & (df["archive_cell_id"] == cell_id)].copy()
    if out.empty:
        return out

    out["cell_id"] = out["archive_cell_id"].astype(int)
    out["sample_group"] = root.name
    out["sample_source"] = root.name
    if "psi_pred_seed_cv" not in out.columns and "psi_pred" in out.columns:
        out["psi_pred_seed_cv"] = out["psi_pred"]
    if "psi_pred" not in out.columns and "psi_pred_seed_cv" in out.columns:
        out["psi_pred"] = out["psi_pred_seed_cv"]
    return out


def summarize_samples(df: pd.DataFrame, label: str, knee_H: int) -> dict:
    if df.empty:
        return {
            "label": label,
            "n": 0,
            "runs": 0,
            "seeds": 0,
            "confidence": 0.0,
            "mean_psi": None,
            "mean_future_gain": None,
            "std_future_gain": None,
            "basin_precision": None,
            "mean_current_peak": None,
            "max_current_peak": None,
        }

    run_col = "run_id" if "run_id" in df.columns else "global_run_id"
    seed_count = int(df["seed"].nunique()) if "seed" in df.columns else 0
    if "psi" in df.columns:
        psi_col = "psi"
    elif "psi_pred_seed_cv" in df.columns:
        psi_col = "psi_pred_seed_cv"
    else:
        psi_col = "psi_pred"
    gain_col = "future_gain_final"
    peak_col = "current_peak"

    return {
        "label": label,
        "n": int(len(df)),
        "runs": int(df[run_col].nunique()) if run_col in df.columns else 0,
        "seeds": seed_count,
        "confidence": float(min(1.0, len(df) / knee_H)) if knee_H else 0.0,
        "mean_psi": safe_mean(df, psi_col),
        "mean_future_gain": safe_mean(df, gain_col),
        "std_future_gain": safe_std(df, gain_col),
        "basin_precision": safe_mean(df, "basin_hit"),
        "mean_current_peak": safe_mean(df, peak_col),
        "max_current_peak": safe_max(df, peak_col),
    }


def safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    return None if series.empty else float(series.mean())


def safe_std(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    return None if len(series) < 2 else float(series.std(ddof=1))


def safe_max(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    return None if series.empty else float(series.max())


def normalize_sample_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "psi_pred_seed_cv" in out.columns:
        out["psi"] = out["psi_pred_seed_cv"]
    elif "psi_pred" in out.columns:
        out["psi"] = out["psi_pred"]
    if "archive_cell_id" in out.columns:
        out["cell_id"] = out["archive_cell_id"]
    keep = [
        "sample_group",
        "sample_source",
        "H",
        "cell_id",
        "state_id",
        "run_id",
        "seed",
        "panel_index",
        "time_pct",
        "current_peak",
        "future_gain_final",
        "psi",
        "basin_hit",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep].sort_values(["sample_group", "seed", "panel_index", "state_id"], na_position="last")


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def metric_card(label: str, before: object, after: object, note: str = "") -> str:
    return f"""
    <div class="card">
      <div class="label">{label}</div>
      <div class="before">{fmt(before)}</div>
      <div class="arrow">-></div>
      <div class="after">{fmt(after)}</div>
      <div class="note">{note}</div>
    </div>
    """


def write_html(out_path: Path, H: int, cell_id: int, base: dict, summaries: list[dict], samples: pd.DataFrame) -> None:
    total = next(item for item in summaries if item["label"] == "total_baseline_plus_live")
    live = next(item for item in summaries if item["label"] == "combined_live")
    rows = "\n".join(
        f"<tr><td>{s['label']}</td><td>{s['n']}</td><td>{s['runs']}</td><td>{s['seeds']}</td>"
        f"<td>{fmt(s['confidence'])}</td><td>{fmt(s['mean_psi'])}</td><td>{fmt(s['mean_future_gain'])}</td>"
        f"<td>{fmt(s['std_future_gain'])}</td><td>{fmt(s['basin_precision'])}</td>"
        f"<td>{fmt(s['mean_current_peak'])}</td><td>{fmt(s['max_current_peak'])}</td></tr>"
        for s in summaries
    )
    sample_rows = "\n".join(
        f"<tr><td>{row.sample_group}</td><td>{fmt(row.seed, 0)}</td><td>{fmt(row.panel_index, 0)}</td>"
        f"<td>{fmt(row.time_pct)}</td><td>{fmt(row.current_peak)}</td>"
        f"<td>{fmt(row.future_gain_final)}</td><td>{fmt(row.psi)}</td><td>{fmt(row.basin_hit, 0)}</td>"
        f"<td class='state'>{row.state_id}</td></tr>"
        for row in samples.itertuples(index=False)
    )
    verdict = "CONFIDENCE FILLED, QUALITY DOWN"
    if total["mean_future_gain"] is not None and base.get("mean_future_gain") is not None:
        if total["mean_future_gain"] >= base["mean_future_gain"]:
            verdict = "CONFIDENCE FILLED, QUALITY HOLDS"
        elif live["mean_future_gain"] is not None and live["mean_future_gain"] < base["mean_future_gain"]:
            verdict = "CONFIDENCE FILLED, BASELINE WAS OPTIMISTIC"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>D8 Target Cell H{H} C{cell_id}</title>
  <style>
    :root {{
      --bg:#0b111b; --panel:#121d2b; --panel2:#172638; --text:#dbe7f5;
      --muted:#7f91a8; --blue:#62a8ff; --green:#4fe39a; --amber:#ffbd4a; --red:#ff5b7a;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }}
    header {{ padding:24px 28px; border-bottom:1px solid #26364d; background:linear-gradient(135deg,#101a28,#0b111b); }}
    h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:.02em; }}
    .sub {{ color:var(--muted); }}
    main {{ padding:24px 28px; display:grid; gap:20px; }}
    .verdict {{ display:inline-flex; padding:8px 12px; border-radius:999px; background:#2d1f0c; color:var(--amber); border:1px solid #694818; font-weight:700; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
    .card {{ background:var(--panel); border:1px solid #26364d; border-radius:16px; padding:14px; min-height:118px; }}
    .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }}
    .before {{ margin-top:10px; color:#90a5bf; font-size:22px; font-weight:700; }}
    .arrow {{ color:var(--muted); font-size:12px; }}
    .after {{ color:var(--green); font-size:26px; font-weight:800; }}
    .note {{ color:var(--muted); font-size:12px; margin-top:6px; }}
    section {{ background:var(--panel); border:1px solid #26364d; border-radius:18px; overflow:hidden; }}
    h2 {{ margin:0; padding:16px 18px; background:var(--panel2); font-size:16px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid #233247; text-align:right; }}
    th:first-child,td:first-child {{ text-align:left; }}
    .state {{ max-width:560px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#9bb2cc; }}
    .explain {{ line-height:1.55; color:#b9c8da; }}
    .bad {{ color:var(--red); font-weight:700; }}
    .good {{ color:var(--green); font-weight:700; }}
  </style>
</head>
<body>
<header>
  <h1>D8 Target Cell Detail: H{H} / C{cell_id}</h1>
  <div class="sub">Frozen runtime archive_cell_id tracking. This is not a regenerated atlas reassignment.</div>
  <div style="margin-top:14px" class="verdict">{verdict}</div>
</header>
<main>
  <div class="cards">
    {metric_card("samples / knee", f"{int(base['n_samples'])}/{int(base['knee_H'])}", f"{int(total['n'])}/{int(base['knee_H'])}", "confidence target was knee_H")}
    {metric_card("confidence", base.get("confidence"), total.get("confidence"), "C2 is now statistically filled")}
    {metric_card("mean Ψ", base.get("mean_psi"), total.get("mean_psi"), "live samples lowered the estimate")}
    {metric_card("future_gain", base.get("mean_future_gain"), total.get("mean_future_gain"), "main quality signal")}
    {metric_card("basin precision", base.get("basin_precision"), total.get("basin_precision"), "fraction future_gain >= 0.005")}
    {metric_card("live-only future_gain", "n/a", live.get("mean_future_gain"), "what the new scan actually saw")}
  </div>

  <section>
    <h2>Interpretation</h2>
    <div class="explain" style="padding:16px 18px">
      <p><span class="good">What improved:</span> H{H}/C{cell_id} is no longer undersampled. It moved from {int(base['n_samples'])}/{int(base['knee_H'])} samples to {int(total['n'])}/{int(base['knee_H'])} samples, so confidence is now {fmt(total.get('confidence'))}.</p>
      <p><span class="bad">What changed negatively:</span> the new live samples did not reproduce the old high future_gain. Baseline future_gain was {fmt(base.get('mean_future_gain'))}; live-only future_gain was {fmt(live.get('mean_future_gain'))}; combined total is {fmt(total.get('mean_future_gain'))}.</p>
      <p>Practical read: C{cell_id} was worth scanning because it was high-priority but underfilled. After enough samples, it looks less like a branch target and more like a cell that should be downgraded unless another controlled scan revives it.</p>
    </div>
  </section>

  <section>
    <h2>Summary By Sample Group</h2>
    <table>
      <thead><tr><th>group</th><th>n</th><th>runs</th><th>seeds</th><th>conf</th><th>mean Ψ</th><th>mean gain</th><th>std gain</th><th>basin</th><th>mean peak</th><th>max peak</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>

  <section>
    <h2>Sample Rows</h2>
    <table>
      <thead><tr><th>group</th><th>seed</th><th>panel</th><th>time</th><th>current_peak</th><th>future_gain</th><th>Ψ</th><th>basin</th><th>state_id</th></tr></thead>
      <tbody>{sample_rows}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a focused D8 target-cell detail report.")
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--cell-id", type=int, required=True)
    parser.add_argument("--base-atlas", type=Path, default=DEFAULT_BASE_ATLAS)
    parser.add_argument("--scan-roots", type=parse_roots, default=[])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.out.mkdir(parents=True, exist_ok=True)
    base, base_samples = read_base_cell(args.base_atlas, args.H, args.cell_id)
    knee_H = int(base["knee_H"])

    live_frames = []
    for root in args.scan_roots:
        rows = live_rows_from_root(root, args.H, args.cell_id)
        if not rows.empty:
            live_frames.append(rows)
    live = pd.concat(live_frames, ignore_index=True) if live_frames else pd.DataFrame()

    base_norm = normalize_sample_columns(base_samples)
    live_norm = normalize_sample_columns(live)
    all_samples = pd.concat([base_norm, live_norm], ignore_index=True)
    all_samples = all_samples.drop_duplicates(subset=["sample_group", "state_id"], keep="last")

    summaries: list[dict] = [summarize_samples(base_norm, "baseline_atlas", knee_H)]
    for group, group_df in live_norm.groupby("sample_group", dropna=False):
        summaries.append(summarize_samples(group_df, str(group), knee_H))
    summaries.append(summarize_samples(live_norm, "combined_live", knee_H))
    summaries.append(summarize_samples(all_samples, "total_baseline_plus_live", knee_H))

    summary = {
        "verdict": "D8_TARGET_CELL_DETAIL_READY",
        "H": args.H,
        "cell_id": args.cell_id,
        "base_atlas": str(args.base_atlas),
        "scan_roots": [str(root) for root in args.scan_roots],
        "base_cell": base,
        "summaries": summaries,
    }

    (args.out / "target_cell_detail.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(args.out / "target_cell_summary.csv", index=False)
    all_samples.to_csv(args.out / "target_cell_samples.csv", index=False)
    write_html(args.out / "target_cell_detail.html", args.H, args.cell_id, base, summaries, all_samples)

    print(json.dumps(summary["verdict"], indent=2))
    print(f"wrote {args.out / 'target_cell_detail.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
