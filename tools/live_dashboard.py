"""
Simple live dashboard for VRAXION logs.

Usage:
  streamlit run tools/live_dashboard.py -- --log logs/current/tournament_phase6.log

Dependencies:
  pip install streamlit plotly pandas
"""

import argparse
import os
import re
from typing import List, Dict

import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

try:
    from streamlit import autorefresh as st_autorefresh
except Exception:
    st_autorefresh = None


LOG_STEP_PATTERN = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[\d\.]+)\s+\|"
    r".*?raw_delta=(?P<raw_delta>[\d\.\-]+).*?shard=(?P<shard_count>[\d\-]+)/(?P<shard_size>[\d\-]+)"
    r"(?:,\s*traction=(?P<traction>[\d\.\-]+))?"
)
LOG_GRAD_PATTERN = re.compile(r"grad_norm\(theta_ptr\)=(?P<grad>[\d\.\+eE\-]+)")


def parse_log(path: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    last_grad = None
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mg = LOG_GRAD_PATTERN.search(line)
            if mg:
                try:
                    last_grad = float(mg.group("grad"))
                except Exception:
                    last_grad = None
                continue
            ms = LOG_STEP_PATTERN.search(line)
            if not ms:
                continue
            try:
                rows.append(
                    {
                        "step": int(ms.group("step")),
                        "loss": float(ms.group("loss")),
                        "raw_delta": float(ms.group("raw_delta")),
                        "shard_count": float(ms.group("shard_count")),
                        "shard_size": float(ms.group("shard_size")),
                        "traction": float(ms.group("traction")) if ms.group("traction") else None,
                        "grad_norm": last_grad,
                    }
                )
            except Exception:
                pass
            last_grad = None
    if not rows:
        return pd.DataFrame()
    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["step"], keep="last")
        .sort_values("step")
        .reset_index(drop=True)
    )
    if "grad_norm" in df.columns and "raw_delta" in df.columns:
        # Derived tension: product scaled to keep reasonable range.
        df["tension"] = df["grad_norm"].fillna(0) * df["raw_delta"].fillna(0) / 100.0
        if df["tension"].notnull().any():
            upper = df["tension"].quantile(0.99)
            if upper > 0:
                df["tension"] = df["tension"].clip(upper=upper)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/current/tournament_phase6.log", help="Path to log file")
    parser.add_argument("--refresh", type=int, default=10, help="Auto-refresh seconds (0 = manual only)")
    args = parser.parse_args()

    st.set_page_config(page_title="VRAXION Live Dashboard", layout="wide")
    st.title("VRAXION Live Dashboard")
    st.caption(f"Log: {os.path.abspath(args.log)} (auto-refresh {args.refresh}s; set 0 to disable)")

    # Auto-refresh (non-flicker). If unavailable or refresh=0, manual button remains.
    if args.refresh > 0 and st_autorefresh:
        st_autorefresh(interval=args.refresh * 1000, key="auto_refresh")

    df = parse_log(args.log)
    if df.empty:
        st.warning("No parsed data yet. Waiting for log lines with shard info...")
        st.stop()

    if st.button("Refresh now"):
        fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if fn:
            fn()

    backend = st.radio("Plot backend", ["matplotlib", "plotly"], index=0, horizontal=True)
    smooth = st.slider("Rolling window (steps)", min_value=1, max_value=200, value=25, step=1)

    def _smooth_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if smooth <= 1:
            return frame
        out = frame.copy()
        for col in ["loss", "shard_count", "shard_size", "traction", "tension", "grad_norm", "raw_delta"]:
            if col in out.columns:
                out[col] = out[col].rolling(window=smooth, min_periods=1).mean()
        return out

    def _normalize(series: pd.Series) -> pd.Series:
        if series is None or series.empty:
            return series
        lo = series.quantile(0.01)
        hi = series.quantile(0.99)
        if hi == lo:
            return series * 0
        clipped = series.clip(lower=lo, upper=hi)
        return (clipped - lo) / (hi - lo)

    df_plot = _smooth_frame(df)

    if backend == "plotly":
        # Composite, normalized for readability; hover shows raw values.
        norm_loss = _normalize(df_plot["loss"])
        norm_tension = _normalize(df_plot.get("tension"))
        norm_traction = _normalize(df_plot.get("traction"))
        frag = df_plot["shard_count"]
        max_frag = max(frag.max(), 1)
        lw_base, lw_span = 1.0, 6.0
        lw = lw_base + lw_span * (frag / max_frag)

        fig = px.line(template="plotly_dark")
        fig.update_layout(
            legend_title=None,
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
        )
        fig.add_scatter(
            x=df_plot["step"],
            y=norm_loss,
            mode="lines",
            name="loss (norm)",
            line=dict(color="#22c55e", width=2.5),  # green
            hovertext=[f"loss={v:.4f}" for v in df_plot["loss"]],
            hoverinfo="text+x",
        )
        # Thickness overlay: use many segments with width driven by shard fragmentation.
        # Fragmentation band: widen area around loss based on shard_count
        frag_norm = frag / max_frag
        band_scale = 0.2  # adjust to make width visible
        upper = norm_loss + band_scale * frag_norm
        lower = norm_loss - band_scale * frag_norm
        fig.add_scatter(
            x=df_plot["step"],
            y=lower,
            mode="lines",
            name="fragmentation",
            line=dict(color="rgba(6,95,70,0.0)", width=0),
            hoverinfo="skip",
            showlegend=False,
        )
        fig.add_scatter(
            x=df_plot["step"],
            y=upper,
            mode="lines",
            name="fragmentation band",
            line=dict(color="#065f46", width=0),
            fill="tonexty",
            fillcolor="rgba(6,95,70,0.35)",
            hoverinfo="skip",
            showlegend=False,
        )
        # Tension
        if norm_tension is not None and not norm_tension.isnull().all():
            fig.add_scatter(
                x=df_plot["step"],
                y=norm_tension,
                mode="lines",
                name="tension (norm)",
                line=dict(color="#ef4444", width=1.5, dash="dot"),
                hovertext=[f"tension={v:.4f}" for v in df_plot["tension"]],
                hoverinfo="text+x",
            )
        # Traction
        if norm_traction is not None and not norm_traction.isnull().all():
            fig.add_scatter(
                x=df_plot["step"],
                y=norm_traction,
                mode="lines+markers",
                name="traction (norm)",
                line=dict(color="#22d3ee", width=1.2),
                marker=dict(size=4),
                hovertext=[f"traction={v:.4f}" for v in df_plot["traction"]],
                hoverinfo="text+x",
            )
        # Shard size/count small band
        fig.add_scatter(
            x=df_plot["step"],
            y=_normalize(frag),
            mode="lines",
            name="shard_count (norm)",
            line=dict(color="#3b82f6", width=1, dash="dash"),
            hovertext=[f"shard_count={int(v)}" for v in frag],
            hoverinfo="text+x",
        )
        fig.add_scatter(
            x=df_plot["step"],
            y=_normalize(df_plot["shard_size"]),
            mode="lines",
            name="shard_size (norm)",
            line=dict(color="#f59e0b", width=1, dash="dashdot"),
            hovertext=[f"shard_size={int(v)}" for v in df_plot["shard_size"]],
            hoverinfo="text+x",
        )
        fig.update_yaxes(title="normalized (0-1)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Single-axis view: loss value; line thickness encodes shard fragmentation (more shards = thicker).
        fig, ax = plt.subplots(figsize=(10, 4.5))
        max_shard = max(df_plot["shard_count"].max(), 1)
        shard_norm = df_plot["shard_count"] / max_shard
        lw_base, lw_span = 1.0, 8.0
        line_w = lw_base + lw_span * shard_norm

        # Baseline thin line for context.
        ax.plot(df_plot["step"], df_plot["loss"], color="tab:red", linewidth=1.0, alpha=0.4, label="loss")

        # Variable-width line segments
        try:
            from matplotlib.collections import LineCollection

            points = list(zip(df_plot["step"], df_plot["loss"]))
            segments = [points[i : i + 2] for i in range(len(points) - 1)]
            lws = line_w.to_list()[:-1]
            lc = LineCollection(segments, linewidths=lws, colors="tab:red", alpha=0.8)
            ax.add_collection(lc)
        except Exception:
            ax.plot(df_plot["step"], df_plot["loss"], color="tab:red", linewidth=2.0, alpha=0.8)

        ax.set_ylabel("loss")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", title="Thickness = fragmentation")

        fig.tight_layout()
        st.pyplot(fig)
    st.caption("Use the Refresh button for manual update; auto-refresh is smooth (no full rerun) when enabled.")


if __name__ == "__main__":
    main()
