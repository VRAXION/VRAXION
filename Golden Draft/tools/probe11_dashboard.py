"""Live dashboard for Probe 11 -- Fibonacci Volume Weight.

Tails probe11_telemetry.jsonl and plots per-ant metrics in real time.
Supports staged multi-ant training with frozen/active ant detection.

Launch:
  streamlit run "S:/AI/Golden Draft/tools/probe11_dashboard.py" --server.port 8511
"""
from __future__ import annotations

import json
import os
from collections import deque

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Probe 11 -- Fib Volume Weight", layout="wide")

TELEMETRY_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "probe11_telemetry.jsonl")
TELEMETRY_PATH = os.environ.get("PROBE11_TELEMETRY", TELEMETRY_DEFAULT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT = dict(
    margin=dict(l=50, r=20, t=50, b=40),
    legend=dict(orientation="h", y=1.02, x=0.01),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,1)",
)


def _ant_colors(n: int) -> list[str]:
    """Cyan-dark -> Cyan-bright -> Magenta-bright gradient for *n* ants."""
    if n <= 1:
        return ["#ff00ff"]
    out: list[str] = []
    for i in range(n):
        t = i / max(1, n - 1)
        if t < 0.5:
            s = t * 2
            r = int(0 + s * 30)
            g = int(80 + s * 175)
            b = int(120 + s * 135)
        else:
            s = (t - 0.5) * 2
            r = int(30 + s * 225)
            g = int(255 - s * 205)
            b = int(255 - s * 50)
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def _detect_frozen(df: pd.DataFrame, n_ants: int, window: int = 20) -> set[int]:
    """Return set of ant indices whose max gnorm in last *window* rows is 0."""
    if "gnorms" not in df.columns or df.empty:
        return set()
    frozen: set[int] = set()
    tail = df["gnorms"].iloc[-window:].tolist()
    for i in range(n_ants):
        max_g = max((row[i] if i < len(row) else 0.0 for row in tail), default=0.0)
        if max_g == 0.0:
            frozen.add(i)
    return frozen


def _add_smoothed(
    fig: go.Figure,
    df: pd.DataFrame,
    x: str,
    y: str,
    label: str,
    color: str,
    window: int = 20,
    frozen: bool = False,
) -> None:
    """Add raw + smoothed traces (active) or single dim trace (frozen)."""
    if frozen:
        smoothed = df[y].rolling(window=window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df[x], y=smoothed,
            name=label + " (frozen)", mode="lines",
            line=dict(color="#888888", width=1.5, dash="dot"),
            opacity=0.45, showlegend=True,
        ))
        return
    # Raw -- faint
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y],
        name=label + " raw", mode="lines",
        line=dict(color=color, width=1),
        opacity=0.15, showlegend=False,
    ))
    # Smoothed -- bold
    smoothed = df[y].rolling(window=window, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df[x], y=smoothed,
        name=label, mode="lines",
        line=dict(color=color, width=3),
    ))


def _simple_plotly(df: pd.DataFrame, x: str, y: str, color: str = "#e879f9",
                   title: str = "", height: int = 300) -> go.Figure:
    """Single-trace plotly chart for scalar metrics."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y], mode="lines",
        line=dict(color=color, width=2), showlegend=False,
    ))
    fig.update_layout(title=title, xaxis_title="Step", height=height, **_PLOTLY_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_telemetry(path: str, tail_n: int = 2000) -> pd.DataFrame:
    """Load JSONL telemetry.  For long runs, only keep last *tail_n* rows."""
    if not os.path.exists(path):
        return pd.DataFrame()
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        # Use deque to keep only tail_n lines for large files
        lines = deque(fh, maxlen=tail_n)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Live dashboard
# ---------------------------------------------------------------------------

@st.fragment(run_every=3)
def live_view() -> None:
    df = load_telemetry(TELEMETRY_PATH)
    if df.empty:
        st.warning(f"Waiting for telemetry at: {TELEMETRY_PATH}")
        return

    max_step = int(df["step"].max())
    active_ants = int(df["active_ants"].iloc[-1]) if "active_ants" in df.columns else None
    total_ants = int(df["total_ants"].iloc[-1]) if "total_ants" in df.columns else None

    # Determine ant count from gnorms array
    n_ants = 0
    if "gnorms" in df.columns and len(df) > 0:
        n_ants = len(df["gnorms"].iloc[-1])
    ring_lens = df["ring_lens"].iloc[-1] if "ring_lens" in df.columns else []
    weights = df["weights"].iloc[-1] if "weights" in df.columns else []
    colors = _ant_colors(n_ants)
    frozen_set = _detect_frozen(df, n_ants)
    has_staged = len(frozen_set) > 0

    # ==================================================================
    # SECTION A: Command Strip
    # ==================================================================
    ant_label = f" | ants {active_ants}/{total_ants}" if active_ants is not None else ""
    st.markdown(f"### Probe 11 -- Fibonacci Volume Weight | step {max_step}{ant_label}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Step", f"{max_step}")
    col2.metric("Loss", f"{df['loss'].iloc[-1]:.4f}")
    col3.metric("Acc (MA100)", f"{df['acc_ma100'].iloc[-1]:.1%}")
    if "gnorm_ratio" in df.columns and df["gnorm_ratio"].notna().any():
        col4.metric("Gnorm Ratio", f"{df['gnorm_ratio'].iloc[-1]:.1f}x")
    elif active_ants is not None:
        col4.metric("Active Ants", f"{active_ants}/{total_ants}")

    # Ant status badges
    if n_ants > 0:
        badges: list[str] = []
        for i in range(n_ants):
            rl = ring_lens[i] if i < len(ring_lens) else "?"
            if i >= (active_ants or n_ants):
                badges.append(f'<span style="color:#666;padding:2px 8px;border:1px solid #444;'
                              f'border-radius:4px;margin-right:6px;">ant[{i}] ring={rl} INACTIVE</span>')
            elif i in frozen_set:
                badges.append(f'<span style="color:#9ca3af;background:#1f2937;padding:2px 8px;'
                              f'border:1px solid #4b5563;border-radius:4px;margin-right:6px;">'
                              f'&#10052; ant[{i}] ring={rl} FROZEN</span>')
            else:
                badges.append(f'<span style="color:#22c55e;background:#052e16;padding:2px 8px;'
                              f'border:1px solid #166534;border-radius:4px;margin-right:6px;">'
                              f'&#9679; ant[{i}] ring={rl} LEARNING</span>')
        st.markdown(" ".join(badges), unsafe_allow_html=True)

    # Staged training banner
    if has_staged:
        frozen_names = ", ".join(f"ant[{i}]" for i in sorted(frozen_set))
        active_names = ", ".join(
            f"ant[{i}]" for i in range(active_ants or n_ants) if i not in frozen_set
        )
        st.info(f"Staged training: {frozen_names} FROZEN, {active_names} LEARNING "
                f"-- residual specialization mode")

    # ==================================================================
    # SECTION B: Primary Signals (aggregate)
    # ==================================================================
    st.subheader("Loss")
    st.plotly_chart(_simple_plotly(df, "step", "loss", "#f97316", "Loss", 280),
                    use_container_width=True)

    # Accuracy trio
    st.subheader("Accuracy")
    acc_cols_map = [("MA100", "acc_ma100", "#22d3ee"), ("MA50", "acc_ma50", "#a78bfa"),
                    ("MA10", "acc_ma10", "#facc15")]
    acc_present = [(label, key, c) for label, key, c in acc_cols_map if key in df.columns]
    if acc_present:
        cols = st.columns(len(acc_present))
        for col, (label, key, color) in zip(cols, acc_present):
            with col:
                val = df[key].iloc[-1]
                st.metric(label, f"{val:.1%}")
                fig = _simple_plotly(df, "step", key, color, "", 220)
                fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.3)
                st.plotly_chart(fig, use_container_width=True)

    # Swarm logit norm
    if "swarm_logit_norm" in df.columns:
        st.subheader("Swarm Logit Norm")
        st.plotly_chart(
            _simple_plotly(df, "step", "swarm_logit_norm", "#06b6d4", "Swarm Logit Norm", 250),
            use_container_width=True,
        )

    # ==================================================================
    # SECTION C: Per-Ant Deep Dive
    # ==================================================================
    if n_ants > 0:
        st.markdown("---")
        st.subheader("Per-Ant Deep Dive")

        # Expand per-ant accuracy MAs (only rows that have them).
        df_acc = df
        has_ant_ma = False
        ant_ma_fields = {}  # {field_name: list of per-ant column names}
        for ma_field in ("ant_accs_ma100", "ant_accs_ma50", "ant_accs_ma10"):
            if ma_field in df.columns:
                has_rows = df[ma_field].apply(lambda x: isinstance(x, list))
                if has_rows.any():
                    if not has_ant_ma:
                        df_acc = df[has_rows].copy()
                        has_ant_ma = True
                    expand = pd.DataFrame(df_acc[ma_field].tolist())
                    col_names = []
                    for ci in range(expand.shape[1]):
                        cname = f"_{ma_field}_{ci}"
                        df_acc[cname] = expand[ci].astype(float).values
                        col_names.append(cname)
                    ant_ma_fields[ma_field] = col_names

        # Expand gnorms / msg_norms into per-ant columns
        gnorm_cols: list[str] = []
        if "gnorms" in df.columns:
            gnorm_expand = pd.DataFrame(df["gnorms"].tolist())
            for ci in range(gnorm_expand.shape[1]):
                cname = f"_gnorm_{ci}"
                df[cname] = gnorm_expand[ci].values
                gnorm_cols.append(cname)

        msg_cols: list[str] = []
        if "msg_norms" in df.columns:
            msg_expand = pd.DataFrame(df["msg_norms"].tolist())
            for ci in range(msg_expand.shape[1]):
                cname = f"_msg_{ci}"
                df[cname] = msg_expand[ci].values
                msg_cols.append(cname)

        # -- Per-Ant Accuracy (3 charts: MA100 / MA50 / MA10, all ants as lines) --
        if has_ant_ma and ant_ma_fields:
            ma_chart_map = [
                ("Per-Ant Accuracy (MA100)", "ant_accs_ma100"),
                ("Per-Ant Accuracy (MA50)", "ant_accs_ma50"),
                ("Per-Ant Accuracy (MA10)", "ant_accs_ma10"),
            ]
            for title, field_key in ma_chart_map:
                if field_key not in ant_ma_fields:
                    continue
                col_names = ant_ma_fields[field_key]
                fig_aa = go.Figure()
                for i, cname in enumerate(col_names):
                    rl = ring_lens[i] if i < len(ring_lens) else "?"
                    lbl = f"ant[{i}] ring={rl}"
                    is_frozen = i in frozen_set
                    _add_smoothed(fig_aa, df_acc, "step", cname, lbl,
                                  colors[i % len(colors)], window=10, frozen=is_frozen)
                fig_aa.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.3)
                fig_aa.update_layout(
                    title=title, xaxis_title="Step", yaxis_title="Accuracy",
                    yaxis_range=[0.3, 1.0], height=300, **_PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_aa, use_container_width=True)

        # -- Per-Ant Gradient Norms --
        if gnorm_cols:
            fig_gn = go.Figure()
            for i, cname in enumerate(gnorm_cols):
                rl = ring_lens[i] if i < len(ring_lens) else "?"
                lbl = f"ant[{i}] ring={rl}"
                is_frozen = i in frozen_set
                _add_smoothed(fig_gn, df, "step", cname, lbl,
                              colors[i % len(colors)], frozen=is_frozen)
            fig_gn.update_layout(
                title="Per-Ant Gradient Norms",
                xaxis_title="Step", yaxis_title="Grad Norm",
                height=350, **_PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_gn, use_container_width=True)

        # -- Per-Ant Message Norms --
        if msg_cols:
            fig_msg = go.Figure()
            for i, cname in enumerate(msg_cols):
                rl = ring_lens[i] if i < len(ring_lens) else "?"
                lbl = f"ant[{i}] ring={rl}"
                is_frozen = i in frozen_set
                _add_smoothed(fig_msg, df, "step", cname, lbl,
                              colors[i % len(colors)], frozen=is_frozen)
            fig_msg.update_layout(
                title="Per-Ant Message Norms",
                xaxis_title="Step", yaxis_title="Msg Norm",
                height=350, **_PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_msg, use_container_width=True)

        # -- Gnorm Ratio with zones --
        if ("gnorm_ratio" in df.columns and df["gnorm_ratio"].notna().any()
                and (active_ants is None or active_ants >= 2)):
            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Scatter(
                x=df["step"], y=df["gnorm_ratio"], mode="lines",
                line=dict(color="#f97316", width=2), showlegend=False,
            ))
            # Zone bands
            fig_ratio.add_hrect(y0=0, y1=2.0, fillcolor="#22c55e", opacity=0.07,
                                line_width=0, annotation_text="healthy",
                                annotation_position="top left")
            fig_ratio.add_hrect(y0=2.0, y1=5.0, fillcolor="#facc15", opacity=0.07,
                                line_width=0, annotation_text="watch",
                                annotation_position="top left")
            fig_ratio.add_hrect(y0=5.0, y1=20.0, fillcolor="#ef4444", opacity=0.07,
                                line_width=0, annotation_text="starvation risk",
                                annotation_position="top left")
            fig_ratio.update_layout(
                title="Gnorm Ratio (ant[0] / ant[-1])",
                xaxis_title="Step", yaxis_title="Ratio",
                height=300, **_PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

        # -- Weight Distribution (bar chart) --
        if weights and ring_lens:
            fig_w = go.Figure()
            bar_colors = []
            opacities = []
            text_labels = []
            for i in range(len(weights)):
                c = colors[i % len(colors)] if i not in frozen_set else "#555555"
                bar_colors.append(c)
                opacities.append(0.3 if i in frozen_set else 1.0)
                rl = ring_lens[i] if i < len(ring_lens) else "?"
                status = "FROZEN" if i in frozen_set else "LEARNING"
                if i >= (active_ants or n_ants):
                    status = "INACTIVE"
                text_labels.append(f"ring={rl}<br>{status}")
            fig_w.add_trace(go.Bar(
                x=[f"ant[{i}]" for i in range(len(weights))],
                y=weights,
                marker_color=bar_colors,
                marker_opacity=opacities,
                text=text_labels,
                textposition="outside",
            ))
            fig_w.update_layout(
                title="Ant Weight Distribution (volume-based)",
                yaxis_title="Weight", height=300, **_PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_w, use_container_width=True)

    # ==================================================================
    # SECTION D: Config Table + Status Footer
    # ==================================================================
    if "weights" in df.columns and "ring_lens" in df.columns and weights:
        st.subheader("Ant Configuration")
        config_rows = []
        for i in range(len(weights)):
            rl = ring_lens[i] if i < len(ring_lens) else "?"
            w = weights[i]
            # Latest gnorm
            latest_g = 0.0
            if "gnorms" in df.columns:
                gn_last = df["gnorms"].iloc[-1]
                if i < len(gn_last):
                    latest_g = gn_last[i]
            # Status
            if i >= (active_ants or n_ants):
                status = "INACTIVE"
            elif i in frozen_set:
                status = "FROZEN"
            else:
                status = "LEARNING"
            config_rows.append({
                "Ant": f"ant[{i}]",
                "Ring Len": rl,
                "Weight": f"{w:.4f}",
                "Status": status,
                "Gnorm": f"{latest_g:.2f}",
            })
        config_df = pd.DataFrame(config_rows)
        st.dataframe(config_df, use_container_width=True, hide_index=True)

    # Status footer
    if max_step > 0:
        last_loss = df["loss"].iloc[-1]
        last_acc = df["acc_ma100"].iloc[-1]
        parts = [f"step {max_step}", f"loss {last_loss:.4f}", f"acc {last_acc:.1%}"]
        if active_ants is not None:
            n_frozen = len(frozen_set)
            n_learning = (active_ants or 0) - n_frozen
            parts.append(f"ants {active_ants}/{total_ants} ({n_learning} learning, {n_frozen} frozen)")
        if "gnorm_ratio" in df.columns and df["gnorm_ratio"].notna().any():
            parts.append(f"gnorm ratio {df['gnorm_ratio'].iloc[-1]:.1f}x")
        st.info("Running -- " + " | ".join(parts))


live_view()
