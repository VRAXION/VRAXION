"""Live dashboard for Probe 11 -- Fibonacci Volume Weight.

Tails probe11_telemetry.jsonl and plots per-ant metrics in real time.

Launch:
  streamlit run "S:/AI/Golden Draft/tools/probe11_dashboard.py"
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Probe 11 -- Fib Volume Weight", layout="wide")

TELEMETRY_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "probe11_telemetry.jsonl")
TELEMETRY_PATH = os.environ.get("PROBE11_TELEMETRY", TELEMETRY_DEFAULT)


def load_telemetry(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
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


@st.fragment(run_every=3)
def live_view():
    df = load_telemetry(TELEMETRY_PATH)
    if df.empty:
        st.warning(f"Waiting for telemetry at: {TELEMETRY_PATH}")
        return

    max_step = int(df["step"].max())

    # Header metrics.
    st.markdown(f"### Probe 11 -- Fibonacci Volume Weight | step {max_step}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Step", f"{max_step}")
    col2.metric("Loss", f"{df['loss'].iloc[-1]:.4f}")
    col3.metric("Acc (MA100)", f"{df['acc_ma100'].iloc[-1]:.3f}")
    if "gnorm_ratio" in df.columns:
        col4.metric("Gnorm ratio (big/small)", f"{df['gnorm_ratio'].iloc[-1]:.1f}x")

    # ---- Loss + Accuracy ----
    st.subheader("Loss & Accuracy")
    c1, c2 = st.columns(2)
    with c1:
        st.line_chart(df.set_index("step")[["loss"]], height=300)
    with c2:
        acc_cols = ["acc_ma100"]
        if "acc" in df.columns:
            acc_cols.insert(0, "acc")
        st.line_chart(df.set_index("step")[acc_cols], height=300)

    # ---- Per-ant gradient norms ----
    if "gnorms" in df.columns:
        st.subheader("Per-Ant Gradient Norms")
        gnorm_df = pd.DataFrame(df["gnorms"].tolist(), index=df["step"])
        n_ants = gnorm_df.shape[1]
        gnorm_df.columns = [f"ant[{i}]" for i in range(n_ants)]
        st.line_chart(gnorm_df, height=350)

    # ---- Gnorm ratio ----
    if "gnorm_ratio" in df.columns:
        st.subheader("Gnorm Ratio (ant[0] / ant[-1])")
        st.line_chart(df.set_index("step")[["gnorm_ratio"]], height=250)

    # ---- Per-ant message norms ----
    if "msg_norms" in df.columns:
        st.subheader("Per-Ant Message Norms")
        msg_df = pd.DataFrame(df["msg_norms"].tolist(), index=df["step"])
        n_ants = msg_df.shape[1]
        msg_df.columns = [f"ant[{i}]" for i in range(n_ants)]
        st.line_chart(msg_df, height=300)

    # ---- Swarm logit norm ----
    if "swarm_logit_norm" in df.columns:
        st.subheader("Swarm Logit Norm")
        st.line_chart(df.set_index("step")[["swarm_logit_norm"]], height=250)

    # ---- Weights table ----
    if "weights" in df.columns and "ring_lens" in df.columns:
        st.subheader("Ant Configuration")
        w = df["weights"].iloc[-1]
        rl = df["ring_lens"].iloc[-1]
        config_df = pd.DataFrame({
            "Ant": [f"ant[{i}]" for i in range(len(w))],
            "Ring Len": rl,
            "Weight": [f"{x:.4f}" for x in w],
        })
        st.dataframe(config_df, use_container_width=True, hide_index=True)

    # Status.
    if max_step > 0:
        last_loss = df["loss"].iloc[-1]
        last_acc = df["acc_ma100"].iloc[-1]
        if "gnorm_ratio" in df.columns:
            ratio = df["gnorm_ratio"].iloc[-1]
            st.info(f"Running -- step {max_step} | loss {last_loss:.4f} | acc {last_acc:.3f} | gnorm ratio {ratio:.1f}x")
        else:
            st.info(f"Running -- step {max_step} | loss {last_loss:.4f} | acc {last_acc:.3f}")


live_view()
