"""Live dashboard for probe 7 — reads CSV and plots curves in real-time.

Launch:  python -m streamlit run tools/_scratch/probe7_dashboard.py
From:    S:/AI/work/VRAXION_DEV/Golden Draft/
"""

import os
import time

import pandas as pd
import streamlit as st

CSV_PATH = os.path.join(os.path.dirname(__file__), "probe7_live.csv")
STATUS_PATH = os.path.join(os.path.dirname(__file__), "probe7_status.txt")

st.set_page_config(page_title="Probe 7 — Infinite Assoc", layout="wide")


@st.fragment(run_every=2)
def live_view():
    # Status bar.
    status = ""
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH) as f:
            status = f.read().strip()

    if not os.path.exists(CSV_PATH):
        st.warning("Waiting for probe7_live.py to start writing data...")
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        st.warning("CSV not ready yet...")
        return

    if df.empty:
        st.warning("No data yet...")
        return

    max_step = int(df["step"].max())
    st.markdown(f"### Probe 7 — Infinite Assoc &nbsp; | &nbsp; `{status}`")

    # Split by model.
    df_a = df[df["model"] == "A_scale1.0"].copy()
    df_b = df[df["model"] == "B_scale0.01"].copy()

    # ── Metrics row ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    tail = 50
    if len(df_a) >= tail:
        acc_a = df_a["avg_acc_20"].iloc[-1]
        acc_b = df_b["avg_acc_20"].iloc[-1]
        loss_a = df_a["avg_loss_20"].iloc[-1]
        loss_b = df_b["avg_loss_20"].iloc[-1]
    else:
        acc_a = df_a["acc"].astype(float).mean()
        acc_b = df_b["acc"].astype(float).mean()
        loss_a = df_a["loss"].astype(float).mean()
        loss_b = df_b["loss"].astype(float).mean()

    delta = float(acc_a) - float(acc_b)
    col1.metric("A (scale=1.0) acc", f"{float(acc_a):.3f}")
    col2.metric("B (scale=0.01) acc", f"{float(acc_b):.3f}")
    col3.metric("Delta (A-B)", f"{delta:+.3f}", delta_color="normal")
    col4.metric("Step", f"{max_step}")

    # ── Accuracy curves ──────────────────────────────────────────────
    st.subheader("Accuracy (20-step rolling avg)")
    chart_acc = pd.DataFrame({
        "step": df_a["step"].values,
        "A (scale=1.0)": df_a["avg_acc_20"].astype(float).values,
        "B (scale=0.01)": df_b["avg_acc_20"].astype(float).values,
    }).set_index("step")
    st.line_chart(chart_acc, height=350)

    # ── Loss curves ──────────────────────────────────────────────────
    st.subheader("Loss (20-step rolling avg)")
    chart_loss = pd.DataFrame({
        "step": df_a["step"].values,
        "A (scale=1.0)": df_a["avg_loss_20"].astype(float).values,
        "B (scale=0.01)": df_b["avg_loss_20"].astype(float).values,
    }).set_index("step")
    st.line_chart(chart_loss, height=350)

    # ── Delta curve ──────────────────────────────────────────────────
    st.subheader("Delta acc (A - B), 20-step rolling")
    delta_vals = df_a["avg_acc_20"].astype(float).values - df_b["avg_acc_20"].astype(float).values
    chart_delta = pd.DataFrame({
        "step": df_a["step"].values,
        "delta_acc": delta_vals,
    }).set_index("step")
    st.line_chart(chart_delta, height=250)

    # Verdict zone.
    if status == "done":
        st.success(f"Run complete at step {max_step}. Final delta: {delta:+.3f} ({delta*100:+.1f}pp)")
    else:
        st.info(f"Running... step {max_step}")


live_view()
