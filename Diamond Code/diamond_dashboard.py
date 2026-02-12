"""
Diamond Code Real-Time Dashboard

Streamlit dashboard for visualizing Diamond Code training in real-time.
Uses @st.fragment for smooth auto-refresh without scroll/widget resets.

Launch:
    python -m streamlit run diamond_dashboard.py -- --log logs/swarm/current.log --refresh 5
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from datetime import timedelta
import argparse

# Add Diamond Code to path
sys.path.insert(0, str(Path(__file__).parent))

from diamond_log_parser import read_and_parse_log, read_masks_from_log, read_masks_from_json


# ============================================================================
# Configuration
# ============================================================================

DIAMOND_BLUE = '#B9F2FF'
DIAMOND_PLATINUM = '#E5E4E2'
DIAMOND_SILVER = '#C0C0C0'

parser = argparse.ArgumentParser(description='Diamond Code Dashboard')
parser.add_argument('--log', type=str, default='logs/swarm/current.log',
                   help='Path to training log file')
parser.add_argument('--refresh', type=int, default=5,
                   help='Auto-refresh interval in seconds (0 to disable)')
args, _ = parser.parse_known_args()


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Diamond Code Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Session State
# ============================================================================

if 'file_position' not in st.session_state:
    st.session_state.file_position = 0
if 'parsed_rows' not in st.session_state:
    st.session_state.parsed_rows = []
if 'max_rows' not in st.session_state:
    st.session_state.max_rows = 5000
if 'masks_data' not in st.session_state:
    st.session_state.masks_data = None


# ============================================================================
# Sidebar (static, outside fragment -- survives refresh)
# ============================================================================

st.sidebar.title("Diamond Code Dashboard")
st.sidebar.markdown("Real-time training visualization")
st.sidebar.markdown("---")

log_path = st.sidebar.text_input("Log File Path", value=args.log, key="sb_log")
refresh_interval = st.sidebar.number_input(
    "Refresh (seconds)", min_value=0, max_value=60, value=args.refresh, key="sb_refresh"
)
max_rows = st.sidebar.number_input(
    "Max Rows", min_value=100, max_value=50000,
    value=st.session_state.max_rows, step=1000, key="sb_maxrows"
)
st.session_state.max_rows = max_rows

st.sidebar.markdown("---")
if os.path.exists(log_path):
    st.sidebar.success("Log file found")
    st.sidebar.info(f"Size: {os.path.getsize(log_path):,} bytes")
else:
    st.sidebar.error("Log file not found")

st.sidebar.markdown("---")
if st.sidebar.button("Refresh Now", key="sb_refresh_btn"):
    st.rerun()

smooth_window = st.sidebar.slider("Smoothing", min_value=1, max_value=200, value=1, key="sb_smooth")
st.session_state._smooth_window = smooth_window

# ============================================================================
# Live Controls (read/write controls.json)
# ============================================================================

_controls_path = str(Path(log_path).parent / "controls.json")

def _load_controls_into_session():
    """Read controls.json and populate session_state defaults (once)."""
    try:
        with open(_controls_path, 'r') as f:
            ctrl = json.load(f)
    except Exception:
        ctrl = {}
    st.session_state.ctrl_lr = ctrl.get('lr', 0.0001)
    st.session_state.ctrl_tt = ctrl.get('think_ticks', 0)
    st.session_state.ctrl_ckpt = ctrl.get('checkpoint_every', 50)
    st.session_state.ctrl_weights = ctrl.get('data_weights', {})
    st.session_state.ctrl_being_states = ctrl.get('being_states', {})
    st.session_state.ctrl_loaded = True

if 'ctrl_loaded' not in st.session_state:
    _load_controls_into_session()

st.sidebar.markdown("---")
st.sidebar.subheader("Live Controls")

ctrl_lr = st.sidebar.number_input(
    "Learning Rate", min_value=0.000001, max_value=0.1,
    value=st.session_state.ctrl_lr, step=0.00001,
    format="%.6f", key="wgt_lr"
)
ctrl_tt = st.sidebar.number_input(
    "Think Ticks", min_value=0, max_value=20,
    value=st.session_state.ctrl_tt, step=1, key="wgt_tt"
)
ctrl_ckpt = st.sidebar.number_input(
    "Checkpoint Every", min_value=10, max_value=5000,
    value=st.session_state.ctrl_ckpt, step=10, key="wgt_ckpt"
)

# Data source checkboxes (scan traindat directory)
_traindat_dir = Path(__file__).parent / "data" / "traindat"
_traindat_files = sorted(f.name for f in _traindat_dir.iterdir() if f.suffix == '.traindat' and f.is_file()) if _traindat_dir.exists() else []
_ctrl_weights = st.session_state.get('ctrl_weights', {})

if _traindat_files:
    st.sidebar.markdown("**Data Sources**")
    ctrl_weight_values = {}
    for fname in _traindat_files:
        is_on = _ctrl_weights.get(fname, 0.0) > 0.0
        safe_key = f"wgt_dw_{fname.replace('.', '_')}"
        checked = st.sidebar.checkbox(fname.replace('.traindat', ''), value=is_on, key=safe_key)
        ctrl_weight_values[fname] = 1.0 if checked else 0.0

# Per-being state controls (null/active/frozen)
_being_states_from_ctrl = st.session_state.get('ctrl_being_states', {})
_NUM_BEINGS = 7  # TODO: read from log header if available

st.sidebar.markdown("---")
st.sidebar.markdown("**Being States**")
_being_state_icons = {'null': ':gray[OFF]', 'active': ':green[TRAIN]', 'frozen': ':blue[FROZEN]'}
_being_new_states = {}
_being_clears = set()
for _bi in range(_NUM_BEINGS):
    _bstate = _being_states_from_ctrl.get(str(_bi), 'null')
    _cols = st.sidebar.columns([2, 2, 1])
    with _cols[0]:
        _icon = _being_state_icons.get(_bstate, ':gray[OFF]')
        st.markdown(f"B{_bi} {_icon}")
    with _cols[1]:
        _is_on = _bstate in ('active',)
        _cb = st.checkbox("Train", value=_is_on, key=f"being_cb_{_bi}")
        _being_new_states[_bi] = (_cb, _bstate)
    with _cols[2]:
        if st.button("X", key=f"being_clr_{_bi}", help=f"Reset B{_bi} to null"):
            _being_clears.add(_bi)

def _compute_being_states():
    """Derive being_states dict from checkboxes and clear buttons."""
    states = {}
    for bi in range(_NUM_BEINGS):
        if bi in _being_clears:
            states[str(bi)] = 'null'
            continue
        cb_on, old_state = _being_new_states[bi]
        if cb_on:
            states[str(bi)] = 'active'
        elif old_state == 'active':
            # Was active, now unticked -> frozen
            states[str(bi)] = 'frozen'
        else:
            # Was null or frozen, checkbox off -> keep state
            states[str(bi)] = old_state
    return states

_ctrl_col1, _ctrl_col2 = st.sidebar.columns(2)
with _ctrl_col1:
    if st.button("Apply", key="ctrl_apply", width='stretch'):
        _final_being_states = _compute_being_states()
        new_controls = {
            "lr": ctrl_lr,
            "think_ticks": int(ctrl_tt),
            "checkpoint_every": int(ctrl_ckpt),
            "being_states": _final_being_states,
            "data_weights": ctrl_weight_values if _traindat_files else {},
        }
        try:
            _tmp_path = _controls_path + '.tmp'
            with open(_tmp_path, 'w') as f:
                json.dump(new_controls, f, indent=2)
            os.replace(_tmp_path, _controls_path)
            st.session_state.ctrl_being_states = _final_being_states
            st.sidebar.success("Applied!")
        except Exception as e:
            st.sidebar.error(f"Write failed: {e}")

with _ctrl_col2:
    if st.button("Reload", key="ctrl_reload", width='stretch'):
        _load_controls_into_session()
        st.rerun()

_sidebar_mask_placeholder = st.sidebar.empty()


# ============================================================================
# Main Dashboard (auto-refreshing fragment -- no scroll reset)
# ============================================================================

_refresh_td = timedelta(seconds=refresh_interval) if refresh_interval > 0 else None


@st.fragment(run_every=_refresh_td)
def dashboard_content():
    # ------------------------------------------------------------------
    # Data loading (single log: current.log has both basic + full eval lines)
    # ------------------------------------------------------------------
    new_rows, new_position = read_and_parse_log(log_path, st.session_state.file_position)
    if new_rows:
        st.session_state.parsed_rows.extend(new_rows)
        st.session_state.file_position = new_position
        if len(st.session_state.parsed_rows) > max_rows:
            st.session_state.parsed_rows = st.session_state.parsed_rows[-max_rows:]

    df = pd.DataFrame(st.session_state.parsed_rows) if st.session_state.parsed_rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Header metrics
    # ------------------------------------------------------------------
    st.title("Diamond Code - Real-Time Training Dashboard")

    # Detect full_view mode from masks data
    _fv_masks = st.session_state.get('masks_data')
    _full_view = _fv_masks.get('full_view', False) if _fv_masks else False

    if _full_view:
        with _sidebar_mask_placeholder.container():
            st.markdown("**Full View Mode**")
            _hidden_dims = _fv_masks.get('hidden_dims', [])
            if _hidden_dims:
                st.metric("Resolution", f"{max(_hidden_dims)}→{min(_hidden_dims)}")
            _being_keys_sb = [k for k in _fv_masks if k.startswith('being_')]
            st.metric("Beings", f"{len(_being_keys_sb)}")
            st.metric("Input", f"{_fv_masks.get('num_bits', 256)} bits (all)")
    elif not df.empty and 'min_cov' in df.columns:
        # Use last row that has eval data (skip mini log lines with NaN)
        eval_rows = df.dropna(subset=['min_cov'])
        if not eval_rows.empty:
            latest_eval = eval_rows.iloc[-1]
            with _sidebar_mask_placeholder.container():
                st.markdown("**Receptive Field Stats**")
                st.metric("Min Coverage", f"{int(latest_eval.get('min_cov', 0))}")
                st.metric("Max Coverage", f"{int(latest_eval.get('max_cov', 0))}")
                st.metric("Mask Diversity", f"{latest_eval.get('mask_div', 0):.3f}")

    if not df.empty:
        # Use train rows for header metrics (ignore eval spikes)
        has_eval_flag = 'is_eval' in df.columns
        df_hdr = df[df['is_eval'] < 0.5] if has_eval_flag else df
        if df_hdr.empty:
            df_hdr = df  # fallback if no train rows yet

        latest_step = df_hdr['step'].max()
        latest_loss = df_hdr.loc[df_hdr['step'] == latest_step, 'loss'].values[0]
        latest_acc = 0.0
        latest_oracle = 0.0
        for acc_col in ('bit_acc', 'train_bacc', 'acc'):
            if acc_col in df_hdr.columns:
                acc_rows = df_hdr.dropna(subset=[acc_col])
                if not acc_rows.empty:
                    latest_acc = acc_rows.iloc[-1][acc_col]
                    break
        if 'oracle' in df_hdr.columns:
            oracle_rows = df_hdr.dropna(subset=['oracle'])
            if not oracle_rows.empty:
                latest_oracle = oracle_rows.iloc[-1]['oracle']

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Step", f"{latest_step:,}")
        c2.metric("Loss", f"{latest_loss:.4f}")
        c3.metric("Bit Acc", f"{latest_acc*100:.1f}%" if latest_acc > 0 else "N/A")
        c4.metric("Oracle", f"{latest_oracle*100:.1f}%" if latest_oracle > 0 else "N/A")
        c5.metric("Log Lines", f"{len(df):,}")
    else:
        st.info("Waiting for training data...")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Loss and Accuracy
    # ------------------------------------------------------------------
    if not df.empty and 'loss' in df.columns:
        st.subheader("Loss and Accuracy")

        fig = go.Figure()
        sw = st.session_state.get('_smooth_window', 1)

        # Split train vs eval rows (parser sets is_eval=1.0 on EVAL lines)
        has_eval_col = 'is_eval' in df.columns
        if has_eval_col:
            df_train = df[df['is_eval'] < 0.5]
            df_eval_loss = df[df['is_eval'] > 0.5]
        else:
            df_train = df
            df_eval_loss = pd.DataFrame()

        # Training loss line (orange)
        if not df_train.empty:
            loss_raw = df_train['loss']
            if sw > 1:
                loss_smooth = loss_raw.rolling(sw, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=df_train['step'], y=loss_raw, mode='lines', name='Loss (raw)',
                    line=dict(color='rgba(255,136,0,0.2)', width=1), yaxis='y1',
                    showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(
                    x=df_train['step'], y=loss_smooth, mode='lines', name='Train Loss',
                    line=dict(color='#FF8800', width=2), yaxis='y1'))
            else:
                fig.add_trace(go.Scatter(
                    x=df_train['step'], y=loss_raw, mode='lines', name='Train Loss',
                    line=dict(color='#FF8800', width=2), yaxis='y1'))

        # Eval loss (separate series, red markers)
        if not df_eval_loss.empty:
            fig.add_trace(go.Scatter(
                x=df_eval_loss['step'], y=df_eval_loss['loss'],
                mode='markers+lines', name='Eval Loss',
                marker=dict(color='#FF4444', size=6, symbol='x'),
                line=dict(color='#FF4444', width=1.5, dash='dot'), yaxis='y1'))

        # Bit accuracy (train rows only — eval rows have their own acc)
        acc_col = 'bit_acc' if 'bit_acc' in df_train.columns else ('train_bacc' if 'train_bacc' in df_train.columns else ('acc' if 'acc' in df_train.columns else None))
        if acc_col and not df_train.empty:
            df_acc = df_train.dropna(subset=[acc_col])
            if not df_acc.empty:
                acc_raw = df_acc[acc_col]
                acc_y = acc_raw.rolling(sw, min_periods=1).mean() if sw > 1 else acc_raw
                fig.add_trace(go.Scatter(
                    x=df_acc['step'], y=acc_y,
                    mode='markers+lines', name='Bit Acc',
                    marker=dict(color='#00BBFF', size=5, symbol='diamond'),
                    line=dict(color='#00BBFF', width=2), yaxis='y2'))

        # Eval bit accuracy (if available)
        if not df_eval_loss.empty and acc_col and acc_col in df_eval_loss.columns:
            df_eval_acc = df_eval_loss.dropna(subset=[acc_col])
            if not df_eval_acc.empty:
                fig.add_trace(go.Scatter(
                    x=df_eval_acc['step'], y=df_eval_acc[acc_col],
                    mode='markers', name='Eval Acc',
                    marker=dict(color='#FF6666', size=7, symbol='diamond-open'),
                    yaxis='y2'))

        # Oracle accuracy overlay (train rows)
        if 'oracle' in df_train.columns and not df_train.empty:
            df_oracle = df_train.dropna(subset=['oracle'])
            if not df_oracle.empty:
                oracle_raw = df_oracle['oracle']
                oracle_y = oracle_raw.rolling(sw, min_periods=1).mean() if sw > 1 else oracle_raw
                fig.add_trace(go.Scatter(
                    x=df_oracle['step'], y=oracle_y,
                    mode='markers+lines', name='Oracle',
                    marker=dict(color='#00FF88', size=4),
                    line=dict(color='#00FF88', width=1.5, dash='dot'), yaxis='y2'))

        fig.update_layout(
            xaxis=dict(title="Step"),
            yaxis=dict(title="Loss", title_font=dict(color='#FF8800')),
            yaxis2=dict(title="Accuracy", title_font=dict(color='#00BBFF'),
                        overlaying='y', side='right'),
            hovermode='x unified', height=350, template='plotly_dark',
            legend=dict(x=0.01, y=0.99), margin=dict(t=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Swarm Diagnostics
    # ------------------------------------------------------------------
    # Filter to train-only rows with full metrics (bit_acc present, not EVAL lines)
    if not df.empty and 'bit_acc' in df.columns:
        df_eval = df.dropna(subset=['bit_acc'])
        if 'is_eval' in df_eval.columns:
            df_eval = df_eval[df_eval['is_eval'] < 0.5]
    else:
        df_eval = df

    has_swarm = any(c in df_eval.columns for c in ['circular_spread', 'coverage', 'being_0']) if not df_eval.empty else False

    if has_swarm:
        st.markdown("---")
        st.subheader("Swarm Diagnostics")

        col1, col2 = st.columns(2)

        with col1:
            if 'circular_spread' in df_eval.columns:
                fig = go.Figure()
                fig.add_hline(y=16, line_dash="dash", line_color="green",
                              annotation_text="Good", annotation_position="right")
                fig.add_trace(go.Scatter(
                    x=df_eval['step'], y=df_eval['circular_spread'], fill='tozeroy',
                    name='Circular Spread', line=dict(color='#00D9FF', width=2),
                    fillcolor='rgba(0, 217, 255, 0.2)'
                ))
                fig.update_layout(
                    title="Circular Pointer Spread", xaxis_title="Step",
                    yaxis_title="Mean Circular Distance",
                    template="plotly_dark", height=280, hovermode='x unified',
                    margin=dict(t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'coverage' in df_eval.columns and 'clustering' in df_eval.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_eval['step'], y=df_eval['coverage'], mode='lines',
                    name='Coverage', line=dict(color='#7DFF8C', width=2), yaxis='y1'
                ))
                fig.add_trace(go.Scatter(
                    x=df_eval['step'], y=df_eval['clustering'], mode='lines',
                    name='Clustering', line=dict(color='#FF6B9D', width=2, dash='dot'),
                    yaxis='y2'
                ))
                fig.update_layout(
                    title="Coverage vs Clustering",
                    yaxis=dict(title="Coverage (avg bits/token)"),
                    yaxis2=dict(title="Clustering", overlaying='y', side='right'),
                    hovermode='x unified', height=280, template='plotly_dark',
                    legend=dict(x=0.01, y=0.99), margin=dict(t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Per-Being Performance
    # ------------------------------------------------------------------
    being_cols = sorted(
        [c for c in df_eval.columns if c.startswith('being_') and c[6:].isdigit()],
        key=lambda c: int(c.split('_')[1])
    ) if not df_eval.empty else []

    if len(being_cols) >= 2:
        st.markdown("---")
        st.subheader(f"Per-Being Performance ({len(being_cols)} beings)")

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            being_colors = px.colors.qualitative.Set3[:len(being_cols)]
            lw = 1 if len(being_cols) > 6 else 2
            opacity = 0.4 if len(being_cols) > 6 else 1.0

            for i, cn in enumerate(being_cols):
                fig.add_trace(go.Scatter(
                    x=df_eval['step'], y=df_eval[cn], mode='lines', name=f'B{i}',
                    line=dict(color=being_colors[i % len(being_colors)], width=lw),
                    opacity=opacity, showlegend=(len(being_cols) <= 10),
                ))
            if 'oracle' in df_eval.columns:
                fig.add_trace(go.Scatter(
                    x=df_eval['step'], y=df_eval['oracle'], mode='lines',
                    name='Oracle', line=dict(color='#00FF00', width=2, dash='dash')
                ))
            if 'bit_oracle' in df_eval.columns:
                fig.add_trace(go.Scatter(
                    x=df_eval['step'], y=df_eval['bit_oracle'], mode='lines',
                    name='Bit Oracle', line=dict(color='#FFFF00', width=2, dash='dot')
                ))
            if 'bit_acc' in df_eval.columns:
                fig.add_trace(go.Scatter(
                    x=df_eval['step'], y=df_eval['bit_acc'], mode='lines+markers',
                    name='Bit Acc (ensemble)', line=dict(color='#FF4444', width=3),
                    marker=dict(size=4)
                ))

            fig.update_layout(
                title="Oracle Gap (yellow vs red = think_ticks signal)", xaxis_title="Step",
                yaxis_title="Accuracy", template="plotly_dark",
                height=320, hovermode='x unified', legend=dict(x=1.02, y=1),
                margin=dict(t=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'ensemble_benefit' in df_eval.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_eval['step'], y=df_eval['ensemble_benefit'], mode='lines',
                    name='Ensemble Benefit', line=dict(color='#FFD700', width=2),
                ))
                fig.add_hline(y=0, line_color="white", line_width=1)
                fig.update_layout(
                    title="Ensemble Benefit", xaxis_title="Step",
                    yaxis_title="Benefit", template="plotly_dark",
                    height=320, hovermode='x unified', margin=dict(t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Specialization & Jump Rates
    # ------------------------------------------------------------------
    jump_cols = sorted(
        [c for c in df_eval.columns if c.startswith('jump_') and c[5:].isdigit()],
        key=lambda c: int(c.split('_')[1])
    ) if not df_eval.empty else []

    if 'specialization' in df_eval.columns or len(jump_cols) >= 2:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if 'specialization' in df_eval.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_eval['step'], y=df_eval['specialization'], fill='tozeroy',
                    name='Specialization', line=dict(color='#B19CD9', width=2),
                    fillcolor='rgba(177, 156, 217, 0.2)'
                ))
                fig.update_layout(
                    title="Specialization Score", xaxis_title="Step",
                    template="plotly_dark", height=280, hovermode='x unified',
                    margin=dict(t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if len(jump_cols) >= 2:
                fig = go.Figure()
                jcolors = px.colors.qualitative.Set3[:len(jump_cols)]
                jlw = 1 if len(jump_cols) > 6 else 2
                for i, cn in enumerate(jump_cols):
                    fig.add_trace(go.Scatter(
                        x=df_eval['step'], y=df_eval[cn], mode='lines', name=f'B{i} Jump',
                        line=dict(color=jcolors[i % len(jcolors)], width=jlw),
                        showlegend=(len(jump_cols) <= 10),
                    ))
                fig.update_layout(
                    title=f"Per-Being Jump Rates ({len(jump_cols)} beings)",
                    xaxis_title="Step", yaxis_title="Jump Rate",
                    template="plotly_dark", height=280, hovermode='x unified',
                    legend=dict(x=1.02, y=1), margin=dict(t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Per-Bit Accuracy
    # ------------------------------------------------------------------
    bit_cols = sorted(
        [c for c in df_eval.columns if c.startswith('bit') and c[3:].isdigit()],
        key=lambda c: int(c[3:])
    ) if not df_eval.empty else []

    if bit_cols:
        st.markdown("---")
        n_bits = len(bit_cols)
        st.subheader(f"Per-Bit Accuracy ({n_bits} bits)")

        col1, col2 = st.columns(2)

        with col1:
            if n_bits > 16:
                # Heatmap for many bits
                step_sample = df_eval['step'].values[::max(1, len(df_eval)//100)]
                bit_matrix = np.array([
                    df_eval[cn].values[::max(1, len(df_eval)//100)] for cn in bit_cols
                ])
                avg_row = bit_matrix.mean(axis=0, keepdims=True)
                bit_matrix = np.vstack([avg_row, bit_matrix])
                fig = go.Figure(data=go.Heatmap(
                    z=bit_matrix, x=step_sample,
                    y=['AVG'] + [f'bit{i}' for i in range(n_bits)],
                    colorscale='Viridis', zmin=0, zmax=1,
                    colorbar_title='Accuracy'
                ))
            else:
                fig = go.Figure()
                colors_16 = ['#00D9FF', '#FF6B9D', '#B19CD9', '#FFD700',
                             '#7DFF8C', '#FFB000', '#FF4444', '#44FFFF',
                             '#FF69B4', '#00FF7F', '#DDA0DD', '#F0E68C',
                             '#87CEEB', '#FFA07A', '#98FB98', '#D8BFD8']
                for idx, cn in enumerate(bit_cols):
                    fig.add_trace(go.Scatter(
                        x=df_eval['step'], y=df_eval[cn], mode='lines', name=cn,
                        line=dict(color=colors_16[idx % 16], width=2)
                    ))

            fig.update_layout(
                title="Per-Bit Accuracy Over Time", xaxis_title="Step",
                template="plotly_dark",
                height=350 if n_bits <= 16 else 450,
                hovermode='x unified' if n_bits <= 16 else 'closest',
                margin=dict(t=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if not df_eval.empty:
                latest = df_eval.iloc[-1]
                bit_accs = [latest.get(c, 0.0) for c in bit_cols]
                fig = go.Figure(data=[go.Bar(
                    x=list(range(n_bits)), y=bit_accs,
                    marker_color='#00D9FF',
                )])
                fig.update_layout(
                    title=f"Per-Bit Accuracy (Step {int(latest['step'])})",
                    yaxis=dict(range=[0, 1.1], title="Accuracy"),
                    template="plotly_dark", height=350, margin=dict(t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Bit Ownership / Resolution Map
    # ------------------------------------------------------------------
    if st.session_state.masks_data is None:
        # Prefer JSON (can be live-patched) over log header parsing
        log_dir = str(Path(log_path).parent)
        masks_data = read_masks_from_json(os.path.join(log_dir, 'current_masks.json'))
        if masks_data is None:
            masks_data = read_masks_from_log(log_path)
        if masks_data is not None:
            st.session_state.masks_data = masks_data

    if st.session_state.masks_data is not None and bit_cols:
        st.markdown("---")
        masks_data = st.session_state.masks_data
        n_bits_mask = masks_data.get('num_bits', 256)
        is_full_view = masks_data.get('full_view', False)
        being_keys = sorted(
            [k for k in masks_data if k.startswith('being_')],
            key=lambda k: int(k.split('_')[1])
        )
        num_beings = len(being_keys)

        being_palette = [
            '#FF4444', '#FF8C00', '#FFD700', '#00CC66',
            '#00AAFF', '#8855FF', '#FF55AA', '#44FFCC',
            '#FFAA44', '#AA44FF'
        ]

        latest = df_eval.iloc[-1] if not df_eval.empty else {}
        bit_accs_arr = np.array([latest.get(f'bit{i}', 0.5) for i in range(n_bits_mask)])

        cols_grid = 16 if n_bits_mask >= 64 else 8
        rows_grid = (n_bits_mask + cols_grid - 1) // cols_grid

        if is_full_view:
            # ==========================================================
            # FULL VIEW: Resolution Architecture + Bit Accuracy
            # ==========================================================
            hidden_dims = masks_data.get('hidden_dims', [])
            being_sizes = {}
            for bk in being_keys:
                idx = int(bk.split('_')[1])
                being_sizes[idx] = len(masks_data[bk])

            st.subheader(f"Full View ({num_beings} beings x {n_bits_mask} bits)")

            c1, c2 = st.columns(2)

            with c1:
                # Being Resolution bar chart
                being_labels = []
                h_values = []
                bar_colors = []
                hover_texts = []
                for i in range(num_beings):
                    h_i = hidden_dims[i] if i < len(hidden_dims) else 0
                    k_i = being_sizes.get(i, 0)
                    being_labels.append(f"B{i}")
                    h_values.append(h_i)
                    bar_colors.append(being_palette[i % len(being_palette)])
                    hover_texts.append(f"Being {i}<br>K={k_i} (resolution level)<br>H={h_i}<br>{n_bits_mask} → {h_i}")

                fig = go.Figure(data=[go.Bar(
                    y=being_labels, x=h_values, orientation='h',
                    marker_color=bar_colors, text=[str(h) for h in h_values],
                    textposition='auto', hovertext=hover_texts, hoverinfo='text',
                )])
                fig.update_layout(
                    title="Being Resolution (H_i)",
                    xaxis=dict(title="Hidden Dim", type='log'),
                    yaxis=dict(autorange='reversed'),
                    template="plotly_dark", height=400, margin=dict(t=40),
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                # 3D Bit Accuracy Terrain (accuracy over bits x steps)
                if len(df_eval) >= 3:
                    stride = max(1, len(df_eval) // 80)
                    df_sampled = df_eval.iloc[::stride]
                    steps_3d = df_sampled['step'].values
                    bit_indices = list(range(n_bits_mask))

                    z_raw = np.array([
                        [r.get(f'bit{i}', 0.5) for i in range(n_bits_mask)]
                        for _, r in df_sampled.iterrows()
                    ])

                    # Detrend: subtract per-step mean to reveal per-bit variation
                    row_means = z_raw.mean(axis=1, keepdims=True)
                    z_detrended = z_raw - row_means

                    # Downsample bits: average groups of 8 → 32 columns
                    group_size = 8
                    n_groups = n_bits_mask // group_size
                    z_grouped = z_detrended[:, :n_groups * group_size].reshape(
                        len(z_detrended), n_groups, group_size
                    ).mean(axis=2)
                    group_labels = list(range(0, n_groups * group_size, group_size))

                    z_abs_max = max(0.01, float(np.abs(z_grouped).max()))

                    fig = go.Figure(data=[go.Surface(
                        z=z_grouped, x=group_labels, y=steps_3d,
                        colorscale='RdYlGn', cmin=-z_abs_max, cmax=z_abs_max,
                        colorbar=dict(title='vs avg', len=0.6),
                        opacity=0.85,
                        hidesurface=False,
                        contours=dict(
                            x=dict(show=True, highlight=False, color='rgba(255,255,255,0.3)', width=1),
                            y=dict(show=True, highlight=False, color='rgba(255,255,255,0.3)', width=1),
                        ),
                        hovertemplate='Bits %{x}-%{x}+7<br>Step %{y}<br>vs avg: %{z:+.3f}<extra></extra>',
                    )])
                    latest_mean = float(row_means[-1])
                    fig.update_layout(
                        title=f"Bit Accuracy Terrain (detrended, avg {latest_mean:.1%})",
                        scene=dict(
                            xaxis_title='Bit Group',
                            yaxis_title='Step',
                            zaxis_title='vs Step Mean',
                            zaxis=dict(range=[-z_abs_max, z_abs_max]),
                            camera=dict(eye=dict(x=1.8, y=-1.5, z=1.0)),
                        ),
                        template="plotly_dark", height=500, margin=dict(t=40, l=0, r=0, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback: static heatmap until enough data
                    acc_grid = bit_accs_arr[:n_bits_mask].reshape(rows_grid, cols_grid)
                    fig = go.Figure(data=go.Heatmap(
                        z=acc_grid, colorscale='RdYlGn', zmin=0, zmax=1,
                        colorbar_title='Accuracy', xgap=1, ygap=1,
                    ))
                    fig.update_layout(
                        title=f"Bit Accuracy Map (step {int(latest.get('step', 0))})",
                        xaxis=dict(showticklabels=False), yaxis=dict(autorange='reversed', showticklabels=False),
                        template="plotly_dark", height=400, margin=dict(t=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Per-being resolution summary
            st.markdown("**Per-Being Resolution:**")
            summary_cols = st.columns(min(num_beings, 7))
            for i in range(num_beings):
                h_i = hidden_dims[i] if i < len(hidden_dims) else 0
                k_i = being_sizes.get(i, 0)
                with summary_cols[i % len(summary_cols)]:
                    color = being_palette[i % len(being_palette)]
                    st.markdown(
                        f"<span style='color:{color}'>**B{i}** H={h_i}</span><br>K={k_i} (res)",
                        unsafe_allow_html=True
                    )

        else:
            # ==========================================================
            # MASKED MODE: Bit Ownership Map (original)
            # ==========================================================
            st.subheader(f"Bit Ownership Map ({num_beings} beings x {n_bits_mask} bits)")

            bit_owner = np.full(n_bits_mask, -1, dtype=int)
            bit_all_owners = [[] for _ in range(n_bits_mask)]
            being_sizes = {}
            for bk in being_keys:
                idx = int(bk.split('_')[1])
                bits = masks_data[bk]
                being_sizes[idx] = len(bits)
                for b in bits:
                    if b < n_bits_mask:
                        bit_all_owners[b].append(idx)
                        if bit_owner[b] == -1 or being_sizes[idx] < being_sizes.get(bit_owner[b], 999):
                            bit_owner[b] = idx

            c1, c2 = st.columns(2)

            with c1:
                owner_grid = bit_owner[:n_bits_mask].reshape(rows_grid, cols_grid)
                colorscale = []
                for i in range(num_beings):
                    lo = i / num_beings
                    hi = (i + 1) / num_beings
                    colorscale.append([lo, being_palette[i % len(being_palette)]])
                    colorscale.append([hi, being_palette[i % len(being_palette)]])

                hover_text = []
                for row in range(rows_grid):
                    hover_row = []
                    for col in range(cols_grid):
                        bi = row * cols_grid + col
                        if bi < n_bits_mask:
                            hover_row.append(
                                f"Bit {bi}<br>Owner: B{bit_owner[bi]} (K={being_sizes.get(bit_owner[bi], '?')})<br>"
                                f"All: {bit_all_owners[bi]}<br>Acc: {bit_accs_arr[bi]:.0%}"
                            )
                        else:
                            hover_row.append("")
                    hover_text.append(hover_row)

                fig = go.Figure(data=go.Heatmap(
                    z=owner_grid, text=hover_text, hoverinfo='text',
                    colorscale=colorscale, zmin=0, zmax=num_beings,
                    showscale=False, xgap=1, ygap=1,
                ))
                for i in range(num_beings):
                    fig.add_annotation(
                        x=cols_grid + 0.5, y=i * (rows_grid / num_beings),
                        text=f"B{i} K={being_sizes.get(i, '?')}",
                        showarrow=False,
                        font=dict(color=being_palette[i % len(being_palette)], size=10),
                        xanchor='left',
                    )
                fig.update_layout(
                    title="Bit Ownership (colored by being)",
                    xaxis=dict(showticklabels=False), yaxis=dict(autorange='reversed', showticklabels=False),
                    template="plotly_dark", height=400, margin=dict(t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                acc_grid = bit_accs_arr[:n_bits_mask].reshape(rows_grid, cols_grid)
                hover_acc = []
                for row in range(rows_grid):
                    hr = []
                    for col in range(cols_grid):
                        bi = row * cols_grid + col
                        if bi < n_bits_mask:
                            hr.append(f"Bit {bi}<br>Acc: {bit_accs_arr[bi]:.0%}<br>Owner: B{bit_owner[bi]}")
                        else:
                            hr.append("")
                    hover_acc.append(hr)

                fig = go.Figure(data=go.Heatmap(
                    z=acc_grid, text=hover_acc, hoverinfo='text',
                    colorscale='RdYlGn', zmin=0, zmax=1,
                    colorbar_title='Accuracy', xgap=1, ygap=1,
                ))
                fig.update_layout(
                    title=f"Bit Accuracy Map (step {int(latest.get('step', 0))})",
                    xaxis=dict(showticklabels=False), yaxis=dict(autorange='reversed', showticklabels=False),
                    template="plotly_dark", height=400, margin=dict(t=40)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Per-being summary
            st.markdown("**Per-Being Bit Accuracy:**")
            summary_cols = st.columns(min(num_beings, 7))
            for i, bk in enumerate(being_keys):
                idx = int(bk.split('_')[1])
                bits = masks_data[bk]
                accs_for_being = [bit_accs_arr[b] for b in bits if b < n_bits_mask]
                mean_acc = np.mean(accs_for_being) if accs_for_being else 0
                with summary_cols[i % len(summary_cols)]:
                    color = being_palette[idx % len(being_palette)]
                    st.markdown(
                        f"<span style='color:{color}'>**B{idx}** K={len(bits)}</span><br>Avg: {mean_acc:.1%}",
                        unsafe_allow_html=True
                    )

    # ------------------------------------------------------------------
    # Raw Log Viewer
    # ------------------------------------------------------------------
    st.markdown("---")
    with st.expander("Raw Logs", expanded=False):
        if not df.empty:
            n_lines = st.slider("Lines to show", 10, 200, 30, step=10, key="log_lines")
            recent = st.session_state.parsed_rows[-n_lines:]
            lines = []
            for row in recent:
                line = f"step {int(row['step'])} | loss {row['loss']:.6f}"
                for m in ['bit_acc', 'byte_match', 'specialization', 's_per_step']:
                    if m in row and row[m] is not None:
                        line += f" | {m}={row[m]:.4f}"
                lines.append(line)
            st.code("\n".join(lines), language="text")
            st.caption(f"Showing {len(lines)} of {len(st.session_state.parsed_rows)} total lines")

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    with st.expander("Debug Info", expanded=False):
        st.write(f"**File Position:** {st.session_state.file_position:,} bytes")
        st.write(f"**Rows:** {len(st.session_state.parsed_rows):,}")
        if not df.empty:
            st.write(f"**Columns:** {list(df.columns)}")


# ============================================================================
# Run the dashboard
# ============================================================================

dashboard_content()
