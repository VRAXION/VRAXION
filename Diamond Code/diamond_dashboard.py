"""
Diamond Code Real-Time Dashboard - Phase 1 MVP

Streamlit dashboard for visualizing Diamond Code training in real-time.

Launch:
    python -m streamlit run diamond_dashboard.py -- --log logs/diamond/train.log --refresh 10

Phase 1 (MVP) includes:
    - Row 1: Loss and Accuracy charts
    - Row 2: Jump Gate Activation chart
    - Auto-refresh every 10s
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import sys
from pathlib import Path
import argparse

# Add Diamond Code to path
sys.path.insert(0, str(Path(__file__).parent))

from diamond_log_parser import read_and_parse_log


# ============================================================================
# Configuration
# ============================================================================

# Diamond color scheme
DIAMOND_BLUE = '#B9F2FF'
DIAMOND_PLATINUM = '#E5E4E2'
DIAMOND_SILVER = '#C0C0C0'

# Parse command-line args
parser = argparse.ArgumentParser(description='Diamond Code Dashboard')
parser.add_argument('--log', type=str, default='logs/diamond/train.log',
                   help='Path to training log file')
parser.add_argument('--refresh', type=int, default=10,
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
# Session State Initialization
# ============================================================================

if 'file_position' not in st.session_state:
    st.session_state.file_position = 0

if 'parsed_rows' not in st.session_state:
    st.session_state.parsed_rows = []

if 'max_rows' not in st.session_state:
    st.session_state.max_rows = 5000


# ============================================================================
# Sidebar Configuration
# ============================================================================

st.sidebar.title("💎 Diamond Code Dashboard")
st.sidebar.markdown("Real-time training visualization")
st.sidebar.markdown("---")

# Log file path
log_path = st.sidebar.text_input(
    "Log File Path",
    value=args.log,
    help="Path to training log file"
)

# Refresh interval
refresh_interval = st.sidebar.number_input(
    "Refresh Interval (seconds)",
    min_value=0,
    max_value=60,
    value=args.refresh,
    help="0 = manual refresh only"
)

# Max rows to keep
max_rows = st.sidebar.number_input(
    "Max Rows in Memory",
    min_value=100,
    max_value=50000,
    value=st.session_state.max_rows,
    step=1000,
    help="Limit memory usage by keeping only recent rows"
)
st.session_state.max_rows = max_rows

st.sidebar.markdown("---")

# Status indicators
if os.path.exists(log_path):
    st.sidebar.success(f"✓ Log file found")
    file_size = os.path.getsize(log_path)
    st.sidebar.info(f"File size: {file_size:,} bytes")
else:
    st.sidebar.error(f"✗ Log file not found")

st.sidebar.markdown("---")

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()


# ============================================================================
# Auto-Refresh
# ============================================================================

if refresh_interval > 0:
    try:
        from streamlit_autorefresh import st_autorefresh
        # Auto-refresh every N seconds
        st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")
    except ImportError:
        st.sidebar.warning("streamlit-autorefresh not installed. Install with: pip install streamlit-autorefresh")


# ============================================================================
# Load and Parse Log Data
# ============================================================================

# Read new lines from log file
new_rows, new_position = read_and_parse_log(log_path, st.session_state.file_position)

# Append new rows to session state
if new_rows:
    st.session_state.parsed_rows.extend(new_rows)
    st.session_state.file_position = new_position

    # Limit memory usage
    if len(st.session_state.parsed_rows) > max_rows:
        st.session_state.parsed_rows = st.session_state.parsed_rows[-max_rows:]

# Convert to DataFrame
if st.session_state.parsed_rows:
    df = pd.DataFrame(st.session_state.parsed_rows)
else:
    df = pd.DataFrame()


# ============================================================================
# Header
# ============================================================================

st.title("💎 Diamond Code - Real-Time Training Dashboard")

if not df.empty:
    latest_step = df['step'].max()
    latest_loss = df.loc[df['step'] == latest_step, 'loss'].values[0]
    latest_acc = df.loc[df['step'] == latest_step, 'acc'].values[0] if 'acc' in df.columns else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Step", f"{latest_step:,}")
    col2.metric("Loss", f"{latest_loss:.4f}")
    col3.metric("Accuracy", f"{latest_acc*100:.1f}%" if latest_acc > 0 else "N/A")
else:
    st.info("Waiting for training data... Start training with: `python train_with_logging.py`")

st.markdown("---")


# ============================================================================
# Row 1: Loss and Accuracy Charts
# ============================================================================

if not df.empty and 'step' in df.columns and 'loss' in df.columns:
    st.subheader("📈 Loss and Accuracy")

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Loss (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['step'],
            y=df['loss'],
            mode='lines',
            name='Loss',
            line=dict(color=DIAMOND_BLUE, width=2),
            yaxis='y1'
        )
    )

    # Accuracy (secondary y-axis)
    if 'acc' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['step'],
                y=df['acc'],
                mode='lines',
                name='Accuracy',
                line=dict(color=DIAMOND_PLATINUM, width=2, dash='dot'),
                yaxis='y2'
            )
        )

    # Layout with dual y-axes
    fig.update_layout(
        title="Training Progress",
        xaxis=dict(title="Step"),
        yaxis=dict(
            title=dict(text="Loss", font=dict(color=DIAMOND_BLUE)),
            tickfont=dict(color=DIAMOND_BLUE)
        ),
        yaxis2=dict(
            title=dict(text="Accuracy", font=dict(color=DIAMOND_PLATINUM)),
            tickfont=dict(color=DIAMOND_PLATINUM),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400,
        template='plotly_dark',
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No data available for Loss/Accuracy chart")


# ============================================================================
# Row 2: Jump Gate Activation
# ============================================================================

if not df.empty and 'jump_gate' in df.columns:
    st.subheader("🎯 Routing Emergence")

    col1, col2 = st.columns(2)

    with col1:
        # Jump Gate Activation chart
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df['step'],
                y=df['jump_gate'] * 100,  # Convert to percentage
                mode='lines',
                name='Jump Gate Activation',
                line=dict(color=DIAMOND_SILVER, width=2),
                fill='tozeroy',
                fillcolor='rgba(192, 192, 192, 0.2)'
            )
        )

        fig.update_layout(
            title="Jump Gate Activation Rate",
            xaxis=dict(title="Step"),
            yaxis=dict(title="Activation Rate (%)", range=[0, 100]),
            hovermode='x',
            height=300,
            template='plotly_dark',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Holonomy Distribution vs Accuracy
        if 'holonomy_pct' in df.columns and 'acc' in df.columns:
            fig_hol = go.Figure()

            # Holonomy distribution (% of samples with holonomy=+1)
            fig_hol.add_trace(
                go.Scatter(
                    x=df['step'],
                    y=df['holonomy_pct'] * 100,  # Convert to percentage
                    mode='lines',
                    name='% Holonomy +1',
                    line=dict(color='#00D9FF', width=2),
                    yaxis='y1'
                )
            )

            # Accuracy overlay (secondary y-axis)
            fig_hol.add_trace(
                go.Scatter(
                    x=df['step'],
                    y=df['acc'] * 100,  # Convert to percentage
                    mode='lines',
                    name='Accuracy',
                    line=dict(color='#FFB000', width=2, dash='dot'),
                    yaxis='y2'
                )
            )

            fig_hol.update_layout(
                title="Holonomy Distribution vs Accuracy",
                xaxis=dict(title="Step"),
                yaxis=dict(
                    title=dict(text="Holonomy % (+1)", font=dict(color='#00D9FF')),
                    tickfont=dict(color='#00D9FF'),
                    range=[0, 100]
                ),
                yaxis2=dict(
                    title=dict(text="Accuracy", font=dict(color='#FFB000')),
                    tickfont=dict(color='#FFB000'),
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                hovermode='x unified',
                height=300,
                template='plotly_dark',
                showlegend=True,
                legend=dict(x=0.01, y=0.99)
            )

            st.plotly_chart(fig_hol, use_container_width=True)
        else:
            st.info("🧬 Holonomy Distribution\n\nWaiting for holonomy_pct data from TRUE Möbius training...")

else:
    st.info("No data available for Jump Gate Activation chart")


# ============================================================================
# Row 3: Möbius Diagnostics
# ============================================================================

if not df.empty and 'ptr_std' in df.columns:
    st.subheader("🔬 Möbius Helix Diagnostics")

    col1, col2 = st.columns(2)

    with col1:
        # Pointer Synchronization (ptr_std)
        fig_sync = go.Figure()

        fig_sync.add_trace(
            go.Scatter(
                x=df['step'],
                y=df['ptr_std'],
                mode='lines',
                name='Pointer Position StdDev',
                line=dict(color='#FF6B9D', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 157, 0.2)'
            )
        )

        # Add reference line for high synchronization
        fig_sync.add_hline(
            y=5.0,
            line_dash="dash",
            line_color="orange",
            annotation_text="Low (synchronized)",
            annotation_position="right"
        )

        fig_sync.update_layout(
            title="Pointer Position Variance (Synchronization Detector)",
            xaxis=dict(title="Step"),
            yaxis=dict(title="Std Dev (positions)"),
            hovermode='x',
            height=300,
            template='plotly_dark',
            showlegend=False
        )

        st.plotly_chart(fig_sync, use_container_width=True)

        st.caption("⚠️ Low variance = pointers synchronized → potential oscillations")

    with col2:
        # Memory Coverage and Wraps
        if 'coverage' in df.columns and 'wraps' in df.columns:
            fig_mem = go.Figure()

            # Coverage (secondary y-axis)
            fig_mem.add_trace(
                go.Scatter(
                    x=df['step'],
                    y=df['coverage'] * 100,  # Convert to percentage
                    mode='lines',
                    name='Memory Coverage',
                    line=dict(color='#7DFF8C', width=2),
                    yaxis='y2'
                )
            )

            # Wrap events (primary y-axis)
            fig_mem.add_trace(
                go.Scatter(
                    x=df['step'],
                    y=df['wraps'],
                    mode='markers+lines',
                    name='Wrap Events',
                    line=dict(color='#FFD700', width=1),
                    marker=dict(size=4),
                    yaxis='y1'
                )
            )

            fig_mem.update_layout(
                title="Memory Coverage & Wrap Events",
                xaxis=dict(title="Step"),
                yaxis=dict(
                    title=dict(text="Wrap Events", font=dict(color='#FFD700')),
                    tickfont=dict(color='#FFD700')
                ),
                yaxis2=dict(
                    title=dict(text="Coverage %", font=dict(color='#7DFF8C')),
                    tickfont=dict(color='#7DFF8C'),
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                hovermode='x unified',
                height=300,
                template='plotly_dark',
                showlegend=True,
                legend=dict(x=0.01, y=0.99)
            )

            st.plotly_chart(fig_mem, use_container_width=True)

            st.caption("💡 Coverage = range of positions used / total positions (64)")
        else:
            st.info("Waiting for coverage and wrap data...")

else:
    st.info("No Möbius diagnostic data available yet")


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
    Diamond Code Dashboard v1.0 (Phase 1 MVP) |
    <a href='https://github.com/VRAXION/VRAXION' target='_blank'>VRAXION Project</a>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# Debug Info (Collapsed)
# ============================================================================

with st.expander("🔧 Debug Info"):
    st.write(f"**File Position:** {st.session_state.file_position:,} bytes")
    st.write(f"**Rows in Memory:** {len(st.session_state.parsed_rows):,}")
    st.write(f"**DataFrame Shape:** {df.shape if not df.empty else 'Empty'}")

    if not df.empty:
        st.write("**Available Columns:**")
        st.write(list(df.columns))

        st.write("**Latest Row:**")
        st.write(df.iloc[-1].to_dict())
