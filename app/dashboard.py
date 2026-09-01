"""
Guardian — Main Dashboard
Instrumentation console landing page for Guardian.
"""

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_theme import apply_theme, render_page_header, render_sidebar_header

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Guardian",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
render_sidebar_header()

# ==============================================================================
# HEADER
# ==============================================================================

render_page_header(
    title="Guardian.",
)

# ==============================================================================
# STREAM ARCHITECTURE OVERVIEW
# ==============================================================================

st.markdown('<div class="g-section-title">Stream Architecture Overview</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0.75rem; margin-bottom: 0.75rem;">
        <div class="g-panel-normal">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem;">
                <span style="font-size: 0.8rem; font-weight: 600; color: #1B222B;">Kinematic Stream</span>
                <span style="font-size: 0.74rem; color: #5B6470;">Wrist IMU</span>
            </div>
            <div style="font-size: 0.83rem; color: #5B6470; line-height: 1.45;">
                Two-stage detection pipeline: 2.5g impact gate, 18-feature Random Forest classifier, and 10s post-impact stillness tracking.
            </div>
        </div>
        <div class="g-panel-normal">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem;">
                <span style="font-size: 0.8rem; font-weight: 600; color: #1B222B;">Context Stream</span>
                <span style="font-size: 0.74rem; color: #5B6470;">4-State Model</span>
            </div>
            <div style="font-size: 0.83rem; color: #5B6470; line-height: 1.45;">
                Activity classification surrounding impact windows (ambulating, stationary, seated hand, lying) for false positive suppression.
            </div>
        </div>
        <div class="g-panel-normal">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem;">
                <span style="font-size: 0.8rem; font-weight: 600; color: #1B222B;">Physiology Stream</span>
                <span style="font-size: 0.74rem; color: #5B6470;">Heart Rate</span>
            </div>
            <div style="font-size: 0.83rem; color: #5B6470; line-height: 1.45;">
                Autonomic stress and sustained heart rate deviation tracking contributing to late-fusion evidence combination.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# PIPELINE SPECIFICATIONS
# ==============================================================================

st.markdown('<div class="g-section-title">Pipeline Specifications & Ingestion</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown(
        """
        <div class="g-panel">
            <div style="font-size: 0.82rem; font-weight: 600; color: #1B222B; margin-bottom: 0.5rem;">Signal Processing & Stream Contracts</div>
            <div style="display: flex; flex-direction: column; gap: 0.45rem; font-size: 0.82rem; color: #5B6470; line-height: 1.45;">
                <div><span style="color: #1B222B; font-weight: 500;">Window Slicing:</span> Uniform 2.5-second windows (<span class="g-telemetry-num">125</span> samples @ <span class="g-telemetry-num">50 Hz</span>) with 50% overlap (<span class="g-telemetry-num">1.24s</span> hop).</div>
                <div><span style="color: #1B222B; font-weight: 500;">Impact Timing:</span> Reconstructed via internal buffer clock <code>t_impact = internal_now - time_since_impact</code>.</div>
                <div><span style="color: #1B222B; font-weight: 500;">Stream Isolation:</span> Independent per-window scoring with stream contract conformity (<code>last_quality</code> in [0, 1]).</div>
                <div><span style="color: #1B222B; font-weight: 500;">Calibration:</span> Temperature scaling with open-set rejection for out-of-distribution events.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        """
        <div class="g-panel">
            <div style="font-size: 0.82rem; font-weight: 600; color: #1B222B; margin-bottom: 0.5rem;">Dataset & Sensor Parameters</div>
            <div style="display: flex; flex-direction: column; gap: 0.45rem; font-size: 0.82rem;">
                <div style="display: flex; justify-content: space-between; color: #5B6470;">
                    <span>Primary Benchmark</span>
                    <span style="color: #1B222B; font-weight: 500;">UMAFall (<span class="g-telemetry-num">208</span> Falls, <span class="g-telemetry-num">538</span> ADLs)</span>
                </div>
                <div style="display: flex; justify-content: space-between; color: #5B6470;">
                    <span>Sensor Configuration</span>
                    <span style="color: #1B222B; font-weight: 500;">Tri-axial Accel + Gyro (Wrist)</span>
                </div>
                <div style="display: flex; justify-content: space-between; color: #5B6470;">
                    <span>Sampling Rate</span>
                    <span class="g-telemetry-num" style="color: #1B222B; font-weight: 500;">50.0 Hz</span>
                </div>
                <div style="display: flex; justify-content: space-between; color: #5B6470;">
                    <span>Fusion Strategy</span>
                    <span style="color: #1B222B; font-weight: 500;">Weighted Evidence Combination</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# CONSOLE NAVIGATION NOTE
# ==============================================================================

st.markdown(
    """
    <div class="g-panel" style="font-size: 0.82rem; color: #5B6470;">
        Select an analytics module from the left sidebar to inspect real-time stream execution.
    </div>
    """,
    unsafe_allow_html=True,
)