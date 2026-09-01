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
# HEADER & BUILD STATUS
# ==============================================================================

render_page_header(
    title="Guardian",
)

st.markdown(
    """
    <div style="font-size: 0.84rem; color: #5B6470; margin-top: -0.35rem; margin-bottom: 1.15rem; line-height: 1.45;">
        Wearable fall and emergency-risk pipeline. Motion and Activity are built; Physiological is in progress; Context is designed, not built.
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# EVIDENCE STREAM STATUS (FLAT LIST IN A SINGLE PANEL)
# ==============================================================================

st.markdown('<div class="g-section-title">Evidence Stream Status</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="g-panel" style="padding: 0; margin-bottom: 1.25rem;">
        <div style="display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; padding: 0.65rem 0.95rem; border-bottom: 1px solid #DCDFE2;">
            <div style="display: flex; align-items: baseline; gap: 0.45rem; min-width: 150px;">
                <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #2F6F62; transform: translateY(-1px);"></span>
                <span style="font-size: 0.84rem; font-weight: 600; color: #1B222B;">Motion</span>
                <span style="font-size: 0.74rem; color: #2F6F62; font-weight: 500;">built</span>
            </div>
            <div style="font-size: 0.82rem; color: #5B6470; flex: 1; text-align: left;">
                Two-stage detection: 2.5g impact gate, 18-feature Random Forest classifier, and 10s post-impact stillness tracking.
            </div>
        </div>
        <div style="display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; padding: 0.65rem 0.95rem; border-bottom: 1px solid #DCDFE2;">
            <div style="display: flex; align-items: baseline; gap: 0.45rem; min-width: 150px;">
                <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #2F6F62; transform: translateY(-1px);"></span>
                <span style="font-size: 0.84rem; font-weight: 600; color: #1B222B;">Activity</span>
                <span style="font-size: 0.74rem; color: #2F6F62; font-weight: 500;">built</span>
            </div>
            <div style="font-size: 0.82rem; color: #5B6470; flex: 1; text-align: left;">
                4-state activity classification (ambulating, stationary, seated hand activity, lying/immobile) around impacts for false alarm suppression.
            </div>
        </div>
        <div style="display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; padding: 0.65rem 0.95rem; border-bottom: 1px solid #DCDFE2;">
            <div style="display: flex; align-items: baseline; gap: 0.45rem; min-width: 150px;">
                <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #B8752B; transform: translateY(-1px);"></span>
                <span style="font-size: 0.84rem; font-weight: 600; color: #1B222B;">Physiological</span>
                <span style="font-size: 0.74rem; color: #B8752B; font-weight: 500;">in progress</span>
            </div>
            <div style="font-size: 0.82rem; color: #5B6470; flex: 1; text-align: left;">
                Rule-based heart-rate deviation tracking with dwell time scaling (scripted trace; hardware integration pending).
            </div>
        </div>
        <div style="display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; padding: 0.65rem 0.95rem;">
            <div style="display: flex; align-items: baseline; gap: 0.45rem; min-width: 150px;">
                <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #9AA1A8; transform: translateY(-1px);"></span>
                <span style="font-size: 0.84rem; font-weight: 600; color: #1B222B;">Context</span>
                <span style="font-size: 0.74rem; color: #5B6470; font-weight: 500;">designed, not built</span>
            </div>
            <div style="font-size: 0.82rem; color: #5B6470; flex: 1; text-align: left;">
                Pre- and post-impact temporal context extraction and behavioral state reconstruction.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# PIPELINE SPECIFICATIONS (SINGLE FLAT SPEC SHEET)
# ==============================================================================

st.markdown('<div class="g-section-title">Pipeline Specifications</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="g-panel" style="padding: 0; margin-bottom: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; padding: 0.6rem 0.95rem; border-bottom: 1px solid #DCDFE2; font-size: 0.82rem;">
            <span style="color: #5B6470;">Window Slicing</span>
            <span style="color: #1B222B; text-align: right;">Uniform <span class="g-telemetry-num">2.5s</span> windows (<span class="g-telemetry-num">125</span> samples @ <span class="g-telemetry-num">50 Hz</span>), <span class="g-telemetry-num">50%</span> overlap (<span class="g-telemetry-num">1.24s</span> hop)</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; padding: 0.6rem 0.95rem; border-bottom: 1px solid #DCDFE2; font-size: 0.82rem;">
            <span style="color: #5B6470;">Impact Timing</span>
            <span style="color: #1B222B; text-align: right;">Reconstructed via internal buffer clock <code>t_impact = internal_now - time_since_impact</code></span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; padding: 0.6rem 0.95rem; border-bottom: 1px solid #DCDFE2; font-size: 0.82rem;">
            <span style="color: #5B6470;">Stream Isolation Contract</span>
            <span style="color: #1B222B; text-align: right;">Independent per-window scoring with stream contract conformity (<code>last_quality</code> in <span class="g-telemetry-num">[0, 1]</span>)</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; padding: 0.6rem 0.95rem; border-bottom: 1px solid #DCDFE2; font-size: 0.82rem;">
            <span style="color: #5B6470;">Model Calibration</span>
            <span style="color: #1B222B; text-align: right;">Temperature scaling with open-set rejection for out-of-distribution windows</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; padding: 0.6rem 0.95rem; border-bottom: 1px solid #DCDFE2; font-size: 0.82rem;">
            <span style="color: #5B6470;">Primary Benchmark Dataset</span>
            <span style="color: #1B222B; text-align: right;">UMAFall (<span class="g-telemetry-num">208</span> Falls, <span class="g-telemetry-num">538</span> ADLs across <span class="g-telemetry-num">19</span> subjects)</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; padding: 0.6rem 0.95rem; border-bottom: 1px solid #DCDFE2; font-size: 0.82rem;">
            <span style="color: #5B6470;">Sensor Configuration</span>
            <span style="color: #1B222B; text-align: right;">Tri-axial Accel + Gyro (Wrist, <span class="g-telemetry-num">SensorID=2</span>)</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; padding: 0.6rem 0.95rem; font-size: 0.82rem;">
            <span style="color: #5B6470;">Target Sampling Rate</span>
            <span style="color: #1B222B; text-align: right;" class="g-telemetry-num">50.0 Hz Uniform Grid</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# CLOSING NAVIGATION LINE
# ==============================================================================

st.markdown(
    """
    <div style="font-size: 0.82rem; color: #5B6470; margin-top: 0.25rem;">
        Select a module from the sidebar to inspect real-time stream execution.
    </div>
    """,
    unsafe_allow_html=True,
)