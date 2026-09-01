"""
Guardian — Motion Module
Motion analysis using the UMAFall dataset and the trained MotionStream model.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# ==========================================================================
# PROJECT PATH & IMPORTS
# ==========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_theme import apply_theme, render_page_header, render_sidebar_header
from data.loader import iter_windows, read_umafall
from streams.motion import MotionStream

# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================

st.set_page_config(
    page_title="Guardian — Motion",
    layout="wide",
)

apply_theme()
render_sidebar_header()

# ==========================================================================
# PATHS
# ==========================================================================

UMAFALL_ROOT = PROJECT_ROOT / "data" / "raw" / "UMAFall"
MOTION_MODEL_PATH = PROJECT_ROOT / "models" / "motion.joblib"


# ==========================================================================
# DATA HELPERS (PRESERVED EXACTLY)
# ==========================================================================

@st.cache_data
def list_umafall_files():
    if not UMAFALL_ROOT.exists():
        return []

    return [
        str(path)
        for path in sorted(UMAFALL_ROOT.rglob("*.csv"))
        if path.name.lower() != "fall_timestamps.csv"
    ]


@st.cache_data
def analyze_motion_record(
    record_path: str,
    model_path: str,
    model_mtime_ns: int,
):
    # Used to invalidate the cache when the model changes.
    del model_mtime_ns

    record = read_umafall(Path(record_path))

    motion_stream = MotionStream.load(
        Path(model_path)
    )

    if getattr(motion_stream, "model", None) is not None:
        if hasattr(motion_stream.model, "n_jobs"):
            motion_stream.model.n_jobs = 1

    rows = []

    for window, _meta in iter_windows(record):

        score = float(
            motion_stream.score(window)
        )

        time_since_impact = getattr(
            motion_stream,
            "time_since_impact",
            None,
        )

        gate_open = bool(
            getattr(
                motion_stream,
                "gate_open",
                False,
            )
        )

        rows.append(
            {
                "t": float(window["t"]),

                "motion_score": score,

                "impact": float(
                    getattr(
                        motion_stream,
                        "last_impact",
                        0.0,
                    )
                ),

                "stillness": float(
                    getattr(
                        motion_stream,
                        "last_stillness",
                        0.0,
                    )
                ),

                "quality": float(
                    getattr(
                        motion_stream,
                        "last_quality",
                        0.0,
                    )
                ),

                "time_since_impact": (
                    None
                    if time_since_impact is None
                    else float(time_since_impact)
                ),

                "gate_open": gate_open,
            }
        )

    info = {
        "path": str(record.path),
        "filename": record.path.name,
        "subject": record.subject,
        "activity": record.activity,
        "trial": record.trial,
        "samples": int(record.acc.shape[0]),
        "duration": float(record.duration),
        "window_count": len(rows),
    }

    if not rows:
        return info, pd.DataFrame(), None, None

    frame = pd.DataFrame(rows)

    latest = rows[-1]

    peak = max(
        rows,
        key=lambda row: row["motion_score"],
    )

    return info, frame, latest, peak


def format_seconds(value) -> str:
    if value is None:
        return "None"

    return f"{float(value):.2f} s"


# ==========================================================================
# PAGE HEADER
# ==========================================================================

render_page_header(
    title="Motion Analysis",
    subtitle="Wrist IMU kinematics, two-stage impact gating, Random Forest scoring, and post-impact stillness tracking.",
    context_label="UMAFall Dataset",
)


# ==========================================================================
# VALIDATION
# ==========================================================================

if not MOTION_MODEL_PATH.exists():
    st.error(f"Missing motion model: {MOTION_MODEL_PATH}")
    st.stop()

record_files = list_umafall_files()

if not record_files:
    st.error(f"No UMAFall CSV files were found under {UMAFALL_ROOT}.")
    st.stop()


# ==========================================================================
# RECORD SELECTION BAR
# ==========================================================================

col_select, col_badge = st.columns([4, 1])

with col_select:
    selected_path = st.selectbox(
        "Select UMAFall Recording",
        options=record_files,
        format_func=lambda path: Path(path).name,
        help="Select any raw UMAFall trial to run the real-time MotionStream pipeline.",
    )

with col_badge:
    st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.8rem; color: #5B6470; text-align: right; padding-top: 0.25rem;">50 Hz Uniform Grid</div>', unsafe_allow_html=True)


# ==========================================================================
# RUN MOTION ANALYSIS
# ==========================================================================

with st.spinner("Processing wrist IMU stream through MotionStream..."):
    info, motion_df, latest, peak = analyze_motion_record(
        selected_path,
        str(MOTION_MODEL_PATH),
        MOTION_MODEL_PATH.stat().st_mtime_ns,
    )


# ==========================================================================
# RECORD METADATA & TELEMETRY
# ==========================================================================

if motion_df.empty or latest is None:
    st.warning("No motion windows were produced for the selected record.")
    st.stop()

st.markdown('<div class="g-section-title">Recording Telemetry</div>', unsafe_allow_html=True)

meta_cols = st.columns(5)
meta_cols[0].metric("Subject", info["subject"])
meta_cols[1].metric("Activity", info["activity"])
meta_cols[2].metric("Trial", f"#{info['trial']}")
meta_cols[3].metric("Samples", f"{info['samples']:,}")
meta_cols[4].metric("Duration", f"{info['duration']:.2f} s")


# ==========================================================================
# STREAM OUTPUTS & DIAGNOSTIC KPIS
# ==========================================================================

st.markdown('<div class="g-section-title">Stream Outputs & Impact Diagnostics</div>', unsafe_allow_html=True)

res_cols = st.columns(5)
res_cols[0].metric("Motion Score", f"{latest['motion_score']:.2f}")
res_cols[1].metric("Impact Stage", f"{latest['impact']:.2f}")
res_cols[2].metric("Stillness Score", f"{latest['stillness']:.2f}")
res_cols[3].metric("Time Since Impact", format_seconds(latest["time_since_impact"]))
res_cols[4].metric("Signal Quality", f"{latest['quality']:.2f}")

gate_status_panel_class = "g-panel-alert" if latest["gate_open"] else "g-panel-normal"
gate_indicator = (
    '<span class="g-indicator g-indicator-alert"><span class="g-dot g-dot-alert"></span> Gate Open (Impact Triggered)</span>'
    if latest["gate_open"]
    else '<span class="g-indicator g-indicator-normal"><span class="g-dot g-dot-normal"></span> Gate Closed (Normal)</span>'
)

st.markdown(
    f"""
    <div class="{gate_status_panel_class}" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.6rem; margin-bottom: 0.75rem;">
        <div style="font-size: 0.83rem; color: #5B6470;">
            Peak motion score: <strong class="g-telemetry-num" style="color: #1B222B;">{peak['motion_score']:.2f}</strong> at <span class="g-telemetry-num">t = {peak['t']:.2f}s</span>
        </div>
        <div>
            {gate_indicator}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# TIME-SERIES SIGNAL VISUALIZATION
# ==========================================================================

st.markdown('<div class="g-section-title">Multi-Signal Time Series</div>', unsafe_allow_html=True)

chart_df = motion_df[
    [
        "t",
        "motion_score",
        "impact",
        "stillness",
        "quality",
    ]
].rename(
    columns={
        "motion_score": "Motion score",
        "impact": "Impact",
        "stillness": "Stillness",
        "quality": "Quality",
    }
)

st.line_chart(
    chart_df.set_index("t"),
    height=380,
)


# ==========================================================================
# WINDOW-LEVEL DATA EXPANDER
# ==========================================================================

with st.expander("Window-Level Telemetry Data"):
    st.caption("Window timeline with per-step feature and gate telemetry.")
    st.dataframe(
        motion_df,
        use_container_width=True,
        hide_index=True,
    )