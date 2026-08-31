"""
Guardian — Motion Module

Motion analysis using the UMAFall dataset and the trained MotionStream model.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# ==========================================================================
# PROJECT PATH
# ==========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================================
# Guardian imports
# ==========================================================================

from data.loader import iter_windows, read_umafall
from streams.motion import MotionStream


# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================

st.set_page_config(
    page_title="Guardian — Motion",
    page_icon="🛡️",
    layout="wide",
)


# ==========================================================================
# PATHS
# ==========================================================================

UMAFALL_ROOT = PROJECT_ROOT / "data" / "raw" / "UMAFall"
MOTION_MODEL_PATH = PROJECT_ROOT / "models" / "motion.joblib"


# ==========================================================================
# DATA HELPERS
# ==========================================================================

@st.cache_data
def list_umafall_files():
    if not UMAFALL_ROOT.exists():
        return []

    return [
        str(path)
        for path in sorted(UMAFALL_ROOT.rglob("*.csv"))
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
# MOTION MODULE
# ==========================================================================

st.title("Motion Module")

st.caption("Dataset: UMAFall")

st.write(
    "The MotionStream processes wrist IMU windows from UMAFall, "
    "uses an impact gate and trained Random Forest model, and "
    "examines post-impact stillness."
)

st.write(
    "This module displays motion-specific outputs only. "
    "Fusion and emergency-risk scoring are not used."
)


# ==========================================================================
# VALIDATION
# ==========================================================================

if not MOTION_MODEL_PATH.exists():

    st.error(
        f"Missing motion model: {MOTION_MODEL_PATH}"
    )

    st.stop()


record_files = list_umafall_files()

if not record_files:

    st.error(
        f"No UMAFall CSV files were found under "
        f"{UMAFALL_ROOT}."
    )

    st.stop()


# ==========================================================================
# RECORD SELECTION
# ==========================================================================

selected_path = st.selectbox(
    "UMAFall record",
    options=record_files,
    format_func=lambda path: Path(path).name,
)


# ==========================================================================
# RUN MOTION ANALYSIS
# ==========================================================================

with st.spinner(
    "Running MotionStream on the selected record..."
):

    info, motion_df, latest, peak = analyze_motion_record(
        selected_path,
        str(MOTION_MODEL_PATH),
        MOTION_MODEL_PATH.stat().st_mtime_ns,
    )


# ==========================================================================
# RECORD INFORMATION
# ==========================================================================

st.subheader("Selected record")

info_cols = st.columns(5)

info_cols[0].metric(
    "Subject",
    info["subject"],
)

info_cols[1].metric(
    "Activity",
    info["activity"],
)

info_cols[2].metric(
    "Trial",
    info["trial"],
)

info_cols[3].metric(
    "Samples",
    info["samples"],
)

info_cols[4].metric(
    "Duration",
    f"{info['duration']:.2f} s",
)


# ==========================================================================
# RESULTS
# ==========================================================================

if motion_df.empty or latest is None:

    st.warning(
        "No motion windows were produced "
        "for the selected record."
    )

    st.stop()


st.subheader("Motion results")

result_cols = st.columns(5)

result_cols[0].metric(
    "Motion score",
    f"{latest['motion_score']:.2f}",
)

result_cols[1].metric(
    "Impact",
    f"{latest['impact']:.2f}",
)

result_cols[2].metric(
    "Stillness",
    f"{latest['stillness']:.2f}",
)

result_cols[3].metric(
    "Time since impact",
    format_seconds(
        latest["time_since_impact"]
    ),
)

result_cols[4].metric(
    "Quality",
    f"{latest['quality']:.2f}",
)


# ==========================================================================
# CURRENT STATUS
# ==========================================================================

gate_status = (
    "Open"
    if latest["gate_open"]
    else "Closed"
)

st.caption(
    f"Current gate status: {gate_status}. "
    f"Peak motion score: "
    f"{peak['motion_score']:.2f} "
    f"at t = {peak['t']:.2f}s."
)


# ==========================================================================
# GRAPH
# ==========================================================================

st.subheader("Motion stream graph")

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
    chart_df.set_index("t")
)