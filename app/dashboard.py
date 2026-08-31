"""
app/dashboard.py

Guardian real-data replay dashboard.

Pipeline:
    UMAFall CSV
        -> data.loader.read_umafall()
        -> data.loader.replay()
        -> ContextStream
        -> MotionStream
        -> PhysiologyStream
        -> fusion
        -> decision state machine
        -> explanation
        -> dashboard
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


# ==========================================================================
# PROJECT PATH
# ==========================================================================
#
# Streamlit executes this file from inside app/, so Python may not have the
# Guardian project root on sys.path.
#
# Without this, imports such as:
#     from data.loader import ...
#
# can fail with:
#     ModuleNotFoundError: No module named 'data'
#
# Add the project root explicitly BEFORE importing project modules.
# ==========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================================
# Guardian imports
# ==========================================================================

from data.loader import (
    read_umafall,
    replay,
    ScriptedTrace,
)

from streams.context import ContextStream
from streams.motion import MotionStream
from streams.physiology import PhysiologyStream

from core.fusion import fuse
from core.decision import DecisionStateMachine
from core.explain import explain


# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================

st.set_page_config(
    page_title="Guardian Dashboard",
    page_icon="🛡️",
    layout="wide",
)


# ==========================================================================
# PATHS
# ==========================================================================

DATA_ROOT = PROJECT_ROOT / "data" / "raw"

# IMPORTANT:
# test_pipeline_real.py uses data/hr_trace.csv
# so the dashboard should use the same location.
HR_TRACE_PATH = PROJECT_ROOT / "data" / "hr_trace.csv"

MOTION_MODEL_PATH = PROJECT_ROOT / "models" / "motion.joblib"
CONTEXT_MODEL_PATH = PROJECT_ROOT / "models" / "context.joblib"


# ==========================================================================
# HEADER
# ==========================================================================

st.title("Guardian — Real Data Replay Dashboard")

st.caption(
    "UMAFall wrist sensor data → Context + Motion + Physiology → "
    "Fusion → Decision → Explanation"
)

st.info(
    "The dashboard uses real UMAFall sensor data and the trained "
    "ContextStream and MotionStream models. Physiology uses the "
    "scripted HR trace until hardware integration is available."
)


# ==========================================================================
# DATASET DISCOVERY
# ==========================================================================

@st.cache_data
def find_records():
    """
    Find all readable UMAFall CSV files under data/raw/.

    hr_trace.csv is explicitly excluded because it is not an UMAFall
    sensor recording.
    """

    records = []

    if not DATA_ROOT.exists():
        return records

    for path in sorted(DATA_ROOT.rglob("*.csv")):

        if path.name.lower() == "hr_trace.csv":
            continue

        try:
            record = read_umafall(path)
            records.append(record)

        except Exception:
            # A bad/unrelated CSV should not crash the dashboard.
            continue

    return records


records = find_records()


# ==========================================================================
# SIDEBAR
# ==========================================================================

st.sidebar.header("Replay controls")


if not records:

    st.error(
        "No readable UMAFall CSV files were found under "
        f"{DATA_ROOT}."
    )

    st.stop()


record_labels = [
    (
        f"{r.subject} — {r.activity} — "
        f"Trial {r.trial} — {r.path.name}"
    )
    for r in records
]


selected_index = st.sidebar.selectbox(
    "UMAFall record",
    range(len(records)),
    format_func=lambda i: record_labels[i],
)


selected_record = records[selected_index]


# --------------------------------------------------------------------------
# Replay speed
# --------------------------------------------------------------------------

replay_speed = st.sidebar.select_slider(
    "Replay speed",
    options=["1x", "4x"],
    value="4x",
)

speed = 1.0 if replay_speed == "1x" else 4.0


# --------------------------------------------------------------------------
# Number of windows
# --------------------------------------------------------------------------

available_windows = max(
    1,
    int(
        (selected_record.acc.shape[0] - 125) // 62 + 1
    ),
)

n_windows = st.sidebar.slider(
    "Number of windows",
    min_value=1,
    max_value=min(100, available_windows),
    value=min(20, available_windows),
)


# --------------------------------------------------------------------------
# Run button
# --------------------------------------------------------------------------

run_clicked = st.sidebar.button(
    "Run replay",
    type="primary",
)


# ==========================================================================
# CLEAR OLD RESULTS WHEN RECORD CHANGES
# ==========================================================================

record_key = str(selected_record.path)

if (
    "last_record_key" not in st.session_state
    or st.session_state.last_record_key != record_key
):

    st.session_state.history = []
    st.session_state.last_record_key = record_key


# ==========================================================================
# SELECTED RECORD INFORMATION
# ==========================================================================

st.subheader("Selected record")


info_cols = st.columns(5)


info_cols[0].metric(
    "Subject",
    selected_record.subject,
)

info_cols[1].metric(
    "Activity",
    selected_record.activity,
)

info_cols[2].metric(
    "Trial",
    selected_record.trial,
)

info_cols[3].metric(
    "Samples",
    selected_record.acc.shape[0],
)

info_cols[4].metric(
    "Duration",
    f"{selected_record.duration:.2f} s",
)


# --------------------------------------------------------------------------
# Loader notes
# --------------------------------------------------------------------------

if selected_record.notes:

    with st.expander("Loader notes"):

        for note in selected_record.notes:
            st.write(f"- {note}")


# ==========================================================================
# MODEL / TRACE STATUS
# ==========================================================================

with st.expander("Pipeline resources"):

    col1, col2, col3 = st.columns(3)

    col1.write("**Context model**")

    if CONTEXT_MODEL_PATH.exists():
        col1.success("context.joblib found")
    else:
        col1.error("context.joblib missing")

    col2.write("**Motion model**")

    if MOTION_MODEL_PATH.exists():
        col2.success("motion.joblib found")
    else:
        col2.error("motion.joblib missing")

    col3.write("**HR trace**")

    if HR_TRACE_PATH.exists():
        col3.success("hr_trace.csv found")
    else:
        col3.warning("hr_trace.csv not found")


# ==========================================================================
# RUN REAL PIPELINE
# ==========================================================================

if run_clicked:

    st.session_state.history = []

    # ----------------------------------------------------------------------
    # Verify models
    # ----------------------------------------------------------------------

    missing_models = []

    if not CONTEXT_MODEL_PATH.exists():
        missing_models.append(str(CONTEXT_MODEL_PATH))

    if not MOTION_MODEL_PATH.exists():
        missing_models.append(str(MOTION_MODEL_PATH))

    if missing_models:

        st.error(
            "Required trained model(s) are missing:\n\n"
            + "\n".join(
                f"- {path}" for path in missing_models
            )
        )

        st.stop()


    # ----------------------------------------------------------------------
    # Load trained streams
    # ----------------------------------------------------------------------

    try:

        context_stream = ContextStream.load(
            CONTEXT_MODEL_PATH
        )

        motion_stream = MotionStream.load(
            MOTION_MODEL_PATH
        )

        physiology_stream = PhysiologyStream()

        decision_machine = DecisionStateMachine()

    except Exception as e:

        st.error(
            "Failed to load the Guardian pipeline:\n\n"
            f"{type(e).__name__}: {e}"
        )

        st.stop()


    # ----------------------------------------------------------------------
    # Load scripted physiology trace
    # ----------------------------------------------------------------------

    trace = None

    if HR_TRACE_PATH.exists():

        try:
            trace = ScriptedTrace(HR_TRACE_PATH)

        except Exception as e:

            st.warning(
                "Could not load hr_trace.csv. "
                f"Physiology will run without the scripted trace.\n\n"
                f"{type(e).__name__}: {e}"
            )

            trace = None

    else:

        st.warning(
            "data/hr_trace.csv was not found. "
            "Physiology will run without the scripted HR trace."
        )


    # ----------------------------------------------------------------------
    # Status indicators
    # ----------------------------------------------------------------------

    status_cols = st.columns(4)

    status_cols[0].success("ContextStream loaded")
    status_cols[1].success("MotionStream loaded")
    status_cols[2].success("PhysiologyStream loaded")

    if trace is not None:
        status_cols[3].success("HR trace loaded")
    else:
        status_cols[3].warning("No HR trace")


    # ----------------------------------------------------------------------
    # Replay
    # ----------------------------------------------------------------------

    history = []

    windows = replay(
        selected_record,
        speed=speed,
        trace=trace,
    )


    # ----------------------------------------------------------------------
    # Progress bar
    # ----------------------------------------------------------------------

    progress = st.progress(0.0)

    status_text = st.empty()


    for i, window in enumerate(windows):

        if i >= n_windows:
            break


        # ==================================================================
        # CONTEXT STREAM
        # ==================================================================

        try:

            context_score = context_stream.score(
                window
            )

            context_state = getattr(
                context_stream,
                "last_state",
                None,
            )

            context_confidence = getattr(
                context_stream,
                "last_confidence",
                0.0,
            )

        except Exception as e:

            context_score = 0.0
            context_state = "unknown"
            context_confidence = 0.0

            st.warning(
                f"ContextStream failed at window {i}: {e}"
            )


        # ==================================================================
        # MOTION STREAM
        # ==================================================================

        try:

            motion_score = motion_stream.score(
                window
            )

            motion_impact = getattr(
                motion_stream,
                "last_impact",
                0.0,
            )

            motion_stillness = getattr(
                motion_stream,
                "last_stillness",
                0.0,
            )

            time_since_impact = getattr(
                motion_stream,
                "time_since_impact",
                None,
            )

        except Exception as e:

            motion_score = 0.0
            motion_impact = 0.0
            motion_stillness = 0.0
            time_since_impact = None

            st.warning(
                f"MotionStream failed at window {i}: {e}"
            )


        # ==================================================================
        # PHYSIOLOGY STREAM
        # ==================================================================

        phys_window = dict(window)

        # PhysiologyStream can use the context classification.
        phys_window["context_state"] = context_state


        try:

            physiology_score = physiology_stream.score(
                phys_window
            )

            physiology_quality = getattr(
                physiology_stream,
                "last_quality",
                0.0,
            )

        except Exception as e:

            physiology_score = 0.0
            physiology_quality = 0.0

            st.warning(
                f"PhysiologyStream failed at window {i}: {e}"
            )


        # ==================================================================
        # FUSION
        # ==================================================================

        scores = {
            "context": float(context_score),
            "motion": float(motion_score),
            "physiology": float(physiology_score),
        }


        try:

            risk, contributions = fuse(
                scores,
                context_stream,
                motion_stream,
            )

        except Exception as e:

            risk = 0.0

            contributions = {
                "context": 0.0,
                "motion": 0.0,
                "physiology": 0.0,
            }

            st.warning(
                f"Fusion failed at window {i}: {e}"
            )


        # ==================================================================
        # DECISION STATE MACHINE
        # ==================================================================

        try:

            state = decision_machine.update(
                risk,
                window["t"],
                contributions=contributions,
            )

        except Exception as e:

            state = "unknown"

            st.warning(
                f"Decision state machine failed at "
                f"window {i}: {e}"
            )


        # ==================================================================
        # EXPLANATION
        # ==================================================================

        try:

            explanation = explain(
                risk,
                contributions,
            )

        except Exception as e:

            explanation = (
                f"Explanation unavailable: "
                f"{type(e).__name__}: {e}"
            )


        # ==================================================================
        # SAVE RESULT
        # ==================================================================

        history.append(
            {
                "t": float(window["t"]),

                "risk": float(risk),

                "state": state,

                "explanation": explanation,

                # Stream scores
                "score_context": float(context_score),
                "score_motion": float(motion_score),
                "score_physiology": float(physiology_score),

                # Fusion contributions
                "contrib_context": float(
                    contributions.get("context", 0.0)
                ),

                "contrib_motion": float(
                    contributions.get("motion", 0.0)
                ),

                "contrib_physiology": float(
                    contributions.get("physiology", 0.0)
                ),

                # Context details
                "context_state": context_state,

                "context_confidence": float(
                    context_confidence
                ),

                # Motion details
                "motion_impact": float(
                    motion_impact
                ),

                "motion_stillness": float(
                    motion_stillness
                ),

                "time_since_impact": time_since_impact,

                # Physiology details
                "physiology_quality": float(
                    physiology_quality
                ),

                # Raw physiology values
                "hr": window.get("hr"),

                "temp": window.get("temp"),
            }
        )


        # ------------------------------------------------------------------
        # Progress
        # ------------------------------------------------------------------

        completed = i + 1

        progress.progress(
            min(completed / n_windows, 1.0)
        )

        status_text.write(
            f"Processing window {completed}/{n_windows} "
            f"— t = {window['t']:.2f}s"
        )


    progress.empty()
    status_text.empty()


    # ----------------------------------------------------------------------
    # Save history
    # ----------------------------------------------------------------------

    st.session_state.history = history


# ==========================================================================
# RESULTS
# ==========================================================================

history = st.session_state.get(
    "history",
    [],
)


if not history:

    st.info(
        "Select a UMAFall record and click "
        "'Run replay' to execute the real pipeline."
    )

    st.stop()


# ==========================================================================
# RESULTS DATAFRAME
# ==========================================================================

df = pd.DataFrame(history)


# ==========================================================================
# CURRENT DECISION STATE
# ==========================================================================

latest = history[-1]


st.subheader("Current decision state")


current_state = str(
    latest.get("state", "unknown")
)


state_display = current_state.lower()


if state_display in ("emergency", "alert", "fall"):

    st.error(
        f"**{current_state}**"
    )

elif state_display in ("warning", "caution"):

    st.warning(
        f"**{current_state}**"
    )

else:

    st.success(
        f"**{current_state}**"
    )


# ==========================================================================
# STREAM SCORES
# ==========================================================================

st.subheader("Stream scores")


score_cols = st.columns(3)


score_cols[0].metric(
    "Context",
    f"{latest['score_context']:.2f}",
)

score_cols[1].metric(
    "Motion",
    f"{latest['score_motion']:.2f}",
)

score_cols[2].metric(
    "Physiology",
    f"{latest['score_physiology']:.2f}",
)


# ==========================================================================
# EMERGENCY RISK
# ==========================================================================

st.subheader("Emergency risk")


risk_df = df[
    [
        "t",
        "risk",
    ]
].copy()


risk_df = risk_df.set_index("t")


st.line_chart(
    risk_df,
    y="risk",
)


# ==========================================================================
# CONTEXT
# ==========================================================================

st.subheader("Context")


context_cols = st.columns(2)


context_cols[0].metric(
    "Detected state",
    str(
        latest.get(
            "context_state",
            "unknown",
        )
    ),
)


context_cols[1].metric(
    "Confidence",
    f"{latest.get('context_confidence', 0.0):.2f}",
)


# ==========================================================================
# MOTION EVIDENCE
# ==========================================================================

st.subheader("Motion evidence")


motion_cols = st.columns(3)


motion_cols[0].metric(
    "Impact",
    f"{latest.get('motion_impact', 0.0):.2f}",
)


motion_cols[1].metric(
    "Stillness",
    f"{latest.get('motion_stillness', 0.0):.2f}",
)


time_since = latest.get(
    "time_since_impact",
    None,
)


if time_since is None:

    time_since_display = "None"

else:

    try:
        time_since_display = f"{float(time_since):.2f} s"
    except Exception:
        time_since_display = str(time_since)


motion_cols[2].metric(
    "Time since impact",
    time_since_display,
)


# ==========================================================================
# PHYSIOLOGY
# ==========================================================================

st.subheader("Physiology")


phys_cols = st.columns(3)


hr_value = latest.get("hr")

if hr_value is None:
    hr_display = "None"
else:
    try:
        hr_display = f"{float(hr_value):.1f} bpm"
    except Exception:
        hr_display = str(hr_value)


temp_value = latest.get("temp")

if temp_value is None:
    temp_display = "None"
else:
    try:
        temp_display = f"{float(temp_value):.2f}"
    except Exception:
        temp_display = str(temp_value)


phys_cols[0].metric(
    "Heart rate",
    hr_display,
)


phys_cols[1].metric(
    "Temperature",
    temp_display,
)


phys_cols[2].metric(
    "Physiology quality",
    f"{latest.get('physiology_quality', 0.0):.2f}",
)


# ==========================================================================
# EXPLANATION
# ==========================================================================

st.subheader("Explanation")


st.code(
    str(
        latest.get(
            "explanation",
            "No explanation available.",
        )
    ),
    language=None,
)


# ==========================================================================
# FUSION CONTRIBUTIONS
# ==========================================================================

st.subheader("Fusion contributions")


contrib_df = df[
    [
        "t",
        "contrib_context",
        "contrib_motion",
        "contrib_physiology",
    ]
].copy()


contrib_df = contrib_df.set_index("t")


st.line_chart(
    contrib_df,
)


# ==========================================================================
# SCORE HISTORY
# ==========================================================================

st.subheader("Stream score history")


score_history_df = df[
    [
        "t",
        "score_context",
        "score_motion",
        "score_physiology",
    ]
].copy()


score_history_df = score_history_df.set_index("t")


st.line_chart(
    score_history_df,
)


# ==========================================================================
# RAW REPLAY HISTORY
# ==========================================================================

with st.expander("Raw replay history"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================================
# PIPELINE SUMMARY
# ==========================================================================

with st.expander("Pipeline details"):

    st.markdown(
        """
### Data

- Source: UMAFall
- Sensor: wrist
- Sampling rate: 50 Hz after resampling
- Window size: 2.5 seconds
- Window samples: 125
- Window overlap: 50%
- Hop: 62 samples

### Streams

- **ContextStream** — estimates the user's current activity/context.
- **MotionStream** — evaluates impact and post-impact stillness.
- **PhysiologyStream** — evaluates physiological evidence such as heart rate.
- **Fusion** — combines the stream evidence.
- **DecisionStateMachine** — converts fused risk into a decision state.
- **Explanation** — describes the evidence contributing to the decision.

### Data separation

Ground-truth labels from UMAFall are used by the dataset loader for
training/evaluation, but labels are not passed into the replay pipeline.
The dashboard therefore processes the replay windows as sensor input.
"""
    )