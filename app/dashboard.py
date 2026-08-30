"""
app/dashboard.py

Streamlit dashboard for the Guardian pipeline.

STAGE 3: runs entirely on the STUB chain built so far:
    data.loader.replay_stub
    streams.{context,motion,physiology,quality}
    core.fusion.fuse_stub
    core.decision.decide_stub
    core.explain.explain_stub

This file does NOT implement any real algorithm. It only wires and
displays what the stub pipeline already produces, exactly as proven
in test_pipeline_stub.py. When real modules replace the stubs later
(Stage 9), this file should not need to change -- it consumes the
same interfaces (score/last_quality, fuse/decide/explain).

Run with:
    streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd

from data.loader import replay_stub
from streams.context import ContextStream
from streams.motion import MotionStream
from streams.physiology import PhysiologyStream
from streams.quality import QualityStream
from core.fusion import fuse_stub
from core.decision import decide_stub
from core.explain import explain_stub


st.set_page_config(page_title="Guardian Dashboard (stub pipeline)", layout="wide")
st.title("Guardian — Replay Dashboard (Stage 3: stub pipeline)")
st.caption(
    "All scores below come from stub modules (constant/placeholder logic). "
    "This confirms the pipeline wiring, not real detection accuracy."
)

# --- Sidebar controls -------------------------------------------------

st.sidebar.header("Replay controls")

# Scenario selector: labels only for now. Real scenario -> dataset-file
# mapping is Stage 9, once data/loader.py replays real recordings.
scenario = st.sidebar.selectbox(
    "Scenario",
    [
        "Scenario 1 — False-positive suppression (not wired yet)",
        "Scenario 2 — True fall (not wired yet)",
        "Scenario 3 — Degraded signal (not wired yet)",
    ],
)

replay_speed = st.sidebar.select_slider(
    "Replay speed", options=["1x", "4x"], value="1x"
)

n_windows = st.sidebar.slider("Number of windows to replay", 5, 100, 20)

run_clicked = st.sidebar.button("Run replay")

st.sidebar.caption(f"Selected: {scenario}")
st.sidebar.caption(f"Speed setting: {replay_speed} (not yet affecting playback)")

# --- Session state: holds replay history across Streamlit reruns ------

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts, one per window

# --- Run the stub pipeline end-to-end on button press ------------------

if run_clicked:
    streams = {
        "context": ContextStream(),
        "motion": MotionStream(),
        "physiology": PhysiologyStream(),
        "quality": QualityStream(),
    }

    history = []
    for window in replay_stub(n_windows=n_windows):
        scores = {name: s.score(window) for name, s in streams.items()}
        qualities = {name: s.last_quality for name, s in streams.items()}
        risk = fuse_stub(scores)
        state = decide_stub(risk)
        explanation = explain_stub(risk, scores)

        history.append(
            {
                "t": window["t"],
                "risk": risk,
                "state": state,
                "explanation": explanation,
                **{f"score_{k}": v for k, v in scores.items()},
                **{f"quality_{k}": v for k, v in qualities.items()},
            }
        )

    st.session_state.history = history

# --- Display ------------------------------------------------------------

history = st.session_state.history

if not history:
    st.info("Click 'Run replay' in the sidebar to run the stub pipeline.")
else:
    df = pd.DataFrame(history)
    latest = df.iloc[-1]

    # 1. Risk score over time
    st.subheader("Risk score over replay time")
    st.line_chart(df.set_index("t")["risk"])

    # 2. Current decision state
    st.subheader("Current decision state")
    st.markdown(f"## `{latest['state']}`")

    # 3. Four stream score/quality bars
    st.subheader("Stream scores and quality")
    stream_names = ["context", "motion", "physiology", "quality"]
    cols = st.columns(4)
    for col, name in zip(cols, stream_names):
        with col:
            st.metric(
                label=name.capitalize(),
                value=f"{latest[f'score_{name}']:.2f}",
            )
            st.progress(latest[f"quality_{name}"], text="quality")

    # 4. Ranked explanation panel
    st.subheader("Explanation (latest window)")
    st.code(latest["explanation"])

    # Raw table, useful for debugging while wiring is stub-only
    with st.expander("Raw replay history"):
        st.dataframe(df)