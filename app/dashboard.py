"""
app/dashboard.py

Streamlit dashboard, updated to use the REAL fusion/decision/explain
implementations in place of the stubs.

Evidence streams shown: context, motion, physiology -- three streams,
per this review's scope. streams/quality.py is not one of them here
and is not called by this file.

Context/motion are still the stub classes at this point (owned by a
teammate, not modified here). Real ContextStream/MotionStream will
drop in later without requiring changes to this file, because fusion
reads their attributes via getattr() with safe defaults.

Physiology needs the current context label to apply its motion-gating
rule. Per fusion's design, context labels are never put into the
window dict for FUSION's sake -- but physiology.py (unchanged, already
implemented) reads context_state from window.get("context_state").
That's satisfied here by passing physiology a shallow COPY of the
window with context_state added, while the original window (and what
fusion receives) stays untouched. This is orchestration glue local to
this file, not a change to the window contract or to fusion's inputs.
"""

import streamlit as st
import pandas as pd

from data.loader import replay_stub
from streams.context import ContextStream
from streams.motion import MotionStream
from streams.physiology import PhysiologyStream
from core.fusion import fuse
from core.decision import DecisionStateMachine
from core.explain import explain


st.set_page_config(page_title="Guardian Dashboard", layout="wide")
st.title("Guardian — Replay Dashboard")
st.caption(
    "Context and motion are still stub streams pending the teammate branch merge. "
    "Physiology is real. Fusion, decision, and explanation are real."
)
st.info("Physiology uses a **scripted physiological trace — hardware pending**. "
        "It is not real sensor data.")

# --- Sidebar controls -------------------------------------------------

st.sidebar.header("Replay controls")

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

# --- Session state ------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --- Run the real pipeline -----------------------------------------------

if run_clicked:
    context_stream = ContextStream()
    motion_stream = MotionStream()
    physiology_stream = PhysiologyStream()
    decision_machine = DecisionStateMachine()

    history = []
    for window in replay_stub(n_windows=n_windows):
        context_score = context_stream.score(window)
        motion_score = motion_stream.score(window)

        phys_window = dict(window)
        phys_window["context_state"] = getattr(context_stream, "last_state", None)
        physiology_score = physiology_stream.score(phys_window)

        scores = {
            "context": context_score,
            "motion": motion_score,
            "physiology": physiology_score,
        }

        risk, contributions = fuse(scores, context_stream, motion_stream)
        state = decision_machine.update(risk, window["t"], contributions=contributions)
        explanation = explain(risk, contributions)

        history.append(
            {
                "t": window["t"],
                "risk": risk,
                "state": state,
                "explanation": explanation,
                "score_context": context_score,
                "score_motion": motion_score,
                "score_physiology": physiology_score,
                "contrib_context": contributions.get("context"),
                "contrib_motion": contributions.get("motion"),
                "contrib_physiology": contributions.get("physiology"),
            }
        )

    st.session_state.history = history

# --- Display ------------------------------------------------------------

history = st.session_state.history

if not history:
    st.info("Click 'Run replay' in the sidebar to run the pipeline.")
else:
    df = pd.DataFrame(history)
    latest = df.iloc[-1]

    st.subheader("Risk score over replay time")
    st.line_chart(df.set_index("t")["risk"])

    st.subheader("Current decision state")
    st.markdown(f"## `{latest['state']}`")

    st.subheader("Stream scores")
    stream_names = ["context", "motion", "physiology"]
    cols = st.columns(3)
    for col, name in zip(cols, stream_names):
        with col:
            st.metric(label=name.capitalize(), value=f"{latest[f'score_{name}']:.2f}")

    st.subheader("Explanation (latest window)")
    st.code(latest["explanation"])

    with st.expander("Raw replay history"):
        st.dataframe(df)