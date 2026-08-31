"""
Guardian — Main Dashboard

Landing page for the three independent Guardian modules.
"""

import streamlit as st


# ==========================================================================
# PAGE CONFIGURATION
# ==========================================================================

st.set_page_config(
    page_title="Guardian",
    page_icon="🛡️",
    layout="wide",
)


# ==========================================================================
# HEADER
# ==========================================================================

st.title("Guardian")

st.write(
    "A modular system for analysing motion, activity, "
    "and physiological signals independently."
)

st.divider()


# ==========================================================================
# MODULES
# ==========================================================================

st.subheader("Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Motion Module")
    st.write("Dataset: UMAFall")
    st.write(
        "Analyses motion and impact characteristics "
        "using wrist IMU data."
    )

with col2:
    st.markdown("### Activity Module")
    st.write("Dataset: WEDA-FALL")
    st.write(
        "Analyses activity and contextual behaviour "
        "using the WEDA-FALL dataset."
    )

with col3:
    st.markdown("### Physiology Module")
    st.write("Physiological data")
    st.write(
        "Analyses physiological signals using the "
        "physiology model."
    )


st.divider()

st.caption(
    "Select a module from the sidebar to view its individual analysis."
)