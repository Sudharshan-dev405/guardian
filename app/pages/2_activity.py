"""
Guardian — Activity Module
Aggregate ContextStream activity states before and after MotionStream impact.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# ==============================================================================
# PROJECT PATH & IMPORTS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ui_theme import apply_theme, render_page_header, render_sidebar_header
from core.activity import (
    POST_SEC,
    PRE_SEC,
    STATES,
    ActivityAggregateResult,
    aggregate_activity_dataset,
    list_fall_files,
)

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Guardian — Activity",
    layout="wide",
)

apply_theme()
render_sidebar_header()

# ==============================================================================
# PATHS
# ==============================================================================

UMAFALL_ROOT = PROJECT_ROOT / "data" / "raw" / "UMAFall"
CONTEXT_MODEL_PATH = PROJECT_ROOT / "models" / "context.joblib"
MOTION_MODEL_PATH = PROJECT_ROOT / "models" / "motion.joblib"


# ==============================================================================
# DATA CACHING (PRESERVED EXACTLY)
# ==============================================================================

@st.cache_data
def get_fall_files() -> list[str]:
    return [str(p) for p in list_fall_files(UMAFALL_ROOT)]


@st.cache_data
def run_activity_aggregation(
    fall_files_tuple: tuple[str, ...],
    dataset_root_str: str,
    context_model_path_str: str,
    motion_model_path_str: str,
    context_mtime_ns: int,
    motion_mtime_ns: int,
) -> ActivityAggregateResult:
    del context_mtime_ns, motion_mtime_ns

    return aggregate_activity_dataset(
        dataset_root=dataset_root_str,
        motion_model_path=motion_model_path_str,
        context_model_path=context_model_path_str,
        fall_files=fall_files_tuple,
    )


# ==============================================================================
# VALIDATION
# ==============================================================================

if not CONTEXT_MODEL_PATH.exists():
    st.error(f"Missing context model: {CONTEXT_MODEL_PATH}")
    st.stop()

if not MOTION_MODEL_PATH.exists():
    st.error(f"Missing motion model: {MOTION_MODEL_PATH}")
    st.stop()

fall_files = get_fall_files()

if not fall_files:
    st.error(f"No UMAFall fall CSV files found under {UMAFALL_ROOT}.")
    st.stop()


# ==============================================================================
# PAGE HEADER
# ==============================================================================

render_page_header(
    title="Activity Context",
    subtitle="Contextual activity state distribution immediately before and after MotionStream-detected impacts.",
    context_label="UMAFall Benchmark",
)


# ==============================================================================
# RUN DATASET AGGREGATION
# ==============================================================================

with st.spinner("Aggregating activity context across all UMAFall fall recordings..."):
    result = run_activity_aggregation(
        fall_files_tuple=tuple(fall_files),
        dataset_root_str=str(UMAFALL_ROOT),
        context_model_path_str=str(CONTEXT_MODEL_PATH),
        motion_model_path_str=str(MOTION_MODEL_PATH),
        context_mtime_ns=CONTEXT_MODEL_PATH.stat().st_mtime_ns,
        motion_mtime_ns=MOTION_MODEL_PATH.stat().st_mtime_ns,
    )


# ==============================================================================
# ANALYSIS SUMMARY (KPI HIERARCHY)
# ==============================================================================

st.markdown('<div class="g-section-title">Dataset Impact & Context Summary</div>', unsafe_allow_html=True)

# Primary cohort metrics
kpi_cols = st.columns(4)
kpi_cols[0].metric("Fall Records Scanned", result.scanned_records)
kpi_cols[1].metric("Impacts Detected", result.records_with_impact)
kpi_cols[2].metric("No Impact Detected", result.records_without_impact)
kpi_cols[3].metric("Excluded Records", result.records_skipped)

# Secondary window metrics
win_cols = st.columns(2)
win_cols[0].metric("Pre-Impact Windows (2.5 s)", f"{result.before_total:,}")
win_cols[1].metric("Post-Impact Windows (2.5 s)", f"{result.after_total:,}")

# Informational panel for skipped records
if result.records_skipped:
    st.markdown(
        f"""
        <div class="g-panel-alert" style="margin-top: 0.6rem; margin-bottom: 0.85rem;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #B8752B; margin-bottom: 0.2rem;">
                Hardware Exclusion Note: {result.records_skipped} Records Skipped
            </div>
            <div style="font-size: 0.82rem; color: #5B6470; line-height: 1.45;">
                {result.records_skipped} fall recordings in UMAFall lack the required wrist IMU sensor (<code>SensorID=2</code>) and contain only chest, pocket, waist, or ankle nodes. These records are excluded without data fabrication.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# CONTEXT INTERPRETATION INSIGHT CARD
# ==============================================================================

if result.before_total and result.after_total:
    before_state = max(result.before_counts, key=result.before_counts.get)
    after_state = max(result.after_counts, key=result.after_counts.get)

    before_pct = (
        100.0 * result.before_counts[before_state] / result.before_total
    )
    after_pct = (
        100.0 * result.after_counts[after_state] / result.after_total
    )

    st.markdown(
        f"""
        <div class="g-panel" style="margin-top: 0.75rem;">
            <div style="font-size: 0.82rem; font-weight: 600; color: #1B222B; margin-bottom: 0.6rem;">
                Contextual State Summary
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
                <div style="border-left: 2px solid #2F6F62; padding-left: 0.75rem;">
                    <div style="font-size: 0.74rem; font-weight: 500; color: #5B6470;">Before Impact (2.5 s)</div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #1B222B; margin: 0.15rem 0;">
                        {before_state} <span class="g-telemetry-num" style="font-size: 0.88rem; font-weight: 500; color: #2F6F62;">({before_pct:.1f}%)</span>
                    </div>
                    <div style="font-size: 0.81rem; color: #5B6470; line-height: 1.4;">Dominant pre-fall baseline indicating active ambulation prior to impact.</div>
                </div>
                <div style="border-left: 2px solid #5B6470; padding-left: 0.75rem;">
                    <div style="font-size: 0.74rem; font-weight: 500; color: #5B6470;">After Impact (2.5 s)</div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #1B222B; margin: 0.15rem 0;">
                        {after_state} <span class="g-telemetry-num" style="font-size: 0.88rem; font-weight: 500; color: #5B6470;">({after_pct:.1f}%)</span>
                    </div>
                    <div style="font-size: 0.81rem; color: #5B6470; line-height: 1.4;">Post-fall transition reflecting ground-level movement and recovery attempts.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# AGGREGATE DISTRIBUTION & COMPARISON CHART
# ==============================================================================

st.markdown('<div class="g-section-title">State Distribution Before vs After Impact</div>', unsafe_allow_html=True)

col_table, col_chart = st.columns([1, 1])

with col_table:
    st.markdown('<div style="font-size: 0.82rem; font-weight: 600; color: #1B222B; margin-bottom: 0.4rem;">Aggregate Distribution Table</div>', unsafe_allow_html=True)

    table_rows = []
    for state in STATES:
        b = result.before_counts.get(state, 0)
        a = result.after_counts.get(state, 0)

        bp = 100.0 * b / result.before_total if result.before_total else 0.0
        ap = 100.0 * a / result.after_total if result.after_total else 0.0

        table_rows.append({
            "State": state,
            "Before (Windows)": b,
            "Before (%)": f"{bp:.1f}%",
            "After (Windows)": a,
            "After (%)": f"{ap:.1f}%",
        })

    aggregate_df = pd.DataFrame(table_rows)

    st.dataframe(
        aggregate_df,
        use_container_width=True,
        hide_index=True,
    )

with col_chart:
    st.markdown('<div style="font-size: 0.82rem; font-weight: 600; color: #1B222B; margin-bottom: 0.4rem;">State Distribution Before vs After Impact (%)</div>', unsafe_allow_html=True)

    graph_df = pd.DataFrame(
        {
            "Before Impact": [
                100.0 * result.before_counts.get(state, 0) / result.before_total
                if result.before_total else 0.0
                for state in STATES
            ],
            "After Impact": [
                100.0 * result.after_counts.get(state, 0) / result.after_total
                if result.after_total else 0.0
                for state in STATES
            ],
        },
        index=list(STATES),
    )

    st.bar_chart(graph_df, height=300)


# ==============================================================================
# RECORD-LEVEL INSPECTION EXPANDER
# ==============================================================================

with st.expander("Record-Level Telemetry Data (208 Total Fall Records)"):
    st.caption(
        "Trial-by-trial analysis telemetry across all scanned UMAFall fall recordings. "
        "Records with detected impacts contribute directly to the 2.5s context aggregate."
    )

    if not result.record_rows.empty:
        st.dataframe(
            result.record_rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No record-level results available.")
