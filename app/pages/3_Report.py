"""
Guardian — System Report
Concise technical report summarizing multi-modal architecture, motion analysis,
activity context aggregation, and cross-dataset evaluation.
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
    page_title="Guardian — Report",
    layout="wide",
)

apply_theme()
render_sidebar_header()

# ==============================================================================
# PATHS & DATA CACHING (REUSING EXACT CORE ACTIVITY LOGIC)
# ==============================================================================

UMAFALL_ROOT = PROJECT_ROOT / "data" / "raw" / "UMAFall"
CONTEXT_MODEL_PATH = PROJECT_ROOT / "models" / "context.joblib"
MOTION_MODEL_PATH = PROJECT_ROOT / "models" / "motion.joblib"


@st.cache_data
def get_report_activity_aggregation(
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
# PAGE HEADER
# ==============================================================================

render_page_header(
    title="System Report",
    subtitle="Multi-modal architecture, motion analysis, activity context, and benchmark evaluation.",
    context_label="Technical Specification",
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

fall_files = [str(p) for p in list_fall_files(UMAFALL_ROOT)]

if not fall_files:
    st.error(f"No UMAFall fall CSV files found under {UMAFALL_ROOT}.")
    st.stop()

with st.spinner("Compiling system report and dataset metrics..."):
    result = get_report_activity_aggregation(
        fall_files_tuple=tuple(fall_files),
        dataset_root_str=str(UMAFALL_ROOT),
        context_model_path_str=str(CONTEXT_MODEL_PATH),
        motion_model_path_str=str(MOTION_MODEL_PATH),
        context_mtime_ns=CONTEXT_MODEL_PATH.stat().st_mtime_ns,
        motion_mtime_ns=MOTION_MODEL_PATH.stat().st_mtime_ns,
    )


st.markdown('<div class="g-section-title">1. System Overview</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="g-panel">
        <p style="margin: 0 0 0.65rem 0; font-size: 0.83rem; color: #5B6470; line-height: 1.5;">
            Guardian is an edge safety analytics architecture designed for continuous fall detection,
            activity context tracking, and physiological risk assessment from wearable sensor streams.
            The pipeline is built on an isolated-stream contract where independent evidence streams process standardized 2.5-second windows (125 samples @ 50 Hz, 50% overlap), each exposing its own continuous score, confidence, and signal quality metrics.
        </p>
        <div style="display: flex; flex-direction: column; gap: 0.45rem; font-size: 0.82rem; color: #5B6470; line-height: 1.45; border-top: 1px solid #DCDFE2; padding-top: 0.6rem;">
            <div><strong style="color: #1B222B;">Motion:</strong> Two-stage detection pipeline with a 2.5g impact gate, 18-feature Random Forest classifier, and 10s post-impact stillness tracking.</div>
            <div><strong style="color: #1B222B;">Activity:</strong> 4-state activity classification (ambulating, stationary, seated hand activity, lying/immobile) with temperature calibration and open-set rejection (implemented via the codebase's <code>ContextStream</code> class).</div>
            <div><strong style="color: #1B222B;">Physiological:</strong> Rule-based heart-rate deviation tracking with dwell time scaling (scripted trace; hardware integration pending).</div>
            <div><strong style="color: #1B222B;">Context:</strong> Derived multi-stream behavioral context signal across motion, activity, and physiology (designed, not built). <em>Note: distinct from the codebase's <code>ContextStream</code> class, which implements the 4-state Activity stream above.</em></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)



# ==============================================================================
# SECTION 2 — DATASET SUMMARY & COMPARISON (UMAFALL VS FALLALLD)
# ==============================================================================

st.markdown('<div class="g-section-title">2. Dataset Summary & Benchmark Comparison</div>', unsafe_allow_html=True)

col_d1, col_d2 = st.columns([1, 1])

with col_d1:
    st.markdown(
        """
        <div class="g-panel">
            <div style="font-size: 0.82rem; font-weight: 600; color: #1B222B; margin-bottom: 0.5rem;">Benchmark Cohort Comparison</div>
            <table style="width: 100%; font-size: 0.81rem; border-collapse: collapse; color: #1B222B;">
                <thead>
                    <tr style="border-bottom: 1px solid #DCDFE2; color: #5B6470; text-align: left;">
                        <th style="padding: 0.35rem 0.4rem; font-weight: 500;">Parameter</th>
                        <th style="padding: 0.35rem 0.4rem; font-weight: 500;">UMAFall</th>
                        <th style="padding: 0.35rem 0.4rem; font-weight: 500;">FallAllD</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #DCDFE2;">
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">Total Segments / Trials</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">746</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">2,515</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #DCDFE2;">
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">Fall Segments</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">208</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">523</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #DCDFE2;">
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">ADL Segments</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">538</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">1,992</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #DCDFE2;">
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">Subject Cohort</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">19 subjects</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">13 subjects</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #DCDFE2;">
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">Gate Threshold (SVM)</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">2.5 g</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">2.5 g</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #DCDFE2;">
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">Gate Fall Recall</td>
                        <td style="padding: 0.35rem 0.4rem; color: #2F6F62; font-weight: 500;" class="g-telemetry-num">84.1%</td>
                        <td style="padding: 0.35rem 0.4rem; color: #2F6F62; font-weight: 500;" class="g-telemetry-num">97.5%</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">Best Gated AUC</td>
                        <td style="padding: 0.35rem 0.4rem; font-weight: 500;" class="g-telemetry-num">0.950+</td>
                        <td style="padding: 0.35rem 0.4rem; font-weight: 500;" class="g-telemetry-num">0.653</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_d2:
    st.markdown(
        """
        <div class="g-panel">
            <div style="font-size: 0.82rem; font-weight: 600; color: #1B222B; margin-bottom: 0.5rem;">FallAllD Feature Set Evaluation</div>
            <table style="width: 100%; font-size: 0.81rem; border-collapse: collapse; color: #1B222B;">
                <thead>
                    <tr style="border-bottom: 1px solid #DCDFE2; color: #5B6470; text-align: left;">
                        <th style="padding: 0.35rem 0.4rem; font-weight: 500;">Feature Set</th>
                        <th style="padding: 0.35rem 0.4rem; font-weight: 500;">Features</th>
                        <th style="padding: 0.35rem 0.4rem; font-weight: 500;">Gate Recall</th>
                        <th style="padding: 0.35rem 0.4rem; font-weight: 500;">Gated AUC</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #DCDFE2; background-color: #F4F5F6;">
                        <td style="padding: 0.35rem 0.4rem; font-weight: 600; color: #2F6F62;">no_gyro (Rank 1)</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">13</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">97.5%</td>
                        <td style="padding: 0.35rem 0.4rem; font-weight: 600; color: #2F6F62;" class="g-telemetry-num">0.653</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #DCDFE2;">
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">robust_core (Rank 2)</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">10</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">97.5%</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">0.644</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #DCDFE2;">
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">all_18 (Rank 3)</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">18</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">97.5%</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">0.604</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #DCDFE2;">
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">no_jerk (Rank 4)</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">16</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">97.5%</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">0.602</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.35rem 0.4rem; color: #5B6470;">no_abs_tilt (Rank 5)</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">17</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">97.5%</td>
                        <td style="padding: 0.35rem 0.4rem;" class="g-telemetry-num">0.593</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 3 — MOTION MODULE
# ==============================================================================

st.markdown('<div class="g-section-title">3. Motion Module Architecture & Parameters</div>', unsafe_allow_html=True)

col_m1, col_m2 = st.columns([1, 1])

with col_m1:
    st.markdown(
        """
        <div class="g-panel">
            <div style="font-size: 0.82rem; font-weight: 600; color: #1B222B; margin-bottom: 0.4rem;">Pipeline Configuration</div>
            <ul style="margin: 0; padding-left: 1.1rem; font-size: 0.82rem; color: #5B6470; line-height: 1.5;">
                <li><strong style="color: #1B222B;">Stage 1 Gate Threshold:</strong> SVM Peak > <span class="g-telemetry-num">2.5 g</span> (wrist threshold). <em>Design justification:</em> Retains <span class="g-telemetry-num">91%</span> fall recall and <span class="g-telemetry-num">91%</span> coverage at <span class="g-telemetry-num">20 Hz</span>.</li>
                <li><strong style="color: #1B222B;">Gyroscope Gate Exclusion:</strong> Excluded from the Stage 1 gate due to high-acceleration ADLs at the wrist, but retained as a Stage 2 feature. <em>Gate-conditional ablation:</em> AUC increases from <span class="g-telemetry-num">0.584</span> to <span class="g-telemetry-num">0.679</span> (evaluated on a <span class="g-telemetry-num">300</span>-segment, <span class="g-telemetry-num">2</span>-subject FallAllD subset).</li>
                <li><strong style="color: #1B222B;">Stage 2 Classifier:</strong> Random Forest (<span class="g-telemetry-num">400</span> estimators, <code>min_samples_leaf=2</code>).</li>
                <li><strong style="color: #1B222B;">Segment Scope:</strong> <span class="g-telemetry-num">-2.0 s</span> pre-trigger to <span class="g-telemetry-num">+1.5 s</span> post-trigger. Centered on the <em>first gate crossing</em> rather than the global peak because a live stream has no oracle peak-finder.</li>
                <li><strong style="color: #1B222B;">Post-Impact Stillness:</strong> Forearm gravity tilt + SVM variance over <span class="g-telemetry-num">10.0 s</span>.</li>
                <li><strong style="color: #1B222B;">Score Combination:</strong> <span class="g-telemetry-num">0.6</span> * Impact + <span class="g-telemetry-num">0.4</span> * Stillness.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        """
        <div class="g-panel">
            <div style="font-size: 0.82rem; font-weight: 600; color: #1B222B; margin-bottom: 0.4rem;">Evaluation & Calibration</div>
            <ul style="margin: 0; padding-left: 1.1rem; font-size: 0.82rem; color: #5B6470; line-height: 1.5;">
                <li><strong style="color: #1B222B;">Validation Protocol:</strong> Leave-One-Subject-Out (LOSO) Cross-Validation.</li>
                <li><strong style="color: #1B222B;">Calibration Strategy:</strong> Platt / Temperature Scaling (<span class="g-telemetry-num">T &ge; 1.0</span>).</li>
                <li><strong style="color: #1B222B;">Contract Telemetry:</strong> Exposes <code>last_quality</code>, <code>last_impact</code>, <code>last_stillness</code>.</li>
                <li><strong style="color: #1B222B;">Event Hold / Decay:</strong> <span class="g-telemetry-num">60s</span> hold time with exponential decay (<span class="g-telemetry-num">&tau; = 30s</span>).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )



# ==============================================================================
# SECTION 4 — ACTIVITY MODULE (DYNAMIC AGGREGATE RESULTS)
# ==============================================================================

st.markdown('<div class="g-section-title">4. Activity Context Module — Impact Aggregation</div>', unsafe_allow_html=True)

# Analysis summary metrics from real pipeline
m_cols = st.columns(4)
m_cols[0].metric("Fall Records Scanned", result.scanned_records)
m_cols[1].metric("Records with Impact", result.records_with_impact)
m_cols[2].metric("Records without Impact", result.records_without_impact)
m_cols[3].metric("Excluded Records", result.records_skipped)

col_act_table, col_act_summary = st.columns([3, 2])

with col_act_table:
    table_rows = []
    for state in STATES:
        b = result.before_counts.get(state, 0)
        a = result.after_counts.get(state, 0)

        bp = 100.0 * b / result.before_total if result.before_total else 0.0
        ap = 100.0 * a / result.after_total if result.after_total else 0.0

        table_rows.append({
            "Activity State": state,
            "Before (Count)": b,
            "Before (%)": f"{bp:.1f}%",
            "After (Count)": a,
            "After (%)": f"{ap:.1f}%",
        })

    # Total row
    table_rows.append({
        "Activity State": "TOTAL",
        "Before (Count)": result.before_total,
        "Before (%)": "100.0%",
        "After (Count)": result.after_total,
        "After (%)": "100.0%",
    })

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

with col_act_summary:
    before_state = max(result.before_counts, key=result.before_counts.get) if result.before_total else "unknown"
    after_state = max(result.after_counts, key=result.after_counts.get) if result.after_total else "unknown"

    before_pct = 100.0 * result.before_counts[before_state] / result.before_total if result.before_total else 0.0
    after_pct = 100.0 * result.after_counts[after_state] / result.after_total if result.after_total else 0.0

    st.markdown(
        f"""
        <div class="g-panel" style="height: 100%;">
            <div style="font-size: 0.82rem; font-weight: 600; color: #1B222B; margin-bottom: 0.4rem;">Aggregation Semantics</div>
            <div style="font-size: 0.81rem; color: #5B6470; line-height: 1.5;">
                Context is classified in two fixed time horizons around the reconstructed impact timestamp (<code>t_impact</code>):
                <ul style="margin: 0.3rem 0; padding-left: 1.1rem;">
                    <li><strong>Before Impact:</strong> [t_impact - 2.5s, t_impact) &rarr; <span class="g-telemetry-num">{result.before_total}</span> windows</li>
                    <li><strong>After Impact:</strong> [t_impact, t_impact + 2.5s] &rarr; <span class="g-telemetry-num">{result.after_total}</span> windows</li>
                </ul>
                Modal baseline before impact: <strong style="color: #1B222B;">{before_state}</strong> (<span class="g-telemetry-num">{before_pct:.1f}%</span>).<br>
                Modal recovery state after impact: <strong style="color: #1B222B;">{after_state}</strong> (<span class="g-telemetry-num">{after_pct:.1f}%</span>).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# SECTION 5 — MOTION -> ACTIVITY PIPELINE FLOW
# ==============================================================================

st.markdown('<div class="g-section-title">5. Signal Processing Flow</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.4rem; background-color: #FFFFFF; border: 1px solid #DCDFE2; padding: 0.75rem 0.9rem; font-size: 0.8rem;">
        <div style="text-align: center; padding: 0.3rem 0.5rem; background-color: #F4F5F6; border: 1px solid #DCDFE2;">
            <div style="color: #5B6470; font-size: 0.7rem;">Input Stream</div>
            <div style="font-weight: 600; color: #1B222B;">Wrist IMU</div>
            <div style="color: #5B6470; font-size: 0.7rem;" class="g-telemetry-num">50 Hz Accel + Gyro</div>
        </div>
        <div style="color: #2F6F62; font-weight: 700;">&rarr;</div>
        <div style="text-align: center; padding: 0.3rem 0.5rem; background-color: #F4F5F6; border: 1px solid #DCDFE2;">
            <div style="color: #5B6470; font-size: 0.7rem;">Stage 1</div>
            <div style="font-weight: 600; color: #1B222B;">SVM Gate</div>
            <div style="color: #5B6470; font-size: 0.7rem;" class="g-telemetry-num">Peak > 2.5 g</div>
        </div>
        <div style="color: #2F6F62; font-weight: 700;">&rarr;</div>
        <div style="text-align: center; padding: 0.3rem 0.5rem; background-color: #F4F5F6; border: 1px solid #DCDFE2;">
            <div style="color: #5B6470; font-size: 0.7rem;">Stage 2</div>
            <div style="font-weight: 600; color: #1B222B;">Motion RF</div>
            <div style="color: #5B6470; font-size: 0.7rem;" class="g-telemetry-num">18 Feats ([-2s, +1.5s])</div>
        </div>
        <div style="color: #2F6F62; font-weight: 700;">&rarr;</div>
        <div style="text-align: center; padding: 0.3rem 0.5rem; background-color: #F4F5F6; border: 1px solid #DCDFE2;">
            <div style="color: #5B6470; font-size: 0.7rem;">Clock</div>
            <div style="font-weight: 600; color: #1B222B;">Impact Time</div>
            <div style="color: #5B6470; font-size: 0.7rem;" class="g-telemetry-num">internal_now - age</div>
        </div>
        <div style="color: #2F6F62; font-weight: 700;">&rarr;</div>
        <div style="text-align: center; padding: 0.3rem 0.5rem; background-color: #F4F5F6; border: 1px solid #DCDFE2;">
            <div style="color: #5B6470; font-size: 0.7rem;">Context</div>
            <div style="font-weight: 600; color: #1B222B;">4-State Slicing</div>
            <div style="color: #5B6470; font-size: 0.7rem;" class="g-telemetry-num">[-2.5s, 0) & [0, +2.5s]</div>
        </div>
        <div style="color: #2F6F62; font-weight: 700;">&rarr;</div>
        <div style="text-align: center; padding: 0.3rem 0.5rem; background-color: #F4F5F6; border: 1px solid #DCDFE2;">
            <div style="color: #5B6470; font-size: 0.7rem;">Synthesis</div>
            <div style="font-weight: 600; color: #2F6F62;">Aggregate</div>
            <div style="color: #5B6470; font-size: 0.7rem;">Context Distribution</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# SECTION 6 — KEY FINDINGS (DYNAMICALLY DERIVED)
# ==============================================================================

st.markdown('<div class="g-section-title">6. Key Findings</div>', unsafe_allow_html=True)

readable_falls = result.scanned_records - result.records_skipped
impact_rate = (100.0 * result.records_with_impact / readable_falls) if readable_falls else 0.0

st.markdown(
    f"""
    <div class="g-panel">
        <ul style="margin: 0; padding-left: 1.15rem; font-size: 0.83rem; color: #5B6470; line-height: 1.6;">
            <li><strong style="color: #1B222B;">Impact Detection Rate:</strong> <span class="g-telemetry-num">{result.records_with_impact}</span> of <span class="g-telemetry-num">{readable_falls}</span> readable fall recordings (<span class="g-telemetry-num">{impact_rate:.1f}%</span>) triggered the MotionStream impact gate with valid timestamp reconstruction.</li>
            <li><strong style="color: #1B222B;">Pre-Impact Baseline:</strong> <span style="color: #1B222B; font-weight: 500;">{before_state}</span> accounted for <span class="g-telemetry-num">{before_pct:.1f}%</span> of all classified pre-impact windows (<span class="g-telemetry-num">{result.before_counts.get(before_state, 0)}</span> of <span class="g-telemetry-num">{result.before_total}</span>), indicating active ambulation prior to impact.</li>
            <li><strong style="color: #1B222B;">Post-Impact Transition:</strong> Ambulation drops post-impact, while <span style="color: #1B222B; font-weight: 500;">{after_state}</span> rises to <span class="g-telemetry-num">{after_pct:.1f}%</span> and lying/immobile accounts for <span class="g-telemetry-num">{100.0 * result.after_counts.get('lying/immobile', 0) / result.after_total:.1f}%</span> (<span class="g-telemetry-num">{result.after_counts.get('lying/immobile', 0)}</span> windows).</li>
            <li><strong style="color: #1B222B;">Cross-Dataset Generalizability:</strong> FallAllD benchmark analysis provides directional support that models omitting gyroscopic rotational rate features (<code>no_gyro</code>, 13 features) achieve higher cross-dataset generalizability (<span class="g-telemetry-num">AUC = 0.653</span>) than the full 18-feature model (<span class="g-telemetry-num">AUC = 0.604</span>). <em>Caveat:</em> Evaluated on a limited subset (<span class="g-telemetry-num">3</span> of <span class="g-telemetry-num">13</span> subjects evaluated &mdash; indicative, not conclusive).</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# SECTION 7 — LIMITATIONS & DATA COVERAGE DISCLOSURE
# ==============================================================================

st.markdown('<div class="g-section-title">7. Limitations & Data Coverage Disclosure</div>', unsafe_allow_html=True)

skipped_pct = (100.0 * result.records_skipped / result.scanned_records) if result.scanned_records else 0.0

st.markdown(
    f"""
    <div class="g-panel" style="font-size: 0.82rem; color: #5B6470; line-height: 1.55;">
        <ul style="margin: 0; padding-left: 1.15rem;">
            <li style="margin-bottom: 0.45rem;">
                <strong style="color: #1B222B;">Hardware & Sensor Coverage Exclusion:</strong>
                {result.records_skipped} fall records ({skipped_pct:.1f}% of the scanned cohort) could not be evaluated because raw UMAFall files lack wrist sensor channels (<code>SensorID=2</code>) and contain only pocket, chest, waist, or ankle sensor nodes. In accordance with clinical validation standards, these records are excluded without data fabrication or sensor-node reinterpretation.
            </li>
            <li style="margin-bottom: 0.45rem;">
                <strong style="color: #1B222B;">Staged vs. Real-World Evaluation:</strong>
                Benchmark Motion AUC metrics (e.g. <span class="g-telemetry-num">0.998</span>) reflect staged, mattress-cushioned falls in controlled laboratory settings. Continuous real-world monitoring yields lower sensitivity and higher false alarm rates (Bagal&agrave; et al., 2012: <span class="g-telemetry-num">57%</span> mean sensitivity on real-world falls, <span class="g-telemetry-num">3&ndash;85</span> false alarms/day).
            </li>
            <li style="margin-bottom: 0.45rem;">
                <strong style="color: #1B222B;">Sampling Rate Uniformity:</strong>
                Sensor streams were standardized via upsampling to a uniform <span class="g-telemetry-num">50 Hz</span> grid rather than recorded natively at that fixed rate across all trials.
            </li>
            <li>
                <strong style="color: #1B222B;">Sensor Placement Tradeoff:</strong>
                The wrist location was chosen for high longitudinal user adherence rather than optimal impact biomechanics (the trunk/waist provides superior fall dynamics but lower continuous wear adherence).
            </li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)


