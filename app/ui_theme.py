"""
app/ui_theme.py -- Guardian Shared UI & Design System Components.

Provides styling, CSS overrides, and reusable visual components for the
Guardian wearable fall-detection instrumentation console.
All components adhere to the light technical token system and strict typography.
"""

from __future__ import annotations

import streamlit as st

# ==============================================================================
# GLOBAL CSS THEME INJECTION (LIGHT TECHNICAL INSTRUMENTATION CONSOLE)
# ==============================================================================

CUSTOM_CSS = """
<style>
/* -------------------------------------------------------------------------- */
/* Typography & Reset                                                         */
/* -------------------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1B222B;
    background-color: #F4F5F6;
}

/* Base app container */
.stApp {
    background-color: #F4F5F6 !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1240px !important;
}

/* -------------------------------------------------------------------------- */
/* Sidebar Polish                                                             */
/* -------------------------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background-color: #ECEEF0 !important;
    border-right: 1px solid #DCDFE2 !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.25rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

[data-testid="stSidebarNav"] {
    padding-top: 0.25rem;
}

[data-testid="stSidebarNav"] ul {
    gap: 0.15rem;
}

[data-testid="stSidebarNav"] a {
    border-radius: 0px !important;
    padding: 0.45rem 0.75rem !important;
    color: #5B6470 !important;
    font-weight: 400 !important;
    font-size: 0.84rem !important;
    border-left: 2px solid transparent !important;
    transition: background-color 0.1s ease, color 0.1s ease !important;
}

[data-testid="stSidebarNav"] a:hover {
    background-color: #E2E5E8 !important;
    color: #1B222B !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: #FFFFFF !important;
    color: #1B222B !important;
    border-left: 2px solid #2F6F62 !important;
    font-weight: 500 !important;
}

/* -------------------------------------------------------------------------- */
/* Flat Instrumentation Panels & Containers                                   */
/* -------------------------------------------------------------------------- */
.g-brand {
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #1B222B;
    margin: 0 0 0.85rem 0;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid #DCDFE2;
}

.g-panel {
    background-color: #FFFFFF;
    border: 1px solid #DCDFE2;
    border-radius: 0px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    box-shadow: none;
}

.g-panel-normal {
    background-color: #FFFFFF;
    border: 1px solid #DCDFE2;
    border-left: 3px solid #2F6F62;
    border-radius: 0px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    box-shadow: none;
}

.g-panel-alert {
    background-color: #FFFFFF;
    border: 1px solid #DCDFE2;
    border-left: 3px solid #B8752B;
    border-radius: 0px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    box-shadow: none;
}

/* -------------------------------------------------------------------------- */
/* Header & Section Typography                                                */
/* -------------------------------------------------------------------------- */
.g-page-header {
    margin-bottom: 1.1rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid #DCDFE2;
}

.g-page-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #1B222B;
    letter-spacing: -0.015em;
    margin: 0 0 0.2rem 0;
}

.g-page-desc {
    font-size: 0.84rem;
    color: #5B6470;
    margin: 0;
    line-height: 1.45;
}

.g-section-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #1B222B;
    letter-spacing: 0;
    margin: 1.25rem 0 0.5rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #DCDFE2;
}

.g-hairline {
    border-bottom: 1px solid #DCDFE2;
    margin: 0.75rem 0;
}

/* -------------------------------------------------------------------------- */
/* State Indicators & Telemetry Readouts                                      */
/* -------------------------------------------------------------------------- */
.g-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    font-weight: 500;
}

.g-indicator-normal {
    color: #2F6F62;
}

.g-indicator-alert {
    color: #B8752B;
    font-weight: 600;
}

.g-indicator-neutral {
    color: #5B6470;
}

.g-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
}

.g-dot-normal {
    background-color: #2F6F62;
}

.g-dot-alert {
    background-color: #B8752B;
}

.g-dot-neutral {
    background-color: #5B6470;
}

.g-telemetry-num {
    font-family: 'IBM Plex Mono', monospace;
    color: #1B222B;
}

/* -------------------------------------------------------------------------- */
/* KPI Metric Component Styling                                               */
/* -------------------------------------------------------------------------- */
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCDFE2 !important;
    border-radius: 0px !important;
    padding: 0.7rem 0.85rem !important;
    box-shadow: none !important;
}

div[data-testid="stMetric"] label {
    font-size: 0.74rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: #5B6470 !important;
    margin-bottom: 0.15rem !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #1B222B !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* -------------------------------------------------------------------------- */
/* Streamlit Alerts, Expanders & Inputs                                       */
/* -------------------------------------------------------------------------- */
div[data-testid="stAlert"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCDFE2 !important;
    border-left: 3px solid #B8752B !important;
    border-radius: 0px !important;
    padding: 0.65rem 0.85rem !important;
    color: #1B222B !important;
}

div[data-testid="stAlert"] p {
    color: #1B222B !important;
    font-size: 0.82rem !important;
}

div[data-testid="stExpander"] {
    border: 1px solid #DCDFE2 !important;
    border-radius: 0px !important;
    background-color: #FFFFFF !important;
    box-shadow: none !important;
}

.streamlit-expanderHeader {
    background-color: #FFFFFF !important;
    border-radius: 0px !important;
    color: #1B222B !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
}

/* Dataframe & Selectbox refinement */
div[data-testid="stDataFrame"] {
    border: 1px solid #DCDFE2 !important;
    border-radius: 0px !important;
    overflow: hidden !important;
}

div[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF !important;
    border: 1px solid #DCDFE2 !important;
    border-radius: 0px !important;
    color: #1B222B !important;
    font-size: 0.84rem !important;
}

code {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    background-color: #ECEEF0 !important;
    color: #1B222B !important;
    padding: 0.1rem 0.3rem !important;
    border-radius: 0px !important;
}

hr {
    border: none !important;
    border-top: 1px solid #DCDFE2 !important;
    margin: 1rem 0 !important;
}
</style>
"""


def apply_theme():
    """Inject the Guardian CSS design system into the current Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar_header():
    """Render Guardian instrumentation branding in the sidebar."""
    with st.sidebar:
        st.markdown(
            '<div class="g-brand">Guardian.</div>',
            unsafe_allow_html=True,
        )


def render_page_header(title: str, subtitle: str | None = None, context_label: str | None = None):
    """Render a clean left-aligned instrumentation page header."""
    header_html = f'<div class="g-page-header">'
    header_html += f'<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem;">'
    header_html += f'<div><h1 class="g-page-title">{title}</h1>'
    if subtitle:
        header_html += f'<p class="g-page-desc">{subtitle}</p>'
    header_html += f'</div>'
    if context_label:
        header_html += f'<div style="font-size: 0.8rem; color: #5B6470;">{context_label}</div>'
    header_html += f'</div></div>'

    st.markdown(header_html, unsafe_allow_html=True)

