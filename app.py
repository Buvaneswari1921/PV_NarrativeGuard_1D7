"""
PV NarrativeGuard - Application Entry Point

The application opens directly on the Dashboard.
All functional pages are maintained under /pages.
"""

from pathlib import Path

import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PV NarrativeGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DASHBOARD LANDING PAGE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"

dashboard_candidates = list(
    PAGES_DIR.glob("1_*Dashboard.py")
)

if not dashboard_candidates:
    st.error(
        "PV NarrativeGuard Dashboard could not be located."
    )
    st.stop()

st.switch_page(
    str(
        dashboard_candidates[0]
        .relative_to(BASE_DIR)
    )
)