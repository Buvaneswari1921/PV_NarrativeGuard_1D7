"""PV NarrativeGuard - Dashboard page."""

import streamlit as st

from src.ui.data_loader import load_application_data
from src.ui.dashboard import render_dashboard
from src.ui.components import render_global_styles


st.set_page_config(
    page_title="PV NarrativeGuard | Dashboard",
    page_icon="🛡️",
    layout="wide",
)

render_global_styles()

data = load_application_data()

render_dashboard(data)