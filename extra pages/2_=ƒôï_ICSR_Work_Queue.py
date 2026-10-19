"""PV NarrativeGuard - ICSR Work Queue."""

import streamlit as st

from src.ui.data_loader import load_application_data
from src.ui.work_queue import render_work_queue
from src.ui.components import render_global_styles


st.set_page_config(
    page_title="PV NarrativeGuard | ICSR Work Queue",
    page_icon="📋",
    layout="wide",
)

render_global_styles()

data = load_application_data()

render_work_queue(data)