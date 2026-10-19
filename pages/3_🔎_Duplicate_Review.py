"""PV NarrativeGuard - Duplicate Review."""

import streamlit as st

from src.ui.data_loader import load_application_data
from src.ui.duplicate_review import render_duplicate_review
from src.ui.components import render_global_styles


st.set_page_config(
    page_title="PV NarrativeGuard | Duplicate Review",
    page_icon="🔎",
    layout="wide",
)

render_global_styles()


data = load_application_data()

render_duplicate_review(data)