"""PV NarrativeGuard - Narrative Generation."""

import streamlit as st

from src.ui.data_loader import load_application_data
from src.ui.narrative_generation import render_narrative_generation
from src.ui.components import render_global_styles


st.set_page_config(
    page_title="PV NarrativeGuard | Narrative Generation",
    page_icon="✍️",
    layout="wide",
)

render_global_styles()


data = load_application_data()

render_narrative_generation(data)