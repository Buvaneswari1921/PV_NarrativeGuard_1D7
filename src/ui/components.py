"""Reusable visual components for the PV NarrativeGuard Streamlit UI.

The visual layer is intentionally presentation-only.  It does not change
pipeline decisions, eligibility, persistence, or reviewer controls.
"""

from html import escape

import streamlit as st


def render_global_styles() -> None:
    """Apply the shared NarrativeGuard visual system to the current page."""
    st.markdown(
        """
        <style>
        :root {
          --ng-bg: #0b1220;
          --ng-panel: #111d30;
          --ng-panel-2: #15243a;
          --ng-border: rgba(151, 180, 213, .17);
          --ng-text: #edf4ff;
          --ng-muted: #9fb0c7;
          --ng-teal: #62e4d1;
          --ng-blue: #7aa7ff;
          --ng-amber: #f2bd68;
        }
        .stApp {
          background:
            radial-gradient(ellipse at 78% -10%, rgba(37, 99, 150, .20), transparent 34%),
            radial-gradient(ellipse at 0% 40%, rgba(24, 113, 119, .10), transparent 30%),
            var(--ng-bg);
          color: var(--ng-text);
        }
        [data-testid="stHeader"] { background: rgba(11,18,32,.76); }
        [data-testid="stToolbar"] { right: 1rem; }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #0d192b 0%, #0a1321 100%);
          border-right: 1px solid var(--ng-border);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
          color: #c6d5e8;
        }
        [data-testid="stSidebarNav"] { padding-top: .4rem; }
        [data-testid="stSidebarNav"] a {
          border-radius: 10px;
          margin: 3px 8px;
          transition: background .18s ease, transform .18s ease;
        }
        [data-testid="stSidebarNav"] a:hover {
          background: rgba(98,228,209,.09);
          transform: translateX(2px);
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
          background: linear-gradient(90deg, rgba(98,228,209,.17), rgba(122,167,255,.08));
          border: 1px solid rgba(98,228,209,.24);
        }
        .ng-sidebar-brand {
          display: flex; align-items: center; gap: 10px;
          padding: 12px 13px; margin: 8px 4px 18px;
          border: 1px solid rgba(98,228,209,.20);
          border-radius: 14px;
          background: linear-gradient(135deg, rgba(98,228,209,.10), rgba(122,167,255,.06));
        }
        .ng-sidebar-mark {
          width: 35px; height: 35px; flex: 0 0 35px;
          display: grid; place-items: center; border-radius: 11px;
          color: #071621; font-size: 1rem; font-weight: 900;
          background: linear-gradient(135deg, #78eedc, #7aa7ff);
        }
        .ng-sidebar-brand strong {
          display: block; color: #edf7ff; font-size: .82rem; letter-spacing: -.01em;
        }
        .ng-sidebar-brand small {
          display: block; color: #8ea5c0; font-size: .67rem; margin-top: 2px;
        }
        .block-container {
          max-width: 1560px;
          padding-top: 2.1rem;
          padding-bottom: 3.2rem;
        }
        h1, h2, h3, h4 { letter-spacing: -.025em; }
        h2 { margin-top: 1.25rem !important; }
        [data-testid="stCaptionContainer"] { color: var(--ng-muted); }
        [data-testid="stMetric"] {
          background: linear-gradient(145deg, rgba(21,36,58,.94), rgba(15,27,44,.94));
          border: 1px solid var(--ng-border);
          border-radius: 16px;
          padding: 15px 17px;
          box-shadow: 0 12px 30px rgba(0,0,0,.10);
          min-height: 102px;
        }
        [data-testid="stMetricLabel"] { color: #a9bdd5; font-size: .82rem; }
        [data-testid="stMetricValue"] { color: #f1f7ff; font-weight: 720; }
        [data-testid="stMetricDelta"] { font-size: .76rem; }
        [data-testid="stVerticalBlockBorderWrapper"] {
          border-color: var(--ng-border) !important;
          border-radius: 16px !important;
          background: rgba(17,29,48,.55);
        }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
          border: 1px solid var(--ng-border);
          border-radius: 12px;
          overflow: hidden;
        }
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input {
          background: #0e1a2c;
          border-color: #2b405b;
          border-radius: 10px;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
          background: #0e1a2c;
          border-color: #2b405b;
          border-radius: 10px;
        }
        .stButton > button, [data-testid="stDownloadButton"] button {
          border-radius: 10px;
          border: 1px solid rgba(98,228,209,.30);
          transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
          font-weight: 650;
        }
        .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
          border-color: var(--ng-teal);
          box-shadow: 0 0 0 3px rgba(98,228,209,.08);
          transform: translateY(-1px);
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        [data-testid="stTabs"] [role="tablist"] {
          gap: 5px;
          border-bottom: 1px solid var(--ng-border);
        }
        [data-testid="stTabs"] button[role="tab"] {
          border-radius: 9px 9px 0 0;
          padding: .65rem 1rem;
        }
        .ng-hero {
          position: relative;
          overflow: hidden;
          border: 1px solid rgba(122,167,255,.20);
          border-radius: 22px;
          padding: 25px 28px 23px;
          margin: 0 0 1.15rem 0;
          background:
            radial-gradient(circle at 88% 10%, rgba(98,228,209,.17), transparent 24%),
            radial-gradient(circle at 100% 100%, rgba(122,167,255,.14), transparent 34%),
            linear-gradient(115deg, #122139 0%, #101c30 60%, #102a39 100%);
          box-shadow: 0 18px 48px rgba(0,0,0,.17);
          animation: ng-rise .42s ease both;
        }
        .ng-hero:after {
          content: "";
          position: absolute;
          width: 220px; height: 220px;
          right: -86px; top: -130px;
          border: 1px solid rgba(98,228,209,.20);
          border-radius: 50%;
          box-shadow: 0 0 0 22px rgba(98,228,209,.025), 0 0 0 48px rgba(122,167,255,.025);
          pointer-events: none;
        }
        .ng-eyebrow {
          display: flex; align-items: center; gap: 9px;
          color: #7de8d8; text-transform: uppercase;
          font-size: .70rem; letter-spacing: .16em; font-weight: 800;
          margin-bottom: 9px;
        }
        .ng-pulse {
          width: 7px; height: 7px; border-radius: 50%;
          background: #62e4d1; box-shadow: 0 0 0 4px rgba(98,228,209,.12);
        }
        .ng-hero h1 {
          color: #f3f7ff; font-size: clamp(1.8rem, 3vw, 2.45rem);
          line-height: 1.12; font-weight: 760; margin: 0 0 9px 0;
          letter-spacing: -.045em;
        }
        .ng-hero p {
          color: #b4c5db; font-size: .98rem; line-height: 1.55;
          max-width: 820px; margin: 0;
        }
        .ng-hero-meta {
          display: flex; gap: 8px; flex-wrap: wrap; margin-top: 17px;
        }
        .ng-chip {
          display: inline-flex; align-items: center; gap: 6px;
          border: 1px solid rgba(168,194,224,.19);
          background: rgba(5,13,25,.22); color: #c8d7e9;
          border-radius: 999px; padding: 5px 10px;
          font-size: .72rem; font-weight: 650;
        }
        .ng-chip-accent { color: #7ce8d7; border-color: rgba(98,228,209,.25); }
        .ng-workflow {
          display: grid; grid-template-columns: repeat(6, minmax(0,1fr));
          gap: 7px; margin: 0 0 1.6rem 0;
        }
        .ng-step {
          min-width: 0; padding: 10px 11px;
          background: rgba(17,29,48,.65); border: 1px solid var(--ng-border);
          border-radius: 11px; color: #879bb5;
          font-size: .72rem; font-weight: 650; line-height: 1.35;
        }
        .ng-step b {
          display: block; color: #6f839e; font-size: .64rem;
          letter-spacing: .08em; margin-bottom: 5px;
        }
        .ng-step.active {
          background: linear-gradient(135deg, rgba(98,228,209,.14), rgba(122,167,255,.08));
          border-color: rgba(98,228,209,.40); color: #e8fffb;
          box-shadow: inset 0 0 0 1px rgba(98,228,209,.04);
        }
        .ng-step.active b { color: #74e5d5; }
        .ng-section-label {
          color: #75dfd0; text-transform: uppercase; letter-spacing: .13em;
          font-size: .69rem; font-weight: 800; margin: .15rem 0 .4rem;
        }
        @keyframes ng-rise {
          from { opacity: 0; transform: translateY(7px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 900px) {
          .block-container { padding: 1.2rem 1rem 2.2rem; }
          .ng-hero { padding: 21px 20px; border-radius: 17px; }
          .ng-workflow { grid-template-columns: repeat(3, minmax(0,1fr)); }
        }
        @media (max-width: 520px) {
          .ng-workflow { grid-template-columns: repeat(2, minmax(0,1fr)); }
          [data-testid="stMetric"] { padding: 11px 12px; }
        }
        @media (prefers-reduced-motion: reduce) {
          *, *:before, *:after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div class="ng-sidebar-brand">
          <div class="ng-sidebar-mark">N</div>
          <div><strong>PV NarrativeGuard</strong><small>Safety intelligence workspace</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero(
    title: str,
    description: str = "",
    *,
    eyebrow: str = "PV NARRATIVEGUARD  /  SAFETY INTELLIGENCE",
    chips=None,
) -> None:
    """Render the branded, high-contrast page hero."""
    safe_title = escape(str(title))
    safe_description = escape(str(description))
    chip_markup = "".join(
        f'<span class="ng-chip {"ng-chip-accent" if i == 0 else ""}">{escape(str(chip))}</span>'
        for i, chip in enumerate(chips or [])
    )
    st.markdown(
        f"""
        <section class="ng-hero">
          <div class="ng-eyebrow"><span class="ng-pulse"></span>{escape(eyebrow)}</div>
          <h1>{safe_title}</h1>
          <p>{safe_description}</p>
          <div class="ng-hero-meta">{chip_markup}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_strip(active_step: int) -> None:
    """Show the six-page workflow as a compact visual orientation aid."""
    steps = [
        ("01", "Dashboard"),
        ("02", "ICSR Work Queue"),
        ("03", "Duplicate Review"),
        ("04", "Narrative Generation"),
        ("05", "Narrative QC"),
        ("06", "Audit Trail"),
    ]
    markup = "".join(
        f'<div class="ng-step {"active" if index == active_step else ""}>'
        f'<b>{number}</b>{escape(label)}</div>'
        for index, (number, label) in enumerate(steps, start=1)
    )
    st.markdown(
        f'<div class="ng-workflow" aria-label="Application workflow">{markup}</div>',
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    description: str = "",
):
    """Compatibility wrapper for a branded page header."""
    render_page_hero(title, description)


def render_status_metric_row(metrics):
    """Render a row of operational metrics."""
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)


def render_empty_state(message: str, *, level: str = "info"):
    """Render a standardized empty-state message."""
    if level == "warning":
        st.warning(message)
    elif level == "success":
        st.success(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def render_evidence_table(evidence):
    """Render RAB evidence when structured evidence is available."""
    if not evidence:
        st.info("No evidence available.")
        return
    st.dataframe(evidence, width="stretch", hide_index=True)
