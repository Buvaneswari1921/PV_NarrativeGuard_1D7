"""PV NarrativeGuard Dashboard.

Phase 1 UI module.

Purpose
-------
Render the operational dashboard using the shared application data model.

Design boundary
---------------
This module is presentation-only. It does not modify backend pipeline logic,
duplicate detection, narrative generation, or narrative QC logic.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.ui.data_loader import load_application_data
from src.ui.page_context import create_page_context
from src.ui.rab_ui import render_rab_panel
from src.ui.components import render_page_hero, render_workflow_strip


# ============================================================
# DASHBOARD METRICS
# ============================================================

def _calculate_dashboard_metrics(
    final_df: pd.DataFrame,
    narrative_qc_df: pd.DataFrame,
) -> Dict[str, int]:
    """Calculate dashboard-level operational metrics."""

    if final_df.empty:
        return {
            "total_cases": 0,
            "valid_cases": 0,
            "invalid_cases": 0,
            "duplicate_review_cases": 0,
            "eligible_cases": 0,
            "generated_cases": 0,
        }

    validation_status = (
        final_df.get(
            "validation_status",
            pd.Series(index=final_df.index, dtype="object"),
        )
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    eligibility = (
        final_df.get(
            "narrative_eligibility",
            pd.Series(index=final_df.index, dtype="object"),
        )
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    total_cases = len(final_df)

    valid_cases = int(
        (validation_status == "VALID").sum()
    )

    invalid_cases = int(
        (validation_status == "INVALID").sum()
    )

    duplicate_review_cases = int(
        (eligibility == "PENDING_HUMAN_REVIEW").sum()
    )

    eligible_cases = int(
        (eligibility == "ELIGIBLE").sum()
    )

    # Count only saved QC records that contain generated
    # narrative text, matching the original app behavior.
    generated_cases = 0

    if (
        not narrative_qc_df.empty
        and "generated_narrative" in narrative_qc_df.columns
        and "case_id" in narrative_qc_df.columns
    ):
        generated_cases = int(
            narrative_qc_df[
                narrative_qc_df["generated_narrative"]
                .fillna("")
                .astype(str)
                .str.strip()
                != ""
            ]["case_id"].nunique()
        )

    return {
        "total_cases": total_cases,
        "valid_cases": valid_cases,
        "invalid_cases": invalid_cases,
        "duplicate_review_cases": duplicate_review_cases,
        "eligible_cases": eligible_cases,
        "generated_cases": generated_cases,
    }


# ============================================================
# DASHBOARD
# ============================================================

def render_dashboard(
    data: Dict[str, Any] | None = None,
) -> None:
    """Render the PV NarrativeGuard operational dashboard."""

    if data is None:
        data = load_application_data()

    final_df = data.get(
        "final_df",
        pd.DataFrame(),
    )

    narrative_qc_df = data.get(
        "narrative_qc_df",
        pd.DataFrame(),
    )

    render_page_hero(
        "Safety intelligence, with human oversight",
        "A focused command center for ICSR validation, duplicate investigation, "
        "narrative quality control, and traceable review decisions.",
        eyebrow="PV NARRATIVEGUARD  /  COMMAND CENTER",
        chips=["Human-in-the-loop", "Evidence-led review", "Prototype workspace"],
    )
    render_workflow_strip(1)

    # --------------------------------------------------------
    # Backend availability
    # --------------------------------------------------------

    if final_df.empty:
        st.warning("No pipeline results found.")

        st.info(
            "Run the backend pipeline first:"
        )

        st.code(
            "python src\\run_pipeline.py"
        )

        # RAB remains available even when pipeline data
        # is unavailable.
        ctx = create_page_context(
            page_name="Dashboard"
        )

        render_rab_panel(
            ctx=ctx,
            page_name="Dashboard",
        )

        return

    # --------------------------------------------------------
    # Pipeline overview
    # --------------------------------------------------------

    st.header("📊 Pipeline Overview")

    metrics = _calculate_dashboard_metrics(
        final_df=final_df,
        narrative_qc_df=narrative_qc_df,
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
        "Total ICSRs",
        metrics["total_cases"],
    )

    col2.metric(
        "Valid",
        metrics["valid_cases"],
    )

    col3.metric(
        "Invalid",
        metrics["invalid_cases"],
    )

    col4.metric(
        "Duplicate Review",
        metrics["duplicate_review_cases"],
    )

    col5.metric(
        "Narrative Eligible",
        metrics["eligible_cases"],
    )

    col6.metric(
        "Narratives Generated",
        metrics["generated_cases"],
    )

    st.divider()

    # --------------------------------------------------------
    # Operational status
    # --------------------------------------------------------

    st.header("📌 Operational Status")

    status_col1, status_col2, status_col3 = st.columns(3)

    qc_status = (
        final_df.get(
            "qc_status",
            pd.Series(
                ["NOT_REVIEWED"] * len(final_df),
                index=final_df.index,
            ),
        )
        .fillna("NOT_REVIEWED")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    qc_review_count = int(
        (qc_status == "REVIEW").sum()
    )

    confirmed_duplicates = int(
        (
            final_df.get(
                "human_duplicate_decision",
                pd.Series(
                    [""] * len(final_df),
                    index=final_df.index,
                ),
            )
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
            == "DUPLICATE"
        ).sum()
    )

    human_review_count = metrics[
        "duplicate_review_cases"
    ]

    status_col1.metric(
        "Awaiting Human Review",
        human_review_count,
    )

    status_col2.metric(
        "Confirmed Duplicates",
        confirmed_duplicates,
    )

    status_col3.metric(
        "Narrative QC Review",
        qc_review_count,
    )

    # --------------------------------------------------------
    # RAB Assistant
    # --------------------------------------------------------

    st.divider()

    ctx = create_page_context(
        page_name="Dashboard",
        metadata={
            "total_cases": metrics["total_cases"],
            "valid_cases": metrics["valid_cases"],
            "invalid_cases": metrics["invalid_cases"],
            "duplicate_review_cases": metrics[
                "duplicate_review_cases"
            ],
            "eligible_cases": metrics[
                "eligible_cases"
            ],
        },
    )

    render_rab_panel(
        ctx=ctx,
        page_name="Dashboard",
        expanded=False,
    )