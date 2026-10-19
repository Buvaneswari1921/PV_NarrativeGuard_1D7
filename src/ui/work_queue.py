"""PV NarrativeGuard - ICSR Work Queue.

Phase 1 UI module.

This module extracts the presentation/orchestration layer of the
existing PV Work Queue. Backend processing and persistence remain
outside this module.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st

from src.ui.page_context import create_page_context
from src.ui.rab_ui import render_rab_panel
from src.ui.components import render_page_hero, render_workflow_strip


# ============================================================
# WORK QUEUE CONSTRUCTION
# ============================================================

def build_work_queues(
    final_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build action-required and ready-for-processing queues.

    Priority follows the existing PV workflow:

    1. Validation review
    2. Duplicate review
    3. Narrative QC review
    4. Ready for narrative processing

    No backend decision is made here.
    """

    action_rows = []
    ready_rows = []

    if final_df.empty or "case_id" not in final_df.columns:
        return (
            pd.DataFrame(
                columns=[
                    "case_id",
                    "priority",
                    "action",
                    "status",
                    "reason",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "case_id",
                    "action",
                    "status",
                    "reason",
                ]
            ),
        )

    for _, row in final_df.iterrows():

        case_id = str(row["case_id"])

        validation_status = str(
            row.get("validation_status", "")
        ).strip()

        duplicate_status = str(
            row.get("duplicate_status", "")
        ).strip()

        eligibility = str(
            row.get("narrative_eligibility", "")
        ).strip()

        narrative_status = str(
            row.get("narrative_status", "")
        ).strip()

        qc_status = str(
            row.get("qc_status", "NOT_REVIEWED")
        ).strip()

        # ----------------------------------------------------
        # Priority 1 — validation
        # ----------------------------------------------------

        if validation_status == "INVALID":

            action_rows.append(
                {
                    "case_id": case_id,
                    "priority": 1,
                    "action": "Validation review",
                    "status": "INVALID",
                    "reason": "ICSR failed validation.",
                }
            )

            continue

        # ----------------------------------------------------
        # Priority 2 — duplicate review
        # ----------------------------------------------------

        if (
            validation_status == "VALID"
            and eligibility == "PENDING_HUMAN_REVIEW"
        ):

            action_rows.append(
                {
                    "case_id": case_id,
                    "priority": 2,
                    "action": "Duplicate review",
                    "status": duplicate_status,
                    "reason": (
                        "Human PV duplicate decision required."
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Priority 3 — narrative QC
        # ----------------------------------------------------

        if (
            validation_status == "VALID"
            and qc_status == "REVIEW"
        ):

            action_rows.append(
                {
                    "case_id": case_id,
                    "priority": 3,
                    "action": "Narrative QC review",
                    "status": "QC REVIEW",
                    "reason": (
                        "Generated narrative requires QC review."
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Ready for processing
        # ----------------------------------------------------

        if (
            validation_status == "VALID"
            and eligibility == "ELIGIBLE"
            and not narrative_status.startswith("GENERATED")
            and not narrative_status.startswith("BLOCKED")
        ):

            ready_rows.append(
                {
                    "case_id": case_id,
                    "action": "Generate Narrative",
                    "status": (
                        narrative_status
                        or "NOT_GENERATED - ON DEMAND"
                    ),
                    "reason": (
                        "Case is eligible for "
                        "on-demand narrative generation."
                    ),
                }
            )

    action_df = pd.DataFrame(action_rows)
    ready_df = pd.DataFrame(ready_rows)

    if not action_df.empty:
        action_df = action_df.sort_values(
            ["priority", "case_id"]
        ).reset_index(drop=True)

    if not ready_df.empty:
        ready_df = ready_df.sort_values(
            "case_id"
        ).reset_index(drop=True)

    return action_df, ready_df


# ============================================================
# FILTERING
# ============================================================

def filter_cases(
    df: pd.DataFrame,
    search: str,
    validation: str,
    duplicate_status: str,
    narrative_eligibility: str,
) -> pd.DataFrame:
    """Apply Work Queue filters without changing source data."""

    if df.empty:
        return df.copy()

    result = df.copy()

    if search and "case_id" in result.columns:
        result = result[
            result["case_id"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False,
            )
        ]

    if (
        validation != "All"
        and "validation_status" in result.columns
    ):
        result = result[
            result["validation_status"]
            .astype(str)
            .str.upper()
            == validation.upper()
        ]

    if (
        duplicate_status != "All"
        and "duplicate_status" in result.columns
    ):
        result = result[
            result["duplicate_status"]
            .astype(str)
            .str.upper()
            == duplicate_status.upper()
        ]

    if (
        narrative_eligibility != "All"
        and "narrative_eligibility" in result.columns
    ):
        result = result[
            result["narrative_eligibility"]
            .astype(str)
            .str.upper()
            == narrative_eligibility.upper()
        ]

    return result


# ============================================================
# CASE DETAILS
# ============================================================

def render_case_details(
    case: pd.Series,
) -> None:
    """Render a selected case in a controlled read-only view."""

    case_id = str(
        case.get("case_id", "Unknown")
    )

    st.subheader(
        f"Case Details — {case_id}"
    )

    details = case.to_dict()

    preferred_fields = [
        "case_id",
        "validation_status",
        "validation_reason",
        "duplicate_status",
        "human_duplicate_decision",
        "narrative_eligibility",
        "narrative_status",
        "qc_status",
    ]

    shown = {
        field: details.get(field)
        for field in preferred_fields
        if field in details
    }

    if shown:
        detail_df = pd.DataFrame(
            list(shown.items()),
            columns=["Field", "Value"],
        )

        st.dataframe(
            detail_df,
            width="stretch",
            hide_index=True,
        )

    remaining = {
        key: value
        for key, value in details.items()
        if key not in shown
    }

    if remaining:

        with st.expander(
            "Additional case data",
            expanded=False,
        ):
            st.json(remaining)


# ============================================================
# WORK QUEUE PAGE
# ============================================================

def render_work_queue(
    data: Dict[str, Any],
) -> None:
    """Render the Phase 1 ICSR Work Queue."""

    final_df = data.get(
        "final_df",
        pd.DataFrame(),
    )

    if not isinstance(final_df, pd.DataFrame):
        final_df = pd.DataFrame()

    render_page_hero(
        "ICSR Work Queue",
        "One prioritized workspace for cases that need attention and cases ready for the next controlled step.",
        eyebrow="WORKSPACE  /  CASE TRIAGE",
        chips=["Priority queue", "Eligibility-aware", "Reviewer controlled"],
    )
    render_workflow_strip(2)

    if final_df.empty:

        st.warning(
            "No pipeline results are currently available."
        )

        st.code(
            "python src\\run_pipeline.py"
        )

        ctx = create_page_context(
            page_name="ICSR Work Queue"
        )

        render_rab_panel(
            ctx=ctx,
            page_name="ICSR Work Queue",
        )

        return

    # --------------------------------------------------------
    # Build operational queues
    # --------------------------------------------------------

    action_df, ready_df = build_work_queues(
        final_df
    )

    # --------------------------------------------------------
    # Queue metrics
    # --------------------------------------------------------

    st.subheader("📌 Queue Summary")

    m1, m2, m3, m4 = st.columns(4)

    validation_count = int(
        (
            action_df["action"]
            == "Validation review"
        ).sum()
    ) if not action_df.empty else 0

    duplicate_count = int(
        (
            action_df["action"]
            == "Duplicate review"
        ).sum()
    ) if not action_df.empty else 0

    qc_count = int(
        (
            action_df["action"]
            == "Narrative QC review"
        ).sum()
    ) if not action_df.empty else 0

    m1.metric(
        "Validation Review",
        validation_count,
    )

    m2.metric(
        "Duplicate Review",
        duplicate_count,
    )

    m3.metric(
        "Narrative QC Review",
        qc_count,
    )

    m4.metric(
        "Ready for Processing",
        len(ready_df),
    )

    # --------------------------------------------------------
    # Action required
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔴 Action Required")

    st.caption(
        "These cases require a PV action before the workflow "
        "can proceed."
    )

    if action_df.empty:

        st.success(
            "No cases currently require immediate PV action."
        )

    else:

        display_action = action_df[
            [
                "case_id",
                "action",
                "status",
                "reason",
            ]
        ].copy()

        st.dataframe(
            display_action,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # Ready queue
    # --------------------------------------------------------

    st.divider()

    st.subheader("🟢 Ready for Processing")

    st.caption(
        "Eligible cases are available for explicit "
        "on-demand narrative generation."
    )

    if ready_df.empty:

        st.info(
            "No cases are currently ready for processing."
        )

    else:

        st.dataframe(
            ready_df,
            width="stretch",
            hide_index=True,
        )

    # --------------------------------------------------------
    # Case Explorer
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔎 Case Explorer")

    st.caption(
        "Search and filter the case-level pipeline results."
    )

    filter_col1, filter_col2 = st.columns(2)

    search = filter_col1.text_input(
        "Search Case ID",
        placeholder="e.g. INCOMING002",
    )

    validation_options = ["All"]

    if "validation_status" in final_df.columns:
        validation_options += sorted(
            final_df["validation_status"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

    validation = filter_col1.selectbox(
        "Validation",
        validation_options,
    )

    duplicate_options = ["All"]

    if "duplicate_status" in final_df.columns:
        duplicate_options += sorted(
            final_df["duplicate_status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    duplicate_status = filter_col2.selectbox(
        "Duplicate Status",
        duplicate_options,
    )

    eligibility_options = ["All"]

    if "narrative_eligibility" in final_df.columns:
        eligibility_options += sorted(
            final_df["narrative_eligibility"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    narrative_eligibility = filter_col2.selectbox(
        "Narrative Eligibility",
        eligibility_options,
    )

    filtered = filter_cases(
        final_df,
        search=search,
        validation=validation,
        duplicate_status=duplicate_status,
        narrative_eligibility=narrative_eligibility,
    )

    st.metric(
        "Cases in view",
        len(filtered),
    )

    if filtered.empty:

        st.info(
            "No cases match the selected filters."
        )

    else:

        sort_options = [
            "case_id"
        ]

        if "validation_status" in filtered.columns:
            sort_options.append(
                "validation_status"
            )

        if "narrative_eligibility" in filtered.columns:
            sort_options.append(
                "narrative_eligibility"
            )

        sort_by = st.selectbox(
            "Sort cases by",
            sort_options,
        )

        sorted_df = filtered.sort_values(
            sort_by,
            kind="stable",
        )

        st.dataframe(
            sorted_df,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # Selected case
        # ----------------------------------------------------

        case_ids = (
            sorted_df["case_id"]
            .astype(str)
            .tolist()
        )

        selected_case = st.selectbox(
            "Select case for details",
            ["Select a case..."] + case_ids,
        )

        selected_record = None

        if selected_case != "Select a case...":

            selected_rows = sorted_df[
                sorted_df["case_id"]
                .astype(str)
                == selected_case
            ]

            if not selected_rows.empty:

                selected_record = (
                    selected_rows.iloc[0]
                )

                render_case_details(
                    selected_record
                )

    # --------------------------------------------------------
    # RAB
    # --------------------------------------------------------

    st.divider()

    selected_case_id = (
        selected_record.get("case_id")
        if selected_record is not None
        else None
    )

    selected_record_dict = (
        selected_record.to_dict()
        if selected_record is not None
        else None
    )

    ctx = create_page_context(
        page_name="ICSR Work Queue",
        selected_case_id=(
            str(selected_case_id)
            if selected_case_id is not None
            else None
        ),
        selected_record=selected_record_dict,
        metadata={
            "action_required": len(action_df),
            "ready_for_processing": len(
                ready_df
            ),
            "cases_in_view": len(filtered),
        },
    )

    render_rab_panel(
        ctx=ctx,
        page_name="ICSR Work Queue",
        expanded=False,
    )