"""PV NarrativeGuard - Duplicate Review UI.

Purpose
-------
Provide the dedicated human duplicate-review workspace.

Design boundary
---------------
This module is presentation/orchestration only.

- Application data comes from ``load_application_data``.
- Human duplicate decisions are persisted to the existing
  ``duplicate_human_review.csv`` workflow.
- Human NEW/DUPLICATE classification is never inferred by this UI.
- Narrative eligibility is subsequently derived by ``data_loader``.
- Page context is created separately and passed to the
  PV/MedDRA Assistant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.ui.data_loader import (
    HUMAN_REVIEW_FILE,
    load_application_data,
)
from src.ui.page_context import create_page_context
from src.ui.rab_ui import render_rab_panel
from src.ui.components import render_page_hero, render_workflow_strip


# ============================================================
# HELPERS
# ============================================================

def _get_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """Return the primary pipeline dataframe from application data."""

    if not data:
        return pd.DataFrame()

    for key in (
        "final_results",
        "final_df",
        "pipeline_results",
        "dataframe",
        "df",
    ):
        value = data.get(key)

        if isinstance(value, pd.DataFrame):
            return value

    return pd.DataFrame()


def _get_review_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Return cases currently awaiting human duplicate review."""

    if df is None or df.empty:
        return df

    if "narrative_eligibility" not in df.columns:
        return df.iloc[0:0].copy()

    review_df = df[
        df["narrative_eligibility"]
        .astype(str)
        .str.upper()
        .str.strip()
        == "PENDING_HUMAN_REVIEW"
    ].copy()

    return review_df


def _get_selected_record(
    review_df: pd.DataFrame,
    case_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return the selected review case as a dictionary."""

    if (
        review_df is None
        or review_df.empty
        or not case_id
        or case_id == "Select a case..."
    ):
        return None

    if "case_id" not in review_df.columns:
        return None

    matches = review_df[
        review_df["case_id"]
        .astype(str)
        .str.strip()
        == str(case_id).strip()
    ]

    if matches.empty:
        return None

    return matches.iloc[-1].to_dict()


def _save_human_decision(
    case_id: str,
    decision: str,
) -> bool:
    """Persist a human PV duplicate decision.

    Existing decisions are retained for auditability. If the same
    case is reviewed again, the latest valid decision becomes the
    active decision because ``load_human_review_decisions`` keeps
    the last record for each case.
    """

    case_id = str(case_id).strip()
    decision = str(decision).upper().strip()

    if not case_id:
        st.error("A valid case ID is required.")
        return False

    if decision not in {"NEW", "DUPLICATE"}:
        st.error("Please select either NEW or DUPLICATE.")
        return False

    try:
        HUMAN_REVIEW_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Load existing review history
        # ----------------------------------------------------

        if HUMAN_REVIEW_FILE.exists():

            try:
                existing_df = pd.read_csv(
                    HUMAN_REVIEW_FILE
                )
            except (
                pd.errors.EmptyDataError,
                OSError,
            ):
                existing_df = pd.DataFrame(
                    columns=[
                        "incoming_case_id",
                        "decision",
                    ]
                )

        else:
            existing_df = pd.DataFrame(
                columns=[
                    "incoming_case_id",
                    "decision",
                ]
            )

        # ----------------------------------------------------
        # Normalize existing history
        # ----------------------------------------------------

        required_columns = [
            "incoming_case_id",
            "decision",
        ]

        if not set(required_columns).issubset(
            existing_df.columns
        ):
            existing_df = pd.DataFrame(
                columns=required_columns
            )
        else:
            existing_df = existing_df[
                required_columns
            ].copy()

        existing_df["incoming_case_id"] = (
            existing_df["incoming_case_id"]
            .astype(str)
            .str.strip()
        )

        existing_df["decision"] = (
            existing_df["decision"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        existing_df = existing_df[
            existing_df["decision"].isin(
                [
                    "NEW",
                    "DUPLICATE",
                ]
            )
        ]

        # ----------------------------------------------------
        # Append latest human decision
        # ----------------------------------------------------

        new_record = pd.DataFrame(
            [
                {
                    "incoming_case_id": case_id,
                    "decision": decision,
                }
            ]
        )

        updated_df = pd.concat(
            [
                existing_df,
                new_record,
            ],
            ignore_index=True,
        )

        # ----------------------------------------------------
        # Keep latest decision for active state
        #
        # The original rows are retained until this point,
        # allowing the latest row to represent the active
        # persisted decision.
        # ----------------------------------------------------

        updated_df = updated_df.drop_duplicates(
            subset=["incoming_case_id"],
            keep="last",
        )

        # ----------------------------------------------------
        # Persist
        # ----------------------------------------------------

        updated_df.to_csv(
            HUMAN_REVIEW_FILE,
            index=False,
        )

        return True

    except (
        OSError,
        PermissionError,
        pd.errors.ParserError,
    ) as exc:

        st.error(
            "Unable to persist the human duplicate decision: "
            f"{exc}"
        )

        return False


def _render_case_details(
    selected_record: Dict[str, Any],
) -> None:
    """Render useful duplicate-review case information."""

    st.markdown("### Case Assessment")

    case_id = str(
        selected_record.get(
            "case_id",
            "—",
        )
    )

    validation_status = str(
        selected_record.get(
            "validation_status",
            "UNKNOWN",
        )
    )

    duplicate_status = str(
        selected_record.get(
            "duplicate_status",
            "UNKNOWN",
        )
    )

    eligibility = str(
        selected_record.get(
            "narrative_eligibility",
            "UNKNOWN",
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Case ID",
            case_id,
        )

    with col2:
        st.metric(
            "Validation",
            validation_status,
        )

    with col3:
        st.metric(
            "Duplicate status",
            duplicate_status,
        )

    with col4:
        st.metric(
            "Narrative eligibility",
            eligibility,
        )

    # --------------------------------------------------------
    # Duplicate evidence
    # --------------------------------------------------------

    evidence_fields = [
        (
            "duplicate_case_id",
            "Potential duplicate case",
        ),
        (
            "duplicate_candidate_case_id",
            "Duplicate candidate",
        ),
        (
            "duplicate_similarity",
            "Similarity score",
        ),
        (
            "similarity_score",
            "Similarity score",
        ),
        (
            "duplicate_reason",
            "Duplicate assessment",
        ),
    ]

    evidence_rows = []

    for field_name, display_name in evidence_fields:

        if field_name not in selected_record:
            continue

        value = selected_record.get(
            field_name
        )

        if value is None:
            continue

        value_text = str(value).strip()

        if not value_text:
            continue

        if value_text.lower() in {
            "nan",
            "none",
        }:
            continue

        evidence_rows.append(
            {
                "Field": display_name,
                "Value": value_text,
            }
        )

    if evidence_rows:

        st.markdown(
            "#### Available Duplicate Evidence"
        )

        st.dataframe(
            pd.DataFrame(evidence_rows),
            width="stretch",
            hide_index=True,
        )

    st.warning(
        "This case requires a human PV decision. "
        "Select NEW if the case is determined to be a new "
        "report, or DUPLICATE if it is confirmed as a duplicate."
    )


# ============================================================
# MAIN UI
# ============================================================

def render_duplicate_review(
    data: Dict[str, Any],
) -> None:
    """Render the Duplicate Review page."""

    df = _get_dataframe(data)

    # ========================================================
    # PAGE HEADER
    # ========================================================

    render_page_hero(
        "Duplicate investigation",
        "Compare candidate cases with visible evidence. Similarity helps prioritize review; it does not make the final duplicate decision.",
        eyebrow="WORKSPACE  /  DUPLICATE INVESTIGATION",
        chips=["Human decision required", "Evidence comparison", "No automatic merge"],
    )
    render_workflow_strip(3)

    # ========================================================
    # DATA AVAILABILITY
    # ========================================================

    if df.empty:

        st.info(
            "No pipeline results are available."
        )

        ctx = create_page_context(
            page_name="Duplicate Review",
            metadata={
                "cases_awaiting_review": 0,
            },
        )

        render_rab_panel(
            ctx=ctx,
            page_name="Duplicate Review",
            expanded=False,
        )

        return

    # ========================================================
    # REVIEW CASES
    # ========================================================

    review_df = _get_review_cases(df)

    st.metric(
        "Cases awaiting human review",
        len(review_df),
    )

    selected_record = None
    selected_case_id = None

    # ========================================================
    # CASE QUEUE
    # ========================================================

    if review_df.empty:

        st.success(
            "No cases are currently awaiting duplicate review."
        )

    else:

        st.markdown(
            "### Cases Awaiting Human Review"
        )

        # Keep the queue operationally focused.
        queue_columns = [
            column
            for column in [
                "case_id",
                "validation_status",
                "duplicate_status",
                "narrative_eligibility",
            ]
            if column in review_df.columns
        ]

        if queue_columns:
            queue_df = review_df[
                queue_columns
            ].copy()
        else:
            queue_df = review_df.copy()

        st.dataframe(
            queue_df,
            width="stretch",
            hide_index=True,
        )

        # ====================================================
        # CASE SELECTION
        # ====================================================

        if "case_id" in review_df.columns:

            case_ids = (
                review_df["case_id"]
                .astype(str)
                .str.strip()
                .drop_duplicates()
                .tolist()
            )

            selected_case_id = st.selectbox(
                "Select case",
                ["Select a case..."] + case_ids,
                key="duplicate_review_selected_case",
            )

            selected_record = _get_selected_record(
                review_df,
                selected_case_id,
            )

        # ====================================================
        # SELECTED CASE
        # ====================================================

        if selected_record is not None:

            st.divider()

            _render_case_details(
                selected_record
            )

            st.divider()

            # =================================================
            # HUMAN DECISION
            # =================================================

            st.markdown(
                "### Human PV Duplicate Decision"
            )

            st.write(
                "The decision below is the formal human "
                "review gate. It will be persisted and used "
                "by the downstream narrative eligibility logic."
            )

            with st.form(
                "human_duplicate_decision_form",
                clear_on_submit=False,
            ):

                decision = st.radio(
                    "Select final duplicate classification",
                    [
                        "NEW",
                        "DUPLICATE",
                    ],
                    index=None,
                    horizontal=True,
                    key=(
                        "human_duplicate_decision_"
                        + str(
                            selected_record.get(
                                "case_id",
                                "",
                            )
                        )
                    ),
                )

                st.caption(
                    "NEW → case becomes eligible for narrative "
                    "generation. "
                    "DUPLICATE → downstream narrative generation "
                    "is blocked."
                )

                submitted = st.form_submit_button(
                    "💾 Save Human Decision",
                    type="primary",
                    width="stretch",
                )

            if submitted:

                if decision not in {
                    "NEW",
                    "DUPLICATE",
                }:

                    st.error(
                        "Please select NEW or DUPLICATE "
                        "before saving."
                    )

                else:

                    case_id = str(
                        selected_record.get(
                            "case_id",
                            "",
                        )
                    ).strip()

                    if _save_human_decision(
                        case_id=case_id,
                        decision=decision,
                    ):

                        if decision == "NEW":

                            st.success(
                                f"{case_id} has been classified "
                                "as NEW. The case is now eligible "
                                "for narrative generation."
                            )

                        else:

                            st.success(
                                f"{case_id} has been classified "
                                "as DUPLICATE. Downstream "
                                "narrative generation is blocked."
                            )

                        st.info(
                            "Refreshing application state..."
                        )

                        st.rerun()

    # ========================================================
    # PAGE CONTEXT
    # ========================================================

    selected_case_value = (
        str(selected_case_id)
        if selected_case_id
        and selected_case_id != "Select a case..."
        else None
    )

    ctx = create_page_context(
        page_name="Duplicate Review",
        selected_case_id=selected_case_value,
        selected_record=selected_record,
        metadata={
            "cases_awaiting_review": len(review_df),
            "cases_in_view": len(df),
            "human_review_file": str(
                HUMAN_REVIEW_FILE
            ),
        },
    )

    render_rab_panel(
        ctx=ctx,
        page_name="Duplicate Review",
        expanded=False,
    )


# ============================================================
# OPTIONAL DIRECT EXECUTION
# ============================================================

def main():
    """Allow direct execution of the module."""

    st.set_page_config(
        page_title="PV NarrativeGuard | Duplicate Review",
        page_icon="🔎",
        layout="wide",
    )

    data = load_application_data()

    render_duplicate_review(data)


if __name__ == "__main__":
    main()