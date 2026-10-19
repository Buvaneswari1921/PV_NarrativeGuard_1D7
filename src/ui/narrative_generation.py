"""PV NarrativeGuard - Narrative Generation UI.

Controlled, on-demand narrative generation for cases that have
passed validation and duplicate review.

Design principles:
- Only ELIGIBLE cases are exposed for generation.
- Existing persisted narratives are displayed read-only.
- Regeneration requires explicit user confirmation.
- Generation is followed immediately by field-level QC.
- Human pharmacovigilance review remains required before final use.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.ui.page_context import create_page_context
from src.ui.rab_ui import render_rab_panel
from src.ui.components import render_page_hero, render_workflow_strip


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
NARRATIVE_QC_FILE = OUTPUT_DIR / "narrative_qc_results.csv"


# ============================================================
# QC FIELD MAP
# ============================================================

QC_FIELD_MAP = {
    "patient_age_qc": (
        "Patient age",
        "patient_age",
    ),
    "sex_qc": (
        "Patient sex",
        "patient_sex",
    ),
    "country_qc": (
        "Country",
        "country",
    ),
    "suspect_drug_qc": (
        "Suspect drug",
        "suspect_drug",
    ),
    "indication_qc": (
        "Indication",
        "indication",
    ),
    "adverse_event_qc": (
        "Adverse event",
        "adverse_event",
    ),
    "event_date_qc": (
        "Event date",
        "event_date",
    ),
    "seriousness_qc": (
        "Seriousness",
        "seriousness",
    ),
    "outcome_qc": (
        "Outcome",
        "outcome",
    ),
    "reporter_type_qc": (
        "Reporter type",
        "reporter_type",
    ),
}


# ============================================================
# HELPERS
# ============================================================

def has_real_narrative(value: Any) -> bool:
    """Return True only when value contains actual narrative text."""

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return False

    return text.lower() not in {
        "nan",
        "none",
        "null",
    }


def safe_int(value: Any) -> int:
    """Safely convert a persisted QC metric to integer."""

    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


# ============================================================
# PERSISTENCE
# ============================================================

def persist_narrative_qc(result_row: Dict[str, Any]) -> None:
    """Persist one narrative/QC result, replacing the same case."""

    case_id = str(result_row["case_id"]).strip()

    result_df = pd.DataFrame([result_row])

    if NARRATIVE_QC_FILE.exists():
        try:
            saved_qc = pd.read_csv(NARRATIVE_QC_FILE)
        except (OSError, pd.errors.EmptyDataError):
            saved_qc = pd.DataFrame()
    else:
        saved_qc = pd.DataFrame()

    if not saved_qc.empty and "case_id" in saved_qc.columns:
        saved_qc = saved_qc[
            saved_qc["case_id"].astype(str).str.strip() != case_id
        ]

    combined_qc = pd.concat(
        [saved_qc, result_df],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_qc.to_csv(
        NARRATIVE_QC_FILE,
        index=False,
    )


# ============================================================
# EXISTING NARRATIVE
# ============================================================

def get_existing_narrative(
    narrative_qc_df: pd.DataFrame,
    case_id: str,
) -> Optional[pd.Series]:
    """Return the latest persisted narrative/QC row for a case."""

    if (
        narrative_qc_df.empty
        or "case_id" not in narrative_qc_df.columns
    ):
        return None

    matches = narrative_qc_df[
        narrative_qc_df["case_id"].astype(str).str.strip()
        == str(case_id).strip()
    ]

    if matches.empty:
        return None

    return matches.iloc[-1]


# ============================================================
# DISPLAY EXISTING NARRATIVE
# ============================================================

def render_existing_narrative(
    existing_narrative: pd.Series,
    case: pd.Series,
) -> bool:
    """Display persisted narrative and its QC evidence."""

    narrative_text = existing_narrative.get(
        "generated_narrative",
        "",
    )

    if not has_real_narrative(narrative_text):
        return False

    st.success("Narrative already generated for this case.")

    st.markdown("### AI-Generated Narrative")

    st.text_area(
        "Narrative",
        value=str(narrative_text).strip(),
        height=180,
        disabled=True,
        label_visibility="collapsed",
    )

    qc_status = str(
        existing_narrative.get(
            "overall_status",
            "UNKNOWN",
        )
    ).upper().strip()

    st.markdown("### Narrative QC")

    if qc_status == "PASS":
        st.success("QC Status: PASS")
    elif qc_status == "REVIEW":
        st.warning("QC Status: REVIEW")
    else:
        st.info(f"QC Status: {qc_status}")

    st.markdown("#### Field-Level QC")

    st.caption(
        "Each available source fact is checked against the generated "
        "narrative. PASS means the fact was covered; REVIEW means PV "
        "attention is required; SOURCE MISSING means the source ICSR "
        "did not provide a verifiable value."
    )

    qc_rows = []

    for qc_column, (display_name, source_column) in QC_FIELD_MAP.items():

        field_result = str(
            existing_narrative.get(
                qc_column,
                "REVIEW",
            )
        ).upper().strip()

        source_value = ""

        if source_column in case.index:
            raw_value = case.get(source_column, "")

            if raw_value is not None and pd.notna(raw_value):
                source_value = str(raw_value).strip()

        if not source_value:
            source_value = "Not available"

            if field_result == "PASS":
                field_result = "SOURCE MISSING"

        if field_result == "PASS":
            result_display = "🟢 PASS"
        elif field_result == "REVIEW":
            result_display = "🔴 REVIEW"
        elif field_result == "SOURCE MISSING":
            result_display = "🟡 SOURCE MISSING"
        else:
            result_display = f"🟡 {field_result}"

        qc_rows.append(
            {
                "QC check": display_name,
                "Source ICSR": source_value,
                "Result": result_display,
            }
        )

    st.dataframe(
        pd.DataFrame(qc_rows),
        width="stretch",
        hide_index=True,
    )

    review_fields = []

    for qc_column, (display_name, source_column) in QC_FIELD_MAP.items():

        field_result = str(
            existing_narrative.get(
                qc_column,
                "REVIEW",
            )
        ).upper().strip()

        if field_result != "REVIEW":
            continue

        source_value = ""

        if source_column in case.index:
            raw_value = case.get(source_column, "")

            if raw_value is not None and pd.notna(raw_value):
                source_value = str(raw_value).strip()

        review_fields.append(
            {
                "name": display_name,
                "source": source_value or "Not available",
            }
        )

    if review_fields:
        st.markdown("#### ⚠️ Fields Requiring PV Review")

        for item in review_fields:
            st.warning(
                f"**{item['name']}** — "
                f"Source ICSR: `{item['source']}`. "
                "The narrative coverage check requires review."
            )

    checked_count = safe_int(
        existing_narrative.get(
            "checked_field_count",
            0,
        )
    )

    failed_count = safe_int(
        existing_narrative.get(
            "failed_field_count",
            0,
        )
    )

    missing_count = safe_int(
        existing_narrative.get(
            "missing_source_field_count",
            0,
        )
    )

    summary_1, summary_2, summary_3 = st.columns(3)

    with summary_1:
        st.metric("Fields checked", checked_count)

    with summary_2:
        st.metric("Fields requiring review", failed_count)

    with summary_3:
        st.metric("Missing source fields", missing_count)

    qc_reason = str(
        existing_narrative.get(
            "qc_reason",
            "",
        )
    ).strip()

    if (
        qc_reason
        and qc_reason.lower() not in {
            "nan",
            "none",
            "null",
        }
    ):
        with st.expander(
            "View QC reason",
            expanded=(qc_status == "REVIEW"),
        ):
            st.write(qc_reason)

    st.warning(
        "Human pharmacovigilance review is required before final use."
    )

    return True


# ============================================================
# GENERATION + QC
# ============================================================

def generate_and_qc(
    selected_case_id: str,
    case: pd.Series,
) -> Dict[str, Any]:
    """Generate one narrative, run QC, and persist the result."""

    from src.narrative_generation import generate_narrative
    from src.narrative_qc import qc_case

    narrative = generate_narrative(selected_case_id)

    if narrative is None:
        narrative = ""

    narrative = str(narrative).strip()

    if not narrative:
        raise ValueError(
            "Narrative generation returned an empty result."
        )

    qc_result = qc_case(
        case,
        narrative,
    )

    result_row = {
        "case_id": selected_case_id,
        "generated_narrative": narrative,
        "overall_status": qc_result.get(
            "overall_status",
            "REVIEW",
        ),
        "qc_reason": qc_result.get(
            "qc_reason",
            "",
        ),
        "checked_field_count": qc_result.get(
            "checked_field_count",
            0,
        ),
        "failed_field_count": qc_result.get(
            "failed_field_count",
            0,
        ),
        "missing_source_field_count": qc_result.get(
            "missing_source_field_count",
            0,
        ),
    }

    for field in QC_FIELD_MAP:
        result_row[field] = qc_result.get(
            field,
            "REVIEW",
        )

    persist_narrative_qc(result_row)

    return result_row


# ============================================================
# MAIN UI
# ============================================================

def render_narrative_generation(data: Dict[str, Any]) -> None:
    """Render the controlled Narrative Generation workspace."""

    final_df = data.get(
        "final_df",
        pd.DataFrame(),
    )

    incoming_df = data.get(
        "incoming_df",
        pd.DataFrame(),
    )

    narrative_qc_df = data.get(
        "narrative_qc_df",
        pd.DataFrame(),
    )

    render_page_hero(
        "Narrative Studio",
        "Generate a structured narrative only for eligible cases, then inspect field-level quality findings before human PV review.",
        eyebrow="WORKSPACE  /  CONTROLLED GENERATION",
        chips=["Eligible cases only", "On-demand generation", "Human PV review"],
    )
    render_workflow_strip(4)

    # --------------------------------------------------------
    # Empty pipeline
    # --------------------------------------------------------

    if final_df.empty:
        st.info("No pipeline results are currently available.")

        ctx = create_page_context(
            page_name="Narrative Generation",
            metadata={"eligible_cases": 0},
        )

        render_rab_panel(
            ctx,
            page_name="Narrative Generation",
            expanded=False,
        )
        return

    if "case_id" not in final_df.columns:
        st.error(
            "Pipeline results do not contain a case_id column."
        )
        return

    # --------------------------------------------------------
    # Eligible cases only
    # --------------------------------------------------------

    if "narrative_eligibility" in final_df.columns:
        eligible_df = final_df[
            final_df["narrative_eligibility"]
            .astype(str)
            .str.upper()
            .str.strip()
            == "ELIGIBLE"
        ].copy()
    else:
        eligible_df = final_df.iloc[0:0].copy()

    # Deduplicate case IDs while preserving order.
    eligible_df = eligible_df.drop_duplicates(
        subset=["case_id"],
        keep="last",
    )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    generated_count = 0

    if (
        not narrative_qc_df.empty
        and "case_id" in narrative_qc_df.columns
        and "generated_narrative" in narrative_qc_df.columns
    ):
        generated_count = (
            narrative_qc_df[
                narrative_qc_df["generated_narrative"]
                .apply(has_real_narrative)
            ]["case_id"]
            .astype(str)
            .nunique()
        )

    pending_review_count = 0

    if "narrative_eligibility" in final_df.columns:
        pending_review_count = int(
            (
                final_df["narrative_eligibility"]
                .astype(str)
                .str.upper()
                .str.strip()
                == "PENDING_HUMAN_REVIEW"
            ).sum()
        )

    metric_1, metric_2, metric_3 = st.columns(3)

    with metric_1:
        st.metric(
            "Eligible cases",
            len(eligible_df),
        )

    with metric_2:
        st.metric(
            "Narratives retained",
            generated_count,
        )

    with metric_3:
        st.metric(
            "Awaiting human duplicate review",
            pending_review_count,
        )

    st.caption(
        "Narratives retained includes previously generated narratives "
        "kept for traceability and audit history. Only currently eligible "
        "cases are available for operational narrative generation."
    )

    st.divider()

    # --------------------------------------------------------
    # Case selection
    # --------------------------------------------------------

    st.subheader("Select Case")

    case_ids = (
        eligible_df["case_id"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    if not case_ids:
        st.info(
            "No cases are currently eligible for narrative generation."
        )

        ctx = create_page_context(
            page_name="Narrative Generation",
            metadata={
                "eligible_cases": 0,
                "narratives_generated": generated_count,
            },
        )

        render_rab_panel(
            ctx,
            page_name="Narrative Generation",
            expanded=False,
        )
        return

    selected_case_id = st.selectbox(
        "Case",
        ["Select a case..."] + case_ids,
        key="narrative_generation_case",
    )

    if selected_case_id == "Select a case...":
        st.info(
            "Select an eligible case to view its narrative "
            "generation workspace."
        )

        ctx = create_page_context(
            page_name="Narrative Generation",
            metadata={
                "eligible_cases": len(eligible_df),
                "narratives_generated": generated_count,
            },
        )

        render_rab_panel(
            ctx,
            page_name="Narrative Generation",
            expanded=False,
        )
        return

    # --------------------------------------------------------
    # Selected pipeline record
    # --------------------------------------------------------

    selected_rows = eligible_df[
        eligible_df["case_id"].astype(str).str.strip()
        == selected_case_id
    ]

    if selected_rows.empty:
        st.error(
            "Selected case could not be found in eligible "
            "pipeline results."
        )
        return

    selected_record = selected_rows.iloc[-1]

    # --------------------------------------------------------
    # Source ICSR
    # --------------------------------------------------------

    case = None

    if (
        not incoming_df.empty
        and "case_id" in incoming_df.columns
    ):
        incoming_rows = incoming_df[
            incoming_df["case_id"].astype(str).str.strip()
            == selected_case_id
        ]

        if not incoming_rows.empty:
            case = incoming_rows.iloc[0]

    if case is None:
        st.error(
            f"Source ICSR {selected_case_id} could not be found "
            "in incoming_icsrs.csv."
        )
        return

    # --------------------------------------------------------
    # Case status
    # --------------------------------------------------------

    st.subheader(f"Case: {selected_case_id}")

    validation_status = str(
        selected_record.get(
            "validation_status",
            "UNKNOWN",
        )
    ).upper().strip()

    duplicate_status = str(
        selected_record.get(
            "duplicate_status",
            "UNKNOWN",
        )
    ).strip()

    eligibility = str(
        selected_record.get(
            "narrative_eligibility",
            "UNKNOWN",
        )
    ).upper().strip()

    status_1, status_2, status_3 = st.columns(3)

    with status_1:
        if validation_status == "VALID":
            st.success(
                f"Validation\n\n{validation_status}"
            )
        else:
            st.warning(
                f"Validation\n\n{validation_status}"
            )

    with status_2:
        st.info(
            f"Duplicate status\n\n{duplicate_status}"
        )

    with status_3:
        if eligibility == "ELIGIBLE":
            st.success(
                f"Narrative eligibility\n\n{eligibility}"
            )
        else:
            st.warning(
                f"Narrative eligibility\n\n{eligibility}"
            )

    # Defensive eligibility gate.
    if eligibility != "ELIGIBLE":
        st.error(
            "Narrative generation is unavailable because "
            "this case is not currently eligible."
        )

        ctx = create_page_context(
            page_name="Narrative Generation",
            selected_case_id=selected_case_id,
            selected_record=case.to_dict(),
            metadata={
                "eligibility": eligibility,
            },
        )

        render_rab_panel(
            ctx,
            page_name="Narrative Generation",
        )
        return

    # --------------------------------------------------------
    # Narrative workspace
    # --------------------------------------------------------

    st.divider()

    st.subheader("🤖 Narrative Workspace")

    st.caption(
        "Generate a narrative only for this selected eligible case. "
        "No narratives are generated automatically."
    )

    existing_narrative = get_existing_narrative(
        narrative_qc_df,
        selected_case_id,
    )

    generate_requested = False

    # --------------------------------------------------------
    # Existing narrative
    # --------------------------------------------------------

    if existing_narrative is not None:

        existing_text = existing_narrative.get(
            "generated_narrative",
            "",
        )

        if has_real_narrative(existing_text):

            render_existing_narrative(
                existing_narrative,
                case,
            )

            st.markdown("### Regenerate Narrative")

            st.caption(
                "Regeneration creates a new AI-generated version "
                "for this selected case and replaces the saved "
                "narrative only after generation and QC complete."
            )

            confirm_regeneration = st.checkbox(
                "I understand that regeneration will replace "
                "the currently saved narrative after QC.",
                key=f"confirm_regeneration_{selected_case_id}",
            )

            if st.button(
                "🔄 Regenerate Narrative",
                type="secondary",
                disabled=not confirm_regeneration,
                key=f"regenerate_narrative_{selected_case_id}",
            ):
                generate_requested = True

        else:
            existing_narrative = None

    # --------------------------------------------------------
    # New narrative
    # --------------------------------------------------------

    if existing_narrative is None:

        st.info(
            "This case is eligible, but no narrative has been "
            "generated yet."
        )

        st.markdown("### Generate on Demand")

        st.write(
            f"Selected case: **{selected_case_id}**"
        )

        st.write(
            "Click the button below only when you want to "
            "generate the narrative for this case."
        )

        if st.button(
            "✨ Generate Narrative",
            type="primary",
            key=f"generate_selected_narrative_{selected_case_id}",
        ):
            generate_requested = True

    # --------------------------------------------------------
    # Generate / regenerate
    # --------------------------------------------------------

    if generate_requested:

        with st.spinner(
            "Generating narrative and performing QC..."
        ):
            try:
                result_row = generate_and_qc(
                    selected_case_id,
                    case,
                )

                st.success(
                    "Narrative generated and QC completed."
                )

                st.write(
                    f"QC result: **{result_row.get('overall_status', 'REVIEW')}**"
                )

                st.cache_data.clear()
                st.rerun()

            except Exception as exc:
                st.error(
                    "Narrative generation failed."
                )
                st.exception(exc)

    # --------------------------------------------------------
    # RAB context
    # --------------------------------------------------------

    ctx = create_page_context(
        page_name="Narrative Generation",
        selected_case_id=selected_case_id,
        selected_record=case.to_dict(),
        metadata={
            "validation_status": validation_status,
            "duplicate_status": duplicate_status,
            "narrative_eligibility": eligibility,
            "narrative_generated": (
                existing_narrative is not None
                and has_real_narrative(
                    existing_narrative.get(
                        "generated_narrative",
                        "",
                    )
                )
            ),
        },
    )

    st.divider()

    render_rab_panel(
        ctx,
        page_name="Narrative Generation",
        expanded=False,
    )
