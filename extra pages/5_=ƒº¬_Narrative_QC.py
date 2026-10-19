"""PV NarrativeGuard - Narrative QC page.

Provides controlled review of generated narratives, persisted QC outcomes,
and current operational eligibility.
"""

import pandas as pd
import streamlit as st

from src.ui.data_loader import load_application_data
from src.ui.page_context import create_page_context
from src.ui.rab_ui import render_rab_panel
from src.ui.narrative_generation import render_existing_narrative
from src.ui.operational_state import determine_operational_state
from src.ui.components import render_global_styles, render_page_hero, render_workflow_strip


st.set_page_config(
    page_title="PV NarrativeGuard | Narrative QC",
    page_icon="QC",
    layout="wide",
)

render_global_styles()


# ============================================================
# LOAD APPLICATION DATA
# ============================================================

data = load_application_data()

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


# ============================================================
# PAGE TITLE
# ============================================================

render_page_hero(
    "Narrative quality control",
    "Inspect the generated narrative against source ICSR facts, field-level checks, and persisted QC outcomes before human PV sign-off.",
    eyebrow="WORKSPACE  /  QUALITY GATE",
    chips=["Field-level evidence", "Review exceptions", "Human sign-off"],
)
render_workflow_strip(5)


# ============================================================
# HELPERS
# ============================================================

def _has_narrative(row):
    """Return True when a persisted row contains a real narrative."""

    value = row.get("generated_narrative", "")

    if pd.isna(value):
        return False

    text = str(value).strip()

    return bool(text) and text.lower() != "nan"


def _latest_qc_records(df):
    """Return the latest persisted QC record for each case."""

    if df.empty or "case_id" not in df.columns:
        return pd.DataFrame()

    result = df.copy()

    result["case_id"] = (
        result["case_id"]
        .astype(str)
        .str.strip()
    )

    return result.drop_duplicates(
        subset=["case_id"],
        keep="last",
    )


def _current_case_record(case_id):
    """Return the current application-level case record."""

    if (
        final_df.empty
        or "case_id" not in final_df.columns
    ):
        return None

    matches = final_df[
        final_df["case_id"]
        .astype(str)
        .str.strip()
        == str(case_id).strip()
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def _source_case(case_id):
    """Return the source ICSR row for a case."""

    if (
        incoming_df.empty
        or "case_id" not in incoming_df.columns
    ):
        return None

    matches = incoming_df[
        incoming_df["case_id"]
        .astype(str)
        .str.strip()
        == str(case_id).strip()
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def _eligibility_label(value):
    """Normalize the current narrative eligibility value."""

    value = str(value or "").strip().upper()

    if not value:
        return "UNKNOWN"

    return value


#def _operational_state(eligibility, has_narrative):
    """Translate eligibility into a clear operational state."""

    if not has_narrative:
        return "NO GENERATED NARRATIVE"

    if eligibility == "ELIGIBLE":
        return "AVAILABLE - HUMAN PV REVIEW REQUIRED"

    if eligibility == "BLOCKED_DUPLICATE":
        return "HISTORICAL / BLOCKED - DUPLICATE"

    if eligibility == "PENDING_HUMAN_REVIEW":
        return "BLOCKED - AWAITING HUMAN DUPLICATE REVIEW"

    if eligibility == "BLOCKED":
        return "BLOCKED"

    return f"NOT CURRENTLY USABLE - {eligibility}"


# ============================================================
# PREPARE LATEST QC DATA
# ============================================================

latest_qc_df = _latest_qc_records(narrative_qc_df)

generated_qc_df = (
    latest_qc_df[
        latest_qc_df.apply(_has_narrative, axis=1)
    ].copy()
    if not latest_qc_df.empty
    else pd.DataFrame()
)

generated_count = len(generated_qc_df)

qc_pass_count = 0
qc_review_count = 0

if not generated_qc_df.empty and "overall_status" in generated_qc_df.columns:
    status_series = (
        generated_qc_df["overall_status"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    qc_pass_count = int((status_series == "PASS").sum())
    qc_review_count = int((status_series == "REVIEW").sum())


# ============================================================
# SUMMARY METRICS
# ============================================================

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Narratives generated",
        generated_count,
    )

with metric_2:
    st.metric(
        "QC PASS",
        qc_pass_count,
    )

with metric_3:
    st.metric(
        "QC REVIEW",
        qc_review_count,
    )

with metric_4:
    st.metric(
        "QC records",
        len(latest_qc_df),
    )


# ============================================================
# NO GENERATED NARRATIVES
# ============================================================

if generated_qc_df.empty:

    st.info(
        "No generated narratives are currently available for QC review."
    )

    ctx = create_page_context(
        page_name="Narrative QC",
        metadata={
            "generated_narratives": 0,
            "qc_pass": 0,
            "qc_review": 0,
            "qc_records": len(latest_qc_df),
        },
    )

    render_rab_panel(
        ctx=ctx,
        page_name="Narrative QC",
        expanded=False,
    )

    st.stop()


# ============================================================
# CASE SELECTION
# ============================================================

st.divider()

st.subheader("Select Case")

case_ids = (
    generated_qc_df["case_id"]
    .astype(str)
    .tolist()
)

selected_case_id = st.selectbox(
    "Case",
    ["Select a case..."] + case_ids,
    key="narrative_qc_case",
)


# ============================================================
# NO CASE SELECTED
# ============================================================

if selected_case_id == "Select a case...":

    st.info(
        "Select a generated case to review its current eligibility, "
        "narrative, and field-level QC evidence."
    )

    ctx = create_page_context(
        page_name="Narrative QC",
        metadata={
            "generated_narratives": generated_count,
            "qc_pass": qc_pass_count,
            "qc_review": qc_review_count,
            "qc_records": len(latest_qc_df),
        },
    )

    render_rab_panel(
        ctx=ctx,
        page_name="Narrative QC",
        expanded=False,
    )

    st.stop()


# ============================================================
# SELECTED RECORDS
# ============================================================

qc_rows = generated_qc_df[
    generated_qc_df["case_id"]
    .astype(str)
    .str.strip()
    == str(selected_case_id).strip()
]

if qc_rows.empty:

    st.error(
        "The selected narrative QC record could not be found."
    )

    st.stop()

existing_narrative = qc_rows.iloc[-1]

selected_record = _current_case_record(
    selected_case_id
)

case = _source_case(
    selected_case_id
)

if selected_record is None:

    st.error(
        "The current application record for this case could not be found."
    )

    st.stop()

if case is None:

    st.error(
        "The source ICSR for this case could not be found."
    )

    st.stop()


# ============================================================
# CURRENT OPERATIONAL STATUS
# ============================================================

eligibility = _eligibility_label(
    selected_record.get(
        "narrative_eligibility",
        "UNKNOWN",
    )
)

validation_status = str(
    selected_record.get(
        "validation_status",
        selected_record.get(
            "validation",
            "UNKNOWN",
        ),
    )
).strip()

duplicate_status = str(
    selected_record.get(
        "duplicate_status",
        "UNKNOWN",
    )
).strip()

has_narrative = _has_narrative(
    existing_narrative
)

operational_state = determine_operational_state(
    validation_status=validation_status,
    duplicate_status=duplicate_status,
    human_duplicate_decision=selected_record.get(
        "human_duplicate_decision",
        "",
    ),
    narrative_eligibility=eligibility,
    narrative_status=selected_record.get(
        "narrative_status",
        "",
    ),
    qc_status=existing_narrative.get(
        "overall_status",
        selected_record.get(
            "qc_status",
            "NOT_REVIEWED",
        ),
    ),
    generated_narrative=existing_narrative.get(
        "generated_narrative",
        "",
    ),
)


st.subheader(
    f"Case: {selected_case_id}"
)

status_1, status_2, status_3 = st.columns(3)

with status_1:

    if validation_status.upper() in {"VALID", "PASS"}:
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

    elif eligibility == "BLOCKED_DUPLICATE":
        st.warning(
            f"Narrative eligibility\n\n{eligibility}"
        )

    elif eligibility == "PENDING_HUMAN_REVIEW":
        st.warning(
            f"Narrative eligibility\n\n{eligibility}"
        )

    else:
        st.error(
            f"Narrative eligibility\n\n{eligibility}"
        )


# ============================================================
# OPERATIONAL STATE
# ============================================================

if operational_state == "READY - QC PASSED":

    st.success(
        f"Operational state: {operational_state}"
    )

    st.caption(
        "The narrative has passed automated factual QC and "
        "is available for qualified human pharmacovigilance review."
    )

elif operational_state == "READY - NARRATIVE GENERATION":

    st.info(
        f"Operational state: {operational_state}"
    )

    st.caption(
        "The case has passed validation and duplicate review "
        "and is available for explicit on-demand narrative generation."
    )

elif operational_state == "HUMAN PV REVIEW REQUIRED":

    st.warning(
        f"Operational state: {operational_state}"
    )

    st.caption(
        "A potential duplicate has been identified and requires "
        "a human pharmacovigilance duplicate decision."
    )

elif operational_state == "BLOCKED - CONFIRMED DUPLICATE":

    st.warning(
        f"Operational state: {operational_state}"
    )

    st.caption(
        "Narrative processing is blocked because the human PV "
        "duplicate decision classified this case as a duplicate."
    )

elif operational_state == "BLOCKED - INVALID ICSR":

    st.error(
        f"Operational state: {operational_state}"
    )

    st.caption(
        "The ICSR did not pass the required validation gate."
    )

elif operational_state == "REVIEW - QC ATTENTION REQUIRED":

    st.warning(
        f"Operational state: {operational_state}"
    )

    st.caption(
        "Narrative factual QC identified an issue requiring "
        "qualified pharmacovigilance review."
    )

else:

    st.info(
        f"Operational state: {operational_state}"
    )


# ============================================================
# HISTORICAL NARRATIVE NOTICE
# ============================================================

if (
    has_narrative
    and eligibility != "ELIGIBLE"
):

    st.info(
        "Historical narrative retained: the persisted narrative and QC "
        "result are displayed for traceability, but current eligibility "
        "controls whether the narrative is operationally usable."
    )


# ============================================================
# EXISTING NARRATIVE + FIELD-LEVEL QC
# ============================================================

render_existing_narrative(
    existing_narrative=existing_narrative,
    case=case,
)


# ============================================================
# PAGE CONTEXT FOR RAB
# ============================================================

ctx = create_page_context(
    page_name="Narrative QC",
    selected_case_id=str(selected_case_id),
    selected_record={
        key: (
            value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value in selected_record.to_dict().items()
        if not isinstance(value, float) or not pd.isna(value)
    },
    metadata={
        "validation_status": validation_status,
        "duplicate_status": duplicate_status,
        "narrative_eligibility": eligibility,
        "operational_state": operational_state,
        "qc_status": str(
            existing_narrative.get(
                "overall_status",
                "UNKNOWN",
            )
        ),
        "has_generated_narrative": has_narrative,
    },
)

render_rab_panel(
    ctx=ctx,
    page_name="Narrative QC",
    expanded=False,
)
