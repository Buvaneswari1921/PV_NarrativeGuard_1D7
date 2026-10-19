"""
PV NarrativeGuard - Operational State Engine.

Centralized case lifecycle/state determination for the UI.

Design principle:
    Individual columns such as validation_status,
    duplicate_status, narrative_eligibility and qc_status
    are evidence.

    operational_state is the single UI-facing lifecycle state.

Human PV duplicate decisions remain authoritative.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# OPERATIONAL STATES
# ============================================================

STATE_BLOCKED_INVALID = "BLOCKED - INVALID ICSR"
STATE_HUMAN_DUPLICATE_REVIEW = "HUMAN PV REVIEW REQUIRED"
STATE_BLOCKED_DUPLICATE = "BLOCKED - CONFIRMED DUPLICATE"
STATE_READY_GENERATION = "READY - NARRATIVE GENERATION"
STATE_QC_REVIEW = "REVIEW - QC ATTENTION REQUIRED"
STATE_QC_PASSED = "READY - QC PASSED"
STATE_NO_NARRATIVE = "NO GENERATED NARRATIVE"
STATE_UNKNOWN = "UNKNOWN"


# ============================================================
# NORMALIZATION
# ============================================================

def _text(value: Any) -> str:
    """Return a normalized uppercase text value."""

    if value is None:
        return ""

    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (ImportError, TypeError, ValueError):
        pass

    return str(value).strip().upper()


def _has_narrative(value: Any) -> bool:
    """Return True only when a real narrative exists."""

    if value is None:
        return False

    try:
        import pandas as pd

        if pd.isna(value):
            return False
    except (ImportError, TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return False

    return text.lower() not in {
        "nan",
        "none",
        "null",
    }


# ============================================================
# STATE DETERMINATION
# ============================================================

def determine_operational_state(
    *,
    validation_status: Any = "",
    duplicate_status: Any = "",
    human_duplicate_decision: Any = "",
    narrative_eligibility: Any = "",
    narrative_status: Any = "",
    qc_status: Any = "",
    generated_narrative: Any = "",
) -> str:
    """
    Determine the current operational lifecycle state.

    Priority:
        1. Invalid ICSR
        2. Human duplicate decision = DUPLICATE
        3. Pending human duplicate review
        4. QC REVIEW
        5. QC PASS
        6. Eligible but no narrative
        7. Eligible narrative without QC
        8. Unknown
    """

    validation = _text(validation_status)
    duplicate = _text(duplicate_status)
    human_decision = _text(human_duplicate_decision)
    eligibility = _text(narrative_eligibility)
    narrative_state = _text(narrative_status)
    qc = _text(qc_status)

    has_narrative = _has_narrative(
        generated_narrative
    )

    # --------------------------------------------------------
    # 1. Validation gate
    # --------------------------------------------------------

    if validation in {
        "INVALID",
        "INVALID / INSUFFICIENT",
        "EXCLUDED",
    }:
        return STATE_BLOCKED_INVALID

    # --------------------------------------------------------
    # 2. Human duplicate decision is authoritative
    # --------------------------------------------------------

    if human_decision == "DUPLICATE":
        return STATE_BLOCKED_DUPLICATE

    # --------------------------------------------------------
    # 3. Potential duplicate awaiting human decision
    # --------------------------------------------------------

    if human_decision not in {
        "NEW",
        "DUPLICATE",
    }:

        if eligibility == "PENDING_HUMAN_REVIEW":
            return STATE_HUMAN_DUPLICATE_REVIEW

        if duplicate in {
            "POTENTIAL DUPLICATE - HUMAN REVIEW",
            "POTENTIAL DUPLICATE - HUMAN REVIEW REQUIRED",
            "HIGH-CONFIDENCE DUPLICATE CANDIDATE",
        }:
            return STATE_HUMAN_DUPLICATE_REVIEW

    # --------------------------------------------------------
    # 4. Narrative QC requires attention
    # --------------------------------------------------------

    if qc == "REVIEW":
        return STATE_QC_REVIEW

    # --------------------------------------------------------
    # 5. Narrative QC passed
    # --------------------------------------------------------

    if qc == "PASS" and has_narrative:
        return STATE_QC_PASSED

    # --------------------------------------------------------
    # 6. Eligible but no narrative yet
    # --------------------------------------------------------

    if eligibility == "ELIGIBLE" and not has_narrative:
        return STATE_READY_GENERATION

    # --------------------------------------------------------
    # 7. Eligible narrative exists but QC is not yet available
    # --------------------------------------------------------

    if eligibility == "ELIGIBLE" and has_narrative:

        if qc in {
            "",
            "NOT_REVIEWED",
            "NOT_RUN",
            "UNKNOWN",
        }:
            return STATE_NO_NARRATIVE

    # --------------------------------------------------------
    # 8. Defensive fallback
    # --------------------------------------------------------

    if narrative_state.startswith("BLOCKED"):
        return STATE_BLOCKED_DUPLICATE

    return STATE_UNKNOWN


# ============================================================
# DATAFRAME ENRICHMENT
# ============================================================

def apply_operational_state(df):
    """
    Add operational_state to a case-level dataframe.

    Existing columns are preserved.
    """

    if df is None or df.empty:
        return df

    result = df.copy()

    def get(row, column):
        if column not in result.columns:
            return ""

        return row.get(column, "")

    result["operational_state"] = [
        determine_operational_state(
            validation_status=get(row, "validation_status"),
            duplicate_status=get(row, "duplicate_status"),
            human_duplicate_decision=get(
                row,
                "human_duplicate_decision",
            ),
            narrative_eligibility=get(
                row,
                "narrative_eligibility",
            ),
            narrative_status=get(
                row,
                "narrative_status",
            ),
            qc_status=get(row, "qc_status"),
            generated_narrative=get(
                row,
                "generated_narrative",
            ),
        )
        for _, row in result.iterrows()
    ]

    return result
