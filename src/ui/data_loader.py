"""Shared data-loading utilities for PV NarrativeGuard Streamlit UI.

This module centralizes access to application output files.

UI pages should use these loaders instead of independently
reading CSV/JSON files. This keeps page implementations
consistent and makes future backend changes easier.
"""

from pathlib import Path
import json

import pandas as pd

from src.ui.operational_state import apply_operational_state


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

FINAL_RESULTS_FILE = OUTPUT_DIR / "final_pipeline_results.csv"
VALIDATION_RESULTS_FILE = OUTPUT_DIR / "validation_results.csv"
INCOMING_FILE = DATA_DIR / "incoming_icsrs.csv"
HUMAN_REVIEW_FILE = OUTPUT_DIR / "duplicate_human_review.csv"
NARRATIVE_QC_FILE = OUTPUT_DIR / "narrative_qc_results.csv"
INVESTIGATION_FILE = OUTPUT_DIR / "duplicate_investigations.json"
AUDIT_LOG_FILE = OUTPUT_DIR / "narrative_audit_log.csv"


# ============================================================
# EMPTY DATA HELPERS
# ============================================================

def _empty_review_dataframe():
    return pd.DataFrame(
        columns=[
            "incoming_case_id",
            "decision",
        ]
    )


# ============================================================
# CSV LOADERS
# ============================================================

def load_final_results():
    """Load final pipeline results."""

    if not FINAL_RESULTS_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(FINAL_RESULTS_FILE)


def load_validation_results():
    """Load validation results."""

    if not VALIDATION_RESULTS_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(VALIDATION_RESULTS_FILE)


def load_incoming_cases():
    """Load incoming ICSR cases."""

    if not INCOMING_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(INCOMING_FILE)


def load_narrative_qc():
    """Load persisted narrative QC results."""

    if not NARRATIVE_QC_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(NARRATIVE_QC_FILE)

def load_audit_log():
    """Load persisted NarrativeGuard audit events."""

    if not AUDIT_LOG_FILE.exists():
        return pd.DataFrame(
            columns=[
                "audit_timestamp_utc",
                "case_id",
                "event_type",
                "actor_type",
                "validation_status",
                "duplicate_status",
                "human_duplicate_decision",
                "narrative_eligibility",
                "narrative_status",
                "qc_status",
                "operational_state",
                "action",
                "reason",
                "narrative_hash",
            ]
        )

    try:
        audit_df = pd.read_csv(
            AUDIT_LOG_FILE
        )

        return audit_df

    except (
        OSError,
        pd.errors.EmptyDataError,
    ):
        return pd.DataFrame()


# ============================================================
# INVESTIGATION DATA
# ============================================================

def load_investigations():
    """Load duplicate investigation records."""

    if not INVESTIGATION_FILE.exists():
        return {}

    try:
        with open(
            INVESTIGATION_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        OSError,
        ValueError,
    ):
        return {}


# ============================================================
# HUMAN REVIEW
# ============================================================

def load_human_review_decisions():
    """Load persisted PV reviewer duplicate decisions.

    The latest valid decision for each case is retained.
    """

    if not HUMAN_REVIEW_FILE.exists():
        return _empty_review_dataframe()

    try:
        review_df = pd.read_csv(
            HUMAN_REVIEW_FILE
        )

        required = {
            "incoming_case_id",
            "decision",
        }

        if not required.issubset(
            review_df.columns
        ):
            return _empty_review_dataframe()

        review_df = review_df[
            [
                "incoming_case_id",
                "decision",
            ]
        ].copy()

        review_df["incoming_case_id"] = (
            review_df["incoming_case_id"]
            .astype(str)
        )

        review_df["decision"] = (
            review_df["decision"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        review_df = review_df[
            review_df["decision"].isin(
                [
                    "NEW",
                    "DUPLICATE",
                ]
            )
        ]

        review_df = review_df.drop_duplicates(
            subset=["incoming_case_id"],
            keep="last",
        )

        return review_df

    except (
        OSError,
        pd.errors.EmptyDataError,
    ):
        return _empty_review_dataframe()


# ============================================================
# APPLICATION-LEVEL DATA ENRICHMENT
# ============================================================

def apply_human_review_decisions(
    pipeline_df,
    review_df,
):
    """Apply persisted human PV decisions to pipeline results.

    Human review remains the source of truth for duplicate
    decisions. The resulting dataframe is intended for UI use.
    """

    result = pipeline_df.copy()

    if result.empty or review_df.empty:
        return result

    if "case_id" not in result.columns:
        return result

    result["case_id"] = (
        result["case_id"]
        .astype(str)
    )

    review_lookup = (
        review_df
        .set_index("incoming_case_id")["decision"]
        .to_dict()
    )

    result["human_duplicate_decision"] = (
        result["case_id"]
        .map(review_lookup)
        .fillna(
            result.get(
                "human_duplicate_decision",
                pd.Series(
                    index=result.index,
                    dtype="object",
                ),
            )
        )
    )

    if "narrative_eligibility" in result.columns:

        result.loc[
            result["human_duplicate_decision"]
            == "NEW",
            "narrative_eligibility",
        ] = "ELIGIBLE"

        result.loc[
            result["human_duplicate_decision"]
            == "DUPLICATE",
            "narrative_eligibility",
        ] = "BLOCKED_DUPLICATE"

    if "narrative_status" in result.columns:

        result.loc[
            result["human_duplicate_decision"]
            == "NEW",
            "narrative_status",
        ] = "NOT_GENERATED - ON DEMAND"

        result.loc[
            result["human_duplicate_decision"]
            == "DUPLICATE",
            "narrative_status",
        ] = "BLOCKED - CONFIRMED DUPLICATE"

    return result


def attach_qc_status(
    pipeline_df,
    narrative_qc_df,
):
    """Attach the latest narrative QC status to case-level data."""

    result = pipeline_df.copy()

    if result.empty:
        return result

    if (
        narrative_qc_df.empty
        or "case_id" not in narrative_qc_df.columns
        or "overall_status" not in narrative_qc_df.columns
    ):
        result["qc_status"] = "NOT_REVIEWED"
        return result

    qc_lookup = (
        narrative_qc_df
        .drop_duplicates(
            subset=["case_id"],
            keep="last",
        )
        .set_index("case_id")["overall_status"]
        .to_dict()
    )

    result["qc_status"] = (
        result["case_id"]
        .astype(str)
        .map(qc_lookup)
        .fillna("NOT_REVIEWED")
    )

    return result


def load_application_data():
    """Load and prepare all primary UI data.

    Returns a dictionary so individual Streamlit pages can
    consume a consistent application-level data model.
    """

    final_df = load_final_results()
    incoming_df = load_incoming_cases()
    validation_results_df = load_validation_results()
    narrative_qc_df = load_narrative_qc()
    investigations = load_investigations()
    human_review_df = load_human_review_decisions()
    audit_log_df = load_audit_log()

    final_df = apply_human_review_decisions(
        final_df,
        human_review_df,
    )

    final_df = attach_qc_status(
        final_df,
        narrative_qc_df,
    )

    final_df = apply_operational_state(
        final_df
    )

    return {
        "final_df": final_df,
        "incoming_df": incoming_df,
        "validation_results_df": validation_results_df,
        "narrative_qc_df": narrative_qc_df,
        "investigations": investigations,
        "human_review_df": human_review_df,
        "audit_log_df": audit_log_df,
    }