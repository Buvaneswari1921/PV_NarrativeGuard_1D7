"""
PV_NarrativeGuard - Audit Log

Purpose
-------
Provide append-only audit trails for:
    1. Core PV NarrativeGuard workflow events
    2. RAB operational events

Design principles
-----------------
- Audit records are append-only.
- Existing workflow audit records are preserved.
- RAB events use a dedicated audit log so the legacy PV schema is not changed.
- Every RAB request can produce one terminal operational event.
- Timestamps are UTC.
- Narrative content is represented by SHA-256 rather than duplicated.
- Audit failures must never crash the primary workflow.
"""

from __future__ import annotations

import csv
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Existing PV audit log
AUDIT_LOG_FILE = OUTPUT_DIR / "narrative_audit_log.csv"

# Dedicated RAB audit log
RAB_AUDIT_LOG_FILE = OUTPUT_DIR / "rab_audit_log.csv"


# ============================================================
# EXISTING PV AUDIT SCHEMA
# ============================================================

AUDIT_FIELDNAMES = [
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


# ============================================================
# RAB AUDIT SCHEMA
# ============================================================

RAB_AUDIT_FIELDNAMES = [
    "audit_timestamp_utc",
    "request_id",
    "query",
    "event_type",
    "actor_type",
    "route",
    "response_mode",
    "operational_state",
    "confidence",
    "evidence_count",
    "action",
    "reason",
]


# ============================================================
# HELPERS
# ============================================================

def utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


def safe_text(value: Any) -> str:
    """Safely convert a value to text."""
    if value is None:
        return ""

    return str(value).strip()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def hash_narrative(
    narrative: Optional[str],
) -> str:
    """
    Generate SHA-256 hash for narrative traceability.

    The narrative itself is not duplicated into the audit log.
    """
    text = safe_text(narrative)

    if not text:
        return ""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def generate_rab_request_id() -> str:
    """Generate a unique identifier for a RAB request."""
    return f"RAB-{uuid.uuid4().hex[:12].upper()}"


# ============================================================
# EXISTING PV AUDIT WRITE
# ============================================================

def write_audit_event(
    *,
    case_id: Any,
    event_type: str,
    actor_type: str = "SYSTEM",
    validation_status: Any = "",
    duplicate_status: Any = "",
    human_duplicate_decision: Any = "",
    narrative_eligibility: Any = "",
    narrative_status: Any = "",
    qc_status: Any = "",
    operational_state: Any = "",
    action: Any = "",
    reason: Any = "",
    narrative: Optional[str] = None,
) -> bool:
    """
    Append one audit event to the existing PV audit log.

    Audit persistence failures do not interrupt the primary workflow.
    """

    try:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_exists = AUDIT_LOG_FILE.exists()

        audit_record: Dict[str, Any] = {
            "audit_timestamp_utc": utc_timestamp(),
            "case_id": safe_text(case_id),
            "event_type": safe_text(event_type),
            "actor_type": safe_text(actor_type),
            "validation_status": safe_text(
                validation_status
            ),
            "duplicate_status": safe_text(
                duplicate_status
            ),
            "human_duplicate_decision": safe_text(
                human_duplicate_decision
            ),
            "narrative_eligibility": safe_text(
                narrative_eligibility
            ),
            "narrative_status": safe_text(
                narrative_status
            ),
            "qc_status": safe_text(
                qc_status
            ),
            "operational_state": safe_text(
                operational_state
            ),
            "action": safe_text(action),
            "reason": safe_text(reason),
            "narrative_hash": hash_narrative(
                narrative
            ),
        }

        with AUDIT_LOG_FILE.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=AUDIT_FIELDNAMES,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                audit_record
            )

        return True

    except Exception as exc:

        print(
            "[AUDIT WARNING] "
            f"Unable to write audit event: {exc}"
        )

        return False


# ============================================================
# RAB AUDIT WRITE
# ============================================================

def write_rab_audit_event(
    *,
    request_id: Any,
    query: Any,
    event_type: str,
    route: Any = "",
    response_mode: Any = "",
    operational_state: Any = "",
    confidence: Any = 0.0,
    evidence_count: Any = 0,
    action: Any = "",
    reason: Any = "",
    actor_type: str = "SYSTEM",
) -> bool:
    """
    Append one RAB operational event.

    RAB-specific audit data is intentionally kept separate from the
    legacy PV narrative audit schema.

    This function is deliberately fail-safe:
    audit persistence failure must never break the RAB response.
    """

    try:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_exists = RAB_AUDIT_LOG_FILE.exists()

        confidence_value = max(
            0.0,
            min(
                1.0,
                safe_float(confidence),
            ),
        )

        try:
            evidence_value = int(
                evidence_count
            )
        except (TypeError, ValueError):
            evidence_value = 0

        evidence_value = max(
            0,
            evidence_value,
        )

        record: Dict[str, Any] = {
            "audit_timestamp_utc": utc_timestamp(),
            "request_id": safe_text(request_id),
            "query": safe_text(query),
            "event_type": safe_text(event_type),
            "actor_type": safe_text(actor_type),
            "route": safe_text(route),
            "response_mode": safe_text(response_mode),
            "operational_state": safe_text(
                operational_state
            ),
            "confidence": confidence_value,
            "evidence_count": evidence_value,
            "action": safe_text(action),
            "reason": safe_text(reason),
        }

        with RAB_AUDIT_LOG_FILE.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=RAB_AUDIT_FIELDNAMES,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(record)

        return True

    except Exception as exc:

        print(
            "[RAB AUDIT WARNING] "
            f"Unable to write RAB audit event: {exc}"
        )

        return False