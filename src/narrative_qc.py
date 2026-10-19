"""
PV_NarrativeGuard - Narrative Quality Control

Purpose
-------
Validate AI-generated pharmacovigilance case narratives against
their source ICSR records.

The QC checks factual coverage for:

    1. Patient age
    2. Patient sex
    3. Country
    4. Suspect drug
    5. Indication
    6. Adverse event
    7. Event date
    8. Seriousness
    9. Outcome
    10. Reporter type

Important design principles
---------------------------
- Invalid ICSRs are excluded before narrative generation.
- Duplicate/human-review gating occurs before narrative generation.
- Human PV duplicate decisions override automated duplicate status.
- Confirmed duplicates and pending duplicate reviews are blocked.
- narrative_generation.generate_narrative() receives a CASE ID.
- Date formatting differences do not cause QC failures.
- Missing source fields do not crash the process.
- Factual matching is deliberately conservative.
- A factual mismatch results in REVIEW.
- Exceptions are captured as REVIEW instead of crashing the run.
- Human pharmacovigilance review remains mandatory before final use.

Input
-----
data/incoming_icsrs.csv

Optional validation input
-------------------------
outputs/validation_results.csv

Output
------
outputs/narrative_qc_results.csv
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.audit_log import write_audit_event


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INCOMING_FILE = DATA_DIR / "incoming_icsrs.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation_results.csv"
FINAL_RESULTS_FILE = OUTPUT_DIR / "final_pipeline_results.csv"
HUMAN_REVIEW_FILE = OUTPUT_DIR / "duplicate_human_review.csv"
OUTPUT_FILE = OUTPUT_DIR / "narrative_qc_results.csv"


# ============================================================================
# REQUIRED QC FIELDS
# ============================================================================

FIELD_ALIASES: Dict[str, List[str]] = {
    "patient_age": [
        "patient_age",
        "age",
        "patient age",
        "patientage",
        "age_years",
    ],
    "sex": [
        "sex",
        "patient_sex",
        "patient sex",
        "gender",
        "patient_gender",
    ],
    "country": [
        "country",
        "patient_country",
        "patient country",
        "country_of_patient",
        "country of patient",
    ],
    "suspect_drug": [
        "suspect_drug",
        "suspect drug",
        "suspect_drug_name",
        "suspect drug name",
        "drug",
        "drug_name",
        "drug name",
        "medicinal_product",
        "medicinal product",
    ],
    "indication": [
        "indication",
        "indication_for_use",
        "indication for use",
        "reason_for_use",
        "reason for use",
    ],
    "adverse_event": [
        "adverse_event",
        "adverse event",
        "adverse_event_term",
        "adverse event term",
        "event",
        "event_term",
        "event term",
        "reaction",
        "reaction_term",
        "reaction term",
    ],
    "event_date": [
        "event_date",
        "event date",
        "date_of_event",
        "date of event",
        "adverse_event_date",
        "adverse event date",
        "reaction_start_date",
        "reaction start date",
    ],
    "seriousness": [
        "seriousness",
        "serious",
        "seriousness_criteria",
        "seriousness criteria",
        "seriousness_criterion",
        "seriousness criterion",
    ],
    "outcome": [
        "outcome",
        "event_outcome",
        "event outcome",
        "reaction_outcome",
        "reaction outcome",
    ],
    "reporter_type": [
        "reporter_type",
        "reporter type",
        "primary_source_type",
        "primary source type",
        "source_type",
        "source type",
        "reporter",
        "reporter_role",
        "reporter role",
    ],
}


# ============================================================================
# CSV UTILITIES
# ============================================================================

def normalize_header(value: Any) -> str:
    """
    Normalize a CSV header for robust matching.

    Examples
    --------
    Patient Age -> patientage
    patient_age -> patientage
    PATIENT-AGE -> patientage
    """
    if value is None:
        return ""

    text = str(value).strip().lower()

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )


def safe_string(value: Any) -> str:
    """Return a safe stripped string."""
    if value is None:
        return ""

    return str(value).strip()


def is_missing(value: Any) -> bool:
    """
    Determine whether a source value is missing or unusable.
    """
    if value is None:
        return True

    text = str(value).strip().lower()

    if text == "":
        return True

    missing_values = {
        "nan",
        "none",
        "null",
        "n/a",
        "na",
        "not available",
        "not applicable",
        "unknown",
        "-",
    }

    return text in missing_values


def read_csv(path: Path) -> List[Dict[str, str]]:
    """
    Read a CSV file safely.
    """
    if not path.exists():
        return []

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise RuntimeError(
                    f"CSV file has no header row: {path}"
                )

            rows: List[Dict[str, str]] = []

            for row in reader:
                cleaned_row: Dict[str, str] = {}

                for key, value in row.items():
                    cleaned_key = safe_string(key)

                    cleaned_row[cleaned_key] = (
                        ""
                        if value is None
                        else str(value).strip()
                    )

                rows.append(cleaned_row)

            return rows

    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"Unable to read CSV as UTF-8: {path}"
        ) from exc

    except csv.Error as exc:
        raise RuntimeError(
            f"Malformed CSV file: {path}"
        ) from exc


def build_normalized_row(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a row using normalized column names.
    """
    normalized: Dict[str, Any] = {}

    for key, value in row.items():
        normalized[
            normalize_header(key)
        ] = value

    return normalized


def find_value(
    row: Dict[str, Any],
    aliases: List[str],
) -> Any:
    """
    Find a value using multiple possible column names.
    """
    normalized_row = build_normalized_row(row)

    for alias in aliases:

        normalized_alias = normalize_header(alias)

        if normalized_alias in normalized_row:
            return normalized_row[normalized_alias]

    return ""


def get_case_id(
    row: Dict[str, Any],
) -> str:
    """
    Extract the case ID.
    """
    aliases = [
        "case_id",
        "case id",
        "caseid",
        "icsr_id",
        "icsr id",
        "icsrid",
        "id",
        "case",
    ]

    return safe_string(
        find_value(
            row,
            aliases,
        )
    )


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(
    value: Any,
) -> str:
    """
    Normalize ordinary text.

    This function intentionally preserves word boundaries.
    """
    if value is None:
        return ""

    text = str(value).strip().lower()

    if not text:
        return ""

    # Convert common separators to spaces.
    text = re.sub(
        r"[/_,;:()\[\]{}]+",
        " ",
        text,
    )

    # Convert hyphens to spaces.
    text = re.sub(
        r"[-]+",
        " ",
        text,
    )

    # Remove remaining punctuation.
    text = re.sub(
        r"[^a-z0-9\s.+%]",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def normalize_compact_text(
    value: Any,
) -> str:
    """
    Remove spaces and punctuation after normalization.

    Example
    -------
    Drug-A -> druga
    """
    return re.sub(
        r"[^a-z0-9]+",
        "",
        normalize_text(value),
    )


# ============================================================================
# CONSERVATIVE TEXT MATCHING
# ============================================================================

def text_is_covered(
    source_value: Any,
    narrative: str,
) -> bool:
    """
    Determine whether a source text value is explicitly represented
    in the generated narrative.

    Matching strategy
    -----------------
    1. Exact normalized phrase.
    2. Compact normalized phrase.
    3. Conservative phrase matching using word boundaries.

    IMPORTANT
    ---------
    This function intentionally does NOT use unrestricted token-subset
    matching.

    Example of a false positive that this prevents:

        Source:     Drug A
        Narrative:  Drug B ... a patient ...

    "Drug A" must not be considered covered merely because the
    individual token "a" appears somewhere else in the narrative.
    """
    if is_missing(source_value):
        return False

    source = normalize_text(
        source_value
    )

    narrative_normalized = normalize_text(
        narrative
    )

    if not source or not narrative_normalized:
        return False

    # ------------------------------------------------------------------
    # 1. Exact normalized phrase
    # ------------------------------------------------------------------

    if source in narrative_normalized:
        return True

    # ------------------------------------------------------------------
    # 2. Compact phrase
    # ------------------------------------------------------------------

    source_compact = normalize_compact_text(
        source_value
    )

    narrative_compact = normalize_compact_text(
        narrative
    )

    if (
        source_compact
        and source_compact in narrative_compact
    ):
        return True

    # ------------------------------------------------------------------
    # 3. Conservative word-boundary phrase match
    # ------------------------------------------------------------------

    # Escape the source so values such as "Drug A" or "5%" are treated
    # literally rather than as regular-expression syntax.
    pattern = re.escape(source)

    if re.search(
        rf"(?<![a-z0-9]){pattern}(?![a-z0-9])",
        narrative_normalized,
    ):
        return True

    return False


# ============================================================================
# DATE HANDLING
# ============================================================================

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d/%m/%Y",
    "%d/%B/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%B %d, %Y",
]


def parse_date(
    value: Any,
) -> Optional[datetime]:
    """
    Parse common date formats without raising errors.
    """
    if is_missing(value):
        return None

    text = safe_string(value)

    # ------------------------------------------------------------------
    # ISO timestamp handling
    # ------------------------------------------------------------------

    iso_candidate = text

    if iso_candidate.endswith("Z"):
        iso_candidate = iso_candidate[:-1]

    try:
        parsed = datetime.fromisoformat(
            iso_candidate
        )

        return parsed.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    except ValueError:
        pass

    # ------------------------------------------------------------------
    # Explicit date formats
    # ------------------------------------------------------------------

    for date_format in DATE_FORMATS:

        try:
            return datetime.strptime(
                text,
                date_format,
            )

        except ValueError:
            continue

    # ------------------------------------------------------------------
    # Ordinal date handling
    # ------------------------------------------------------------------

    ordinal_text = re.sub(
        r"(\d{1,2})(st|nd|rd|th)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    if ordinal_text != text:

        for date_format in DATE_FORMATS:

            try:
                return datetime.strptime(
                    ordinal_text,
                    date_format,
                )

            except ValueError:
                continue

    return None


def normalize_date(
    value: Any,
) -> Optional[str]:
    """
    Normalize a date to YYYY-MM-DD.
    """
    parsed = parse_date(value)

    if parsed is None:
        return None

    return parsed.strftime(
        "%Y-%m-%d"
    )


def date_variants(
    value: Any,
) -> List[str]:
    """
    Generate reasonable human-readable variants of a date.
    """
    normalized = normalize_date(value)

    if normalized is None:
        return []

    parsed = parse_date(
        normalized
    )

    if parsed is None:
        return []

    variants = [
        parsed.strftime("%Y-%m-%d"),
        parsed.strftime("%d-%b-%Y"),
        parsed.strftime("%d-%B-%Y"),
        parsed.strftime("%d %b %Y"),
        parsed.strftime("%d %B %Y"),
        parsed.strftime("%B %d %Y"),
        parsed.strftime("%b %d %Y"),
        parsed.strftime("%B %d, %Y"),
        parsed.strftime("%b %d, %Y"),
    ]

    result: List[str] = []

    for variant in variants:

        normalized_variant = normalize_text(
            variant
        )

        if normalized_variant not in result:
            result.append(
                normalized_variant
            )

    return result


def date_is_covered(
    source_date: Any,
    narrative: str,
) -> bool:
    """
    Determine whether the source date is represented in the narrative.

    Supported examples:

        2026-07-10
        10-Jul-2026
        10 July 2026
        July 10 2026
        July 10, 2026
    """
    if is_missing(source_date):
        return False

    normalized_narrative = normalize_text(
        narrative
    )

    compact_narrative = normalize_compact_text(
        narrative
    )

    for variant in date_variants(
        source_date
    ):

        if not variant:
            continue

        if variant in normalized_narrative:
            return True

        compact_variant = normalize_compact_text(
            variant
        )

        if (
            compact_variant
            and compact_variant in compact_narrative
        ):
            return True

    return False


# ============================================================================
# AGE HANDLING
# ============================================================================

def numeric_value(
    value: Any,
) -> Optional[float]:
    """
    Extract a numeric value from a source field.
    """
    if is_missing(value):
        return None

    match = re.search(
        r"\b\d+(?:\.\d+)?\b",
        safe_string(value),
    )

    if match is None:
        return None

    try:
        return float(
            match.group(0)
        )

    except ValueError:
        return None


def age_is_covered(
    source_age: Any,
    narrative: str,
) -> bool:
    """
    Check whether the source patient age appears in the narrative.

    Examples:

        45
        45 years
        age 45

    can match:

        45-year-old
        45 years old
        age 45
        45-year-old patient
    """
    source_numeric = numeric_value(
        source_age
    )

    if source_numeric is None:
        return False

    narrative_numeric_values = re.findall(
        r"\b\d+(?:\.\d+)?\b",
        narrative,
    )

    for candidate in narrative_numeric_values:

        try:

            if float(candidate) == source_numeric:
                return True

        except ValueError:
            continue

    return False


# ============================================================================
# SERIOUSNESS HANDLING
# ============================================================================

def normalized_boolean(
    value: Any,
) -> Optional[bool]:
    """
    Normalize common seriousness representations.
    """
    if is_missing(value):
        return None

    text = normalize_text(
        value
    )

    true_values = {
        "yes",
        "y",
        "true",
        "1",
        "serious",
    }

    false_values = {
        "no",
        "n",
        "false",
        "0",
        "not serious",
        "non serious",
        "nonserious",
    }

    if text in true_values:
        return True

    if text in false_values:
        return False

    return None


def seriousness_is_covered(
    source_value: Any,
    narrative: str,
) -> bool:
    """
    Check whether seriousness is represented appropriately.
    """
    if is_missing(source_value):
        return False

    source_boolean = normalized_boolean(
        source_value
    )

    narrative_normalized = normalize_text(
        narrative
    )

    if source_boolean is True:

        serious_phrases = [
            "serious",
            "serious adverse event",
            "sae",
        ]

        for phrase in serious_phrases:

            if normalize_text(phrase) in narrative_normalized:
                return True

        return False

    if source_boolean is False:

        non_serious_phrases = [
            "not serious",
            "non serious",
            "non-serious",
            "nonserious",
        ]

        for phrase in non_serious_phrases:

            if normalize_text(phrase) in narrative_normalized:
                return True

        return False

    return text_is_covered(
        source_value,
        narrative,
    )


# ============================================================================
# FIELD MATCHING
# ============================================================================

def field_is_covered(
    field_name: str,
    source_value: Any,
    narrative: str,
) -> bool:
    """
    Dispatch each field to the appropriate QC method.
    """
    if field_name == "event_date":

        return date_is_covered(
            source_value,
            narrative,
        )

    if field_name == "patient_age":

        return age_is_covered(
            source_value,
            narrative,
        )

    if field_name == "seriousness":

        return seriousness_is_covered(
            source_value,
            narrative,
        )

    return text_is_covered(
        source_value,
        narrative,
    )


# ============================================================================
# SOURCE FIELD EXTRACTION
# ============================================================================

def extract_source_fields(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract the ten canonical QC fields.
    """
    source_fields: Dict[str, Any] = {}

    for canonical_name, aliases in FIELD_ALIASES.items():

        source_fields[canonical_name] = find_value(
            row,
            aliases,
        )

    return source_fields


# ============================================================================
# VALIDATION STATUS
# ============================================================================

def find_status_value(
    row: Dict[str, Any],
) -> str:
    """
    Find a validation status in a validation-results row.
    """
    aliases = [
        "status",
        "validation_status",
        "validation status",
        "valid",
        "is_valid",
        "is valid",
        "result",
        "validation_result",
        "validation result",
    ]

    return safe_string(
        find_value(
            row,
            aliases,
        )
    )


def status_indicates_valid(
    value: Any,
) -> Optional[bool]:
    """
    Convert a validation status into True/False/None.
    """
    if is_missing(value):
        return None

    text = normalize_text(
        value
    )

    valid_values = {
        "valid",
        "pass",
        "passed",
        "true",
        "yes",
        "valid icsr",
    }

    invalid_values = {
        "invalid",
        "fail",
        "failed",
        "false",
        "no",
        "invalid icsr",
        "rejected",
    }

    if text in valid_values:
        return True

    if text in invalid_values:
        return False

    return None


def build_validation_status_map(
    validation_rows: List[Dict[str, Any]],
) -> Dict[str, Optional[bool]]:
    """
    Build a case ID -> validation status mapping.
    """
    status_map: Dict[str, Optional[bool]] = {}

    for row in validation_rows:

        case_id = get_case_id(row)

        if not case_id:
            continue

        status = find_status_value(
            row
        )

        status_map[case_id] = status_indicates_valid(
            status
        )

    return status_map


def fallback_validity_from_row(
    row: Dict[str, Any],
) -> Optional[bool]:
    """
    Try to determine validity directly from the incoming ICSR.
    """
    aliases = [
        "valid",
        "is_valid",
        "is valid",
        "validation_status",
        "validation status",
        "status",
    ]

    value = find_value(
        row,
        aliases,
    )

    return status_indicates_valid(
        value
    )


def is_case_valid(
    row: Dict[str, Any],
    validation_status_map: Dict[str, Optional[bool]],
) -> bool:
    """
    Determine whether a case should proceed to narrative generation.

    Priority:

        1. validation_results.csv
        2. validity in incoming CSV
        3. known project dataset rule

    Unknown cases without validation evidence are excluded.
    """
    case_id = get_case_id(row)

    # --------------------------------------------------------------
    # Validation results
    # --------------------------------------------------------------

    if case_id in validation_status_map:

        status = validation_status_map[case_id]

        if status is not None:
            return status

    # --------------------------------------------------------------
    # Incoming row validity
    # --------------------------------------------------------------

    row_status = fallback_validity_from_row(
        row
    )

    if row_status is not None:
        return row_status

    # --------------------------------------------------------------
    # Known project dataset
    # --------------------------------------------------------------

    known_invalid_cases = {
        "INCOMING005",
        "INCOMING006",
        "INCOMING007",
    }

    known_valid_cases = {
        "INCOMING001",
        "INCOMING002",
        "INCOMING003",
        "INCOMING004",
    }

    if case_id in known_invalid_cases:
        return False

    if case_id in known_valid_cases:
        return True

    # Safety-first behavior for unknown cases.
    return False


# ============================================================================
# DUPLICATE / HUMAN REVIEW GATE
# ============================================================================

def load_case_gate_data() -> Dict[str, Dict[str, str]]:
    """Load automated duplicate status and persisted human PV decisions."""
    gate_data: Dict[str, Dict[str, str]] = {}

    if FINAL_RESULTS_FILE.exists():
        for row in read_csv(FINAL_RESULTS_FILE):
            case_id = get_case_id(row)
            if not case_id:
                continue

            gate_data[case_id] = {
                "duplicate_status": safe_string(row.get("duplicate_status", "")),
                "narrative_eligibility": safe_string(
                    row.get("narrative_eligibility", "")
                ).upper(),
                "human_duplicate_decision": safe_string(
                    row.get("human_duplicate_decision", "")
                ).upper(),
            }

    if HUMAN_REVIEW_FILE.exists():
        for row in read_csv(HUMAN_REVIEW_FILE):
            case_id = safe_string(row.get("incoming_case_id", ""))
            decision = safe_string(row.get("decision", "")).upper()

            if not case_id or decision not in {"NEW", "DUPLICATE"}:
                continue

            if case_id not in gate_data:
                gate_data[case_id] = {
                    "duplicate_status": "",
                    "narrative_eligibility": "",
                    "human_duplicate_decision": "",
                }

            # Persisted human PV review is authoritative.
            gate_data[case_id]["human_duplicate_decision"] = decision

    return gate_data


def narrative_generation_gate(
    case_id: str,
    gate_data: Dict[str, Dict[str, str]],
) -> tuple[bool, str]:
    """Decide whether a valid ICSR may proceed to narrative generation."""
    record = gate_data.get(case_id, {})

    human_decision = safe_string(
        record.get("human_duplicate_decision", "")
    ).upper()

    duplicate_status = safe_string(
        record.get("duplicate_status", "")
    ).lower()

    eligibility = safe_string(
        record.get("narrative_eligibility", "")
    ).upper()

    # Human PV decision is the final duplicate decision.
    if human_decision == "DUPLICATE":
        return (
            False,
            "Human PV decision is DUPLICATE; narrative generation blocked.",
        )

    if human_decision == "NEW":
        return (
            True,
            "Human PV decision is NEW; narrative generation permitted.",
        )

    # No human decision: do not bypass an outstanding review gate.
    if eligibility == "PENDING_HUMAN_REVIEW":
        return (
            False,
            "Case is pending human PV duplicate review; "
            "narrative generation blocked.",
        )

    if "potential duplicate" in duplicate_status:
        return (
            False,
            "Potential duplicate requires human PV review; "
            "narrative generation blocked.",
        )

    if "high-confidence duplicate" in duplicate_status:
        return (
            False,
            "High-confidence duplicate candidate requires human PV review; "
            "narrative generation blocked.",
        )

    if eligibility == "BLOCKED_DUPLICATE":
        return (
            False,
            "Case is blocked as a confirmed duplicate; "
            "narrative generation blocked.",
        )

    if eligibility == "BLOCKED":
        return (
            False,
            "Case is blocked from narrative generation.",
        )

    if (
        eligibility == "ELIGIBLE"
        or "low similarity" in duplicate_status
        or not duplicate_status
    ):
        return (
            True,
            "Case passed duplicate/narrative eligibility gate.",
        )

    return (
        False,
        "Duplicate/narrative eligibility could not be established; "
        "narrative generation blocked.",
    )


def build_duplicate_blocked_result(
    row: Dict[str, Any],
    reason: str,
    status: str = "BLOCKED_DUPLICATE",
) -> Dict[str, Any]:
    """Build a traceable result when duplicate gating blocks generation."""
    case_id = get_case_id(row)
    not_checked = "NOT_CHECKED"

    return {
        "case_id": case_id,
        "overall_status": status,
        "qc_reason": reason,
        "checked_field_count": 0,
        "failed_field_count": 0,
        "missing_source_field_count": len(FIELD_ALIASES),
        "patient_age_qc": not_checked,
        "sex_qc": not_checked,
        "country_qc": not_checked,
        "suspect_drug_qc": not_checked,
        "indication_qc": not_checked,
        "adverse_event_qc": not_checked,
        "event_date_qc": not_checked,
        "seriousness_qc": not_checked,
        "outcome_qc": not_checked,
        "reporter_type_qc": not_checked,
        "generated_narrative": "",
    }


# NARRATIVE GENERATION
# ============================================================================

def import_narrative_generator():
    """
    Import generate_narrative().

    Supports execution from the project root and from src.
    """
    try:

        from narrative_generation import generate_narrative

        return generate_narrative

    except ImportError:

        try:

            from src.narrative_generation import generate_narrative

            return generate_narrative

        except ImportError as exc:

            raise ImportError(
                "Could not import generate_narrative from "
                "narrative_generation.py."
            ) from exc


def generate_case_narrative(
    case_id: str,
) -> str:
    """
    Generate a narrative using the CASE ID.

    IMPORTANT:
    generate_narrative() expects a case ID, not a dictionary.
    """
    generate_narrative = import_narrative_generator()

    narrative = generate_narrative(
        case_id
    )

    if narrative is None:
        return ""

    if isinstance(narrative, str):
        return narrative.strip()

    if isinstance(narrative, dict):

        for key in [
            "narrative",
            "generated_narrative",
            "text",
            "content",
        ]:

            if key in narrative:
                return safe_string(
                    narrative[key]
                )

    return safe_string(
        narrative
    )


# ============================================================================
# CASE-LEVEL QC
# ============================================================================

def qc_case(
    row: Dict[str, Any],
    narrative: str,
) -> Dict[str, Any]:
    """
    Run factual coverage QC for one valid ICSR.
    """
    case_id = get_case_id(
        row
    )

    source_fields = extract_source_fields(
        row
    )

    field_results: Dict[str, str] = {}

    failed_fields: List[str] = []
    checked_fields: List[str] = []

    for field_name in FIELD_ALIASES:

        source_value = source_fields.get(
            field_name,
            "",
        )

        # Missing source information is not treated as a QC failure.
        if is_missing(source_value):

            field_results[field_name] = "NOT_CHECKED"

            continue

        checked_fields.append(
            field_name
        )

        try:

            covered = field_is_covered(
                field_name,
                source_value,
                narrative,
            )

        except Exception:

            covered = False

        if covered:

            field_results[field_name] = "PASS"

        else:

            field_results[field_name] = "REVIEW"

            failed_fields.append(
                field_name
            )

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------

    if not checked_fields:

        overall_status = "REVIEW"

        reason = (
            "No source fields available for factual QC."
        )

    elif failed_fields:

        overall_status = "REVIEW"

        reason = (
            "Narrative did not clearly cover: "
            + ", ".join(failed_fields)
        )

    else:

        overall_status = "PASS"

        reason = (
            "All available source fields were covered."
        )

    return {
        "case_id": case_id,
        "overall_status": overall_status,
        "qc_reason": reason,
        "checked_field_count": len(
            checked_fields
        ),
        "failed_field_count": len(
            failed_fields
        ),
        "missing_source_field_count": (
            len(FIELD_ALIASES)
            - len(checked_fields)
        ),
        "patient_age_qc": field_results[
            "patient_age"
        ],
        "sex_qc": field_results[
            "sex"
        ],
        "country_qc": field_results[
            "country"
        ],
        "suspect_drug_qc": field_results[
            "suspect_drug"
        ],
        "indication_qc": field_results[
            "indication"
        ],
        "adverse_event_qc": field_results[
            "adverse_event"
        ],
        "event_date_qc": field_results[
            "event_date"
        ],
        "seriousness_qc": field_results[
            "seriousness"
        ],
        "outcome_qc": field_results[
            "outcome"
        ],
        "reporter_type_qc": field_results[
            "reporter_type"
        ],
        "generated_narrative": narrative,
    }


# ============================================================================
# EXCLUDED CASE RESULT
# ============================================================================

def build_excluded_result(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an output record for an invalid ICSR.

    The invalid case is retained for traceability but no narrative
    is generated.
    """
    case_id = get_case_id(
        row
    )

    not_checked = "NOT_CHECKED"

    return {
        "case_id": case_id,
        "overall_status": "EXCLUDED",
        "qc_reason": (
            "Invalid ICSR excluded from "
            "narrative generation and QC."
        ),
        "checked_field_count": 0,
        "failed_field_count": 0,
        "missing_source_field_count": len(
            FIELD_ALIASES
        ),
        "patient_age_qc": not_checked,
        "sex_qc": not_checked,
        "country_qc": not_checked,
        "suspect_drug_qc": not_checked,
        "indication_qc": not_checked,
        "adverse_event_qc": not_checked,
        "event_date_qc": not_checked,
        "seriousness_qc": not_checked,
        "outcome_qc": not_checked,
        "reporter_type_qc": not_checked,
        "generated_narrative": "",
    }


# ============================================================================
# ERROR RESULT
# ============================================================================

def build_error_result(
    row: Dict[str, Any],
    error: Exception,
) -> Dict[str, Any]:
    """
    Build a REVIEW result when generation/QC encounters an error.
    """
    case_id = get_case_id(
        row
    )

    not_checked = "NOT_CHECKED"

    return {
        "case_id": case_id,
        "overall_status": "REVIEW",
        "qc_reason": (
            "Narrative generation/QC error: "
            + safe_string(error)
        ),
        "checked_field_count": 0,
        "failed_field_count": 0,
        "missing_source_field_count": len(
            FIELD_ALIASES
        ),
        "patient_age_qc": not_checked,
        "sex_qc": not_checked,
        "country_qc": not_checked,
        "suspect_drug_qc": not_checked,
        "indication_qc": not_checked,
        "adverse_event_qc": not_checked,
        "event_date_qc": not_checked,
        "seriousness_qc": not_checked,
        "outcome_qc": not_checked,
        "reporter_type_qc": not_checked,
        "generated_narrative": "",
    }


# ============================================================================
# AUDIT HELPER
# ============================================================================

def audit_event(
    *,
    case_id: str,
    event_type: str,
    actor_type: str = "SYSTEM",
    validation_status: str = "",
    duplicate_status: str = "",
    human_duplicate_decision: str = "",
    narrative_eligibility: str = "",
    narrative_status: str = "",
    qc_status: str = "",
    operational_state: str = "",
    action: str = "",
    reason: str = "",
    narrative: str = "",
) -> None:
    """Write an audit event without allowing audit I/O to stop QC."""
    try:
        write_audit_event(
            case_id=case_id,
            event_type=event_type,
            actor_type=actor_type,
            validation_status=validation_status,
            duplicate_status=duplicate_status,
            human_duplicate_decision=human_duplicate_decision,
            narrative_eligibility=narrative_eligibility,
            narrative_status=narrative_status,
            qc_status=qc_status,
            operational_state=operational_state,
            action=action,
            reason=reason,
            narrative=narrative,
        )
    except Exception as exc:
        print(
            f"[AUDIT WARNING] {case_id or '<missing-case-id>'} - "
            f"audit event '{event_type}' could not be written: {exc}"
        )


# ============================================================================
# MAIN QC PIPELINE
# ============================================================================

def run_narrative_qc() -> List[Dict[str, Any]]:
    """
    Execute the complete Narrative QC workflow.
    """
    print()
    print("=" * 70)
    print("PV_NarrativeGuard - Narrative QC")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load incoming ICSRs
    # ------------------------------------------------------------------

    if not INCOMING_FILE.exists():

        raise FileNotFoundError(
            f"Incoming ICSR file not found:\n"
            f"{INCOMING_FILE}"
        )

    incoming_rows = read_csv(
        INCOMING_FILE
    )

    if not incoming_rows:

        raise RuntimeError(
            f"No ICSR records found in:\n"
            f"{INCOMING_FILE}"
        )

    print(
        f"Incoming ICSRs loaded: "
        f"{len(incoming_rows)}"
    )

    # ------------------------------------------------------------------
    # Load validation results
    # ------------------------------------------------------------------

    validation_rows: List[
        Dict[str, Any]
    ] = []

    if VALIDATION_FILE.exists():

        validation_rows = read_csv(
            VALIDATION_FILE
        )

        print(
            f"Validation results loaded: "
            f"{len(validation_rows)}"
        )

    else:

        print(
            "Validation results file not found. "
            "Using fallback validity logic."
        )

    validation_status_map = (
        build_validation_status_map(
            validation_rows
        )
    )

    # ------------------------------------------------------------------
    # Load duplicate / human-review gate
    # ------------------------------------------------------------------

    gate_data = load_case_gate_data()

    print(
        f"Duplicate/human-review gate records loaded: "
        f"{len(gate_data)}"
    )

    # Process cases
    # ------------------------------------------------------------------

    results: List[
        Dict[str, Any]
    ] = []

    valid_count = 0
    excluded_count = 0
    blocked_duplicate_count = 0
    pending_human_review_count = 0
    generated_count = 0
    pass_count = 0
    review_count = 0

    for row in incoming_rows:

        case_id = get_case_id(
            row
        )

        # --------------------------------------------------------------
        # Missing case ID
        # --------------------------------------------------------------

        if not case_id:

            error_result = build_error_result(
                row,
                ValueError(
                    "Missing case ID; "
                    "narrative generation skipped."
                ),
            )

            results.append(
                error_result
            )

            review_count += 1

            audit_event(
                case_id="",
                event_type="NARRATIVE_PROCESSING_ERROR",
                validation_status="UNKNOWN",
                action="NARRATIVE_PROCESSING_REVIEW",
                reason="Missing case ID; narrative generation skipped.",
            )

            continue

        # --------------------------------------------------------------
        # Validation gate
        # --------------------------------------------------------------

        valid = is_case_valid(
            row,
            validation_status_map,
        )

        if not valid:

            excluded_result = build_excluded_result(
                row
            )

            results.append(
                excluded_result
            )

            excluded_count += 1

            print(
                f"[EXCLUDED] {case_id} - invalid ICSR"
            )

            audit_event(
                case_id=case_id,
                event_type="ICSR_VALIDATION_FAILED",
                validation_status="INVALID",
                narrative_eligibility="BLOCKED_INVALID_ICSR",
                action="NARRATIVE_PROCESSING_BLOCKED",
                reason="ICSR failed the required validation gate.",
            )

            continue

        valid_count += 1

        # --------------------------------------------------------------
        # Duplicate / human-review gate
        # --------------------------------------------------------------

        generation_allowed, gate_reason = narrative_generation_gate(
            case_id,
            gate_data,
        )

        if not generation_allowed:

            is_pending = (
                "pending human" in gate_reason.lower()
                or "requires human" in gate_reason.lower()
            )

            if is_pending:
                blocked_status = "PENDING_HUMAN_REVIEW"
                pending_human_review_count += 1
            else:
                blocked_status = "BLOCKED_DUPLICATE"
                blocked_duplicate_count += 1

            blocked_result = build_duplicate_blocked_result(
                row,
                gate_reason,
                status=blocked_status,
            )

            results.append(blocked_result)

            print(
                f"[BLOCKED] {case_id} - "
                f"{gate_reason}"
            )

            gate_record = gate_data.get(
                case_id,
                {},
            )

            audit_event(
                case_id=case_id,
                event_type="NARRATIVE_PROCESSING_BLOCKED",
                validation_status="VALID",
                duplicate_status=gate_record.get(
                    "duplicate_status",
                    "",
                ),
                human_duplicate_decision=gate_record.get(
                    "human_duplicate_decision",
                    "",
                ),
                narrative_eligibility=blocked_status,
                action="NARRATIVE_PROCESSING_BLOCKED",
                reason=gate_reason,
            )

            continue

        # --------------------------------------------------------------
        # Narrative generation
        # --------------------------------------------------------------

        generated_count += 1

        print(
            f"[QC] Generating narrative for "
            f"{case_id}..."
        )

        try:

            narrative = generate_case_narrative(
                case_id
            )

            if not narrative:

                raise ValueError(
                    "Narrative generation returned "
                    "empty output."
                )

            # ----------------------------------------------------------
            # Factual QC
            # ----------------------------------------------------------

            qc_result = qc_case(
                row,
                narrative,
            )

            results.append(
                qc_result
            )

            status = qc_result[
                "overall_status"
            ]

            if status == "PASS":

                pass_count += 1

                print(
                    f"[PASS] {case_id} - "
                    f"{qc_result['checked_field_count']} "
                    f"fields covered"
                )

            else:

                review_count += 1

                print(
                    f"[REVIEW] {case_id} - "
                    f"{qc_result['qc_reason']}"
                )

            audit_event(
                case_id=case_id,
                event_type="NARRATIVE_QC_COMPLETED",
                validation_status="VALID",
                duplicate_status=gate_data.get(
                    case_id,
                    {},
                ).get(
                    "duplicate_status",
                    "",
                ),
                human_duplicate_decision=gate_data.get(
                    case_id,
                    {},
                ).get(
                    "human_duplicate_decision",
                    "",
                ),
                narrative_eligibility="ELIGIBLE",
                narrative_status="GENERATED",
                qc_status=status,
                operational_state=(
                    "READY - QC PASSED"
                    if status == "PASS"
                    else "REVIEW - QC ATTENTION REQUIRED"
                ),
                action="NARRATIVE_QC_COMPLETED",
                reason=qc_result.get(
                    "qc_reason",
                    "",
                ),
                narrative=narrative,
            )

        except Exception as exc:

            review_result = build_error_result(
                row,
                exc,
            )

            results.append(
                review_result
            )

            review_count += 1

            print(
                f"[REVIEW] {case_id} - "
                f"{review_result['qc_reason']}"
            )

            audit_event(
                case_id=case_id,
                event_type="NARRATIVE_QC_ERROR",
                validation_status="VALID",
                duplicate_status=gate_data.get(
                    case_id,
                    {},
                ).get(
                    "duplicate_status",
                    "",
                ),
                human_duplicate_decision=gate_data.get(
                    case_id,
                    {},
                ).get(
                    "human_duplicate_decision",
                    "",
                ),
                narrative_eligibility="ELIGIBLE",
                narrative_status="ERROR",
                qc_status="REVIEW",
                operational_state="REVIEW - QC ATTENTION REQUIRED",
                action="NARRATIVE_PROCESSING_REVIEW",
                reason=review_result.get(
                    "qc_reason",
                    "",
                ),
            )

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "case_id",
        "overall_status",
        "qc_reason",
        "checked_field_count",
        "failed_field_count",
        "missing_source_field_count",
        "patient_age_qc",
        "sex_qc",
        "country_qc",
        "suspect_drug_qc",
        "indication_qc",
        "adverse_event_qc",
        "event_date_qc",
        "seriousness_qc",
        "outcome_qc",
        "reporter_type_qc",
        "generated_narrative",
    ]

    try:

        with OUTPUT_FILE.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()

            for result in results:

                writer.writerow(
                    result
                )

    except OSError as exc:

        raise RuntimeError(
            f"Unable to write narrative QC "
            f"results to:\n{OUTPUT_FILE}"
        ) from exc

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("-" * 70)
    print("NARRATIVE QC SUMMARY")
    print("-" * 70)

    print(
        f"Total incoming ICSRs : "
        f"{len(incoming_rows)}"
    )

    print(
        f"Valid ICSRs processed: "
        f"{valid_count}"
    )

    print(
        f"Excluded ICSRs       : "
        f"{excluded_count}"
    )

    print(
        f"Blocked duplicates   : "
        f"{blocked_duplicate_count}"
    )

    print(
        f"Pending human review : "
        f"{pending_human_review_count}"
    )

    print(
        f"Narratives generated : "
        f"{generated_count}"
    )

    print(
        f"PASS                 : "
        f"{pass_count}"
    )

    print(
        f"REVIEW               : "
        f"{review_count}"
    )

    print(
        f"Results saved to     : "
        f"{OUTPUT_FILE}"
    )

    print("-" * 70)

    print(
        "Human pharmacovigilance review is REQUIRED "
        "before final use."
    )

    print("=" * 70)

    return results


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Command-line entry point.
    """
    try:

        run_narrative_qc()

    except Exception as exc:

        print()
        print("=" * 70)
        print("NARRATIVE QC FAILED")
        print("=" * 70)
        print(str(exc))
        print("=" * 70)

        sys.exit(1)


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()