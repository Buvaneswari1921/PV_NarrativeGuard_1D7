"""Controlled response builder for the PV NarrativeGuard RAB Bot.

This module is presentation-only.

It does NOT:
    - perform retrieval
    - perform MedDRA matching
    - change routing decisions
    - override terminology decisions
    - generate unsupported medical conclusions

It converts an already-controlled RAB decision into a
structured, human-readable response.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BuiltResponse:
    """Final controlled response representation."""

    message: str
    evidence: List[Dict[str, Any]]
    confidence: float
    response_mode: str
    route: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float without breaking response generation."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_confidence(confidence: Any) -> str:
    """Format confidence as a percentage."""
    value = max(0.0, min(1.0, _safe_float(confidence)))
    return f"{value * 100:.1f}%"


def _clean_label(value: Any) -> str:
    """Clean terminology labels for display."""
    if value is None:
        return ""

    label = str(value).strip()

    if label.endswith(" AE"):
        label = label[:-3].strip()

    return label


def _format_meddra_ids(value: Any) -> str:
    """Format pipe/comma-delimited MedDRA IDs without presenting them as one ID."""
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return ""
    parts = [p.strip() for p in raw.replace(",", "|").split("|") if p.strip()]
    return ", ".join(dict.fromkeys(parts))


def _id_count(value: Any) -> int:
    formatted = _format_meddra_ids(value)
    return len([x for x in formatted.split(", ") if x]) if formatted else 0


def build_meddra_response(
    decision: Dict[str, Any],
) -> BuiltResponse:
    """Build a response from an existing MedDRA decision."""

    status = decision.get("status", "")
    result = decision.get("result") or {}
    candidates = decision.get("candidates") or []

    decision_score = _safe_float(
        result.get("decision_score", 0.0)
    )

    # --------------------------------------------------------
    # Supported terminology
    # --------------------------------------------------------

    if status == "SUPPORTED" and result:

        label = _clean_label(
            result.get("preferred_label")
        )

        meddra_id = _format_meddra_ids(
            result.get("meddra_id", "")
        )

        match_method = result.get(
            "match_method",
            "",
        )

        message = (
            "MedDRA terminology result\n\n"
            f"Preferred Term: {label}\n"
            f"MedDRA ID(s): {meddra_id}\n"
            f"Confidence: {_format_confidence(decision_score)}\n"
        )

        if _id_count(result.get("meddra_id", "")) > 1:
            message += (
                "Source mapping note: multiple MedDRA IDs are associated "
                "with this retrieved terminology record. Verify the applicable "
                "licensed MedDRA concept before final coding.\n"
            )

        if match_method:
            message += (
                f"Match Method: {match_method}\n"
            )

        matched_expression = result.get("matched_pv_expression", "")
        if matched_expression:
            message += (
                f"Matched PV Expression: {matched_expression}\n"
            )

        pv_category = result.get("pv_category", "")
        if pv_category:
            message += (
                f"PV Category: {pv_category}\n"
            )

        coding_note = result.get("coding_note", "")
        if coding_note:
            message += (
                f"\nCoding Note: {coding_note}\n"
            )

        message += (
            "\nDecision: "
            f"{decision.get('message', 'Terminology match found.')}"
        )

        return BuiltResponse(
            message=message,
            evidence=[result],
            confidence=decision_score,
            response_mode="MEDDRA_ANSWER",
            route="MEDDRA",
        )

    # --------------------------------------------------------
    # Ambiguous terminology
    # --------------------------------------------------------

    if status == "AMBIGUOUS":

        evidence = candidates[:5]

        lines = [
            "Clarification required: terminology is ambiguous.",
            "",
            decision.get(
                "message",
                "Additional clinical context is required before selecting a terminology concept.",
            ),
        ]

        if evidence:
            lines.extend([
                "",
                "Candidate concepts:",
            ])

            for index, candidate in enumerate(
                evidence,
                start=1,
            ):
                label = _clean_label(
                    candidate.get("preferred_label")
                )

                meddra_id = _format_meddra_ids(
                    candidate.get("meddra_id", "")
                )

                lines.append(
                    f"{index}. {label} "
                    f"(MedDRA ID(s): {meddra_id})"
                )

        return BuiltResponse(
            message="\n".join(lines),
            evidence=evidence,
            confidence=decision_score,
            response_mode="CLARIFICATION_REQUIRED",
            route="MEDDRA",
        )

    # --------------------------------------------------------
    # Unsupported / not found
    # --------------------------------------------------------

    return BuiltResponse(
        message=(
            "No sufficiently supported MedDRA terminology "
            "was found in the current knowledge dataset.\n\n"
            "This is a controlled knowledge gap and requires "
            "human review before any terminology is added."
        ),
        evidence=candidates[:5],
        confidence=decision_score,
        response_mode="ASK_ADD_TO_DATASET",
        route="MEDDRA",
    )


def build_rag_response(
    decision,
    route: str,
) -> BuiltResponse:
    """Build a response from a retrieval-guard decision."""

    evidence = decision.evidence or []

    if decision.status == "SUFFICIENT":

        message = (
            f"Relevant knowledge was found in the "
            f"{route} knowledge route.\n\n"
            f"Confidence: "
            f"{_format_confidence(decision.confidence)}\n\n"
            "The retrieved evidence can be used as the "
            "grounding context for the final answer."
        )

        return BuiltResponse(
            message=message,
            evidence=evidence,
            confidence=decision.confidence,
            response_mode="RAG_ANSWER",
            route=route,
        )

    return BuiltResponse(
        message=(
            "This topic is relevant to PV NarrativeGuard, "
            "but there is not currently sufficient evidence "
            "in the knowledge dataset.\n\n"
            "This has been identified as a controlled "
            "knowledge gap requiring human review."
        ),
        evidence=evidence,
        confidence=decision.confidence,
        response_mode="ASK_ADD_TO_DATASET",
        route=route,
    )


def build_out_of_context_response(
    confidence: float,
    reason: str = "",
) -> BuiltResponse:
    """Build the controlled out-of-context response."""

    message = (
        "This query is outside my current PV NarrativeGuard "
        "context. I can assist with pharmacovigilance, ICSR "
        "processing, MedDRA terminology, narrative "
        "generation/QC, regulatory concepts, and the "
        "project's AI/RAG architecture."
    )

    return BuiltResponse(
        message=message,
        evidence=[],
        confidence=confidence,
        response_mode="OUT_OF_CONTEXT",
        route="OUT_OF_CONTEXT",
    )


def build_clarification_response(
    confidence: float,
    reason: str = "",
) -> BuiltResponse:
    """Build a controlled routing clarification response."""

    return BuiltResponse(
        message=(
            "I can identify this as related to the "
            "PV NarrativeGuard domain, but I cannot "
            "determine the correct knowledge source "
            "with enough confidence.\n\n"
            "Please clarify whether you mean MedDRA "
            "terminology, general pharmacovigilance, "
            "or PV NarrativeGuard/project architecture."
        ),
        evidence=[],
        confidence=confidence,
        response_mode="CLARIFICATION_REQUIRED",
        route="UNKNOWN",
    )