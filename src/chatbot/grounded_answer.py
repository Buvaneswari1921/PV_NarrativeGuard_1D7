"""Grounded answer generation for PV NarrativeGuard.

This module converts an already-controlled RAG decision into a
natural-language answer using ONLY the supplied evidence.

Safety principles:
    - No retrieval is performed here.
    - No new knowledge is introduced.
    - Low-confidence evidence is rejected.
    - Empty or malformed evidence is rejected.
    - MedDRA decisions are preserved rather than regenerated.
    - Knowledge gaps are never converted into speculative answers.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


GROUNDING_MIN_CONFIDENCE = 0.65


@dataclass
class GroundedAnswer:
    """Controlled final answer produced from approved evidence."""

    answer: str
    evidence: List[Dict[str, Any]]
    confidence: float
    grounded: bool
    reason: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_text(item: Dict[str, Any]) -> str:
    """Extract usable evidence text."""
    if not isinstance(item, dict):
        return ""

    text = item.get("text")

    if text is None:
        return ""

    return str(text).strip()


def _normalise_evidence(
    evidence: Any,
) -> List[Dict[str, Any]]:
    """Return only valid evidence records containing text."""
    if not evidence:
        return []

    valid = []

    for item in evidence:

        if not isinstance(item, dict):
            continue

        text = _extract_text(item)

        if not text:
            continue

        valid.append(item)

    return valid


def _build_evidence_section(
    evidence: List[Dict[str, Any]],
) -> str:
    """Build a compact evidence attribution section."""

    lines = [
        "",
        "Evidence:",
    ]

    for index, item in enumerate(
        evidence,
        start=1,
    ):

        source = str(
            item.get(
                "source",
                "Unknown source",
            )
        )

        chunk_id = str(
            item.get(
                "chunk_id",
                "",
            )
        )

        text = _extract_text(item)

        reference = source

        if chunk_id:
            reference = (
                f"{source} "
                f"({chunk_id})"
            )

        lines.append(
            f"{index}. {reference}: {text}"
        )

    return "\n".join(lines)


def generate_grounded_rag_answer(
    query: str,
    evidence: Any,
    confidence: Any,
    min_confidence: float = GROUNDING_MIN_CONFIDENCE,
) -> GroundedAnswer:
    """Generate a controlled answer using only supplied evidence.

    The function intentionally performs no semantic retrieval and makes
    no attempt to answer from model knowledge.

    For this initial implementation, the answer is constructed directly
    from retrieved evidence. A later LLM-based generation layer can be
    introduced behind this same contract, provided the evidence-only
    constraint is preserved.
    """

    confidence_value = max(
        0.0,
        min(
            1.0,
            _safe_float(confidence),
        ),
    )

    valid_evidence = _normalise_evidence(
        evidence
    )

    # --------------------------------------------------------
    # Evidence validation
    # --------------------------------------------------------

    if not valid_evidence:

        return GroundedAnswer(
            answer=(
                "I cannot provide a grounded answer because "
                "no usable supporting evidence was retrieved."
            ),
            evidence=[],
            confidence=confidence_value,
            grounded=False,
            reason="EMPTY_OR_INVALID_EVIDENCE",
        )

    # --------------------------------------------------------
    # Confidence gate
    # --------------------------------------------------------

    if confidence_value < min_confidence:

        return GroundedAnswer(
            answer=(
                "I cannot provide a grounded answer because "
                "the retrieved evidence does not meet the "
                "minimum confidence threshold."
            ),
            evidence=valid_evidence,
            confidence=confidence_value,
            grounded=False,
            reason="LOW_CONFIDENCE_EVIDENCE",
        )

    # --------------------------------------------------------
    # Grounded answer
    #
    # IMPORTANT:
    # We deliberately do not invent a synthesis here.
    # The evidence itself becomes the grounding content.
    # --------------------------------------------------------

    answer_parts = []

    for item in valid_evidence:

        text = _extract_text(item)

        if text:
            answer_parts.append(
                text
            )

    answer = "\n\n".join(
        answer_parts
    )

    answer += _build_evidence_section(
        valid_evidence
    )

    return GroundedAnswer(
        answer=answer,
        evidence=valid_evidence,
        confidence=confidence_value,
        grounded=True,
        reason="GROUNDED_BY_RETRIEVED_EVIDENCE",
    )


def preserve_meddra_decision(
    decision: Dict[str, Any],
) -> GroundedAnswer:
    """Preserve an existing controlled MedDRA decision.

    This function does not reinterpret terminology. It simply exposes
    an already-approved MedDRA result through the grounded-answer
    contract.
    """

    status = decision.get(
        "status",
        "",
    )

    result = decision.get(
        "result"
    )

    if status != "SUPPORTED" or not result:

        return GroundedAnswer(
            answer=(
                "A grounded MedDRA answer cannot be generated "
                "because the terminology decision is not supported."
            ),
            evidence=decision.get(
                "candidates",
                [],
            )[:5],
            confidence=0.0,
            grounded=False,
            reason="MEDDRA_DECISION_NOT_SUPPORTED",
        )

    label = str(
        result.get(
            "preferred_label",
            "",
        )
    ).replace(
        " AE",
        ""
    ).strip()

    raw_meddra_id = str(
        result.get(
            "meddra_id",
            "",
        )
    ).strip()

    # Some upstream OAE->MedDRA records can contain multiple retrieved
    # IDs for one label. Never render the pipe-delimited value as though
    # it were one MedDRA ID. Preserve the evidence, but present the IDs
    # explicitly as a set requiring verification.
    meddra_ids = [
        item.strip()
        for item in raw_meddra_id.split("|")
        if item.strip()
    ]

    confidence = _safe_float(
        result.get(
            "decision_score",
            0.0,
        )
    )

    if len(meddra_ids) <= 1:
        displayed_id = meddra_ids[0] if meddra_ids else "Not available"
        answer = (
            f"MedDRA terminology result: {label} "
            f"(MedDRA ID: {displayed_id})."
        )
    else:
        displayed_ids = ", ".join(meddra_ids)
        answer = (
            f"MedDRA terminology result: {label}.\n"
            f"Retrieved MedDRA ID(s): {displayed_ids}.\n"
            "Multiple IDs are associated with this retrieved label; "
            "verify the applicable MedDRA concept before final coding."
        )

    return GroundedAnswer(
        answer=answer,
        evidence=[result],
        confidence=confidence,
        grounded=True,
        reason="SUPPORTED_MEDDRA_DECISION_PRESERVED",
    )