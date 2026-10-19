"""
Scope classification for the PV NarrativeGuard RAB bot.

Purpose
-------
Determine whether a user query is relevant to PV NarrativeGuard before
retrieval begins.

The classifier recognizes four broad signals:

    1. Explicit PV / pharmacovigilance terminology
    2. PV NarrativeGuard project terminology
    3. Clinical-event / adverse-event phrasing
    4. Standalone clinical-event / symptom terminology

Important design principle
--------------------------
The classifier only determines whether a query is relevant.

It does NOT decide the final MedDRA concept.

Clinical terminology is subsequently evaluated by the MedDRA intent route
and terminology engine.

This separation prevents the scope classifier from incorrectly rejecting
short clinical expressions such as:

    itching
    rash
    feels like vomiting
    loss of consciousness
    passed out
    difficulty breathing
    lightheaded

while still keeping unrelated questions outside the system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from meddra_rag.terminology_engine import (
    get_pv_core_alias_match,
    load_pv_core_vocabulary,
)


# ============================================================
# EXPLICIT PV TERMS
# ============================================================

PV_TERMS = {
    "pharmacovigilance",
    "pv",
    "drug safety",
    "safety",
    "adverse event",
    "adverse drug reaction",
    "adr",
    "ae",
    "sae",
    "susar",
    "icsr",
    "individual case safety report",
    "meddra",
    "medical dictionary for regulatory activities",
    "causality",
    "naranjo",
    "who-umc",
    "signal detection",
    "risk management",
    "aggregate reporting",
    "narrative",
    "narrative qc",
    "narrative quality",
    "case processing",
    "duplicate",
    "duplicate screening",
    "duplicate investigation",
    "gvp",
    "ich",
    "cdsco",
    "pvpi",
    "regulatory",
    "safety database",
    "safety reporting",
    "seriousness",
    "expectedness",
    "dechallenge",
    "rechallenge",
    "ai in pharmacovigilance",
    "nlp",
    "rag",
    "faiss",
    "sentence transformer",
    "terminology",
    "drug safety ai",
}


# ============================================================
# PROJECT TERMS
# ============================================================

PROJECT_TERMS = {
    "narrativeguard",
    "rab bot",
    "rag engine",
    "knowledge dataset",
    "knowledge base",
    "vector store",
    "vectorstore",
    "faiss index",
    "terminology engine",
    "narrative generation",
}


# ============================================================
# CLINICAL EVENT PATTERNS
# ============================================================

CLINICAL_EVENT_PATTERNS = (
    r"\bpatient\s+(?:has|had|developed|experienced|reports?|reported|"
    r"complains?\s+of|complained\s+of|presented\s+with|"
    r"feels?|felt|is\s+experiencing|was\s+experiencing)\b",

    r"\bsubject\s+(?:has|had|developed|experienced|reports?|reported|"
    r"complains?\s+of|complained\s+of|presented\s+with|"
    r"feels?|felt|is\s+experiencing|was\s+experiencing)\b",

    r"\bpatient's\s+\w+",

    r"\bsubject's\s+\w+",
)


# ============================================================
# STANDALONE CLINICAL / AE SIGNALS
# ============================================================
#
# These are NOT intended to be a MedDRA vocabulary.
#
# They are routing indicators only.
#
# The terminology engine remains responsible for determining whether
# the expression maps to a supported MedDRA concept.
#
# Keep this list conservative and focused on expressions that strongly
# resemble clinical-event terminology.
# ============================================================

CLINICAL_EVENT_TERMS = {
    # Symptoms / events
    "itching",
    "itch",
    "pruritus",
    "rash",
    "hives",
    "urticaria",

    "nausea",
    "vomiting",
    "feels like vomiting",
    "feeling like vomiting",

    "diarrhea",
    "diarrhoea",
    "loose stools",

    "headache",
    "dizziness",
    "dizzy",
    "lightheaded",
    "lightheadedness",

    "loss of consciousness",
    "lost consciousness",
    "passed out",
    "fainted",
    "fainting",
    "syncope",

    "difficulty breathing",
    "breathing difficulty",
    "shortness of breath",
    "dyspnea",
    "dyspnoea",

    "swelling",
    "edema",
    "oedema",

    "abdominal pain",
    "stomach pain",
    "chest pain",
    "back pain",

    "fever",
    "fatigue",
    "weakness",
    "drowsiness",
    "somnolence",

    "constipation",
    "insomnia",

    "cough",
    "sore throat",

    "palpitations",

    "bleeding",
    "bruising",

    "seizure",
    "convulsion",

    "anxiety",
    "depression",

    "pain",
}


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass
class ScopeResult:
    scope: str
    confidence: float
    matched_terms: List[str]
    reason: str


# ============================================================
# NORMALIZATION
# ============================================================

def _normalise(text: str) -> str:
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# MATCHING HELPERS
# ============================================================

# ============================================================
# PV CORE VOCABULARY ROUTING
# ============================================================

@lru_cache(maxsize=1)
def _get_pv_core_vocabulary():
    """Load the curated PV core vocabulary once for routing.

    The classifier uses this vocabulary only as a relevance signal.
    It never selects the final MedDRA concept.
    """
    return load_pv_core_vocabulary()


def _get_pv_core_routing_match(query: str):
    """Return a curated PV expression match for scope classification."""
    try:
        return get_pv_core_alias_match(
            query,
            _get_pv_core_vocabulary(),
        )
    except Exception:
        # Scope classification must remain available even if the
        # optional curated vocabulary cannot be loaded.
        return None



def _contains_term(q: str, term: str) -> bool:
    """
    Match a vocabulary term safely.

    Multi-word expressions use substring matching.
    Single words use word boundaries to avoid accidental matches.
    """

    if " " in term:
        return term in q

    return bool(
        re.search(
            rf"\b{re.escape(term)}\b",
            q,
        )
    )


def _find_matches(q: str, vocabulary) -> List[str]:
    matches = []

    for term in sorted(
        vocabulary,
        key=len,
        reverse=True,
    ):
        if _contains_term(q, term):
            matches.append(term)

    return matches


def _is_clinical_event_query(q: str) -> bool:
    """Return True when the query resembles a clinical-event narrative."""

    return any(
        re.search(pattern, q)
        for pattern in CLINICAL_EVENT_PATTERNS
    )


def _is_standalone_clinical_query(q: str) -> bool:
    """
    Return True when the query contains a recognized clinical-event
    expression.

    This deliberately does NOT mean that the expression is guaranteed
    to exist in the MedDRA dataset.

    It only means the query should be allowed to reach the terminology
    routing/decision layer.
    """

    return bool(
        _find_matches(
            q,
            CLINICAL_EVENT_TERMS,
        )
    )


# ============================================================
# MAIN CLASSIFIER
# ============================================================

def classify_query(query: str) -> ScopeResult:
    """
    Classify a query as RELEVANT or OUT_OF_SCOPE.

    Relevance is established through:

        1. Explicit PV terminology
        2. Project terminology
        3. Clinical-event narrative phrasing
        4. Standalone clinical-event terminology

    The classifier does not determine the final MedDRA concept.
    """

    q = _normalise(query)

    # --------------------------------------------------------
    # Empty query
    # --------------------------------------------------------

    if not q:

        return ScopeResult(
            scope="OUT_OF_SCOPE",
            confidence=1.0,
            matched_terms=[],
            reason="Empty query.",
        )

    # --------------------------------------------------------
    # Explicit PV / project terminology
    # --------------------------------------------------------

    matched = _find_matches(
        q,
        PV_TERMS | PROJECT_TERMS,
    )

    if matched:

        unique = list(
            dict.fromkeys(matched)
        )

        confidence = min(
            0.99,
            0.72 + 0.08 * min(
                len(unique),
                3,
            ),
        )

        return ScopeResult(
            scope="RELEVANT",
            confidence=confidence,
            matched_terms=unique[:10],
            reason=(
                "PV NarrativeGuard/domain terminology detected."
            ),
        )

    # --------------------------------------------------------
    # Curated PV Core terminology
    # --------------------------------------------------------
    #
    # Natural PV expressions such as:
    #   patient threw up twice
    #   itchy skin
    #   fast heartbeat
    #   high blood pressure
    # may not match the generic clinical-event verb patterns above.
    # The curated 150-concept layer therefore acts as an explicit
    # routing signal before the query can be rejected as out of scope.
    #
    # IMPORTANT: this is only a scope signal. Final terminology
    # selection remains the responsibility of the MedDRA route.

    pv_core_match = _get_pv_core_routing_match(q)

    if pv_core_match:

        return ScopeResult(
            scope="RELEVANT",
            confidence=0.91,
            matched_terms=[
                pv_core_match.get("alias", ""),
                pv_core_match.get("preferred_label", ""),
            ],
            reason=(
                "Curated PV Core terminology expression detected; "
                "query will be evaluated through the MedDRA "
                "terminology route."
            ),
        )

    # --------------------------------------------------------
    # Clinical-event narrative
    # --------------------------------------------------------

    if _is_clinical_event_query(q):

        return ScopeResult(
            scope="RELEVANT",
            confidence=0.88,
            matched_terms=[
                "clinical_event_narrative"
            ],
            reason=(
                "Clinical-event narrative detected; "
                "query will be evaluated through the "
                "MedDRA terminology route."
            ),
        )

    # --------------------------------------------------------
    # Standalone clinical-event terminology
    # --------------------------------------------------------

    clinical_matches = _find_matches(
        q,
        CLINICAL_EVENT_TERMS,
    )

    if clinical_matches:

        return ScopeResult(
            scope="RELEVANT",
            confidence=0.86,
            matched_terms=clinical_matches[:10],
            reason=(
                "Clinical-event terminology detected; "
                "query will be evaluated through the "
                "MedDRA terminology route."
            ),
        )

    # --------------------------------------------------------
    # No supported relevance signal
    # --------------------------------------------------------

    return ScopeResult(
        scope="OUT_OF_SCOPE",
        confidence=0.95,
        matched_terms=[],
        reason=(
            "No PV NarrativeGuard, pharmacovigilance, "
            "or clinical-event indicator found."
        ),
    )