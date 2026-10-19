"""
PV_NarrativeGuard - RAB Intent Router

Purpose
-------
Route an already in-scope query to the appropriate knowledge source.

Routes
------
    MEDDRA       -> MedDRA FAISS + terminology decision engine
    GENERAL_PV   -> General PV RAG
    PROJECT      -> PV NarrativeGuard project knowledge
    UNKNOWN      -> routing confidence is insufficient

Routing principles
------------------
1. Explicit MedDRA terminology is routed to MEDDRA.
2. Clinical-event narratives are routed to MEDDRA.
3. Standalone clinical concepts are treated as terminology
   lookup candidates and routed to MEDDRA.
4. Explicit PV questions are routed to GENERAL_PV.
5. NarrativeGuard / RAG / architecture questions are routed
   to PROJECT.
6. Ambiguous or unsupported queries remain UNKNOWN.

Important
---------
The router does NOT decide whether a clinical concept actually
exists in the MedDRA dataset.

The router only decides which knowledge source should evaluate
the query.

The terminology engine remains responsible for:
    - retrieval
    - similarity scoring
    - candidate selection
    - terminology support
    - final terminology decision
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from meddra_rag.terminology_engine import (
    get_pv_core_alias_match,
    load_pv_core_vocabulary,
)


# ============================================================================
# ROUTE CONSTANTS
# ============================================================================

MEDDRA = "MEDDRA"
GENERAL_PV = "GENERAL_PV"
PROJECT = "PROJECT"
UNKNOWN = "UNKNOWN"


# ============================================================================
# MEDDRA / TERMINOLOGY VOCABULARY
# ============================================================================

MEDDRA_TERMS = {
    # ------------------------------------------------------------------------
    # Explicit MedDRA terminology
    # ------------------------------------------------------------------------
    "meddra",
    "medical dictionary for regulatory activities",

    "preferred term",
    "preferred terms",
    "pt",

    "lowest level term",
    "llt",

    "high level term",
    "hlt",

    "high level group term",
    "hlgt",

    "system organ class",
    "soc",

    # ------------------------------------------------------------------------
    # Coding / terminology language
    # ------------------------------------------------------------------------
    "meddra coding",
    "meddra code",
    "medical coding",
    "terminology coding",
    "preferred label",
    "synonym",

    # ------------------------------------------------------------------------
    # Explicit terminology lookup language
    # ------------------------------------------------------------------------
    "term for",
    "meddra term for",
    "medical term for",
    "preferred term for",
    "coded term for",
    "coding term for",
    "how to code",
    "code this",
    "coding this",
}


# ============================================================================
# GENERAL PHARMACOVIGILANCE VOCABULARY
# ============================================================================

GENERAL_PV_TERMS = {
    "pharmacovigilance",
    "drug safety",

    "icsr",
    "individual case safety report",

    "adverse event",
    "adverse drug reaction",
    "adr",
    "ae",
    "serious adverse event",
    "sae",
    "susar",

    "case processing",
    "case intake",
    "case assessment",
    "case follow up",
    "case follow-up",

    "follow-up",
    "seriousness",
    "expectedness",
    "causality",

    "naranjo",
    "who-umc",

    "signal detection",
    "risk management",
    "aggregate reporting",
    "safety reporting",
    "regulatory reporting",

    "gpp",
    "gvp",
    "ich",
    "cdsco",
    "pvpi",

    "duplicate case",
    "duplicate detection",
}


# ============================================================================
# PROJECT VOCABULARY
# ============================================================================

PROJECT_TERMS = {
    "narrativeguard",
    "pv narrativeguard",

    "rab bot",
    "rag engine",
    "rag",

    "faiss",
    "sentence transformer",
    "sentence-transformers",

    "vector store",
    "vectorstore",

    "knowledge dataset",
    "knowledge base",
    "knowledge source",

    "narrative generation",
    "narrative qc",
    "narrative quality control",
    "narrative validation",

    "terminology engine",
    "retrieval guard",
    "query classifier",
    "intent router",
    "pipeline",

    "streamlit",
}


# ============================================================================
# CLINICAL EVENT PATTERNS
# ============================================================================
#
# These patterns identify the SHAPE of a clinical statement.
#
# Examples:
#
#   patient has loose stools
#   patient developed rash
#   patient experienced dizziness
#   patient reports nausea
#   patient complained of headache
#   patient feels like vomiting
#   patient feels lightheaded
#   patient is experiencing itching
#
# The terminology engine remains responsible for deciding whether
# the clinical event maps to a supported MedDRA concept.
#

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


# ============================================================================
# TERMINOLOGY QUERY PATTERNS
# ============================================================================

TERMINOLOGY_PHRASES = (
    "term for",
    "meddra term",
    "preferred term",
    "preferred label",
    "medical term",
    "coding term",
    "meddra code",
    "how to code",
    "code this",
    "coding this",
)


# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass
class IntentResult:
    intent: str
    confidence: float
    matched_terms: list
    reason: str


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_query(query):
    """
    Normalize a user query before routing.

    Operations:
        - convert to string
        - lowercase
        - trim whitespace
        - collapse repeated whitespace
    """

    if query is None:
        return ""

    query = str(query).lower().strip()
    query = re.sub(r"\s+", " ", query)

    return query


# ============================================================================
# VOCABULARY MATCHING
# ============================================================================

def find_matches(query, vocabulary):
    """
    Find vocabulary terms present in the normalized query.

    Longer terms are checked first so that more specific phrases
    are retained ahead of generic terms.
    """

    matches = []

    for term in sorted(
        vocabulary,
        key=len,
        reverse=True,
    ):
        if term in query:
            matches.append(term)

    return matches


# ============================================================================
# CLINICAL EVENT DETECTION
# ============================================================================

def is_clinical_event_query(query):
    """
    Detect whether a query resembles a clinical-event narrative.
    """

    return any(
        re.search(pattern, query)
        for pattern in CLINICAL_EVENT_PATTERNS
    )


# ============================================================================
# TERMINOLOGY QUERY DETECTION
# ============================================================================

def is_terminology_query(query):
    """
    Detect terminology lookup requests that may not explicitly
    mention MedDRA.

    Examples
    --------
    itching
    nausea
    loss of consciousness
    passed out
    difficulty breathing

    Also supports explicit terminology requests such as:

    What is the term for itching?
    What is the preferred term for nausea?
    How do I code loss of consciousness?

    Important
    ---------
    This function does NOT determine whether the term exists
    in the MedDRA dataset.

    It only determines whether the query should be routed to
    the terminology engine.
    """

    query = normalize_query(query)

    if not query:
        return False

    # ------------------------------------------------------------------------
    # Explicit terminology language
    # ------------------------------------------------------------------------

    if any(
        phrase in query
        for phrase in TERMINOLOGY_PHRASES
    ):
        return True

    # ------------------------------------------------------------------------
    # Standalone clinical concept
    # ------------------------------------------------------------------------
    #
    # Short phrases such as:
    #
    #     itching
    #     loss of consciousness
    #     passed out
    #     difficulty breathing
    #
    # are likely terminology lookup candidates.
    #
    # We deliberately keep this conservative.
    #

    words = query.split()

    if not (1 <= len(words) <= 6):
        return False

    question_words = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "can",
        "could",
        "should",
        "does",
        "do",
        "is",
        "are",
    }

    if any(
        word in question_words
        for word in words
    ):
        return False

    # ------------------------------------------------------------------------
    # Reject obvious non-clinical conversational fragments
    # ------------------------------------------------------------------------

    non_clinical_phrases = {
        "thank you",
        "thanks",
        "hello",
        "hi",
        "okay",
        "ok",
        "good morning",
        "good evening",
        "good afternoon",
    }

    if query in non_clinical_phrases:
        return False

    return True


# ============================================================================
# CURATED PV CORE ROUTING
# ============================================================================

@lru_cache(maxsize=1)
def _get_pv_core_vocabulary():
    """Load the curated PV Core vocabulary once for intent routing."""
    return load_pv_core_vocabulary()


def get_pv_core_route_match(query):
    """Return the longest curated PV expression for routing only."""
    try:
        return get_pv_core_alias_match(
            query,
            _get_pv_core_vocabulary(),
        )
    except Exception:
        return None


# ============================================================================
# MAIN ROUTER
# ============================================================================

def route_query(query):
    """
    Route an already in-scope query.

    Routing order
    -------------
    1. Empty query
    2. Clinical-event narrative
    3. Standalone terminology lookup
    4. Vocabulary-based scoring
    5. Ambiguity handling
    6. Final route
    """

    query = normalize_query(query)

    # =========================================================================
    # EMPTY QUERY
    # =========================================================================

    if not query:
        return IntentResult(
            intent=UNKNOWN,
            confidence=1.0,
            matched_terms=[],
            reason="Empty query.",
        )

    # =========================================================================
    # CURATED PV CORE TERMINOLOGY
    # =========================================================================
    #
    # This explicit route makes the architecture resilient to natural
    # PV wording that is not covered by the generic clinical-event
    # patterns. The match is a routing signal only; MedDRA retrieval
    # and terminology validation still determine the final result.

    pv_core_match = get_pv_core_route_match(query)

    if pv_core_match:
        return IntentResult(
            intent=MEDDRA,
            confidence=0.91,
            matched_terms=[
                pv_core_match.get("alias", ""),
                pv_core_match.get("preferred_label", ""),
            ],
            reason=(
                "Curated PV Core terminology expression detected; "
                "routed to the MedDRA terminology engine."
            ),
        )

    # =========================================================================
    # CLINICAL EVENT NARRATIVE
    # =========================================================================
    #
    # Examples:
    #
    #     patient has itching
    #     patient developed rash
    #     patient feels like vomiting
    #     patient lost consciousness
    #
    # These should go directly to MedDRA.
    #

    if is_clinical_event_query(query):
        return IntentResult(
            intent=MEDDRA,
            confidence=0.88,
            matched_terms=["clinical_event_narrative"],
            reason=(
                "Clinical-event narrative detected; "
                "routed to the MedDRA terminology engine."
            ),
        )

    # =========================================================================
    # STANDALONE TERMINOLOGY / CLINICAL CONCEPT
    # =========================================================================
    #
    # Examples:
    #
    #     itching
    #     loss of consciousness
    #     passed out
    #     difficulty breathing
    #     feels like vomiting
    #
    # These queries do not necessarily contain the word "MedDRA".
    #

    if is_terminology_query(query):
        return IntentResult(
            intent=MEDDRA,
            confidence=0.82,
            matched_terms=["terminology_lookup"],
            reason=(
                "Clinical terminology lookup pattern detected; "
                "routed to the MedDRA terminology engine."
            ),
        )

    # =========================================================================
    # VOCABULARY MATCHING
    # =========================================================================

    meddra_matches = find_matches(
        query,
        MEDDRA_TERMS,
    )

    pv_matches = find_matches(
        query,
        GENERAL_PV_TERMS,
    )

    project_matches = find_matches(
        query,
        PROJECT_TERMS,
    )

    # =========================================================================
    # INITIAL SCORES
    # =========================================================================

    scores = {
        MEDDRA: 0.0,
        GENERAL_PV: 0.0,
        PROJECT: 0.0,
    }

    # Base contribution from vocabulary matches

    scores[MEDDRA] += 0.40 * len(meddra_matches)

    scores[GENERAL_PV] += 0.30 * len(pv_matches)

    scores[PROJECT] += 0.30 * len(project_matches)

    # =========================================================================
    # STRONG MEDDRA SIGNALS
    # =========================================================================

    if (
        "meddra" in query
        or "medical dictionary for regulatory activities" in query
        or "preferred term" in query
        or "preferred terms" in query
        or "meddra coding" in query
        or "meddra code" in query
    ):
        scores[MEDDRA] += 0.70

    # =========================================================================
    # STRONG PROJECT SIGNALS
    # =========================================================================

    if (
        "narrativeguard" in query
        or "pv narrativeguard" in query
        or "rab bot" in query
        or "terminology engine" in query
        or "retrieval guard" in query
        or "intent router" in query
    ):
        scores[PROJECT] += 0.70

    # =========================================================================
    # STRONG GENERAL PV SIGNALS
    # =========================================================================

    if (
        "icsr" in query
        or "individual case safety report" in query
        or "adverse event" in query
        or "adverse drug reaction" in query
        or "susar" in query
    ):
        scores[GENERAL_PV] += 0.60

    # =========================================================================
    # NO SIGNAL
    # =========================================================================

    if max(scores.values()) == 0:
        return IntentResult(
            intent=UNKNOWN,
            confidence=0.0,
            matched_terms=[],
            reason="No supported intent indicators found.",
        )

    # =========================================================================
    # BEST ROUTE
    # =========================================================================

    best_intent = max(
        scores,
        key=scores.get,
    )

    best_score = scores[best_intent]

    sorted_scores = sorted(
        scores.values(),
        reverse=True,
    )

    second_score = (
        sorted_scores[1]
        if len(sorted_scores) > 1
        else 0.0
    )

    # =========================================================================
    # CONFIDENCE
    # =========================================================================

    confidence = min(
        0.99,
        0.55 + min(best_score, 1.0) * 0.35,
    )

    # =========================================================================
    # AMBIGUITY CHECK
    # =========================================================================

    if (
        best_score > 0
        and abs(best_score - second_score) < 0.15
    ):
        return IntentResult(
            intent=UNKNOWN,
            confidence=confidence,
            matched_terms=(
                meddra_matches
                + pv_matches
                + project_matches
            )[:10],
            reason=(
                "Multiple knowledge domains were detected "
                "without sufficient routing separation."
            ),
        )

    # =========================================================================
    # SELECT MATCHED TERMS
    # =========================================================================

    if best_intent == MEDDRA:
        matched = meddra_matches

    elif best_intent == GENERAL_PV:
        matched = pv_matches

    else:
        matched = project_matches

    # =========================================================================
    # FINAL RESULT
    # =========================================================================

    return IntentResult(
        intent=best_intent,
        confidence=confidence,
        matched_terms=matched[:10],
        reason=(
            f"Query routed to {best_intent} "
            f"based on domain-specific terminology."
        ),
    )


# ============================================================================
# OPTIONAL DIRECT TEST
# ============================================================================

if __name__ == "__main__":

    tests = [
        "itching",
        "What is the MedDRA term for itching?",
        "loss of consciousness",
        "patient lost consciousness",
        "passed out",
        "difficulty breathing",
        "feels like vomiting",
        "patient feels like vomiting",
        "What is an ICSR?",
        "What is narrative QC?",
        "How does FAISS work in NarrativeGuard?",
        "What is the weather today?",
    ]

    print("=" * 90)
    print("PV NarrativeGuard - Intent Router Test")
    print("=" * 90)

    for query in tests:

        result = route_query(query)

        print(
            f"{query:<50} -> "
            f"{result.intent:<12} "
            f"{result.confidence:.2f} "
            f"{result.matched_terms}"
        )