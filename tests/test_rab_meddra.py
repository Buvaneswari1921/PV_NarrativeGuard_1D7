"""
PV_NarrativeGuard - RAB / MedDRA Regression Tests

Purpose:
    Validate the contract between:

        Query
          ↓
        RABBot
          ↓
        MedDRA route
          ↓
        MedDRA Adapter
          ↓
        Terminology decision
          ↓
        RABResponse

The tests deliberately separate:

    1. Unit-level RAB routing/response behavior
    2. Real MedDRA integration smoke tests

This prevents changes to the chatbot orchestration layer from
silently breaking the controlled MedDRA workflow.
"""

from __future__ import annotations

import pytest

from src.chatbot.rab_bot import (
    RABBot,
    MEDDRA_ANSWER,
    ASK_ADD_TO_DATASET,
    CLARIFICATION_REQUIRED,
)

from src.chatbot.grounded_answer import (
    preserve_meddra_decision,
)


# ============================================================
# TEST FIXTURES
# ============================================================

class FakeMedDRAAdapter:
    """
    Controlled fake adapter used to test RABBot behavior
    independently from the embedding model and FAISS.
    """

    def __init__(self, decision):
        self.decision = decision
        self.queries = []

    def query(self, user_query):
        self.queries.append(user_query)
        return self.decision


def make_supported_decision(
    label="headache AE",
    meddra_id="10019211",
    similarity=0.5722,
    decision_score=0.8222,
    mapped_term=None,
    match_method=None,
):
    result = {
        "similarity_score": similarity,
        "oae_id": "test:oae",
        "preferred_label": label,
        "synonyms": None,
        "definition": "Test definition.",
        "parents": "test:parent",
        "meddra_id": meddra_id,
        "decision_score": decision_score,
        "lexical_match": False,
        "clinical_phrase_match": True,
        "unsupported_specificity": False,
        "general_term": True,
    }

    if mapped_term is not None:
        result["mapped_term"] = mapped_term

    if match_method is not None:
        result["match_method"] = match_method

    return {
        "status": "SUPPORTED",
        "message": "Terminology match found.",
        "result": result,
        "candidates": [result],
    }


def make_ambiguous_decision():
    candidate_1 = {
        "similarity_score": 0.72,
        "preferred_label": "candidate one AE",
        "meddra_id": "10000001",
        "decision_score": 0.72,
    }

    candidate_2 = {
        "similarity_score": 0.71,
        "preferred_label": "candidate two AE",
        "meddra_id": "10000002",
        "decision_score": 0.71,
    }

    return {
        "status": "AMBIGUOUS",
        "message": (
            "Multiple MedDRA concepts may fit. "
            "Please provide more clinical context."
        ),
        "result": {
            "decision_score": 0.72,
        },
        "candidates": [
            candidate_1,
            candidate_2,
        ],
    }


def make_unsupported_decision():
    return {
        "status": "NOT_SUPPORTED",
        "message": (
            "This is out of my context."
        ),
        "result": None,
        "candidates": [],
    }


# ============================================================
# UNIT TESTS - GROUNDED MEDDRA DECISION
# ============================================================

def test_preserve_supported_meddra_decision():
    """
    A supported terminology decision must be preserved exactly
    rather than regenerated or reinterpreted.
    """

    decision = make_supported_decision()

    grounded = preserve_meddra_decision(
        decision
    )

    assert grounded.grounded is True

    assert grounded.reason == (
        "SUPPORTED_MEDDRA_DECISION_PRESERVED"
    )

    assert grounded.confidence == pytest.approx(
        0.8222
    )

    assert (
        grounded.answer
        == "MedDRA terminology result: headache "
        "(MedDRA ID: 10019211)."
    )

    assert len(grounded.evidence) == 1

    assert (
        grounded.evidence[0]["meddra_id"]
        == "10019211"
    )

    assert (
        grounded.evidence[0]["similarity_score"]
        == pytest.approx(0.5722)
    )


# ============================================================
# UNIT TESTS - RAB MEDDRA ROUTE
# ============================================================

def test_rab_routes_supported_query_to_meddra():
    """
    A clinical terminology query must go through the MedDRA route
    and produce MEDDRA_ANSWER when the decision is supported.
    """

    decision = make_supported_decision()

    adapter = FakeMedDRAAdapter(
        decision
    )

    bot = RABBot(
        meddra_adapter=adapter
    )

    response = bot.ask(
        "patient has headache"
    )

    assert response.response_mode == (
        MEDDRA_ANSWER
    )

    assert response.route == "MEDDRA"

    assert response.confidence == pytest.approx(
        0.8222
    )

    assert (
        response.message
        == "MedDRA terminology result: headache "
        "(MedDRA ID: 10019211)."
    )

    assert adapter.queries == [
        "patient has headache"
    ]

    assert len(response.evidence) == 1

    assert (
        response.evidence[0]["meddra_id"]
        == "10019211"
    )


def test_rab_preserves_known_mapping():
    """
    A known terminology mapping must survive the complete RAB
    orchestration layer.
    """

    decision = make_supported_decision(
        label="loose stools AE",
        meddra_id="10024840",
        similarity=0.6368,
        decision_score=0.8868,
        mapped_term="loose stools",
        match_method="known_mapping",
    )

    adapter = FakeMedDRAAdapter(
        decision
    )

    bot = RABBot(
        meddra_adapter=adapter
    )

    response = bot.ask(
        "patient has loose stools"
    )

    assert response.response_mode == (
        MEDDRA_ANSWER
    )

    assert response.route == "MEDDRA"

    assert response.confidence == pytest.approx(
        0.8868
    )

    assert (
        response.evidence[0]["meddra_id"]
        == "10024840"
    )

    assert (
        response.evidence[0]["similarity_score"]
        == pytest.approx(0.6368)
    )

    assert (
        response.evidence[0]["decision_score"]
        == pytest.approx(0.8868)
    )

    assert (
        response.evidence[0]["clinical_phrase_match"]
        is True
    )

    assert (
        response.evidence[0]["match_method"]
        == "known_mapping"
    )

    assert (
        response.evidence[0]["mapped_term"]
        == "loose stools"
    )


# ============================================================
# UNIT TESTS - AMBIGUITY
# ============================================================

def test_rab_returns_clarification_for_ambiguous_meddra():
    """
    Ambiguous terminology decisions must never be converted into
    a definitive MedDRA answer.
    """

    decision = make_ambiguous_decision()

    adapter = FakeMedDRAAdapter(
        decision
    )

    bot = RABBot(
        meddra_adapter=adapter
    )

    response = bot.ask(
        "patient has an unusual neurological symptom"
    )

    assert response.response_mode == (
        CLARIFICATION_REQUIRED
    )

    assert response.route == "MEDDRA"

    assert response.confidence == pytest.approx(
        0.72
    )

    assert len(response.evidence) == 2

    assert (
        "Multiple MedDRA concepts"
        in response.message
    )


# ============================================================
# UNIT TESTS - KNOWLEDGE GAP
# ============================================================

def test_rab_returns_knowledge_gap_for_unsupported_meddra():
    """
    Unsupported terminology must produce a controlled knowledge
    gap rather than a fabricated terminology result.
    """

    decision = make_unsupported_decision()

    adapter = FakeMedDRAAdapter(
        decision
    )

    bot = RABBot(
        meddra_adapter=adapter
    )

    response = bot.ask(
        "patient developed zorbaflux syndrome"
    )

    assert response.response_mode == (
        ASK_ADD_TO_DATASET
    )

    assert response.route == "MEDDRA"

    assert response.confidence == 0.0

    assert (
        "current MedDRA terminology dataset"
        in response.message
    )

    assert response.evidence == []


# ============================================================
# UNIT TESTS - SAFETY CONTRACT
# ============================================================

def test_unsupported_meddra_decision_cannot_be_grounded():
    """
    preserve_meddra_decision must reject anything that is not
    explicitly SUPPORTED.
    """

    decision = make_unsupported_decision()

    grounded = preserve_meddra_decision(
        decision
    )

    assert grounded.grounded is False

    assert grounded.confidence == 0.0

    assert grounded.reason == (
        "MEDDRA_DECISION_NOT_SUPPORTED"
    )


# ============================================================
# REAL MEDDRA INTEGRATION TESTS
# ============================================================

@pytest.mark.integration
def test_real_meddra_headache():
    """
    Real end-to-end MedDRA integration smoke test.

    This loads:
        Sentence Transformer
        FAISS index
        MedDRA metadata
        terminology engine
    """

    from src.chatbot.meddra_adapter import (
        MedDRAAdapter,
    )

    result = MedDRAAdapter().query(
        "patient has headache"
    )

    assert result["status"] == "SUPPORTED"

    assert (
        result["result"]["meddra_id"]
        == "10019211"
    )

    assert (
        result["result"]["preferred_label"]
        == "headache AE"
    )

    assert (
        result["result"]["clinical_phrase_match"]
        is True
    )

    assert (
        result["result"]["decision_score"]
        > 0.65
    )


@pytest.mark.integration
def test_real_meddra_loose_stools():
    """
    Real known-mapping smoke test.

    Validates the specific mapping already confirmed manually.
    """

    from src.chatbot.meddra_adapter import (
        MedDRAAdapter,
    )

    result = MedDRAAdapter().query(
        "patient has loose stools"
    )

    assert result["status"] == "SUPPORTED"

    terminology = result["result"]

    assert (
        terminology["meddra_id"]
        == "10024840"
    )

    assert (
        terminology["preferred_label"]
        == "loose stools AE"
    )

    assert (
        terminology["mapped_term"]
        == "loose stools"
    )

    assert (
        terminology["match_method"]
        == "known_mapping"
    )

    assert (
        terminology["clinical_phrase_match"]
        is True
    )

    assert (
        terminology["decision_score"]
        > 0.65
    )


@pytest.mark.integration
def test_real_meddra_unknown_term_is_not_supported():
    """
    Real MedDRA integration must not fabricate a result for an
    unsupported terminology query.
    """

    from src.chatbot.meddra_adapter import (
        MedDRAAdapter,
    )

    result = MedDRAAdapter().query(
        "patient developed zorbaflux syndrome"
    )

    assert result["status"] != "SUPPORTED"

    assert (
        result["result"] is None
        or result["result"].get("meddra_id")
        in (None, "")
    )



def test_real_meddra_fast_heartbeat_is_controlled_ambiguity():
    from src.chatbot.meddra_adapter import MedDRAAdapter

    result = MedDRAAdapter().query("fast heartbeat")

    assert result["status"] == "AMBIGUOUS"
    assert result["result"] is None
    assert len(result["candidates"]) >= 2


def test_real_meddra_measured_heart_rate_resolves_to_tachycardia():
    from src.chatbot.meddra_adapter import MedDRAAdapter

    result = MedDRAAdapter().query("heart rate was 130 bpm")

    assert result["status"] == "SUPPORTED"
    assert result["result"]["meddra_id"] == "10043071"


# ============================================================
# END
# ============================================================

def test_curated_candidate_enrichment_prevents_faiss_shadowing():
    """
    A FAISS hit with the same MedDRA ID as a curated narrative candidate
    must inherit the curated provenance so low raw similarity cannot shadow
    a verified PV expression.
    """
    from src.chatbot.meddra_adapter import MedDRAAdapter

    adapter = MedDRAAdapter()
    adapter._model = object()
    adapter._index = object()
    adapter._metadata = object()

    # Bypass the real FAISS/model call and exercise the merge logic directly.
    adapter.retrieve_candidates = lambda query: [
        {
            "similarity_score": 0.20,
            "preferred_label": "vomiting AE",
            "synonyms": "",
            "definition": "",
            "parents": "",
            "meddra_id": "10047700",
        }
    ]

    result = adapter.query("patient threw up twice")
    assert result["status"] == "SUPPORTED"
    assert result["result"]["meddra_id"] == "10047700"
    assert result["result"]["matched_pv_expression"] == "threw up"


@pytest.mark.integration
def test_real_meddra_threw_up_narrative_resolves_to_vomiting():
    from src.chatbot.meddra_adapter import MedDRAAdapter
    result = MedDRAAdapter().query("patient threw up twice")
    assert result["status"] == "SUPPORTED"
    assert result["result"]["meddra_id"] == "10047700"
    assert result["result"]["preferred_label"] == "vomiting AE"
    assert result["result"]["matched_pv_expression"] == "threw up"


def test_rab_negated_event_is_not_a_dataset_gap():
    from src.chatbot.rab_bot import RABBot, CLARIFICATION_REQUIRED
    from src.chatbot.meddra_adapter import MedDRAAdapter
    bot = RABBot(meddra_adapter=MedDRAAdapter())
    response = bot.ask("patient denies headache")
    assert response.response_mode == CLARIFICATION_REQUIRED
    assert "No MedDRA coding recommendation" in response.message
    assert response.operational_state == "COMPLETED"
