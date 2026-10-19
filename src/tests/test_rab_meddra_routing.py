import pandas as pd

from src.chatbot.rab_bot import (
    RABBot,
    OUT_OF_CONTEXT,
    MEDDRA_ANSWER,
    RAG_ANSWER,
    ASK_ADD_TO_DATASET,
)

from src.chatbot.intent_router import route_query, MEDDRA


class FakeMedDRAAdapter:
    def query(self, query):
        return {
            "status": "SUPPORTED",
            "message": "Terminology decision supported.",
            "result": {
                "preferred_label": "Headache AE",
                "meddra_id": "10019211",
                "decision_score": 0.91,
                "similarity_score": 0.88,
            },
            "candidates": [],
        }


def fake_general_retriever(query, top_k=3):
    return pd.DataFrame([
        {
            "chunk_id": "PV_001",
            "source": "pv.txt",
            "text": "An ICSR is an individual case safety report.",
            "similarity_score": 0.91,
        }
    ])


def test_irrelevant_query_does_not_retrieve():
    bot = RABBot(
        retriever=fake_general_retriever,
        meddra_adapter=FakeMedDRAAdapter(),
    )
    result = bot.ask("What is the capital of France?")
    assert result.response_mode == OUT_OF_CONTEXT


def test_meddra_query_uses_meddra_route():
    bot = RABBot(
        retriever=fake_general_retriever,
        meddra_adapter=FakeMedDRAAdapter(),
    )
    result = bot.ask("What is MedDRA and what is the Preferred Term for headache?")
    assert result.response_mode == MEDDRA_ANSWER
    assert result.route == "MEDDRA"
    assert "Headache" in result.message


def test_general_pv_query_uses_general_rag():
    bot = RABBot(
        retriever=fake_general_retriever,
        meddra_adapter=FakeMedDRAAdapter(),
    )
    result = bot.ask("What is an ICSR?")
    assert result.response_mode == RAG_ANSWER
    assert result.route == "GENERAL_PV"


def test_router_uses_pv_core_for_natural_language_expression():
    result = route_query("patient threw up twice")
    assert result.intent == MEDDRA
    assert "threw up" in result.matched_terms


def test_router_uses_pv_core_for_fast_heartbeat():
    result = route_query("fast heartbeat")
    assert result.intent == MEDDRA
    assert "fast heartbeat" in result.matched_terms
