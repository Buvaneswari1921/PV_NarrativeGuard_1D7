import pandas as pd

from src.chatbot.rab_bot import (
    RABBot,
    OUT_OF_CONTEXT,
    RAG_ANSWER,
    ASK_ADD_TO_DATASET,
    MEDDRA_ANSWER,
)


def fake_retriever(query, top_k=3):
    if "meddra" in query.lower():
        return pd.DataFrame([
            {
                "chunk_id": "CHUNK_001",
                "source": "meddra.txt",
                "text": "MedDRA is a standardized medical terminology.",
                "similarity_score": 0.91,
            }
        ])

    if "pharmacovigilance" in query.lower():
        return pd.DataFrame([
            {
                "chunk_id": "CHUNK_002",
                "source": "pv.txt",
                "text": (
                    "Pharmacovigilance is concerned with the detection "
                    "and assessment of adverse effects."
                ),
                "similarity_score": 0.91,
            }
        ])

    return pd.DataFrame([
        {
            "chunk_id": "CHUNK_999",
            "source": "weak.txt",
            "text": "Unrelated weak result.",
            "similarity_score": 0.42,
        }
    ])


def test_out_of_context():
    bot = RABBot(retriever=fake_retriever)
    result = bot.ask("What is the capital of France?")
    assert result.response_mode == OUT_OF_CONTEXT


def test_relevant_and_found():
    bot = RABBot(retriever=fake_retriever)
    result = bot.ask("What is pharmacovigilance?")

    assert result.response_mode == RAG_ANSWER
    assert result.evidence
    assert "Pharmacovigilance is concerned" in result.message
    assert "CHUNK_002" in result.message
    assert "pv.txt" in result.message


def test_relevant_but_missing():
    bot = RABBot(retriever=fake_retriever)
    result = bot.ask("Explain SUSAR signal triage methodology.")
    assert result.response_mode == ASK_ADD_TO_DATASET


def test_meddra_query_returns_controlled_answer():
    class FakeMedDRAAdapter:

        def query(self, query):
            return {
                "status": "SUPPORTED",
                "message": "Terminology match found.",
                "result": {
                    "preferred_label": "Nausea AE",
                    "meddra_id": "10028813",
                    "decision_score": 0.88,
                },
                "candidates": [],
            }

    bot = RABBot(
        retriever=fake_retriever,
        meddra_adapter=FakeMedDRAAdapter(),
    )

    result = bot.ask("patient feels like vomiting")

    assert result.response_mode == MEDDRA_ANSWER
    assert "Nausea" in result.message
    assert "10028813" in result.message
    assert result.confidence == 0.88
