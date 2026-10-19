from src.chatbot.grounded_answer import (
    generate_grounded_rag_answer,
    preserve_meddra_decision,
)


def test_grounded_rag_answer_uses_only_evidence():

    evidence = [
        {
            "chunk_id": "CHUNK_001",
            "source": "pv.txt",
            "text": "Pharmacovigilance concerns the detection and assessment of adverse effects.",
            "similarity_score": 0.91,
        }
    ]

    result = generate_grounded_rag_answer(
        query="What is pharmacovigilance?",
        evidence=evidence,
        confidence=0.91,
    )

    assert result.grounded is True
    assert "Pharmacovigilance concerns" in result.answer
    assert "CHUNK_001" in result.answer
    assert "pv.txt" in result.answer


def test_grounded_rag_rejects_low_confidence():

    evidence = [
        {
            "chunk_id": "CHUNK_002",
            "source": "weak.txt",
            "text": "Weak evidence.",
        }
    ]

    result = generate_grounded_rag_answer(
        query="test",
        evidence=evidence,
        confidence=0.42,
    )

    assert result.grounded is False
    assert result.reason == "LOW_CONFIDENCE_EVIDENCE"


def test_grounded_rag_rejects_empty_evidence():

    result = generate_grounded_rag_answer(
        query="test",
        evidence=[],
        confidence=0.95,
    )

    assert result.grounded is False
    assert result.reason == "EMPTY_OR_INVALID_EVIDENCE"


def test_grounded_rag_rejects_evidence_without_text():

    evidence = [
        {
            "chunk_id": "CHUNK_003",
            "source": "empty.txt",
        }
    ]

    result = generate_grounded_rag_answer(
        query="test",
        evidence=evidence,
        confidence=0.95,
    )

    assert result.grounded is False
    assert result.reason == "EMPTY_OR_INVALID_EVIDENCE"


def test_meddra_answer_preserves_supported_decision():

    decision = {
        "status": "SUPPORTED",
        "result": {
            "preferred_label": "Nausea AE",
            "meddra_id": "10028813",
            "decision_score": 0.88,
        },
    }

    result = preserve_meddra_decision(
        decision
    )

    assert result.grounded is True
    assert "Nausea" in result.answer
    assert "10028813" in result.answer


def test_meddra_ambiguous_is_not_generated():

    decision = {
        "status": "AMBIGUOUS",
        "result": None,
        "candidates": [
            {
                "preferred_label": "Concept A",
                "meddra_id": "111",
            },
            {
                "preferred_label": "Concept B",
                "meddra_id": "222",
            },
        ],
    }

    result = preserve_meddra_decision(
        decision
    )

    assert result.grounded is False
    assert result.reason == "MEDDRA_DECISION_NOT_SUPPORTED"


def test_meddra_unsupported_is_not_generated():

    decision = {
        "status": "NOT_FOUND",
        "result": None,
        "candidates": [],
    }

    result = preserve_meddra_decision(
        decision
    )

    assert result.grounded is False
    assert result.reason == "MEDDRA_DECISION_NOT_SUPPORTED"

def test_preserve_meddra_decision_formats_multiple_ids_without_pipe_delimiter():
    decision = {
        "status": "SUPPORTED",
        "result": {
            "preferred_label": "dyspnea AE",
            "meddra_id": "10013968|10013963",
            "decision_score": 0.80,
        },
        "candidates": [],
    }

    grounded = preserve_meddra_decision(decision)

    assert grounded.grounded is True
    assert "Retrieved MedDRA ID(s): 10013968, 10013963." in grounded.answer
    assert "10013968|10013963" not in grounded.answer
    assert "verify the applicable MedDRA concept" in grounded.answer
