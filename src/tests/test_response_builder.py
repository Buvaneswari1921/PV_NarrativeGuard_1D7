from src.chatbot.response_builder import (
    build_meddra_response,
    build_rag_response,
    build_out_of_context_response,
    build_clarification_response,
)


def test_meddra_supported_response():

    decision = {
        "status": "SUPPORTED",
        "message": "Terminology match found.",
        "result": {
            "preferred_label": "pruritus",
            "meddra_id": "10037087",
            "decision_score": 0.72,
            "match_method": "clinical_phrase",
        },
        "candidates": [],
    }

    result = build_meddra_response(decision)

    assert result.response_mode == "MEDDRA_ANSWER"
    assert result.route == "MEDDRA"
    assert result.confidence == 0.72
    assert "pruritus" in result.message
    assert "10037087" in result.message
    assert "72.0%" in result.message
    assert len(result.evidence) == 1


def test_meddra_ambiguous_response():

    decision = {
        "status": "AMBIGUOUS",
        "message": "Additional clinical context required.",
        "result": {
            "decision_score": 0.0,
        },
        "candidates": [
            {
                "preferred_label": "nausea",
                "meddra_id": "10028813",
            },
            {
                "preferred_label": "vomiting",
                "meddra_id": "10047700",
            },
        ],
    }

    result = build_meddra_response(decision)

    assert result.response_mode == "CLARIFICATION_REQUIRED"
    assert result.route == "MEDDRA"
    assert "ambiguous" in result.message.lower()
    assert "nausea" in result.message
    assert "vomiting" in result.message
    assert len(result.evidence) == 2


def test_meddra_knowledge_gap_response():

    decision = {
        "status": "NOT_FOUND",
        "message": "No sufficiently supported terminology.",
        "result": None,
        "candidates": [],
    }

    result = build_meddra_response(decision)

    assert result.response_mode == "ASK_ADD_TO_DATASET"
    assert result.route == "MEDDRA"
    assert "knowledge gap" in result.message.lower()


def test_general_pv_sufficient_response():

    decision = type(
        "Decision",
        (),
        {
            "status": "SUFFICIENT",
            "confidence": 0.91,
            "evidence": [
                {
                    "chunk_id": "CHUNK_001",
                    "source": "general_pv.txt",
                    "text": "PV knowledge.",
                    "similarity_score": 0.91,
                }
            ],
            "reason": "Strong evidence.",
        },
    )()

    result = build_rag_response(
        decision,
        "GENERAL_PV",
    )

    assert result.response_mode == "RAG_ANSWER"
    assert result.route == "GENERAL_PV"
    assert result.confidence == 0.91
    assert "91.0%" in result.message
    assert len(result.evidence) == 1


def test_general_pv_knowledge_gap_response():

    decision = type(
        "Decision",
        (),
        {
            "status": "INSUFFICIENT",
            "confidence": 0.42,
            "evidence": [],
            "reason": "Insufficient evidence.",
        },
    )()

    result = build_rag_response(
        decision,
        "GENERAL_PV",
    )

    assert result.response_mode == "ASK_ADD_TO_DATASET"
    assert result.route == "GENERAL_PV"
    assert result.confidence == 0.42
    assert "knowledge gap" in result.message.lower()


def test_out_of_context_response():

    result = build_out_of_context_response(
        confidence=0.95,
    )

    assert result.response_mode == "OUT_OF_CONTEXT"
    assert result.route == "OUT_OF_CONTEXT"
    assert result.confidence == 0.95
    assert result.evidence == []
    assert "outside" in result.message.lower()


def test_clarification_response():

    result = build_clarification_response(
        confidence=0.55,
    )

    assert result.response_mode == "CLARIFICATION_REQUIRED"
    assert result.route == "UNKNOWN"
    assert result.confidence == 0.55
    assert result.evidence == []
    assert "clarify" in result.message.lower()
def test_meddra_response_formats_multiple_ids_separately():
    from src.chatbot.response_builder import build_meddra_response
    built = build_meddra_response({
        "status": "SUPPORTED", "message": "Terminology match found.",
        "result": {"preferred_label": "hypertension AE", "meddra_id": "10020772|10005750", "decision_score": 0.94},
        "candidates": [],
    })
    assert "MedDRA ID(s): 10020772, 10005750" in built.message
    assert "multiple MedDRA IDs" in built.message
