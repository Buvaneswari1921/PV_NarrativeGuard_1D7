from src.chatbot.intent_router import (
    route_query,
    MEDDRA,
    GENERAL_PV,
    PROJECT,
    UNKNOWN,
)


def test_meddra_query():
    result = route_query(
        "What is MedDRA and what is a Preferred Term?"
    )
    assert result.intent == MEDDRA


def test_general_pv_query():
    result = route_query(
        "What is an Individual Case Safety Report?"
    )
    assert result.intent == GENERAL_PV


def test_project_query():
    result = route_query(
        "How does the PV NarrativeGuard RAG engine work?"
    )
    assert result.intent == PROJECT


def test_unknown_query():
    result = route_query(
        "What is the capital of France?"
    )
    assert result.intent == UNKNOWN


def test_clinical_event_routes_to_meddra():
    result = route_query(
        "patient has loose stools"
    )
    assert result.intent == MEDDRA


def test_clinical_event_developed_routes_to_meddra():
    result = route_query(
        "patient developed skin itching"
    )
    assert result.intent == MEDDRA
