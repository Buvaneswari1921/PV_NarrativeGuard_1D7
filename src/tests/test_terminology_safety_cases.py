from src.chatbot.query_classifier import classify_query
from src.chatbot.intent_router import route_query, MEDDRA


def test_vomiting_is_relevant():
    assert classify_query("patient feels like vomiting").scope == "RELEVANT"


def test_lightheaded_is_relevant():
    assert classify_query("patient feels lightheaded").scope == "RELEVANT"


def test_skin_itching_is_relevant():
    assert classify_query("patient developed skin itching").scope == "RELEVANT"


def test_vomiting_routes_to_meddra():
    assert route_query("patient feels like vomiting").intent == MEDDRA


def test_lightheaded_routes_to_meddra():
    assert route_query("patient feels lightheaded").intent == MEDDRA


def test_skin_itching_routes_to_meddra():
    assert route_query("patient developed skin itching").intent == MEDDRA


def test_unrelated_query_stays_out_of_scope():
    assert classify_query("What is the capital of France?").scope == "OUT_OF_SCOPE"
