from src.chatbot.query_classifier import classify_query


def test_irrelevant_query():
    result = classify_query("What is the capital of France?")
    assert result.scope == "OUT_OF_SCOPE"


def test_meddra_explicit_query():
    result = classify_query("What is MedDRA?")
    assert result.scope == "RELEVANT"


def test_clinical_event_loose_stools():
    result = classify_query("patient has loose stools")
    assert result.scope == "RELEVANT"


def test_clinical_event_developed_rash():
    result = classify_query("patient developed skin itching")
    assert result.scope == "RELEVANT"


def test_clinical_event_experienced_dizziness():
    result = classify_query("patient experienced dizziness")
    assert result.scope == "RELEVANT"

def test_pv_core_routes_natural_language_vomiting():
    result = classify_query("patient threw up twice")
    assert result.scope == "RELEVANT"
    assert "threw up" in result.matched_terms


def test_pv_core_routes_natural_language_pruritus():
    result = classify_query("itchy skin")
    assert result.scope == "RELEVANT"
    assert "itchy skin" in result.matched_terms


def test_pv_core_routes_natural_language_tachycardia():
    result = classify_query("fast heartbeat")
    assert result.scope == "RELEVANT"
    assert "fast heartbeat" in result.matched_terms


def test_pv_core_routes_natural_language_hypertension():
    result = classify_query("high blood pressure")
    assert result.scope == "RELEVANT"
    assert "high blood pressure" in result.matched_terms
