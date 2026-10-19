from meddra_rag.terminology_engine import (
    load_pv_core_vocabulary,
    get_pv_core_alias_match,
    decide,
)


def test_pv_core_contains_150_concepts():
    vocab = load_pv_core_vocabulary()
    assert len(vocab) == 150
    assert vocab["preferred_label"].nunique() == 150


def test_longest_alias_wins():
    vocab = load_pv_core_vocabulary()
    match = get_pv_core_alias_match(
        "patient developed severe chest pain",
        vocab,
    )
    assert match is not None
    assert match["preferred_label"] == "chest pain"


def test_pv_core_alias_selects_retrieved_candidate():
    vocab = load_pv_core_vocabulary()
    match = get_pv_core_alias_match(
        "patient complained of head pain",
        vocab,
    )
    assert match is not None
    assert match["preferred_label"] == "headache"

    candidates = [{
        "similarity_score": 0.61,
        "preferred_label": "headache",
        "synonyms": "",
        "definition": "",
        "parents": "",
        "meddra_id": match["meddra_id"],
    }]

    result = decide(
        "patient complained of head pain",
        candidates,
    )
    assert result["status"] == "SUPPORTED"
    assert result["result"]["preferred_label"] == "headache"
    assert result["result"]["match_method"] == "pv_core_alias"
    assert result["result"]["matched_pv_expression"] == "head pain"


def test_core_layer_cannot_invent_missing_retrieval_candidate():
    result = decide("patient complained of head pain", [])
    assert result["status"] == "OUT_OF_CONTEXT"

def test_curated_core_candidate_can_be_selected_without_fabricated_similarity():
    vocab = load_pv_core_vocabulary()
    match = get_pv_core_alias_match("patient threw up twice", vocab)
    assert match is not None and match["preferred_label"] == "vomiting"
    candidates = [{
        "similarity_score": 0.0, "preferred_label": "vomiting AE",
        "synonyms": "regurgitation AE", "definition": "vomiting definition",
        "parents": "", "meddra_id": "10047700",
        "curated_core_candidate": True, "curated_match_score": 0.94,
    }]
    result = decide("patient threw up twice", candidates)
    assert result["status"] == "SUPPORTED"
    assert result["result"]["meddra_id"] == "10047700"
    assert result["result"]["match_method"] == "pv_core_alias"
    assert result["result"]["matched_pv_expression"] == "threw up"
    assert result["result"]["decision_score"] >= 0.94

def test_fast_heartbeat_returns_controlled_ambiguity():
    candidates = [
        {"similarity_score": 0.0, "preferred_label": "tachycardia AE", "synonyms": "", "definition": "increased heart rate", "parents": "", "meddra_id": "10043071", "curated_core_candidate": True, "curated_match_score": 0.90, "ambiguity_candidate": True},
        {"similarity_score": 0.0, "preferred_label": "palpitations AE", "synonyms": "", "definition": "awareness of heartbeat", "parents": "", "meddra_id": "10033557", "curated_core_candidate": True, "curated_match_score": 0.90, "ambiguity_candidate": True},
    ]
    result = decide("fast heartbeat", candidates)
    assert result["status"] == "AMBIGUOUS"
    assert "measured elevated heart rate" in result["message"]
    assert len(result["candidates"]) == 2

def test_loose_stools_does_not_shadow_validated_legacy_mapping():
    """
    The generic diarrhea core concept must not shadow the validated
    loose-stools terminology mapping (10024840 / loose stools AE).
    """
    from meddra_rag.terminology_engine import (
        get_pv_core_alias_match,
        load_pv_core_vocabulary,
    )

    match = get_pv_core_alias_match(
        "patient has loose stools",
        load_pv_core_vocabulary(),
    )

    assert match is None
