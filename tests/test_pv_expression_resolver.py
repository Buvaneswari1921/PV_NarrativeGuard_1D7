"""Phase 1B.3 tests for natural-language PV expression resolution."""
from meddra_rag.pv_expression_resolver import resolve_core_expression, is_negated_expression

def test_threw_up_resolves_to_vomiting_metadata():
    result = resolve_core_expression("patient threw up twice")
    assert result is not None and result["status"] == "RESOLVED"
    assert result["match"]["alias"] == "threw up"
    assert any(c["preferred_label"] == "vomiting AE" and c["meddra_id"] == "10047700" for c in result["candidates"])

def test_itchy_skin_resolves_to_pruritus_metadata():
    result = resolve_core_expression("itchy skin")
    assert result is not None and result["status"] == "RESOLVED"
    assert result["match"]["alias"] == "itchy skin"
    assert any(c["preferred_label"] == "pruritus AE" and c["meddra_id"] == "10037087" for c in result["candidates"])

def test_fast_heartbeat_requires_clinical_context():
    result = resolve_core_expression("fast heartbeat")
    assert result is not None and result["status"] == "AMBIGUOUS"
    labels = {c["preferred_label"] for c in result["candidates"]}
    assert "tachycardia AE" in labels and "palpitations AE" in labels
    assert "measured elevated heart rate" in result["ambiguity_reason"]

def test_negated_core_expression_is_not_resolved_as_an_event():
    result = resolve_core_expression("patient denies headache")
    assert result is not None and result["status"] == "NEGATED"
    assert result["candidates"] == []
    assert is_negated_expression("patient denies headache", "headache") is True

def test_all_150_pv_core_concepts_have_local_metadata_resolution():
    import pandas as pd
    from meddra_rag.terminology_engine import PV_CORE_FILE

    core = pd.read_csv(PV_CORE_FILE).fillna("")
    unresolved = []
    for _, row in core.iterrows():
        alias = str(row["pv_query_aliases"]).split("|")[0].strip()
        result = resolve_core_expression(alias)
        if not result or not result.get("candidates"):
            unresolved.append((row["core_id"], row["preferred_label"], alias))

    assert unresolved == []


def test_narrative_resolution_supports_measured_heart_rate_context():
    from meddra_rag.pv_expression_resolver import resolve_narrative_expression
    result = resolve_narrative_expression("heart rate was 130 bpm")
    assert result is not None
    assert result["status"] == "RESOLVED"
    assert any(c["meddra_id"] == "10043071" for c in result["candidates"])
    assert result["clinical_context"]


def test_narrative_resolution_negation_is_explicit():
    from meddra_rag.pv_expression_resolver import resolve_narrative_expression
    result = resolve_narrative_expression("patient denies headache")
    assert result is not None
    assert result["status"] == "NEGATED"
    assert result["candidates"] == []
