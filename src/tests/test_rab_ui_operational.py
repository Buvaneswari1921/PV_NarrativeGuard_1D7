from types import SimpleNamespace

import src.ui.rab_ui as rab_ui


def test_knowledge_gap_action_only_applies_to_knowledge_gap():
    response = SimpleNamespace(
        response_mode="RAG_ANSWER",
        query="what is pharmacovigilance",
    )

    # Function should return without attempting
    # to create a review request.
    assert (
        rab_ui._render_knowledge_gap_action(response)
        is None
    )


def test_knowledge_gap_action_handles_missing_query():
    response = SimpleNamespace(
        response_mode="ASK_ADD_TO_DATASET",
        query="",
    )

    assert (
        rab_ui._render_knowledge_gap_action(response)
        is None
    )