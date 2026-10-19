"""
PV_NarrativeGuard - Terminology Engine Unit Tests

Purpose:
    Validate the controlled terminology decision logic independently
    of the RAB orchestration layer.

These tests use synthetic candidate records so that terminology-engine
decision behavior can be tested deterministically without loading the
Sentence Transformer or FAISS index.

The existing meddra_rag/test_terminology_engine.py remains available as
a manual integration/diagnostic runner.
"""

import pytest

from meddra_rag.terminology_engine import (
    decide,
    load_controlled_inventory,
    get_controlled_inventory_match,
)


# ============================================================
# Candidate factory
# ============================================================

def candidate(
    preferred_label="headache AE",
    similarity_score=0.80,
    meddra_id="10019211",
    definition="A pain AE that has an outcome of headache.",
):
    """Create a minimal terminology candidate."""
    return {
        "similarity_score": similarity_score,
        "oae_id": "http://purl.obolibrary.org/obo/OAE_TEST",
        "preferred_label": preferred_label,
        "synonyms": "",
        "definition": definition,
        "parents": "http://purl.obolibrary.org/obo/OAE_PARENT",
        "meddra_id": meddra_id,
    }


# ============================================================
# Basic decision contract
# ============================================================

def test_supported_known_mapping():
    """Known clinical terminology mapping should be supported."""

    result = decide(
        "patient has loose stools",
        [
            candidate(
                preferred_label="loose stools AE",
                similarity_score=0.6368,
                meddra_id="10024840",
            )
        ],
    )

    assert result["status"] == "SUPPORTED"
    assert result["result"] is not None
    assert result["result"]["meddra_id"] == "10024840"
    assert result["result"]["match_method"] == "known_mapping"
    assert result["result"]["mapped_term"] == "loose stools"


def test_supported_headache():
    """A clinically matching headache candidate should be supported."""

    result = decide(
        "patient has headache",
        [
            candidate(
                preferred_label="headache AE",
                similarity_score=0.5722,
                meddra_id="10019211",
            )
        ],
    )

    assert result["status"] == "SUPPORTED"
    assert result["result"]["meddra_id"] == "10019211"


# ============================================================
# Unsupported terminology
# ============================================================

def test_unknown_term_is_not_supported():
    """A weak unrelated candidate must not become a supported answer."""

    result = decide(
        "patient developed zorbaflux syndrome",
        [
            candidate(
                preferred_label="headache AE",
                similarity_score=0.20,
                meddra_id="10019211",
            )
        ],
    )

    assert result["status"] != "SUPPORTED"


def test_empty_candidates_are_not_supported():
    """No terminology evidence must never produce a supported answer."""

    result = decide(
        "patient has an unknown clinical event",
        [],
    )

    assert result["status"] != "SUPPORTED"
    assert result["result"] is None


# ============================================================
# Candidate filtering
# ============================================================

def test_low_similarity_candidate_is_not_selected():
    """Candidates below the retrieval threshold should not be selected."""

    result = decide(
        "patient has headache",
        [
            candidate(
                preferred_label="headache AE",
                similarity_score=0.20,
                meddra_id="10019211",
            )
        ],
    )

    assert result["status"] != "SUPPORTED"


def test_stronger_candidate_is_preferred():
    """A stronger terminology candidate should outrank a weaker one."""

    result = decide(
        "patient has headache",
        [
            candidate(
                preferred_label="pain AE",
                similarity_score=0.50,
                meddra_id="10033371",
            ),
            candidate(
                preferred_label="headache AE",
                similarity_score=0.72,
                meddra_id="10019211",
            ),
        ],
    )

    assert result["result"] is not None
    assert result["result"]["meddra_id"] == "10019211"


# ============================================================
# Safety / specificity
# ============================================================

def test_unsupported_specificity_is_not_promoted():
    """
    A candidate containing unsupported clinical specificity should not
    automatically become the terminology decision.
    """

    result = decide(
        "patient has meningitis",
        [
            candidate(
                preferred_label="viral meningitis AE",
                similarity_score=0.80,
                meddra_id="10012345",
            )
        ],
    )

    assert result["status"] != "SUPPORTED"


# ============================================================
# Decision metadata
# ============================================================

def test_decision_preserves_similarity_score():
    """The original FAISS similarity should remain available."""

    result = decide(
        "patient has headache",
        [
            candidate(
                preferred_label="headache AE",
                similarity_score=0.72,
                meddra_id="10019211",
            )
        ],
    )

    assert result["result"] is not None
    assert result["result"]["similarity_score"] == pytest.approx(
        0.72
    )


def test_supported_result_contains_meddra_id():
    """Supported terminology decisions must expose the MedDRA ID."""

    result = decide(
        "patient has headache",
        [
            candidate(
                preferred_label="headache AE",
                similarity_score=0.75,
                meddra_id="10019211",
            )
        ],
    )

    assert result["status"] == "SUPPORTED"
    assert result["result"]["meddra_id"]

# ============================================================
# Controlled Decision Inventory
# ============================================================

def test_controlled_inventory_safe_exact_is_supported():
    """
    SAFE_EXACT inventory decisions should produce a supported
    terminology result using the controlled inventory.
    """

    inventory = load_controlled_inventory()

    assert not inventory.empty

    safe_exact = inventory[
        inventory["controlled_decision"] == "SAFE_EXACT"
    ]

    assert not safe_exact.empty

    row = safe_exact.iloc[0]

    term = row["Preferred Label"]
    controlled_id = str(
        row["candidate_meddra_id"]
    )

    candidate_record = candidate(
        preferred_label=term,
        similarity_score=1.0,
        meddra_id=controlled_id,
    )

    result = decide(
        term,
        [candidate_record],
    )

    assert result["status"] == "SUPPORTED"
    assert result["result"] is not None
    assert (
        result["result"]["match_method"]
        == "controlled_inventory"
    )
    assert (
        str(result["result"]["meddra_id"])
        == controlled_id
    )


def test_controlled_inventory_ambiguous_overrides_lexical_match():
    """
    An AMBIGUOUS controlled-inventory decision must not be
    overridden by a perfect lexical/semantic candidate.

    This protects the safety boundary between:
        retrieval similarity
    and
        controlled terminology resolution.
    """

    inventory = load_controlled_inventory()

    assert not inventory.empty

    ambiguous = inventory[
        inventory["controlled_decision"] == "AMBIGUOUS"
    ]

    assert not ambiguous.empty

    # Use a deterministic ambiguous record that has
    # multiple controlled MedDRA IDs.
    row = ambiguous.iloc[0]

    term = row["Preferred Label"]

    candidate_ids = str(
        row["candidate_meddra_id"]
    ).split("|")

    candidate_ids = [
        item.strip()
        for item in candidate_ids
        if item.strip()
    ]

    assert len(candidate_ids) >= 2

    candidates = [
        candidate(
            preferred_label=term,
            similarity_score=1.0,
            meddra_id=meddra_id,
        )
        for meddra_id in candidate_ids
    ]

    result = decide(
        term,
        candidates,
    )

    assert result["status"] == "AMBIGUOUS"
    assert result["result"] is None


def test_controlled_inventory_lookup_returns_ambiguous():
    """
    The inventory lookup itself must expose AMBIGUOUS rather than
    silently selecting one candidate.
    """

    inventory = load_controlled_inventory()

    assert not inventory.empty

    ambiguous = inventory[
        inventory["controlled_decision"] == "AMBIGUOUS"
    ]

    assert not ambiguous.empty

    row = ambiguous.iloc[0]

    term = row["Preferred Label"]

    match = get_controlled_inventory_match(
        term,
        inventory,
    )

    assert match is not None
    assert (
        match["controlled_decision"]
        == "AMBIGUOUS"
    )


def test_controlled_inventory_ambiguity_blocks_known_similarity_selection():
    """
    Even when one candidate has perfect similarity and lexical
    matching, an AMBIGUOUS controlled decision must remain unresolved.
    """

    inventory = load_controlled_inventory()

    ambiguous = inventory[
        inventory["controlled_decision"] == "AMBIGUOUS"
    ]

    assert not ambiguous.empty

    row = ambiguous.iloc[0]

    term = row["Preferred Label"]

    candidate_ids = str(
        row["candidate_meddra_id"]
    ).split("|")

    candidate_ids = [
        item.strip()
        for item in candidate_ids
        if item.strip()
    ]

    assert len(candidate_ids) >= 2

    # Deliberately make the first candidate appear maximally strong.
    candidates = [
        candidate(
            preferred_label=term,
            similarity_score=1.0,
            meddra_id=candidate_ids[0],
        ),
        candidate(
            preferred_label=term,
            similarity_score=0.99,
            meddra_id=candidate_ids[1],
        ),
    ]

    result = decide(
        term,
        candidates,
    )

    assert result["status"] == "AMBIGUOUS"
    assert result["result"] is None