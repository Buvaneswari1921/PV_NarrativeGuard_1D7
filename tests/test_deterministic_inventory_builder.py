"""
PV_NarrativeGuard - Deterministic Inventory Builder Unit Tests

Purpose
-------
Validate the deterministic classification rules used by
build_oae_deterministic_inventory.py without rebuilding the full
7,425-record production inventory.

These tests exercise the decision logic directly and keep the production
OAE.csv and generated inventory files untouched.
"""

from build_oae_deterministic_inventory import (
    classify_mapping,
    normalize_text,
)


def make_row(
    label="headache AE",
    meddra_ids=None,
    synonyms=None,
):
    return {
        "_normalized_label": normalize_text(label),
        "_normalized_synonyms": synonyms or [],
        "_meddra_ids": meddra_ids if meddra_ids is not None else ["10019211"],
    }


def test_unique_label_single_id_is_safe():
    row = make_row()

    status, reason = classify_mapping(
        row,
        {"headache": {"10019211"}},
        {},
    )

    assert status == "SAFE"
    assert "Unique deterministic" in reason


def test_multiple_ids_in_source_row_require_review():
    row = make_row(
        meddra_ids=["10021028", "10021027"],
    )

    status, reason = classify_mapping(
        row,
        {"hypomagnesemia": {"10021027", "10021028"}},
        {},
    )

    assert status == "NEEDS_REVIEW"
    assert reason == "Source row contains multiple MedDRA IDs"


def test_same_normalized_label_mapping_to_multiple_ids_is_ambiguous():
    row = make_row(
        label="hemoglobinuria AE",
        meddra_ids=["10019489"],
    )

    status, reason = classify_mapping(
        row,
        {"hemoglobinuria": {"10018906", "10019489"}},
        {},
    )

    assert status == "AMBIGUOUS"
    assert reason == "Same normalized OAE label maps to multiple MedDRA IDs"


def test_synonym_mapping_to_multiple_ids_is_ambiguous():
    row = make_row(
        label="atrial fibrillation AE",
        meddra_ids=["10003658"],
        synonyms=["atrial fibrillation"],
    )

    status, reason = classify_mapping(
        row,
        {"atrial fibrillation": {"10003658"}},
        {"atrial fibrillation": {"10003658", "10003662"}},
    )

    assert status == "AMBIGUOUS"
    assert reason == "Synonym maps to multiple MedDRA IDs"


def test_row_id_conflicting_with_unique_label_mapping_requires_review():
    row = make_row(
        label="headache AE",
        meddra_ids=["99999999"],
    )

    status, reason = classify_mapping(
        row,
        {"headache": {"10019211"}},
        {},
    )

    assert status == "NEEDS_REVIEW"
    assert reason == "Row MedDRA ID conflicts with normalized-label mapping"


def test_missing_meddra_id_requires_review():
    row = make_row(
        label="headache AE",
        meddra_ids=[],
    )

    status, reason = classify_mapping(
        row,
        {"headache": {"10019211"}},
        {},
    )

    assert status == "NEEDS_REVIEW"
    assert reason == "No MedDRA ID present"


def test_missing_preferred_label_requires_review():
    row = make_row(
        label="",
        meddra_ids=["10019211"],
    )

    status, reason = classify_mapping(
        row,
        {},
        {},
    )

    assert status == "NEEDS_REVIEW"
    assert reason == "Preferred Label is empty"


def test_unique_synonym_without_conflict_remains_safe():
    row = make_row(
        label="headache AE",
        meddra_ids=["10019211"],
        synonyms=["head pain"],
    )

    status, reason = classify_mapping(
        row,
        {"headache": {"10019211"}},
        {"head pain": {"10019211"}},
    )

    assert status == "SAFE"
    assert "Unique deterministic" in reason


def test_normalization_removes_ae_suffix_and_punctuation():
    assert normalize_text("Hemoglobinuria AE") == "hemoglobinuria"
    assert normalize_text("Black-Hairy Tongue AE") == "black hairy tongue"
