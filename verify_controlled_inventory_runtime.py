from pathlib import Path
import pandas as pd

from meddra_rag.terminology_engine import (
    load_controlled_inventory,
    get_controlled_inventory_match,
    decide,
    normalize_text,
)


BASE_DIR = Path(__file__).resolve().parent

INVENTORY_FILE = (
    BASE_DIR
    / "meddra_rag"
    / "data"
    / "OAE_MedDRA_Deterministic_Decision_Inventory.csv"
)

METADATA_FILE = (
    BASE_DIR
    / "meddra_rag"
    / "vectorstore"
    / "oae_metadata.csv"
)


def make_candidate(row):
    """
    Convert an OAE metadata row into the candidate structure
    expected by terminology_engine.decide().
    """

    result = dict(row)

    result["preferred_label"] = str(
        row.get(
            "preferred_label",
            row.get("Preferred Label", ""),
        )
    )

    result["synonyms"] = str(
        row.get(
            "synonyms",
            row.get("Synonyms", ""),
        )
    )

    result["meddra_id"] = str(
        row.get(
            "meddra_id",
            row.get(
                "MedDRA ID",
                row.get("candidate_meddra_id", ""),
            ),
        )
    )

    result["similarity_score"] = 1.0

    return result


def find_candidate(
    metadata,
    meddra_id,
    preferred_label,
):
    """
    Find the corresponding metadata candidate using either
    MedDRA ID or normalized preferred label.
    """

    normalized_target = normalize_text(
        preferred_label
    )

    for _, row in metadata.iterrows():

        candidate = make_candidate(row)

        if (
            str(candidate["meddra_id"]).strip()
            == str(meddra_id).strip()
        ):
            return candidate

    for _, row in metadata.iterrows():

        candidate = make_candidate(row)

        if (
            normalize_text(
                candidate["preferred_label"]
            )
            == normalized_target
        ):
            return candidate

    return None


def main():

    print("=" * 100)
    print(
        "PV NarrativeGuard - Controlled Decision Inventory Runtime Verification"
    )
    print("=" * 100)

    inventory = load_controlled_inventory()

    if inventory.empty:
        raise RuntimeError(
            "Controlled decision inventory could not be loaded."
        )

    metadata = pd.read_csv(
        METADATA_FILE,
        low_memory=False,
    )

    print()
    print("INVENTORY")
    print("-" * 100)
    print(f"Rows: {len(inventory)}")

    print(
        inventory[
            "controlled_decision"
        ].value_counts(
            dropna=False
        ).to_string()
    )

    # ========================================================
    # TEST 1 - SAFE_EXACT
    # ========================================================

    safe_exact = inventory[
        inventory[
            "controlled_decision"
        ]
        == "SAFE_EXACT"
    ]

    if safe_exact.empty:
        raise RuntimeError(
            "No SAFE_EXACT records found."
        )

    row = safe_exact.iloc[0]

    term = row["Preferred Label"]
    meddra_id = row["candidate_meddra_id"]

    print()
    print("=" * 100)
    print("TEST 1 - SAFE_EXACT")
    print("=" * 100)
    print(f"Term       : {term}")
    print(f"MedDRA ID  : {meddra_id}")

    controlled_match = get_controlled_inventory_match(
        term,
        inventory,
    )

    print(
        "Inventory decision:",
        controlled_match.get(
            "controlled_decision"
        )
        if controlled_match
        else None,
    )

    candidate = find_candidate(
        metadata,
        meddra_id,
        term,
    )

    if candidate is None:
        raise RuntimeError(
            "Could not construct SAFE_EXACT candidate."
        )

    result = decide(
        term,
        [candidate],
    )

    print(
        "Engine status   :",
        result["status"],
    )

    print(
        "Match method    :",
        (
            result["result"] or {}
        ).get(
            "match_method"
        ),
    )

    print(
        "Returned ID     :",
        (
            result["result"] or {}
        ).get(
            "meddra_id"
        ),
    )

    assert result["status"] == "SUPPORTED"
    assert (
        (
            result["result"] or {}
        ).get(
            "match_method"
        )
        == "controlled_inventory"
    )

        # ========================================================
    # TEST 2 - AMBIGUOUS
    # ========================================================

    ambiguous = inventory[
        inventory[
            "controlled_decision"
        ]
        == "AMBIGUOUS"
    ]

    if ambiguous.empty:
        raise RuntimeError(
            "No AMBIGUOUS records found."
        )

    row = ambiguous.iloc[0]

    term = str(
        row["Preferred Label"]
    ).strip()

    print()
    print("=" * 100)
    print("TEST 2 - AMBIGUOUS")
    print("=" * 100)
    print(f"Term       : {term}")

    print(
        "Candidate IDs:",
        row.get(
            "label_candidate_ids",
            "",
        ),
    )

    controlled_match = (
        get_controlled_inventory_match(
            term,
            inventory,
        )
    )

    print(
        "Inventory decision:",
        (
            controlled_match or {}
        ).get(
            "controlled_decision"
        ),
    )

    # --------------------------------------------------------
    # Build candidates directly from metadata.
    #
    # Important:
    # The metadata itself may contain pipe-delimited MedDRA
    # IDs. We therefore construct candidates from the metadata
    # rows matching the normalized preferred label.
    # --------------------------------------------------------

    normalized_term = normalize_text(
        term
    )

    candidates = []

    for _, metadata_row in metadata.iterrows():

        candidate = make_candidate(
            metadata_row
        )

        if (
            normalize_text(
                candidate["preferred_label"]
            )
            != normalized_term
        ):
            continue

        candidate_ids = [
            x.strip()
            for x in str(
                candidate["meddra_id"]
            ).split("|")
            if x.strip()
        ]

        # Keep the metadata row only when it represents
        # a distinct MedDRA candidate.
        if candidate_ids:

            candidates.append(
                candidate
            )

    print(
        "Constructed candidates:",
        len(candidates),
    )

    for candidate in candidates:

        print(
            "  -",
            candidate.get(
                "meddra_id"
            ),
            "|",
            candidate.get(
                "preferred_label"
            ),
        )

    if len(candidates) < 2:
        raise RuntimeError(
            "Could not construct AMBIGUOUS candidates "
            "from metadata."
        )

    result = decide(
        term,
        candidates,
    )

    print(
        "Engine status:",
        result["status"],
    )

    print(
        "Engine result:",
        result.get(
            "result"
        ),
    )

    assert result["status"] == "AMBIGUOUS"

    # ========================================================
    # TEST 3 - UNKNOWN
    # ========================================================

    unknown_term = (
        "completely unknown adverse event xyz"
    )

    print()
    print("=" * 100)
    print("TEST 3 - UNKNOWN TERM")
    print("=" * 100)
    print(f"Term: {unknown_term}")

    result = decide(
        unknown_term,
        [],
    )

    print(
        "Engine status:",
        result["status"],
    )

    assert result["status"] == "OUT_OF_CONTEXT"

    # ========================================================
    # TEST 4 - EXISTING KNOWN MAPPING
    # ========================================================

    lightheaded = inventory[
        inventory[
            "Preferred Label"
        ].str.contains(
            "lightheadedness",
            case=False,
            na=False,
        )
    ]

    if not lightheaded.empty:

        row = lightheaded.iloc[0]

        candidate = find_candidate(
            metadata,
            row["candidate_meddra_id"],
            row["Preferred Label"],
        )

        if candidate:

            print()
            print("=" * 100)
            print("TEST 4 - EXISTING KNOWN MAPPING")
            print("=" * 100)

            query = (
                "patient feels lightheaded"
            )

            result = decide(
                query,
                [candidate],
            )

            print(
                "Query:",
                query,
            )

            print(
                "Engine status:",
                result["status"],
            )

            print(
                "Match method:",
                (
                    result["result"] or {}
                ).get(
                    "match_method"
                ),
            )

            assert (
                result["status"]
                == "SUPPORTED"
            )

    print()
    print("=" * 100)
    print("RUNTIME VERIFICATION PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()