"""
PV_NarrativeGuard
Deterministic OAE -> MedDRA candidate mapping inventory

Purpose
-------
Build a complete, review-first inventory from the 7,425 OAE records that
already contain a MedDRA ID.

IMPORTANT
---------
This script does NOT automatically accept any mapping.
Every candidate is classified as:
    SAFE
    AMBIGUOUS
    NEEDS_REVIEW

The original OAE.csv is never modified.

Expected source:
    D:\PV_NarrativeGuard\meddra_rag\data\OAE.csv

Outputs:
    OAE_MedDRA_Deterministic_Candidate_Inventory.csv
    OAE_MedDRA_Deterministic_Summary.csv
    OAE_MedDRA_Ambiguous.csv
    OAE_MedDRA_Needs_Review.csv
"""

from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = Path(r"D:\PV_NarrativeGuard")
DATA_DIR = PROJECT_ROOT / "meddra_rag" / "data"
SOURCE_FILE = DATA_DIR / "OAE.csv"

OUTPUT_INVENTORY = DATA_DIR / "OAE_MedDRA_Deterministic_Candidate_Inventory.csv"
OUTPUT_SUMMARY = DATA_DIR / "OAE_MedDRA_Deterministic_Summary.csv"
OUTPUT_AMBIGUOUS = DATA_DIR / "OAE_MedDRA_Ambiguous.csv"
OUTPUT_REVIEW = DATA_DIR / "OAE_MedDRA_Needs_Review.csv"


def normalize_text(value) -> str:
    """Normalize terminology text for deterministic comparison."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    # Remove the OAE suffix used in Preferred Label values.
    text = re.sub(r"\s+ae\s*$", "", text)

    # Normalize punctuation/spacing.
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_med_dra_ids(value):
    """
    Preserve every MedDRA ID appearing in a source cell.

    Some OAE rows contain multiple IDs separated by '|'. Those are retained
    as separate candidate IDs and flagged for review.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    parts = [p.strip() for p in re.split(r"[|;,]", text) if p.strip()]
    return parts


def split_synonyms(value):
    """Split OAE synonym fields into individual normalized terms."""
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    # Existing OAE exports commonly use | between synonyms.
    parts = re.split(r"\s*\|\s*|\s*;\s*", text)

    return [normalize_text(p) for p in parts if normalize_text(p)]


def find_column(df, candidates, required=True):
    """Find a column case-insensitively."""
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]

    if required:
        raise ValueError(
            f"Required column not found. Expected one of: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def classify_mapping(
    row,
    label_to_ids,
    synonym_to_ids,
):
    """
    Deterministic review classification.

    SAFE
        - exactly one MedDRA ID in the row
        - normalized preferred label resolves to exactly one MedDRA ID
        - no conflicting synonym mapping

    AMBIGUOUS
        - the normalized preferred label maps to >1 MedDRA IDs, OR
        - a synonym resolves to >1 MedDRA IDs

    NEEDS_REVIEW
        - row contains multiple MedDRA IDs, OR
        - mapping evidence is incomplete/inconsistent
    """
    ids = row["_meddra_ids"]
    label_key = row["_normalized_label"]
    synonym_keys = row["_normalized_synonyms"]

    if not ids:
        return "NEEDS_REVIEW", "No MedDRA ID present"

    if len(ids) > 1:
        return (
            "NEEDS_REVIEW",
            "Source row contains multiple MedDRA IDs",
        )

    row_id = ids[0]

    label_ids = label_to_ids.get(label_key, set()) if label_key else set()

    conflicting_synonym_ids = set()
    matched_synonyms = []

    for synonym in synonym_keys:
        synonym_ids = synonym_to_ids.get(synonym, set())

        if synonym_ids:
            matched_synonyms.append(synonym)

        if len(synonym_ids) > 1:
            conflicting_synonym_ids.update(synonym_ids)

    if len(label_ids) > 1:
        return (
            "AMBIGUOUS",
            "Same normalized OAE label maps to multiple MedDRA IDs",
        )

    if conflicting_synonym_ids:
        return (
            "AMBIGUOUS",
            "Synonym maps to multiple MedDRA IDs",
        )

    if label_ids and row_id not in label_ids:
        return (
            "NEEDS_REVIEW",
            "Row MedDRA ID conflicts with normalized-label mapping",
        )

    if not label_key:
        return (
            "NEEDS_REVIEW",
            "Preferred Label is empty",
        )

    # A single source ID with a unique normalized label is a candidate
    # deterministic mapping. It remains a candidate until human review.
    return (
        "SAFE",
        "Unique deterministic label-to-MedDRA candidate",
    )


def main():
    print("=" * 100)
    print("PV NarrativeGuard - Deterministic OAE -> MedDRA Candidate Inventory")
    print("=" * 100)

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"OAE dataset not found:\n{SOURCE_FILE}\n\n"
            "Place OAE.csv in the expected project data directory and rerun."
        )

    df = pd.read_csv(SOURCE_FILE, low_memory=False)

    print(f"Source rows                 : {len(df):,}")

    meddra_col = find_column(
        df,
        ["has_MedDRA_id", "MedDRA ID", "meddra_id"],
    )

    label_col = find_column(
        df,
        ["Preferred Label", "preferred_label", "label"],
    )

    synonym_col = find_column(
        df,
        ["Synonyms", "synonyms"],
        required=False,
    )

    oae_col = find_column(
        df,
        ["OAE ID", "oae_id", "Class ID", "class_id"],
        required=False,
    )

    # Work only from the 7,425 source records that contain MedDRA linkage.
    mapped = df[
        df[meddra_col].notna()
        & df[meddra_col].astype(str).str.strip().ne("")
        & df[meddra_col].astype(str).str.lower().ne("nan")
    ].copy()

    print(f"MedDRA-linked records       : {len(mapped):,}")

    mapped["_normalized_label"] = mapped[label_col].apply(normalize_text)

    if synonym_col:
        mapped["_normalized_synonyms"] = mapped[synonym_col].apply(
            split_synonyms
        )
    else:
        mapped["_normalized_synonyms"] = [[] for _ in range(len(mapped))]

    mapped["_meddra_ids"] = mapped[meddra_col].apply(split_med_dra_ids)

    # ------------------------------------------------------------------
    # Build deterministic lookup maps across the complete 7,425-record
    # MedDRA-linked reference set.
    # ------------------------------------------------------------------
    label_to_ids = {}
    synonym_to_ids = {}

    for _, row in mapped.iterrows():
        ids = set(row["_meddra_ids"])

        label = row["_normalized_label"]
        if label:
            label_to_ids.setdefault(label, set()).update(ids)

        for synonym in row["_normalized_synonyms"]:
            synonym_to_ids.setdefault(synonym, set()).update(ids)

    # ------------------------------------------------------------------
    # Classify every source record.
    # ------------------------------------------------------------------
    statuses = []
    reasons = []

    for _, row in mapped.iterrows():
        status, reason = classify_mapping(
            row,
            label_to_ids,
            synonym_to_ids,
        )
        statuses.append(status)
        reasons.append(reason)

    mapped["deterministic_status"] = statuses
    mapped["deterministic_reason"] = reasons

    mapped["meddra_id_count"] = mapped["_meddra_ids"].apply(len)
    mapped["normalized_label"] = mapped["_normalized_label"]

    mapped["label_candidate_ids"] = mapped["_normalized_label"].apply(
        lambda x: "|".join(sorted(label_to_ids.get(x, set())))
    )

    def synonym_candidate_ids(synonyms):
        ids = set()
        for synonym in synonyms:
            ids.update(synonym_to_ids.get(synonym, set()))
        return "|".join(sorted(ids))

    mapped["synonym_candidate_ids"] = mapped[
        "_normalized_synonyms"
    ].apply(synonym_candidate_ids)

    mapped["mapping_review_required"] = mapped[
        "deterministic_status"
    ].ne("SAFE")

    # Human-readable candidate fields.
    mapped["candidate_meddra_id"] = mapped[meddra_col].astype(str)

    # Keep source columns plus deterministic audit fields.
    helper_columns = [
        "_normalized_label",
        "_normalized_synonyms",
        "_meddra_ids",
    ]

    output = mapped.drop(columns=helper_columns, errors="ignore")

    # Stable, review-friendly ordering.
    preferred_order = [
        c for c in [
            oae_col,
            "Class ID",
            label_col,
            synonym_col,
            meddra_col,
            "normalized_label",
            "candidate_meddra_id",
            "deterministic_status",
            "deterministic_reason",
            "meddra_id_count",
            "label_candidate_ids",
            "synonym_candidate_ids",
            "mapping_review_required",
        ]
        if c and c in output.columns
    ]

    remaining = [c for c in output.columns if c not in preferred_order]
    output = output[preferred_order + remaining]

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    output.to_csv(
        OUTPUT_INVENTORY,
        index=False,
        encoding="utf-8-sig",
    )

    ambiguous = output[
        output["deterministic_status"] == "AMBIGUOUS"
    ].copy()

    needs_review = output[
        output["deterministic_status"] == "NEEDS_REVIEW"
    ].copy()

    ambiguous.to_csv(
        OUTPUT_AMBIGUOUS,
        index=False,
        encoding="utf-8-sig",
    )

    needs_review.to_csv(
        OUTPUT_REVIEW,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    status_counts = (
        output["deterministic_status"]
        .value_counts()
        .reindex(
            ["SAFE", "AMBIGUOUS", "NEEDS_REVIEW"],
            fill_value=0,
        )
    )

    summary_rows = [
        {
            "metric": "Source OAE rows",
            "value": len(df),
        },
        {
            "metric": "MedDRA-linked rows evaluated",
            "value": len(output),
        },
        {
            "metric": "SAFE candidates",
            "value": int(status_counts["SAFE"]),
        },
        {
            "metric": "AMBIGUOUS candidates",
            "value": int(status_counts["AMBIGUOUS"]),
        },
        {
            "metric": "NEEDS_REVIEW candidates",
            "value": int(status_counts["NEEDS_REVIEW"]),
        },
        {
            "metric": "Unique normalized labels",
            "value": len(label_to_ids),
        },
        {
            "metric": "Labels mapping to >1 MedDRA ID",
            "value": sum(
                1 for ids in label_to_ids.values()
                if len(ids) > 1
            ),
        },
        {
            "metric": "Synonyms mapping to >1 MedDRA ID",
            "value": sum(
                1 for ids in synonym_to_ids.values()
                if len(ids) > 1
            ),
        },
    ]

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("-" * 100)
    print("DETERMINISTIC MAPPING INVENTORY SUMMARY")
    print("-" * 100)

    for status in ["SAFE", "AMBIGUOUS", "NEEDS_REVIEW"]:
        count = int(status_counts[status])
        pct = (count / len(output) * 100) if len(output) else 0
        print(f"{status:<15}: {count:>6,} ({pct:6.2f}%)")

    print()
    print(f"Inventory saved : {OUTPUT_INVENTORY}")
    print(f"Summary saved   : {OUTPUT_SUMMARY}")
    print(f"Ambiguous saved : {OUTPUT_AMBIGUOUS}")
    print(f"Review saved    : {OUTPUT_REVIEW}")

    print()
    print("IMPORTANT:")
    print("SAFE means 'candidate deterministic mapping', NOT automatically accepted.")
    print("AMBIGUOUS and NEEDS_REVIEW must remain gated from automatic MedDRA decisions.")
    print("=" * 100)


if __name__ == "__main__":
    main()
