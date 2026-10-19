import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

RESULTS_FILE = (
    BASE_DIR
    / "data"
    / "rag_evaluation_results.csv"
)

FAILURE_FILE = (
    BASE_DIR
    / "data"
    / "rag_failure_analysis.csv"
)


print("=" * 90)
print("PV NARRATIVEGUARD - RAG FAILURE ANALYSIS")
print("=" * 90)

print("\nLoading evaluation results...")

df = pd.read_csv(
    RESULTS_FILE,
    low_memory=False
)

# ---------------------------------------------------------
# Identify Top-5 failures
# ---------------------------------------------------------

failures = df[
    df["top5_correct"] == False
].copy()

print(
    f"\nTotal concepts evaluated : {len(df)}"
)

print(
    f"Top-5 failures            : {len(failures)}"
)


# ---------------------------------------------------------
# Classify failure
# ---------------------------------------------------------

def classify_failure(row):

    target = str(
        row["target_preferred_label"]
    ).lower()

    rank1 = str(
        row["rank1_label"]
    ).lower()

    rank2 = str(
        row["rank2_label"]
    ).lower()

    rank3 = str(
        row["rank3_label"]
    ).lower()

    # -----------------------------------------------------
    # General concept appears inside a more specific term
    # -----------------------------------------------------

    target_core = (
        target
        .replace(" ae", "")
        .strip()
    )

    specific_match = (
        target_core in rank1
        or target_core in rank2
        or target_core in rank3
    )

    if specific_match:
        return "GENERAL_VS_SPECIFIC"

    # -----------------------------------------------------
    # Similar clinical neighborhood
    # -----------------------------------------------------

    target_words = set(
        target_core.split()
    )

    candidate_words = set(
        (
            rank1
            + " "
            + rank2
            + " "
            + rank3
        ).split()
    )

    overlap = len(
        target_words & candidate_words
    )

    if overlap > 0:
        return "SEMANTIC_NEAR_MISS"

    # -----------------------------------------------------
    # Otherwise
    # -----------------------------------------------------

    return "RETRIEVAL_MISS"


failures["failure_type"] = failures.apply(
    classify_failure,
    axis=1
)


# ---------------------------------------------------------
# Save failure report
# ---------------------------------------------------------

failures.to_csv(
    FAILURE_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\nFailure classification:")
print(
    failures["failure_type"]
    .value_counts()
)


print("\nDetailed failures:")
print("=" * 90)

for _, row in failures.iterrows():

    print(
        f"\nTarget : "
        f"{row['target_preferred_label']}"
    )

    print(
        f"MedDRA : "
        f"{row['target_meddra_id']}"
    )

    print(
        f"Failure : "
        f"{row['failure_type']}"
    )

    print(
        f"Rank 1 : "
        f"{row['rank1_label']} "
        f"({row['rank1_score']:.4f})"
    )

    print(
        f"Rank 2 : "
        f"{row['rank2_label']} "
        f"({row['rank2_score']:.4f})"
    )

    print(
        f"Rank 3 : "
        f"{row['rank3_label']} "
        f"({row['rank3_score']:.4f})"
    )

    print(
        f"Rank 4 : "
        f"{row['rank4_label']}"
    )

    print(
        f"Rank 5 : "
        f"{row['rank5_label']}"
    )


print("\n" + "=" * 90)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 90)

print(
    f"\nDetailed report saved to:\n"
    f"{FAILURE_FILE}"
)