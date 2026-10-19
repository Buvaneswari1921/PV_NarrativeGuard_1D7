import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = BASE_DIR / "vectorstore" / "oae_faiss.index"
METADATA_FILE = BASE_DIR / "vectorstore" / "oae_metadata.csv"

OUTPUT_FILE = BASE_DIR / "data" / "rag_evaluation_results.csv"

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TOP_K = 5

# Number of concepts to evaluate.
# Set to None to evaluate all 7,425 concepts.
MAX_TESTS = None


# ---------------------------------------------------------
# Load resources
# ---------------------------------------------------------

print("=" * 80)
print("PV NARRATIVEGUARD - RAG EVALUATION")
print("=" * 80)

print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_FILE))

print("Loading metadata...")
metadata = pd.read_csv(
    METADATA_FILE,
    low_memory=False
)

print(f"\nTotal concepts available: {len(metadata)}")


# ---------------------------------------------------------
# Prepare evaluation cases
# ---------------------------------------------------------

evaluation_data = metadata.copy()

if MAX_TESTS is not None:
    evaluation_data = evaluation_data.head(MAX_TESTS)

print(
    f"Concepts selected for evaluation: "
    f"{len(evaluation_data)}"
)


# ---------------------------------------------------------
# Create evaluation queries
# ---------------------------------------------------------

def build_query(row):
    """
    Use the preferred terminology and synonyms as the
    reference query.

    This establishes a controlled baseline for the
    retrieval system before we introduce human-generated
    paraphrases.
    """

    preferred = str(
        row.get("preferred_label", "")
    ).strip()

    synonyms = row.get("synonyms", "")

    if pd.isna(synonyms):
        synonyms = ""

    synonyms = str(synonyms).strip()

    if synonyms:
        return f"{preferred}. Also known as {synonyms}."

    return preferred


evaluation_data["evaluation_query"] = (
    evaluation_data.apply(
        build_query,
        axis=1
    )
)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

results = []

top1_correct = 0
top3_correct = 0
top5_correct = 0


print("\nStarting retrieval evaluation...")
print("This may take several minutes.\n")


for _, row in tqdm(
    evaluation_data.iterrows(),
    total=len(evaluation_data),
    desc="Evaluating concepts"
):

    query = row["evaluation_query"]

    target_oae_id = str(
        row["oae_id"]
    )

    target_meddra_id = str(
        row["meddra_id"]
    )

    target_label = str(
        row["preferred_label"]
    )

    # -----------------------------------------------------
    # Encode query
    # -----------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # -----------------------------------------------------
    # Retrieve candidates
    # -----------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        TOP_K
    )

    retrieved_oae_ids = []

    retrieved_labels = []

    retrieved_meddra_ids = []

    retrieved_scores = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        candidate = metadata.iloc[idx]

        retrieved_oae_ids.append(
            str(candidate["oae_id"])
        )

        retrieved_labels.append(
            str(candidate["preferred_label"])
        )

        retrieved_meddra_ids.append(
            str(candidate["meddra_id"])
        )

        retrieved_scores.append(
            float(score)
        )

    # -----------------------------------------------------
    # Determine rank
    # -----------------------------------------------------

    rank = None

    for position, oae_id in enumerate(
        retrieved_oae_ids,
        start=1
    ):

        if oae_id == target_oae_id:

            rank = position
            break

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    is_top1 = rank == 1

    is_top3 = (
        rank is not None
        and rank <= 3
    )

    is_top5 = (
        rank is not None
        and rank <= 5
    )

    if is_top1:
        top1_correct += 1

    if is_top3:
        top3_correct += 1

    if is_top5:
        top5_correct += 1

    # -----------------------------------------------------
    # Store result
    # -----------------------------------------------------

    results.append({

        "target_oae_id":
            target_oae_id,

        "target_preferred_label":
            target_label,

        "target_meddra_id":
            target_meddra_id,

        "evaluation_query":
            query,

        "retrieved_rank":
            rank,

        "top1_correct":
            is_top1,

        "top3_correct":
            is_top3,

        "top5_correct":
            is_top5,

        "rank1_label":
            retrieved_labels[0]
            if len(retrieved_labels) > 0
            else "",

        "rank1_meddra_id":
            retrieved_meddra_ids[0]
            if len(retrieved_meddra_ids) > 0
            else "",

        "rank1_score":
            retrieved_scores[0]
            if len(retrieved_scores) > 0
            else None,

        "rank2_label":
            retrieved_labels[1]
            if len(retrieved_labels) > 1
            else "",

        "rank2_score":
            retrieved_scores[1]
            if len(retrieved_scores) > 1
            else None,

        "rank3_label":
            retrieved_labels[2]
            if len(retrieved_labels) > 2
            else "",

        "rank3_score":
            retrieved_scores[2]
            if len(retrieved_scores) > 2
            else None,

        "rank4_label":
            retrieved_labels[3]
            if len(retrieved_labels) > 3
            else "",

        "rank4_score":
            retrieved_scores[3]
            if len(retrieved_scores) > 3
            else None,

        "rank5_label":
            retrieved_labels[4]
            if len(retrieved_labels) > 4
            else "",

        "rank5_score":
            retrieved_scores[4]
            if len(retrieved_scores) > 4
            else None,
    })


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

total = len(evaluation_data)

top1_accuracy = (
    top1_correct / total
    if total
    else 0
)

top3_accuracy = (
    top3_correct / total
    if total
    else 0
)

top5_accuracy = (
    top5_correct / total
    if total
    else 0
)


# ---------------------------------------------------------
# Save detailed results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print("\n")
print("=" * 80)
print("RAG EVALUATION COMPLETE")
print("=" * 80)

print(
    f"\nTotal concepts tested : {total}"
)

print(
    f"Top-1 Accuracy        : "
    f"{top1_accuracy:.2%}"
)

print(
    f"Top-3 Accuracy        : "
    f"{top3_accuracy:.2%}"
)

print(
    f"Top-5 Accuracy        : "
    f"{top5_accuracy:.2%}"
)

print(
    f"\nDetailed results saved to:"
)

print(
    OUTPUT_FILE
)


# ---------------------------------------------------------
# Failure summary
# ---------------------------------------------------------

failures = results_df[
    results_df["top5_correct"] == False
]

print(
    f"\nConcepts NOT found in Top-5: "
    f"{len(failures)}"
)

if len(failures) > 0:

    print("\nSample retrieval failures:")

    print(
        failures[
            [
                "target_preferred_label",
                "target_meddra_id",
                "retrieved_rank",
                "rank1_label",
                "rank1_score",
                "rank2_label",
                "rank2_score",
                "rank3_label",
                "rank3_score"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


print("\n" + "=" * 80)