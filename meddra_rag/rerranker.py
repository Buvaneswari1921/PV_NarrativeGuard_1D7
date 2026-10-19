import re
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = BASE_DIR / "vectorstore" / "oae_faiss.index"
METADATA_FILE = BASE_DIR / "vectorstore" / "oae_metadata.csv"

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load resources
# ---------------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_FILE))

print("Loading terminology metadata...")
metadata = pd.read_csv(
    METADATA_FILE,
    low_memory=False
)


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize_text(text):
    """Normalize text for terminology comparison."""

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove AE suffix used in OAE labels
    text = re.sub(r"\bae\b", " ", text)

    # Normalize separators
    text = text.replace("|", " ")

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s-]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    """Return normalized tokens."""

    stop_words = {
        "patient",
        "developed",
        "develop",
        "has",
        "have",
        "had",
        "with",
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "feels",
        "feel",
        "feeling",
        "experiencing",
        "experienced",
        "reported",
        "reports",
        "complains",
        "complaint",
        "severe",
        "mild",
        "moderate",
        "new"
    }

    return {
        word
        for word in normalize_text(text).split()
        if word not in stop_words
    }


# ---------------------------------------------------------
# Clinical synonym normalization
# ---------------------------------------------------------

CLINICAL_SYNONYMS = {

    "itching": {"itch", "pruritus"},
    "itch": {"itching", "pruritus"},

    "shortness": {"dyspnea", "dyspnoea"},
    "breathlessness": {"dyspnea", "dyspnoea"},
    "difficulty": {"dyspnea", "dyspnoea"},
    "breathing": {"dyspnea", "dyspnoea"},
    "air hunger": {"dyspnea", "dyspnoea"},

    "head pain": {"headache"},
    "headache": {"head pain"},

    "vomiting": {"emesis"},
    "emesis": {"vomiting"},

    "throwing up": {"vomiting", "emesis"},

    "lightheaded": {"lightheadedness"},
    "lightheadedness": {"lightheaded"},

    "loose stools": {"diarrhea", "diarrhoea"},
    "loose stool": {"diarrhea", "diarrhoea"},

}


def synonym_match_score(query, row):
    """
    Determine whether the user wording matches terminology
    or known clinical synonyms.
    """

    query_text = normalize_text(query)

    label = normalize_text(
        row.get("preferred_label", "")
    )

    synonyms = normalize_text(
        row.get("synonyms", "")
    )

    terminology = f"{label} {synonyms}"

    score = 0.0
    evidence = []

    # -----------------------------------------------------
    # Direct phrase matching
    # -----------------------------------------------------

    if label and label in query_text:
        score += 1.00
        evidence.append("preferred-term phrase match")

    # Check each synonym phrase
    if synonyms:

        synonym_list = [
            s.strip()
            for s in str(row.get("synonyms", "")).split("|")
            if s.strip()
        ]

        for synonym in synonym_list:

            synonym_norm = normalize_text(synonym)

            if synonym_norm and synonym_norm in query_text:
                score += 1.20
                evidence.append(
                    f"synonym match: {synonym}"
                )

    # -----------------------------------------------------
    # Clinical synonym expansion
    # -----------------------------------------------------

    query_tokens = tokenize(query)

    for phrase, alternatives in CLINICAL_SYNONYMS.items():

        if phrase in query_text:

            for alternative in alternatives:

                if alternative in terminology:

                    score += 1.50

                    evidence.append(
                        f"clinical synonym: {phrase} → {alternative}"
                    )

    return score, evidence


# ---------------------------------------------------------
# Token overlap
# ---------------------------------------------------------

def token_overlap_score(query, row):

    query_tokens = tokenize(query)

    label_tokens = set(
        normalize_text(
            row.get("preferred_label", "")
        ).split()
    )

    synonym_tokens = set(
        normalize_text(
            row.get("synonyms", "")
        ).split()
    )

    label_overlap = len(
        query_tokens & label_tokens
    )

    synonym_overlap = len(
        query_tokens & synonym_tokens
    )

    return (
        label_overlap * 0.20
        + synonym_overlap * 0.30
    )


# ---------------------------------------------------------
# Specificity detection
# ---------------------------------------------------------

SPECIFICITY_TERMS = {

    "cervicogenic",
    "postural",
    "procedural",
    "vascular",
    "premenstrual",
    "neonatal",
    "congenital",
    "drug induced",
    "drug-induced",
    "radiation",
    "traumatic",
    "infectious",
    "allergic",
    "occupational",
    "projectile",
    "faecal",
    "fecal",
    "herpetiformis",
    "paralysis",
    "pregnancy",
}


def specificity_penalty(query, label):

    query_text = normalize_text(query)
    label_text = normalize_text(label)

    penalty = 0.0
    evidence = []

    for term in SPECIFICITY_TERMS:

        if term in label_text:

            if term not in query_text:

                penalty += 0.50

                evidence.append(
                    f"unsupported specificity: {term}"
                )

    return penalty, evidence


# ---------------------------------------------------------
# General concept preference
# ---------------------------------------------------------

def general_concept_bonus(query, row):

    """
    Prefer a general concept when the user gives only
    a general symptom and does not specify a subtype.
    """

    label = normalize_text(
        row.get("preferred_label", "")
    )

    bonus = 0.0

    # General headache
    if label == "headache":
        if "head pain" in normalize_text(query):
            bonus += 1.00

    # General pruritus
    if label == "pruritus":
        if "itch" in normalize_text(query) or \
           "itching" in normalize_text(query):
            bonus += 1.00

    # General dyspnea
    if label == "dyspnea":
        query_text = normalize_text(query)

        if (
            "difficulty breathing" in query_text
            or "shortness of breath" in query_text
            or "breathlessness" in query_text
        ):
            bonus += 1.00

    # General vomiting
    if label == "vomiting":
        query_text = normalize_text(query)

        if (
            "vomiting" in query_text
            or "throwing up" in query_text
        ):
            bonus += 1.00

    return bonus


# ---------------------------------------------------------
# Retrieve and rerank
# ---------------------------------------------------------

def retrieve_and_rerank(
    query,
    top_k=20,
    final_k=5
):

    # -----------------------------------------------------
    # Semantic retrieval
    # -----------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    semantic_scores, indices = index.search(
        query_embedding,
        top_k
    )

    candidates = []

    # -----------------------------------------------------
    # Candidate scoring
    # -----------------------------------------------------

    for semantic_score, idx in zip(
        semantic_scores[0],
        indices[0]
    ):

        row = metadata.iloc[idx]

        semantic_score = float(
            semantic_score
        )

        synonym_score, synonym_evidence = (
            synonym_match_score(
                query,
                row
            )
        )

        token_score = token_overlap_score(
            query,
            row
        )

        specificity_score, specificity_evidence = (
            specificity_penalty(
                query,
                row["preferred_label"]
            )
        )

        general_bonus = general_concept_bonus(
            query,
            row
        )

        # -------------------------------------------------
        # Final score
        # -------------------------------------------------

        final_score = (
            semantic_score
            + synonym_score
            + token_score
            + general_bonus
            - specificity_score
        )

        candidates.append({

            "semantic_score": semantic_score,

            "synonym_score": synonym_score,

            "token_score": token_score,

            "general_bonus": general_bonus,

            "specificity_penalty": specificity_score,

            "final_score": final_score,

            "preferred_label": row[
                "preferred_label"
            ],

            "meddra_id": row[
                "meddra_id"
            ],

            "oae_id": row[
                "oae_id"
            ],

            "synonyms": row.get(
                "synonyms",
                ""
            ),

            "evidence":
                synonym_evidence
                + specificity_evidence
        })

    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return candidates[:final_k]


# ---------------------------------------------------------
# Test display
# ---------------------------------------------------------

def test_query(query):

    print("\n" + "=" * 110)
    print(f"USER INPUT: {query}")
    print("=" * 110)

    results = retrieve_and_rerank(
        query,
        top_k=20,
        final_k=5
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(f"\nRank {rank}")

        print(
            f"Final Score        : "
            f"{result['final_score']:.4f}"
        )

        print(
            f"Semantic Score     : "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"Synonym Score      : "
            f"{result['synonym_score']:.4f}"
        )

        print(
            f"Token Score        : "
            f"{result['token_score']:.4f}"
        )

        print(
            f"General Bonus      : "
            f"{result['general_bonus']:.4f}"
        )

        print(
            f"Specificity Penalty: "
            f"{result['specificity_penalty']:.4f}"
        )

        print(
            f"Preferred Label    : "
            f"{result['preferred_label']}"
        )

        print(
            f"MedDRA ID          : "
            f"{result['meddra_id']}"
        )

        print(
            f"OAE ID             : "
            f"{result['oae_id']}"
        )

        print(
            f"Synonyms           : "
            f"{result['synonyms']}"
        )

        if result["evidence"]:

            print(
                "Evidence           : "
                + "; ".join(
                    result["evidence"]
                )
            )


# ---------------------------------------------------------
# Test cases
# ---------------------------------------------------------

TEST_QUERIES = [

    "patient developed severe head pain",

    "patient feels like vomiting",

    "patient has loose stools",

    "patient feels lightheaded",

    "patient developed skin itching",

    "patient has difficulty breathing",

]


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":

    for query in TEST_QUERIES:

        test_query(query)

    print(
        "\n"
        + "=" * 110
    )

    print(
        "DOMAIN-AWARE RETRIEVAL TEST COMPLETE"
    )

    print(
        "=" * 110
    )