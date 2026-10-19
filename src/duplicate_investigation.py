"""
PV_NarrativeGuard - RAG Duplicate Investigation

Purpose:
    Generate a grounded, assistive investigation for a candidate
    duplicate and return structured evidence that can be consumed
    by both the backend pipeline and the Streamlit UI.

IMPORTANT:
    The AI investigation does NOT make a final duplicate decision.
    Final duplicate determination remains with a qualified
    pharmacovigilance professional.
"""

from pathlib import Path
import json
import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from rag_retriever import retrieve


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

HISTORICAL_FILE = DATA_DIR / "existing_icsrs.csv"
INCOMING_FILE = DATA_DIR / "incoming_icsrs.csv"
COMBINED_FILE = OUTPUT_DIR / "combined_duplicate_results.csv"

INVESTIGATION_FILE = (
    OUTPUT_DIR / "duplicate_investigations.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not configured in the .env file."
    )

client = Groq(
    api_key=GROQ_API_KEY
)

MODEL_NAME = "openai/gpt-oss-20b"


# ============================================================
# DATA HELPERS
# ============================================================

def load_case_data():
    """Load incoming and historical ICSR datasets."""

    incoming_df = pd.read_csv(
        INCOMING_FILE
    )

    historical_df = pd.read_csv(
        HISTORICAL_FILE
    )

    return incoming_df, historical_df


def get_case(case_id, dataframe):
    """Return one case by case_id."""

    rows = dataframe[
        dataframe["case_id"].astype(str)
        == str(case_id)
    ]

    if rows.empty:
        raise ValueError(
            f"Case not found: {case_id}"
        )

    return rows.iloc[0]


def format_value(value):
    if pd.isna(value):
        return "Not available"

    return str(value)


def format_case(case):
    """Format an ICSR into grounded evidence text."""

    return f"""
Case ID: {format_value(case.get("case_id"))}
Patient age: {format_value(case.get("patient_age"))}
Patient sex: {format_value(case.get("patient_sex"))}
Country: {format_value(case.get("country"))}
Suspect drug: {format_value(case.get("suspect_drug"))}
Indication: {format_value(case.get("indication"))}
Adverse event: {format_value(case.get("adverse_event"))}
Event date: {format_value(case.get("event_date"))}
Seriousness: {format_value(case.get("seriousness"))}
Outcome: {format_value(case.get("outcome"))}
Reporter type: {format_value(case.get("reporter_type"))}
Description: {format_value(case.get("description"))}
""".strip()


# ============================================================
# CANDIDATE DATA
# ============================================================

def get_candidate_scores(
    incoming_case_id,
    candidate_case_id,
):
    """Get rule and semantic scores for one candidate pair."""

    if not COMBINED_FILE.exists():
        raise FileNotFoundError(
            f"Combined duplicate results not found: "
            f"{COMBINED_FILE}"
        )

    combined_df = pd.read_csv(
        COMBINED_FILE
    )

    rows = combined_df[
        (
            combined_df["incoming_case_id"].astype(str)
            == str(incoming_case_id)
        )
        &
        (
            combined_df["candidate_case_id"].astype(str)
            == str(candidate_case_id)
        )
    ]

    if rows.empty:
        raise ValueError(
            f"Candidate pair not found: "
            f"{incoming_case_id} vs {candidate_case_id}"
        )

    row = rows.iloc[0]

    return {
        "rule_score": float(row["rule_score"]),
        "semantic_score": float(row["semantic_score"]),
        "combined_score": float(
            row.get("combined_score", 0)
        ),
        "classification": str(
            row.get("classification", "")
        ),
    }


# ============================================================
# STRUCTURED DIFFERENCES
# ============================================================

COMPARISON_FIELDS = [
    ("patient_age", "Patient age"),
    ("patient_sex", "Patient sex"),
    ("country", "Country"),
    ("suspect_drug", "Suspect drug"),
    ("indication", "Indication"),
    ("adverse_event", "Adverse event"),
    ("event_date", "Event date"),
    ("seriousness", "Seriousness"),
    ("outcome", "Outcome"),
    ("reporter_type", "Reporter type"),
]


def values_equal(value_a, value_b):
    if pd.isna(value_a) and pd.isna(value_b):
        return True

    if pd.isna(value_a) or pd.isna(value_b):
        return False

    return str(value_a).strip().lower() == str(
        value_b
    ).strip().lower()


def build_structured_comparison(
    incoming_case,
    historical_case,
):
    """Create explicit similarities and differences."""

    similarities = []
    differences = []

    for field, label in COMPARISON_FIELDS:

        incoming_value = incoming_case.get(field)
        historical_value = historical_case.get(field)

        if values_equal(
            incoming_value,
            historical_value,
        ):

            similarities.append({
                "field": field,
                "label": label,
                "incoming": format_value(incoming_value),
                "historical": format_value(
                    historical_value
                ),
            })

        else:

            differences.append({
                "field": field,
                "label": label,
                "incoming": format_value(incoming_value),
                "historical": format_value(
                    historical_value
                ),
            })

    return similarities, differences


# ============================================================
# AI INVESTIGATION
# ============================================================

def build_prompt(
    incoming_case,
    historical_case,
    scores,
    retrieved_context,
):
    return f"""
You are assisting a qualified pharmacovigilance professional
with duplicate-case assessment.

Your task is to provide a GROUNDED DUPLICATE INVESTIGATION.

IMPORTANT RULES:
1. Use only information explicitly present in the incoming case,
   historical candidate, similarity scores, and retrieved PV guidance.
2. Do not invent patient details, clinical facts, or missing information.
3. Do not make a final duplicate/non-duplicate decision.
4. High semantic similarity does NOT prove duplication.
5. Explicit differences must be clearly identified.
6. Explain why qualified human PV review is required.
7. If something cannot be established, say:
   "Cannot be determined from available data."

INCOMING ICSR:
{format_case(incoming_case)}

HISTORICAL CANDIDATE:
{format_case(historical_case)}

SIMILARITY SCORES:
Rule-based score: {scores["rule_score"]:.2f}/100
Semantic score: {scores["semantic_score"]:.4f}
Combined score: {scores["combined_score"]:.4f}
Classification: {scores["classification"]}

RETRIEVED PV GUIDANCE:
{retrieved_context}

Provide the response using exactly these sections:

1. Similarity evidence
2. Differences
3. Evidence interpretation
4. Human review recommendation

The final section must clearly state that the final duplicate
determination remains with a qualified pharmacovigilance professional.
""".strip()


def call_llm(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content


# ============================================================
# PERSISTENCE
# ============================================================

def load_saved_investigations():
    if not INVESTIGATION_FILE.exists():
        return {}

    try:
        with open(
            INVESTIGATION_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}


def save_investigation(result):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = load_saved_investigations()

    key = (
        f'{result["incoming_case_id"]}'
        f'__'
        f'{result["candidate_case_id"]}'
    )

    saved[key] = result

    with open(
        INVESTIGATION_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            saved,
            file,
            indent=2,
            ensure_ascii=False,
        )


def get_saved_investigation(
    incoming_case_id,
    candidate_case_id,
):
    saved = load_saved_investigations()

    key = (
        f"{incoming_case_id}"
        f"__"
        f"{candidate_case_id}"
    )

    return saved.get(key)


# ============================================================
# MAIN INVESTIGATION FUNCTION
# ============================================================

def investigate(
    incoming_case_id,
    candidate_case_id,
    force_refresh=False,
):
    """
    Run or retrieve a duplicate investigation.

    Returns a dictionary containing:
        - case IDs
        - similarity scores
        - structured similarities
        - structured differences
        - retrieved guidance
        - AI investigation
        - human-review requirement

    The result is persisted so Streamlit can display it
    without repeatedly calling the LLM.
    """

    if not force_refresh:

        saved = get_saved_investigation(
            incoming_case_id,
            candidate_case_id,
        )

        if saved is not None:

            print(
                f"[INVESTIGATION] Using saved result for "
                f"{incoming_case_id} vs {candidate_case_id}"
            )

            return saved

    incoming_df, historical_df = load_case_data()

    incoming_case = get_case(
        incoming_case_id,
        incoming_df,
    )

    historical_case = get_case(
        candidate_case_id,
        historical_df,
    )

    scores = get_candidate_scores(
        incoming_case_id,
        candidate_case_id,
    )

    similarities, differences = (
        build_structured_comparison(
            incoming_case,
            historical_case,
        )
    )

    query = f"""
PV duplicate investigation for:
Incoming case {incoming_case_id}
Historical candidate {candidate_case_id}

Suspect drug:
{format_value(incoming_case.get("suspect_drug"))}

Indication:
{format_value(incoming_case.get("indication"))}

Adverse event:
{format_value(incoming_case.get("adverse_event"))}

Event date:
{format_value(incoming_case.get("event_date"))}

Rule score:
{scores["rule_score"]}

Semantic score:
{scores["semantic_score"]}
""".strip()

    print("Retrieving relevant PV guidance...")

    retrieved = retrieve(
        query,
        top_k=3,
    )

    context_parts = []

    for item in retrieved:

        if isinstance(item, dict):

            chunk_id = item.get(
                "chunk_id",
                "Unknown",
            )

            source = item.get(
                "source",
                "Unknown source",
            )

            similarity = item.get(
                "similarity",
                None,
            )

            content = item.get(
                "content",
                "",
            )

            header = (
                f"{chunk_id} | {source}"
            )

            if similarity is not None:
                header += (
                    f" | similarity={float(similarity):.4f}"
                )

            context_parts.append(
                f"{header}\n{content}"
            )

        else:

            context_parts.append(
                str(item)
            )

    retrieved_context = "\n\n".join(
        context_parts
    )

    print(
        "Sending grounded case evidence + "
        "RAG context to Groq..."
    )

    prompt = build_prompt(
        incoming_case,
        historical_case,
        scores,
        retrieved_context,
    )

    investigation_text = call_llm(
        prompt
    )

    result = {
        "incoming_case_id":
            str(incoming_case_id),

        "candidate_case_id":
            str(candidate_case_id),

        "rule_score":
            scores["rule_score"],

        "semantic_score":
            scores["semantic_score"],

        "combined_score":
            scores["combined_score"],

        "classification":
            scores["classification"],

        "similarities":
            similarities,

        "differences":
            differences,

        "retrieved_guidance":
            retrieved_context,

        "investigation":
            investigation_text,

        "human_review_required":
            True,
    }

    save_investigation(
        result
    )

    print()
    print("=" * 70)
    print("GROUNDED DUPLICATE INVESTIGATION")
    print("=" * 70)
    print()
    print(
        f"Incoming case: {incoming_case_id}"
    )
    print(
        f"Historical candidate: {candidate_case_id}"
    )
    print(
        f"Rule score: {scores['rule_score']:.2f}/100"
    )
    print(
        f"Semantic score: {scores['semantic_score']:.4f}"
    )
    print()
    print(investigation_text)

    return result


if __name__ == "__main__":

    # Keep a simple direct test when this module is
    # executed manually.
    investigate(
        "INCOMING004",
        "CASE001",
    )
