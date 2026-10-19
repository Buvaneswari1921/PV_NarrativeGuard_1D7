import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------
# 1. Load configuration
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY was not found in the .env file."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

EMBEDDING_MODEL = "openai/text-embedding-3-small"


# --------------------------------------------------
# 2. Create a text representation of an ICSR
# --------------------------------------------------

def create_case_text(case):
    return f"""
Patient: {case['patient_age']}-year-old {case['patient_sex']}
Country: {case['country']}
Suspect drug: {case['suspect_drug']}
Indication: {case['indication']}
Adverse event: {case['adverse_event']}
Event date: {case['event_date']}
Seriousness: {case['seriousness']}
Outcome: {case['outcome']}
Reporter: {case['reporter_type']}
Description: {case.get('description', '')}
""".strip()


# --------------------------------------------------
# 3. Generate embeddings
# --------------------------------------------------

def generate_embeddings(texts):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


# --------------------------------------------------
# 4. Calculate cosine similarity
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# --------------------------------------------------
# 5. Main workflow
# --------------------------------------------------

def main():

    print("Loading ICSR datasets...")

    historical = pd.read_csv(
        "data/existing_icsrs.csv"
    )

    incoming = pd.read_csv(
        "data/incoming_icsrs.csv"
    )

    validation = pd.read_csv(
        "outputs/validation_results.csv"
    )

    # Only process valid incoming ICSRs
    valid_case_ids = validation.loc[
        validation["is_valid"] == True,
        "case_id"
    ]

    incoming_valid = incoming[
        incoming["case_id"].isin(valid_case_ids)
    ].copy()

    print(f"Historical cases: {len(historical)}")
    print(f"Valid incoming cases: {len(incoming_valid)}")

    # --------------------------------------------------
    # Create text representations
    # --------------------------------------------------

    historical["case_text"] = historical.apply(
        create_case_text,
        axis=1
    )

    incoming_valid["case_text"] = incoming_valid.apply(
        create_case_text,
        axis=1
    )

    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    print("Generating historical case embeddings...")

    historical_embeddings = generate_embeddings(
        historical["case_text"].tolist()
    )

    print("Generating incoming case embeddings...")

    incoming_embeddings = generate_embeddings(
        incoming_valid["case_text"].tolist()
    )

    # --------------------------------------------------
    # Compare incoming cases with historical cases
    # --------------------------------------------------

    results = []

    for incoming_index, incoming_row in incoming_valid.iterrows():

        incoming_vector = incoming_embeddings[
            list(incoming_valid.index).index(incoming_index)
        ]

        similarities = []

        for historical_index, historical_row in historical.iterrows():

            historical_vector = historical_embeddings[
                historical_index
            ]

            score = cosine_similarity(
                incoming_vector,
                historical_vector
            )

            similarities.append({
                "incoming_case_id": incoming_row["case_id"],
                "candidate_case_id": historical_row["case_id"],
                "semantic_score": round(score, 4),
            })

        # Sort highest similarity first
        similarities = sorted(
            similarities,
            key=lambda x: x["semantic_score"],
            reverse=True
        )

        # Keep top 3 candidates
        results.extend(similarities[:3])

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path = (
        "outputs/semantic_similarity_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nSemantic similarity screening completed.")
    print(f"Results saved to: {output_path}")

    print("\nTop semantic candidates:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()