import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Duplicate screening configuration
# ---------------------------------------------------------

FIELD_WEIGHTS = {
    "patient_age": 15,
    "patient_sex": 10,
    "country": 10,
    "suspect_drug": 20,
    "adverse_event": 20,
    "event_date": 20,
    "reporter_type": 5,
}


# ---------------------------------------------------------
# Compare two values
# ---------------------------------------------------------

def values_match(value1, value2):
    """
    Compare two structured ICSR values.

    Returns True when the values match after
    basic normalization.
    """

    if pd.isna(value1) or pd.isna(value2):
        return False

    value1 = str(value1).strip().lower()
    value2 = str(value2).strip().lower()

    return value1 == value2


# ---------------------------------------------------------
# Calculate structured similarity
# ---------------------------------------------------------

def calculate_similarity(incoming_case, historical_case):
    """
    Calculate a weighted structured similarity score
    between an incoming ICSR and a historical ICSR.

    Returns:
        score between 0 and 100
        matching fields
    """

    score = 0
    matching_fields = []

    for field, weight in FIELD_WEIGHTS.items():

        incoming_value = incoming_case.get(field)
        historical_value = historical_case.get(field)

        if values_match(incoming_value, historical_value):

            score += weight
            matching_fields.append(field)

    return score, matching_fields


# ---------------------------------------------------------
# Screen one incoming case
# ---------------------------------------------------------

def screen_case(incoming_case, historical_df):

    candidates = []

    for _, historical_case in historical_df.iterrows():

        score, matching_fields = calculate_similarity(
            incoming_case,
            historical_case,
        )

        candidates.append(
            {
                "incoming_case_id": incoming_case["case_id"],
                "candidate_case_id": historical_case["case_id"],
                "similarity_score": score,
                "matching_fields": ", ".join(
                    matching_fields
                ),
            }
        )

    candidates_df = pd.DataFrame(candidates)

    # Highest similarity first
    candidates_df = candidates_df.sort_values(
        by="similarity_score",
        ascending=False,
    )

    return candidates_df


# ---------------------------------------------------------
# Main duplicate screening
# ---------------------------------------------------------

if __name__ == "__main__":

    # Load datasets
    historical_path = DATA_DIR / "existing_icsrs.csv"
    incoming_path = DATA_DIR / "incoming_icsrs.csv"
    validation_path = OUTPUT_DIR / "validation_results.csv"

    historical_df = pd.read_csv(historical_path)
    incoming_df = pd.read_csv(incoming_path)
    validation_df = pd.read_csv(validation_path)

    # Keep only valid incoming ICSRs
    valid_case_ids = validation_df.loc[
        validation_df["is_valid"] == True,
        "case_id",
    ].tolist()

    valid_incoming_df = incoming_df[
        incoming_df["case_id"].isin(valid_case_ids)
    ]

    all_results = []

    # Screen each valid incoming case
    for _, incoming_case in valid_incoming_df.iterrows():

        case_results = screen_case(
            incoming_case,
            historical_df,
        )

        # Keep top 3 candidates
        top_candidates = case_results.head(3)

        all_results.append(top_candidates)

    # Combine results
    if all_results:

        results_df = pd.concat(
            all_results,
            ignore_index=True,
        )

    else:

        results_df = pd.DataFrame()

    # Save results
    output_path = (
        OUTPUT_DIR / "duplicate_screening_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("Duplicate screening completed.")
    print()

    print(
        results_df.to_string(index=False)
    )

    print()

    print(
        f"Results saved to: {output_path}"
    )