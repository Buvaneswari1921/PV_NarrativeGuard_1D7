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
# Validation configuration
# ---------------------------------------------------------

REQUIRED_FIELDS = [
    "case_id",
    "patient_age",
    "patient_sex",
    "country",
    "suspect_drug",
    "adverse_event",
    "event_date",
]


VALID_SEX_VALUES = {
    "Male",
    "Female",
    "Unknown",
}


# ---------------------------------------------------------
# Validate a single ICSR
# ---------------------------------------------------------

def validate_icsr(case):
    """
    Validate one ICSR record.

    Returns:
        dict containing validation status and issues.
    """

    issues = []

    # -----------------------------------------------------
    # Check required fields
    # -----------------------------------------------------

    for field in REQUIRED_FIELDS:

        value = case.get(field)

        if pd.isna(value) or str(value).strip() == "":
            issues.append(
                f"Missing required field: {field}"
            )

    # -----------------------------------------------------
    # Stop deeper validation if required fields are missing
    # -----------------------------------------------------

    if issues:
        return {
            "is_valid": False,
            "issues": issues,
        }

    # -----------------------------------------------------
    # Validate patient age
    # -----------------------------------------------------

    try:
        age = float(case["patient_age"])

        if age < 0 or age > 120:
            issues.append(
                "Patient age must be between 0 and 120."
            )

    except (ValueError, TypeError):
        issues.append(
            "Patient age must be numeric."
        )

    # -----------------------------------------------------
    # Validate patient sex
    # -----------------------------------------------------

    if case["patient_sex"] not in VALID_SEX_VALUES:
        issues.append(
            f"Invalid patient sex: {case['patient_sex']}"
        )

    # -----------------------------------------------------
    # Validate event date
    # -----------------------------------------------------

    parsed_date = pd.to_datetime(
        case["event_date"],
        errors="coerce",
    )

    if pd.isna(parsed_date):
        issues.append(
            "Event date is not a valid date."
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
    }


# ---------------------------------------------------------
# Validate incoming ICSR dataset
# ---------------------------------------------------------

def validate_dataset(input_path):

    df = pd.read_csv(input_path)

    results = []

    for _, case in df.iterrows():

        validation_result = validate_icsr(case)

        results.append(
            {
                "case_id": case["case_id"],
                "is_valid": validation_result["is_valid"],
                "validation_issues": "; ".join(
                    validation_result["issues"]
                ),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

if __name__ == "__main__":

    input_path = DATA_DIR / "incoming_icsrs.csv"

    results_df = validate_dataset(input_path)

    output_path = OUTPUT_DIR / "validation_results.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("ICSR validation completed.")
    print()
    print(results_df.to_string(index=False))
    print()
    print(f"Results saved to: {output_path}")