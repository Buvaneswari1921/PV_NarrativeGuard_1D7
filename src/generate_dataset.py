import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Historical ICSR dataset
# ---------------------------------------------------------

existing_cases = [
    {
        "case_id": "CASE001",
        "patient_age": 45,
        "patient_sex": "Female",
        "country": "India",
        "suspect_drug": "Drug A",
        "indication": "Hypertension",
        "adverse_event": "Acute kidney injury",
        "event_date": "2026-07-10",
        "seriousness": "Serious",
        "outcome": "Recovering",
        "reporter_type": "Physician",
        "case_description": (
            "A 45-year-old female receiving Drug A for hypertension "
            "developed acute kidney injury on 10-Jul-2026."
        ),
    },
    {
        "case_id": "CASE002",
        "patient_age": 62,
        "patient_sex": "Male",
        "country": "USA",
        "suspect_drug": "Drug B",
        "indication": "Diabetes",
        "adverse_event": "Hypoglycemia",
        "event_date": "2026-07-15",
        "seriousness": "Non-serious",
        "outcome": "Recovered",
        "reporter_type": "Pharmacist",
        "case_description": (
            "A 62-year-old male receiving Drug B for diabetes "
            "experienced hypoglycemia on 15-Jul-2026."
        ),
    },
    {
        "case_id": "CASE003",
        "patient_age": 34,
        "patient_sex": "Female",
        "country": "UK",
        "suspect_drug": "Drug C",
        "indication": "Rheumatoid arthritis",
        "adverse_event": "Headache",
        "event_date": "2026-07-20",
        "seriousness": "Non-serious",
        "outcome": "Recovered",
        "reporter_type": "Consumer",
        "case_description": (
            "A 34-year-old female treated with Drug C for rheumatoid "
            "arthritis reported headache on 20-Jul-2026."
        ),
    },
    {
        "case_id": "CASE004",
        "patient_age": 51,
        "patient_sex": "Male",
        "country": "India",
        "suspect_drug": "Drug D",
        "indication": "Asthma",
        "adverse_event": "Anaphylactic reaction",
        "event_date": "2026-07-25",
        "seriousness": "Serious",
        "outcome": "Recovered",
        "reporter_type": "Physician",
        "case_description": (
            "A 51-year-old male receiving Drug D for asthma "
            "experienced an anaphylactic reaction on 25-Jul-2026."
        ),
    },
    {
        "case_id": "CASE005",
        "patient_age": 39,
        "patient_sex": "Female",
        "country": "Germany",
        "suspect_drug": "Drug E",
        "indication": "Migraine",
        "adverse_event": "Nausea",
        "event_date": "2026-08-01",
        "seriousness": "Non-serious",
        "outcome": "Recovering",
        "reporter_type": "Physician",
        "case_description": (
            "A 39-year-old female treated with Drug E for migraine "
            "developed nausea on 01-Aug-2026."
        ),
    },
]


# ---------------------------------------------------------
# Incoming ICSR dataset
# ---------------------------------------------------------

incoming_cases = [
    {
        "case_id": "INCOMING001",
        "patient_age": 45,
        "patient_sex": "Female",
        "country": "India",
        "suspect_drug": "Drug A",
        "indication": "Hypertension",
        "adverse_event": "Acute renal impairment",
        "event_date": "2026-07-10",
        "seriousness": "Serious",
        "outcome": "Recovering",
        "reporter_type": "Physician",
        "case_description": (
            "A 45-year-old woman treated with Drug A for hypertension "
            "developed acute renal impairment on 10-Jul-2026."
        ),
    },
    {
        "case_id": "INCOMING002",
        "patient_age": 70,
        "patient_sex": "Male",
        "country": "Canada",
        "suspect_drug": "Drug F",
        "indication": "Hyperlipidemia",
        "adverse_event": "Dizziness",
        "event_date": "2026-08-05",
        "seriousness": "Non-serious",
        "outcome": "Recovered",
        "reporter_type": "Physician",
        "case_description": (
            "A 70-year-old male receiving Drug F for hyperlipidemia "
            "reported dizziness on 05-Aug-2026."
        ),
    },
    {
        "case_id": "INCOMING003",
        "patient_age": 62,
        "patient_sex": "Male",
        "country": "USA",
        "suspect_drug": "Drug B",
        "indication": "Diabetes",
        "adverse_event": "Low blood glucose",
        "event_date": "2026-07-15",
        "seriousness": "Non-serious",
        "outcome": "Recovered",
        "reporter_type": "Pharmacist",
        "case_description": (
            "A 62-year-old man using Drug B for diabetes experienced "
            "low blood glucose on 15-Jul-2026."
        ),
    },
    {
        "case_id": "INCOMING004",
        "patient_age": 28,
        "patient_sex": "Female",
        "country": "India",
        "suspect_drug": "Drug A",
        "indication": "Hypertension",
        "adverse_event": "Acute kidney injury",
        "event_date": "2026-08-10",
        "seriousness": "Serious",
        "outcome": "Recovering",
        "reporter_type": "Physician",
        "case_description": (
            "A 28-year-old female receiving Drug A for hypertension "
            "developed acute kidney injury on 10-Aug-2026."
        ),
    },

            
    

    # -----------------------------------------------------
    # Deliberately invalid test cases
    # -----------------------------------------------------

    {
        "case_id": "INCOMING005",
        "patient_age": -5,
        "patient_sex": "Female",
        "country": "India",
        "suspect_drug": "Drug G",
        "indication": "Pain",
        "adverse_event": "Nausea",
        "event_date": "2026-08-12",
        "seriousness": "Non-serious",
        "outcome": "Recovered",
        "reporter_type": "Physician",
        "case_description": (
            "Invalid test case with a negative patient age."
        ),
    },

    {
        "case_id": "INCOMING006",
        "patient_age": 55,
        "patient_sex": "Male",
        "country": "India",
        "suspect_drug": "Drug H",
        "indication": "Diabetes",
        "adverse_event": "Dizziness",
        "event_date": "not-a-date",
        "seriousness": "Non-serious",
        "outcome": "Recovered",
        "reporter_type": "Physician",
        "case_description": (
            "Invalid test case with an invalid event date."
        ),
    },

    {
        "case_id": "INCOMING007",
        "patient_age": 41,
        "patient_sex": "Female",
        "country": "India",
        "suspect_drug": "",
        "indication": "Hypertension",
        "adverse_event": "Headache",
        "event_date": "2026-08-15",
        "seriousness": "Non-serious",
        "outcome": "Recovered",
        "reporter_type": "Physician",
        "case_description": (
            "Invalid test case with a missing suspect drug."
        ),
    },
]


# ---------------------------------------------------------
# Ground-truth labels
# ---------------------------------------------------------

expected_results = [
    {
        "incoming_case_id": "INCOMING001",
        "expected_duplicate": True,
        "expected_case_id": "CASE001",
        "duplicate_type": "Paraphrased duplicate",
    },
    {
        "incoming_case_id": "INCOMING002",
        "expected_duplicate": False,
        "expected_case_id": "",
        "duplicate_type": "New case",
    },
    {
        "incoming_case_id": "INCOMING003",
        "expected_duplicate": True,
        "expected_case_id": "CASE002",
        "duplicate_type": "Paraphrased duplicate",
    },
    {
        "incoming_case_id": "INCOMING004",
        "expected_duplicate": False,
        "expected_case_id": "",
        "duplicate_type": "Similar drug/event but different patient and event date",
    },
]


# ---------------------------------------------------------
# Convert to DataFrames
# ---------------------------------------------------------

existing_df = pd.DataFrame(existing_cases)
incoming_df = pd.DataFrame(incoming_cases)
expected_df = pd.DataFrame(expected_results)


# ---------------------------------------------------------
# Save datasets
# ---------------------------------------------------------

existing_path = DATA_DIR / "existing_icsrs.csv"
incoming_path = DATA_DIR / "incoming_icsrs.csv"
expected_path = DATA_DIR / "expected_results.csv"

existing_df.to_csv(existing_path, index=False)
incoming_df.to_csv(incoming_path, index=False)
expected_df.to_csv(expected_path, index=False)


# ---------------------------------------------------------
# Display summary
# ---------------------------------------------------------

print("Synthetic ICSR datasets generated successfully.")
print()
print(f"Historical cases : {len(existing_df)}")
print(f"Incoming cases   : {len(incoming_df)}")
print(f"Expected labels  : {len(expected_df)}")
print()
print(f"Saved to: {DATA_DIR}")