import pandas as pd

FILE = "meddra_rag/data/OAE.csv"

df = pd.read_csv(FILE, low_memory=False)

print("Total rows:", len(df))

mapped = df[df["has_MedDRA_id"].notna()].copy()

print("Rows with MedDRA ID:", len(mapped))
print("Rows without MedDRA ID:", len(df) - len(mapped))

terms = [
    "headache",
    "nausea",
    "vomiting",
    "diarrhea",
    "constipation",
    "dizziness",
    "fatigue",
    "rash",
    "pruritus",
    "cough",
    "dyspnea",
    "pyrexia",
    "fever",
    "abdominal pain",
    "back pain",
    "chest pain",
    "pain",
    "edema",
    "insomnia",
    "somnolence",
    "anxiety",
    "depression",
    "hypertension",
    "hypotension",
    "tachycardia",
    "palpitations",
    "urinary tract infection",
    "anemia",
    "thrombocytopenia",
    "leukopenia",
    "neutropenia",
    "increased liver enzymes",
    "alanine aminotransferase",
    "aspartate aminotransferase",
]

labels = df["Preferred Label"].fillna("").astype(str).str.lower()

print("\n" + "=" * 100)
print("COMMON AE COVERAGE")
print("=" * 100)

for term in terms:
    matches = df[labels.str.contains(term, regex=False)]

    if len(matches) == 0:
        print(f"\n{term:30} -> NOT FOUND")
    else:
        print(f"\n{term:30} -> {len(matches)} match(es)")

        for _, row in matches.head(3).iterrows():
            print(
                f"   Label: {row['Preferred Label']}"
            )
            print(
                f"   MedDRA ID: {row['has_MedDRA_id']}"
            )
            print(
                f"   OAE ID: {row['Class ID']}"
            )