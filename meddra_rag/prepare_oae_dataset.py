import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "OAE.csv"
OUTPUT_FILE = BASE_DIR / "data" / "oae_mapped.csv"

# ---------------------------------------------------------
# Load OAE
# ---------------------------------------------------------

print("Loading OAE dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Total OAE records: {len(df)}")

# ---------------------------------------------------------
# Keep only concepts having a MedDRA mapping
# ---------------------------------------------------------

mapped = df[
    df["has_MedDRA_id"].notna()
].copy()

print(f"Records with MedDRA ID: {len(mapped)}")

# ---------------------------------------------------------
# Select useful fields for RAG
# ---------------------------------------------------------

columns_to_keep = [
    "Class ID",
    "Preferred Label",
    "Synonyms",
    "Definitions",
    "Parents",
    "has_MedDRA_id",
]

# Keep only columns that actually exist
available_columns = [
    col for col in columns_to_keep
    if col in mapped.columns
]

rag_df = mapped[available_columns].copy()

# ---------------------------------------------------------
# Rename columns to cleaner names
# ---------------------------------------------------------

rename_map = {
    "Class ID": "oae_id",
    "Preferred Label": "preferred_label",
    "Synonyms": "synonyms",
    "Definitions": "definition",
    "Parents": "parents",
    "has_MedDRA_id": "meddra_id",
}

rag_df = rag_df.rename(columns=rename_map)

# ---------------------------------------------------------
# Remove completely empty labels
# ---------------------------------------------------------

rag_df = rag_df[
    rag_df["preferred_label"].notna()
].copy()

# ---------------------------------------------------------
# Clean text fields
# ---------------------------------------------------------

for column in ["preferred_label", "synonyms", "definition", "parents"]:
    if column in rag_df.columns:
        rag_df[column] = (
            rag_df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

# ---------------------------------------------------------
# Clean MedDRA IDs
# ---------------------------------------------------------

rag_df["meddra_id"] = (
    rag_df["meddra_id"]
    .astype(str)
    .str.strip()
)

# ---------------------------------------------------------
# Remove accidental "nan"
# ---------------------------------------------------------

rag_df = rag_df[
    rag_df["meddra_id"].str.lower() != "nan"
].copy()

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

rag_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nDataset preparation complete.")
print(f"RAG dataset: {OUTPUT_FILE}")
print(f"RAG records: {len(rag_df)}")

print("\nColumns:")
print(rag_df.columns.tolist())

print("\nFirst 10 records:")
print(
    rag_df.head(10).to_string(index=False)
)