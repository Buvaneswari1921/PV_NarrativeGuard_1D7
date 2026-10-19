import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "oae_mapped.csv"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = VECTORSTORE_DIR / "oae_faiss.index"
METADATA_FILE = VECTORSTORE_DIR / "oae_metadata.csv"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("=" * 70)
print("OAE RAG VECTOR STORE BUILDER")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_FILE,
    low_memory=False
)

print(f"Records loaded: {len(df)}")


# ---------------------------------------------------------
# Prepare text for embedding
# ---------------------------------------------------------

def create_embedding_text(row):
    """
    Combine terminology information into one searchable text.
    """

    parts = []

    preferred_label = str(row.get("preferred_label", "")).strip()
    synonyms = str(row.get("synonyms", "")).strip()
    definition = str(row.get("definition", "")).strip()

    if preferred_label and preferred_label.lower() != "nan":
        parts.append(f"Preferred term: {preferred_label}")

    if synonyms and synonyms.lower() != "nan":
        parts.append(f"Synonyms: {synonyms}")

    if definition and definition.lower() != "nan":
        parts.append(f"Definition: {definition}")

    return " | ".join(parts)


print("\nPreparing terminology text...")

df["embedding_text"] = df.apply(
    create_embedding_text,
    axis=1
)

print("Terminology text prepared.")


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

print(f"\nLoading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

print("\nGenerating embeddings...")
print("This may take a few minutes.")

texts = df["embedding_text"].tolist()

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = embeddings.astype("float32")

print(f"\nEmbedding shape: {embeddings.shape}")


# ---------------------------------------------------------
# Build FAISS index
# ---------------------------------------------------------

print("\nBuilding FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS vectors stored: {index.ntotal}")
print(f"Vector dimension: {dimension}")


# ---------------------------------------------------------
# Save FAISS index
# ---------------------------------------------------------

print("\nSaving FAISS index...")

faiss.write_index(
    index,
    str(INDEX_FILE)
)

print(f"Index saved to: {INDEX_FILE}")


# ---------------------------------------------------------
# Save metadata
# ---------------------------------------------------------

metadata_columns = [
    "oae_id",
    "preferred_label",
    "synonyms",
    "definition",
    "parents",
    "meddra_id",
    "embedding_text",
]

metadata_columns = [
    col for col in metadata_columns
    if col in df.columns
]

metadata = df[metadata_columns].copy()

metadata.to_csv(
    METADATA_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"Metadata saved to: {METADATA_FILE}")


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("VECTOR STORE BUILD COMPLETE")
print("=" * 70)

print(f"Total concepts : {len(df)}")
print(f"FAISS vectors  : {index.ntotal}")
print(f"Vector size    : {dimension}")
print(f"Index file     : {INDEX_FILE}")
print(f"Metadata file  : {METADATA_FILE}")

print("\nNext step:")
print("Test semantic retrieval using real adverse-event queries.")