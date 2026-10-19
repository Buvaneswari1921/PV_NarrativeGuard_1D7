import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# 1. Configuration
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

CHUNKS_PATH = "outputs/rag_chunks.csv"
EMBEDDINGS_PATH = "outputs/rag_chunk_embeddings.npy"
METADATA_PATH = "outputs/rag_chunk_metadata.csv"


# --------------------------------------------------
# 2. Generate embeddings
# --------------------------------------------------

def generate_embeddings(texts):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    # Sort by index to preserve the original chunk order
    data = sorted(
        response.data,
        key=lambda item: item.index
    )

    return [item.embedding for item in data]


# --------------------------------------------------
# 3. Build embedding index
# --------------------------------------------------

def main():

    print("Loading RAG chunks...")

    chunks = pd.read_csv(
        CHUNKS_PATH
    )

    print(f"Chunks found: {len(chunks)}")

    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(
        chunks["text"].tolist()
    )

    embedding_matrix = np.array(
        embeddings,
        dtype=np.float32
    )

    # --------------------------------------------------
    # Save vectors
    # --------------------------------------------------

    np.save(
        EMBEDDINGS_PATH,
        embedding_matrix
    )

    # --------------------------------------------------
    # Save metadata separately
    # --------------------------------------------------

    metadata = chunks[
        ["chunk_id", "source", "text"]
    ].copy()

    metadata.to_csv(
        METADATA_PATH,
        index=False
    )

    print("\nRAG embedding index created successfully.")

    print(
        f"Embedding matrix shape: "
        f"{embedding_matrix.shape}"
    )

    print(
        f"Embeddings saved to: "
        f"{EMBEDDINGS_PATH}"
    )

    print(
        f"Metadata saved to: "
        f"{METADATA_PATH}"
    )


if __name__ == "__main__":
    main()