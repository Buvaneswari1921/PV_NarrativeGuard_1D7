import os
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

EMBEDDINGS_PATH = (
    "outputs/rag_chunk_embeddings.npy"
)

METADATA_PATH = (
    "outputs/rag_chunk_metadata.csv"
)


# --------------------------------------------------
# 2. Generate query embedding
# --------------------------------------------------

def generate_query_embedding(query):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    return response.data[0].embedding


# --------------------------------------------------
# 3. Calculate cosine similarity
# --------------------------------------------------

def cosine_similarity(vector_a, vector_b):

    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# --------------------------------------------------
# 4. Retrieve relevant chunks
# --------------------------------------------------

def retrieve(query, top_k=3):

    # Load stored embeddings
    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    # Load chunk metadata
    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Loaded {len(metadata)} knowledge chunks."
    )

    # Generate embedding ONLY for the query
    print(
        "Generating query embedding..."
    )

    query_embedding = generate_query_embedding(
        query
    )

    # Calculate similarity against stored vectors
    results = []

    for index, row in metadata.iterrows():

        score = cosine_similarity(
            query_embedding,
            embeddings[index]
        )

        results.append({
            "chunk_id": row["chunk_id"],
            "source": row["source"],
            "text": row["text"],
            "similarity_score": score,
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "similarity_score",
        ascending=False
    )

    return results_df.head(top_k)


# --------------------------------------------------
# 5. Test retrieval
# --------------------------------------------------

def main():

    query = (
        "A potential duplicate has very high semantic "
        "similarity but represents a different patient "
        "and event date. How should it be handled?"
    )

    print(
        f"\nQuery:\n{query}"
    )

    results = retrieve(
        query,
        top_k=3
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP RETRIEVED KNOWLEDGE CHUNKS"
    )

    print(
        "=" * 70
    )

    for _, row in results.iterrows():

        print(
            f"\nChunk: {row['chunk_id']}"
        )

        print(
            f"Source: {row['source']}"
        )

        print(
            f"Similarity: "
            f"{row['similarity_score']:.4f}"
        )

        print(
            "\nContent:"
        )

        print(
            row["text"]
        )

        print(
            "-" * 70
        )


if __name__ == "__main__":
    main()