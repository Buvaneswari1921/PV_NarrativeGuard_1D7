from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DOCUMENTS_DIR = Path("documents")
OUTPUT_PATH = Path("outputs/rag_chunks.csv")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# --------------------------------------------------
# 2. Read documents
# --------------------------------------------------

def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if text:
            documents.append({
                "source": file_path.name,
                "text": text,
            })

    return documents


# --------------------------------------------------
# 3. Split text into chunks
# --------------------------------------------------

def create_chunks(text, chunk_size=CHUNK_SIZE,
                  overlap=CHUNK_OVERLAP):

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# --------------------------------------------------
# 4. Build the RAG knowledge base
# --------------------------------------------------

def main():

    print("Loading RAG documents...")

    documents = load_documents()

    print(f"Documents found: {len(documents)}")

    if not documents:
        raise ValueError(
            "No .txt documents found in the documents folder."
        )

    records = []

    chunk_number = 1

    for document in documents:

        chunks = create_chunks(
            document["text"]
        )

        print(
            f"{document['source']}: "
            f"{len(chunks)} chunks"
        )

        for chunk in chunks:

            records.append({
                "chunk_id": f"CHUNK_{chunk_number:03d}",
                "source": document["source"],
                "text": chunk,
            })

            chunk_number += 1


    # --------------------------------------------------
    # 5. Save chunks
    # --------------------------------------------------

    chunks_df = pd.DataFrame(records)

    chunks_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nRAG ingestion completed.")
    print(f"Total chunks: {len(chunks_df)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nChunk preview:")
    print(
        chunks_df[
            ["chunk_id", "source", "text"]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()