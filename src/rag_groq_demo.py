import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import retrieve


# --------------------------------------------------
# 1. Load configuration
# --------------------------------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY was not found in the .env file."
    )


# --------------------------------------------------
# 2. Configure Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_api_key,
)


# --------------------------------------------------
# 3. Define grounded prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI assistant supporting a
pharmacovigilance professional.

Use ONLY the provided case information and
retrieved guidance to answer the question.

Do not invent clinical facts.

Do not make the final duplicate determination.

If the evidence suggests that human review
is required, clearly state that.

Retrieved PV guidance:
{context}
"""
        ),
        (
            "human",
            """
Investigation question:

{question}

Provide a concise, evidence-grounded explanation
for the pharmacovigilance reviewer.
"""
        ),
    ]
)


# --------------------------------------------------
# 4. Run RAG + LLM
# --------------------------------------------------

def main():

    question = (
        "A potential duplicate has very high semantic "
        "similarity but represents a different patient "
        "and event date. How should it be handled?"
    )

    print("Retrieving relevant PV guidance...")

    retrieved_chunks = retrieve(
        question,
        top_k=3
    )

    # --------------------------------------------------
    # Build RAG context
    # --------------------------------------------------

    context_parts = []

    for _, row in retrieved_chunks.iterrows():

        context_parts.append(
            f"Source: {row['source']}\n"
            f"Chunk: {row['chunk_id']}\n"
            f"Similarity: "
            f"{row['similarity_score']:.4f}\n"
            f"Content:\n{row['text']}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # --------------------------------------------------
    # Create LangChain prompt
    # --------------------------------------------------

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    # --------------------------------------------------
    # Call Groq
    # --------------------------------------------------

    print("\nSending retrieved context to Groq...")

    response = llm.invoke(messages)

    # --------------------------------------------------
    # Display result
    # --------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG + GROQ RESPONSE"
    )

    print(
        "=" * 70
    )

    print(response.content)


if __name__ == "__main__":
    main()