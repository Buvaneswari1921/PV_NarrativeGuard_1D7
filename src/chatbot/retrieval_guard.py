"""Decision layer between the existing RAG retriever and RAB Bot."""

from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_MIN_SIMILARITY = 0.65
DEFAULT_STRONG_SIMILARITY = 0.80


@dataclass
class RetrievalDecision:
    status: str
    confidence: float
    evidence: list
    reason: str


def evaluate_retrieval(
    retrieved,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    strong_similarity: float = DEFAULT_STRONG_SIMILARITY,
) -> RetrievalDecision:
    """Evaluate retrieved rows without treating retrieval as truth.

    Expected input: pandas DataFrame returned by src.rag_retriever.retrieve().
    """
    if retrieved is None or len(retrieved) == 0:
        return RetrievalDecision(
            status="INSUFFICIENT",
            confidence=0.0,
            evidence=[],
            reason="No knowledge chunks were retrieved.",
        )

    if "similarity_score" not in retrieved.columns:
        raise ValueError("Retriever output must contain similarity_score.")

    top_score = float(retrieved.iloc[0]["similarity_score"])

    evidence = []
    for _, row in retrieved.iterrows():
        score = float(row["similarity_score"])
        if score >= min_similarity:
            evidence.append({
                "chunk_id": row.get("chunk_id"),
                "source": row.get("source"),
                "text": row.get("text"),
                "similarity_score": score,
            })

    if top_score >= strong_similarity and evidence:
        return RetrievalDecision(
            status="SUFFICIENT",
            confidence=top_score,
            evidence=evidence,
            reason="Top retrieved evidence exceeds the strong-evidence threshold.",
        )

    if evidence:
        return RetrievalDecision(
            status="WEAK",
            confidence=top_score,
            evidence=evidence,
            reason="Related evidence was retrieved, but confidence is below the strong-evidence threshold.",
        )

    return RetrievalDecision(
        status="INSUFFICIENT",
        confidence=top_score,
        evidence=[],
        reason="Retrieved results did not reach the minimum evidence threshold.",
    )
