"""Adapter between RAB Bot and the existing MedDRA FAISS + terminology engine.

This module preserves the existing MedDRA architecture:
    Sentence Transformer -> FAISS -> candidate records -> terminology_engine.decide()

It deliberately loads the model/index lazily so importing RAB does not
load a large model unnecessarily.
"""

from pathlib import Path
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from meddra_rag.terminology_engine import decide
from meddra_rag.pv_expression_resolver import resolve_narrative_expression


BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_FILE = BASE_DIR / "meddra_rag" / "vectorstore" / "oae_faiss.index"
METADATA_FILE = BASE_DIR / "meddra_rag" / "vectorstore" / "oae_metadata.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 20


class MedDRAAdapter:
    def __init__(self, top_k=TOP_K):
        self.top_k = top_k
        self._model = None
        self._index = None
        self._metadata = None

    def _load(self):
        if self._model is None:
            print("Loading MedDRA embedding model...")
            self._model = SentenceTransformer(MODEL_NAME)

        if self._index is None:
            print("Loading MedDRA FAISS index...")
            self._index = faiss.read_index(str(INDEX_FILE))

        if self._metadata is None:
            print("Loading MedDRA metadata...")
            self._metadata = pd.read_csv(
                METADATA_FILE,
                low_memory=False,
            )

    def _enrich_curated_candidates(self, query, candidates):
        """Merge verified PV-expression candidates into retrieved candidates.

        This is deliberately separate from FAISS retrieval so callers/tests can
        replace retrieval without bypassing the deterministic enrichment layer.
        """
        resolution = resolve_narrative_expression(query)
        if not resolution or resolution.get("status") == "NEGATED":
            return candidates

        by_id = {}
        for idx, existing in enumerate(candidates):
            eid = str(existing.get("meddra_id", "")).strip()
            if eid:
                by_id[eid] = idx

        for candidate in resolution.get("candidates", []):
            cid = str(candidate.get("meddra_id", "")).strip()
            if not cid:
                continue

            if cid in by_id:
                idx = by_id[cid]
                merged = dict(candidates[idx])
                for key, value in candidate.items():
                    if key != "similarity_score":
                        merged[key] = value
                merged["similarity_score"] = candidates[idx].get(
                    "similarity_score", 0.0
                )
                candidates[idx] = merged
            else:
                candidates.append(candidate)
                by_id[cid] = len(candidates) - 1

        return candidates

    def retrieve_candidates(self, query):
        self._load()

        embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self._index.search(
            embedding,
            self.top_k,
        )

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            row = self._metadata.iloc[int(idx)]
            candidates.append({
                "similarity_score": float(score),
                "oae_id": row["oae_id"],
                "preferred_label": row["preferred_label"],
                "synonyms": row["synonyms"],
                "definition": row["definition"],
                "parents": row["parents"],
                "meddra_id": row["meddra_id"],
            })

        return self._enrich_curated_candidates(query, candidates)

    def query(self, user_query):
        resolution = resolve_narrative_expression(user_query)

        if resolution and resolution.get("status") == "NEGATED":
            return {
                "status": "NEGATED",
                "message": (
                    "The matched clinical expression is explicitly negated; "
                    "it should not be treated as a reported adverse event."
                ),
                "result": None,
                "candidates": [],
            }

        # Hard safety gate for explicitly curated clinical ambiguities.
        # Do not let a high-similarity FAISS result silently override a
        # resolver-level ambiguity such as 'fast heartbeat'.  The resolver
        # only changes this to RESOLVED when measured heart-rate context is
        # explicitly present.
        if resolution and resolution.get("status") == "AMBIGUOUS":
            ambiguity_candidates = resolution.get("candidates", []) or []
            return {
                "status": "AMBIGUOUS",
                "message": resolution.get(
                    "ambiguity_reason",
                    "Additional clinical context is required before coding.",
                ),
                "result": None,
                "candidates": ambiguity_candidates[:5],
            }

        candidates = self.retrieve_candidates(user_query)

        # Always apply deterministic PV-expression enrichment at the adapter
        # boundary.  This remains effective when retrieval is mocked/replaced
        # in tests or by another retrieval backend, preventing a raw FAISS
        # candidate from shadowing a verified curated expression.
        candidates = self._enrich_curated_candidates(
            user_query,
            candidates,
        )

        decision = decide(user_query, candidates)

        # Narrative expressions such as 'patient threw up twice' may be
        # recognized deterministically even when FAISS's raw ranking does not
        # retain the curated candidate strongly enough. In that case, rerun the
        # controlled decision on the verified alias. This does not invent a
        # concept: the resolver has already verified the alias against local
        # terminology metadata.
        if (
            resolution
            and resolution.get("status") == "RESOLVED"
            and decision.get("status") != "SUPPORTED"
        ):
            alias = str(resolution.get("match", {}).get("alias", "")).strip()
            if alias:
                alias_decision = decide(alias, candidates)
                if alias_decision.get("status") == "SUPPORTED":
                    result = dict(alias_decision.get("result") or {})
                    result["original_query"] = user_query
                    result["matched_pv_expression"] = alias
                    result["narrative_resolution"] = True
                    alias_decision["result"] = result
                    return alias_decision

        return decision
