"""Curated PV expression resolution for the MedDRA assistant.

Phase 1B.3 adds a deterministic normalization layer between user language
and the existing FAISS terminology retrieval.  The resolver does not invent
MedDRA concepts: every curated result must resolve to a record already present
in the local MedDRA/OAE metadata table and the curated PV Core 150 dataset.

It is intentionally small and auditable.  Its job is to recognize a supported
PV expression such as ``threw up`` or ``itchy skin`` inside a natural-language
case phrase and expose the corresponding controlled terminology record to the
existing decision engine.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .terminology_engine import (
    get_pv_core_alias_match,
    load_pv_core_vocabulary,
)


BASE_DIR = Path(__file__).resolve().parents[1]
METADATA_FILE = BASE_DIR / "meddra_rag" / "vectorstore" / "oae_metadata.csv"

NEGATION_CUES = {
    "denies",
    "deny",
    "denied",
    "no",
    "not",
    "without",
    "never",
    "negative",
}

# Expressions for which a direct one-to-one coding recommendation would be
# clinically overconfident without additional context.  A patient-reported
# "fast heartbeat" can describe palpitations, whereas tachycardia generally
# implies an increased heart rate rather than only awareness of it.
CLINICAL_AMBIGUITIES = {
    "fast heartbeat": {
        "reason": (
            "The expression describes a rapid-heartbeat complaint but does not "
            "establish a measured elevated heart rate; palpitations and "
            "tachycardia may therefore both be relevant."
        ),
        "candidate_labels": ("tachycardia AE", "palpitations AE"),
    },
    "rapid heartbeat": {
        "reason": (
            "The expression describes a rapid-heartbeat complaint but does not "
            "establish a measured elevated heart rate; palpitations and "
            "tachycardia may therefore both be relevant."
        ),
        "candidate_labels": ("tachycardia AE", "palpitations AE"),
    },
}


MEASURED_HEART_RATE_PATTERNS = (
    re.compile(r"\b(?:heart|pulse)\s+rate\s+(?:was|is|of|measured\s+at)\s+\d+(?:\.\d+)?\s*(?:bpm|beats?\s*/?\s*min(?:ute)?)\b"),
    re.compile(r"\b(?:heart|pulse)\s+rate\s+(?:was|is)\s+(?:elevated|increased|rapid|fast)\b"),
)


def _metadata_candidate_for_id(meddra_id: str, alias: str, source: str, *, core_id: str = "") -> Optional[Dict[str, Any]]:
    by_id, _ = _metadata_indexes()
    records = by_id.get(str(meddra_id), [])
    if not records:
        return None
    candidate = _row_to_candidate(
        _record_to_series(records[0]),
        alias=alias,
        core_match={"core_id": core_id, "category": "Cardiac disorders", "coding_note": "Verify against source case and current licensed MedDRA hierarchy before final coding."},
        source=source,
    )
    return candidate


def has_measured_heart_rate_context(query: str) -> bool:
    """Return True only when the wording explicitly documents heart/pulse rate measurement context."""
    q = _normalise(query)
    return any(pattern.search(q) for pattern in MEASURED_HEART_RATE_PATTERNS)


def resolve_narrative_expression(query: str) -> Optional[Dict[str, Any]]:
    """Resolve supported PV expressions in a case narrative, with negation and measured-context guards."""
    resolution = resolve_core_expression(query)
    if resolution:
        if resolution.get("status") == "NEGATED":
            return resolution
        if resolution.get("status") == "AMBIGUOUS" and has_measured_heart_rate_context(query):
            tachy = _metadata_candidate_for_id("10043071", "measured heart rate", "measured_heart_rate_context", core_id="PV077")
            if tachy:
                resolution = dict(resolution)
                resolution["status"] = "RESOLVED"
                resolution["match"] = dict(resolution["match"])
                resolution["match"]["alias"] = "measured heart rate"
                resolution["candidates"] = [tachy]
                resolution["ambiguity_reason"] = ""
                resolution["clinical_context"] = "Measured heart/pulse rate context is explicitly documented."
        return resolution

    # Explicit measured heart-rate wording may not contain the curated
    # "fast heartbeat" alias. We only surface the local tachycardia record;
    # no threshold or new terminology concept is invented here.
    if has_measured_heart_rate_context(query):
        candidate = _metadata_candidate_for_id("10043071", "measured heart rate", "measured_heart_rate_context", core_id="PV077")
        if candidate:
            return {
                "status": "RESOLVED",
                "match": {
                    "alias": "measured heart rate",
                    "preferred_label": "tachycardia",
                    "meddra_id": "10043071",
                    "core_id": "PV077",
                    "category": "Cardiac disorders",
                    "coding_note": "Verify against source case and current licensed MedDRA hierarchy before final coding.",
                },
                "candidates": [candidate],
                "clinical_context": "Measured heart/pulse rate context is explicitly documented.",
            }
    return None


@lru_cache(maxsize=1)
def load_meddra_metadata() -> pd.DataFrame:
    """Load local MedDRA/OAE metadata once for deterministic core resolution."""
    if not METADATA_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(METADATA_FILE, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _normalise(value: Any) -> str:
    text = "" if value is None else str(value).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_negated_expression(query: str, alias: str) -> bool:
    """Detect simple local negation immediately before a matched expression."""
    q = _normalise(query)
    a = _normalise(alias)
    if not q or not a:
        return False

    start = q.find(a)
    if start < 0:
        return False

    context = q[:start].split()
    return any(token in NEGATION_CUES for token in context[-4:])


def _ids(value: Any) -> List[str]:
    """Return normalized MedDRA IDs from pipe/comma-delimited source values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = re.split(r"[|,]", str(value))
    return [p.strip() for p in parts if p.strip()]


def _row_to_candidate(row: pd.Series, *, alias: str, core_match: Dict[str, Any],
                      source: str = "pv_core_curated") -> Dict[str, Any]:
    """Convert a metadata row into the candidate contract expected by decide()."""
    return {
        "similarity_score": 0.0,
        "oae_id": row.get("oae_id", ""),
        "preferred_label": row.get("preferred_label", ""),
        "synonyms": row.get("synonyms", ""),
        "definition": row.get("definition", ""),
        "parents": row.get("parents", ""),
        "meddra_id": row.get("meddra_id", ""),
        "curated_core_candidate": True,
        "curated_match_score": 0.94,
        "retrieval_source": source,
        "matched_pv_expression": alias,
        "pv_core_id": core_match.get("core_id", ""),
        "pv_category": core_match.get("category", ""),
        "coding_note": core_match.get("coding_note", ""),
    }


@lru_cache(maxsize=1)
def _metadata_indexes():
    """Build compact ID and label indexes for fast per-query resolution."""
    metadata = load_meddra_metadata()
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    by_label: Dict[str, List[Dict[str, Any]]] = {}

    if metadata.empty:
        return by_id, by_label

    for _, row in metadata.iterrows():
        record = row.to_dict()
        for mid in _ids(record.get("meddra_id")):
            by_id.setdefault(mid, []).append(record)
        label = _normalise(record.get("preferred_label", ""))
        if label:
            by_label.setdefault(label, []).append(record)

    return by_id, by_label


def _record_to_series(record: Dict[str, Any]) -> pd.Series:
    return pd.Series(record)


def _find_metadata_rows_by_ids(metadata: pd.DataFrame, ids: List[str]) -> List[pd.Series]:
    if metadata.empty or not ids:
        return []
    by_id, _ = _metadata_indexes()
    seen = set()
    rows: List[pd.Series] = []
    for mid in ids:
        for record in by_id.get(mid, []):
            key = (str(record.get("oae_id", "")), str(record.get("meddra_id", "")))
            if key not in seen:
                rows.append(_record_to_series(record))
                seen.add(key)
    return rows


def resolve_core_expression(query: str) -> Optional[Dict[str, Any]]:
    """Resolve a natural-language query to a controlled PV Core record.

    Returns a structured resolution containing the matched alias and local
    metadata candidates.  Returns ``None`` when no curated expression is found
    or when the expression is explicitly negated.
    """
    vocabulary = load_pv_core_vocabulary()
    match = get_pv_core_alias_match(query, vocabulary)
    if not match:
        return None

    alias = str(match.get("alias", "")).strip()
    if is_negated_expression(query, alias):
        return {
            "status": "NEGATED",
            "match": match,
            "candidates": [],
            "reason": f"Negation detected near curated expression '{alias}'.",
        }

    metadata = load_meddra_metadata()
    rows = _find_metadata_rows_by_ids(metadata, _ids(match.get("meddra_id")))
    candidates = [
        _row_to_candidate(row, alias=alias, core_match=match)
        for row in rows
    ]

    ambiguity = CLINICAL_AMBIGUITIES.get(alias)
    if ambiguity:
        for label in ambiguity["candidate_labels"]:
            _, by_label = _metadata_indexes()
            for record in by_label.get(_normalise(label), []):
                row = _record_to_series(record)
                candidate = _row_to_candidate(
                    row,
                    alias=alias,
                    core_match=match,
                    source="pv_core_clinical_ambiguity",
                )
                candidate["curated_match_score"] = 0.90
                candidate["ambiguity_candidate"] = True
                if not any(str(x.get("meddra_id")) == str(candidate.get("meddra_id")) for x in candidates):
                    candidates.append(candidate)

    return {
        "status": "AMBIGUOUS" if ambiguity else "RESOLVED",
        "match": match,
        "candidates": candidates,
        "ambiguity_reason": ambiguity["reason"] if ambiguity else "",
    }
