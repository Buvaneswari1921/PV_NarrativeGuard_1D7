"""Controlled knowledge-gap handling for RAB Bot.

This module manages a human-review queue for topics that the RAB Bot cannot
safely answer from its current knowledge sources.

Safety boundary
---------------
Creating or approving a request NEVER updates the production FAISS index
or any production knowledge dataset.

Ingestion, dataset modification, embedding generation, and FAISS
re-indexing remain separate controlled operations.

Operational robustness
----------------------
- Empty queries are rejected.
- Queue directories are created automatically.
- Review requests receive unique identifiers.
- Existing reviewed requests cannot be reviewed again.
- Invalid review decisions are rejected.
- Queue persistence failures are allowed to propagate to the caller so that
  the RAB orchestration layer can handle and audit the failure explicitly.
- No operation in this module modifies the production knowledge base.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid


# ============================================================
# REQUEST STATUS
# ============================================================

PENDING_REVIEW = "PENDING_REVIEW"
APPROVED = "APPROVED"
REJECTED = "REJECTED"


VALID_REVIEW_DECISIONS = {
    APPROVED,
    REJECTED,
}


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class KnowledgeRequest:
    """A controlled human-review knowledge-gap request."""

    request_id: str
    query: str
    created_at: str
    status: str = PENDING_REVIEW
    reviewed_at: str = ""
    reviewer: str = ""
    review_notes: str = ""

    def to_dict(self):
        """Return the request as a serializable dictionary."""
        return asdict(self)


# ============================================================
# TIME
# ============================================================

def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# QUEUE HELPERS
# ============================================================

def _ensure_queue_directory(
    path: Path,
) -> None:
    """Ensure the parent directory for the queue exists."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def _load_requests(
    path: Path,
):
    """Load knowledge requests from the JSONL queue."""

    if not path.exists():
        return []

    requests = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                requests.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    "Invalid JSON in knowledge-gap queue "
                    f"at line {line_number}: {exc}"
                ) from exc

    return requests


def _write_requests(
    path: Path,
    requests,
) -> None:
    """Rewrite the complete knowledge-gap queue."""

    _ensure_queue_directory(
        path
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for request in requests:

            f.write(
                json.dumps(
                    request,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# CREATE REQUEST
# ============================================================

def create_pending_request(
    query: str,
    queue_path="outputs/pending_knowledge_requests.jsonl",
):
    """Create a human-review knowledge-gap request.

    This operation ONLY writes to the review queue.

    It does NOT:
        - ingest the query
        - modify the knowledge dataset
        - modify the MedDRA dataset
        - create embeddings
        - modify the FAISS index
        - rebuild the vector store
    """

    query = str(query).strip()

    if not query:

        raise ValueError(
            "Knowledge-gap query cannot be empty."
        )

    path = Path(
        queue_path
    )

    # --------------------------------------------------------
    # Operational robustness:
    # create the queue directory if it does not already exist.
    # --------------------------------------------------------

    _ensure_queue_directory(
        path
    )

    request = KnowledgeRequest(
        request_id=(
            f"KR-"
            f"{uuid.uuid4().hex[:12].upper()}"
        ),
        query=query,
        created_at=_utc_now(),
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                request.to_dict(),
                ensure_ascii=False,
            )
            + "\n"
        )

    return request


# ============================================================
# GET PENDING REQUESTS
# ============================================================

def get_pending_requests(
    queue_path="outputs/pending_knowledge_requests.jsonl",
):
    """Return requests currently waiting for human review."""

    path = Path(
        queue_path
    )

    return [
        item
        for item in _load_requests(path)
        if item.get("status") == PENDING_REVIEW
    ]


# ============================================================
# REVIEW REQUEST
# ============================================================

def review_request(
    request_id: str,
    decision: str,
    reviewer: str = "",
    review_notes: str = "",
    queue_path="outputs/pending_knowledge_requests.jsonl",
):
    """Approve or reject a pending knowledge-gap request.

    Approval records the human decision only.

    IMPORTANT:
        Approval does NOT ingest the query, update the knowledge base,
        create embeddings, or rebuild FAISS.
    """

    request_id = str(
        request_id
    ).strip()

    if not request_id:

        raise ValueError(
            "Knowledge request ID cannot be empty."
        )

    decision = str(
        decision
    ).strip().upper()

    if decision not in VALID_REVIEW_DECISIONS:

        raise ValueError(
            f"Invalid decision '{decision}'. "
            f"Use {APPROVED} or {REJECTED}."
        )

    path = Path(
        queue_path
    )

    requests = _load_requests(
        path
    )

    for request in requests:

        if request.get(
            "request_id"
        ) != request_id:

            continue

        # ----------------------------------------------------
        # Prevent duplicate review
        # ----------------------------------------------------

        if request.get(
            "status"
        ) != PENDING_REVIEW:

            raise ValueError(
                f"Request {request_id} is already "
                f"{request.get('status')}."
            )

        # ----------------------------------------------------
        # Record human decision
        # ----------------------------------------------------

        request["status"] = decision

        request["reviewed_at"] = _utc_now()

        request["reviewer"] = str(
            reviewer
        ).strip()

        request["review_notes"] = str(
            review_notes
        ).strip()

        # ----------------------------------------------------
        # Persist updated queue
        # ----------------------------------------------------

        _write_requests(
            path,
            requests,
        )

        return request

    raise KeyError(
        f"Knowledge request not found: {request_id}"
    )


# ============================================================
# APPROVE REQUEST
# ============================================================

def approve_request(
    request_id: str,
    reviewer: str = "",
    review_notes: str = "",
    queue_path="outputs/pending_knowledge_requests.jsonl",
):
    """Convenience wrapper for human approval."""

    return review_request(
        request_id=request_id,
        decision=APPROVED,
        reviewer=reviewer,
        review_notes=review_notes,
        queue_path=queue_path,
    )


# ============================================================
# REJECT REQUEST
# ============================================================

def reject_request(
    request_id: str,
    reviewer: str = "",
    review_notes: str = "",
    queue_path="outputs/pending_knowledge_requests.jsonl",
):
    """Convenience wrapper for human rejection."""

    return review_request(
        request_id=request_id,
        decision=REJECTED,
        reviewer=reviewer,
        review_notes=review_notes,
        queue_path=queue_path,
    )