"""Tests for controlled RAB knowledge-gap handling."""

import json

import pytest

from src.chatbot.knowledge_gap import (
    APPROVED,
    PENDING_REVIEW,
    REJECTED,
    approve_request,
    create_pending_request,
    get_pending_requests,
    reject_request,
)


def test_create_pending_request_creates_review_record(tmp_path):
    queue = tmp_path / "pending.jsonl"

    request = create_pending_request(
        "patient feels like vomiting",
        queue_path=queue,
    )

    assert request.request_id.startswith("KR-")
    assert request.query == "patient feels like vomiting"
    assert request.status == PENDING_REVIEW

    rows = [
        json.loads(line)
        for line in queue.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 1
    assert rows[0]["request_id"] == request.request_id
    assert rows[0]["status"] == PENDING_REVIEW


def test_empty_query_is_rejected(tmp_path):
    queue = tmp_path / "pending.jsonl"

    with pytest.raises(ValueError):
        create_pending_request("   ", queue_path=queue)


def test_pending_requests_can_be_listed(tmp_path):
    queue = tmp_path / "pending.jsonl"

    create_pending_request("topic one", queue_path=queue)
    create_pending_request("topic two", queue_path=queue)

    pending = get_pending_requests(queue_path=queue)

    assert len(pending) == 2
    assert all(item["status"] == PENDING_REVIEW for item in pending)


def test_request_can_be_approved_without_ingestion(tmp_path):
    queue = tmp_path / "pending.jsonl"

    request = create_pending_request(
        "patient feels like vomiting",
        queue_path=queue,
    )

    reviewed = approve_request(
        request.request_id,
        reviewer="PV Reviewer",
        review_notes="Validated for controlled dataset update.",
        queue_path=queue,
    )

    assert reviewed["status"] == APPROVED
    assert reviewed["reviewer"] == "PV Reviewer"
    assert reviewed["review_notes"] == (
        "Validated for controlled dataset update."
    )
    assert reviewed["reviewed_at"]

    rows = queue.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


def test_request_can_be_rejected(tmp_path):
    queue = tmp_path / "pending.jsonl"

    request = create_pending_request(
        "unsupported topic",
        queue_path=queue,
    )

    reviewed = reject_request(
        request.request_id,
        reviewer="PV Reviewer",
        review_notes="Insufficient source evidence.",
        queue_path=queue,
    )

    assert reviewed["status"] == REJECTED


def test_already_reviewed_request_cannot_be_reviewed_again(tmp_path):
    queue = tmp_path / "pending.jsonl"

    request = create_pending_request(
        "topic",
        queue_path=queue,
    )

    approve_request(
        request.request_id,
        queue_path=queue,
    )

    with pytest.raises(ValueError):
        reject_request(
            request.request_id,
            queue_path=queue,
        )


def test_unknown_request_id_is_rejected(tmp_path):
    queue = tmp_path / "pending.jsonl"

    create_pending_request(
        "topic",
        queue_path=queue,
    )

    with pytest.raises(KeyError):
        approve_request(
            "KR-DOESNOTEXIST",
            queue_path=queue,
        )


def test_invalid_review_decision_is_rejected(tmp_path):
    queue = tmp_path / "pending.jsonl"

    request = create_pending_request(
        "topic",
        queue_path=queue,
    )

    from src.chatbot.knowledge_gap import review_request

    with pytest.raises(ValueError):
        review_request(
            request.request_id,
            "MAYBE",
            queue_path=queue,
        )
