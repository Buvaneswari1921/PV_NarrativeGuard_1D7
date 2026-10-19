from types import SimpleNamespace

import src.audit_log as audit_log
from src.chatbot.rab_bot import RABBot


def _read_audit_rows(path):
    import csv

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


class FailingRetriever:

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Simulated retrieval failure"
        )


class FailingMedDRAAdapter:

    def query(self, query):
        raise RuntimeError(
            "Simulated MedDRA failure"
        )


def test_retriever_failure_returns_controlled_response(
    tmp_path,
):

    audit_file = (
        tmp_path
        / "rab_audit_log.csv"
    )

    original = audit_log.RAB_AUDIT_LOG_FILE

    try:

        audit_log.RAB_AUDIT_LOG_FILE = audit_file

        bot = RABBot(
            retriever=FailingRetriever()
        )

        response = bot.ask(
            "what is pharmacovigilance"
        )

        assert response.response_mode == (
            "SYSTEM_ERROR"
        )

        assert response.operational_state == (
            "FAILED"
        )

        assert response.route == "GENERAL_PV"

        assert response.request_id.startswith(
            "RAB-"
        )

        assert response.audit_status == (
            "WRITTEN"
        )

        assert (
            "retrieval"
            in response.message.lower()
        )

        rows = _read_audit_rows(
            audit_file
        )

        assert len(rows) == 1

        row = rows[0]

        assert row["request_id"] == (
            response.request_id
        )

        assert row["operational_state"] == (
            "FAILED"
        )

        assert row["action"] == (
            "SYSTEM_ERROR"
        )

    finally:

        audit_log.RAB_AUDIT_LOG_FILE = original


def test_meddra_failure_returns_controlled_response(
    tmp_path,
):

    audit_file = (
        tmp_path
        / "rab_audit_log.csv"
    )

    original = audit_log.RAB_AUDIT_LOG_FILE

    try:

        audit_log.RAB_AUDIT_LOG_FILE = audit_file

        bot = RABBot(
            meddra_adapter=FailingMedDRAAdapter()
        )

        response = bot.ask(
            "patient has headache"
        )

        assert response.response_mode == (
            "SYSTEM_ERROR"
        )

        assert response.operational_state == (
            "FAILED"
        )

        assert response.route == "MEDDRA"

        assert response.request_id.startswith(
            "RAB-"
        )

        assert response.audit_status == (
            "WRITTEN"
        )

        assert (
            "meddra"
            in response.message.lower()
        )

        rows = _read_audit_rows(
            audit_file
        )

        assert len(rows) == 1

        row = rows[0]

        assert row["request_id"] == (
            response.request_id
        )

        assert row["route"] == "MEDDRA"

        assert row["operational_state"] == (
            "FAILED"
        )

        assert row["action"] == (
            "SYSTEM_ERROR"
        )

    finally:

        audit_log.RAB_AUDIT_LOG_FILE = original


def test_invalid_query_returns_controlled_response(
    tmp_path,
):

    audit_file = (
        tmp_path
        / "rab_audit_log.csv"
    )

    original = audit_log.RAB_AUDIT_LOG_FILE

    try:

        audit_log.RAB_AUDIT_LOG_FILE = audit_file

        bot = RABBot()

        response = bot.ask("")

        assert response.response_mode in {
            "CLARIFICATION_REQUIRED",
            "OUT_OF_CONTEXT",
        }

        assert response.request_id.startswith(
            "RAB-"
        )

        assert response.audit_status == (
            "WRITTEN"
        )

        rows = _read_audit_rows(
            audit_file
        )

        assert len(rows) == 1

        assert rows[0]["request_id"] == (
            response.request_id
        )

    finally:

        audit_log.RAB_AUDIT_LOG_FILE = original


def test_component_failure_does_not_expose_exception_details(
    tmp_path,
):

    audit_file = (
        tmp_path
        / "rab_audit_log.csv"
    )

    original = audit_log.RAB_AUDIT_LOG_FILE

    try:

        audit_log.RAB_AUDIT_LOG_FILE = audit_file

        bot = RABBot(
            retriever=FailingRetriever()
        )

        response = bot.ask(
            "what is pharmacovigilance"
        )

        assert response.response_mode == (
            "SYSTEM_ERROR"
        )

        assert (
            "Simulated retrieval failure"
            not in response.message
        )

        assert (
            "traceback"
            not in response.message.lower()
        )

    finally:

        audit_log.RAB_AUDIT_LOG_FILE = original