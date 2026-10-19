import csv

from src.chatbot.rab_bot import RABBot


def _read_audit_rows(path):
    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def test_supported_meddra_has_terminal_audit(
    tmp_path,
):

    audit_dir = tmp_path / "outputs"
    audit_dir.mkdir()

    # We patch the audit target through the module itself.
    import src.audit_log as audit_log

    original = audit_log.RAB_AUDIT_LOG_FILE

    try:

        audit_log.RAB_AUDIT_LOG_FILE = (
            audit_dir / "rab_audit_log.csv"
        )

        bot = RABBot()

        response = bot.ask(
            "patient has headache"
        )

        assert response.response_mode == "MEDDRA_ANSWER"
        assert response.operational_state == "COMPLETED"
        assert response.audit_status == "WRITTEN"
        assert response.request_id.startswith("RAB-")

        rows = _read_audit_rows(
            audit_log.RAB_AUDIT_LOG_FILE
        )

        assert len(rows) == 1

        row = rows[0]

        assert row["request_id"] == response.request_id
        assert row["route"] == "MEDDRA"
        assert row["response_mode"] == "MEDDRA_ANSWER"
        assert row["operational_state"] == "COMPLETED"
        assert row["action"] == "RETURN_SUPPORTED_MEDDRA"

    finally:

        audit_log.RAB_AUDIT_LOG_FILE = original


def test_unsupported_meddra_is_knowledge_gap(
    tmp_path,
):

    audit_dir = tmp_path / "outputs"
    audit_dir.mkdir()

    import src.audit_log as audit_log

    original = audit_log.RAB_AUDIT_LOG_FILE

    try:

        audit_log.RAB_AUDIT_LOG_FILE = (
            audit_dir / "rab_audit_log.csv"
        )

        bot = RABBot()

        response = bot.ask(
            "patient developed zorbaflux syndrome"
        )

        assert response.response_mode == "ASK_ADD_TO_DATASET"
        assert response.operational_state == "KNOWLEDGE_GAP"
        assert response.route == "MEDDRA"
        assert response.audit_status == "WRITTEN"

        rows = _read_audit_rows(
            audit_log.RAB_AUDIT_LOG_FILE
        )

        assert len(rows) == 1

        row = rows[0]

        assert row["operational_state"] == "KNOWLEDGE_GAP"
        assert row["action"] == "REQUEST_DATASET_REVIEW"

        assert (
            "sufficient support"
            in row["reason"].lower()
        )

        assert (
            "out of my context"
            not in row["reason"].lower()
        )

    finally:

        audit_log.RAB_AUDIT_LOG_FILE = original


def test_out_of_context_has_terminal_audit(
    tmp_path,
):

    audit_dir = tmp_path / "outputs"
    audit_dir.mkdir()

    import src.audit_log as audit_log

    original = audit_log.RAB_AUDIT_LOG_FILE

    try:

        audit_log.RAB_AUDIT_LOG_FILE = (
            audit_dir / "rab_audit_log.csv"
        )

        bot = RABBot()

        response = bot.ask(
            "what is the weather today"
        )

        assert response.response_mode == "OUT_OF_CONTEXT"
        assert response.operational_state == "OUT_OF_CONTEXT"
        assert response.audit_status == "WRITTEN"

        rows = _read_audit_rows(
            audit_log.RAB_AUDIT_LOG_FILE
        )

        assert len(rows) == 1

        row = rows[0]

        assert row["operational_state"] == "OUT_OF_CONTEXT"
        assert row["action"] == "RETURN_OUT_OF_CONTEXT"

    finally:

        audit_log.RAB_AUDIT_LOG_FILE = original


def test_each_request_generates_one_terminal_event(
    tmp_path,
):

    audit_dir = tmp_path / "outputs"
    audit_dir.mkdir()

    import src.audit_log as audit_log

    original = audit_log.RAB_AUDIT_LOG_FILE

    try:

        audit_log.RAB_AUDIT_LOG_FILE = (
            audit_dir / "rab_audit_log.csv"
        )

        bot = RABBot()

        responses = [
            bot.ask(
                "patient has headache"
            ),
            bot.ask(
                "patient developed zorbaflux syndrome"
            ),
            bot.ask(
                "what is the weather today"
            ),
        ]

        rows = _read_audit_rows(
            audit_log.RAB_AUDIT_LOG_FILE
        )

        assert len(rows) == 3

        request_ids = {
            row["request_id"]
            for row in rows
        }

        assert len(request_ids) == 3

        for response in responses:

            assert response.request_id in request_ids
            assert response.audit_status == "WRITTEN"

    finally:

        audit_log.RAB_AUDIT_LOG_FILE = original