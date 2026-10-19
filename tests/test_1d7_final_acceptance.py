"""
PV NarrativeGuard 1D7 final-release acceptance tests.

These tests validate the consolidated project tree and representative
business/safety scenarios without creating duplicate production modules.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = PROJECT_ROOT / "pages"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def test_release_identity_and_entrypoint():
    assert (PROJECT_ROOT / "VERSION.txt").exists()
    assert "1D7" in (PROJECT_ROOT / "VERSION.txt").read_text(encoding="utf-8")
    assert (PROJECT_ROOT / "app.py").exists()


def test_final_navigation_contains_only_canonical_workflow_pages():
    page_names = sorted(p.name for p in PAGES_DIR.glob("*.py"))
    assert page_names == [
        "1_🏠_Dashboard.py",
        "2_📋_ICSR_Work_Queue.py",
        "3_🔎_Duplicate_Review.py",
        "4_✍️_Narrative_Generation.py",
        "5_🧪_Narrative_QC.py",
        "6_📜_Audit_Trail.py",
    ]


def test_no_backup_or_duplicate_module_artifacts_remain():
    forbidden = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".pytest_cache", "__pycache__"} for part in path.parts):
            continue
        if path.name.endswith(".bak") or "_backup" in path.stem:
            forbidden.append(str(path.relative_to(PROJECT_ROOT)))
    assert forbidden == []


def test_required_medddra_runtime_artifacts_exist():
    required = [
        "meddra_rag/data/pv_core_150.csv",
        "meddra_rag/data/OAE_MedDRA_Deterministic_Candidate_Inventory.csv",
        "meddra_rag/data/OAE_MedDRA_Deterministic_Decision_Inventory.csv",
        "meddra_rag/vectorstore/oae_faiss.index",
        "meddra_rag/vectorstore/oae_metadata.csv",
        "src/chatbot/meddra_adapter.py",
        "src/chatbot/rab_bot.py",
        "src/chatbot/knowledge_gap.py",
    ]
    for rel in required:
        assert (PROJECT_ROOT / rel).exists(), rel


def test_pv_core_has_exactly_150_concepts():
    df = pd.read_csv(PROJECT_ROOT / "meddra_rag/data/pv_core_150.csv")
    assert len(df) == 150


def test_demo_dataset_and_expected_scenarios_are_intact():
    incoming = pd.read_csv(DATA_DIR / "incoming_icsrs.csv")
    existing = pd.read_csv(DATA_DIR / "existing_icsrs.csv")
    expected = pd.read_csv(DATA_DIR / "expected_results.csv")

    assert len(incoming) == 12
    assert len(existing) == 5
    assert set(expected["incoming_case_id"]) == {
        "INCOMING001",
        "INCOMING002",
        "INCOMING003",
        "INCOMING004",
    }


def test_pipeline_acceptance_counts():
    df = pd.read_csv(OUTPUT_DIR / "final_pipeline_results.csv")

    assert len(df) == 12
    assert (df["validation_status"] == "VALID").sum() == 9
    assert (df["validation_status"] == "INVALID").sum() == 3
    assert (df["narrative_eligibility"] == "ELIGIBLE").sum() == 5
    assert (df["human_duplicate_decision"] == "DUPLICATE").sum() == 1
    assert (df["narrative_status"] == "BLOCKED - DUPLICATE").sum() == 1
    assert (
        df["narrative_status"]
        == "BLOCKED - HUMAN DUPLICATE REVIEW REQUIRED"
    ).sum() == 3


def test_narrative_qc_has_persisted_pass_records():
    df = pd.read_csv(OUTPUT_DIR / "narrative_qc_results.csv")

    assert not df.empty
    assert "overall_status" in df.columns
    assert (df["overall_status"] == "PASS").any()


def test_negative_narrative_qc_detects_factual_error():
    import sys

    src_dir = PROJECT_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from narrative_qc import qc_case, read_csv, get_case_id

    rows = read_csv(DATA_DIR / "incoming_icsrs.csv")
    source = next(row for row in rows if get_case_id(row) == "INCOMING001")

    incorrect_narrative = (
        "A 45-year-old female from India was receiving Drug B "
        "for hypertension when she developed acute renal impairment "
        "on 2026-07-10. The event was classified as serious. "
        "The patient is recovering. The case was reported by a physician."
    )

    result = qc_case(source, incorrect_narrative)

    assert result["suspect_drug_qc"] == "REVIEW"
    assert result["overall_status"] == "REVIEW"


def test_knowledge_gap_is_human_review_only():
    import tempfile

    from src.chatbot.knowledge_gap import (
        APPROVED,
        create_pending_request,
        get_pending_requests,
        approve_request,
    )

    with tempfile.TemporaryDirectory() as tmp:
        queue = Path(tmp) / "knowledge_gap.jsonl"

        request = create_pending_request(
            "unsupported terminology example",
            queue_path=queue,
        )

        assert len(get_pending_requests(queue)) == 1

        approved = approve_request(
            request.request_id,
            reviewer="PV_REVIEWER",
            review_notes="Approved for controlled future ingestion.",
            queue_path=queue,
        )

        assert approved["status"] == APPROVED
        assert get_pending_requests(queue) == []


def test_audit_hash_is_deterministic_and_sha256():
    from src.audit_log import hash_narrative

    value = "A controlled PV narrative."
    first = hash_narrative(value)
    second = hash_narrative(value)

    assert first == second
    assert len(first) == 64
    assert hash_narrative("") == ""


def test_final_acceptance_test_inventory_is_present():
    test_files = sorted(p.name for p in (PROJECT_ROOT / "tests").glob("test_*.py"))
    assert "test_1d7_final_acceptance.py" in test_files
    assert len(test_files) >= 6
