"""
PV_NarrativeGuard - End-to-End Backend Pipeline

Workflow:
    1. ICSR validation
    2. Structured duplicate screening
    3. Semantic similarity
    4. Combined duplicate scoring
    5. RAG-based duplicate investigation
    6. Human duplicate-review gate
    7. Final case-level backend output

IMPORTANT:
    Narrative generation is intentionally NOT performed automatically.

    Narrative generation will be triggered on-demand from the Streamlit UI
    after the case is eligible for narrative generation.

Human PV review remains required before final use.
"""

from pathlib import Path
import importlib

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INCOMING_FILE = DATA_DIR / "incoming_icsrs.csv"
HISTORICAL_FILE = DATA_DIR / "existing_icsrs.csv"

VALIDATION_FILE = OUTPUT_DIR / "validation_results.csv"
STRUCTURED_FILE = OUTPUT_DIR / "duplicate_screening_results.csv"
SEMANTIC_FILE = OUTPUT_DIR / "semantic_similarity_results.csv"
COMBINED_FILE = OUTPUT_DIR / "combined_duplicate_results.csv"

HUMAN_REVIEW_FILE = OUTPUT_DIR / "duplicate_human_review.csv"
FINAL_FILE = OUTPUT_DIR / "final_pipeline_results.csv"


# ============================================================
# CONFIGURATION
# ============================================================

HIGH_RULE_THRESHOLD = 80
HIGH_SEMANTIC_THRESHOLD = 0.90

POTENTIAL_RULE_THRESHOLD = 60
POTENTIAL_SEMANTIC_THRESHOLD = 0.80

RULE_WEIGHT = 0.40
SEMANTIC_WEIGHT = 0.60


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ensure_output_directory():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STAGE 1 - VALIDATION
# ============================================================

def run_validation(incoming_file):
    from validation import validate_dataset

    validation_df = validate_dataset(incoming_file)

    validation_df.to_csv(
        VALIDATION_FILE,
        index=False
    )

    valid_count = int(validation_df["is_valid"].sum())
    invalid_count = len(validation_df) - valid_count

    print(f"Total ICSRs : {len(validation_df)}")
    print(f"Valid       : {valid_count}")
    print(f"Invalid     : {invalid_count}")
    print(f"Saved       : {VALIDATION_FILE}")

    return validation_df


# ============================================================
# STAGE 2 - STRUCTURED DUPLICATE SCREENING
# ============================================================

def run_structured_screening(
    incoming_df,
    historical_df,
    validation_df
):
    from duplicate_rules import screen_case

    valid_ids = validation_df.loc[
        validation_df["is_valid"] == True,
        "case_id"
    ].tolist()

    valid_incoming = incoming_df[
        incoming_df["case_id"].isin(valid_ids)
    ].copy()

    all_results = []

    for _, incoming_case in valid_incoming.iterrows():

        results = screen_case(
            incoming_case,
            historical_df
        )

        if not results.empty:
            results = results.head(3)
            all_results.append(results)

    if all_results:
        structured_df = pd.concat(
            all_results,
            ignore_index=True
        )
    else:
        structured_df = pd.DataFrame(
            columns=[
                "incoming_case_id",
                "candidate_case_id",
                "similarity_score",
                "matching_fields"
            ]
        )

    structured_df.to_csv(
        STRUCTURED_FILE,
        index=False
    )

    print(f"Valid cases screened : {len(valid_incoming)}")
    print(f"Candidate pairs      : {len(structured_df)}")
    print(f"Saved                : {STRUCTURED_FILE}")

    return structured_df


# ============================================================
# STAGE 3 - SEMANTIC SIMILARITY
# ============================================================

def run_semantic_similarity(
    incoming_df,
    historical_df,
    validation_df
):
    from semantic_similarity import (
        create_case_text,
        generate_embeddings,
        cosine_similarity
    )

    valid_ids = validation_df.loc[
        validation_df["is_valid"] == True,
        "case_id"
    ].tolist()

    valid_incoming = incoming_df[
        incoming_df["case_id"].isin(valid_ids)
    ].copy()

    print("Generating historical case embeddings...")

    historical_texts = [
        create_case_text(row)
        for _, row in historical_df.iterrows()
    ]

    historical_embeddings = generate_embeddings(
        historical_texts
    )

    print("Generating incoming case embeddings...")

    incoming_texts = [
        create_case_text(row)
        for _, row in valid_incoming.iterrows()
    ]

    incoming_embeddings = generate_embeddings(
        incoming_texts
    )

    results = []

    for incoming_position, (_, incoming_case) in enumerate(
        valid_incoming.iterrows()
    ):

        incoming_embedding = incoming_embeddings[
            incoming_position
        ]

        case_scores = []

        for historical_position, (_, historical_case) in enumerate(
            historical_df.iterrows()
        ):

            historical_embedding = historical_embeddings[
                historical_position
            ]

            score = cosine_similarity(
                incoming_embedding,
                historical_embedding
            )

            case_scores.append({
                "incoming_case_id": incoming_case["case_id"],
                "candidate_case_id": historical_case["case_id"],
                "semantic_score": score
            })

        case_scores = sorted(
            case_scores,
            key=lambda x: x["semantic_score"],
            reverse=True
        )

        results.extend(case_scores[:3])

    semantic_df = pd.DataFrame(results)

    semantic_df.to_csv(
        SEMANTIC_FILE,
        index=False
    )

    print(f"Valid cases processed : {len(valid_incoming)}")
    print(f"Semantic candidates   : {len(semantic_df)}")
    print(f"Saved                 : {SEMANTIC_FILE}")

    return semantic_df


# ============================================================
# STAGE 4 - COMBINED DUPLICATE SCORING
# ============================================================

def classify_candidate(row):
    rule_score = float(row["rule_score"])
    semantic_score = float(row["semantic_score"])

    if (
        rule_score >= HIGH_RULE_THRESHOLD
        and semantic_score >= HIGH_SEMANTIC_THRESHOLD
    ):
        return "High-confidence duplicate candidate"

    elif (
        rule_score >= POTENTIAL_RULE_THRESHOLD
        or semantic_score >= POTENTIAL_SEMANTIC_THRESHOLD
    ):
        return "Potential duplicate - human review"

    else:
        return "Low similarity"


def run_combined_scoring(
    structured_df,
    semantic_df
):

    if structured_df.empty or semantic_df.empty:

        combined_df = pd.DataFrame(
            columns=[
                "incoming_case_id",
                "candidate_case_id",
                "rule_score",
                "semantic_score",
                "combined_score",
                "classification"
            ]
        )

    else:

        rule_df = structured_df.copy()

        rule_df = rule_df.rename(
            columns={
                "similarity_score": "rule_score"
            }
        )

        semantic_df = semantic_df.copy()

        combined_df = pd.merge(
            rule_df,
            semantic_df,
            on=[
                "incoming_case_id",
                "candidate_case_id"
            ],
            how="inner"
        )

        combined_df["combined_score"] = (
            RULE_WEIGHT
            * (
                combined_df["rule_score"]
                / 100
            )
            +
            SEMANTIC_WEIGHT
            * combined_df["semantic_score"]
        )

        combined_df["classification"] = (
            combined_df.apply(
                classify_candidate,
                axis=1
            )
        )

        combined_df = combined_df.sort_values(
            [
                "incoming_case_id",
                "combined_score"
            ],
            ascending=[
                True,
                False
            ]
        )

    combined_df.to_csv(
        COMBINED_FILE,
        index=False
    )

    print(f"Candidate pairs : {len(combined_df)}")
    print(f"Saved           : {COMBINED_FILE}")

    return combined_df


# ============================================================
# STAGE 5 - DUPLICATE INVESTIGATION
# ============================================================

def run_duplicate_investigation(
    incoming_case_id,
    candidate_case_id
):
    """
    Run the existing RAG-based duplicate investigation.

    The investigation is assistive only.
    It does not make a final duplicate decision.
    """

    try:

        investigation_module = importlib.import_module(
            "duplicate_investigation"
        )

        investigation_module.investigate(
            incoming_case_id,
            candidate_case_id
        )

        return "Investigation completed"

    except Exception as exc:

        print(
            f"[WARNING] Investigation failed for "
            f"{incoming_case_id} vs {candidate_case_id}: {exc}"
        )

        return f"Investigation failed: {exc}"


def investigate_review_candidates(combined_df):

    review_candidates = combined_df[
        combined_df["classification"] != "Low similarity"
    ].copy()

    investigation_status = {}

    for _, row in review_candidates.iterrows():

        incoming_case_id = row["incoming_case_id"]
        candidate_case_id = row["candidate_case_id"]

        print(
            f"\n[INVESTIGATION] "
            f"{incoming_case_id} vs {candidate_case_id}"
        )

        status = run_duplicate_investigation(
            incoming_case_id,
            candidate_case_id
        )

        key = (
            incoming_case_id,
            candidate_case_id
        )

        investigation_status[key] = status

    return investigation_status


# ============================================================
# HUMAN DUPLICATE REVIEW
# ============================================================

def load_human_review():

    if not HUMAN_REVIEW_FILE.exists():

        print()
        print(
            "[INFO] No human duplicate-review file found."
        )

        print(
            f"[INFO] Expected file: {HUMAN_REVIEW_FILE}"
        )

        return pd.DataFrame(
            columns=[
                "incoming_case_id",
                "decision"
            ]
        )

    review_df = pd.read_csv(
        HUMAN_REVIEW_FILE
    )

    required_columns = {
        "incoming_case_id",
        "decision"
    }

    missing_columns = (
        required_columns
        - set(review_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Human review file is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    review_df["decision"] = (
        review_df["decision"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_decisions = review_df[
        ~review_df["decision"].isin(
            ["NEW", "DUPLICATE"]
        )
    ]

    if not invalid_decisions.empty:

        raise ValueError(
            "Invalid human review decision found. "
            "Allowed values are NEW or DUPLICATE."
        )

    return review_df


def determine_narrative_eligibility(
    case_id,
    case_classification,
    human_review_df
):
    """
    Determine whether narrative generation is currently allowed.

    Rules:
        Low similarity
            -> eligible

        Potential/high-confidence duplicate
            -> requires human review

        Human decision NEW
            -> eligible

        Human decision DUPLICATE
            -> blocked
    """

    review_rows = human_review_df[
        human_review_df["incoming_case_id"] == case_id
    ]

    if not review_rows.empty:

        decision = review_rows.iloc[-1]["decision"]

        if decision == "NEW":
            return (
                "ELIGIBLE",
                "Human PV review confirmed NEW case"
            )

        if decision == "DUPLICATE":
            return (
                "BLOCKED_DUPLICATE",
                "Human PV review confirmed DUPLICATE"
            )

    if case_classification == "Low similarity":

        return (
            "ELIGIBLE",
            "No significant duplicate signal"
        )

    return (
        "PENDING_HUMAN_REVIEW",
        "Potential duplicate requires qualified PV review"
    )


# ============================================================
# CASE-LEVEL FINAL RESULTS
# ============================================================

def build_final_results(
    incoming_df,
    validation_df,
    combined_df,
    human_review_df
):

    rows = []

    for _, incoming_case in incoming_df.iterrows():

        case_id = incoming_case["case_id"]

        validation_row = validation_df[
            validation_df["case_id"] == case_id
        ]

        if validation_row.empty:

            rows.append({
                "case_id": case_id,
                "validation_status": "UNKNOWN",
                "duplicate_status": "NOT_PROCESSED",
                "narrative_eligibility": "BLOCKED",
                "narrative_status": "NOT_GENERATED",
                "reason": "Validation result unavailable"
            })

            continue

        is_valid = bool(
            validation_row.iloc[0]["is_valid"]
        )

        if not is_valid:

            rows.append({
                "case_id": case_id,
                "validation_status": "INVALID",
                "duplicate_status": "EXCLUDED",
                "narrative_eligibility": "BLOCKED",
                "narrative_status": "EXCLUDED",
                "reason": "Invalid ICSR"
            })

            continue

        case_candidates = combined_df[
            combined_df["incoming_case_id"] == case_id
        ].copy()

        if case_candidates.empty:

            duplicate_status = "No candidate"

        else:

            duplicate_status = (
                case_candidates.iloc[0]["classification"]
            )

        eligibility, eligibility_reason = (
            determine_narrative_eligibility(
                case_id,
                duplicate_status,
                human_review_df
            )
        )

        human_decision = ""

        review_rows = human_review_df[
            human_review_df["incoming_case_id"] == case_id
        ]

        if not review_rows.empty:
            human_decision = (
                review_rows.iloc[-1]["decision"]
            )

        if eligibility == "ELIGIBLE":

            narrative_status = "NOT_GENERATED - ON DEMAND"

        elif eligibility == "BLOCKED_DUPLICATE":

            narrative_status = "BLOCKED - DUPLICATE"

        else:

            narrative_status = (
                "BLOCKED - HUMAN DUPLICATE REVIEW REQUIRED"
            )

        rows.append({

            "case_id": case_id,

            "validation_status": "VALID",

            "duplicate_status": duplicate_status,

            "human_duplicate_decision": human_decision,

            "narrative_eligibility": eligibility,

            "narrative_status": narrative_status,

            "narrative_qc_status": "NOT_RUN",

            "reason": eligibility_reason
        })

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def print_final_summary(final_df):

    print_section(
        "PV_NarrativeGuard - FINAL PIPELINE SUMMARY"
    )

    total = len(final_df)

    valid = len(
        final_df[
            final_df["validation_status"] == "VALID"
        ]
    )

    invalid = len(
        final_df[
            final_df["validation_status"] == "INVALID"
        ]
    )

    duplicate_review = len(
        final_df[
            final_df["narrative_eligibility"]
            == "PENDING_HUMAN_REVIEW"
        ]
    )

    eligible = len(
        final_df[
            final_df["narrative_eligibility"]
            == "ELIGIBLE"
        ]
    )

    blocked_duplicate = len(
        final_df[
            final_df["narrative_eligibility"]
            == "BLOCKED_DUPLICATE"
        ]
    )

    print(
        f"Total incoming ICSRs       : {total}"
    )

    print(
        f"Valid ICSRs                : {valid}"
    )

    print(
        f"Invalid ICSRs              : {invalid}"
    )

    print(
        f"Duplicate review required  : {duplicate_review}"
    )

    print(
        f"Narrative eligible         : {eligible}"
    )

    print(
        f"Confirmed duplicates       : {blocked_duplicate}"
    )

    print(
        "Narratives generated       : 0 "
        "(on-demand)"
    )

    print()
    print(
        f"Final output saved to:"
    )
    print(FINAL_FILE)

    print()
    print(
        "Narratives are NOT generated automatically."
    )

    print(
        "Use the Streamlit interface to generate "
        "eligible narratives on demand."
    )

    print()
    print(
        "IMPORTANT: AI duplicate investigation is "
        "assistive only."
    )

    print(
        "Qualified human pharmacovigilance review "
        "is required before final use."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    print_section(
        "PV_NarrativeGuard - END-TO-END BACKEND PIPELINE"
    )

    ensure_output_directory()

    print()
    print("Loading ICSR datasets...")

    incoming_df = pd.read_csv(
        INCOMING_FILE
    )

    historical_df = pd.read_csv(
        HISTORICAL_FILE
    )

    print(
        f"Incoming cases  : {len(incoming_df)}"
    )

    print(
        f"Historical cases: {len(historical_df)}"
    )

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    print_section(
        "STAGE 1 - ICSR VALIDATION"
    )

    validation_df = run_validation(
        INCOMING_FILE
    )

    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    print_section(
        "STAGE 2 - STRUCTURED DUPLICATE SCREENING"
    )

    structured_df = run_structured_screening(
        incoming_df,
        historical_df,
        validation_df
    )

    # --------------------------------------------------------
    # STAGE 3
    # --------------------------------------------------------

    print_section(
        "STAGE 3 - SEMANTIC SIMILARITY"
    )

    semantic_df = run_semantic_similarity(
        incoming_df,
        historical_df,
        validation_df
    )

    # --------------------------------------------------------
    # STAGE 4
    # --------------------------------------------------------

    print_section(
        "STAGE 4 - COMBINED DUPLICATE SCORING"
    )

    combined_df = run_combined_scoring(
        structured_df,
        semantic_df
    )

    # --------------------------------------------------------
    # STAGE 5
    # --------------------------------------------------------

    print_section(
        "STAGE 5 - RAG DUPLICATE INVESTIGATION"
    )

    investigate_review_candidates(
        combined_df
    )

    # --------------------------------------------------------
    # STAGE 6
    # --------------------------------------------------------

    print_section(
        "STAGE 6 - HUMAN DUPLICATE REVIEW GATE"
    )

    human_review_df = load_human_review()

    if human_review_df.empty:

        print(
            "No human decisions recorded."
        )

        print(
            "Potential duplicate cases will remain "
            "PENDING_HUMAN_REVIEW."
        )

    else:

        print(
            f"Human review decisions loaded : "
            f"{len(human_review_df)}"
        )

    # --------------------------------------------------------
    # STAGE 7
    # --------------------------------------------------------

    print_section(
        "STAGE 7 - CASE ELIGIBILITY"
    )

    final_df = build_final_results(
        incoming_df,
        validation_df,
        combined_df,
        human_review_df
    )

    final_df.to_csv(
        FINAL_FILE,
        index=False
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_final_summary(
        final_df
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_pipeline()

    except Exception as exc:

        print()
        print("=" * 70)
        print("PIPELINE ERROR")
        print("=" * 70)
        print(str(exc))

        raise