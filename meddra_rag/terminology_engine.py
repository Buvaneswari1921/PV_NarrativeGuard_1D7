import re
import pandas as pd
from pathlib import Path


# ============================================================
# PV NarrativeGuard - Terminology Decision Engine
# ============================================================
#
# FAISS = candidate retrieval
# This module = terminology decision / validation
#
# Decision states:
#   SUPPORTED
#   AMBIGUOUS
#   OUT_OF_CONTEXT
#
# Important:
# The engine must NOT select a more specific PT when the
# user has not supplied that specificity.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

METADATA_FILE = (
    BASE_DIR
    / "vectorstore"
    / "oae_metadata.csv"
)

CONTROLLED_INVENTORY_FILE = (
    BASE_DIR
    / "data"
    / "OAE_MedDRA_Deterministic_Decision_Inventory.csv"
)

PV_CORE_FILE = (
    BASE_DIR
    / "data"
    / "pv_core_150.csv"
)


# ============================================================
# Configuration
# ============================================================

MIN_SIMILARITY = 0.45
STRONG_SIMILARITY = 0.65
AMBIGUITY_MARGIN = 0.035


# ============================================================
# Controlled lay-language mappings
# ============================================================

KNOWN_TERM_MAPPINGS = {

    # GI
    # NOTE:
    # The Validated MedDRA Mapping Dataset will eventually
    # establish the formal loose stools -> Diarrhea mapping.
    # Do not alter this legacy mapping in this integration.
    "loose stools": "loose stools",
    "loose stool": "loose stools",

    "diarrhea": "loose stools",
    "diarrhoea": "loose stools",

    # Nausea / vomiting
    "feels like vomiting": "nausea",
    "feel like vomiting": "nausea",
    "feeling like vomiting": "nausea",

    # Pruritus
    "itchy skin": "pruritus",
    "skin itching": "pruritus",
    "skin itch": "pruritus",
    "itching": "pruritus",
    "itch": "pruritus",

    # Dyspnea
    "difficulty breathing": "dyspnea",
    "difficulty in breathing": "dyspnea",
    "shortness of breath": "dyspnea",
    "breathlessness": "dyspnea",

    # Lightheadedness
    "feels lightheaded": "lightheadedness",
    "feel lightheaded": "lightheadedness",
    "lightheaded": "lightheadedness",

    # Lymphadenopathy
    "swollen lymph nodes": "lymphadenopathy",
    "swollen lymph node": "lymphadenopathy",

    # General disease terms
    "meningitis": "meningitis",
    "encephalitis": "encephalitis",
    "abscess": "abscess",
    "parotitis": "parotitis",
    "thrombosis": "thrombosis",
}


# ============================================================
# Negation protection
# ============================================================

NEGATION_CUES = {
    "denies", "deny", "denied", "no", "not", "without", "never", "negative",
}


def is_negated_query_expression(user_input, pv_core_match):
    """Detect simple local negation before a curated clinical expression."""
    if not pv_core_match:
        return False
    q = normalize_text(user_input)
    alias = normalize_text(pv_core_match.get("alias", ""))
    if not q or not alias:
        return False
    start = q.find(alias)
    if start < 0:
        return False
    return any(token in NEGATION_CUES for token in q[:start].split()[-4:])


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\bae\b",
        "",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# Controlled OAE decision inventory
# ============================================================

def load_controlled_inventory():

    if not CONTROLLED_INVENTORY_FILE.exists():
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            CONTROLLED_INVENTORY_FILE,
            low_memory=False
        )

        required_columns = {
            "Preferred Label",
            "Synonyms",
            "candidate_meddra_id",
            "controlled_decision",
            "controlled_decision_reason",
        }

        if not required_columns.issubset(
            set(df.columns)
        ):
            return pd.DataFrame()

        df = df[
            list(required_columns)
        ].copy()

        df[
            "Preferred Label"
        ] = df[
            "Preferred Label"
        ].fillna("").astype(str)

        df[
            "Synonyms"
        ] = df[
            "Synonyms"
        ].fillna("").astype(str)

        df[
            "controlled_decision"
        ] = df[
            "controlled_decision"
        ].fillna("").astype(str)

        df[
            "controlled_decision_reason"
        ] = df[
            "controlled_decision_reason"
        ].fillna("").astype(str)

        df[
            "normalized_label"
        ] = df[
            "Preferred Label"
        ].map(
            normalize_text
        )

        return df

    except Exception:
        return pd.DataFrame()


def load_pv_core_vocabulary():
    """Load the curated high-frequency PV vocabulary."""
    if not PV_CORE_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PV_CORE_FILE, low_memory=False)
        required = {"core_id", "preferred_label", "meddra_id", "category", "pv_query_aliases"}
        if not required.issubset(df.columns):
            return pd.DataFrame()
        for col in ["preferred_label", "category", "pv_query_aliases",
                    "source_synonyms", "example_pv_expressions", "coding_note"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        df["normalized_label"] = df["preferred_label"].map(normalize_text)
        return df
    except Exception:
        return pd.DataFrame()


def get_pv_core_alias_match(user_input, vocabulary):
    """Find the longest curated PV alias in a query.

    This is a normalization hint only. Final selection still requires
    an underlying retrieved terminology candidate.
    """
    if vocabulary is None or vocabulary.empty:
        return None
    normalized = normalize_text(user_input)
    if not normalized:
        return None

    matches = []
    for _, row in vocabulary.iterrows():
        for alias in str(row.get("pv_query_aliases", "")).split("|"):
            alias_norm = normalize_text(alias)
            if not alias_norm:
                continue
            if normalized == alias_norm or re.search(
                rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])",
                normalized,
            ):
                matches.append({
                    "alias": alias_norm,
                    "preferred_label": str(row.get("preferred_label", "")).strip(),
                    "meddra_id": row.get("meddra_id"),
                    "core_id": row.get("core_id"),
                    "category": row.get("category", ""),
                    "coding_note": row.get("coding_note", ""),
                    "match_length": len(alias_norm),
                })

    if not matches:
        return None

    matches.sort(
        key=lambda x: (x["match_length"], len(x["preferred_label"])),
        reverse=True,
    )
    return matches[0]


def find_pv_core_candidate(core_match, candidates):
    """Find the retrieved candidate corresponding to a PV core concept."""
    if not core_match:
        return None
    target_id = str(core_match.get("meddra_id", "")).strip()
    if target_id and target_id.lower() != "nan":
        for candidate in candidates:
            if str(candidate.get("meddra_id", "")).strip() == target_id:
                return candidate

    target_label = normalize_text(core_match.get("preferred_label", ""))
    if target_label:
        for candidate in candidates:
            if normalize_text(candidate.get("preferred_label", "")) == target_label:
                return candidate
    return None


def get_controlled_inventory_match(
    user_input,
    inventory
):

    if inventory is None or inventory.empty:
        return None

    normalized = normalize_text(
        user_input
    )

    if not normalized:
        return None

    exact = inventory[
        inventory[
            "normalized_label"
        ] == normalized
    ]

    if exact.empty:
        return None

    # ========================================================
    # CONTROLLED DECISION PRECEDENCE
    #
    # Multiple inventory rows may exist for the same
    # normalized terminology label.
    #
    # If ANY matching row is AMBIGUOUS, ambiguity takes
    # precedence over SAFE_EXACT / SAFE_ALIAS.
    # ========================================================

    decisions = (
        exact[
            "controlled_decision"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ========================================================
    # STEP 1
    # AMBIGUOUS takes precedence
    # ========================================================

    ambiguous_rows = exact[
        decisions == "AMBIGUOUS"
    ]

    if not ambiguous_rows.empty:

        reasons = (
            ambiguous_rows[
                "controlled_decision_reason"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        reasons = [
            reason
            for reason in reasons
            if reason
        ]

        return {
            "controlled_decision":
                "AMBIGUOUS",

            "controlled_decision_reason":
                "; ".join(
                    dict.fromkeys(
                        reasons
                    )
                ),

            "inventory_match_count":
                len(exact),
        }

    # ========================================================
    # STEP 2
    # SAFE_EXACT / SAFE_ALIAS
    #
    # Only used when there is no ambiguity.
    # ========================================================

    safe_rows = exact[
        decisions.isin(
            {
                "SAFE_EXACT",
                "SAFE_ALIAS",
            }
        )
    ]

    if not safe_rows.empty:

        exact_rows = safe_rows[
            decisions.loc[
                safe_rows.index
            ] == "SAFE_EXACT"
        ]

        if not exact_rows.empty:

            row = exact_rows.iloc[0]

        else:

            row = safe_rows.iloc[0]

        return row.to_dict()

    return None


# ============================================================
# Remove conversational language
# ============================================================

NON_CLINICAL_WORDS = {

    "patient",
    "developed",
    "develop",
    "has",
    "have",
    "had",
    "having",

    "feels",
    "feel",
    "felt",

    "reports",
    "reported",

    "experiencing",
    "experienced",

    "complains",
    "complained",

    "of",
    "with",
    "the",
    "a",
    "an",

    "is",
    "was",
    "were",

    "and",
    "or",
}


def clinical_tokens(text):

    normalized = normalize_text(text)

    if not normalized:
        return set()

    tokens = set(
        normalized.split()
    )

    return (
        tokens
        - NON_CLINICAL_WORDS
    )


# ============================================================
# Specificity qualifiers
# ============================================================

QUALIFIER_WORDS = {

    "viral",
    "bacterial",
    "fungal",
    "parasitic",

    "allergic",

    "acute",
    "chronic",

    "severe",
    "mild",
    "moderate",

    "congenital",
    "neonatal",

    "postoperative",

    "immunisation",
    "immunization",

    "post",
    "varicella",

    "injection",

    "right",
    "left",
    "bilateral",
    "unilateral",

    "mediastinal",

    "hilar",

    "abdominal",
    "anal",
    "perianal",

    "mesenteric",

    "limb",

    "procedural",

    "postpartum",

    "herpes",
    "herpetic",

    "sterile",

    "pneumococcal",
    "meningococcal",

    "tuberculous",
    "mumps",
}


def get_qualifiers(text):

    return (
        clinical_tokens(text)
        & QUALIFIER_WORDS
    )


def has_unsupported_specificity(
    user_input,
    candidate_label
):

    candidate_qualifiers = (
        get_qualifiers(
            candidate_label
        )
    )

    user_qualifiers = (
        get_qualifiers(
            user_input
        )
    )

    unsupported = (
        candidate_qualifiers
        - user_qualifiers
    )

    return len(
        unsupported
    ) > 0


# ============================================================
# Detect general candidate
# ============================================================

def candidate_is_general(
    candidate_label
):

    return len(
        get_qualifiers(
            candidate_label
        )
    ) == 0


# ============================================================
# Exact / synonym matching
# ============================================================

def lexical_match(
    user_input,
    candidate
):

    user_text = normalize_text(
        user_input
    )

    preferred = normalize_text(
        candidate.get(
            "preferred_label",
            ""
        )
    )

    if user_text == preferred:

        return True

    synonyms = candidate.get(
        "synonyms",
        ""
    )

    if pd.isna(
        synonyms
    ):

        synonyms = ""

    for synonym in str(
        synonyms
    ).split("|"):

        if user_text == normalize_text(
            synonym
        ):

            return True

    return False


# ============================================================
# Phrase matching after removing conversational language
# ============================================================

def clinical_phrase(
    text
):

    tokens = clinical_tokens(
        text
    )

    return " ".join(
        sorted(tokens)
    )


def clinical_phrase_match(
    user_input,
    candidate
):

    user_phrase = clinical_phrase(
        user_input
    )

    preferred_phrase = clinical_phrase(
        candidate.get(
            "preferred_label",
            ""
        )
    )

    if (
        user_phrase
        and preferred_phrase
        and user_phrase
        == preferred_phrase
    ):

        return True

    synonyms = candidate.get(
        "synonyms",
        ""
    )

    if pd.isna(
        synonyms
    ):

        synonyms = ""

    for synonym in str(
        synonyms
    ).split("|"):

        if (
            clinical_phrase(
                synonym
            )
            == user_phrase
        ):

            return True

    return False


# ============================================================
# Known mapping lookup
# ============================================================

def get_known_mapping(
    user_input
):

    normalized = normalize_text(
        user_input
    )

    normalized = re.sub(
        r"^patient\s+",
        "",
        normalized
    )

    normalized = re.sub(
        r"^(developed|has|have|had|feels|feel)\s+",
        "",
        normalized
    )

    normalized = normalized.strip()

    return KNOWN_TERM_MAPPINGS.get(
        normalized
    )


# ============================================================
# Find candidate matching a known PT
# ============================================================

def find_known_pt_candidate(
    target_pt,
    candidates
):

    target = normalize_text(
        target_pt
    )

    for candidate in candidates:

        preferred = normalize_text(
            candidate.get(
                "preferred_label",
                ""
            )
        )

        if preferred == target:

            return candidate

    for candidate in candidates:

        synonyms = candidate.get(
            "synonyms",
            ""
        )

        if pd.isna(
            synonyms
        ):

            synonyms = ""

        for synonym in str(
            synonyms
        ).split("|"):

            if normalize_text(
                synonym
            ) == target:

                return candidate

    return None


# ============================================================
# Candidate scoring
# ============================================================

def calculate_candidate_score(
    user_input,
    candidate
):

    similarity = float(
        candidate.get(
            "similarity_score",
            0
        )
    )

    score = similarity

    if candidate.get("curated_core_candidate", False):
        # This is a deterministic curated-match score, not a fabricated
        # embedding similarity.  The raw similarity remains 0.0 when the
        # candidate was supplied by the curated resolver rather than FAISS.
        score = max(
            score,
            float(candidate.get("curated_match_score", 0.90)),
        )

    if lexical_match(
        user_input,
        candidate
    ):

        score += 0.30

    elif clinical_phrase_match(
        user_input,
        candidate
    ):

        score += 0.25

    if has_unsupported_specificity(
        user_input,
        candidate.get(
            "preferred_label",
            ""
        )
    ):

        score -= 0.30

    # Decision score is a confidence-like value consumed by
    # downstream UI and response layers. Keep the public score
    # strictly within the normalized [0.0, 1.0] range.
    return max(
        0.0,
        min(
            1.0,
            float(score),
        ),
    )


# ============================================================
# Evaluate candidates
# ============================================================

def evaluate_candidates(
    user_input,
    candidates
):

    evaluated = []

    for candidate in candidates:

        similarity = float(
            candidate.get(
                "similarity_score",
                0
            )
        )

        if (
            similarity < MIN_SIMILARITY
            and not candidate.get("curated_core_candidate", False)
        ):

            continue

        item = dict(
            candidate
        )

        item[
            "decision_score"
        ] = calculate_candidate_score(
            user_input,
            candidate
        )

        item[
            "lexical_match"
        ] = lexical_match(
            user_input,
            candidate
        )

        item[
            "clinical_phrase_match"
        ] = clinical_phrase_match(
            user_input,
            candidate
        )

        item[
            "unsupported_specificity"
        ] = has_unsupported_specificity(
            user_input,
            candidate.get(
                "preferred_label",
                ""
            )
        )

        item[
            "general_term"
        ] = candidate_is_general(
            candidate.get(
                "preferred_label",
                ""
            )
        )

        evaluated.append(
            item
        )

    evaluated.sort(
        key=lambda x:
            x["decision_score"],
        reverse=True
    )

    return evaluated


# ============================================================
# MAIN DECISION FUNCTION
# ============================================================

def decide(
    user_input,
    candidates
):

    if not user_input or not str(
        user_input
    ).strip():

        return {
            "status":
                "OUT_OF_CONTEXT",

            "message":
                "This is out of my context.",

            "result":
                None,

            "candidates":
                []
        }

    controlled_inventory = (
        load_controlled_inventory()
    )

    pv_core_vocabulary = (
        load_pv_core_vocabulary()
    )

    pv_core_match = (
        get_pv_core_alias_match(
            user_input,
            pv_core_vocabulary,
        )
    )

    controlled_match = (
        get_controlled_inventory_match(
            user_input,
            controlled_inventory
        )
    )

    if is_negated_query_expression(user_input, pv_core_match):
        return {
            "status": "NEGATED",
            "message": (
                "The matched clinical expression is explicitly negated; "
                "it should not be treated as a reported adverse event."
            ),
            "result": None,
            "candidates": [],
        }

    if not candidates:

        return {
            "status":
                "OUT_OF_CONTEXT",

            "message":
                "This is out of my context.",

            "result":
                None,

            "candidates":
                []
        }

    evaluated = evaluate_candidates(
        user_input,
        candidates
    )

    if not evaluated:

        return {
            "status":
                "OUT_OF_CONTEXT",

            "message":
                "This is out of my context.",

            "result":
                None,

            "candidates":
                []
        }

    # ========================================================
    # STEP 0
    # Curated PV Core Vocabulary normalization
    # ========================================================

    if pv_core_match:
        # Some lay expressions are intentionally not one-to-one coding
        # decisions.  Preserve the curated candidates but require context.
        ambiguous_aliases = {
            "fast heartbeat",
            "rapid heartbeat",
        }

        if pv_core_match.get("alias") in ambiguous_aliases:
            ambiguity_candidates = [
                item
                for item in evaluated
                if item.get("ambiguity_candidate")
                or normalize_text(item.get("preferred_label", ""))
                in {"tachycardia ae", "palpitations ae"}
            ]
            if len(ambiguity_candidates) >= 2:
                return {
                    "status": "AMBIGUOUS",
                    "message": (
                        "The expression describes a rapid-heartbeat complaint "
                        "but does not establish a measured elevated heart rate; "
                        "palpitations and tachycardia may both be relevant. "
                        "Additional clinical context is required before coding."
                    ),
                    "result": None,
                    "candidates": ambiguity_candidates[:5],
                }
        core_candidate = find_pv_core_candidate(
            pv_core_match,
            evaluated,
        )

        if core_candidate:
            core_candidate = dict(core_candidate)
            core_candidate["match_method"] = "pv_core_alias"
            core_candidate["matched_pv_expression"] = pv_core_match["alias"]
            core_candidate["pv_core_id"] = pv_core_match["core_id"]
            core_candidate["pv_category"] = pv_core_match["category"]
            core_candidate["coding_note"] = pv_core_match["coding_note"]
            core_candidate["decision_score"] = max(
                core_candidate["decision_score"],
                float(core_candidate.get("curated_match_score", 0.90)),
                core_candidate["similarity_score"] + 0.20,
            )

            return {
                "status": "SUPPORTED",
                "message": "Terminology match found.",
                "result": core_candidate,
                "candidates": evaluated[:5],
            }

    # ========================================================
    # STEP 1
    # Known terminology mapping
    # ========================================================

    known_pt = get_known_mapping(
        user_input
    )

    if known_pt:

        known_candidate = (
            find_known_pt_candidate(
                known_pt,
                evaluated
            )
        )

        if known_candidate:

            known_candidate = dict(
                known_candidate
            )

            known_candidate[
                "match_method"
            ] = "known_mapping"

            known_candidate[
                "mapped_term"
            ] = known_pt

            known_candidate[
                "decision_score"
            ] = max(
                known_candidate[
                    "decision_score"
                ],
                known_candidate[
                    "similarity_score"
                ] + 0.20
            )

            return {
                "status":
                    "SUPPORTED",

                "message":
                    "Terminology match found.",

                "result":
                    known_candidate,

                "candidates":
                    evaluated[:5]
            }

    # ========================================================
    # STEP 1.5
    # Controlled OAE decision inventory
    # ========================================================

    if controlled_match:

        controlled_decision = str(
            controlled_match.get(
                "controlled_decision",
                ""
            )
        ).upper()

        if controlled_decision == "AMBIGUOUS":

            return {
                "status":
                    "AMBIGUOUS",

                "message":
                    "The terminology mapping is ambiguous and requires review.",

                "result":
                    None,

                "candidates":
                    evaluated[:5]
            }

        if controlled_decision in {
            "SAFE_EXACT",
            "SAFE_ALIAS",
        }:

            controlled_id = (
                controlled_match.get(
                    "candidate_meddra_id"
                )
            )

            for candidate in evaluated:

                candidate_id = candidate.get(
                    "meddra_id"
                )

                if (
                    controlled_id
                    and str(candidate_id)
                    == str(controlled_id)
                ):

                    controlled_candidate = dict(
                        candidate
                    )

                    controlled_candidate[
                        "match_method"
                    ] = (
                        "controlled_inventory"
                    )

                    controlled_candidate[
                        "controlled_decision"
                    ] = controlled_decision

                    controlled_candidate[
                        "controlled_decision_reason"
                    ] = (
                        controlled_match.get(
                            "controlled_decision_reason",
                            ""
                        )
                    )

                    return {
                        "status":
                            "SUPPORTED",

                        "message":
                            "Terminology match found.",

                        "result":
                            controlled_candidate,

                        "candidates":
                            evaluated[:5]
                    }

    # ========================================================
    # STEP 2
    # Exact lexical match
    # ========================================================

    lexical_candidates = [

        item
        for item in evaluated

        if (
            item[
                "lexical_match"
            ]
            and not item[
                "unsupported_specificity"
            ]
        )
    ]

    if lexical_candidates:

        lexical_candidates.sort(
            key=lambda x:
                x["similarity_score"],
            reverse=True
        )

        best = lexical_candidates[0]

        return {
            "status":
                "SUPPORTED",

            "message":
                "Terminology match found.",

            "result":
                best,

            "candidates":
                evaluated[:5]
        }

    # ========================================================
    # STEP 3
    # Clinical phrase match
    # ========================================================

    phrase_candidates = [

        item
        for item in evaluated

        if (
            item[
                "clinical_phrase_match"
            ]
            and not item[
                "unsupported_specificity"
            ]
        )
    ]

    if phrase_candidates:

        best = phrase_candidates[0]

        return {
            "status":
                "SUPPORTED",

            "message":
                "Terminology match found.",

            "result":
                best,

            "candidates":
                evaluated[:5]
        }

    # ========================================================
    # STEP 4
    # Specificity protection
    # ========================================================

    supported_candidates = [

        item
        for item in evaluated

        if not item[
            "unsupported_specificity"
        ]
    ]

    if not supported_candidates:

        return {
            "status":
                "AMBIGUOUS",

            "message":
                "The available terminology candidates require additional specificity.",

            "result":
                None,

            "candidates":
                evaluated[:5]
        }

    best = supported_candidates[0]

    # ========================================================
    # STEP 5
    # Similarity / ambiguity decision
    # ========================================================

    best_score = float(
        best[
            "decision_score"
        ]
    )

    second_score = 0.0

    if len(
        supported_candidates
    ) > 1:

        second_score = float(
            supported_candidates[1][
                "decision_score"
            ]
        )

    margin = (
        best_score
        - second_score
    )

    if (
        best_score >= STRONG_SIMILARITY
        and margin >= AMBIGUITY_MARGIN
    ):

        return {
            "status":
                "SUPPORTED",

            "message":
                "Terminology match found.",

            "result":
                best,

            "candidates":
                evaluated[:5]
        }

    return {
        "status":
            "AMBIGUOUS",

        "message":
            "The terminology candidates are ambiguous.",

        "result":
            None,

        "candidates":
            evaluated[:5]
    }


# ============================================================
# Backward-compatible engine wrapper
# ============================================================

class TerminologyEngine:

    def __init__(
        self,
        metadata_file=METADATA_FILE
    ):

        self.metadata_file = Path(
            metadata_file
        )

        self.metadata = (
            pd.read_csv(
                self.metadata_file
            )
            if self.metadata_file.exists()
            else pd.DataFrame()
        )

        self.controlled_inventory = (
            load_controlled_inventory()
        )

    def decide(
        self,
        user_input,
        candidates
    ):

        return decide(
            user_input,
            candidates
        )

