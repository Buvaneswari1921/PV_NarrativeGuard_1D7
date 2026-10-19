"""
PV Narrative Generation v4
==========================

Source-grounded ICSR narrative generation.

Key design:
    Structured ICSR facts
        -> deterministic clinical narrative
        -> optional controlled LLM editing
        -> final narrative

The deterministic layer owns factual content and sentence construction.
The LLM, when configured, is only an editor. This prevents raw source phrases
from being inserted into grammatically incorrect wrappers.

Public interfaces:
    generate_narrative(case_id)
    generate_case_narrative(case_id)
    generate_narratives_for_cases(case_ids)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INCOMING_FILE = BASE_DIR / "data" / "incoming_icsrs.csv"


FIELD_ALIASES = {
    "case_id": ["case_id", "caseid", "icsr_id", "report_id"],
    "patient_age": ["patient_age", "age", "patient age"],
    "patient_sex": ["patient_sex", "sex", "gender"],
    "country": ["country", "patient_country", "country_of_patient"],
    "suspect_drug": [
        "suspect_drug", "suspect medicinal product", "suspect_drug_name",
        "drug", "medication",
    ],
    "indication": ["indication", "reason_for_use", "therapy_indication"],
    "adverse_event": [
        "adverse_event", "adverse event", "event", "reaction", "reaction_term",
    ],
    "event_date": ["event_date", "reaction_start_date", "onset_date"],
    "seriousness": ["seriousness", "serious", "seriousness_criteria"],
    "outcome": ["outcome", "event_outcome"],
    "reporter_type": ["reporter_type", "reporter", "reporter_qualification"],
    "dose": ["dose", "dose_amount", "dose_value"],
    "dose_unit": ["dose_unit", "dose_units", "dose_uom"],
    "route": ["route", "route_of_administration"],
    "frequency": ["frequency", "dose_frequency", "regimen"],
    "therapy_start_date": [
        "therapy_start_date", "treatment_start_date",
        "drug_start_date", "start_date",
    ],
    "therapy_stop_date": [
        "therapy_stop_date", "treatment_stop_date",
        "drug_stop_date", "stop_date",
    ],
    "duration": ["duration", "treatment_duration"],
    "medical_history": [
        "medical_history", "relevant_medical_history", "past_medical_history",
    ],
    "concomitant_medications": [
        "concomitant_medications", "concomitant_drugs",
        "concomitant_medicines",
    ],
    "investigations": [
        "investigations", "laboratory_results", "lab_results",
        "diagnostic_results", "test_results",
    ],
    "diagnosis": ["diagnosis", "clinical_diagnosis"],
    "action_taken": [
        "action_taken", "action_taken_with_drug", "drug_action", "action",
    ],
    "treatment": [
        "treatment", "therapeutic_measures",
        "treatment_for_event", "management",
    ],
    "follow_up": [
        "follow_up", "followup", "follow_up_information",
        "additional_information",
    ],
    "case_description": [
        "case_description", "description",
        "clinical_description", "narrative_source",
    ],
}

MISSING_VALUES = {
    "", "nan", "none", "null", "n/a", "na",
    "not available", "not reported", "unknown",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in MISSING_VALUES


def _clean(value: Any) -> str:
    if _is_missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _get_value(case: Dict[str, Any], field: str) -> str:
    normalized = {_key(k): v for k, v in case.items()}
    for alias in FIELD_ALIASES.get(field, [field]):
        value = _clean(normalized.get(_key(alias)))
        if value:
            return value
    return ""


def _parse_date(value: str) -> Optional[pd.Timestamp]:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return parsed if pd.notna(parsed) else None


def _format_date(value: str) -> str:
    parsed = _parse_date(value)
    return parsed.strftime("%d %B %Y") if parsed is not None else value


def _format_age(value: str) -> str:
    if not value:
        return ""
    if re.fullmatch(r"\d+(\.\d+)?", value):
        # Avoid changing an integer source value to 50.0 etc.
        if "." in value:
            return f"{value}-year-old"
        return f"{int(value)}-year-old"
    return value


def _strip_terminal_punctuation(value: str) -> str:
    return re.sub(r"[.!?]+$", "", value.strip())


def _lower_first(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    return value[:1].lower() + value[1:]


def extract_case_facts(case: Dict[str, Any]) -> Dict[str, str]:
    """Normalize source fields without clinical interpretation."""
    return {field: _get_value(case, field) for field in FIELD_ALIASES}


def classify_case_complexity(facts: Dict[str, str]) -> str:
    """Classify narrative depth from available source information."""
    if facts.get("follow_up"):
        return "follow-up"

    optional = [
        "medical_history", "concomitant_medications", "investigations",
        "diagnosis", "action_taken", "treatment",
        "therapy_start_date", "therapy_stop_date", "duration",
        "dose", "route", "frequency",
    ]
    count = sum(bool(facts.get(x)) for x in optional)

    if count >= 7:
        return "complex"
    if count >= 3:
        return "moderate"
    return "standard"


# ---------------------------------------------------------------------------
# Controlled sentence construction
# ---------------------------------------------------------------------------

def _build_therapy_phrase(facts: Dict[str, str]) -> str:
    drug = facts.get("suspect_drug", "")
    dose = facts.get("dose", "")
    unit = facts.get("dose_unit", "")
    route = facts.get("route", "")
    frequency = facts.get("frequency", "")
    indication = facts.get("indication", "")

    if not drug:
        phrase = "the suspect medicinal product"
    else:
        phrase = drug

    if dose:
        phrase += f" {dose}"
        if unit:
            phrase += f" {unit}"

    # Convert common administration forms into natural narrative wording.
    if route:
        route_l = route.strip().lower()
        route_map = {
            "subcutaneous": "subcutaneously",
            "intravenous": "intravenously",
            "intramuscular": "intramuscularly",
            "oral": "orally",
            "topical": "topically",
            "inhaled": "by inhalation",
        }
        phrase += f" {route_map.get(route_l, 'by ' + route)}"

    if frequency:
        phrase += f" {frequency}"

    if indication:
        phrase += f" for {indication}"

    return phrase


def _build_patient_therapy_paragraph(facts: Dict[str, str]) -> str:
    age = _format_age(facts.get("patient_age", ""))
    sex = facts.get("patient_sex", "")
    country = facts.get("country", "")

    descriptors = []
    if age:
        descriptors.append(age)
    if sex and sex.lower() != "unknown":
        descriptors.append(sex.lower())

    if descriptors:
        subject = "A " + " ".join(descriptors) + " patient"
    else:
        subject = "The patient"

    if country:
        if country.lower().startswith(("the ", "united ")):
            subject += f" from {country}"
        else:
            subject += f" from {country}"

    therapy = _build_therapy_phrase(facts)
    paragraph = f"{subject} was receiving {therapy}."

    if facts.get("therapy_start_date"):
        paragraph += (
            f" Treatment with the suspect medicinal product was reported "
            f"to have started on {_format_date(facts['therapy_start_date'])}."
        )

    if facts.get("therapy_stop_date"):
        paragraph += (
            f" Treatment was reported to have stopped on "
            f"{_format_date(facts['therapy_stop_date'])}."
        )

    if facts.get("duration"):
        paragraph += f" The reported treatment duration was {facts['duration']}."

    return paragraph


def _build_background_paragraph(facts: Dict[str, str]) -> str:
    sentences = []

    if facts.get("medical_history"):
        sentences.append(
            f"Relevant medical history included {facts['medical_history']}."
        )

    if facts.get("concomitant_medications"):
        sentences.append(
            f"Concomitant medications included "
            f"{facts['concomitant_medications']}."
        )

    return " ".join(sentences)


def _build_event_paragraph(facts: Dict[str, str]) -> str:
    event = facts.get("adverse_event", "")
    if not event:
        return ""

    event_text = _lower_first(event)
    sentence = f"The patient experienced {event_text}"

    if facts.get("event_date"):
        sentence += f" on {_format_date(facts['event_date'])}"

    return sentence + "."


def _clean_diagnosis(value: str) -> str:
    value = _strip_terminal_punctuation(value)

    if value.lower() in {
        "no specific diagnosis was reported",
        "no diagnosis was reported",
        "no specific diagnosis reported",
    }:
        return "No specific diagnosis was reported."

    return f"The reported diagnosis was {value}."


def _clean_investigations(value: str) -> str:
    """
    Turn common source-style investigation phrases into grammatical narrative
    sentences without changing their meaning.
    """
    value = _strip_terminal_punctuation(value)

    # Example source:
    # "blood pressure was 128/78 mmHg; laboratory results were reported as
    # within normal limits"
    clauses = [x.strip() for x in value.split(";") if x.strip()]

    if not clauses:
        return ""

    cleaned = []

    for clause in clauses:
        lower = clause.lower()

        if lower.startswith("blood pressure was "):
            remainder = clause[len("blood pressure was "):].strip()
            cleaned.append(f"Blood pressure was {remainder}.")
        elif lower.startswith("laboratory results were "):
            cleaned.append(clause[0].upper() + clause[1:] + ".")
        else:
            cleaned.append(clause[0].upper() + clause[1:] + ".")

    return " ".join(cleaned)


def _build_clinical_evaluation_paragraph(facts: Dict[str, str]) -> str:
    sentences = []

    if facts.get("diagnosis"):
        sentences.append(_clean_diagnosis(facts["diagnosis"]))

    if facts.get("investigations"):
        sentences.append(_clean_investigations(facts["investigations"]))

    return " ".join(sentences)


def _build_management_paragraph(facts: Dict[str, str]) -> str:
    sentences = []

    if facts.get("action_taken"):
        action = _strip_terminal_punctuation(facts["action_taken"])

        # If the source field already contains a complete statement, preserve
        # it instead of wrapping it in "was reported as".
        if re.search(
            r"\b(was|were|is|are|continued|discontinued|interrupted|stopped|"
            r"withdrawn|reduced|increased|not changed)\b",
            action,
            flags=re.I,
        ):
            sentences.append(action + ".")
        else:
            sentences.append(
                "The action taken with the suspect medicinal product was "
                f"reported as {action}."
            )

    if facts.get("treatment"):
        treatment = _strip_terminal_punctuation(facts["treatment"])

        # Source may already contain "were reported".
        treatment = re.sub(
            r"\s+were reported$",
            "",
            treatment,
            flags=re.I,
        )
        treatment = re.sub(
            r"\s+was reported$",
            "",
            treatment,
            flags=re.I,
        )

        sentences.append(
            f"Therapeutic or other management measures included "
            f"{_lower_first(treatment)}."
        )

    return " ".join(sentences)


def _build_status_paragraph(facts: Dict[str, str]) -> str:
    sentences = []

    if facts.get("seriousness"):
        seriousness = _lower_first(
            _strip_terminal_punctuation(facts["seriousness"])
        )
        sentences.append(f"The event was reported as {seriousness}.")

    if facts.get("outcome"):
        outcome = _strip_terminal_punctuation(facts["outcome"])

        outcome_map = {
            "recovered": "The patient recovered from the event.",
            "recovered/resolved": "The event resolved.",
            "resolved": "The event resolved.",
            "recovering/resolving": "The event was reported as recovering.",
            "not recovered": "The event was reported as not recovered.",
            "fatal": "The reported outcome was fatal.",
            "unknown": "The outcome was reported as unknown.",
        }

        sentences.append(
            outcome_map.get(
                outcome.lower(),
                f"The reported outcome was {outcome}.",
            )
        )

    return " ".join(sentences)


def _build_follow_up_paragraph(facts: Dict[str, str]) -> str:
    value = facts.get("follow_up", "")
    if not value:
        return ""

    value = value.strip()

    # Follow-up source is often already a complete narrative statement.
    if re.match(r"^at follow[- ]?up\b", value, flags=re.I):
        return _strip_terminal_punctuation(value) + "."

    return f"Follow-up information reported in the case was: {value}."


def _build_reporter_sentence(facts: Dict[str, str]) -> str:
    reporter = facts.get("reporter_type", "")
    if not reporter:
        return ""

    reporter = _lower_first(_strip_terminal_punctuation(reporter))
    article = "an" if reporter[:1].lower() in "aeiou" else "a"

    return f"The report was submitted by {article} {reporter}."


def build_narrative_plan(facts: Dict[str, str]) -> List[str]:
    """Build the complete factual narrative in clinical sequence."""

    paragraphs = [
        _build_patient_therapy_paragraph(facts),
        _build_background_paragraph(facts),
        _build_event_paragraph(facts),
        _build_clinical_evaluation_paragraph(facts),
        _build_management_paragraph(facts),
        _build_status_paragraph(facts),
        _build_follow_up_paragraph(facts),
        _build_reporter_sentence(facts),
    ]

    return [p.strip() for p in paragraphs if p and p.strip()]


# ---------------------------------------------------------------------------
# Controlled LLM editing
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """
You are the final language editor for a pharmacovigilance case narrative.

The supplied narrative is already factually constructed from structured ICSR
data. Improve only grammar, flow, punctuation, readability and professional
PV style.

STRICTLY PROHIBITED:
- adding facts;
- removing facts;
- changing dates, drug names, event terms, age, sex, country, seriousness or
  outcome;
- inferring causality;
- inferring diagnosis, severity, treatment response, dechallenge or rechallenge;
- changing reported uncertainty into certainty;
- repeating information;
- adding filler.

Preserve the chronological clinical sequence.

Return ONLY the edited narrative.
"""


def _clean_llm_output(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    text = str(text).strip()
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^\s*Narrative\s*:\s*", "", text, flags=re.I)

    return text.strip() or None


def _polish_with_groq(skeleton: str) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        model = os.getenv("NARRATIVE_LLM_MODEL", "openai/gpt-oss-20b")

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": skeleton},
            ],
        )

        return _clean_llm_output(response.choices[0].message.content)

    except Exception as exc:
        print(f"Groq unavailable; using deterministic narrative: {exc}")
        return None


def _polish_with_openrouter(skeleton: str) -> Optional[str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        model = os.getenv("NARRATIVE_LLM_MODEL", "openai/gpt-oss-20b")

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": skeleton},
            ],
        )

        return _clean_llm_output(response.choices[0].message.content)

    except Exception as exc:
        print(f"OpenRouter unavailable; using deterministic narrative: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _load_case(case_id: str) -> Dict[str, Any]:
    if not INCOMING_FILE.exists():
        raise FileNotFoundError(f"Incoming ICSR file not found: {INCOMING_FILE}")

    df = pd.read_csv(INCOMING_FILE)

    if "case_id" not in df.columns:
        raise ValueError("incoming_icsrs.csv must contain a 'case_id' column.")

    matches = df[
        df["case_id"].astype(str).str.strip() == str(case_id).strip()
    ]

    if matches.empty:
        raise ValueError(f"Case ID '{case_id}' was not found.")

    return matches.iloc[0].to_dict()


def generate_narrative(case_id: str) -> str:
    case = _load_case(case_id)
    facts = extract_case_facts(case)

    paragraphs = build_narrative_plan(facts)
    skeleton = "\n\n".join(paragraphs)

    # Optional language polishing.
    narrative = _polish_with_groq(skeleton)

    if narrative is None:
        narrative = _polish_with_openrouter(skeleton)

    if narrative is None:
        narrative = skeleton

    return narrative.strip()


def generate_case_narrative(case_id: str) -> str:
    """Compatibility alias used by narrative_qc.py."""
    return generate_narrative(case_id)


def generate_narratives_for_cases(
    case_ids: Iterable[str],
) -> Dict[str, str]:
    """Generate narratives only for explicitly supplied case IDs."""

    results: Dict[str, str] = {}

    for case_id in case_ids:
        case_id = str(case_id)
        try:
            results[case_id] = generate_narrative(case_id)
        except Exception as exc:
            results[case_id] = f"[Narrative generation failed: {exc}]"

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src\\narrative_generation.py <CASE_ID>")
        raise SystemExit(1)

    print(generate_narrative(sys.argv[1]))
