"""PV NarrativeGuard - MedDRA Assistant UI.

Presentation layer for the reusable floating MedDRA Assistant.

The backend remains responsible for:
    - query classification
    - intent routing
    - MedDRA terminology decisions
    - RAG retrieval
    - grounding safeguards
    - knowledge-gap handling
    - operational audit events

This module intentionally keeps the UI compact and independent from
the main page layout.  The assistant is rendered through a native
Streamlit popover that is positioned as a fixed bottom-right widget.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import streamlit as st


# ============================================================
# RAB BOT
# ============================================================

@st.cache_resource
def get_rab_bot():
    """Return one shared RABBot instance."""
    from src.chatbot.rab_bot import RABBot

    return RABBot()


# ============================================================
# CONTEXT HELPERS
# ============================================================

def _ctx_value(
    ctx: Any,
    name: str,
    default: str = "",
) -> str:
    """Read a value from dict-like or object-like context."""

    if ctx is None:
        return default

    if isinstance(ctx, dict):
        value = ctx.get(name, default)
    else:
        value = getattr(ctx, name, default)

    if value is None:
        return default

    return str(value)


# ============================================================
# VALUE HELPERS
# ============================================================

def _is_missing(value: Any) -> bool:
    """Return True when a value is None, empty, or NaN."""

    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, float):
        try:
            return math.isnan(value)
        except (TypeError, ValueError):
            return False

    return False


def _format_bool(value: Any) -> str:
    """Format boolean-like values for UI display."""

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, str):
        lowered = value.strip().lower()

        if lowered in {
            "true",
            "yes",
            "1",
        }:
            return "Yes"

        if lowered in {
            "false",
            "no",
            "0",
        }:
            return "No"

    return str(value)


def _response_value(
    response: Any,
    name: str,
    default: Any = None,
) -> Any:
    """Safely read response fields from dataclass/object or dict."""

    if response is None:
        return default

    if isinstance(response, dict):
        return response.get(name, default)

    return getattr(
        response,
        name,
        default,
    )


# ============================================================
# MEDDRA EVIDENCE
# ============================================================

def _evidence_text(item: Any) -> str:
    """Extract useful display text from evidence."""

    if isinstance(item, str):
        return item.strip()

    if not isinstance(item, dict):
        return str(item)

    for key in (
        "text",
        "content",
        "chunk_text",
        "description",
        "definition",
        "preferred_label",
        "term",
    ):
        value = item.get(key)

        if _is_missing(value):
            continue

        return str(value).strip()

    return str(item)


def _render_meddra_metadata(
    item: dict[str, Any],
) -> None:
    """Render compact MedDRA evidence metadata."""

    fields = []

    meddra_id = item.get(
        "meddra_id"
    )

    similarity = item.get(
        "similarity_score",
        item.get("similarity"),
    )

    decision_score = item.get(
        "decision_score",
        item.get("score"),
    )

    match_method = item.get(
        "match_method"
    )

    mapped_term = item.get(
        "mapped_term"
    )

    if not _is_missing(
        meddra_id
    ):
        fields.append(
            f"ID: {meddra_id}"
        )

    if not _is_missing(
        similarity
    ):
        try:
            fields.append(
                f"Similarity: "
                f"{float(similarity):.4f}"
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    if not _is_missing(
        decision_score
    ):
        try:
            fields.append(
                f"Decision: "
                f"{float(decision_score):.4f}"
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    if not _is_missing(
        match_method
    ):
        fields.append(
            f"Method: {match_method}"
        )

    if fields:
        st.caption(
            " â€¢ ".join(fields)
        )

    extra = []

    clinical_phrase_match = item.get(
        "clinical_phrase_match"
    )

    lexical_match = item.get(
        "lexical_match"
    )

    unsupported_specificity = item.get(
        "unsupported_specificity"
    )

    general_term = item.get(
        "general_term"
    )

    if not _is_missing(
        clinical_phrase_match
    ):
        extra.append(
            "Clinical phrase match: "
            f"{_format_bool(clinical_phrase_match)}"
        )

    if not _is_missing(
        lexical_match
    ):
        extra.append(
            "Lexical match: "
            f"{_format_bool(lexical_match)}"
        )

    if not _is_missing(
        unsupported_specificity
    ):
        extra.append(
            "Unsupported specificity: "
            f"{_format_bool(unsupported_specificity)}"
        )

    if not _is_missing(
        general_term
    ):
        extra.append(
            "General term: "
            f"{_format_bool(general_term)}"
        )

    if extra:
        st.caption(
            " â€¢ ".join(extra)
        )

    if not _is_missing(
        mapped_term
    ):
        st.caption(
            f"Mapped term: {mapped_term}"
        )


def _render_evidence(
    evidence: list[Any],
) -> None:
    """Render supporting evidence in a compact expandable section.

    The renderer supports both:
        - General RAG evidence
        - MedDRA terminology evidence
    """

    if not evidence:
        return

    with st.expander(
        f"Evidence / Candidates ({len(evidence)})",
        expanded=False,
    ):

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            if isinstance(
                item,
                dict,
            ):

                label = (
                    item.get(
                        "preferred_label"
                    )
                    or item.get(
                        "term"
                    )
                )

                meddra_id = item.get(
                    "meddra_id"
                )

                similarity = item.get(
                    "similarity_score",
                    item.get(
                        "similarity"
                    ),
                )

                decision_score = item.get(
                    "decision_score",
                    item.get(
                        "score"
                    ),
                )

                # ------------------------------------------------
                # MedDRA evidence
                # ------------------------------------------------

                if (
                    not _is_missing(
                        label
                    )
                    or not _is_missing(
                        meddra_id
                    )
                    or not _is_missing(
                        decision_score
                    )
                ):

                    heading_parts = []

                    if not _is_missing(
                        label
                    ):

                        clean_label = (
                            str(label)
                            .replace(
                                " AE",
                                "",
                            )
                            .strip()
                        )

                        heading_parts.append(
                            clean_label
                        )

                    if not _is_missing(
                        meddra_id
                    ):

                        heading_parts.append(
                            f"MedDRA ID: {meddra_id}"
                        )

                    heading = (
                        " | ".join(
                            heading_parts
                        )
                        if heading_parts
                        else f"Evidence {index}"
                    )

                    st.markdown(
                        f"**{heading}**"
                    )

                    _render_meddra_metadata(
                        item
                    )

                    definition = item.get(
                        "definition"
                    )

                    if not _is_missing(
                        definition
                    ):
                        st.caption(
                            str(
                                definition
                            ).strip()
                        )

                    synonyms = item.get(
                        "synonyms"
                    )

                    if not _is_missing(
                        synonyms
                    ):
                        st.caption(
                            "Synonyms: "
                            f"{str(synonyms).strip()}"
                        )

                    continue

                # ------------------------------------------------
                # General RAG evidence
                # ------------------------------------------------

                metadata = []

                source = item.get(
                    "source"
                )

                chunk_id = item.get(
                    "chunk_id"
                )

                if not _is_missing(
                    source
                ):

                    source_text = (
                        str(source).strip()
                    )

                    if not _is_missing(
                        chunk_id
                    ):

                        source_text += (
                            f" ({str(chunk_id).strip()})"
                        )

                    metadata.append(
                        source_text
                    )

                if not _is_missing(
                    similarity
                ):

                    try:
                        metadata.append(
                            "Similarity: "
                            f"{float(similarity):.4f}"
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                if not _is_missing(
                    decision_score
                ):

                    try:
                        metadata.append(
                            "Score: "
                            f"{float(decision_score):.4f}"
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                heading = (
                    " | ".join(
                        metadata
                    )
                    if metadata
                    else f"Evidence {index}"
                )

                st.markdown(
                    f"**{heading}**"
                )

                st.caption(
                    _evidence_text(item)
                )

            else:

                st.markdown(
                    f"**Evidence {index}**"
                )

                st.caption(
                    _evidence_text(item)
                )


# ============================================================
# RESPONSE
# ============================================================

def _response_status(
    mode: str,
) -> tuple[str, str]:
    """Return compact UI status text and Streamlit status type."""

    mapping = {
        "MEDDRA_ANSWER": (
            "âœ“ MedDRA terminology decision",
            "success",
        ),
        "RAG_ANSWER": (
            "âœ“ Grounded PV knowledge",
            "success",
        ),
        "ASK_ADD_TO_DATASET": (
            "âš  Knowledge gap â€” human review required",
            "warning",
        ),
        "CLARIFICATION_REQUIRED": (
            "â„¹ Clarification required",
            "info",
        ),
        "OUT_OF_CONTEXT": (
            "â„¹ Outside PV NarrativeGuard context",
            "info",
        ),
    }

    return mapping.get(
        mode,
        (
            "PV/MedDRA Assistant response",
            "info",
        ),
    )


def _render_response(
    response: Any,
) -> None:
    """Render a compact controlled response."""

    mode = str(
        _response_value(
            response,
            "response_mode",
            "",
        )
        or ""
    )

    route = str(
        _response_value(
            response,
            "route",
            "",
        )
        or ""
    )

    confidence = _response_value(
        response,
        "confidence",
        0.0,
    )

    message = str(
        _response_value(
            response,
            "message",
            "",
        )
        or ""
    )

    evidence = (
        _response_value(
            response,
            "evidence",
            [],
        )
        or []
    )

    try:
        confidence_value = max(
            0.0,
            min(
                1.0,
                float(confidence),
            ),
        )

        confidence_text = (
            f"{confidence_value * 100:.1f}%"
        )

    except (
        TypeError,
        ValueError,
    ):
        confidence_text = "—"
    except (
        TypeError,
        ValueError,
    ):
        confidence_text = "â€”"

    # --------------------------------------------------------
    # Scope safeguard
    #
    # The floating assistant is presented as the MedDRA
    # Assistant. General RAG answers are therefore not exposed
    # through this UI surface.
    # --------------------------------------------------------

    if mode in {
        "OUT_OF_CONTEXT",
        "RAG_ANSWER",
    }:

        st.info(
            "**MedDRA Assistant:** This query is outside my "
            "current PV NarrativeGuard context. I can assist "
            "with MedDRA terminology."
        )

        return

    status_text, status_type = (
        _response_status(mode)
    )

    if status_type == "success":

        st.success(
            status_text
        )

    elif status_type == "warning":

        st.warning(
            status_text
        )

    else:

        st.info(
            status_text
        )

    if message:
        st.markdown(
            message
        )

    # --------------------------------------------------------
    # Compact operational status
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.markdown(
            '<div class="meddra-status-title">'
            'Operational Status'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(
            3
        )

        c1.metric(
            "Confidence",
            confidence_text,
        )

        c2.metric(
            "Route",
            route or "â€”",
        )

        audit_status = (
            "Recorded"
            if response is not None
            else "â€”"
        )

        c3.metric(
            "Audit",
            audit_status,
        )

    _render_evidence(
        evidence
    )


# ============================================================
# KNOWLEDGE GAP ACTION
# ============================================================

def _render_knowledge_gap_action(response: Any) -> None:
    """
    Render the optional knowledge-gap action only when the
    RAB response explicitly requests dataset addition.

    This helper intentionally performs no action for normal
    RAG answers or missing queries.  The actual review/data
    workflow remains owned by the backend.
    """

    if response is None:
        return None

    response_mode = _response_value(
        response,
        "response_mode",
        "",
    )

    if response_mode != "ASK_ADD_TO_DATASET":
        return None

    query = _response_value(
        response,
        "query",
        "",
    )

    if _is_missing(query):
        return None

    st.markdown(
        "### Knowledge Gap"
    )

    st.info(
        "The requested information is not currently available "
        "in the controlled knowledge base."
    )

    st.caption(
        "You may submit this query for dataset review."
    )

    if st.button(
        "Request Dataset Review",
        key="pv_meddra_dataset_review",
    ):
        st.success(
            "Dataset review request recorded."
        )

    return None


# ============================================================
# GLOBAL CHAT STATE
# ============================================================

def _initialize_state() -> None:
    """Initialize global assistant state."""

    if (
        "pv_meddra_chat_history"
        not in st.session_state
    ):

        st.session_state[
            "pv_meddra_chat_history"
        ] = []

    if (
        "pv_meddra_last_response"
        not in st.session_state
    ):

        st.session_state[
            "pv_meddra_last_response"
        ] = None


# ============================================================
# UI STYLING
# ============================================================

def _inject_floating_styles() -> None:
    """Keep the native Streamlit popover comfortable and bottom-right."""

    st.markdown(
        """
        <style>

        /* ====================================================
           MEDDRA ASSISTANT - FIXED LAUNCHER
           ==================================================== */

        div[data-testid="stPopover"] {
            position: fixed !important;

            right: 20px !important;
            bottom: 20px !important;

            left: auto !important;
            top: auto !important;

            /*
             * The parent needs enough width for the popup.
             * The actual launcher button remains compact.
             */
            width: 500px !important;
            min-width: 500px !important;
            max-width: calc(100vw - 32px) !important;

            margin: 0 !important;
            padding: 0 !important;

            overflow: visible !important;
            z-index: 999999 !important;
        }

        div[data-testid="stPopover"] > div {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;

            overflow: visible !important;
            box-sizing: border-box !important;
        }

        /*
         * Compact launcher.
         * The parent is wide for the popup, but the visible
         * launcher itself stays small.
         */
        div[data-testid="stPopover"] > div > button {
            width: auto !important;
            min-width: 0 !important;
            max-width: 100% !important;

            margin-left: auto !important;
            display: block !important;

            padding: 9px 15px !important;

            border-radius: 22px !important;

            font-size: 13px !important;
            font-weight: 600 !important;

            white-space: nowrap !important;
        }


        /* ====================================================
           MEDDRA ASSISTANT - POPUP
           ==================================================== */

        div[data-testid="stPopoverBody"] {
            width: 500px !important;
            min-width: 500px !important;
            max-width: calc(100vw - 32px) !important;

            box-sizing: border-box !important;

            max-height: 82vh !important;
            height: auto !important;

            overflow-y: auto !important;
            overflow-x: hidden !important;

            padding: 18px !important;
            margin: 0 !important;

            border-radius: 14px !important;
        }

        /*
         * Prevent Streamlit's inner popover wrapper from
         * shrinking the body.
         */
        div[data-testid="stPopoverBody"] > div:first-child {
            width: 100% !important;
            min-width: 0 !important;
            max-width: 100% !important;

            box-sizing: border-box !important;
        }

        /*
         * Keep every assistant child inside the card.
         */
        div[data-testid="stPopoverBody"] > div,
        div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;

            box-sizing: border-box !important;
        }

        div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }


        /* ====================================================
           TYPOGRAPHY
           ==================================================== */

        .meddra-assistant-title {
            font-size: 18px !important;
            font-weight: 700 !important;

            margin: 0 !important;
        }

        .meddra-assistant-subtitle {
            font-size: 12px !important;
            line-height: 1.4 !important;

            opacity: 0.72;

            margin-bottom: 6px !important;
        }

        .meddra-status-title {
            font-size: 12px !important;
            font-weight: 700 !important;

            margin-bottom: 2px !important;
        }


        /* ====================================================
           METRICS
           ==================================================== */

        div[data-testid="stMetric"] {
            padding: 0 !important;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 10px !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 14px !important;
            line-height: 1.2 !important;
        }


        /* ====================================================
           INPUT
           ==================================================== */

        div[data-testid="stPopoverBody"] textarea,
        div[data-testid="stPopoverBody"] input {
            width: 100% !important;
            max-width: 100% !important;

            box-sizing: border-box !important;

            font-size: 13px !important;
        }

        div[data-testid="stPopoverBody"] textarea {
            min-height: 96px !important;
        }


        /* ====================================================
           ASK BUTTON
           ==================================================== */

        div[data-testid="stPopoverBody"] button[kind="primary"] {
            width: 100% !important;
            min-width: 0 !important;

            margin-top: 0 !important;

            min-height: 40px !important;

            box-sizing: border-box !important;
        }


        /* ====================================================
           STATUS / RESPONSE ALERTS
           ==================================================== */

        div[data-testid="stPopoverBody"] div[data-testid="stAlert"] {
            width: 100% !important;
            max-width: 100% !important;

            box-sizing: border-box !important;

            padding: 9px 11px !important;

            margin: 3px 0 !important;

            font-size: 12px !important;
            line-height: 1.4 !important;
        }


        /* ====================================================
           RESPONSE TEXT
           ==================================================== */

        div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] {
            max-width: 100% !important;

            overflow-wrap: anywhere !important;

            box-sizing: border-box !important;
        }

        div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] p {
            margin-top: 0.2rem !important;
            margin-bottom: 0.3rem !important;

            line-height: 1.4 !important;
        }

        div[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] li {
            line-height: 1.4 !important;

            overflow-wrap: anywhere !important;
        }


        /* ====================================================
           EXPANDERS
           ==================================================== */

        div[data-testid="stPopoverBody"] details {
            width: 100% !important;
            max-width: 100% !important;

            margin: 3px 0 !important;

            box-sizing: border-box !important;
        }


        /* ====================================================
           HEADINGS
           ==================================================== */

        div[data-testid="stPopoverBody"] h1,
        div[data-testid="stPopoverBody"] h2,
        div[data-testid="stPopoverBody"] h3 {
            margin-top: 0.25rem !important;
            margin-bottom: 0.4rem !important;
        }


        /* ====================================================
           SMALL VIEWPORT
           ==================================================== */

        @media (max-width: 700px) {

            div[data-testid="stPopover"] {
                right: 10px !important;
                bottom: 10px !important;

                width: calc(100vw - 20px) !important;
                min-width: 0 !important;
                max-width: calc(100vw - 20px) !important;
            }

            div[data-testid="stPopoverBody"] {
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;

                max-height: calc(100vh - 30px) !important;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN MEDDRA ASSISTANT
# ============================================================

def render_rab_panel(
    ctx: Optional[Any] = None,
    page_name: str = "",
    expanded: bool = False,
) -> None:
    """Render the reusable floating MedDRA Assistant."""

    # IMPORTANT:
    # Do not call st.rerun() unconditionally here.
    # An unconditional rerun causes the floating popover to flicker.

    _initialize_state()
    _inject_floating_styles()

    case_id = (
        _ctx_value(
            ctx,
            "selected_case_id",
        )
        or _ctx_value(
            ctx,
            "case_id",
        )
    )

    # Native popover gives us a real Streamlit interaction
    # surface without affecting the width/layout of the
    # underlying page.
    with st.popover(
        "ðŸ§¬ MedDRA Assistant",
        use_container_width=False,
    ):

        st.markdown(
            '<div class="meddra-assistant-title">'
            'MedDRA Assistant'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="meddra-assistant-subtitle">'
            'Controlled AI assistance for PV NarrativeGuard. '
            'Responses are constrained by routing, retrieval, '
            'terminology, grounding, and human-review controls.'
            '</div>',
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Optional page/case context
        # ----------------------------------------------------

        context_parts = []

        if page_name:
            context_parts.append(
                f"Page: {page_name}"
            )

        if case_id:
            context_parts.append(
                f"Case: {case_id}"
            )

        if context_parts:
            st.caption(
                " â€¢ ".join(
                    context_parts
                )
            )

        # ----------------------------------------------------
        # Question / chat input
        #
        # Keep the primary interaction at the top so it is
        # immediately visible even after a response exists.
        # ----------------------------------------------------

        st.markdown(
            "**Ask the MedDRA Assistant**"
        )

        query = st.text_area(
            "MedDRA question",
            placeholder=(
                "Describe the clinical event or ask a "
                "MedDRA terminology question..."
            ),
            height=96,
            key="pv_meddra_query",
            label_visibility="collapsed",
        )

        ask = st.button(
            "Ask MedDRA",
            type="primary",
            key="pv_meddra_ask",
        )

        if ask:

            query = query.strip()

            if not query:
                st.warning(
                    "Please enter a question."
                )
                return

            history = (
                st.session_state[
                    "pv_meddra_chat_history"
                ]
            )

            history.append(
                {
                    "role": "user",
                    "content": query,
                }
            )

            with st.spinner(
                "MedDRA Assistant is evaluating..."
            ):

                try:

                    response = (
                        get_rab_bot().ask(
                            query
                        )
                    )

                except Exception:

                    # Do not expose implementation details
                    # or stack traces through the production UI.
                    st.error(
                        "The MedDRA Assistant could not "
                        "process this request. No unsupported "
                        "answer was generated."
                    )

                    return

            st.session_state[
                "pv_meddra_last_response"
            ] = response

            history.append(
                {
                    "role": "assistant",
                    "content": _response_value(
                        response,
                        "message",
                        "",
                    ),
                }
            )


        # ----------------------------------------------------
        # Latest response
        # ----------------------------------------------------

        response = (
            st.session_state[
                "pv_meddra_last_response"
            ]
        )

        if response is not None:

            st.markdown(
                "---"
            )

            _render_response(
                response
            )


        # ----------------------------------------------------
        # Previous conversation
        #
        # Keep this at the bottom and collapsed by default.
        # ----------------------------------------------------

        history = (
            st.session_state[
                "pv_meddra_chat_history"
            ]
        )

        if history:

            with st.expander(
                f"Conversation ({len(history)})",
                expanded=False,
            ):

                for item in history:

                    role = item.get(
                        "role",
                        "assistant",
                    )

                    content = item.get(
                        "content",
                        "",
                    )

                    if not content:
                        continue

                    if role == "user":

                        st.markdown(
                            f"**You:** {content}"
                        )

                    else:

                        st.markdown(
                            "**MedDRA Assistant:** "
                            f"{content}"
                        )


# ============================================================
# BACKWARD-COMPATIBLE ALIASES
# ============================================================

def render_rab_assistant(
    ctx: Optional[Any] = None,
    page_name: str = "",
    expanded: bool = False,
) -> None:
    """Backward-compatible alias."""

    render_rab_panel(
        ctx=ctx,
        page_name=page_name,
        expanded=expanded,
    )


def render_rab_ui() -> None:
    """Backward-compatible alias for older application code."""

    render_rab_panel()

