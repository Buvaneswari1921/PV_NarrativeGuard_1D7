
"""
PV_NarrativeGuard - RAB Bot

RAB = Retrieval-Augmented Backend

Purpose
-------
Orchestrate the controlled PV/MedDRA assistant workflow.

Architecture
------------
1. Query classification
2. Intent routing
3. MedDRA terminology route OR general PV/project RAG route
4. Retrieval/terminology safety decision
5. Grounded response construction
6. Knowledge-gap handling
7. Operational audit finalization

Safety principles
-----------------
- Unsupported terminology is never promoted to a supported answer.
- General RAG answers require sufficient retrieved evidence.
- Component failures return SYSTEM_ERROR responses with FAILED operational state.
- Internal exception details are never exposed to the user.
- Every request receives one terminal operational audit event.
- Audit persistence failure does not crash the workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from .query_classifier import classify_query

from .intent_router import (
    route_query,
    MEDDRA,
    GENERAL_PV,
    PROJECT,
    UNKNOWN,
)

from .retrieval_guard import evaluate_retrieval

from .knowledge_gap import (
    create_pending_request,
)

from .grounded_answer import (
    generate_grounded_rag_answer,
    preserve_meddra_decision,
)

from src.audit_log import (
    generate_rab_request_id,
    write_rab_audit_event,
)


# ============================================================
# RESPONSE MODES
# ============================================================

OUT_OF_CONTEXT = "OUT_OF_CONTEXT"

RAG_ANSWER = "RAG_ANSWER"

ASK_ADD_TO_DATASET = "ASK_ADD_TO_DATASET"

MEDDRA_ANSWER = "MEDDRA_ANSWER"

CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"

SYSTEM_ERROR = "SYSTEM_ERROR"

# Retained for internal/component classification compatibility.
COMPONENT_FAILURE = "COMPONENT_FAILURE"


# ============================================================
# OPERATIONAL STATES
# ============================================================

COMPLETED = "COMPLETED"

KNOWLEDGE_GAP = "KNOWLEDGE_GAP"

OUT_OF_CONTEXT_STATE = "OUT_OF_CONTEXT"

CLARIFICATION_STATE = "CLARIFICATION_REQUIRED"

FAILED = "FAILED"

# Internal component classification; terminal user-facing state is FAILED.
COMPONENT_FAILURE_STATE = FAILED


# ============================================================
# CONTROLLED USER MESSAGES
# ============================================================

RETRIEVAL_FAILURE_MESSAGE = (
    "The PV/MedDRA Assistant encountered a retrieval failure "
    "while obtaining the required knowledge for this request. "
    "No unsupported answer was generated."

)

MEDDRA_FAILURE_MESSAGE = (
    "The PV/MedDRA Assistant could not complete the "
    "MedDRA terminology evaluation for this request. "
    "No unsupported terminology decision was generated."
)

GENERAL_COMPONENT_FAILURE_MESSAGE = (
    "The PV/MedDRA Assistant could not complete this "
    "request because a required component was unavailable. "
    "No unsupported answer was generated."
)


# ============================================================
# RESPONSE MODEL
# ============================================================

@dataclass
class RABResponse:
    query: str
    response_mode: str
    message: str
    confidence: float
    evidence: list
    scope_reason: str
    retrieval_reason: str = ""
    route: str = ""
    operational_state: str = ""
    audit_status: str = ""
    request_id: str = ""


# ============================================================
# RAB BOT
# ============================================================

class RABBot:

    def __init__(
        self,
        retriever=None,
        meddra_adapter=None,
        top_k=3,
        min_similarity=0.65,
        strong_similarity=0.80,
        queue_path="outputs/pending_knowledge_requests.jsonl",
    ):

        if retriever is None:

            from src.rag_retriever import retrieve

            retriever = retrieve

        self.retriever = retriever

        self._meddra_adapter = meddra_adapter

        self.top_k = top_k

        self.min_similarity = min_similarity

        self.strong_similarity = strong_similarity

        self.queue_path = queue_path

    # ========================================================
    # MEDDRA ADAPTER
    # ========================================================

    def _get_meddra_adapter(self):

        if self._meddra_adapter is None:

            from .meddra_adapter import MedDRAAdapter

            self._meddra_adapter = MedDRAAdapter()

        return self._meddra_adapter

    # ========================================================
    # OPERATIONAL FINALIZATION
    # ========================================================

    def _finalize_response(
        self,
        response: RABResponse,
    ) -> RABResponse:
        """
        Write exactly one terminal RAB audit event.

        Audit failure never interrupts the response.
        """

        if not response.request_id:

            response.request_id = (
                generate_rab_request_id()
            )

        audit_written = write_rab_audit_event(
            request_id=response.request_id,
            query=response.query,
            event_type=self._audit_event_type(
                response
            ),
            route=response.route,
            response_mode=response.response_mode,
            operational_state=response.operational_state,
            confidence=response.confidence,
            evidence_count=len(
                response.evidence or []
            ),
            action=self._audit_action(
                response
            ),
            reason=(
                response.retrieval_reason
                or response.scope_reason
                or "RAB request completed."
            ),
        )

        response.audit_status = (
            "WRITTEN"
            if audit_written
            else "FAILED"
        )

        return response

    # ========================================================
    # AUDIT EVENT TYPE
    # ========================================================

    @staticmethod
    def _audit_event_type(
        response: RABResponse,
    ) -> str:

        if (
            response.operational_state
            == KNOWLEDGE_GAP
        ):

            if response.route == MEDDRA:

                return (
                    "RAB_MEDDRA_KNOWLEDGE_GAP"
                )

            return "RAB_RAG_KNOWLEDGE_GAP"

        if (
            response.operational_state
            == FAILED
        ):

            return "RAB_SYSTEM_ERROR"

        if (
            response.operational_state
            == OUT_OF_CONTEXT_STATE
        ):

            return "RAB_QUERY_COMPLETED"

        if (
            response.operational_state
            == CLARIFICATION_STATE
        ):

            return "RAB_QUERY_COMPLETED"

        if response.route == MEDDRA:

            return "RAB_MEDDRA_DECISION"

        return "RAB_RAG_DECISION"

    # ========================================================
    # AUDIT ACTION
    # ========================================================

    @staticmethod
    def _audit_action(
        response: RABResponse,
    ) -> str:

        if (
            response.operational_state
            == FAILED
        ):

            return "SYSTEM_ERROR"

        if response.response_mode == MEDDRA_ANSWER:

            return "RETURN_SUPPORTED_MEDDRA"

        if (
            response.response_mode
            == CLARIFICATION_REQUIRED
        ):

            return "REQUEST_CLARIFICATION"

        if (
            response.response_mode
            == ASK_ADD_TO_DATASET
        ):

            return "REQUEST_DATASET_REVIEW"

        if response.response_mode == RAG_ANSWER:

            return "RETURN_GROUNDED_ANSWER"

        if response.response_mode == OUT_OF_CONTEXT:

            return "RETURN_OUT_OF_CONTEXT"

        return "RETURN_CONTROLLED_RESPONSE"

    # ========================================================
    # FAILURE RESPONSE HELPERS
    # ========================================================

    def _retriever_failure_response(
        self,
        *,
        query: str,
        scope_reason: str,
        route: str,
        request_id: str,
    ) -> RABResponse:
        """
        Return a controlled response when general RAG
        retrieval fails.

        Internal exception details are intentionally not
        included in the response.
        """

        return RABResponse(
            query=query,
            response_mode=SYSTEM_ERROR,
            message=RETRIEVAL_FAILURE_MESSAGE,
            confidence=0.0,
            evidence=[],
            scope_reason=scope_reason,
            retrieval_reason=(
                "General knowledge retrieval "
                "component failed."
            ),
            route=route,
            operational_state=FAILED,
            request_id=request_id,
        )

    def _meddra_failure_response(
        self,
        *,
        query: str,
        scope_reason: str,
        request_id: str,
    ) -> RABResponse:
        """
        Return a controlled response when the MedDRA
        terminology component fails.
        """

        return RABResponse(
            query=query,
            response_mode=SYSTEM_ERROR,
            message=MEDDRA_FAILURE_MESSAGE,
            confidence=0.0,
            evidence=[],
            scope_reason=scope_reason,
            retrieval_reason=(
                "MedDRA terminology component failed."
            ),
            route=MEDDRA,
            operational_state=FAILED,
            request_id=request_id,
        )

    # ========================================================
    # MAIN QUERY ENTRY POINT
    # ========================================================

    def ask(
        self,
        query: str,
    ) -> RABResponse:

        # ----------------------------------------------------
        # Normalize input safely
        # ----------------------------------------------------

        try:

            query = str(
                query
            ).strip()

        except Exception:

            query = ""

        request_id = generate_rab_request_id()

        # ----------------------------------------------------
        # Invalid / empty query
        # ----------------------------------------------------

        if not query:

            response = RABResponse(
                query="",
                response_mode=CLARIFICATION_REQUIRED,
                message=(
                    "Please provide a question or "
                    "clinical query."
                ),
                confidence=0.0,
                evidence=[],
                scope_reason=(
                    "No usable query was provided."
                ),
                retrieval_reason=(
                    "Query validation failed."
                ),
                route=UNKNOWN,
                operational_state=(
                    CLARIFICATION_STATE
                ),
                request_id=request_id,
            )

            return self._finalize_response(
                response
            )

        # ====================================================
        # QUERY CLASSIFICATION
        # ====================================================

        try:

            scope = classify_query(
                query
            )

        except Exception:

            response = RABResponse(
                query=query,
                response_mode=SYSTEM_ERROR,
                message=GENERAL_COMPONENT_FAILURE_MESSAGE,
                confidence=0.0,
                evidence=[],
                scope_reason=(
                    "Query classification "
                    "component failed."
                ),
                retrieval_reason=(
                    "Query classifier unavailable."
                ),
                route="",
                operational_state=FAILED,
                request_id=request_id,
            )

            return self._finalize_response(
                response
            )

        # ----------------------------------------------------
        # OUT OF SCOPE
        # ----------------------------------------------------

        if scope.scope == "OUT_OF_SCOPE":

            response = RABResponse(
                query=query,
                response_mode=OUT_OF_CONTEXT,
                message=(
                    "This query is outside my current "
                    "PV NarrativeGuard context. "
                    "I can assist with pharmacovigilance, "
                    "ICSR processing, MedDRA terminology, "
                    "narrative generation/QC, regulatory "
                    "concepts, and the project's AI/RAG "
                    "architecture."
                ),
                confidence=scope.confidence,
                evidence=[],
                scope_reason=scope.reason,
                route="OUT_OF_CONTEXT",
                operational_state=(
                    OUT_OF_CONTEXT_STATE
                ),
                request_id=request_id,
            )

            return self._finalize_response(
                response
            )

        # ====================================================
        # INTENT ROUTING
        # ====================================================

        try:

            intent = route_query(
                query
            )

        except Exception:

            response = RABResponse(
                query=query,
                response_mode=SYSTEM_ERROR,
                message=GENERAL_COMPONENT_FAILURE_MESSAGE,
                confidence=0.0,
                evidence=[],
                scope_reason=scope.reason,
                retrieval_reason=(
                    "Intent routing component failed."
                ),
                route="",
                operational_state=FAILED,
                request_id=request_id,
            )

            return self._finalize_response(
                response
            )

        # ----------------------------------------------------
        # UNKNOWN INTENT
        # ----------------------------------------------------

        if intent.intent == UNKNOWN:

            response = RABResponse(
                query=query,
                response_mode=CLARIFICATION_REQUIRED,
                message=(
                    "I can identify this as related to "
                    "the PV NarrativeGuard domain, but "
                    "I cannot determine the correct "
                    "knowledge source with enough confidence. "
                    "Please clarify whether you mean MedDRA "
                    "terminology, general pharmacovigilance, "
                    "or PV NarrativeGuard/project architecture."
                ),
                confidence=intent.confidence,
                evidence=[],
                scope_reason=scope.reason,
                route=UNKNOWN,
                operational_state=(
                    CLARIFICATION_STATE
                ),
                request_id=request_id,
            )

            return self._finalize_response(
                response
            )

        # ====================================================
        # MEDDRA ROUTE
        # ====================================================

        if intent.intent == MEDDRA:

            # ------------------------------------------------
            # MedDRA adapter execution
            # ------------------------------------------------

            try:

                adapter = (
                    self
                    ._get_meddra_adapter()
                )

                decision = adapter.query(
                    query
                )

            except Exception:

                response = (
                    self
                    ._meddra_failure_response(
                        query=query,
                        scope_reason=scope.reason,
                        request_id=request_id,
                    )
                )

                return self._finalize_response(
                    response
                )

            # ------------------------------------------------
            # Defensive decision handling
            # ------------------------------------------------

            if not isinstance(
                decision,
                dict,
            ):

                response = (
                    self
                    ._meddra_failure_response(
                        query=query,
                        scope_reason=scope.reason,
                        request_id=request_id,
                    )
                )

                return self._finalize_response(
                    response
                )

            result = decision.get(
                "result"
            )

            status = decision.get(
                "status",
                "",
            )

            # ------------------------------------------------
            # NEGATED CLINICAL EXPRESSION
            # ------------------------------------------------

            if status == "NEGATED":
                response = RABResponse(
                    query=query,
                    response_mode=CLARIFICATION_REQUIRED,
                    message=(
                        "Negated clinical expression detected. No MedDRA "
                        "coding recommendation was generated because the "
                        "event is documented as denied/not present. Verify "
                        "the source case before final data entry."
                    ),
                    confidence=1.0,
                    evidence=[],
                    scope_reason=scope.reason,
                    retrieval_reason=decision.get("message", ""),
                    route=MEDDRA,
                    operational_state=COMPLETED,
                    request_id=request_id,
                )
                return self._finalize_response(response)

            # ------------------------------------------------
            # SUPPORTED
            # ------------------------------------------------

            if (
                status == "SUPPORTED"
                and result
            ):

                try:

                    grounded = (
                        preserve_meddra_decision(
                            decision
                        )
                    )

                except Exception:

                    response = (
                        self
                        ._meddra_failure_response(
                            query=query,
                            scope_reason=scope.reason,
                            request_id=request_id,
                        )
                    )

                    return self._finalize_response(
                        response
                    )

                response = RABResponse(
                    query=query,
                    response_mode=MEDDRA_ANSWER,
                    message=grounded.answer,
                    confidence=grounded.confidence,
                    evidence=grounded.evidence,
                    scope_reason=scope.reason,
                    retrieval_reason=decision.get(
                        "message",
                        "",
                    ),
                    route=MEDDRA,
                    operational_state=COMPLETED,
                    request_id=request_id,
                )

                return self._finalize_response(
                    response
                )

            # ------------------------------------------------
            # AMBIGUOUS
            # ------------------------------------------------

            if status == "AMBIGUOUS":

                result_data = (
                    result
                    if isinstance(
                        result,
                        dict,
                    )
                    else {}
                )

                try:

                    confidence = float(
                        result_data.get(
                            "decision_score",
                            0.0,
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    confidence = 0.0

                candidates = decision.get(
                    "candidates",
                    [],
                )

                if not isinstance(
                    candidates,
                    list,
                ):

                    candidates = []

                response = RABResponse(
                    query=query,
                    response_mode=(
                        CLARIFICATION_REQUIRED
                    ),
                    message=decision.get(
                        "message",
                        (
                            "Multiple MedDRA concepts "
                            "may fit. Please provide "
                            "more clinical context."
                        ),
                    ),
                    confidence=confidence,
                    evidence=candidates[:5],
                    scope_reason=scope.reason,
                    retrieval_reason=decision.get(
                        "message",
                        "",
                    ),
                    route=MEDDRA,
                    operational_state=(
                        CLARIFICATION_STATE
                    ),
                    request_id=request_id,
                )

                return self._finalize_response(
                    response
                )

            # ------------------------------------------------
            # UNSUPPORTED / KNOWLEDGE GAP
            # ------------------------------------------------

            result_data = (
                result
                if isinstance(
                    result,
                    dict,
                )
                else {}
            )

            try:

                confidence = float(
                    result_data.get(
                        "decision_score",
                        0.0,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                confidence = 0.0

            candidates = decision.get(
                "candidates",
                [],
            )

            if not isinstance(
                candidates,
                list,
            ):

                candidates = []

            response = RABResponse(
                query=query,
                response_mode=ASK_ADD_TO_DATASET,
                message=(
                    "This appears to be a "
                    "MedDRA-related query, but "
                    "the current MedDRA terminology "
                    "dataset does not provide sufficient "
                    "support for a safe terminology decision. "
                    "Would you like to add this topic "
                    "to the dataset?"
                ),
                confidence=confidence,
                evidence=candidates[:5],
                scope_reason=scope.reason,
                retrieval_reason=(
                    "Terminology dataset does not "
                    "provide sufficient support for "
                    "a safe MedDRA decision."
                ),
                route=MEDDRA,
                operational_state=KNOWLEDGE_GAP,
                request_id=request_id,
            )

            return self._finalize_response(
                response
            )

        # ====================================================
        # GENERAL PV / PROJECT RAG
        # ====================================================

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        try:

            retrieved = self.retriever(
                query,
                top_k=self.top_k,
            )

        except Exception:

            response = (
                self
                ._retriever_failure_response(
                    query=query,
                    scope_reason=scope.reason,
                    route=intent.intent,
                    request_id=request_id,
                )
            )

            return self._finalize_response(
                response
            )

        # ----------------------------------------------------
        # Retrieval guard
        # ----------------------------------------------------

        try:

            decision = evaluate_retrieval(
                retrieved,
                min_similarity=self.min_similarity,
                strong_similarity=self.strong_similarity,
            )

        except Exception:

            response = RABResponse(
                query=query,
                response_mode=SYSTEM_ERROR,
                message=GENERAL_COMPONENT_FAILURE_MESSAGE,
                confidence=0.0,
                evidence=[],
                scope_reason=scope.reason,
                retrieval_reason=(
                    "Retrieval evaluation "
                    "component failed."
                ),
                route=intent.intent,
                operational_state=FAILED,
                request_id=request_id,
            )

            return self._finalize_response(
                response
            )

        # ----------------------------------------------------
        # SUFFICIENT RETRIEVAL
        # ----------------------------------------------------

        if decision.status == "SUFFICIENT":

            try:

                grounded = (
                    generate_grounded_rag_answer(
                        query=query,
                        evidence=decision.evidence,
                        confidence=decision.confidence,
                    )
                )

            except Exception:

                response = RABResponse(
                    query=query,
                    response_mode=SYSTEM_ERROR,
                    message=GENERAL_COMPONENT_FAILURE_MESSAGE,
                    confidence=0.0,
                    evidence=[],
                    scope_reason=scope.reason,
                    retrieval_reason=(
                        "Grounded answer component "
                        "failed."
                    ),
                    route=intent.intent,
                    operational_state=(
                        COMPONENT_FAILURE_STATE
                    ),
                    request_id=request_id,
                )

                return self._finalize_response(
                    response
                )

            # ------------------------------------------------
            # Grounded answer accepted
            # ------------------------------------------------

            if grounded.grounded:

                response = RABResponse(
                    query=query,
                    response_mode=RAG_ANSWER,
                    message=grounded.answer,
                    confidence=grounded.confidence,
                    evidence=grounded.evidence,
                    scope_reason=scope.reason,
                    retrieval_reason=decision.reason,
                    route=intent.intent,
                    operational_state=COMPLETED,
                    request_id=request_id,
                )

                return self._finalize_response(
                    response
                )

            # ------------------------------------------------
            # Evidence exists but answer cannot be grounded
            # ------------------------------------------------

            response = RABResponse(
                query=query,
                response_mode=ASK_ADD_TO_DATASET,
                message=(
                    "The retrieval system identified "
                    "potentially relevant information, "
                    "but it could not be converted into "
                    "a sufficiently grounded answer."
                ),
                confidence=grounded.confidence,
                evidence=grounded.evidence,
                scope_reason=scope.reason,
                retrieval_reason=grounded.reason,
                route=intent.intent,
                operational_state=KNOWLEDGE_GAP,
                request_id=request_id,
            )

            return self._finalize_response(
                response
            )

        # ====================================================
        # GENERAL RAG KNOWLEDGE GAP
        # ====================================================

        response = RABResponse(
            query=query,
            response_mode=ASK_ADD_TO_DATASET,
            message=(
                "This topic is relevant to "
                "PV NarrativeGuard, but I do not "
                "currently have sufficient evidence "
                "for it in the knowledge dataset. "
                "Would you like to add this topic "
                "to the dataset?"
            ),
            confidence=decision.confidence,
            evidence=decision.evidence,
            scope_reason=scope.reason,
            retrieval_reason=decision.reason,
            route=intent.intent,
            operational_state=KNOWLEDGE_GAP,
            request_id=request_id,
        )

        return self._finalize_response(
            response
        )

    # ========================================================
    # KNOWLEDGE-GAP REQUEST
    # ========================================================

    def approve_add_to_dataset(
        self,
        query: str,
    ):

        return create_pending_request(
            query,
            self.queue_path,
        )
