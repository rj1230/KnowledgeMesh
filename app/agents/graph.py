from __future__ import annotations

from typing import Any, Dict, List

import logfire
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.nodes.citation_check import citation_check_node
from app.agents.nodes.context_evaluator_node import context_evaluator_node
from app.agents.nodes.grader import grade_documents_node
from app.agents.nodes.grounding_critic import grounding_critic_node
from app.agents.nodes.planner import planner_node
from app.agents.nodes.query_rewriter import rewrite_query_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.private_retry import retry_private_retrieval_node
from app.agents.nodes.web_search_node import web_search_node
from app.agents.state import AgentState


# =====================================================================
# GRAPH LIMITS
# =====================================================================

MAX_RETRIEVAL_REWRITES = 2
MAX_ANSWER_REVISIONS = 2


# =====================================================================
# PLANNER ROUTING
# =====================================================================


def route_after_planner(
    state: AgentState,
) -> str:
    """
    Route planner output.

    Conversational requests bypass retrieval.

    Technical / knowledge requests enter the retrieval pipeline.
    """

    plan = state.get("plan") or []

    intent = (
        str(state.get("intent") or state.get("query_type") or state.get("route") or "")
        .strip()
        .upper()
    )

    plan_text = " ".join(str(item) for item in plan).upper()

    if intent == "CONVERSATIONAL" or "CONVERSATIONAL" in plan_text:
        return "responder"

    return "retriever"


# =====================================================================
# CONTEXT ROUTING
# =====================================================================


def route_after_context_evaluator(
    state: AgentState,
) -> str:
    """
    Decide whether retrieved private evidence is sufficient.

    Strong private context:
        -> responder

    Explicitly insufficient / external-needed context:
        -> web search

    Otherwise:
        -> bounded retrieval rewrite
    """

    # ================================================================
    # GRADER FAILURE / UNVERIFIED PRIVATE EVIDENCE
    # ================================================================
    #
    # A grader failure must NEVER be interpreted as:
    #
    #     strong private evidence
    #
    # even when the retrieved/reranked documents have high vector
    # or rerank scores.
    #
    # Retrieval confidence and evidence validation are separate
    # contracts.
    # ================================================================

    grader_status = str(state.get("grader_status") or "").strip().lower()

    grading_mode = str(state.get("grading_mode") or "").strip().lower()

    if grader_status == "failed" or grading_mode == "unavailable":
        logfire.warning(
            "Private evidence is unverified because document grading "
            "failed; routing to controlled web fallback.",
            grader_status=grader_status,
            grading_mode=grading_mode,
            grader_error_type=state.get("grader_error_type"),
            document_count=len(state.get("documents") or []),
        )

        return "web_search"

    # ================================================================
    # NORMAL CONTEXT ROUTING
    # ================================================================

    context_quality = str(state.get("context_quality") or "").strip().lower()

    web_search_required = bool(
        state.get(
            "web_search_required",
            False,
        )
    )

    rewrite_count = int(
        state.get(
            "retrieval_rewrite_count",
            0,
        )
        or 0
    )

    if context_quality in {
        "strong",
        "good",
        "sufficient",
        "excellent",
    }:
        return "responder"

    if web_search_required or context_quality in {
        "empty",
        "needs_web",
        "web",
        "insufficient_external",
    }:
        return "web_search"

    if rewrite_count >= MAX_RETRIEVAL_REWRITES:
        logfire.warning(
            "Retrieval rewrite budget exhausted; routing to web search.",
            rewrite_count=rewrite_count,
            max_rewrites=MAX_RETRIEVAL_REWRITES,
        )

        return "web_search"

    return "query_rewriter"


# =====================================================================
# RESPONDER ROUTING
# =====================================================================


def route_after_responder(
    state: AgentState,
) -> str:
    """
    Every successful generated answer must first pass citation
    validation.
    """

    if bool(
        state.get(
            "generation_failed",
            False,
        )
    ):
        logfire.warning(
            "Generation failed; ending graph.",
            status=state.get("status"),
        )

        return "end"

    status = str(state.get("status") or "").strip().lower()

    generation_failure_markers = (
        "generation unavailable",
        "generation degraded",
        "generation failed",
        "llm generation failed",
        "llm unavailable",
    )

    if any(marker in status for marker in generation_failure_markers):
        logfire.warning(
            "Generation degraded or unavailable; ending graph.",
            status=state.get("status"),
        )

        return "end"

    return "citation_check"


# =====================================================================
# CITATION CHECK ROUTING
# =====================================================================


def route_after_citation_check(
    state: AgentState,
) -> str:
    """
    Route after citation validation.

    Citation failure:
        -> bounded revision

    Citation success:
        -> grounding critic
    """

    if bool(
        state.get(
            "generation_failed",
            False,
        )
    ):
        return "end"

    if bool(
        state.get(
            "simple_response",
            False,
        )
    ):
        return "end"

    citation_valid = state.get("citation_valid")

    # ================================================================
    # CITATION FAILURE
    # ================================================================

    if citation_valid is not True:
        revision_count = int(
            state.get(
                "revision_count",
                0,
            )
            or 0
        )

        if revision_count >= MAX_ANSWER_REVISIONS:
            logfire.warning(
                "Citation revision budget exhausted.",
                revision_count=revision_count,
                max_revisions=MAX_ANSWER_REVISIONS,
                citation_valid=citation_valid,
            )

            return "end"

        logfire.info(
            "Citation validation failed; routing to bounded answer revision.",
            revision_count=revision_count,
            max_revisions=MAX_ANSWER_REVISIONS,
        )

        return "revision"

    logfire.info("Citation validation passed; routing to grounding critic.")

    return "grounding_critic"


# =====================================================================
# GROUNDING ROUTING
# =====================================================================


def route_after_grounding_critic(
    state: AgentState,
) -> str:
    """
    Route after evidence grounding.

    Fully supported answer:
        -> END

    Unsupported / partially supported answer:
        -> bounded revision
    """

    if bool(
        state.get(
            "simple_response",
            False,
        )
    ):
        return "end"

    is_grounded = bool(
        state.get(
            "is_grounded",
            False,
        )
    )

    answer_supported = bool(
        state.get(
            "answer_supported",
            False,
        )
    )

    if is_grounded and answer_supported:
        logfire.info(
            "Grounding validation passed; finalizing answer.",
            support_score=state.get("support_score"),
        )

        return "end"

    revision_count = int(
        state.get(
            "revision_count",
            0,
        )
        or 0
    )

    if revision_count >= MAX_ANSWER_REVISIONS:
        logfire.warning(
            "Grounding revision budget exhausted.",
            revision_count=revision_count,
            max_revisions=MAX_ANSWER_REVISIONS,
            is_grounded=is_grounded,
            answer_supported=answer_supported,
            support_score=state.get("support_score"),
        )

        return "end"

    logfire.info(
        "Grounding validation failed; routing to bounded answer revision.",
        revision_count=revision_count,
        max_revisions=MAX_ANSWER_REVISIONS,
        is_grounded=is_grounded,
        answer_supported=answer_supported,
        support_score=state.get("support_score"),
    )

    return "revision"


# =====================================================================
# SAFE HELPERS
# =====================================================================


def _safe_list(
    value: Any,
) -> List[Any]:
    """
    Convert a state value into a safe list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def _normalise_claim_record(
    record: Any,
) -> Dict[str, Any] | None:
    """
    Normalize one grounding-critic claim record.

    Supports both the current atomic-claim structure and older
    grounding-score structures.
    """

    if not isinstance(record, dict):
        return None

    supported = record.get("supported")

    # Only explicitly unsupported claims belong in the revision prompt.
    if supported is True:
        return None

    claim = str(
        record.get("claim") or record.get("atomic_claim") or record.get("text") or ""
    ).strip()

    if not claim:
        return None

    evidence = str(
        record.get("best_evidence")
        or record.get("best_evidence_unit")
        or record.get("evidence")
        or ""
    ).strip()

    citation = str(record.get("best_citation") or record.get("citation") or "").strip()

    score = record.get(
        "score",
        record.get(
            "support_score",
            record.get("claim_support_score"),
        ),
    )

    citations = record.get("citations")

    if not citation and citations:
        if isinstance(
            citations,
            (list, tuple, set),
        ):
            citation = str(
                next(
                    iter(citations),
                    "",
                )
            ).strip()
        elif isinstance(
            citations,
            str,
        ):
            citation = citations.strip()

    return {
        "claim": claim,
        "score": score,
        "best_evidence": evidence,
        "best_citation": citation,
        "supported": False,
    }


# =====================================================================
# GROUNDING DIAGNOSTIC EXTRACTION
# =====================================================================


def _extract_unsupported_atomic_claims(
    state: AgentState,
) -> List[Dict[str, Any]]:
    """
    Extract unsupported atomic claims from grounding_critic output.

    Expected structure:

        grounding_scores = {
            "atomic_claims": [
                {
                    "claim": "...",
                    "supported": False,
                    "score": 0.12,
                    "best_evidence_unit": "...",
                    "best_citation": "chunk_5",
                }
            ]
        }

    Defensive support is retained for older grounding-score formats.
    """

    grounding_scores = state.get("grounding_scores")

    if not isinstance(
        grounding_scores,
        dict,
    ):
        return []

    records: List[Any] = []

    # ---------------------------------------------------------------
    # Preferred current format
    # ---------------------------------------------------------------

    atomic_claims = grounding_scores.get("atomic_claims")

    if isinstance(
        atomic_claims,
        list,
    ):
        records.extend(atomic_claims)

    # ---------------------------------------------------------------
    # Defensive fallback
    # ---------------------------------------------------------------

    if not records:
        for key, value in grounding_scores.items():
            if key == "atomic_claims":
                continue

            if isinstance(
                value,
                list,
            ):
                records.extend(value)
                continue

            if isinstance(
                value,
                dict,
            ):
                if "claim" in value or "atomic_claim" in value or "supported" in value:
                    records.append(value)

    unsupported: List[Dict[str, Any]] = []

    for record in records:
        normalized = _normalise_claim_record(record)

        if normalized is None:
            continue

        unsupported.append(normalized)

    # ---------------------------------------------------------------
    # Deduplicate while preserving order.
    # ---------------------------------------------------------------

    seen: set[str] = set()
    deduplicated: List[Dict[str, Any]] = []

    for item in unsupported:
        claim_key = " ".join(str(item.get("claim") or "").lower().split())

        if not claim_key:
            continue

        if claim_key in seen:
            continue

        seen.add(claim_key)
        deduplicated.append(item)

    return deduplicated


# =====================================================================
# REVISION PROMPT
# =====================================================================


def _build_grounding_revision_prompt(
    state: AgentState,
    unsupported_claims: List[Dict[str, Any]],
) -> str:
    """
    Build an evidence-constrained revision instruction.

    The responder receives:

        - exact unsupported claim
        - support score
        - strongest evidence unit
        - citation

    The responder must narrow, rewrite, or delete unsupported claims.
    """

    if not unsupported_claims:
        return (
            "GROUNDING REVISION REQUIRED\n\n"
            "The previous answer failed grounding validation, but the "
            "critic did not expose a specific unsupported atomic claim.\n\n"
            "Re-read ONLY the supplied evidence and previous answer. "
            "Preserve statements directly supported by the evidence. "
            "Delete unsupported factual details. Do not add outside "
            "knowledge, inference, entities, dates, numbers, mechanisms, "
            "examples, or explanations.\n\n"
            "Every retained factual statement must have a valid citation."
        )

    lines: List[str] = [
        "GROUNDING REVISION REQUIRED",
        "",
        "The previous answer failed evidence-grounding validation.",
        "Revise the answer using ONLY the supplied evidence.",
        "",
        "MANDATORY REVISION RULES:",
        "",
        "1. Preserve claims that are directly supported.",
        "",
        "2. For each unsupported claim below, narrow it to the portion "
        "that is explicitly supported by the supplied evidence.",
        "",
        "3. If a compound sentence contains both supported and unsupported "
        "clauses, keep ONLY the supported clause.",
        "",
        "4. If no safe supported wording exists, DELETE the claim.",
        "",
        "5. Do NOT add outside knowledge.",
        "",
        "6. Do NOT infer facts from general knowledge.",
        "",
        "7. Do NOT introduce new entities, dates, numbers, mechanisms, "
        "examples, architectures, technologies, causal explanations, "
        "or performance characteristics.",
        "",
        "8. Do NOT strengthen a weak evidence statement into a stronger claim.",
        "",
        "9. Every retained factual statement MUST have a valid citation.",
        "",
        "10. A citation does NOT make an unsupported claim supported.",
        "",
        "11. The grounding critic is authoritative for support decisions.",
        "",
        "UNSUPPORTED ATOMIC CLAIMS:",
    ]

    for index, item in enumerate(
        unsupported_claims,
        start=1,
    ):
        claim = item.get("claim") or "(missing claim)"

        score = item.get("score")

        evidence = item.get("best_evidence") or "(no supporting evidence identified)"

        citation = item.get("best_citation") or "(no citation identified)"

        lines.extend(
            [
                "",
                f"{index}. UNSUPPORTED CLAIM",
                f"Claim: {claim}",
                f"Grounding score: {score}",
                f"Best evidence: {evidence}",
                f"Best citation: {citation}",
            ]
        )

    lines.extend(
        [
            "",
            "CRITICAL:",
            "If the evidence supports only one clause of an unsupported "
            "compound claim, retain only that supported clause.",
            "",
            "Return ONLY the revised answer.",
            "Do not mention the revision process.",
            "Do not mention the grounding critic.",
            "Do not mention unsupported claims.",
        ]
    )

    return "\n".join(lines)


# =====================================================================
# CITATION REVISION PROMPT
# =====================================================================


def _build_citation_revision_prompt(
    state: AgentState,
) -> str:
    """
    Build a citation-focused revision instruction when citation
    validation fails before grounding validation.
    """

    feedback = _safe_list(state.get("grounding_feedback"))

    feedback_lines = [str(item).strip() for item in feedback if str(item).strip()]

    prompt = [
        "CITATION REVISION REQUIRED",
        "",
        "The previous answer failed citation validation.",
        "",
        "Revise the answer using ONLY the supplied evidence.",
        "",
        "RULES:",
        "1. Preserve only factual claims supported by supplied evidence.",
        "2. Every retained factual statement must have a valid citation.",
        "3. Do not invent citations.",
        "4. Do not introduce new facts.",
        "5. Remove unsupported factual statements.",
        "6. Keep citations attached to the claims they support.",
    ]

    if feedback_lines:
        prompt.extend(
            [
                "",
                "VALIDATION FEEDBACK:",
                *feedback_lines[:20],
            ]
        )

    prompt.extend(
        [
            "",
            "Return ONLY the revised answer.",
        ]
    )

    return "\n".join(prompt)


# =====================================================================
# REVISION NODE
# =====================================================================


def revision_node(
    state: AgentState,
) -> Dict[str, Any]:
    """
    Prepare a bounded evidence-constrained answer revision.

    This node does not itself call the LLM.

    It creates revision instructions consumed by responder.py.
    """

    revision_count = int(
        state.get(
            "revision_count",
            0,
        )
        or 0
    )

    next_revision_count = revision_count + 1

    unsupported_claims = _extract_unsupported_atomic_claims(state)

    # ---------------------------------------------------------------
    # Prefer grounding-specific instructions when available.
    # Otherwise build citation-specific instructions.
    # ---------------------------------------------------------------

    if unsupported_claims:
        revision_prompt = _build_grounding_revision_prompt(
            state,
            unsupported_claims,
        )
    else:
        revision_prompt = _build_citation_revision_prompt(state)

    existing_plan = list(state.get("plan") or [])

    logfire.info(
        "Preparing evidence-constrained answer revision.",
        revision_count=next_revision_count,
        max_revisions=MAX_ANSWER_REVISIONS,
        citation_valid=state.get("citation_valid"),
        is_grounded=state.get("is_grounded"),
        answer_supported=state.get("answer_supported"),
        support_score=state.get("support_score"),
        unsupported_claim_count=len(unsupported_claims),
        has_revision_prompt=bool(revision_prompt),
    )

    for index, item in enumerate(
        unsupported_claims,
        start=1,
    ):
        logfire.warning(
            "Unsupported atomic claim scheduled for revision.",
            revision_count=next_revision_count,
            claim_index=index,
            claim=item.get("claim"),
            score=item.get("score"),
            best_citation=item.get("best_citation"),
        )

    return {
        # -----------------------------------------------------------
        # Revision control
        # -----------------------------------------------------------
        "revision_count": next_revision_count,
        # -----------------------------------------------------------
        # Explicit responder instruction
        # -----------------------------------------------------------
        "revision_prompt": revision_prompt,
        # -----------------------------------------------------------
        # IMPORTANT:
        # Replace stale diagnostics rather than endlessly appending
        # previous failed claims.
        # -----------------------------------------------------------
        "unsupported_atomic_claims": unsupported_claims,
        # -----------------------------------------------------------
        # Status / trace
        # -----------------------------------------------------------
        "status": (f"Answer revision {next_revision_count}/{MAX_ANSWER_REVISIONS}"),
        "plan": [
            *existing_plan,
            (f"Revision {next_revision_count}/{MAX_ANSWER_REVISIONS}"),
        ],
    }


# =====================================================================
# GRAPH CONSTRUCTION
# =====================================================================


def build_graph():
    workflow = StateGraph(AgentState)

    # ================================================================
    # NODES
    # ================================================================

    workflow.add_node(
        "planner",
        planner_node,
    )

    workflow.add_node(
        "retriever",
        retrieve_node,
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # private_retry is registered exactly ONCE.
    # ---------------------------------------------------------------

    workflow.add_node(
        "private_retry",
        retry_private_retrieval_node,
    )

    workflow.add_node(
        "grader",
        grade_documents_node,
    )

    workflow.add_node(
        "context_evaluator",
        context_evaluator_node,
    )

    workflow.add_node(
        "query_rewriter",
        rewrite_query_node,
    )

    workflow.add_node(
        "web_search",
        web_search_node,
    )

    workflow.add_node(
        "responder",
        generate_node,
    )

    workflow.add_node(
        "citation_check",
        citation_check_node,
    )

    workflow.add_node(
        "grounding_critic",
        grounding_critic_node,
    )

    workflow.add_node(
        "revision",
        revision_node,
    )

    # ================================================================
    # START
    # ================================================================

    workflow.add_edge(
        START,
        "planner",
    )

    # ================================================================
    # PLANNER
    # ================================================================

    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retriever": "retriever",
            "responder": "responder",
        },
    )

    # ================================================================
    # PRIVATE RETRIEVAL
    # ================================================================

    workflow.add_edge(
        "retriever",
        "grader",
    )

    workflow.add_edge(
        "grader",
        "context_evaluator",
    )

    # ================================================================
    # CONTEXT EVALUATION
    # ================================================================

    workflow.add_conditional_edges(
        "context_evaluator",
        route_after_context_evaluator,
        {
            "responder": "responder",
            "query_rewriter": "query_rewriter",
            "web_search": "web_search",
        },
    )

    # ================================================================
    # PRIVATE RETRIEVAL REWRITE
    # ================================================================

    workflow.add_edge(
        "query_rewriter",
        "private_retry",
    )

    workflow.add_edge(
        "private_retry",
        "grader",
    )

    # ================================================================
    # WEB SEARCH
    # ================================================================

    workflow.add_edge(
        "web_search",
        "responder",
    )

    # ================================================================
    # RESPONDER
    # ================================================================

    workflow.add_conditional_edges(
        "responder",
        route_after_responder,
        {
            "citation_check": "citation_check",
            "end": END,
        },
    )

    # ================================================================
    # CITATION GATE
    # ================================================================

    workflow.add_conditional_edges(
        "citation_check",
        route_after_citation_check,
        {
            "grounding_critic": "grounding_critic",
            "revision": "revision",
            "end": END,
        },
    )

    # ================================================================
    # GROUNDING GATE
    # ================================================================

    workflow.add_conditional_edges(
        "grounding_critic",
        route_after_grounding_critic,
        {
            "revision": "revision",
            "end": END,
        },
    )

    # ================================================================
    # REVISION
    #
    # revision -> responder
    #
    # responder.py MUST consume:
    #
    #     state["revision_prompt"]
    #
    # ================================================================

    workflow.add_edge(
        "revision",
        "responder",
    )

    # ================================================================
    # COMPILE
    # ================================================================

    return workflow.compile(
        checkpointer=MemorySaver(),
    )


# =====================================================================
# GLOBAL GRAPH INSTANCE
# =====================================================================

rag_agent = build_graph()
