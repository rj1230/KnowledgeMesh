"""
Web-search fallback node for KnowledgeMesh.

Agentic RAG flow:

    Private Retrieval
          ↓
    Private Grader
          ↓
    Context Evaluator
          ↓
    Web Search
          ↓
    Web Grader
          ↓
    Web Context Evaluator
          ↓
    Responder

Important design rules:

1. Private evidence and web evidence remain distinguishable.

2. `documents` represents the CURRENT retrieval batch.
   Therefore, after web fallback:
       documents = web_documents

3. `private_documents` preserves the original private evidence.

4. `all_documents` preserves the complete evidence history.

5. `web_search_attempted` records that the web node actually
   executed, even when zero web documents were returned.

6. `web_search_used` means usable web documents were returned.

7. Web evidence is sent back through the normal grader before
   generation.

8. The graph uses `web_search_attempted` to prevent an infinite
   web-search fallback loop.
"""

from __future__ import annotations

import logging

import logfire

from app.agents.state import AgentState
from app.services.web_search import search_web


logger = logging.getLogger("knowledgemesh")


def web_search_node(state: AgentState):
    """
    Execute external web retrieval after private evidence is
    determined to be insufficient.

    Important state contract
    ------------------------

    `documents`
        Current retrieval batch. After this node executes,
        this contains WEB documents only.

    `private_documents`
        Original private/internal evidence.

    `web_documents`
        External web evidence.

    `all_documents`
        Combined private + web evidence.

    `web_search_attempted`
        True whenever this node executes.

    `web_search_used`
        True only when usable web documents are returned.

    The graph then executes:

        web_search → grader → context_evaluator

    so web evidence is independently graded before generation.
    """

    # ============================================================
    # RESOLVE QUERY
    # ============================================================

    query = (
        state.get("search_query")
        or state.get("rewritten_query")
        or state.get("current_query")
        or state.get("original_query")
        or ""
    ).strip()

    # ============================================================
    # PRESERVE PRIVATE EVIDENCE
    # ============================================================

    current_documents = list(state.get("documents") or [])

    existing_private_documents = list(state.get("private_documents") or [])

    # If private_documents has not yet been populated, the current
    # documents are the private retrieval batch entering this node.
    if existing_private_documents:
        preserved_private_documents = existing_private_documents
    else:
        preserved_private_documents = current_documents

    logger.info(
        "🌐 WEB SEARCH START | query=%r | private_documents=%s | already_attempted=%s",
        query,
        len(preserved_private_documents),
        bool(state.get("web_search_attempted", False)),
    )

    # ============================================================
    # EMPTY QUERY
    # ============================================================
    #
    # IMPORTANT:
    # Even though no provider call can be made, the web-search
    # routing step has been attempted/executed.
    #
    # Therefore:
    #
    #     web_search_attempted = True
    #
    # This prevents the graph from repeatedly returning to
    # web_search because of an empty query.
    # ============================================================

    if not query:
        logger.warning("🌐 WEB SEARCH | empty query received")

        context_reason = (
            "Web search was attempted but skipped because the query was empty."
        )

        plan = list(state.get("plan") or [])

        plan.append("Web Search: attempted but skipped because query was empty")

        return {
            # Current retrieval batch is empty.
            "documents": [],
            # Preserve private evidence.
            "private_documents": preserved_private_documents,
            # No web evidence.
            "web_documents": [],
            # Preserve complete evidence history.
            "all_documents": list(
                state.get("all_documents") or preserved_private_documents
            ),
            # ----------------------------------------------------
            # WEB EXECUTION STATE
            # ----------------------------------------------------
            # Critical loop-prevention flag.
            "web_search_attempted": True,
            # No usable web documents.
            "web_search_used": False,
            # Web routing decision has been executed.
            "web_search_required": False,
            "should_search_web": False,
            # ----------------------------------------------------
            # CONTEXT STATE
            # ----------------------------------------------------
            "context_quality": "web_empty",
            "context_reason": context_reason,
            # ----------------------------------------------------
            # GRADING STATE
            # ----------------------------------------------------
            "grader_status": "skipped",
            "grader_error_type": None,
            "grading_mode": "no_documents",
            "graded_documents": [],
            "generation_documents": [],
            # ----------------------------------------------------
            # EXECUTION TRACE
            # ----------------------------------------------------
            "status": context_reason,
            "plan": plan,
        }

    # ============================================================
    # EXECUTE WEB SEARCH
    # ============================================================

    try:
        with logfire.span(
            "🌐 Web Search Node",
            query=query,
            private_document_count=len(preserved_private_documents),
        ):
            web_documents = search_web(
                query=query,
                max_results=5,
            )

        web_documents = list(web_documents or [])

    except Exception as exc:
        # --------------------------------------------------------
        # IMPORTANT:
        # The web node itself was attempted.
        #
        # Therefore web_search_attempted MUST be True even when
        # the provider fails.
        # --------------------------------------------------------

        logger.exception(
            "🌐 WEB SEARCH FAILED | query=%r | error=%s",
            query,
            exc,
        )

        context_reason = (
            "Web search was attempted but the external search provider failed."
        )

        plan = list(state.get("plan") or [])

        plan.extend(
            [
                "Web Search: attempted",
                "Web Search: provider failed",
            ]
        )

        return {
            "documents": [],
            "private_documents": preserved_private_documents,
            "web_documents": [],
            "all_documents": list(
                state.get("all_documents") or preserved_private_documents
            ),
            # Critical loop-prevention state.
            "web_search_attempted": True,
            "web_search_used": False,
            "web_search_required": False,
            "should_search_web": False,
            "context_quality": "web_empty",
            "context_reason": context_reason,
            "grader_status": "failed",
            "grader_error_type": type(exc).__name__,
            "grading_mode": "unavailable",
            "graded_documents": [],
            "generation_documents": [],
            "status": context_reason,
            "plan": plan,
        }

    # ============================================================
    # COMBINE EVIDENCE
    # ============================================================

    all_documents = [
        *preserved_private_documents,
        *web_documents,
    ]

    web_search_used = bool(web_documents)

    logger.info(
        "🌐 WEB SEARCH COMPLETE | "
        "query=%r | "
        "web_documents=%s | "
        "private_documents=%s | "
        "all_documents=%s",
        query,
        len(web_documents),
        len(preserved_private_documents),
        len(all_documents),
    )

    # ============================================================
    # CONTEXT STATUS
    # ============================================================
    #
    # Do NOT mark web evidence as "strong" here.
    #
    # It still needs to pass through the document grader.
    # ============================================================

    if web_search_used:
        context_quality = "web_pending_grading"

        context_reason = (
            f"Private retrieval was insufficient; web search "
            f"returned {len(web_documents)} external sources. "
            "External evidence will now be graded."
        )

    else:
        context_quality = "web_empty"

        context_reason = (
            "Private retrieval was insufficient and external "
            "search returned no usable sources."
        )

    # ============================================================
    # EXECUTION PLAN
    # ============================================================

    plan = list(state.get("plan") or [])

    plan.extend(
        [
            f"Web Search: {len(web_documents)} external sources",
            (
                "Web Evidence: sent to document grader"
                if web_documents
                else "Web Evidence: no usable sources"
            ),
        ]
    )

    # ============================================================
    # RETURN STATE
    # ============================================================

    return {
        # --------------------------------------------------------
        # CURRENT RETRIEVAL BATCH
        #
        # IMPORTANT:
        #
        #     web_search → grader
        #
        # must grade WEB documents, not the rejected private
        # documents.
        # --------------------------------------------------------
        "documents": web_documents,
        # --------------------------------------------------------
        # PRIVATE EVIDENCE
        # --------------------------------------------------------
        "private_documents": preserved_private_documents,
        # --------------------------------------------------------
        # WEB EVIDENCE
        # --------------------------------------------------------
        "web_documents": web_documents,
        # --------------------------------------------------------
        # COMPLETE EVIDENCE HISTORY
        # --------------------------------------------------------
        "all_documents": all_documents,
        # --------------------------------------------------------
        # WEB EXECUTION STATE
        # --------------------------------------------------------
        # The node executed regardless of whether results existed.
        "web_search_attempted": True,
        # True only if usable web documents were returned.
        "web_search_used": web_search_used,
        # The web-search routing decision has now been executed.
        "web_search_required": False,
        "should_search_web": False,
        # --------------------------------------------------------
        # CONTEXT STATUS
        # --------------------------------------------------------
        # Do not mark as strong before grading.
        "context_quality": context_quality,
        "context_reason": context_reason,
        # --------------------------------------------------------
        # RESET GRADING OUTPUTS
        #
        # The next graph node is the grader.
        # --------------------------------------------------------
        "graded_documents": [],
        "generation_documents": [],
        # --------------------------------------------------------
        # EXECUTION TRACE
        # --------------------------------------------------------
        "status": context_reason,
        "plan": plan,
    }
