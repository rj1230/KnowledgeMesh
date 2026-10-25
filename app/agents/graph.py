"""
KnowledgeMesh · Agentic RAG Graph (lean)
========================================

LangGraph orchestration for the Agentic RAG pipeline.

Design rule
-----------
A retry is only worth its cost if it can change the outcome:

  * Regenerating from IDENTICAL context cannot fix a relevance problem,
    so relevance failures go back to RETRIEVAL, never to the responder.
  * The cheap check runs before the expensive one: answer_relevance
    (~2s, one LLM call) runs BEFORE grounding_critic (~10-15s of HHEM
    inference), so an off-topic answer never pays for grounding.
  * A pruned (salvaged) answer is derived from an answer that already
    passed relevance, so it is NOT re-checked for relevance.

Pipeline
--------

START
  ↓
planner
  ├── direct/conversational → responder
  └── retrieval → retriever
                    ↓
                  grader
                    ↓
              context_evaluator
                ├── strong → responder
                ├── insufficient → query_rewriter → private_retry → grader
                └── web fallback → web_search → grader

responder
  ├── generation failed, first attempt → trajectory_evaluation
  ├── generation failed during a revision
  │      ├── supported lines can be salvaged → apply_pruned_answer
  │      └── otherwise → answer_abstention
  └── answer generated → citation_check
                           ├── PASS → answer_relevance
                           │            ├── useful → grounding_critic
                           │            ├── evaluator failed → answer_abstention
                           │            ├── not useful, retrieval budget left
                           │            │      → prepare_reretrieval → query_rewriter
                           │            │        → private_retry → grader → ...
                           │            └── not useful, budget spent → answer_abstention
                           ├── FAIL, budget left → prepare_revision
                           └── FAIL, budget spent → answer_abstention

grounding_critic
  ├── PASS → trajectory_evaluation
  ├── FAIL, salvage viable → apply_pruned_answer → grounding_critic
  ├── FAIL, budget left → prepare_revision → responder
  └── FAIL, budget spent, nothing salvageable → answer_abstention

answer_abstention → trajectory_evaluation → END


Budgets (all independent)
-------------------------
MAX_RETRIEVAL_REWRITES     query rewrites inside the retrieval loop
MAX_ANSWER_REVISIONS       regenerations for citation/grounding failures
MAX_RELEVANCE_RETRIEVALS   broader re-retrievals after a relevance failure

Relevance failures no longer consume the grounding/citation revision
budget.


Assumptions to verify in the nodes this file does not contain
-------------------------------------------------------------
1. answer_relevance_node must not depend on grounding_critic output
   (it now runs first). It should only need the question and the
   candidate/final answer.
2. private_retry should merge with the documents already in state and
   raise final_k when state["expand_retrieval"] is True, then clear the
   flag. The query_rewriter should rewrite from the ORIGINAL user query.
3. grounding_critic only helps SALVAGE_FIRST if it offers `pruned_answer`
   even while revision budget remains. If it only offers it once the
   budget is spent, SALVAGE_FIRST is a harmless no-op.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict
import re

import logfire

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.nodes.citation_check import (
    citation_check_node,
)
from app.agents.nodes.context_evaluator_node import (
    context_evaluator_node,
)
from app.agents.nodes.grader import (
    grade_documents_node,
)
from app.agents.nodes.grounding_critic import (
    grounding_critic_node,
)
from app.agents.nodes.answer_relevance import (
    answer_abstention_node,
    answer_relevance_node,
)
from app.agents.nodes.planner import (
    planner_node,
)
from app.agents.nodes.private_retry import (
    retry_private_retrieval_node,
)
from app.agents.nodes.query_rewriter import (
    rewrite_query_node,
)
from app.agents.nodes.responder import (
    generate_node,
)
from app.agents.nodes.retriever import (
    retrieve_node,
)
from app.agents.nodes.web_search_node import (
    web_search_node,
)
from app.agents.state import AgentState
from app.evaluation.trace import append_trace_event
from app.evaluation.trajectory import evaluate_trajectory


# ============================================================
# Configuration
# ============================================================

MAX_RETRIEVAL_REWRITES = 2

MAX_ANSWER_REVISIONS = 2

# One broader retrieval after a relevance failure. Regenerating from the
# same context is never attempted for relevance.
MAX_RELEVANCE_RETRIEVALS = 1

# When grounding fails and the critic can delete just the unsupported
# lines, ship that instead of regenerating (saves one generation plus a
# full HHEM pass). Only used if the pruned answer keeps at least
# MIN_SALVAGE_RATIO of the original length.
SALVAGE_FIRST = True
MIN_SALVAGE_RATIO = 0.5


logger = logging.getLogger(__name__)


def _timed_graph_node(
    node_name: str,
    node_func,
):
    """Wrap a LangGraph node with wall-clock timing instrumentation."""

    def wrapped(state):
        started = time.perf_counter()

        try:
            return node_func(state)
        finally:
            elapsed_ms = round(
                (time.perf_counter() - started) * 1000,
                2,
            )

            logfire.info(
                "Graph node timing",
                node=node_name,
                elapsed_ms=elapsed_ms,
            )

            logger.warning(
                "GRAPH_NODE_TIMING node=%s elapsed_ms=%s",
                node_name,
                elapsed_ms,
            )

    return wrapped


# ============================================================
# Helpers
# ============================================================


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a state value to int."""

    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _revision_count(
    state: AgentState,
) -> int:
    """
    Canonical answer revision count.

    answer_revision_count is preferred; revision_count is a
    compatibility fallback for older nodes/state.
    """

    value = state.get("answer_revision_count")

    if value is None:
        value = state.get("revision_count", 0)

    return max(0, _safe_int(value))


def _normalise_feedback(
    value: Any,
) -> list[str]:
    """
    Normalize grounding/citation feedback into a list of strings.

    Never iterate directly over a string (that yields characters).
    """

    if value is None:
        return []

    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []

    if isinstance(value, (list, tuple, set)):
        return [c for c in (str(item).strip() for item in value) if c]

    cleaned = str(value).strip()
    return [cleaned] if cleaned else []


def _salvage_available(state: AgentState) -> bool:
    """A pruned answer exists and salvage has not been used yet."""

    return bool(state.get("pruned_answer")) and not state.get("answer_pruned", False)


def _salvage_is_substantial(state: AgentState) -> bool:
    """The pruned answer kept enough of the original to be worth shipping."""

    pruned = str(state.get("pruned_answer") or "").strip()

    if not pruned:
        return False

    original = str(
        state.get("final_answer") or state.get("candidate_answer") or ""
    ).strip()

    if not original:
        return True

    return len(pruned) >= MIN_SALVAGE_RATIO * len(original)


# ============================================================
# Planner routing
# ============================================================


def route_after_planner(
    state: AgentState,
) -> str:
    """Send the request straight to the responder or through retrieval."""

    route = str(state.get("route") or "").strip().lower()

    retrieval_required = bool(state.get("retrieval_required", True))

    if (
        route in {"responder", "direct", "conversation", "chat", "simple"}
        and not retrieval_required
    ):
        logger.info("Planner route: responder")
        return "responder"

    logger.info("Planner route: retriever")
    return "retriever"


# ============================================================
# Context evaluator routing
# ============================================================


def route_after_context_evaluator(
    state: AgentState,
) -> str:
    """
    Decide whether context is sufficient, should be rewritten,
    or should escalate to web search.
    """

    quality = str(state.get("context_quality") or "").strip().lower()

    web_search_attempted = bool(state.get("web_search_attempted", False))
    web_search_required = bool(state.get("web_search_required", False))
    grader_status = str(state.get("grader_status") or "").strip().lower()
    grading_mode = str(state.get("grading_mode") or "").strip().lower()

    rewrite_count = _safe_int(state.get("retrieval_rewrite_count", 0))
    rewrites_left = rewrite_count < MAX_RETRIEVAL_REWRITES

    # 1. Strong validated context.
    if quality == "strong":
        logger.info("Context route: responder | strong context")
        return "responder"

    # 2. Web already attempted: never loop on web search.
    if web_search_attempted:
        logger.warning("Web search already attempted; routing to responder.")
        return "responder"

    # 3. Insufficient private context: rewrite while budget lasts,
    #    then escalate to web. Takes precedence over web_search_required
    #    because insufficient context sets that flag too.
    if quality == "insufficient":
        if not rewrites_left:
            logger.info("Rewrite budget exhausted; routing to web_search.")
            return "web_search"

        logger.info(
            "Context route: query_rewriter | rewrite_count=%s/%s",
            rewrite_count,
            MAX_RETRIEVAL_REWRITES,
        )
        return "query_rewriter"

    # 4. Explicit web requirement.
    if quality == "needs_web" or web_search_required:
        logger.info("Context route: web_search | explicit web requirement")
        return "web_search"

    # 5. Grader failure: private evidence could not be validated.
    if grader_status == "failed" or grading_mode == "unavailable":
        logger.warning("Context route: web_search | grader unavailable")
        return "web_search"

    # 6. Default.
    if not rewrites_left:
        logger.info("Rewrite budget exhausted; routing to web_search.")
        return "web_search"

    logger.info("Context route: query_rewriter | insufficient private context")
    return "query_rewriter"


# ============================================================
# Responder routing
# ============================================================
_ABSENCE = re.compile(
    r"\b(does|do|did) not (contain|provide|include|mention|state|specify|establish)"
    r"|\bno (evidence|information|data)\b|could(n't| not) find",
    re.IGNORECASE,
)


def _is_absence_answer(state) -> bool:
    answer = str(
        state.get("final_answer") or state.get("candidate_answer") or ""
    ).strip()
    lines = [ln for ln in answer.splitlines() if ln.strip()]
    # EVERY line must be an absence statement, so mixed answers are untouched.
    return 0 < len(lines) <= 3 and all(_ABSENCE.search(ln) for ln in lines)


def route_after_responder(
    state: AgentState,
) -> str:
    """
    Route a generated answer into citation validation.

    First-attempt generation failure ends the graph. A failure during a
    revision falls back to the previous (already failed) answer, which
    must not be shipped as valid: salvage its supported lines or abstain.

    An answer that only says the sources lack the information skips
    validation and revisions, since regenerating from the same context
    cannot change the outcome.
    """

    if bool(state.get("generation_failed", False)):
        if _revision_count(state) > 0:
            if _salvage_available(state):
                logger.warning(
                    "Responder failed during a revision; salvaging the "
                    "supported lines of the previous answer."
                )
                return "salvage"

            logger.warning(
                "Responder failed during a revision and nothing is "
                "salvageable; abstaining."
            )
            return "abstain"

        logger.warning("Responder generation failed; ending graph.")
        return "end"

    if _is_absence_answer(state):
        logger.info(
            "Responder route: abstain | answer says the sources have no evidence"
        )
        return "abstain"

    logger.info("Responder route: citation_check | answer generated")
    return "citation_check"


# ============================================================
# Citation routing
# ============================================================


def route_after_citation_check(
    state: AgentState,
) -> str:
    """
    Citation PASS → answer_relevance (cheap check first).
    Citation FAIL → bounded revision, then abstain.
    """

    if bool(state.get("citation_valid", False)):
        logger.info("Citation route: answer_relevance | citation validation passed")
        return "answer_relevance"

    revision_count = _revision_count(state)

    if revision_count >= MAX_ANSWER_REVISIONS:
        logger.warning(
            "Citation validation failed and revision budget is exhausted. Abstaining.",
            extra={
                "revision_count": revision_count,
                "max_answer_revisions": MAX_ANSWER_REVISIONS,
            },
        )
        return "abstain"

    logger.warning(
        "Citation route: revision | revisions=%s/%s",
        revision_count,
        MAX_ANSWER_REVISIONS,
    )
    return "revision"


# ============================================================
# Answer relevance routing
# ============================================================


def route_after_answer_relevance(
    state: AgentState,
) -> str:
    """
    Useful answer → grounding_critic.

    Not useful → the CONTEXT is the likely problem, so retry retrieval
    (once, broader). Never regenerate from the same evidence.

    Evaluator unavailable → abstain; regenerating cannot fix an
    infrastructure failure.
    """

    if bool(state.get("answer_useful", False)):
        logger.info("Answer relevance route: grounding_critic | useful")
        return "grounding_critic"

    if bool(state.get("relevance_evaluator_failed", False)):
        logger.warning("Answer relevance evaluator failed; abstaining without retry.")
        return "abstain"

    relevance_retrievals = _safe_int(state.get("relevance_retrieval_count", 0))
    rewrite_count = _safe_int(state.get("retrieval_rewrite_count", 0))

    can_reretrieve = (
        relevance_retrievals < MAX_RELEVANCE_RETRIEVALS
        and rewrite_count < MAX_RETRIEVAL_REWRITES
        and not bool(state.get("web_search_attempted", False))
    )

    if can_reretrieve:
        logger.warning(
            "Answer relevance route: re_retrieve | retrievals=%s/%s",
            relevance_retrievals,
            MAX_RELEVANCE_RETRIEVALS,
        )
        return "re_retrieve"

    logger.warning(
        "Answer relevance failed and no new context is available. Abstaining."
    )
    return "abstain"


# ============================================================
# Grounding routing
# ============================================================


def route_after_grounding_critic(
    state: AgentState,
) -> str:
    """
    Strict grounding validation (the expensive check, run last).

    Canonical signal: is_grounded. Fallbacks: grounding_valid,
    answer_supported.

    PASS → trajectory_evaluation (relevance already passed).

    FAIL:
      1. salvage when a substantial pruned answer exists (SALVAGE_FIRST),
      2. else revise while budget remains,
      3. else salvage any pruned answer, else abstain.
    """

    is_grounded_value = state.get("is_grounded")

    if is_grounded_value is None:
        is_grounded_value = state.get("grounding_valid")

    if is_grounded_value is None:
        is_grounded_value = state.get("answer_supported")

    grounding_valid = bool(is_grounded_value)

    revision_count = _revision_count(state)

    logger.info(
        "Grounding routing | is_grounded=%s | answer_supported=%s | "
        "resolved=%s | revision_count=%s/%s",
        state.get("is_grounded"),
        state.get("answer_supported"),
        grounding_valid,
        revision_count,
        MAX_ANSWER_REVISIONS,
    )

    if grounding_valid:
        logger.info("Grounding route: end | grounding validation passed")
        return "end"

    # Cheapest recovery: cut the unsupported lines, no regeneration.
    if SALVAGE_FIRST and _salvage_available(state) and _salvage_is_substantial(state):
        logger.warning(
            "Grounding route: salvage | removing %s unsupported claim(s) "
            "instead of regenerating.",
            len(state.get("prunable_claims") or []),
        )
        return "salvage"

    if revision_count < MAX_ANSWER_REVISIONS:
        logger.warning(
            "Grounding route: revision | revisions=%s/%s",
            revision_count,
            MAX_ANSWER_REVISIONS,
        )
        return "revision"

    # Budget spent: last-chance salvage, otherwise abstain.
    if _salvage_available(state):
        logger.warning(
            "Grounding route: salvage | revision budget exhausted; "
            "removing %s unsupported claim(s) and re-validating.",
            len(state.get("prunable_claims") or []),
        )
        return "salvage"

    logger.warning(
        "Grounding failed, revision budget exhausted, nothing salvageable. Abstaining.",
        extra={
            "revision_count": revision_count,
            "max_answer_revisions": MAX_ANSWER_REVISIONS,
        },
    )
    return "abstain"


# ============================================================
# Revision preparation (citation / grounding failures only)
# ============================================================


def prepare_revision_node(
    state: AgentState,
) -> Dict[str, Any]:
    """
    Prepare exactly one bounded answer revision.

    Only citation and grounding failures reach this node. Relevance
    failures go to re-retrieval instead, because regenerating from the
    same evidence cannot fix them.

    Stage classification:
        citation_valid is False → citation failed; grounding did not run,
                                  so grounding feedback/prompt is stale.
        otherwise               → grounding failed.
    """

    current_count = _revision_count(state)
    next_count = current_count + 1

    if next_count > MAX_ANSWER_REVISIONS:
        logger.warning(
            "Revision request blocked because revision budget is exhausted.",
            extra={
                "revision_count": current_count,
                "max_answer_revisions": MAX_ANSWER_REVISIONS,
            },
        )

        trace_update = append_trace_event(
            state,
            step="revision",
            status="blocked",
            revision_count=current_count,
            max_answer_revisions=MAX_ANSWER_REVISIONS,
            reason="revision_budget_exhausted",
        )

        return {
            **trace_update,
            "revision_requested": False,
            "answer_revision_count": current_count,
            "revision_count": current_count,
            "status": "Revision blocked: maximum answer revisions reached.",
        }

    citation_failure = state.get("citation_valid") is False

    citation_feedback = str(state.get("citation_feedback") or "").strip()
    grounding_feedback = _normalise_feedback(state.get("grounding_feedback"))

    if citation_failure:
        grounding_feedback = []

    feedback_parts: list[str] = []

    if citation_feedback:
        feedback_parts.append(citation_feedback)

    feedback_parts.extend(grounding_feedback)

    revision_prompt = str(state.get("revision_prompt") or "").strip()

    if citation_failure:
        # Written by the critic for an earlier answer: stale.
        revision_prompt = ""

    if not revision_prompt:
        revision_prompt = (
            "Revise the previous answer using only the supplied "
            "evidence. Remove unsupported claims and ensure every "
            "substantive factual statement has a valid citation."
        )

    missing_feedback = [part for part in feedback_parts if part not in revision_prompt]

    if missing_feedback:
        revision_prompt = f"{revision_prompt}\n\nValidation feedback:\n" + "\n\n".join(
            missing_feedback
        )

    logger.info(
        "Preparing answer revision %s/%s",
        next_count,
        MAX_ANSWER_REVISIONS,
    )

    with logfire.span(
        "🔄 Prepare Answer Revision",
        current_revision=current_count,
        next_revision=next_count,
        max_answer_revisions=MAX_ANSWER_REVISIONS,
        has_citation_feedback=bool(citation_feedback),
        grounding_feedback_count=len(grounding_feedback),
        has_revision_prompt=bool(revision_prompt),
    ):
        revision_trigger = []

        if citation_failure or citation_feedback:
            revision_trigger.append("citation_failure")

        if grounding_feedback:
            revision_trigger.append("grounding_failure")

        trace_update = append_trace_event(
            state,
            step="revision",
            status="requested",
            revision_count=next_count,
            previous_revision_count=current_count,
            max_answer_revisions=MAX_ANSWER_REVISIONS,
            trigger=revision_trigger or ["validation_failure"],
            has_citation_feedback=bool(citation_feedback),
            grounding_feedback_count=len(grounding_feedback),
            has_revision_prompt=bool(revision_prompt),
        )

        return {
            **trace_update,
            "revision_requested": True,
            "answer_revision_count": next_count,
            "revision_count": next_count,
            "revision_prompt": revision_prompt,
            "grounding_feedback": grounding_feedback,
            "citation_feedback": citation_feedback,
            # Reset per-pass flags so the next pass is classified from
            # its own results. generate_node must not read these.
            "is_grounded": False,
            "answer_supported": False,
            "answer_useful": False,
            "relevance_evaluator_failed": False,
            "status": (
                f"Preparing answer revision {next_count}/{MAX_ANSWER_REVISIONS}."
            ),
        }


# ============================================================
# Re-retrieval after a relevance failure
# ============================================================


def prepare_reretrieval_node(
    state: AgentState,
) -> Dict[str, Any]:
    """
    Send a relevance failure back to retrieval instead of the responder.

    Resets all per-answer state so the fresh context produces a fresh
    answer, and gives that answer a fresh citation/grounding revision
    budget (the old one was spent on a different context). This is
    bounded: MAX_RELEVANCE_RETRIEVALS total.

    `expand_retrieval` tells private_retry to merge with existing
    documents and raise final_k. private_retry should clear it.
    """

    count = _safe_int(state.get("relevance_retrieval_count", 0)) + 1

    trace_update = append_trace_event(
        state,
        step="re_retrieval",
        status="requested",
        trigger=["answer_relevance_failure"],
        relevance_retrieval_count=count,
        max_relevance_retrievals=MAX_RELEVANCE_RETRIEVALS,
    )

    logger.info(
        "Preparing re-retrieval %s/%s after relevance failure",
        count,
        MAX_RELEVANCE_RETRIEVALS,
    )

    return {
        **trace_update,
        "relevance_retrieval_count": count,
        "expand_retrieval": True,
        # Fresh answer, fresh validation budget.
        "answer_revision_count": 0,
        "revision_count": 0,
        "revision_requested": False,
        "revision_prompt": "",
        "generation_failed": False,
        "citation_valid": False,
        "is_grounded": False,
        "answer_supported": False,
        "answer_useful": False,
        "relevance_evaluator_failed": False,
        "answer_pruned": False,
        "pruned_answer": "",
        "pruned_claims": [],
        "prunable_claims": [],
        "grounding_feedback": [],
        "citation_feedback": "",
        "status": "Answer was off-topic for the retrieved context; retrieving again.",
    }


# ============================================================
# Salvage
# ============================================================


def apply_pruned_answer_node(
    state: AgentState,
) -> Dict[str, Any]:
    """
    Swap in the critic's pruned answer (unsupported lines deleted).

    Control returns to the grounding critic, which re-validates the
    pruned answer from cached scores. answer_pruned=True ensures salvage
    is attempted at most once. Relevance is not re-checked: the pruned
    answer is a subset of an answer that already passed relevance.
    """

    pruned = str(state.get("pruned_answer") or "").strip()
    removed = list(state.get("prunable_claims") or [])

    trace_update = append_trace_event(
        state,
        step="answer_pruned",
        status="applied",
        reason="grounding_failure_salvage",
        removed_claim_count=len(removed),
    )

    return {
        **trace_update,
        "answer_before_pruning": state.get("final_answer")
        or state.get("candidate_answer"),
        "final_answer": pruned,
        "candidate_answer": pruned,
        "answer_pruned": True,
        "generation_failed": False,
        "pruned_claims": removed,
        "pruned_answer": "",
        "revision_prompt": "",
        "revision_requested": False,
        "status": (
            f"Removed {len(removed)} unsupported claim(s); "
            "re-validating the remaining answer."
        ),
    }


# ============================================================
# Trajectory evaluation
# ============================================================


def trajectory_evaluation_node(state: AgentState):
    """
    Evaluate the completed Agentic RAG trajectory.

    Observational only; does not affect routing.
    """

    logger.info(
        "TRAJECTORY EVALUATION START | trace_events=%s | "
        "citation_valid=%s | is_grounded=%s | abstained=%s",
        len(state.get("evaluation_trace") or []),
        state.get("citation_valid"),
        state.get("is_grounded"),
        state.get("answer_abstained"),
    )

    evaluation = evaluate_trajectory(state)

    logger.info(
        "TRAJECTORY EVALUATION COMPLETE | status=%s | hop_count=%s | "
        "required_hops=%s | failures=%s",
        evaluation.get("trajectory_status"),
        evaluation.get("hop_count"),
        evaluation.get("required_hops"),
        evaluation.get("failure_taxonomy"),
    )

    return {
        "trajectory_status": evaluation["trajectory_status"],
        "hop_count": evaluation["hop_count"],
        "required_hops": evaluation["required_hops"],
        "failure_taxonomy": evaluation["failure_taxonomy"],
        "recovered_failures": evaluation["recovered_failures"],
        "terminal_failures": evaluation["terminal_failures"],
        "trajectory_metrics": evaluation["metrics"],
        "evaluation_trace": evaluation["evaluation_trace"],
    }


# ============================================================
# Graph construction
# ============================================================


def build_graph():
    """Build and compile the KnowledgeMesh Agentic RAG graph."""

    workflow = StateGraph(AgentState)

    # ---------------- Nodes ----------------

    workflow.add_node("planner", _timed_graph_node("planner", planner_node))
    workflow.add_node("retriever", _timed_graph_node("retriever", retrieve_node))
    workflow.add_node("grader", _timed_graph_node("grader", grade_documents_node))
    workflow.add_node(
        "context_evaluator",
        _timed_graph_node("context_evaluator", context_evaluator_node),
    )
    workflow.add_node(
        "query_rewriter",
        _timed_graph_node("query_rewriter", rewrite_query_node),
    )
    workflow.add_node(
        "private_retry",
        _timed_graph_node("private_retry", retry_private_retrieval_node),
    )
    workflow.add_node(
        "web_search",
        _timed_graph_node("web_search", web_search_node),
    )

    # The responder is also the revision generator.
    workflow.add_node("responder", _timed_graph_node("responder", generate_node))
    workflow.add_node(
        "citation_check",
        _timed_graph_node("citation_check", citation_check_node),
    )
    workflow.add_node(
        "answer_relevance",
        _timed_graph_node("answer_relevance", answer_relevance_node),
    )
    workflow.add_node(
        "grounding_critic",
        _timed_graph_node("grounding_critic", grounding_critic_node),
    )
    workflow.add_node(
        "answer_abstention",
        _timed_graph_node("answer_abstention", answer_abstention_node),
    )
    workflow.add_node(
        "trajectory_evaluation",
        _timed_graph_node("trajectory_evaluation", trajectory_evaluation_node),
    )
    workflow.add_node(
        "apply_pruned_answer",
        _timed_graph_node("apply_pruned_answer", apply_pruned_answer_node),
    )
    workflow.add_node(
        "prepare_revision",
        _timed_graph_node("prepare_revision", prepare_revision_node),
    )
    workflow.add_node(
        "prepare_reretrieval",
        _timed_graph_node("prepare_reretrieval", prepare_reretrieval_node),
    )

    # ---------------- Entry / planner ----------------

    workflow.add_edge(START, "planner")

    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {"retriever": "retriever", "responder": "responder"},
    )

    # ---------------- Retrieval loop ----------------

    workflow.add_edge("retriever", "grader")
    workflow.add_edge("grader", "context_evaluator")

    workflow.add_conditional_edges(
        "context_evaluator",
        route_after_context_evaluator,
        {
            "responder": "responder",
            "query_rewriter": "query_rewriter",
            "web_search": "web_search",
        },
    )

    workflow.add_edge("query_rewriter", "private_retry")
    workflow.add_edge("private_retry", "grader")
    workflow.add_edge("web_search", "grader")

    # ---------------- Responder ----------------

    workflow.add_conditional_edges(
        "responder",
        route_after_responder,
        {
            "citation_check": "citation_check",
            "salvage": "apply_pruned_answer",
            "abstain": "answer_abstention",
            "end": "trajectory_evaluation",
        },
    )

    # ---------------- Citation → relevance → grounding ----------------

    workflow.add_conditional_edges(
        "citation_check",
        route_after_citation_check,
        {
            "answer_relevance": "answer_relevance",
            "revision": "prepare_revision",
            "abstain": "answer_abstention",
        },
    )

    workflow.add_conditional_edges(
        "answer_relevance",
        route_after_answer_relevance,
        {
            "grounding_critic": "grounding_critic",
            "re_retrieve": "prepare_reretrieval",
            "abstain": "answer_abstention",
        },
    )

    workflow.add_conditional_edges(
        "grounding_critic",
        route_after_grounding_critic,
        {
            "end": "trajectory_evaluation",
            "revision": "prepare_revision",
            "salvage": "apply_pruned_answer",
            "abstain": "answer_abstention",
        },
    )

    # Salvage: re-validate grounding only (cached scores, ~0.3s).
    workflow.add_edge("apply_pruned_answer", "grounding_critic")

    # ---------------- Revision / re-retrieval ----------------

    workflow.add_edge("prepare_revision", "responder")

    # Relevance failure goes back to the start of the retrieval loop.
    workflow.add_edge("prepare_reretrieval", "query_rewriter")

    # ---------------- Terminal ----------------

    workflow.add_edge("answer_abstention", "trajectory_evaluation")
    workflow.add_edge("trajectory_evaluation", END)

    return workflow.compile(checkpointer=MemorySaver())


# ============================================================
# Compiled graph
# ============================================================

rag_agent = build_graph()
