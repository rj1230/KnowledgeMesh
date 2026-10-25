from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from typing import Any, Dict

from app.gateway.client import LLMGatewayError, create_chat_completion
from app.evaluation.trace import append_trace_event

logger = logging.getLogger(__name__)

# One retry for transient gateway errors or malformed JSON. Relevance is
# now the first validation gate, so a single blip would otherwise abstain
# before grounding even runs. The retry only costs time when something
# actually failed.
RELEVANCE_EVALUATOR_ATTEMPTS = 2


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM response."""

    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

    if match:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value

    raise ValueError("LLM relevance evaluator did not return valid JSON.")


def _safe_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(score, 1.0))


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "yes",
            "1",
        }

    return bool(value)


def _evaluate_answer_relevance(
    *,
    question: str,
    answer: str,
) -> Dict[str, Any]:
    """
    Semantically evaluate whether the generated answer actually answers
    the user's original question.

    This evaluator is intentionally separate from grounding:

        grounding  = evidence supports the answer
        relevance  = answer addresses the question
    """

    prompt = f"""
You are the final answer-relevance evaluator for an enterprise RAG system.

Determine whether the ANSWER actually addresses the USER QUESTION.

Important:
- Judge semantic relevance, not keyword overlap.
- An answer can be factually correct and well-cited but still be irrelevant
  to the user's question.
- Do not judge whether the retrieved evidence is good; grounding is handled
  by another evaluator.
- Do not reward an answer merely because it discusses the same broad topic.
- The answer must address the actual request, entity, relationship, policy,
  procedure, or question asked by the user.
- If the answer discusses a related but different subject, mark it NOT useful.
- Use the original user question exactly as the evaluation target.
- An answer that clearly states what the sources cover and what they do not,
  while still giving substantive content for the parts they do cover, is
  useful. Do not mark it not useful only because the sources lack a requested
  judgment such as a recommendation or ranking.
Return ONLY valid JSON with this exact structure:

{{
  "useful": true,
  "score": 0.95,
  "reason": "The answer directly addresses the user's question."
}}

USER QUESTION:
{question}

ANSWER:
{answer}
""".strip()

    response = create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict semantic answer-relevance evaluator. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content or ""

    result = _extract_json(content)

    return {
        "useful": _safe_bool(result.get("useful")),
        "score": _safe_score(result.get("score")),
        "reason": str(result.get("reason") or "").strip(),
    }


def _evaluate_with_retry(
    *,
    question: str,
    answer: str,
    attempts: int = RELEVANCE_EVALUATOR_ATTEMPTS,
) -> Dict[str, Any]:
    """
    Run the relevance evaluator, retrying transient failures.

    Retries gateway errors and unparseable output (JSONDecodeError is a
    ValueError). The last failure is re-raised so the node can mark the
    evaluator as failed.
    """

    last_exc: Exception | None = None

    for attempt in range(1, max(1, attempts) + 1):
        try:
            return _evaluate_answer_relevance(
                question=question,
                answer=answer,
            )
        except (LLMGatewayError, ValueError) as exc:
            last_exc = exc

            if attempt < attempts:
                logger.warning(
                    "Relevance evaluator attempt %s/%s failed (%s); retrying.",
                    attempt,
                    attempts,
                    type(exc).__name__,
                )

    assert last_exc is not None
    raise last_exc


def answer_relevance_node(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate that the final answer actually answers the original question.

    Pipeline position: runs AFTER citation_check and BEFORE
    grounding_critic. It reads only the question and the answer text, so
    it does not depend on any grounding output.

    The evaluator fails closed: if the evaluator itself cannot run,
    answer_useful is False and the trajectory cannot be considered
    successful.

    `relevance_evaluator_failed` distinguishes "the evaluator could not
    run" (infrastructure problem; retrieving again cannot help) from "the
    evaluator ran and judged the answer irrelevant" (a broader retrieval
    may help). The router uses it to abstain without a re-retrieval.
    """

    question = str(state.get("original_query") or state.get("message") or "").strip()

    answer = str(
        state.get("final_answer")
        or state.get("candidate_answer")
        or state.get("answer")
        or ""
    ).strip()

    if not question:
        trace_update = append_trace_event(
            state,
            step="answer_relevance",
            status="failed",
            answer_useful=False,
            usefulness_score=0.0,
            reason="missing_original_query",
        )

        return {
            **trace_update,
            "answer_useful": False,
            "usefulness_score": 0.0,
            "relevance_evaluator_failed": False,
            "status": "Answer relevance evaluation failed: missing question.",
        }

    if not answer:
        trace_update = append_trace_event(
            state,
            step="answer_relevance",
            status="failed",
            answer_useful=False,
            usefulness_score=0.0,
            reason="empty_answer",
        )

        return {
            **trace_update,
            "answer_useful": False,
            "usefulness_score": 0.0,
            "relevance_evaluator_failed": False,
            "status": "Answer relevance evaluation failed: empty answer.",
        }

    try:
        relevance_started = perf_counter()
        result = _evaluate_with_retry(
            question=question,
            answer=answer,
        )

        relevance_ms = round((perf_counter() - relevance_started) * 1000, 2)
    except LLMGatewayError as exc:
        logger.exception("Answer relevance evaluator gateway failure.")

        trace_update = append_trace_event(
            state,
            step="answer_relevance",
            status="failed",
            answer_useful=False,
            usefulness_score=0.0,
            reason="llm_gateway_failure",
            error_type=type(exc).__name__,
        )

        return {
            **trace_update,
            "answer_useful": False,
            "usefulness_score": 0.0,
            "relevance_evaluator_failed": True,
            "status": (
                "Answer relevance evaluation failed because "
                "the LLM gateway was unavailable."
            ),
        }

    except Exception as exc:
        logger.exception("Answer relevance evaluator failed.")

        trace_update = append_trace_event(
            state,
            step="answer_relevance",
            status="failed",
            answer_useful=False,
            usefulness_score=0.0,
            reason="evaluator_failure",
            error_type=type(exc).__name__,
        )

        return {
            **trace_update,
            "answer_useful": False,
            "usefulness_score": 0.0,
            "relevance_evaluator_failed": True,
            "status": ("Answer relevance evaluation failed unexpectedly."),
        }

    useful = bool(result["useful"])
    score = float(result["score"])
    reason = result["reason"]

    trace_update = append_trace_event(
        state,
        step="answer_relevance",
        status="passed" if useful else "failed",
        answer_useful=useful,
        usefulness_score=score,
        reason=reason,
        relevance_ms=relevance_ms,
    )

    return {
        **trace_update,
        "answer_useful": useful,
        "usefulness_score": score,
        "relevance_evaluator_failed": False,
        "status": (
            "Answer relevance validation passed."
            if useful
            else "Answer relevance validation failed."
        ),
    }


def answer_abstention_node(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Replace an answer that cannot pass validation with a safe abstention.

    This node records THAT an abstention happened and WHY. It deliberately
    does not rewrite citation_valid / is_grounded / answer_supported /
    support_score / answer_useful. Those describe the last answer that was
    actually evaluated, and overwriting them made a single failure look
    like a citation failure, a grounding failure and zero support at once.

    The reason is chosen in PIPELINE ORDER: generation, named-document
    scope, relevance evaluator, citation, relevance, grounding. Relevance
    runs before grounding, so a relevance failure must be recognised before
    the (not yet evaluated, or reset) grounding flags are consulted.

    Consumers must check `answer_abstained` before presenting or scoring
    the answer.
    """

    if bool(state.get("generation_failed", False)):
        reason = "generation_failed_during_revision"
        status_message = (
            "Final answer abstained because the revision could not be "
            "generated and the previous answer had failed validation."
        )
    elif (
        bool(state.get("document_scoped"))
        and not str(state.get("candidate_answer") or "").strip()
    ):
        # The question named a document, retrieval found nothing in it, and
        # the web fallback was deliberately skipped. No answer was ever
        # generated, so no validation stage failed.
        reason = "named_document_has_no_evidence"
        status_message = (
            "Final answer abstained because the document named in the "
            "question has no supporting evidence."
        )
    elif bool(state.get("relevance_evaluator_failed", False)):
        reason = "answer_relevance_evaluator_unavailable"
        status_message = (
            "Final answer abstained because the relevance evaluator was unavailable."
        )
    elif state.get("citation_valid") is False:
        reason = "citation_validation_revision_budget_exhausted"
        status_message = "Final answer abstained because citation validation failed."
    elif not bool(state.get("answer_useful", False)):
        # Relevance is evaluated before grounding, so an answer that was
        # judged off-topic never reached the grounding critic.
        reason = "answer_relevance_revision_budget_exhausted"
        status_message = (
            "Final answer abstained because relevance could not be recovered."
        )
    else:
        # Passed citation and relevance, so grounding is what failed.
        reason = "grounding_validation_revision_budget_exhausted"
        status_message = "Final answer abstained because grounding validation failed."

    if reason == "named_document_has_no_evidence":
        abstention = (
            "The document you referred to does not contain an answer to this "
            "question, and I did not substitute other sources."
        )
    else:
        abstention = (
            "I couldn't find evidence in the available sources that "
            "establishes an answer to your specific question."
        )

    trace_update = append_trace_event(
        state,
        step="answer_abstention",
        status="abstained",
        reason=reason,
    )

    return {
        **trace_update,
        "final_answer": abstention,
        "candidate_answer": abstention,
        # Single source of truth for "this response is an abstention".
        "answer_abstained": True,
        "abstention_reason": reason,
        # Do not let stale revision state leak out of a terminal node.
        "revision_requested": False,
        "revision_prompt": "",
        # Terminal abstention supersedes feedback from earlier attempts.
        "grounding_feedback": [],
        "citation_feedback": "",
        "status": status_message,
    }
