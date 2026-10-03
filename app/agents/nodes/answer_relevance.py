from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from app.gateway.client import LLMGatewayError, create_chat_completion
from app.evaluation.trace import append_trace_event

logger = logging.getLogger(__name__)


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


def answer_relevance_node(
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate that the final answer actually answers the original question.

    The evaluator fails closed: if the evaluator itself cannot run,
    answer_useful is False and the trajectory cannot be considered
    successful.
    """

    question = str(
        state.get("original_query")
        or state.get("message")
        or ""
    ).strip()

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
            "status": "Answer relevance evaluation failed: empty answer.",
        }

    try:
        result = _evaluate_answer_relevance(
            question=question,
            answer=answer,
        )

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
            "status": (
                "Answer relevance evaluation failed unexpectedly."
            ),
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
    )

    return {
        **trace_update,
        "answer_useful": useful,
        "usefulness_score": score,
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
    Replace an unrecoverably irrelevant answer with a safe abstention.

    This prevents a grounded-but-irrelevant answer from being exposed as
    a successful final answer after the revision budget is exhausted.
    """

    abstention = (
        "I couldn't find evidence in the available sources that "
        "establishes an answer to your specific question."
    )

    trace_update = append_trace_event(
        state,
        step="answer_abstention",
        status="abstained",
        reason="answer_relevance_revision_budget_exhausted",
    )

    return {
        **trace_update,
        "final_answer": abstention,
        "candidate_answer": abstention,
        "answer_useful": False,
        "usefulness_score": 0.0,
        "answer_supported": False,
        "is_grounded": False,
        "grounding_valid": False,
        "citation_valid": False,
        "support_score": 0.0,
        "status": "Final answer abstained because relevance could not be recovered.",
    }
