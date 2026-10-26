from app.agents.graph import (
    _is_absence_answer,
    route_after_context_evaluator_scoped,
    route_after_responder,
)
from app.agents.nodes.answer_relevance import answer_abstention_node
from app.agents.nodes.grounding_critic import _is_pure_absence


def _web_state(**overrides):
    state = {"context_quality": "needs_web", "retrieval_rewrite_count": 0}
    state.update(overrides)
    return state


def test_scoped_question_rewrites_once_before_abstaining():
    state = _web_state(document_scoped=True, retrieval_rewrite_count=0)
    assert route_after_context_evaluator_scoped(state) == "query_rewriter"


def test_scoped_question_abstains_instead_of_web_search():
    state = _web_state(document_scoped=True, retrieval_rewrite_count=1)
    assert route_after_context_evaluator_scoped(state) == "abstain"


def test_unscoped_question_still_uses_web_search():
    state = _web_state(document_scoped=False)
    assert route_after_context_evaluator_scoped(state) == "web_search"


def test_pure_absence_answer_detected():
    state = {
        "candidate_answer": "The supplied evidence does not contain a direct comparison."
    }
    assert _is_absence_answer(state) is True


def test_mixed_answer_is_not_absence():
    state = {
        "candidate_answer": (
            "HyDE improves recall.\n"
            "The supplied evidence does not contain latency numbers."
        )
    }
    assert _is_absence_answer(state) is False


def test_responder_routes_absence_answer_to_abstain():
    state = {"candidate_answer": "The sources do not contain information on this."}
    assert route_after_responder(state) == "abstain"


def test_named_document_abstention_reason_and_text():
    result = answer_abstention_node(
        {"document_scoped": True, "candidate_answer": "", "evaluation_trace": []}
    )
    assert result["abstention_reason"] == "named_document_has_no_evidence"
    assert result["final_answer"].startswith("The document you referred to")
    assert result["answer_abstained"] is True


def test_generic_abstention_text_unchanged():
    result = answer_abstention_node(
        {"document_scoped": False, "answer_useful": True, "evaluation_trace": []}
    )
    assert result["final_answer"].startswith("I couldn't find evidence")


def test_grounding_critic_skips_pure_absence_only():
    assert _is_pure_absence("The supplied evidence does not contain latency numbers.")
    assert not _is_pure_absence(
        "HyDE improves recall, while the evidence does not contain latency numbers."
    )


def test_apply_pruned_answer_writes_the_pruned_text():
    from app.agents.graph import apply_pruned_answer_node

    state = {
        "final_answer": "Supported line.\nUnsupported line.",
        "candidate_answer": "Supported line.\nUnsupported line.",
        "pruned_answer": "Supported line.",
        "prunable_claims": ["Unsupported line."],
        "evaluation_trace": [],
    }

    result = apply_pruned_answer_node(state)

    assert result["final_answer"] == "Supported line."
    assert result["candidate_answer"] == "Supported line."
    assert result["answer_pruned"] is True
