from unittest.mock import patch

import pytest

from app.agents.nodes.answer_relevance import (
    _evaluate_answer_relevance,
    _extract_json,
    _safe_bool,
    _safe_score,
    answer_abstention_node,
    answer_relevance_node,
)


def test_extract_json_plain_object():
    result = _extract_json(
        '{"useful": true, "score": 0.95, "reason": "Direct answer."}'
    )

    assert result["useful"] is True
    assert result["score"] == 0.95


def test_extract_json_fenced_json():
    result = _extract_json(
        """```json
{"useful": false, "score": 0.12, "reason": "Different subject."}
```"""
    )

    assert result["useful"] is False
    assert result["score"] == 0.12


def test_extract_json_embedded_object():
    result = _extract_json(
        'Evaluator output: {"useful": true, "score": 0.88, "reason": "Relevant."}'
    )

    assert result["useful"] is True
    assert result["score"] == 0.88


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        _extract_json("not valid json")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("0", False),
        ("anything-else", False),
    ],
)
def test_safe_bool_supported_values(value, expected):
    assert _safe_bool(value) is expected


def test_safe_score_clamps_values():
    assert _safe_score(1.5) == 1.0
    assert _safe_score(-0.5) == 0.0
    assert _safe_score("0.75") == 0.75


def test_safe_score_invalid_value_fails_closed():
    assert _safe_score("not-a-score") == 0.0
    assert _safe_score(None) == 0.0


@patch("app.agents.nodes.answer_relevance.create_chat_completion")
def test_evaluate_answer_relevance_parses_llm_result(mock_completion):
    mock_completion.return_value.choices[0].message.content = (
        '{"useful": true, "score": 0.96, '
        '"reason": "The answer directly addresses the question."}'
    )

    result = _evaluate_answer_relevance(
        question="What is KnowledgeMesh?",
        answer="KnowledgeMesh is an enterprise RAG system.",
    )

    assert result == {
        "useful": True,
        "score": 0.96,
        "reason": "The answer directly addresses the question.",
    }


def test_answer_relevance_missing_question_fails_closed():
    state = {
        "final_answer": "Some answer.",
        "evaluation_trace": [],
    }

    result = answer_relevance_node(state)

    assert result["answer_useful"] is False
    assert result["usefulness_score"] == 0.0
    assert "missing question" in result["status"].lower()

    event = result["evaluation_trace"][-1]
    assert event["step"] == "answer_relevance"
    assert event["status"] == "failed"
    assert event["details"]["reason"] == "missing_original_query"


def test_answer_relevance_empty_answer_fails_closed():
    state = {
        "original_query": "What is KnowledgeMesh?",
        "final_answer": "",
        "evaluation_trace": [],
    }

    result = answer_relevance_node(state)

    assert result["answer_useful"] is False
    assert result["usefulness_score"] == 0.0
    assert "empty answer" in result["status"].lower()

    event = result["evaluation_trace"][-1]
    assert event["step"] == "answer_relevance"
    assert event["status"] == "failed"
    assert event["details"]["reason"] == "empty_answer"


@patch("app.agents.nodes.answer_relevance.create_chat_completion")
def test_answer_relevance_success(mock_completion):
    mock_completion.return_value.choices[0].message.content = (
        '{"useful": true, "score": 0.91, "reason": "Directly relevant."}'
    )

    state = {
        "original_query": "What is KnowledgeMesh?",
        "final_answer": "KnowledgeMesh is an enterprise RAG system.",
        "evaluation_trace": [],
    }

    result = answer_relevance_node(state)

    assert result["answer_useful"] is True
    assert result["usefulness_score"] == 0.91
    assert result["status"] == "Answer relevance validation passed."

    event = result["evaluation_trace"][-1]
    assert event["step"] == "answer_relevance"
    assert event["status"] == "passed"
    assert event["details"]["answer_useful"] is True
    assert event["details"]["usefulness_score"] == 0.91


@patch("app.agents.nodes.answer_relevance.create_chat_completion")
def test_answer_relevance_irrelevant_answer_fails(mock_completion):
    mock_completion.return_value.choices[0].message.content = (
        '{"useful": false, "score": 0.08, '
        '"reason": "The answer discusses a different subject."}'
    )

    state = {
        "original_query": (
            "What is KnowledgeMesh's failure handling policy "
            "for quantum database synchronization failures?"
        ),
        "final_answer": "Quantum computers can experience decoherence.",
        "evaluation_trace": [],
    }

    result = answer_relevance_node(state)

    assert result["answer_useful"] is False
    assert result["usefulness_score"] == 0.08
    assert result["status"] == "Answer relevance validation failed."

    event = result["evaluation_trace"][-1]
    assert event["step"] == "answer_relevance"
    assert event["status"] == "failed"


@patch("app.agents.nodes.answer_relevance.create_chat_completion")
def test_answer_relevance_gateway_failure_fails_closed(mock_completion):
    from app.gateway.client import LLMGatewayError

    mock_completion.side_effect = LLMGatewayError("gateway unavailable")

    state = {
        "original_query": "What is KnowledgeMesh?",
        "final_answer": "KnowledgeMesh is an enterprise RAG system.",
        "evaluation_trace": [],
    }

    result = answer_relevance_node(state)

    assert result["answer_useful"] is False
    assert result["usefulness_score"] == 0.0

    event = result["evaluation_trace"][-1]
    assert event["step"] == "answer_relevance"
    assert event["status"] == "failed"
    assert event["details"]["reason"] == "llm_gateway_failure"


def test_answer_abstention_node():
    state = {
        "evaluation_trace": [],
        "answer_useful": False,
        "usefulness_score": 0.08,
    }

    result = answer_abstention_node(state)

    assert result["answer_useful"] is False
    assert result["usefulness_score"] == 0.0
    assert result["answer_supported"] is False
    assert result["is_grounded"] is False
    assert result["grounding_valid"] is False
    assert result["citation_valid"] is False
    assert result["support_score"] == 0.0
    assert result["final_answer"].startswith(
        "I couldn't find evidence"
    )

    event = result["evaluation_trace"][-1]
    assert event["step"] == "answer_abstention"
    assert event["status"] == "abstained"
