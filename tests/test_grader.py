from unittest.mock import patch

import pytest

from app.agents.nodes.grader import (
    MAX_GENERATION_DOCUMENTS,
    MIN_CONTEXT_SCORE,
    _build_document_payload,
    _extract_json,
    _select_generation_documents,
    grade_documents_node,
)


def make_doc(
    idx,
    *,
    evidence_type="CONTENT",
    relevant=True,
    score=0.90,
    content=None,
):
    return {
        "id": f"doc-{idx}",
        "content": content or f"Document {idx} content.",
        "evidence_type": evidence_type,
        "grader_relevant": relevant,
        "grader_score": score,
    }


def test_extract_json_direct_array():
    result = _extract_json(
        '[{"index": 0, "relevant": true, "score": 0.9}]'
    )

    assert isinstance(result, list)
    assert result[0]["index"] == 0


def test_extract_json_fenced_array():
    result = _extract_json(
        """```json
[{"index": 0, "relevant": true, "score": 0.9}]
```"""
    )

    assert result[0]["relevant"] is True


def test_extract_json_embedded_array():
    result = _extract_json(
        'Grader output: [{"index": 0, "relevant": false, "score": 0.2}]'
    )

    assert result[0]["relevant"] is False


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        _extract_json("not valid JSON")


def test_build_document_payload_truncates_content():
    documents = [
        {
            "content": "x" * 5000,
        }
    ]

    payload = _build_document_payload(documents)

    assert payload[0]["index"] == 0
    assert len(payload[0]["content"]) == 2500


def test_select_generation_documents_prefers_content():
    documents = [
        make_doc(0, evidence_type="REFERENCE", score=0.99),
        make_doc(1, evidence_type="CONTENT", score=0.75),
    ]

    selected, fallback_reference_count, excluded_navigation_count = (
        _select_generation_documents(documents)
    )

    assert len(selected) == 1
    assert selected[0]["evidence_type"] == "CONTENT"
    assert fallback_reference_count == 0
    assert excluded_navigation_count == 0


def test_select_generation_documents_uses_reference_fallback():
    documents = [
        make_doc(0, evidence_type="REFERENCE", score=0.90),
        make_doc(1, evidence_type="REFERENCE", score=0.80),
    ]

    selected, fallback_reference_count, excluded_navigation_count = (
        _select_generation_documents(documents)
    )

    assert len(selected) == 2
    assert all(
        doc["evidence_type"] == "REFERENCE"
        for doc in selected
    )
    assert fallback_reference_count == 2
    assert excluded_navigation_count == 0


def test_select_generation_documents_excludes_navigation():
    documents = [
        make_doc(0, evidence_type="NAVIGATION", score=0.99),
        make_doc(1, evidence_type="CONTENT", score=0.90),
    ]

    selected, fallback_reference_count, excluded_navigation_count = (
        _select_generation_documents(documents)
    )

    assert len(selected) == 1
    assert selected[0]["evidence_type"] == "CONTENT"
    assert excluded_navigation_count == 1
    assert fallback_reference_count == 0


def test_select_generation_documents_ignores_below_threshold():
    documents = [
        make_doc(
            0,
            evidence_type="CONTENT",
            score=MIN_CONTEXT_SCORE - 0.01,
        ),
        make_doc(
            1,
            evidence_type="CONTENT",
            score=MIN_CONTEXT_SCORE,
        ),
    ]

    selected, _, _ = _select_generation_documents(documents)

    assert len(selected) == 1
    assert selected[0]["id"] == "doc-1"


def test_select_generation_documents_ignores_non_relevant_documents():
    documents = [
        make_doc(
            0,
            evidence_type="CONTENT",
            relevant=False,
            score=0.99,
        ),
        make_doc(
            1,
            evidence_type="CONTENT",
            relevant=True,
            score=0.90,
        ),
    ]

    selected, _, _ = _select_generation_documents(documents)

    assert len(selected) == 1
    assert selected[0]["id"] == "doc-1"


def test_select_generation_documents_caps_at_five():
    documents = [
        make_doc(i, evidence_type="CONTENT", score=0.90 - i * 0.01)
        for i in range(8)
    ]

    selected, _, _ = _select_generation_documents(documents)

    assert len(selected) == MAX_GENERATION_DOCUMENTS
    assert len(selected) == 5


@patch("app.agents.nodes.grader.rank_evidence", side_effect=lambda docs: docs)
@patch("app.agents.nodes.grader.create_chat_completion")
def test_grade_documents_node_success(
    mock_completion,
    mock_rank,
):
    mock_completion.return_value.choices[0].message.content = """
[
  {
    "index": 0,
    "relevant": true,
    "score": 0.95,
    "reason": "Direct evidence."
  },
  {
    "index": 1,
    "relevant": false,
    "score": 0.20,
    "reason": "Not useful."
  },
  {
    "index": 2,
    "relevant": true,
    "score": 0.85,
    "reason": "Supporting evidence."
  }
]
"""

    documents = [
        make_doc(0, evidence_type="CONTENT"),
        make_doc(1, evidence_type="NAVIGATION"),
        make_doc(2, evidence_type="CONTENT"),
    ]

    state = {
        "original_query": "What is the system failure policy?",
        "documents": documents,
        "evaluation_trace": [],
        "plan": [],
    }

    result = grade_documents_node(state)

    assert result["grader_status"] == "success"
    assert result["grading_mode"] == "graded"
    assert result["context_quality"] == "strong"

    assert len(result["graded_documents"]) == 3
    assert len(result["generation_documents"]) == 2

    assert result["graded_documents"][0]["grader_relevant"] is True
    assert result["graded_documents"][0]["grader_score"] == 0.95

    assert result["graded_documents"][1]["grader_relevant"] is False
    assert result["graded_documents"][1]["grader_score"] == 0.20

    assert result["graded_documents"][2]["grader_relevant"] is True
    assert result["graded_documents"][2]["grader_score"] == 0.85

    assert all(
        doc["evidence_type"] == "CONTENT"
        for doc in result["generation_documents"]
    )

    event = result["evaluation_trace"][-1]
    assert event["step"] == "grader"
    assert event["status"] == "success"
    assert event["details"]["document_count"] == 3
    assert event["details"]["graded_count"] == 3
    assert event["details"]["generation_count"] == 2


def test_grade_documents_node_no_documents():
    state = {
        "original_query": "What is the system failure policy?",
        "documents": [],
        "evaluation_trace": [],
        "plan": [],
    }

    result = grade_documents_node(state)

    assert result["grader_status"] == "skipped"
    assert result["grading_mode"] == "no_documents"
    assert result["context_quality"] == "empty"
    assert result["graded_documents"] == []
    assert result["generation_documents"] == []
    assert result["grader_latency_ms"] == 0.0

    event = result["evaluation_trace"][-1]
    assert event["step"] == "grader"
    assert event["status"] == "skipped"


@patch(
    "app.agents.nodes.grader._batch_grade_documents",
    side_effect=RuntimeError("grader unavailable"),
)
def test_grade_documents_node_failure_preserves_private_documents(
    mock_batch_grade,
):
    documents = [
        make_doc(0, evidence_type="CONTENT"),
        make_doc(1, evidence_type="CONTENT"),
    ]

    state = {
        "original_query": "What is the system failure policy?",
        "documents": documents,
        "evaluation_trace": [],
        "plan": [],
    }

    result = grade_documents_node(state)

    assert result["grader_status"] == "failed"
    assert result["grader_error_type"] == "RuntimeError"
    assert result["grading_mode"] == "unavailable"
    assert result["context_quality"] == "unverified"

    assert result["documents"] == documents
    assert result["graded_documents"] == []
    assert result["generation_documents"] == []
    assert result["grader_latency_ms"] == 0.0

    event = result["evaluation_trace"][-1]
    assert event["step"] == "grader"
    assert event["status"] == "failed"
    assert event["details"]["document_count"] == 2
    assert event["details"]["generation_count"] == 0


@patch("app.agents.nodes.grader.rank_evidence", side_effect=lambda docs: docs)
@patch("app.agents.nodes.grader.create_chat_completion")
def test_grader_missing_result_marks_document_unverified(
    mock_completion,
    mock_rank,
):
    mock_completion.return_value.choices[0].message.content = """
[
  {
    "index": 0,
    "relevant": true,
    "score": 0.95,
    "reason": "Direct evidence."
  }
]
"""

    documents = [
        make_doc(0, evidence_type="CONTENT"),
        make_doc(1, evidence_type="CONTENT"),
    ]

    state = {
        "original_query": "What is the system failure policy?",
        "documents": documents,
        "evaluation_trace": [],
        "plan": [],
    }

    result = grade_documents_node(state)

    assert len(result["graded_documents"]) == 2

    missing = result["graded_documents"][1]

    assert missing["grader_relevant"] is False
    assert missing["grader_score"] == 0.0
    assert missing["grader_reason"] == "No valid grader result."


@patch("app.agents.nodes.grader.rank_evidence", side_effect=lambda docs: docs)
@patch("app.agents.nodes.grader.create_chat_completion")
def test_grader_score_is_clamped(
    mock_completion,
    mock_rank,
):
    mock_completion.return_value.choices[0].message.content = """
[
  {
    "index": 0,
    "relevant": true,
    "score": 4.0,
    "reason": "Over-range score."
  },
  {
    "index": 1,
    "relevant": true,
    "score": -2.0,
    "reason": "Under-range score."
  }
]
"""

    documents = [
        make_doc(0, evidence_type="CONTENT"),
        make_doc(1, evidence_type="CONTENT"),
    ]

    state = {
        "original_query": "What is the system failure policy?",
        "documents": documents,
        "evaluation_trace": [],
        "plan": [],
    }

    result = grade_documents_node(state)

    assert result["graded_documents"][0]["grader_score"] == 1.0
    assert result["graded_documents"][1]["grader_score"] == 0.0

@patch("app.agents.nodes.grader.rank_evidence", side_effect=lambda docs: docs)
@patch("app.agents.nodes.grader.create_chat_completion")
def test_grader_string_false_is_not_relevant(
    mock_completion,
    mock_rank,
):
    mock_completion.return_value.choices[0].message.content = """
[
  {
    "index": 0,
    "relevant": "false",
    "score": 0.20,
    "reason": "Not relevant."
  }
]
"""

    documents = [
        make_doc(0, evidence_type="CONTENT"),
    ]

    state = {
        "original_query": "What is the system failure policy?",
        "documents": documents,
        "evaluation_trace": [],
        "plan": [],
    }

    result = grade_documents_node(state)

    assert result["graded_documents"][0]["grader_relevant"] is False
    assert result["graded_documents"][0]["grader_score"] == 0.20
    assert result["generation_documents"] == []
