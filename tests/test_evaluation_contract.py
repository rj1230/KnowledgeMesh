import sys

import pytest

from evals import evaluate as evaluate_cli
from evals.evaluator import EvaluationCase, KnowledgeMeshEvaluator
from evals.metrics import evaluate_quality_gate


def test_missing_adversarial_dataset_fails_validation(monkeypatch, tmp_path):
    missing = tmp_path / "missing_adversarial.json"
    monkeypatch.setattr(evaluate_cli, "DEFAULT_ADVERSARIAL_DATASET", missing)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--skip-retrieval",
            "--skip-unanswerable",
            "--validate-only",
        ],
    )

    assert evaluate_cli.main() == 2


def test_missing_unsupported_dataset_fails_validation(monkeypatch, tmp_path):
    missing = tmp_path / "missing_unsupported.json"
    monkeypatch.setattr(evaluate_cli, "DEFAULT_UNANSWERABLE_DATASET", missing)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--skip-retrieval",
            "--skip-adversarial",
            "--validate-only",
        ],
    )

    assert evaluate_cli.main() == 2


def test_missing_grounding_telemetry_fails_pipeline_quality_gate():
    evaluator = KnowledgeMeshEvaluator(
        pipeline=lambda question: {"answer": "A test answer."},
        thresholds={
            "grounding_score": 0.90,
            "unsupported_claim_rate": 0.10,
        },
    )

    summary = evaluator.evaluate(
        [EvaluationCase(id="missing-grounding", question="Explain the test.")]
    )

    assert summary.metrics["grounding_score"] is None
    assert summary.metrics["unsupported_claim_rate"] is None
    assert summary.quality_gate["checks"]["grounding_score"]["value"] == 0.0
    assert summary.quality_gate["checks"]["grounding_score"]["passed"] is False
    assert (
        summary.quality_gate["checks"]["unsupported_claim_rate"]["value"] == 1.0
    )
    assert (
        summary.quality_gate["checks"]["unsupported_claim_rate"]["passed"] is False
    )
    assert summary.quality_gate["passed"] is False


def test_generic_quality_gate_skips_inapplicable_metric():
    result = evaluate_quality_gate(
        {"grounding_score": None},
        {"grounding_score": 0.90},
    )

    assert result["checks"]["grounding_score"]["skipped"] is True
    assert result["checks"]["grounding_score"]["passed"] is True
    assert result["passed"] is True


def test_explicit_zero_claims_are_not_missing_grounding_telemetry():
    evaluator = KnowledgeMeshEvaluator(
        pipeline=lambda question: {
            "answer": "No factual claims were made.",
            "total_claims": 0,
            "unsupported_claims": 0,
        },
        thresholds={
            "grounding_score": 0.90,
            "unsupported_claim_rate": 0.10,
        },
    )

    summary = evaluator.evaluate(
        [EvaluationCase(id="zero-claims", question="Ask a test question.")]
    )

    assert summary.metrics["grounding_score"] == 1.0
    assert summary.metrics["unsupported_claim_rate"] == 0.0
    assert summary.quality_gate["checks"]["grounding_score"]["passed"] is True
    assert (
        summary.quality_gate["checks"]["unsupported_claim_rate"]["passed"] is True
    )


def _run_validation_with_dataset(tmp_path, monkeypatch, cases, *, allow_unknown=False):
    import json

    dataset = tmp_path / "golden_questions.json"
    inventory = tmp_path / "inventory.csv"

    dataset.write_text(json.dumps(cases), encoding="utf-8")
    inventory.write_text(
        '"DocumentId","Filename","SourceType","ChunkId","Characters","Text"\n'
        '"known-doc","RAG.pdf","private","0","10","0123456789"\n',
        encoding="utf-8",
    )

    argv = [
        "evaluate",
        "--dataset", str(dataset),
        "--inventory", str(inventory),
        "--skip-adversarial",
        "--skip-unanswerable",
        "--validate-only",
    ]

    if allow_unknown:
        argv.append("--allow-unknown-gold")

    monkeypatch.setattr(sys, "argv", argv)
    return evaluate_cli.main()


def test_unknown_gold_fails_validation_by_default(tmp_path, monkeypatch):
    cases = [{
        "id": "unknown-gold",
        "question": "What does the document say?",
        "relevant_chunks": [{"document_id": "missing-doc", "chunk_id": 99}],
        "expected_route": "private",
    }]

    assert _run_validation_with_dataset(
        tmp_path, monkeypatch, cases
    ) == 2


def test_allow_unknown_gold_permits_unknown_chunks(tmp_path, monkeypatch):
    cases = [{
        "id": "unknown-gold",
        "question": "What does the document say?",
        "relevant_chunks": [{"document_id": "missing-doc", "chunk_id": 99}],
        "expected_route": "private",
    }]

    assert _run_validation_with_dataset(
        tmp_path, monkeypatch, cases, allow_unknown=True
    ) == 0


def test_allow_unknown_gold_does_not_permit_cases_without_gold(
    tmp_path, monkeypatch
):
    cases = [{
        "id": "missing-gold",
        "question": "What does the document say?",
        "expected_route": "private",
    }]

    assert _run_validation_with_dataset(
        tmp_path, monkeypatch, cases, allow_unknown=True
    ) == 2


def test_allow_unknown_gold_does_not_permit_unrecognized_routes(
    tmp_path, monkeypatch
):
    cases = [{
        "id": "invalid-route",
        "question": "What does the document say?",
        "relevant_chunks": [{"document_id": "known-doc", "chunk_id": 0}],
        "expected_route": "not-a-real-route",
    }]

    assert _run_validation_with_dataset(
        tmp_path, monkeypatch, cases, allow_unknown=True
    ) == 2


def test_api_normalization_preserves_runtime_usefulness_signals():
    useful = evaluate_cli.normalize_api_response({
        "answer": "The answer is grounded.",
        "answer_useful": True,
        "usefulness_score": 0.82,
    })
    not_useful = evaluate_cli.normalize_api_response({
        "answer": "The answer is incomplete.",
        "answer_useful": False,
        "usefulness_score": 0.0,
    })

    assert useful["runtime_answer_useful"] is True
    assert useful["runtime_answer_usefulness_score"] == 0.82
    assert not_useful["runtime_answer_useful"] is False
    assert not_useful["runtime_answer_usefulness_score"] == 0.0


def test_api_normalization_rejects_missing_or_invalid_judge_values():
    invalid = evaluate_cli.normalize_api_response({
        "answer": "A response.",
        "answer_useful": "false",
        "usefulness_score": True,
    })
    assert invalid["runtime_answer_useful"] is None
    assert invalid["runtime_answer_usefulness_score"] is None

    out_of_range = evaluate_cli.normalize_api_response({
        "answer": "A response.",
        "answer_useful": None,
        "usefulness_score": 1.5,
    })
    assert out_of_range["runtime_answer_useful"] is None
    assert out_of_range["runtime_answer_usefulness_score"] is None


def test_runtime_usefulness_metrics_aggregate_only_valid_values():
    responses = {
        "positive": {
            "answer": "A useful answer.",
            "runtime_answer_useful": True,
            "runtime_answer_usefulness_score": 0.8,
        },
        "negative": {
            "answer": "An unhelpful answer.",
            "runtime_answer_useful": False,
            "runtime_answer_usefulness_score": 0.2,
        },
        "missing": {
            "answer": "A response without judge telemetry.",
            "runtime_answer_useful": None,
            "runtime_answer_usefulness_score": None,
        },
    }
    evaluator = KnowledgeMeshEvaluator(
        pipeline=lambda question: responses[question],
        thresholds={},
    )
    summary = evaluator.evaluate([
        EvaluationCase(id="positive", question="positive"),
        EvaluationCase(id="negative", question="negative"),
        EvaluationCase(id="missing", question="missing"),
    ])

    assert summary.metrics["runtime_answer_usefulness_rate"] == 0.5
    assert summary.metrics_n["runtime_answer_usefulness_rate"] == 2
    assert summary.metrics["runtime_answer_usefulness_score"] == 0.5
    assert summary.metrics_n["runtime_answer_usefulness_score"] == 2

    case_metrics = {
        case["case_id"]: case["metrics"]
        for case in summary.cases
    }
    assert case_metrics["missing"]["runtime_answer_usefulness_rate"] is None
    assert case_metrics["missing"]["runtime_answer_usefulness_score"] is None



def test_api_normalization_separates_cited_and_supported_citations():
    response = {
        "answer": "A mixed-support answer.",
        "claims": [
            {
                "claim": "The first claim is supported.",
                "citations": ["chunk_1"],
            },
            {
                "claim": "The second claim is unsupported.",
                "citations": ["chunk_2"],
            },
        ],
        "grounding_scores": [
            {
                "claim": "The first claim is supported.",
                "supported": True,
                "citations": ["chunk_1"],
            },
            {
                "claim": "The second claim is unsupported.",
                "supported": False,
                "citations": ["chunk_2"],
            },
        ],
        "citation_valid": True,
        "is_grounded": False,
        "validation_passed": False,
    }

    normalized = evaluate_cli.normalize_api_response(response)

    assert normalized["cited_ids"] == ["chunk_1", "chunk_2"]
    assert normalized["supported_citation_ids"] == ["chunk_1"]
    assert normalized["total_claims"] == 2
    assert normalized["unsupported_claims"] == 1
    assert normalized["final_gate_passed"] is False
    assert normalized["raw"] == response

def test_normalization_detects_empty_private_retrieval_from_trace():
    normalized = evaluate_cli.normalize_api_response({
        "answer": "No matching evidence was found.",
        "private_sources": [],
        "trace": [
            {
                "step": "private_retrieval",
                "status": "empty",
                "details": {
                    "candidate_count": 0,
                    "selected_count": 0,
                },
            }
        ],
    })

    assert normalized["private_retrieval_attempted"] is True
    assert normalized["used_private"] is False


def test_normalization_distinguishes_web_attempt_from_usable_sources():
    normalized = evaluate_cli.normalize_api_response({
        "answer": "No usable web evidence was found.",
        "web_sources": [],
        "web_search_used": False,
        "trace": [
            {
                "step": "web_search",
                "status": "empty",
                "details": {
                    "attempted": True,
                    "used": False,
                    "source_count": 0,
                },
            }
        ],
    })

    assert normalized["web_search_node_executed"] is True
    assert normalized["web_search_attempted"] is True
    assert normalized["used_web"] is False


def test_normalization_marks_empty_query_as_node_execution_not_search():
    normalized = evaluate_cli.normalize_api_response({
        "answer": "No web query was issued.",
        "web_sources": [],
        "trace": [
            {
                "step": "web_search",
                "status": "empty_query",
                "details": {
                    "attempted": True,
                    "used": False,
                    "source_count": 0,
                },
            }
        ],
    })

    assert normalized["web_search_node_executed"] is True
    assert normalized["web_search_attempted"] is False
    assert normalized["used_web"] is False


def test_normalization_does_not_infer_execution_from_retrieval_required_flag():
    normalized = evaluate_cli.normalize_api_response({
        "answer": "Answer generated.",
        "retrieval_used": True,
        "web_search_used": False,
        "private_sources": [],
        "web_sources": [],
    })

    # retrieval_used may indicate a requirement rather than actual execution.
    assert normalized["private_retrieval_attempted"] is None
    assert normalized["web_search_node_executed"] is None
    assert normalized["web_search_attempted"] is None


def test_normalization_reads_nested_trajectory_trace():
    normalized = evaluate_cli.normalize_api_response({
        "answer": "No matching evidence was found.",
        "private_sources": [],
        "trajectory": {
            "trace": [
                {
                    "step": "private_retrieval",
                    "status": "empty",
                    "details": {"selected_count": 0},
                }
            ]
        },
    })

    assert normalized["private_retrieval_attempted"] is True
    assert normalized["used_private"] is False


def test_normalization_skips_empty_top_level_trace():
    normalized = evaluate_cli.normalize_api_response({
        "answer": "No usable web evidence was found.",
        "web_sources": [],
        "trace": [],
        "evaluation_trace": [],
        "trajectory": {
            "trace": [
                {
                    "step": "web_search",
                    "status": "empty",
                    "details": {
                        "attempted": True,
                        "used": False,
                        "source_count": 0,
                    },
                }
            ]
        },
    })

    assert normalized["web_search_node_executed"] is True
    assert normalized["web_search_attempted"] is True
    assert normalized["used_web"] is False
