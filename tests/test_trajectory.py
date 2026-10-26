from app.evaluation.trajectory import evaluate_trajectory


def test_conversational_query():
    state = {
        "retrieval_required": False,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "conversational",
            }
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["required_hops"] == 0
    assert result["hop_count"] == 0
    assert result["trajectory_status"] == "success"
    assert result["failure_taxonomy"] == []


def test_normal_technical_single_hop():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 5,
                },
            },
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["required_hops"] == 1
    assert result["hop_count"] == 1
    assert result["retrieval_attempts"] == 1
    assert result["trajectory_status"] == "success"


def test_empty_retrieval_rewrite_then_success():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "empty",
                "details": {
                    "selected_count": 0,
                },
            },
            {
                "step": "query_rewriter",
                "status": "rewritten",
                "details": {
                    "query_changed": True,
                },
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 4,
                },
            },
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["required_hops"] == 1
    assert result["hop_count"] == 1
    assert result["retrieval_attempts"] == 2
    assert result["trajectory_status"] == "recovered"
    assert "RETRIEVAL_FAILURE" in result["failure_taxonomy"]
    assert "QUERY_REWRITE" in result["recovery_events"]


def test_private_retrieval_then_web_escalation():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 2,
                },
            },
            {
                "step": "web_search",
                "status": "recovered",
                "details": {
                    "attempted": True,
                    "used": True,
                    "source_count": 5,
                },
            },
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["hop_count"] == 1
    assert result["trajectory_status"] == "success"
    assert "WEB_ESCALATION" in result["recovery_events"]


def test_graded_private_context_rewrite_then_success():
    state = {
        "retrieval_required": True,
        "context_quality": "insufficient",
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 5,
                },
            },
            {
                "step": "grader",
                "status": "success",
                "details": {
                    "document_count": 5,
                    "generation_count": 0,
                    "content_count": 5,
                    "context_quality": "empty",
                },
            },
            {
                "step": "context_evaluator",
                "status": "insufficient",
                "details": {
                    "should_search_web": True,
                    "generation_documents": 0,
                },
            },
            {
                "step": "query_rewriter",
                "status": "rewritten",
                "details": {
                    "query_changed": True,
                },
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 4,
                },
            },
            {
                "step": "grader",
                "status": "success",
                "details": {
                    "document_count": 4,
                    "generation_count": 3,
                    "content_count": 4,
                    "context_quality": "strong",
                },
            },
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["required_hops"] == 1
    assert result["hop_count"] == 2
    assert result["retrieval_attempts"] == 2
    assert result["trajectory_status"] == "recovered"
    assert "EVIDENCE_SELECTION_FAILURE" in result["failure_taxonomy"]
    assert "QUERY_REWRITE" in result["recovery_events"]


def test_citation_failure_then_revision_success():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 5,
                },
            },
            {
                "step": "citation_check",
                "status": "failed",
                "details": {
                    "uncited_claims": 2,
                },
            },
            {
                "step": "revision",
                "status": "requested",
            },
            {
                "step": "citation_check",
                "status": "passed",
            },
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["trajectory_status"] == "recovered"
    assert "CITATION_FAILURE" in result["failure_taxonomy"]
    assert "ANSWER_REVISION" in result["recovery_events"]
    assert result["metrics"]["recovery_after_citation_failure"] == 1.0


def test_grounding_failure_then_revision_success():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 5,
                },
            },
            {
                "step": "grounding_critic",
                "status": "failed",
                "details": {
                    "unsupported_atomic_count": 2,
                },
            },
            {
                "step": "revision",
                "status": "requested",
            },
            {
                "step": "grounding_critic",
                "status": "passed",
            },
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["trajectory_status"] == "recovered"
    assert "GROUNDING_FAILURE" in result["failure_taxonomy"]
    assert "ANSWER_REVISION" in result["recovery_events"]
    assert result["metrics"]["recovery_after_grounding_failure"] == 1.0


def test_unrecovered_grounding_failure():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 5,
                },
            },
            {
                "step": "grounding_critic",
                "status": "failed",
                "details": {
                    "unsupported_atomic_count": 3,
                },
            },
        ],
        "citation_valid": True,
        "grounding_valid": False,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["trajectory_status"] == "failed"
    assert "GROUNDING_FAILURE" in result["failure_taxonomy"]
    assert result["metrics"]["recovery_after_grounding_failure"] == 0.0


def test_unrecovered_citation_failure():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 5,
                },
            },
            {
                "step": "citation_check",
                "status": "failed",
                "details": {
                    "uncited_claims": 2,
                },
            },
        ],
        "citation_valid": False,
        "grounding_valid": True,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["trajectory_status"] == "failed"
    assert "CITATION_FAILURE" in result["failure_taxonomy"]
    assert result["metrics"]["recovery_after_citation_failure"] == 0.0


def test_grounded_but_irrelevant_answer_is_not_success():
    state = {
        "retrieval_required": True,
        "original_query": (
            "What is KnowledgeMesh's failure handling policy for "
            "quantum database synchronization failures?"
        ),
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 3,
                },
            },
            {
                "step": "grounding_critic",
                "status": "passed",
                "details": {
                    "atomic_claim_count": 1,
                    "unsupported_atomic_count": 0,
                },
            },
            {
                "step": "answer_relevance",
                "status": "failed",
                "details": {
                    "answer_useful": False,
                    "usefulness_score": 0.08,
                    "reason": (
                        "The answer discusses quantum computing errors "
                        "but does not answer the KnowledgeMesh policy question."
                    ),
                },
            },
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_supported": True,
        "answer_useful": False,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["trajectory_status"] == "failed"
    assert "ANSWER_RELEVANCE_FAILURE" in result["failure_taxonomy"]


def test_web_search_failure_blocks_generation():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {
                    "selected_count": 3,
                },
            },
            {
                "step": "grader",
                "status": "success",
                "details": {
                    "document_count": 3,
                    "generation_count": 0,
                    "content_count": 3,
                    "context_quality": "empty",
                },
            },
            {
                "step": "context_evaluator",
                "status": "insufficient",
                "details": {
                    "should_search_web": True,
                    "generation_documents": 0,
                },
            },
            {
                "step": "web_search",
                "status": "failed",
                "details": {
                    "attempted": True,
                    "used": False,
                    "source_count": 0,
                    "error_type": "ConnectionError",
                },
            },
            {
                "step": "responder",
                "status": "blocked",
                "details": {
                    "generation_evidence_count": 0,
                    "reason": "no_approved_evidence",
                },
            },
        ],
        "citation_valid": None,
        "grounding_valid": None,
        "answer_useful": None,
        "generation_failed": True,
    }

    result = evaluate_trajectory(state)

    assert result["trajectory_status"] == "failed"
    assert "GENERATION_FAILURE" in result["failure_taxonomy"]
    assert "WEB_ESCALATION" in result["recovery_events"]
    assert result["hop_count"] == 1
    assert result["metrics"]["web_fallback_rate"] == 0.0


def test_prepare_revision_preserves_grounding_feedback_on_grounding_failure():
    from app.agents.graph import prepare_revision_node

    state = {
        "answer_useful": False,
        "is_grounded": False,
        "answer_supported": False,
        "grounding_feedback": ["Claim 1 is not supported by the supplied evidence."],
        "citation_feedback": "",
        "revision_prompt": "",
        "answer_revision_count": 0,
        "revision_count": 0,
        "max_revisions": 2,
        "evaluation_trace": [],
        "entailment_cache": [
            {
                "claim": "Retrieved passages are ranked before answering.",
                "evidence": "The system ranks retrieved passages before generating an answer.",
                "score": 0.90,
            }
        ],
    }

    result = prepare_revision_node(state)

    # Revision returns a partial state update; it should not overwrite the cache.
    assert state["entailment_cache"][0]["score"] == 0.90
    assert "entailment_cache" not in result
    assert result["revision_requested"] is True
    assert result["answer_revision_count"] == 1
    assert result["revision_count"] == 1
    assert result["grounding_feedback"] == [
        "Claim 1 is not supported by the supplied evidence."
    ]
    assert "ANSWER-RELEVANCE REVISION REQUIRED" not in result["revision_prompt"]
    assert "grounding_failure" in result["evaluation_trace"][-1]["details"]["trigger"]


def test_prepare_revision_replaces_stale_grounding_feedback_on_relevance_failure():
    from app.agents.graph import prepare_revision_node

    state = {
        "answer_useful": False,
        "is_grounded": True,
        "answer_supported": True,
        "grounding_feedback": [
            "All substantive claims passed strict grounding validation."
        ],
        "citation_feedback": "",
        "revision_prompt": "",
        "answer_revision_count": 1,
        "revision_count": 1,
        "max_revisions": 2,
        "evaluation_trace": [],
    }

    result = prepare_revision_node(state)

    assert result["revision_requested"] is True
    assert result["answer_revision_count"] == 2
    assert result["revision_count"] == 2
    assert result["grounding_feedback"] == []
    assert "ANSWER-RELEVANCE REVISION REQUIRED" in result["revision_prompt"]
    assert (
        "answer_relevance_failure"
        in result["evaluation_trace"][-1]["details"]["trigger"]
    )


def test_prepare_revision_relevance_failure_targets_question_not_related_evidence():
    from app.agents.graph import prepare_revision_node

    state = {
        "answer_useful": False,
        "is_grounded": True,
        "answer_supported": True,
        "grounding_feedback": [],
        "citation_feedback": "",
        "revision_prompt": "",
        "answer_revision_count": 1,
        "revision_count": 1,
        "max_revisions": 2,
        "evaluation_trace": [],
    }

    result = prepare_revision_node(state)

    prompt = result["revision_prompt"]

    assert result["revision_requested"] is True
    assert result["answer_revision_count"] == 2
    assert result["revision_count"] == 2
    assert "directly answer the user's question" in prompt
    assert "Do not substitute a related model" in prompt
    assert "specific one requested by the user" in prompt


def test_prepare_revision_preserves_citation_feedback_with_grounding_prompt():
    from app.agents.graph import prepare_revision_node

    state = {
        "answer_useful": True,
        "is_grounded": False,
        "answer_supported": False,
        "grounding_feedback": ["Claim X is unsupported by the evidence."],
        "citation_feedback": "Citation [2] does not support the claim.",
        "revision_prompt": "Grounding validation failed: remove unsupported atomic claims.",
        "answer_revision_count": 0,
        "revision_count": 0,
        "max_revisions": 2,
        "evaluation_trace": [],
    }

    result = prepare_revision_node(state)
    prompt = result["revision_prompt"]

    assert "Citation [2] does not support the claim." in prompt
    assert prompt.count("Citation [2] does not support the claim.") == 1


def test_prepare_revision_appends_grounding_feedback_to_generic_prompt():
    from app.agents.graph import prepare_revision_node

    state = {
        "answer_useful": True,
        "is_grounded": False,
        "answer_supported": False,
        "grounding_feedback": ["Claim X is unsupported by the evidence."],
        "citation_feedback": "",
        "revision_prompt": (
            "Revise the previous answer using only the supplied evidence. "
            "Remove unsupported claims and ensure every factual statement "
            "has a valid citation."
        ),
        "answer_revision_count": 0,
        "revision_count": 0,
        "max_revisions": 2,
        "evaluation_trace": [],
    }

    result = prepare_revision_node(state)

    assert "Validation feedback:" in result["revision_prompt"]
    assert "Claim X is unsupported by the evidence." in result["revision_prompt"]


def test_prepare_revision_does_not_duplicate_grounding_feedback():
    from app.agents.graph import prepare_revision_node

    feedback = "Claim X is unsupported by the evidence."
    state = {
        "answer_useful": True,
        "is_grounded": False,
        "answer_supported": False,
        "grounding_feedback": [feedback],
        "citation_feedback": "",
        "revision_prompt": f"Grounding feedback: {feedback}",
        "answer_revision_count": 0,
        "revision_count": 0,
        "max_revisions": 2,
        "evaluation_trace": [],
    }

    result = prepare_revision_node(state)

    assert result["revision_prompt"].count(feedback) == 1


def test_prepare_revision_appends_missing_detail_to_detailed_grounding_prompt():
    from app.agents.graph import prepare_revision_node

    missing_detail = "Claim Y is unsupported by the retrieved evidence."
    state = {
        "answer_useful": True,
        "is_grounded": False,
        "answer_supported": False,
        "grounding_feedback": [missing_detail],
        "citation_feedback": "",
        "revision_prompt": (
            "Grounding validation failed. Remove each unsupported atomic claim "
            "and use only claims supported by the evidence."
        ),
        "answer_revision_count": 0,
        "revision_count": 0,
        "max_revisions": 2,
        "evaluation_trace": [],
    }

    result = prepare_revision_node(state)
    prompt = result["revision_prompt"]

    assert missing_detail in prompt
    assert prompt.count(missing_detail) == 1
def test_blocked_revision_is_classified_as_revision_failure():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "technical",
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {"selected_count": 3},
            },
            {
                "step": "revision",
                "status": "blocked",
                "details": {
                    "reason": "revision_budget_exhausted",
                    "revision_count": 2,
                    "max_answer_revisions": 2,
                },
            },
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_supported": True,
        "answer_useful": False,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)

    assert result["trajectory_status"] == "failed"
    assert "REVISION_FAILURE" in result["failure_taxonomy"]



def test_unsupported_claim_metric_ignores_malformed_counts():
    invalid_values = [
        "unknown",
        True,
        -1,
        float("inf"),
        float("nan"),
    ]

    for invalid_value in invalid_values:
        state = {
            "retrieval_required": True,
            "evaluation_trace": [
                {"step": "planner", "status": "technical"},
                {
                    "step": "private_retrieval",
                    "status": "success",
                    "details": {"selected_count": 1},
                },
                {
                    "step": "grounding_critic",
                    "status": "passed",
                    "details": {
                        "atomic_claim_count": invalid_value,
                        "unsupported_atomic_count": 1,
                    },
                },
                {
                    "step": "grounding_critic",
                    "status": "passed",
                    "details": {
                        "atomic_claim_count": 2,
                        "unsupported_atomic_count": invalid_value,
                    },
                },
            ],
            "citation_valid": True,
            "grounding_valid": True,
            "answer_supported": True,
            "answer_useful": True,
            "generation_failed": False,
        }

        result = evaluate_trajectory(state)
        assert result["metrics"]["unsupported_claim_rate"] == 0.0
        assert result["metrics"]["unsupported_claim_rate_available"] is False


def test_unsupported_claim_metric_measured_zero_is_available():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {"step": "grounding_critic", "status": "passed", "details": {
                "atomic_claim_count": 3,
                "unsupported_atomic_count": 0,
            }},
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_supported": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)
    assert result["metrics"]["unsupported_claim_rate"] == 0.0
    assert result["metrics"]["unsupported_claim_rate_available"] is True


def test_unsupported_claim_metric_caps_rate_at_one():
    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {"step": "planner", "status": "technical"},
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {"selected_count": 1},
            },
            {
                "step": "grounding_critic",
                "status": "passed",
                "details": {
                    "atomic_claim_count": 2,
                    "unsupported_atomic_count": 3,
                },
            },
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_supported": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    result = evaluate_trajectory(state)
    assert result["metrics"]["unsupported_claim_rate"] == 1.0
    assert result["metrics"]["unsupported_claim_rate_available"] is True
