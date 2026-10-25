from app.agents import graph as graph_module


def test_compiled_graph_runs_trajectory_evaluation_before_end(monkeypatch):
    evaluation_calls = []

    def fake_planner(state):
        return {
            "route": "conversation",
            "retrieval_required": False,
            "evaluation_trace": [
                {"step": "planner", "status": "conversational"}
            ],
        }

    def fake_responder(state):
        # Exercise the responder's terminal route without an LLM.
        return {
            "generation_failed": True,
            "final_answer": "",
        }

    def fake_evaluate_trajectory(state):
        evaluation_calls.append(dict(state))
        return {
            "trajectory_status": "failed",
            "hop_count": 0,
            "required_hops": 0,
            "failure_taxonomy": ["GENERATION_FAILURE"],
            "metrics": {"answer_useful": 0.0},
            "evaluation_trace": [
                {"step": "planner", "status": "conversational"}
            ],
        }

    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "generate_node", fake_responder)
    monkeypatch.setattr(
        graph_module, "evaluate_trajectory", fake_evaluate_trajectory
    )

    compiled_graph = graph_module.build_graph()

    result = compiled_graph.invoke(
        {
            "route": "conversation",
            "retrieval_required": False,
            "evaluation_trace": [],
        },
        config={"configurable": {"thread_id": "trajectory-regression-test"}},
    )

    assert len(evaluation_calls) == 1
    assert result["trajectory_status"] == "failed"
    assert result["hop_count"] == 0
    assert result["required_hops"] == 0
    assert result["failure_taxonomy"] == ["GENERATION_FAILURE"]
    assert result["trajectory_metrics"] == {"answer_useful": 0.0}
    assert result["evaluation_trace"] == [
        {"step": "planner", "status": "conversational"}
    ]

def test_compiled_graph_preserves_entailment_cache_across_revision(monkeypatch):
    grounding_inputs = []
    responder_calls = []

    cache_entry = {
        "claim": "Retrieved passages are ranked before answering.",
        "evidence": "The system ranks retrieved passages before generating an answer.",
        "score": 0.90,
    }

    def fake_planner(state):
        return {
            "route": "conversation",
            "retrieval_required": False,
        }

    def fake_responder(state):
        responder_calls.append(dict(state))
        return {
            "generation_failed": False,
            "final_answer": "A revised answer. [chunk_1]",
        }

    def fake_citation_check(state):
        return {
            "citation_valid": True,
            "citation_feedback": "",
        }

    def fake_grounding_critic(state):
        grounding_inputs.append(dict(state))
        pass_number = len(grounding_inputs)

        if pass_number == 1:
            return {
                "is_grounded": False,
                "answer_supported": False,
                "grounding_feedback": [
                    "A claim is not supported by the evidence."
                ],
                "entailment_cache": [cache_entry],
                "grounding_pass_count": 1,
            }

        return {
            "is_grounded": True,
            "answer_supported": True,
            "grounding_feedback": [],
            "grounding_pass_count": pass_number,
            "request_hhem_call_count": 4,
            "request_hhem_total_ms": 123.5,
            "entailment_cache_hits": 3,
            "entailment_cache_misses": 2,
            "cache_entries_received": 2,
            "cache_entries_returned": 4,
        }

    def fake_answer_relevance(state):
        return {"answer_useful": True}

    def fake_evaluate_trajectory(state):
        return {
            "trajectory_status": "passed",
            "hop_count": 1,
            "required_hops": 1,
            "failure_taxonomy": [],
            "metrics": {"answer_useful": 1.0},
            "evaluation_trace": state.get("evaluation_trace", []),
        }

    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "generate_node", fake_responder)
    monkeypatch.setattr(graph_module, "citation_check_node", fake_citation_check)
    monkeypatch.setattr(graph_module, "grounding_critic_node", fake_grounding_critic)
    monkeypatch.setattr(graph_module, "answer_relevance_node", fake_answer_relevance)
    monkeypatch.setattr(graph_module, "evaluate_trajectory", fake_evaluate_trajectory)

    compiled_graph = graph_module.build_graph()

    result = compiled_graph.invoke(
        {
            "route": "conversation",
            "retrieval_required": False,
            "final_answer": "An initial answer. [chunk_1]",
            "citation_valid": True,
            "is_grounded": False,
            "answer_supported": False,
            "answer_useful": False,
            "grounding_feedback": [
                "A claim is not supported by the evidence."
            ],
            "answer_revision_count": 0,
            "revision_count": 0,
            "max_revisions": 2,
            "evaluation_trace": [],
            "entailment_cache": [],
        },
        config={
            "configurable": {
                "thread_id": "entailment-cache-revision-regression"
            }
        },
    )

    assert len(grounding_inputs) == 2
    assert grounding_inputs[1].get("entailment_cache") == [cache_entry]
    assert len(responder_calls) == 2

    # Final graph state must retain the grounding node's observability fields.
    assert result["grounding_pass_count"] == 2
    assert result["request_hhem_call_count"] == 4
    assert result["request_hhem_total_ms"] == 123.5
    assert result["entailment_cache_hits"] == 3
    assert result["entailment_cache_misses"] == 2
    assert result["cache_entries_received"] == 2
    assert result["cache_entries_returned"] == 4
    assert result["trajectory_status"] == "passed"
    assert result["grounding_pass_count"] == 2

def test_citation_failure_abstains_when_revision_budget_is_exhausted():
    state = {
        "citation_valid": False,
        "answer_revision_count": graph_module.MAX_ANSWER_REVISIONS,
        "revision_count": graph_module.MAX_ANSWER_REVISIONS,
    }

    assert graph_module.route_after_citation_check(state) == "abstain"


def test_grounding_failure_abstains_when_revision_budget_is_exhausted():
    state = {
        "citation_valid": True,
        "is_grounded": False,
        "answer_supported": False,
        "answer_revision_count": graph_module.MAX_ANSWER_REVISIONS,
        "revision_count": graph_module.MAX_ANSWER_REVISIONS,
    }

    assert graph_module.route_after_grounding_critic(state) == "abstain"


def test_citation_failure_can_still_request_revision():
    state = {
        "citation_valid": False,
        "answer_revision_count": 0,
        "revision_count": 0,
    }

    assert graph_module.route_after_citation_check(state) == "revision"


def test_grounding_failure_can_still_request_revision():
    state = {
        "is_grounded": False,
        "answer_supported": False,
        "answer_revision_count": 0,
        "revision_count": 0,
    }

    assert graph_module.route_after_grounding_critic(state) == "revision"



def test_compiled_graph_fails_trajectory_when_revision_generation_fails(monkeypatch):
    responder_calls = []
    grounding_calls = []
    trajectory_inputs = []

    def fake_planner(state):
        return {
            "route": "conversation",
            "retrieval_required": False,
        }

    def fake_responder(state):
        responder_calls.append(dict(state))

        if len(responder_calls) == 1:
            return {
                "generation_failed": False,
                "final_answer": "Initial answer. [chunk_1]",
                "candidate_answer": "Initial answer. [chunk_1]",
            }

        # Simulate a provider failure during revision. The previous answer
        # remains present, but this generation attempt has failed.
        return {
            "generation_failed": True,
            "final_answer": state.get("final_answer", ""),
            "candidate_answer": state.get("final_answer", ""),
        }

    def fake_citation_check(state):
        return {
            "citation_valid": True,
            "citation_feedback": "",
        }

    def fake_grounding_critic(state):
        grounding_calls.append(dict(state))
        return {
            "is_grounded": False,
            "answer_supported": False,
            "grounding_feedback": ["The answer lacks sufficient evidence."],
        }

    def fake_evaluate_trajectory(state):
        trajectory_inputs.append(dict(state))
        assert state.get("generation_failed") is True
        return {
            "trajectory_status": "failed",
            "hop_count": 0,
            "required_hops": 1,
            "failure_taxonomy": ["GENERATION_FAILURE"],
            "metrics": {"answer_useful": 0.0},
            "evaluation_trace": state.get("evaluation_trace", []),
        }

    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "generate_node", fake_responder)
    monkeypatch.setattr(
        graph_module, "citation_check_node", fake_citation_check
    )
    monkeypatch.setattr(
        graph_module, "grounding_critic_node", fake_grounding_critic
    )
    monkeypatch.setattr(
        graph_module, "evaluate_trajectory", fake_evaluate_trajectory
    )

    compiled_graph = graph_module.build_graph()
    result = compiled_graph.invoke(
        {
            "route": "conversation",
            "retrieval_required": False,
            "original_query": "What does the evidence establish?",
            "final_answer": "",
            "citation_valid": True,
            "is_grounded": False,
            "answer_supported": False,
            "answer_useful": False,
            "answer_revision_count": 0,
            "revision_count": 0,
            "max_revisions": graph_module.MAX_ANSWER_REVISIONS,
            "grounding_feedback": [],
            "evaluation_trace": [],
        },
        config={
            "configurable": {
                "thread_id": "revision-generation-failure-regression"
            }
        },
    )

    assert len(responder_calls) == 2
    assert len(grounding_calls) == 1
    assert len(trajectory_inputs) == 1
    assert result["generation_failed"] is True
    assert result["trajectory_status"] == "failed"
    assert result["failure_taxonomy"] == ["GENERATION_FAILURE"]
