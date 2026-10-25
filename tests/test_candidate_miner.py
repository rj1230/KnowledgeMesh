import json

import pytest

from evals.mine_candidates import candidate_from_record, mine_candidates


def _record(status="failed", failures=None, **extra):
    trajectory = {
        "trajectory_status": status,
        "failure_taxonomy": (
            failures if failures is not None else ["GROUNDING_FAILURE"]
        ),
        "evaluation_trace": [
            {"step": "grounding_critic", "status": "failed",
             "details": {"question": "PRIVATE QUESTION", "text": "PRIVATE TEXT"}}
        ],
        "metrics": {"unsupported_claim_rate": 0.5},
    }
    trajectory.update(extra)
    return {"trajectory": trajectory, "question": "PRIVATE QUESTION"}


def test_failed_trajectory_becomes_review_candidate():
    candidate = candidate_from_record(_record())

    assert candidate is not None
    assert candidate["trajectory_status"] == "failed"
    assert candidate["failure_taxonomy"] == ["GROUNDING_FAILURE"]
    assert candidate["review_status"] == "needs_review"
    assert candidate["gold_answer"] is None
    assert candidate["promotion_status"] == "not_promoted"


def test_recovered_trajectory_is_a_candidate():
    candidate = candidate_from_record(_record(status="recovered"))

    assert candidate is not None
    assert candidate["trajectory_status"] == "recovered"


def test_success_without_failure_is_skipped():
    assert candidate_from_record(
        _record(status="success", failures=[])
    ) is None


def test_raw_question_and_event_details_are_not_exported():
    candidate = candidate_from_record(_record())
    serialized = json.dumps(candidate)

    assert "PRIVATE QUESTION" not in serialized
    assert "PRIVATE TEXT" not in serialized
    assert "details" not in serialized


def test_unknown_failure_category_is_rejected():
    with pytest.raises(ValueError, match="unrecognized failure categories"):
        candidate_from_record(_record(failures=["MADE_UP_FAILURE"]))


def test_deduplicates_equivalent_candidates_and_writes_jsonl(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "candidates.jsonl"
    record = _record()

    source.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    result = mine_candidates(source, output)
    candidates = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert result["input_records"] == 2
    assert result["unique_candidates"] == 1
    assert len(candidates) == 1


def test_malformed_input_fails_without_replacing_existing_output(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "candidates.jsonl"
    output.write_text("existing output\n", encoding="utf-8")
    source.write_text('{"trajectory":\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        mine_candidates(source, output)

    assert output.read_text(encoding="utf-8") == "existing output\n"

def test_real_evaluator_recovery_becomes_candidate():
    from app.evaluation.trajectory import evaluate_trajectory

    state = {
        "retrieval_required": True,
        "evaluation_trace": [
            {"step": "planner", "status": "technical"},
            {
                "step": "private_retrieval",
                "status": "empty",
                "details": {"selected_count": 0},
            },
            {
                "step": "query_rewriter",
                "status": "rewritten",
                "details": {"query_changed": True},
            },
            {
                "step": "private_retrieval",
                "status": "success",
                "details": {"selected_count": 4},
            },
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    trajectory = evaluate_trajectory(state)
    candidate = candidate_from_record({
        "trajectory": trajectory,
        "question": "PRIVATE SYNTHETIC QUESTION",
    })

    assert trajectory["trajectory_status"] == "recovered"
    assert "RETRIEVAL_FAILURE" in trajectory["failure_taxonomy"]
    assert candidate is not None
    assert candidate["trajectory_status"] == "recovered"
    assert candidate["failure_taxonomy"] == ["RETRIEVAL_FAILURE"]
    assert candidate["review_status"] == "needs_review"
    assert candidate["gold_answer"] is None
    assert candidate["promotion_status"] == "not_promoted"

    serialized = json.dumps(candidate)
    assert "PRIVATE SYNTHETIC QUESTION" not in serialized
    assert "selected_count" not in serialized


def test_real_evaluator_clean_conversation_is_not_candidate():
    from app.evaluation.trajectory import evaluate_trajectory

    state = {
        "retrieval_required": False,
        "evaluation_trace": [
            {"step": "planner", "status": "conversational"},
        ],
        "citation_valid": True,
        "grounding_valid": True,
        "answer_useful": True,
        "generation_failed": False,
    }

    trajectory = evaluate_trajectory(state)

    assert trajectory["trajectory_status"] == "success"
    assert trajectory["failure_taxonomy"] == []
    assert candidate_from_record({"trajectory": trajectory}) is None

def test_candidate_miner_redacts_untrusted_step_and_status():
    import json

    from evals.mine_candidates import candidate_from_record

    secret_step = "customer@example.com confidential query"
    secret_status = "private answer: secret-token-123"

    candidate = candidate_from_record({
        "trajectory_status": "failed",
        "failure_taxonomy": ["GROUNDING_FAILURE"],
        "evaluation_trace": [
            {"step": secret_step, "status": secret_status,
             "details": {"query": "sensitive user question"}},
        ],
        "metrics": {},
        "question": "sensitive user question",
        "answer": "confidential answer",
    })

    assert candidate is not None
    assert candidate["event_summary"] == [
        {"step": "unknown", "status": "unknown"}
    ]

    serialized = json.dumps(candidate)
    for secret in (
        secret_step,
        secret_status,
        "secret-token-123",
        "sensitive user question",
        "confidential answer",
    ):
        assert secret not in serialized


def test_candidate_miner_preserves_known_canonical_event_labels():
    from evals.mine_candidates import candidate_from_record

    candidate = candidate_from_record({
        "trajectory_status": "recovered",
        "failure_taxonomy": ["RETRIEVAL_FAILURE"],
        "evaluation_trace": [
            {"step": "private_retrieval", "status": "empty",
             "details": {"query": "must not appear"}},
            {"step": "query_rewriter", "status": "rewritten"},
        ],
        "metrics": {},
    })

    assert candidate is not None
    assert candidate["event_summary"] == [
        {"step": "private_retrieval", "status": "empty"},
        {"step": "query_rewriter", "status": "rewritten"},
    ]

@pytest.mark.parametrize(
    "value",
    [True, float("nan"), float("inf"), float("-inf"), -0.01, 1.01, "0.5"],
)
def test_candidate_miner_rejects_invalid_rate_metrics(value):
    record = _record()
    record["trajectory"]["metrics"]["unsupported_claim_rate"] = value

    with pytest.raises(ValueError, match="unsupported_claim_rate"):
        candidate_from_record(record)


def test_candidate_miner_rejects_non_string_trajectory_status():
    record = _record()
    record["trajectory"]["trajectory_status"] = {"unexpected": "object"}

    with pytest.raises(ValueError, match="trajectory_status must be a string"):
        candidate_from_record(record)


def test_candidate_miner_accepts_boundary_rate_metrics():
    record = _record()
    record["trajectory"]["metrics"] = {
        "retrieval_success_rate": 0.0,
        "unsupported_claim_rate": 1.0,
    }

    candidate = candidate_from_record(record)

    assert candidate is not None
    assert candidate["metrics"] == {
        "retrieval_success_rate": 0.0,
        "unsupported_claim_rate": 1.0,
    }


def test_candidate_miner_rejects_oversized_integer_metric():
    record = _record()
    record["trajectory"]["metrics"] = {
        "retrieval_success_rate": 10**10000
    }

    with pytest.raises(
        ValueError,
        match=r"finite number in \[0, 1\]",
    ):
        candidate_from_record(record)

def test_duplicate_candidates_aggregate_metrics_deterministically(tmp_path):
    records = []

    first = _record()
    first["trajectory"]["metrics"] = {
        "retrieval_success_rate": 0.0,
        "unsupported_claim_rate": 0.2,
        "secret_metric": 0.9,
    }
    first["question"] = "PRIVATE QUESTION ONE"
    records.append(first)

    second = _record()
    second["trajectory"]["metrics"] = {
        "retrieval_success_rate": 0.5,
        "unsupported_claim_rate": 0.8,
    }
    second["question"] = "PRIVATE QUESTION TWO"
    records.append(second)

    third = _record()
    third["trajectory"]["metrics"] = {
        "retrieval_success_rate": 1.0,
    }
    third["question"] = "PRIVATE QUESTION THREE"
    records.append(third)

    outputs = []
    for index, ordered_records in enumerate((records, list(reversed(records)))):
        source = tmp_path / f"input_{index}.jsonl"
        output = tmp_path / f"output_{index}.jsonl"
        source.write_text(
            "".join(json.dumps(record) + "\n" for record in ordered_records),
            encoding="utf-8",
        )

        result = mine_candidates(source, output)
        assert result["input_records"] == 3
        assert result["unique_candidates"] == 1
        outputs.append(output.read_text(encoding="utf-8"))

    # Reordering equivalent input records must not change the exported result.
    assert outputs[0] == outputs[1]

    candidate = json.loads(outputs[0])
    assert candidate["occurrence_count"] == 3
    assert candidate["metrics"] == {
        "retrieval_success_rate": 0.5,
        "unsupported_claim_rate": 0.5,
    }
    assert candidate["metric_observation_counts"] == {
        "retrieval_success_rate": 3,
        "unsupported_claim_rate": 2,
    }
    assert candidate["review_status"] == "needs_review"
    assert candidate["gold_answer"] is None
    assert candidate["promotion_status"] == "not_promoted"

    serialized = outputs[0]
    for private_value in (
        "PRIVATE QUESTION ONE",
        "PRIVATE QUESTION TWO",
        "PRIVATE QUESTION THREE",
        "secret_metric",
    ):
        assert private_value not in serialized


def test_duplicate_candidates_count_occurrences_without_metrics(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "candidates.jsonl"
    record = _record()
    record["trajectory"]["metrics"] = {}

    source.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    result = mine_candidates(source, output)
    candidate = json.loads(output.read_text(encoding="utf-8"))

    assert result["unique_candidates"] == 1
    assert candidate["occurrence_count"] == 2
    assert candidate["metrics"] == {}
    assert candidate["metric_observation_counts"] == {}
