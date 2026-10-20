from app.evaluation.trajectory import evaluate_trajectory


def _blocked_state():
    return {
        "retrieval_required": True,
        "evaluation_trace": [
            {
                "step": "planner",
                "status": "success",
                "details": {},
            },
            {
                "step": "private_retrieval",
                "status": "empty",
                "details": {"selected_count": 0},
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


def test_no_approved_evidence_has_distinct_failure_category():
    result = evaluate_trajectory(_blocked_state())

    assert result["trajectory_status"] == "failed"
    assert "NO_APPROVED_EVIDENCE" in result["failure_taxonomy"]
    assert "GENERATION_FAILURE" not in result["failure_taxonomy"]


def test_generation_error_remains_generation_failure():
    state = _blocked_state()
    state["evaluation_trace"][-1] = {
        "step": "responder",
        "status": "failed",
        "details": {"error_type": "ProviderError"},
    }

    result = evaluate_trajectory(state)

    assert result["trajectory_status"] == "failed"
    assert "GENERATION_FAILURE" in result["failure_taxonomy"]
    assert "NO_APPROVED_EVIDENCE" not in result["failure_taxonomy"]


def test_new_category_is_accepted_by_persistence_contract():
    from app.evaluation.contracts import FAILURE_CATEGORIES

    assert "NO_APPROVED_EVIDENCE" in FAILURE_CATEGORIES


def test_no_approved_evidence_survives_persistence_round_trip(tmp_path):
    import json
    import sqlite3

    from app.evaluation.store import record_trajectory

    result = evaluate_trajectory(_blocked_state())
    database = tmp_path / "trajectory_test.db"

    record_trajectory(database, result)

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT trajectory_status, failure_taxonomy_json "
            "FROM evaluated_trajectories"
        ).fetchone()

    assert row is not None
    assert row[0] == "failed"
    assert "NO_APPROVED_EVIDENCE" in json.loads(row[1])
    assert "GENERATION_FAILURE" not in json.loads(row[1])
