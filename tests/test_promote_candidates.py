import json
from pathlib import Path

import pytest

from evals.promote_candidates import promote_candidates


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def make_candidate(candidate_id="cand_aaaaaaaaaaaaaaaa"):
    suffix = candidate_id.removeprefix("cand_")
    fingerprint = suffix + ("0" * (64 - len(suffix)))
    assert len(suffix) == 16
    assert all(char in "0123456789abcdef" for char in suffix)
    return {
        "candidate_id": candidate_id,
        "fingerprint": fingerprint,
        "trajectory_status": "failed",
        "failure_taxonomy": ["RETRIEVAL_FAILURE"],
        "review_status": "needs_review",
        "event_summary": [],
        "metrics": {},
    }


def make_review(candidate_id="cand_aaaaaaaaaaaaaaaa", **overrides):
    review = {
        "candidate_id": candidate_id,
        "review_status": "approved",
        "id": "traj001",
        "question": "Which two memory types does RAG combine?",
        "expected_answer_points": [
            "Parametric memory",
            "Non-parametric memory",
        ],
        "expected_route": "private",
        "relevant_chunks": [
            {"document_id": "doc123", "chunk_id": 0},
        ],
    }
    review.update(overrides)
    return review


def run_promotion(
    tmp_path, candidates, reviews, existing=None, output=None, inventory=True
):
    candidates_path = tmp_path / "candidates.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    output_path = output or (tmp_path / "promoted.json")

    write_jsonl(candidates_path, candidates)
    write_jsonl(reviews_path, reviews)

    existing_paths = []
    for index, records in enumerate(existing or []):
        path = tmp_path / f"existing_{index}.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        existing_paths.append(path)

    inventory_path = None
    if inventory:
        inventory_path = tmp_path / "true_chunk_inventory.csv"
        inventory_path.write_text(
            '"DocumentId","Filename","SourceType","ChunkId",'
            '"Characters","Text"\n'
            '"doc123","doc123.pdf","true","0","0",""\n',
            encoding="utf-8",
        )

    result = promote_candidates(
        candidates_path,
        reviews_path,
        output_path,
        existing_paths,
        inventory_path=inventory_path,
    )
    return result, output_path


def test_approved_candidate_promotes_to_evaluator_compatible_dataset(tmp_path):
    result, output = run_promotion(
        tmp_path,
        [make_candidate()],
        [make_review()],
    )

    assert result["promoted_cases"] == 1
    dataset = json.loads(output.read_text(encoding="utf-8"))
    assert len(dataset) == 1

    case = dataset[0]
    assert case["id"] == "traj001"
    assert case["expected_route"] == "private"
    assert case["expected_answer_points"]
    assert case["relevant_chunks"] == [
        {"document_id": "doc123", "chunk_id": 0}
    ]
    assert case["metadata"]["candidate_id"] == "cand_aaaaaaaaaaaaaaaa"
    assert case["metadata"]["failure_taxonomy"] == ["RETRIEVAL_FAILURE"]
    assert "gold_answer" not in case
    assert "question" not in case["metadata"]


def test_unknown_private_chunk_is_rejected_without_output(tmp_path):
    review = make_review(
        relevant_chunks=[{"document_id": "unknown-document", "chunk_id": 0}]
    )
    with pytest.raises(ValueError, match="absent from inventory"):
        run_promotion(tmp_path, [make_candidate()], [review])
    assert not (tmp_path / "promoted.json").exists()


def test_private_gold_requires_inventory_without_output(tmp_path):
    with pytest.raises(ValueError, match="require --inventory"):
        run_promotion(
            tmp_path,
            [make_candidate()],
            [make_review()],
            inventory=False,
        )
    assert not (tmp_path / "promoted.json").exists()


def test_unapproved_candidate_is_rejected_without_output(tmp_path):
    review = make_review(review_status="needs_review")
    output = tmp_path / "promoted.json"

    with pytest.raises(ValueError, match="explicitly be 'approved'"):
        run_promotion(tmp_path, [make_candidate()], [review], output=output)

    assert not output.exists()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"expected_answer_points": []}, "expected_answer_points"),
        ({"question": "   "}, "question"),
        ({"expected_route": "unknown"}, "expected_route"),
        ({"relevant_chunks": []}, "requires relevant_chunks"),
        (
            {"relevant_chunks": [{"document_id": "doc123", "chunk_id": True}]},
            "non-negative integer",
        ),
        (
            {"relevant_chunks": [{"document_id": "", "chunk_id": 0}]},
            "document_id",
        ),
    ],
)
def test_invalid_private_review_is_rejected(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        run_promotion(
            tmp_path,
            [make_candidate()],
            [make_review(**overrides)],
        )

    assert not (tmp_path / "promoted.json").exists()


@pytest.mark.parametrize(
    "route, extra, message",
    [
        ("web", {}, "requires web sources or domains"),
        ("both", {}, "requires web sources or domains"),
        (
            "both",
            {"expected_domains": ["example.org"]},
            None,
        ),
    ],
)
def test_web_and_both_routes_require_matching_evidence(
    tmp_path, route, extra, message
):
    review = make_review(expected_route=route, **extra)
    if message:
        with pytest.raises(ValueError, match=message):
            run_promotion(tmp_path, [make_candidate()], [review])
        assert not (tmp_path / "promoted.json").exists()
    else:
        result, output = run_promotion(
            tmp_path, [make_candidate()], [review]
        )
        assert result["promoted_cases"] == 1
        assert json.loads(output.read_text(encoding="utf-8"))[0][
            "expected_route"
        ] == "both"


def test_abstention_can_promote_without_retrieval_gold(tmp_path):
    review = make_review(
        expected_route="abstain",
        relevant_chunks=[],
        expected_answer_points=["The corpus does not contain this information."],
    )
    result, output = run_promotion(
        tmp_path, [make_candidate()], [review]
    )

    assert result["promoted_cases"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))[0][
        "expected_route"
    ] == "abstain"


def test_duplicate_promoted_case_ids_are_rejected(tmp_path):
    first = make_review(id="same_id")
    second = make_review(
        id="same_id",
        candidate_id="cand_bbbbbbbbbbbbbbbb",
    )
    second_candidate = make_candidate("cand_bbbbbbbbbbbbbbbb")
    second_candidate["fingerprint"] = "b" * 64

    with pytest.raises(ValueError, match="duplicate promoted case ID"):
        run_promotion(
            tmp_path,
            [make_candidate(), second_candidate],
            [first, second],
        )

    assert not (tmp_path / "promoted.json").exists()


def test_existing_benchmark_id_collision_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="collide with an existing benchmark"):
        run_promotion(
            tmp_path,
            [make_candidate()],
            [make_review()],
            existing=[[{"id": "traj001"}]],
        )

    assert not (tmp_path / "promoted.json").exists()


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "promoted.json"
    original = '[{"preserve": true}]'
    output.write_text(original, encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_promotion(
            tmp_path,
            [make_candidate()],
            [make_review()],
            output=output,
        )

    assert output.read_text(encoding="utf-8") == original


def test_review_for_unknown_candidate_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown candidate"):
        run_promotion(
            tmp_path,
            [make_candidate()],
            [make_review(candidate_id="cand_" + "c" * 16)],
        )

    assert not (tmp_path / "promoted.json").exists()


def test_duplicate_reviews_for_candidate_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate review"):
        run_promotion(
            tmp_path,
            [make_candidate()],
            [make_review(), make_review(id="traj002")],
        )

    assert not (tmp_path / "promoted.json").exists()



@pytest.mark.parametrize(
    "taxonomy",
    [
        None,
        [],
        "RETRIEVAL_FAILURE",
        ["NOT_A_REAL_FAILURE"],
        ["RETRIEVAL_FAILURE", 123],
        ["RETRIEVAL_FAILURE", "RETRIEVAL_FAILURE"],
    ],
)
def test_invalid_candidate_taxonomy_is_rejected_without_output(
    tmp_path, taxonomy
):
    candidate = make_candidate()
    candidate["failure_taxonomy"] = taxonomy

    with pytest.raises(ValueError, match="failure_taxonomy"):
        run_promotion(
            tmp_path,
            [candidate],
            [make_review()],
        )

    assert not (tmp_path / "promoted.json").exists()


def test_valid_multiple_failure_categories_are_preserved(tmp_path):
    candidate = make_candidate()
    candidate["failure_taxonomy"] = [
        "RETRIEVAL_FAILURE",
        "GROUNDING_FAILURE",
    ]

    result, output = run_promotion(
        tmp_path,
        [candidate],
        [make_review()],
    )

    assert result["promoted_cases"] == 1
    dataset = json.loads(output.read_text(encoding="utf-8"))
    assert dataset[0]["metadata"]["failure_taxonomy"] == [
        "RETRIEVAL_FAILURE",
        "GROUNDING_FAILURE",
    ]


def test_promotion_cli_help_runs_directly():
    import subprocess
    import sys

    repository_root = Path(__file__).resolve().parents[1]
    script_path = repository_root / "evals" / "promote_candidates.py"

    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--candidates" in completed.stdout
    assert "--reviews" in completed.stdout
    assert "--output" in completed.stdout
    assert "--inventory" in completed.stdout
