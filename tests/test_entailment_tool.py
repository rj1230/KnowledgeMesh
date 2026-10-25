import pytest

import app.tools.entailment_tool as entailment


@pytest.fixture(autouse=True)
def _clean_cache():
    """The score cache is process-global; isolate every test from it."""
    entailment.clear_hhem_cache()
    yield
    entailment.clear_hhem_cache()


def _fake_scorer(value=0.8):
    """Stand-in for _score_uncached(model, pairs) -> list[float]."""

    def fake(model, pairs):
        return [value] * len(pairs)

    return fake


def test_score_entailment_success_updates_profile(monkeypatch):
    monkeypatch.setattr(entailment, "_hhem_call_count", 0)
    monkeypatch.setattr(entailment, "_hhem_total_ms", 0.0)
    monkeypatch.setattr(entailment, "_get_model", lambda: object())
    monkeypatch.setattr(entailment, "_score_uncached", _fake_scorer(0.8))

    score = entailment.score_entailment("evidence", "claim")

    profile = entailment.get_hhem_profile()
    assert score == 0.8
    assert profile["call_count"] == 1
    assert profile["total_ms"] >= 0


def test_score_entailment_empty_input_returns_zero_without_model(monkeypatch):
    monkeypatch.setattr(entailment, "_hhem_call_count", 0)
    monkeypatch.setattr(entailment, "_hhem_total_ms", 0.0)

    model_calls = []

    def tracking_get_model():
        model_calls.append(1)
        return object()

    monkeypatch.setattr(entailment, "_get_model", tracking_get_model)

    assert entailment.score_entailment("", "claim") == 0.0
    assert entailment.score_entailment("evidence", "") == 0.0

    # Empty inputs never reach the model, so nothing is counted as scored.
    assert model_calls == []
    assert entailment.get_hhem_profile()["call_count"] == 0


def test_score_entailment_model_load_failure_returns_zero(monkeypatch):
    monkeypatch.setattr(entailment, "_hhem_call_count", 0)
    monkeypatch.setattr(entailment, "_hhem_total_ms", 0.0)
    monkeypatch.setattr(
        entailment,
        "_get_model",
        lambda: (_ for _ in ()).throw(RuntimeError("test failure")),
    )

    score = entailment.score_entailment("evidence", "claim")

    assert score == 0.0
    # The model never scored anything.
    assert entailment.get_hhem_profile()["call_count"] == 0


def test_scoring_failure_returns_zero_and_is_not_cached(monkeypatch):
    monkeypatch.setattr(entailment, "_get_model", lambda: object())

    def broken(model, pairs):
        raise RuntimeError("inference blew up")

    monkeypatch.setattr(entailment, "_score_uncached", broken)
    assert entailment.score_entailment("evidence", "claim") == 0.0

    # A failure must not poison the cache: the same pair scores normally
    # once the model works again.
    monkeypatch.setattr(entailment, "_score_uncached", _fake_scorer(0.8))
    assert entailment.score_entailment("evidence", "claim") == 0.8


def test_get_hhem_profile_returns_expected_fields():
    profile = entailment.get_hhem_profile()

    assert "call_count" in profile
    assert "total_ms" in profile


def test_cache_avoids_second_model_call(monkeypatch):
    monkeypatch.setattr(entailment, "_get_model", lambda: object())

    calls = []

    def counting(model, pairs):
        calls.append(len(pairs))
        return [0.8] * len(pairs)

    monkeypatch.setattr(entailment, "_score_uncached", counting)

    assert entailment.score_entailment("evidence", "claim") == 0.8
    assert entailment.score_entailment("evidence", "claim") == 0.8

    assert calls == [1]


def test_batch_scores_duplicates_once_and_keeps_order(monkeypatch):
    monkeypatch.setattr(entailment, "_get_model", lambda: object())

    seen = []

    def by_claim_number(model, pairs):
        seen.append([hypothesis for _, hypothesis in pairs])
        return [int(hypothesis[1:]) / 10 for _, hypothesis in pairs]

    monkeypatch.setattr(entailment, "_score_uncached", by_claim_number)

    scores = entailment.score_entailment_batch(
        [
            ("evidence", "c1"),
            ("evidence", "c2"),
            ("evidence", "c1"),  # duplicate of the first pair
            ("evidence", "c3"),
        ]
    )

    assert scores == pytest.approx([0.1, 0.2, 0.1, 0.3])
    # The duplicate was sent to the model only once.
    assert seen == [["c1", "c2", "c3"]]


def test_concurrent_profiles_are_isolated(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2, timeout=10)

    monkeypatch.setattr(entailment, "_get_model", lambda: object())

    def fake_score(model, pairs):
        barrier.wait()
        return [0.8] * len(pairs)

    monkeypatch.setattr(entailment, "_score_uncached", fake_score)

    def run_profile(index):
        token, profile = entailment.start_hhem_profile()
        try:
            # Different premises so the shared cache cannot let one
            # thread skip the model (and the barrier).
            score = entailment.score_entailment(f"evidence {index}", "claim")
            return score, dict(profile)
        finally:
            entailment.stop_hhem_profile(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_profile, range(2)))

    for score, profile in results:
        assert score == 0.8
        assert profile["call_count"] == 1
        assert profile["total_ms"] >= 0


def test_profile_context_is_restored_after_exception():
    previous = entailment._active_hhem_profile.get()
    token, profile = entailment.start_hhem_profile()

    try:
        assert entailment._active_hhem_profile.get() is profile
        raise RuntimeError("simulated grounding failure")
    except RuntimeError as exc:
        assert str(exc) == "simulated grounding failure"
    finally:
        entailment.stop_hhem_profile(token)

    assert entailment._active_hhem_profile.get() is previous


def test_request_profile_ignores_empty_input_and_model_failure(monkeypatch):
    monkeypatch.setattr(
        entailment,
        "_get_model",
        lambda: (_ for _ in ()).throw(RuntimeError("intentional failure")),
    )

    previous_profile = entailment._active_hhem_profile.get()
    token, profile = entailment.start_hhem_profile()
    try:
        assert entailment.score_entailment("", "claim") == 0.0
        assert profile["call_count"] == 0

        assert entailment.score_entailment("evidence", "claim") == 0.0
        assert profile["call_count"] == 0
        assert profile["total_ms"] >= 0.0
    finally:
        entailment.stop_hhem_profile(token)

    assert entailment._active_hhem_profile.get() is previous_profile


def test_select_relevant_sentences_keeps_the_matching_sentence():
    def fake_tokenizer(text, add_special_tokens=False, **kwargs):
        return {"input_ids": text.split()}

    premise = "Cats purr loudly. The capital of France is Paris. Rain fell on Tuesday."

    result = entailment._select_relevant_sentences(
        premise,
        "Paris is the capital of France",
        fake_tokenizer,
        budget=7,
    )

    assert result == "The capital of France is Paris."
    assert "Cats" not in result
