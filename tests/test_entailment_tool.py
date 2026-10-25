import threading

import app.tools.entailment_tool as entailment


def test_score_entailment_success_updates_profile(monkeypatch):
    monkeypatch.setattr(entailment, "_hhem_call_count", 0)
    monkeypatch.setattr(entailment, "_hhem_total_ms", 0.0)
    monkeypatch.setattr(entailment, "_get_model", lambda: object())
    monkeypatch.setattr(
        entailment,
        "_predict_with_safe_truncation",
        lambda **kwargs: [0.8],
    )
    monkeypatch.setattr(entailment, "_extract_score", lambda output: 0.8)
    monkeypatch.setattr(entailment, "_normalize_score", lambda score: score)

    score = entailment.score_entailment("evidence", "claim")

    profile = entailment.get_hhem_profile()
    assert score == 0.8
    assert profile["call_count"] == 1
    assert profile["total_ms"] >= 0


def test_score_entailment_empty_input_returns_zero(monkeypatch):
    monkeypatch.setattr(entailment, "_hhem_call_count", 0)
    monkeypatch.setattr(entailment, "_hhem_total_ms", 0.0)

    score = entailment.score_entailment("", "claim")

    assert score == 0.0
    assert entailment.get_hhem_profile()["call_count"] == 1


def test_score_entailment_failure_returns_zero(monkeypatch):
    monkeypatch.setattr(entailment, "_hhem_call_count", 0)
    monkeypatch.setattr(entailment, "_hhem_total_ms", 0.0)
    monkeypatch.setattr(
        entailment, "_get_model",
        lambda: (_ for _ in ()).throw(RuntimeError("test failure")),
    )

    score = entailment.score_entailment("evidence", "claim")

    assert score == 0.0
    assert entailment.get_hhem_profile()["call_count"] == 1


def test_get_hhem_profile_returns_expected_fields():
    profile = entailment.get_hhem_profile()

    assert "call_count" in profile
    assert "total_ms" in profile



def test_concurrent_profiles_are_isolated(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2, timeout=10)

    monkeypatch.setattr(entailment, "_get_model", lambda: object())

    def fake_predict(**kwargs):
        barrier.wait()
        return [0.8]

    monkeypatch.setattr(
        entailment, "_predict_with_safe_truncation", fake_predict
    )
    monkeypatch.setattr(entailment, "_extract_score", lambda output: 0.8)
    monkeypatch.setattr(entailment, "_normalize_score", lambda score: score)

    def run_profile():
        token, profile = entailment.start_hhem_profile()
        try:
            score = entailment.score_entailment("evidence", "claim")
            return score, dict(profile)
        finally:
            entailment.stop_hhem_profile(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run_profile(), range(2)))

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


def test_request_profile_counts_empty_input_and_inference_failure(monkeypatch):
    monkeypatch.setattr(entailment, "_get_model", lambda: (
        (_ for _ in ()).throw(RuntimeError("intentional inference failure"))
    ))

    previous_profile = entailment._active_hhem_profile.get()
    token, profile = entailment.start_hhem_profile()
    try:
        assert entailment.score_entailment("", "claim") == 0.0
        assert profile["call_count"] == 1

        assert entailment.score_entailment("evidence", "claim") == 0.0
        assert profile["call_count"] == 2
        assert profile["total_ms"] >= 0.0
    finally:
        entailment.stop_hhem_profile(token)

    assert entailment._active_hhem_profile.get() is previous_profile

