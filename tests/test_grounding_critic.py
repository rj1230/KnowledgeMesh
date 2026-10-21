import math
from app.agents.nodes.grounding_critic import _resolve_citation_metadata


def test_resolve_citation_metadata_preserves_document_scoped_identity():
    provenance = {
        "chunk_1": {
            "citation_id": "chunk_1",
            "document_id": "document_A",
            "chunk_id": 7,
            "point_id": "point-A-7",
        },
        "chunk_2": {
            "citation_id": "chunk_2",
            "document_id": "document_B",
            "chunk_id": 7,
            "point_id": "point-B-7",
        },
    }

    result = _resolve_citation_metadata(provenance)

    assert result["chunk_1"]["document_id"] == "document_A"
    assert result["chunk_1"]["chunk_id"] == 7
    assert result["chunk_1"]["point_id"] == "point-A-7"
    assert result["chunk_1"]["canonical_evidence_id"] == "document_A::7"

    assert result["chunk_2"]["document_id"] == "document_B"
    assert result["chunk_2"]["chunk_id"] == 7
    assert result["chunk_2"]["point_id"] == "point-B-7"
    assert result["chunk_2"]["canonical_evidence_id"] == "document_B::7"

    assert (
        result["chunk_1"]["canonical_evidence_id"]
        != result["chunk_2"]["canonical_evidence_id"]
    )


def test_score_claim_against_evidence_reuses_cached_score(monkeypatch):
    import app.agents.nodes.grounding_critic as grounding

    calls = []

    def fake_scorer(*, premise, hypothesis):
        calls.append((premise, hypothesis))
        return 0.40

    monkeypatch.setattr(
        grounding, "_get_entailment_scorer", lambda: fake_scorer
    )
    monkeypatch.setattr(
        grounding,
        "_select_entailment_candidates",
        lambda claim, evidence_units: [
            "Evidence supports the system design."
        ],
    )

    cache = {}
    claim = "The system uses a design process."
    evidence = ["Evidence supports the system design."]

    first_score, first_evidence = grounding._score_claim_against_evidence(
        claim, evidence, entailment_cache=cache
    )
    second_score, second_evidence = grounding._score_claim_against_evidence(
        claim, evidence, entailment_cache=cache
    )

    assert first_score == second_score == 0.40
    assert first_evidence == second_evidence == evidence[0]
    assert len(calls) == 1
from app.agents.nodes.grounding_critic import grounding_critic_node


def test_grounding_node_restores_serialized_entailment_cache(monkeypatch):
    import app.agents.nodes.grounding_critic as grounding
    import app.tools.entailment_tool as entailment_tool

    scorer_calls = []

    evidence = (
        "The platform stores indexed passages in a vector database "
        "and ranks retrieved passages before generating an answer."
    )
    claim = "Retrieved passages are ranked before the system produces its answer."

    def fake_scorer(*, premise, hypothesis):
        scorer_calls.append((premise, hypothesis))
        return 0.90

    monkeypatch.setattr(grounding, "_get_entailment_scorer", lambda: fake_scorer)
    monkeypatch.setattr(grounding, "_split_compound_claim", lambda text: [text])
    monkeypatch.setattr(
        grounding,
        "_select_global_entailment_candidates",
        lambda claim, evidence_map, excluded_citations=(),
               included_citations=(), limit=grounding.MAX_DISCOVERY_ENTAILMENT_CANDIDATES:
            [
                (citation_id, window)
                for citation_id, windows in evidence_map.items()
                if not excluded_citations or citation_id not in excluded_citations
                for window in windows
                if not included_citations or citation_id in included_citations
            ][:limit],
    )

    # Stable profile values isolate this test from real HHEMv2 inference.
    profile = {"call_count": 0, "total_ms": 0.0}
    monkeypatch.setattr(entailment_tool, "get_hhem_profile", lambda: profile.copy())

    initial_state = {
        "final_answer": f"{claim} [chunk_1]",
        "citation_provenance": {
            "chunk_1": {
                "citation_id": "chunk_1",
                "text": evidence,
                "document_id": "doc_1",
                "chunk_id": 1,
            }
        },
        "evaluation_trace": [],
    }

    first = grounding_critic_node(initial_state)

    assert len(scorer_calls) == 1
    assert first["entailment_cache"]
    assert first["cache_entries_returned"] >= 1

    second = grounding_critic_node(
        {
            **initial_state,
            "entailment_cache": first["entailment_cache"],
            "grounding_pass_count": first["grounding_pass_count"],
        }
    )

    assert len(scorer_calls) == 1, "Second pass repeated an entailment score"
    assert second["cache_entries_received"] >= 1
    assert second["entailment_cache_hits"] >= 1
    assert second["cache_entries_returned"] >= 1

    # Invalid restored scores must be rejected and recomputed.
    invalid_cache = [
        {**entry, "score": float("nan")}
        for entry in first["entailment_cache"]
    ]

    third = grounding_critic_node(
        {
            **initial_state,
            "entailment_cache": invalid_cache,
            "grounding_pass_count": first["grounding_pass_count"],
        }
    )

    assert len(scorer_calls) > 1, (
        "An invalid cached score was reused instead of recomputed"
    )
    assert third["cache_entries_returned"] >= 1
    assert all(
        isinstance(entry["score"], (int, float))
        and not isinstance(entry["score"], bool)
        and math.isfinite(entry["score"])
        and 0.0 <= entry["score"] <= 1.0
        for entry in third["entailment_cache"]
    )




def test_discovery_support_requires_valid_citation(monkeypatch):
    import app.agents.nodes.grounding_critic as grounding

    calls = []

    def fake_discovery(**kwargs):
        calls.append(kwargs)
        return {
            "score": 0.95,
            "supported": True,
            "best_evidence_unit": "Apparently supporting evidence.",
            "best_citation": "",
            "support_method": "entailment",
        }

    monkeypatch.setattr(
        grounding,
        "_discover_supporting_evidence",
        fake_discovery,
    )

    result = grounding._validate_atomic_claim(
        "The system supports this capability.",
        ["rejected_citation"],
        {"other_citation": ["Unrelated evidence."]},
    )

    assert calls, "The test did not exercise independent discovery"
    assert result["supported"] is False
    assert result["best_citation"] == ""
    assert result["citations"] == ["rejected_citation"]


def test_discovery_support_requires_citation_in_evidence_map(monkeypatch):
    import app.agents.nodes.grounding_critic as grounding

    calls = []

    def fake_discovery(**kwargs):
        calls.append(kwargs)
        return {
            "score": 0.95,
            "supported": True,
            "best_evidence_unit": "Apparently supporting evidence.",
            "best_citation": "missing_citation",
            "support_method": "entailment",
        }

    monkeypatch.setattr(
        grounding,
        "_discover_supporting_evidence",
        fake_discovery,
    )

    result = grounding._validate_atomic_claim(
        "The system supports this capability.",
        ["rejected_citation"],
        {"other_citation": ["Unrelated evidence."]},
    )

    assert calls, "The test did not exercise independent discovery"
    assert result["supported"] is False
    assert result["best_citation"] == "missing_citation"
    assert result["citations"] == ["rejected_citation"]


def test_citation_audit_cannot_override_unsupported_exact_match():
    import app.agents.nodes.grounding_critic as grounding

    result = grounding._audit_citation_consistency(
        atomic_results=[
            {
                "claim": "The system supports this capability.",
                "citations": ["chunk_1"],
                "best_citation": "chunk_1",
                "score": 1.0,
                "supported": False,
                "support_method": "deterministic_exact_match",
                "best_evidence_unit": "Some evidence text.",
            }
        ],
        evidence_map={
            "chunk_1": ["Some evidence text."]
        },
    )

    assert result["passed"] is False
    assert result["violation_count"] == 1
    assert result["violations"][0]["supported"] is False
    assert result["violations"][0]["support_method"] == (
        "deterministic_exact_match"
    )
