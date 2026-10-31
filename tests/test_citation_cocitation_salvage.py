from app.agents.nodes.grounding_critic import _strip_unsupported_co_citations as strip

CLAIM = "Participants identified control articles with about 86 % accuracy and GPT-3 175B articles with about 52 % accuracy ."


def _cc(supported_cids, bad_cids, method="not_selected"):
    results = [
        {"claim": CLAIM, "citation": c, "supported": True, "support_method": "deterministic_exact_match"}
        for c in supported_cids
    ] + [
        {"claim": CLAIM, "citation": c, "supported": False, "support_method": method, "severity": "hard"}
        for c in bad_cids
    ]
    return {
        "passed": not bad_cids,
        "violations": [r for r in results if not r["supported"]],
        "results": results,
    }


ATOMIC = [{"claim": CLAIM, "supported": True}]


def test_extra_unsupported_co_citation_is_removed():
    answer = CLAIM + " [chunk_2][chunk_5]"
    out = strip(answer, _cc(["chunk_2"], ["chunk_5"]), ATOMIC)
    assert "[chunk_5]" not in out
    assert "[chunk_2]" in out


def test_claim_with_no_supporting_citation_is_not_salvaged():
    answer = CLAIM + " [chunk_5]"
    out = strip(answer, _cc([], ["chunk_5"]), [{"claim": CLAIM, "supported": False}])
    assert out == ""


def test_missing_evidence_is_not_salvaged():
    answer = CLAIM + " [chunk_2][chunk_9]"
    out = strip(answer, _cc(["chunk_2"], ["chunk_9"], method="missing_evidence"), ATOMIC)
    assert out == ""


def test_no_violations_returns_empty():
    assert strip(CLAIM + " [chunk_2]", _cc(["chunk_2"], []), ATOMIC) == ""


def test_violations_on_already_failed_claims_are_ignored():
    other = "Another claim that was unsupported and already pruned away ."
    cc = _cc(["chunk_2"], ["chunk_5"])
    cc["violations"].append(
        {"claim": other, "citation": "chunk_7", "supported": False,
         "support_method": "not_selected", "severity": "hard"}
    )
    atomic = [{"claim": CLAIM, "supported": True}, {"claim": other, "supported": False}]
    out = strip(CLAIM + " [chunk_2][chunk_5]", cc, atomic)
    assert "[chunk_5]" not in out
    assert "[chunk_2]" in out
