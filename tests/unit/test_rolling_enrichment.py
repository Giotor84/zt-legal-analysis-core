import pytest

from zt_legal_analysis_core.core.rolling_enrichment import documents_covered, plan_batches, rolling_synthesis

FIRST = "Sintetizza:\n{documents}"
ENRICH = "Sintesi attuale:\n{synthesis}\n\nArricchisci con:\n{documents}"


def _docs(n, size=100):
    return [(f"D{i:02d}", "x" * size) for i in range(n)]


def test_plan_batches_respects_budget_and_order():
    batches = plan_batches(_docs(10), first_budget=400, enrich_budget=280)
    assert [d for b in batches for d, _ in b] == [f"D{i:02d}" for i in range(10)]
    assert len(batches[0]) == 3 and all(len(b) <= 2 for b in batches[1:])


def test_oversized_document_stays_alone():
    batches = plan_batches([("A", "x" * 10), ("BIG", "x" * 1000), ("C", "x" * 10)], 100, 100)
    assert [[d for d, _ in b] for b in batches] == [["A"], ["BIG"], ["C"]]


def test_passes_enrich_progressively_within_context():
    seen = []

    def model(prompt):
        seen.append(prompt)
        n = prompt.count("### DOCUMENTO")
        prev = prompt.split("Sintesi attuale:\n")[1].split("\n\nArricchisci")[0] if "Sintesi attuale" in prompt else ""
        return (prev + f"\n## Parte {len(seen)}\n{n} documenti").strip()

    r = rolling_synthesis(_docs(9, 1000), model, first_prompt=FIRST, enrich_prompt=ENRICH,
                          context_chars=3200, reserve_chars=1000)
    assert len(r.passes) == 4 and not r.degraded          # 3 documenti nella prima passata, poi 2 alla volta
    assert all(p.chars_in <= 3200 for p in r.passes)
    assert "## Parte 1" in r.synthesis and "## Parte 4" in r.synthesis
    ok, ko = documents_covered(r)
    assert len(ok) == 9 and ko == []


def test_enrichment_that_loses_sections_is_rejected():
    answers = iter(["## Debitore\nRossi\n## Garanzie\nIpoteca", "## Debitore\nRossi"])
    r = rolling_synthesis(_docs(2, 1000), lambda p: next(answers), first_prompt=FIRST, enrich_prompt=ENRICH,
                          context_chars=1100, reserve_chars=50)
    assert "## Garanzie" in r.synthesis and r.degraded
    assert "sezioni perse: garanzie" in r.passes[1].reason
    assert documents_covered(r)[1] == ["D01"]


def test_failing_pass_keeps_previous_synthesis():
    calls = {"n": 0}

    def model(prompt):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TimeoutError("bedrock")
        return f"## Sintesi\nversione {calls['n']}"

    r = rolling_synthesis(_docs(3, 1000), model, first_prompt=FIRST, enrich_prompt=ENRICH,
                          context_chars=1100, reserve_chars=50)
    assert r.synthesis.endswith("versione 3") and r.passes[1].kept is False


def test_no_documents():
    with pytest.raises(ValueError):
        rolling_synthesis([], lambda p: "x", first_prompt=FIRST, enrich_prompt=ENRICH)
