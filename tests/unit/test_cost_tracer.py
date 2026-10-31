# -*- coding: utf-8 -*-
"""
tests/unit/test_cost_tracer.py

Test per zt_legal_analysis_core.infrastructure.cost_tracer (A7 piano evolutivo).

Copertura:
- CostPolicy: caricamento YAML, default, price_for_model
- CostCall: calcolo total_tokens
- CostTracer: record, aggregati, check_cap, write_json, to_dict
"""
import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers di fixture
# ---------------------------------------------------------------------------

def _make_call(agent="TestAgent", model="qwen3-30b", input_t=1000, output_t=500,
               cost_usd=None, cache_hit=False, latency_ms=None):
    from zt_legal_analysis_core.infrastructure.cost_tracer import CostCall
    return CostCall(
        agent=agent,
        model=model,
        input_tokens=input_t,
        output_tokens=output_t,
        cost_usd=cost_usd,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
    )


def _make_tracer(fascicolo_id="FASC-TEST", profile="bedrock_managed", policy=None):
    from zt_legal_analysis_core.infrastructure.cost_tracer import CostTracer
    return CostTracer(fascicolo_id, profile, policy)


# ---------------------------------------------------------------------------
# Tests: CostCall
# ---------------------------------------------------------------------------

class TestCostCall:
    def test_total_tokens(self):
        call = _make_call(input_t=1000, output_t=500)
        assert call.total_tokens() == 1500

    def test_default_cost_is_none(self):
        call = _make_call()
        assert call.cost_usd is None

    def test_explicit_cost(self):
        call = _make_call(cost_usd=0.018)
        assert call.cost_usd == pytest.approx(0.018)

    def test_timestamp_is_set(self):
        call = _make_call()
        assert call.timestamp is not None
        assert "T" in call.timestamp  # ISO 8601


# ---------------------------------------------------------------------------
# Tests: CostPolicy
# ---------------------------------------------------------------------------

class TestCostPolicy:
    def test_default_hard_cap(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy
        policy = CostPolicy()
        assert policy.hard_cap_usd == pytest.approx(0.50)

    def test_default_alert_threshold(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy
        policy = CostPolicy()
        assert policy.alert_threshold_usd == pytest.approx(0.30)

    def test_load_from_yaml(self):
        """CostPolicy.from_yaml() carica correttamente config/cost_policy.yaml."""
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy
        policy = CostPolicy.from_yaml()  # usa il file reale in config/
        assert policy.hard_cap_usd > 0
        assert policy.alert_threshold_usd > 0
        assert len(policy.model_prices) > 0

    def test_price_for_known_model(self):
        """qwen3-30b deve avere prezzi > 0 (Bedrock managed)."""
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy
        policy = CostPolicy.from_yaml()
        inp, out = policy.price_for_model("qwen3-30b")
        assert inp > 0.0
        assert out > 0.0

    def test_price_for_local_model_is_zero(self):
        """Modelli Ollama locali devono avere costo zero."""
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy
        policy = CostPolicy.from_yaml()
        inp, out = policy.price_for_model("qwen2.5:14b")
        assert inp == pytest.approx(0.0)
        assert out == pytest.approx(0.0)

    def test_price_for_unknown_model_uses_default(self):
        """Modello sconosciuto usa _default."""
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy
        policy = CostPolicy.from_yaml()
        inp, out = policy.price_for_model("modello-non-censito-xyz")
        # il default è > 0 (fail safe: non gratis)
        assert inp > 0.0
        assert out > 0.0

    def test_from_yaml_missing_file_uses_defaults(self, tmp_path):
        """Se il file non esiste, usa valori di default senza crash."""
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy
        policy = CostPolicy.from_yaml(tmp_path / "nonexistent.yaml")
        assert policy.hard_cap_usd == pytest.approx(0.50)

    def test_from_yaml_custom_values(self, tmp_path):
        """Valori custom nel YAML vengono caricati correttamente."""
        yaml_content = """
per_fascicolo:
  hard_cap_usd: 1.25
  alert_threshold_usd: 0.75
per_batch:
  hard_cap_usd: 100.0
per_month:
  budget_usd: 1000.0
  alert_at_pct: 90
model_prices:
  my-model:
    input_per_1k: 0.005
    output_per_1k: 0.015
  _default:
    input_per_1k: 0.001
    output_per_1k: 0.003
"""
        p = tmp_path / "cost_policy.yaml"
        p.write_text(yaml_content)
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy
        policy = CostPolicy.from_yaml(p)
        assert policy.hard_cap_usd == pytest.approx(1.25)
        assert policy.alert_threshold_usd == pytest.approx(0.75)
        assert policy.month_budget_usd == pytest.approx(1000.0)
        inp, out = policy.price_for_model("my-model")
        assert inp == pytest.approx(0.005)
        assert out == pytest.approx(0.015)


# ---------------------------------------------------------------------------
# Tests: CostTracer — registrazione e aggregati
# ---------------------------------------------------------------------------

class TestCostTracerRecording:
    def test_record_computes_cost_when_none(self):
        """Se cost_usd è None, deve essere calcolato dalla price table."""
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy, CostTracer
        policy = CostPolicy(
            model_prices={"my-model": {"input_per_1k": 1.0, "output_per_1k": 2.0}}
        )
        tracer = CostTracer("FASC-001", policy=policy)
        tracer.record(_make_call(model="my-model", input_t=1000, output_t=500, cost_usd=None))
        # costo = 1000/1000 * 1.0 + 500/1000 * 2.0 = 1.0 + 1.0 = 2.0
        assert tracer.total_cost_usd() == pytest.approx(2.0)

    def test_record_preserves_explicit_cost(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostTracer
        tracer = _make_tracer()
        tracer.record(_make_call(cost_usd=0.042))
        assert tracer.total_cost_usd() == pytest.approx(0.042)

    def test_total_tokens_sums_all_calls(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostTracer
        tracer = _make_tracer()
        tracer.record(_make_call(input_t=1000, output_t=500, cost_usd=0.0))
        tracer.record(_make_call(input_t=2000, output_t=1000, cost_usd=0.0))
        assert tracer.total_tokens() == 4500

    def test_cache_hits_count(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostTracer
        tracer = _make_tracer()
        tracer.record(_make_call(cost_usd=0.0, cache_hit=True))
        tracer.record(_make_call(cost_usd=0.0, cache_hit=False))
        tracer.record(_make_call(cost_usd=0.0, cache_hit=True))
        assert tracer.cache_hits() == 2

    def test_multiple_calls_summed(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostTracer
        tracer = _make_tracer()
        tracer.record(_make_call(cost_usd=0.10))
        tracer.record(_make_call(cost_usd=0.15))
        tracer.record(_make_call(cost_usd=0.05))
        assert tracer.total_cost_usd() == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Tests: CostTracer — hard cap
# ---------------------------------------------------------------------------

class TestCostTracerCap:
    def test_check_cap_passes_when_under(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy, CostTracer
        policy = CostPolicy(hard_cap_usd=1.0)
        tracer = CostTracer("FASC-CAP", policy=policy)
        tracer.record(_make_call(cost_usd=0.50))
        tracer.check_cap()  # deve passare senza eccezione

    def test_check_cap_raises_when_exceeded(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostCapExceeded, CostPolicy, CostTracer
        policy = CostPolicy(hard_cap_usd=0.10)
        tracer = CostTracer("FASC-OVER", policy=policy)
        tracer.record(_make_call(cost_usd=0.12))
        with pytest.raises(CostCapExceeded) as exc_info:
            tracer.check_cap()
        assert exc_info.value.fascicolo_id == "FASC-OVER"
        assert exc_info.value.total_usd == pytest.approx(0.12)
        assert exc_info.value.cap_usd == pytest.approx(0.10)
        assert "COST_OVERRUN" in str(exc_info.value)

    def test_cap_exceeded_exactly_at_boundary(self):
        """Esattamente al cap non deve raise (solo > cap scatta)."""
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy, CostTracer
        policy = CostPolicy(hard_cap_usd=0.50)
        tracer = CostTracer("FASC-BOUNDARY", policy=policy)
        tracer.record(_make_call(cost_usd=0.50))
        tracer.check_cap()  # deve passare


# ---------------------------------------------------------------------------
# Tests: CostTracer — serializzazione
# ---------------------------------------------------------------------------

class TestCostTracerSerialization:
    def test_to_dict_structure(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostTracer
        tracer = CostTracer("FASC-SER", "bedrock_managed")
        tracer.record(_make_call(agent="A1", model="qwen3-30b", cost_usd=0.018,
                                 input_t=1000, output_t=300, cache_hit=False, latency_ms=1200))
        d = tracer.to_dict()
        assert d["fascicolo_id"] == "FASC-SER"
        assert d["runtime_profile"] == "bedrock_managed"
        assert len(d["calls"]) == 1
        assert d["calls"][0]["agent"] == "A1"
        assert d["calls"][0]["cost_usd"] == pytest.approx(0.018)
        assert d["calls"][0]["cache_hit"] is False
        assert d["calls"][0]["latency_ms"] == 1200
        assert d["total_cost_usd"] == pytest.approx(0.018)
        assert "started_at" in d
        assert "completed_at" in d

    def test_cap_exceeded_flag_in_dict(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostPolicy, CostTracer
        policy = CostPolicy(hard_cap_usd=0.10)
        tracer = CostTracer("FASC-FLAG", policy=policy)
        tracer.record(_make_call(cost_usd=0.20))
        d = tracer.to_dict()
        assert d["cap_exceeded"] is True

    def test_write_json_creates_file(self, tmp_path):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostTracer
        tracer = CostTracer("FASC-WRITE", "local_default")
        tracer.record(_make_call(cost_usd=0.001))
        out_file = tmp_path / "traces" / "FASC-WRITE" / "cost_trace.json"
        written = tracer.write_json(out_file)
        assert written == out_file
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["fascicolo_id"] == "FASC-WRITE"
        assert len(data["calls"]) == 1

    def test_write_json_creates_parent_dirs(self, tmp_path):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostTracer
        tracer = CostTracer("FASC-DIR", "local_default")
        deep_path = tmp_path / "a" / "b" / "c" / "cost_trace.json"
        tracer.write_json(deep_path)
        assert deep_path.exists()

    def test_empty_tracer_to_dict(self):
        """Un tracer senza chiamate deve comunque produrre un dict valido."""
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostTracer
        tracer = CostTracer("FASC-EMPTY", "local_default")
        d = tracer.to_dict()
        assert d["total_cost_usd"] == 0.0
        assert d["total_tokens"] == 0
        assert d["calls"] == []
        assert d["cap_exceeded"] is False

    def test_factory_for_fascicolo(self):
        from zt_legal_analysis_core.infrastructure.cost_tracer import CostTracer
        tracer = CostTracer.for_fascicolo("4038276", profile="bedrock_managed")
        assert tracer._fascicolo_id == "4038276"
        assert tracer._runtime_profile == "bedrock_managed"
        assert tracer._policy is not None
