# -*- coding: utf-8 -*-
"""
tests/unit/test_tracer.py

Test per AgentTracer (A8 — Observability stack).

Tutto gira offline, zero dipendenze esterne, zero file system permanenti
(usiamo tmp_path di pytest).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from zt_legal_analysis_core.infrastructure.telemetry.tracer import AgentTracer, SpanRecord


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tracer(tmp_path: Path, fascicolo_id: str = "TEST001", enabled: bool = True) -> AgentTracer:
    return AgentTracer(
        fascicolo_id=fascicolo_id,
        runtime_profile="local_default",
        traces_dir=tmp_path,
        enabled=enabled,
    )


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# TestSpanRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestSpanRecord:
    def test_to_dict_campi_base(self):
        r = SpanRecord(
            fascicolo_id="F1",
            run_id="RUN1",
            runtime_profile="local_default",
            agent_id="TestAgent",
            model="qwen2.5:14b",
        )
        r._start()
        time.sleep(0.01)
        r._end()
        d = r.to_dict()

        assert d["fascicolo_id"] == "F1"
        assert d["agent_id"] == "TestAgent"
        assert d["model"] == "qwen2.5:14b"
        assert d["status"] == "ok"
        assert d["duration_ms"] is not None
        assert d["duration_ms"] >= 10

    def test_set_tokens(self):
        r = SpanRecord("F", "R", "local_default", "A")
        r.set_tokens(input_tokens=500, output_tokens=200)
        assert r.to_dict()["input_tokens"] == 500
        assert r.to_dict()["output_tokens"] == 200

    def test_set_cost(self):
        r = SpanRecord("F", "R", "local_default", "A")
        r.set_cost(0.018)
        assert r.to_dict()["cost_usd"] == pytest.approx(0.018)

    def test_set_degraded(self):
        r = SpanRecord("F", "R", "local_default", "A")
        r.set_degraded("kuzu_unavailable")
        d = r.to_dict()
        assert d["status"] == "degraded"
        assert d["degraded_reason"] == "kuzu_unavailable"

    def test_set_error(self):
        r = SpanRecord("F", "R", "local_default", "A")
        r.set_error("timeout")
        d = r.to_dict()
        assert d["status"] == "error"
        assert d["error_message"] == "timeout"

    def test_set_meta(self):
        r = SpanRecord("F", "R", "local_default", "A")
        r.set_meta(doc_count=5, gate_blocked=True)
        d = r.to_dict()
        assert d["meta"]["doc_count"] == 5
        assert d["meta"]["gate_blocked"] is True

    def test_duration_none_before_end(self):
        r = SpanRecord("F", "R", "local_default", "A")
        r._start()
        assert r.duration_ms is None  # _end non ancora chiamato


# ─────────────────────────────────────────────────────────────────────────────
# TestAgentTracerDisabled
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentTracerDisabled:
    """Con enabled=False tutto è no-op: nessun file scritto."""

    def test_span_noop(self, tmp_path):
        tracer = _tracer(tmp_path, enabled=False)
        with tracer.span("AgentA") as span:
            span.set_tokens(100, 50)
        result = tracer.flush()
        assert result is None
        assert not (tmp_path / "TEST001.jsonl").exists()

    def test_flush_restituisce_none(self, tmp_path):
        tracer = _tracer(tmp_path, enabled=False)
        assert tracer.flush() is None


# ─────────────────────────────────────────────────────────────────────────────
# TestAgentTracerSpan
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentTracerSpan:
    def test_span_singolo_scrive_jsonl(self, tmp_path):
        tracer = _tracer(tmp_path, "FAX001")
        with tracer.span("FactExtractorAgent", model="qwen2.5:14b") as span:
            span.set_tokens(1200, 340)
            span.set_meta(doc_count=3)
        path = tracer.flush()

        assert path is not None
        assert path.exists()
        records = _read_jsonl(path)
        assert len(records) == 1
        r = records[0]
        assert r["fascicolo_id"] == "FAX001"
        assert r["agent_id"] == "FactExtractorAgent"
        assert r["model"] == "qwen2.5:14b"
        assert r["input_tokens"] == 1200
        assert r["output_tokens"] == 340
        assert r["meta"]["doc_count"] == 3
        assert r["status"] == "ok"

    def test_span_multipli_stesso_tracer(self, tmp_path):
        tracer = _tracer(tmp_path, "FAX002")
        with tracer.span("AgentA") as s:
            s.set_tokens(100, 50)
        with tracer.span("AgentB", model="gemma3:4b") as s:
            s.set_meta(gate_verdict="pass")
        tracer.flush()

        records = _read_jsonl(tmp_path / "FAX002.jsonl")
        assert len(records) == 2
        assert records[0]["agent_id"] == "AgentA"
        assert records[1]["agent_id"] == "AgentB"
        assert records[1]["model"] == "gemma3:4b"

    def test_flush_append_su_file_esistente(self, tmp_path):
        """Flush multipli sullo stesso fascicolo accumulano righe."""
        tracer = _tracer(tmp_path, "FAX003")
        with tracer.span("AgentA"):
            pass
        tracer.flush()

        tracer2 = _tracer(tmp_path, "FAX003")
        tracer2.run_id = tracer.run_id  # stesso run per semplicità
        with tracer2.span("AgentB"):
            pass
        tracer2.flush()

        records = _read_jsonl(tmp_path / "FAX003.jsonl")
        assert len(records) == 2

    def test_run_id_unico_per_tracer(self, tmp_path):
        t1 = _tracer(tmp_path, "X1")
        t2 = _tracer(tmp_path, "X1")
        assert t1.run_id != t2.run_id

    def test_span_cattura_eccezione_e_rilancia(self, tmp_path):
        tracer = _tracer(tmp_path, "FAX004")
        with pytest.raises(ValueError, match="test error"):
            with tracer.span("BrokenAgent") as span:
                raise ValueError("test error")
        # lo span deve essere stato registrato comunque con status=error
        tracer.flush()
        records = _read_jsonl(tmp_path / "FAX004.jsonl")
        assert len(records) == 1
        assert records[0]["status"] == "error"
        assert "test error" in records[0]["error_message"]

    def test_span_degraded(self, tmp_path):
        tracer = _tracer(tmp_path, "FAX005")
        with tracer.span("GraphGateAgent") as span:
            span.set_degraded("kuzu_unavailable")
        tracer.flush()
        records = _read_jsonl(tmp_path / "FAX005.jsonl")
        assert records[0]["status"] == "degraded"
        assert records[0]["degraded_reason"] == "kuzu_unavailable"

    def test_span_contiene_timestamps_iso(self, tmp_path):
        tracer = _tracer(tmp_path, "FAX006")
        with tracer.span("TimingAgent"):
            pass
        tracer.flush()
        records = _read_jsonl(tmp_path / "FAX006.jsonl")
        r = records[0]
        assert r["started_at"] is not None
        assert "T" in r["started_at"]
        assert r["ended_at"] is not None
        assert r["duration_ms"] >= 0

    def test_span_duration_ragionevole(self, tmp_path):
        tracer = _tracer(tmp_path, "FAX007")
        with tracer.span("SlowAgent"):
            time.sleep(0.05)
        tracer.flush()
        records = _read_jsonl(tmp_path / "FAX007.jsonl")
        assert records[0]["duration_ms"] >= 50


# ─────────────────────────────────────────────────────────────────────────────
# TestAgentTracerReadTrace
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentTracerReadTrace:
    def test_read_trace_restituisce_lista(self, tmp_path):
        tracer = _tracer(tmp_path, "FAX010")
        with tracer.span("A"):
            pass
        tracer.flush()

        records = tracer.read_trace()
        assert isinstance(records, list)
        assert len(records) == 1
        assert records[0]["agent_id"] == "A"

    def test_read_trace_file_mancante(self, tmp_path):
        tracer = _tracer(tmp_path, "NONEXISTENT")
        assert tracer.read_trace() == []

    def test_read_trace_dopo_flush_multipli(self, tmp_path):
        tracer = _tracer(tmp_path, "FAX011")
        with tracer.span("AgentA"):
            pass
        tracer.flush()
        with tracer.span("AgentB"):
            pass
        tracer.flush()
        records = tracer.read_trace()
        assert len(records) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestAgentTracerRunId
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentTracerRunId:
    def test_tutti_span_stesso_run_id(self, tmp_path):
        tracer = _tracer(tmp_path, "FAX020")
        with tracer.span("A"):
            pass
        with tracer.span("B"):
            pass
        tracer.flush()
        records = _read_jsonl(tmp_path / "FAX020.jsonl")
        run_ids = {r["run_id"] for r in records}
        assert len(run_ids) == 1  # tutti gli span dello stesso tracer = stesso run_id

    def test_runtime_profile_nel_record(self, tmp_path):
        tracer = AgentTracer(
            fascicolo_id="FAX021",
            runtime_profile="sagemaker_highcap",
            traces_dir=tmp_path,
        )
        with tracer.span("X"):
            pass
        tracer.flush()
        records = _read_jsonl(tmp_path / "FAX021.jsonl")
        assert records[0]["runtime_profile"] == "sagemaker_highcap"
