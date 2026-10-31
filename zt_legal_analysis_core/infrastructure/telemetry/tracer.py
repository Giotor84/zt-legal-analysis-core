# -*- coding: utf-8 -*-
"""
zt_legal_analysis_core/infrastructure/telemetry/tracer.py

Observability stack — A8 del piano evolutivo rev. 3.

Design:
- Zero dipendenze obbligatorie: funziona con solo stdlib Python.
- File exporter locale: ogni span viene accodato a
  data/quality_traces/{fascicolo_id}.jsonl (una riga JSON per span).
- Langfuse opzionale: se LANGFUSE_HOST è impostato nell'env e langfuse
  è installato, gli span vengono inviati anche lì.
- Compatibile locale + SageMaker: su SageMaker i file JSONL finiscono
  su S3 insieme al resto degli output, senza infrastruttura aggiuntiva.

Utilizzo base:
    from zt_legal_analysis_core.infrastructure.telemetry.tracer import AgentTracer

    tracer = AgentTracer(fascicolo_id="4038276", runtime_profile="local_default")

    with tracer.span("FactExtractorAgent", model="qwen2.5:14b") as span:
        # ... lavoro dell'agente ...
        span.set_tokens(input_tokens=1200, output_tokens=340)
        span.set_meta(doc_count=5)

    tracer.flush()  # scrive il file (chiamato anche da __del__ come safety net)

Struttura di ogni riga nel JSONL:
    {
      "fascicolo_id": "4038276",
      "run_id": "uuid4",
      "runtime_profile": "local_default",
      "agent_id": "FactExtractorAgent",
      "model": "qwen2.5:14b",
      "started_at": "2026-04-25T10:00:00.123456",
      "ended_at":   "2026-04-25T10:00:05.678901",
      "duration_ms": 5555,
      "input_tokens": 1200,
      "output_tokens": 340,
      "cost_usd": null,
      "status": "ok",          # ok | error | degraded
      "degraded_reason": null,
      "meta": {"doc_count": 5}
    }
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Percorso base per i file JSONL.  Può essere sovrascritto via env.
_DEFAULT_TRACES_DIR = Path("data") / "quality_traces"


# ─────────────────────────────────────────────────────────────────────────────
# SpanRecord — dati accumulati durante un singolo span
# ─────────────────────────────────────────────────────────────────────────────

class SpanRecord:
    """Accumulatore mutabile per un singolo span agente.

    Non istanziare direttamente: usa AgentTracer.span().
    """

    def __init__(
        self,
        fascicolo_id: str,
        run_id: str,
        runtime_profile: str,
        agent_id: str,
        model: Optional[str] = None,
    ) -> None:
        self.fascicolo_id = fascicolo_id
        self.run_id = run_id
        self.runtime_profile = runtime_profile
        self.agent_id = agent_id
        self.model = model

        self._started_at: Optional[datetime] = None
        self._ended_at: Optional[datetime] = None
        self._start_perf: Optional[float] = None

        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.cost_usd: Optional[float] = None
        self.status: str = "ok"
        self.degraded_reason: Optional[str] = None
        self.error_message: Optional[str] = None
        self.meta: Dict[str, Any] = {}

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _start(self) -> None:
        self._started_at = datetime.now(tz=timezone.utc)
        self._start_perf = time.perf_counter()

    def _end(self) -> None:
        self._ended_at = datetime.now(tz=timezone.utc)

    # ── Setter pubblici ───────────────────────────────────────────────────────

    def set_tokens(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """Registra conteggio token input/output."""
        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens

    def set_cost(self, cost_usd: float) -> None:
        """Registra costo in USD (da CostTracer se disponibile)."""
        self.cost_usd = cost_usd

    def set_degraded(self, reason: str) -> None:
        """Marca lo span come degraded (fallback silenzioso)."""
        self.status = "degraded"
        self.degraded_reason = reason

    def set_error(self, message: str) -> None:
        """Marca lo span come errore."""
        self.status = "error"
        self.error_message = message

    def set_meta(self, **kwargs: Any) -> None:
        """Aggiunge metadati liberi (es. doc_count, gate_verdict)."""
        self.meta.update(kwargs)

    # ── Serializzazione ───────────────────────────────────────────────────────

    @property
    def duration_ms(self) -> Optional[int]:
        if self._started_at and self._ended_at:
            return int((self._ended_at - self._started_at).total_seconds() * 1000)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fascicolo_id": self.fascicolo_id,
            "run_id": self.run_id,
            "runtime_profile": self.runtime_profile,
            "agent_id": self.agent_id,
            "model": self.model,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "ended_at": self._ended_at.isoformat() if self._ended_at else None,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "status": self.status,
            "degraded_reason": self.degraded_reason,
            "error_message": self.error_message,
            "meta": self.meta,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AgentTracer — entry point principale
# ─────────────────────────────────────────────────────────────────────────────

class AgentTracer:
    """Tracer per un singolo run di fascicolo.

    Ogni run ottiene un run_id univoco.  Gli span vengono accumulati in
    memoria e scritti atomicamente su JSONL al flush().

    Args:
        fascicolo_id:     ID del fascicolo in elaborazione.
        runtime_profile:  Profilo runtime attivo (local_default, sagemaker_highcap, ...).
        traces_dir:       Directory dove scrivere i file JSONL.
                          Default: data/quality_traces/ (relativa a cwd).
        enabled:          Se False, tutti gli span sono no-op.  Utile nei test.
    """

    def __init__(
        self,
        fascicolo_id: str,
        runtime_profile: str = "local_default",
        traces_dir: Optional[Path] = None,
        enabled: bool = True,
    ) -> None:
        self.fascicolo_id = fascicolo_id
        self.runtime_profile = runtime_profile
        self.run_id = str(uuid.uuid4())
        self.enabled = enabled

        self._traces_dir = traces_dir or Path(
            os.environ.get("ZT_LEGAL_TRACES_DIR", str(_DEFAULT_TRACES_DIR))
        )
        self._spans: list[SpanRecord] = []
        self._flushed = False

        # Langfuse opzionale
        self._langfuse: Any = None
        self._try_init_langfuse()

    # ── Langfuse opt-in ──────────────────────────────────────────────────────

    def _try_init_langfuse(self) -> None:
        """Inizializza Langfuse se LANGFUSE_HOST è nel env e SDK installato."""
        host = os.environ.get("LANGFUSE_HOST")
        if not host:
            return
        try:
            from langfuse import Langfuse  # type: ignore[import]
            self._langfuse = Langfuse(
                host=host,
                public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            )
        except ImportError:
            logger.debug("langfuse non installato — solo file exporter attivo")
        except Exception as exc:  # pragma: no cover
            logger.warning("Langfuse init fallito: %s — continuo senza", exc)

    # ── Context manager span ─────────────────────────────────────────────────

    @contextmanager
    def span(
        self,
        agent_id: str,
        model: Optional[str] = None,
    ) -> Generator[SpanRecord, None, None]:
        """Context manager che traccia un singolo span agente.

        Esempio:
            with tracer.span("EvidenceGateAgent") as span:
                result = gate.evaluate(canonical)
                if result.has_blockers:
                    span.set_meta(gate_blocked=True, blocker_count=len(result.blockers))
        """
        if not self.enabled:
            # no-op span
            record = SpanRecord(
                fascicolo_id=self.fascicolo_id,
                run_id=self.run_id,
                runtime_profile=self.runtime_profile,
                agent_id=agent_id,
                model=model,
            )
            yield record
            return

        record = SpanRecord(
            fascicolo_id=self.fascicolo_id,
            run_id=self.run_id,
            runtime_profile=self.runtime_profile,
            agent_id=agent_id,
            model=model,
        )
        record._start()
        try:
            yield record
        except Exception as exc:
            record.set_error(str(exc))
            raise
        finally:
            record._end()
            self._spans.append(record)
            self._emit_to_langfuse(record)

    # ── Flush → file JSONL ───────────────────────────────────────────────────

    def flush(self) -> Optional[Path]:
        """Scrive tutti gli span accumulati nel file JSONL del fascicolo.

        Ritorna il path del file scritto, o None se disabled/no spans.
        Idempotente: una seconda chiamata aggiunge gli eventuali nuovi span.
        """
        if not self.enabled or not self._spans:
            return None

        self._traces_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._traces_dir / f"{self.fascicolo_id}.jsonl"

        lines = [json.dumps(s.to_dict(), ensure_ascii=False, default=str) for s in self._spans]
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        self._spans.clear()
        self._flushed = True

        if self._langfuse:
            try:
                self._langfuse.flush()
            except Exception as exc:  # pragma: no cover
                logger.warning("Langfuse flush fallito: %s", exc)

        return out_path

    # ── Langfuse emission ────────────────────────────────────────────────────

    def _emit_to_langfuse(self, record: SpanRecord) -> None:
        """Invia uno span a Langfuse (best-effort, non blocca)."""
        if not self._langfuse:
            return
        try:
            trace = self._langfuse.trace(
                id=record.run_id,
                name=f"fascicolo_{self.fascicolo_id}",
                metadata={"fascicolo_id": self.fascicolo_id, "runtime_profile": self.runtime_profile},
            )
            trace.span(
                name=record.agent_id,
                start_time=record._started_at,
                end_time=record._ended_at,
                metadata={
                    "model": record.model,
                    "status": record.status,
                    "duration_ms": record.duration_ms,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "cost_usd": record.cost_usd,
                    "degraded_reason": record.degraded_reason,
                    **record.meta,
                },
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse emit fallito per %s: %s", record.agent_id, exc)

    # ── Safety net ───────────────────────────────────────────────────────────

    def __del__(self) -> None:
        """Flush automatico se dimenticato (best-effort)."""
        if self._spans:
            try:
                self.flush()
            except Exception:  # pragma: no cover
                pass

    # ── Utility ──────────────────────────────────────────────────────────────

    def read_trace(self) -> list[Dict[str, Any]]:
        """Legge e restituisce tutti gli span del fascicolo dal file JSONL.

        Utile nei test e nel tooling di analisi.
        """
        path = self._traces_dir / f"{self.fascicolo_id}.jsonl"
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records
