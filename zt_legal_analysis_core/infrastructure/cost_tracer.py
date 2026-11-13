# -*- coding: utf-8 -*-
# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
zt_legal_analysis_core/infrastructure/cost_tracer.py

Cost-as-first-class citizen (A7 del piano evolutivo).

Ogni fascicolo emette un cost_trace.json con:
- ogni chiamata LLM (agente, modello, token in/out, costo, cache hit)
- totale fascicolo
- verifica hard cap (abort con CostCapExceeded se sforato)

Uso tipico:
    tracer = CostTracer.for_fascicolo("4038276", profile="bedrock_managed")

    # dopo ogni chiamata LLM:
    tracer.record(CostCall(
        agent="FactExtractor",
        model="qwen3-30b",
        input_tokens=12500,
        output_tokens=3200,
    ))

    # a fine elaborazione:
    tracer.check_cap()          # raise CostCapExceeded se sforato
    tracer.write_json(path)     # scrive cost_trace.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_CONFIG_DIR = Path(__file__).parent.parent / "config"
_DEFAULT_POLICY_FILE = _CONFIG_DIR / "cost_policy.yaml"


# ---------------------------------------------------------------------------
# Eccezioni
# ---------------------------------------------------------------------------

class CostCapExceeded(Exception):
    """Sollevata quando il costo totale del fascicolo supera hard_cap_usd."""

    def __init__(self, fascicolo_id: str, total_usd: float, cap_usd: float) -> None:
        self.fascicolo_id = fascicolo_id
        self.total_usd = total_usd
        self.cap_usd = cap_usd
        super().__init__(
            f"[COST_OVERRUN] Fascicolo {fascicolo_id}: "
            f"{total_usd:.4f} USD > hard cap {cap_usd:.4f} USD"
        )


# ---------------------------------------------------------------------------
# Dataclass CostCall
# ---------------------------------------------------------------------------

@dataclass
class CostCall:
    """
    Singola chiamata LLM da registrare nel trace.

    Args:
        agent:         Nome dell'agente che ha generato la chiamata.
        model:         Nome del modello usato (come in config/models.yaml).
        input_tokens:  Token in input.
        output_tokens: Token in output.
        cost_usd:      Costo effettivo in USD (se noto dal backend). Se None,
                       viene calcolato da CostTracer usando la price table.
        cache_hit:     True se la risposta era in cache (Bedrock prompt caching).
        latency_ms:    Latenza in millisecondi (opzionale, per observability).
        timestamp:     Timestamp UTC della chiamata.
    """
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float] = None
    cache_hit: bool = False
    latency_ms: Optional[int] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# CostPolicy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostPolicy:
    """Valori letti da config/cost_policy.yaml."""
    hard_cap_usd: float = 0.50
    alert_threshold_usd: float = 0.30
    batch_hard_cap_usd: float = 50.0
    month_budget_usd: float = 500.0
    month_alert_at_pct: int = 80
    model_prices: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "CostPolicy":
        """Carica la policy dal file YAML. Usa i default se il file manca."""
        resolved = Path(path) if path else _DEFAULT_POLICY_FILE

        if not resolved.exists():
            logger.warning(
                f"[CostPolicy] File non trovato: {resolved} — uso default."
            )
            return cls()

        if not _YAML_AVAILABLE:
            logger.warning("[CostPolicy] PyYAML non installato — uso default.")
            return cls()

        with resolved.open(encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        per_f = data.get("per_fascicolo", {})
        per_b = data.get("per_batch", {})
        per_m = data.get("per_month", {})

        return cls(
            hard_cap_usd=float(per_f.get("hard_cap_usd", 0.50)),
            alert_threshold_usd=float(per_f.get("alert_threshold_usd", 0.30)),
            batch_hard_cap_usd=float(per_b.get("hard_cap_usd", 50.0)),
            month_budget_usd=float(per_m.get("budget_usd", 500.0)),
            month_alert_at_pct=int(per_m.get("alert_at_pct", 80)),
            model_prices=data.get("model_prices", {}),
        )

    def price_for_model(self, model: str) -> tuple[float, float]:
        """
        Ritorna (input_per_1k, output_per_1k) per il modello dato.
        Fallback su '_default' se il modello non è censito.
        """
        entry = self.model_prices.get(model) or self.model_prices.get("_default", {})
        return (
            float(entry.get("input_per_1k", 0.001)),
            float(entry.get("output_per_1k", 0.003)),
        )


# ---------------------------------------------------------------------------
# CostTracer
# ---------------------------------------------------------------------------

class CostTracer:
    """
    Traccia il costo LLM per un singolo fascicolo.

    Thread-safety: NON thread-safe. Un'istanza per fascicolo.
    """

    def __init__(
        self,
        fascicolo_id: str,
        runtime_profile: str = "local_default",
        policy: Optional[CostPolicy] = None,
    ) -> None:
        self._fascicolo_id = fascicolo_id
        self._runtime_profile = runtime_profile
        self._policy = policy or CostPolicy.from_yaml()
        self._calls: list[CostCall] = []
        self._started_at: str = datetime.now(timezone.utc).isoformat()

    # -- factory -------------------------------------------------------------

    @classmethod
    def for_fascicolo(
        cls,
        fascicolo_id: str,
        profile: str = "local_default",
        policy_path: Path | str | None = None,
    ) -> "CostTracer":
        """Factory con caricamento policy da YAML."""
        policy = CostPolicy.from_yaml(policy_path)
        return cls(fascicolo_id, profile, policy)

    # -- recording -----------------------------------------------------------

    def record(self, call: CostCall) -> None:
        """
        Registra una chiamata LLM.

        Se `call.cost_usd` è None, lo calcola dalla price table.
        Emette WARNING se si supera alert_threshold_usd.
        """
        if call.cost_usd is None:
            in_price, out_price = self._policy.price_for_model(call.model)
            computed = (
                call.input_tokens / 1000 * in_price
                + call.output_tokens / 1000 * out_price
            )
            call = CostCall(
                agent=call.agent,
                model=call.model,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                cost_usd=computed,
                cache_hit=call.cache_hit,
                latency_ms=call.latency_ms,
                timestamp=call.timestamp,
            )

        self._calls.append(call)

        total = self.total_cost_usd()
        if total >= self._policy.alert_threshold_usd:
            logger.warning(
                f"[CostTracer] Fascicolo {self._fascicolo_id}: "
                f"costo {total:.4f} USD >= soglia {self._policy.alert_threshold_usd:.2f} USD"
            )

    # -- cap enforcement -----------------------------------------------------

    def check_cap(self) -> None:
        """
        Verifica hard cap per fascicolo.
        Raise CostCapExceeded se sforato.
        """
        total = self.total_cost_usd()
        cap = self._policy.hard_cap_usd
        if total > cap:
            raise CostCapExceeded(self._fascicolo_id, total, cap)

    # -- aggregates ----------------------------------------------------------

    def total_cost_usd(self) -> float:
        return sum(c.cost_usd or 0.0 for c in self._calls)

    def total_tokens(self) -> int:
        return sum(c.total_tokens() for c in self._calls)

    def cache_hits(self) -> int:
        return sum(1 for c in self._calls if c.cache_hit)

    def cache_savings_usd(self) -> float:
        """
        Stima del risparmio per cache hit.
        Convenzione: un cache hit risparmia l'80% del costo input
        (Bedrock prompt caching: ~50%; usiamo 80% come stima conservativa).
        """
        savings = 0.0
        for c in self._calls:
            if c.cache_hit:
                in_price, _ = self._policy.price_for_model(c.model)
                savings += c.input_tokens / 1000 * in_price * 0.80
        return savings

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Struttura per cost_trace.json."""
        return {
            "fascicolo_id": self._fascicolo_id,
            "runtime_profile": self._runtime_profile,
            "started_at": self._started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "calls": [
                {
                    "agent": c.agent,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "total_tokens": c.total_tokens(),
                    "cost_usd": round(c.cost_usd or 0.0, 6),
                    "cache_hit": c.cache_hit,
                    "latency_ms": c.latency_ms,
                    "timestamp": c.timestamp,
                }
                for c in self._calls
            ],
            "total_cost_usd": round(self.total_cost_usd(), 6),
            "total_tokens": self.total_tokens(),
            "cache_hits": self.cache_hits(),
            "cache_savings_usd": round(self.cache_savings_usd(), 6),
            "hard_cap_usd": self._policy.hard_cap_usd,
            "cap_exceeded": self.total_cost_usd() > self._policy.hard_cap_usd,
        }

    def write_json(self, path: Path | str) -> Path:
        """
        Scrive cost_trace.json nel percorso specificato.

        Returns:
            Il Path del file scritto.
        """
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        logger.debug(
            f"[CostTracer] cost_trace.json scritto: {resolved} "
            f"(totale {self.total_cost_usd():.4f} USD)"
        )
        return resolved
