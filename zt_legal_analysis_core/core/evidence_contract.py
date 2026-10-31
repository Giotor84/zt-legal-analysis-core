"""Contratti Pydantic per gli EvidenceFact prodotti da FactAssembler.

Filosofia (Step C — modalita' advisory):
- ZERO impatto sull'output: i fact non vengono trasformati ne' rimossi.
- Validazione opzionale, controllata da env var EVIDENCE_FACT_VALIDATION:
    * "off"      -> nessuna validazione (default sicuro per produzione)
    * "advisory" -> valida ma logga solo warning, NON solleva
    * "strict"   -> solleva ValidationError sul primo errore (uso test/CI)
- I modelli sono permissivi: extra="allow" sui sub-payload (value),
  cosi' possiamo tipizzare i campi noti senza rompere su nuovi campi futuri.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# Tipi noti — list "soft": se compare un nuovo tipo, andra' in WARNING ma non blocca.
KNOWN_FACT_TYPES = {
    "PARTY",
    "CREDITOR",
    "AMOUNT",
    "COLLATERAL",
    "PROCEEDING",
    "DOCUMENT",
    "DATE",
}

KNOWN_STATUS = {"RESOLVED", "CONFLICTED", "UNRESOLVED", "DRAFT"}
KNOWN_CONFIDENCE_LABELS = {"LOW", "MEDIUM", "HIGH"}


class ConfidenceBundleV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    score: float = Field(..., ge=0.0, le=1.0)
    label: str

    @field_validator("label")
    @classmethod
    def _label_known(cls, v: str) -> str:
        if v not in KNOWN_CONFIDENCE_LABELS:
            # advisory: accettiamo nuovi label ma logghiamo
            logger.debug(f"[evidence-contract] confidence label inattesa: {v!r}")
        return v


class ProvenanceItemV1(BaseModel):
    """Riferimento documentale.

    Permissivo: i fact_assembler aggiungono campi extra (page, snippet, ecc.).
    """
    model_config = ConfigDict(extra="allow")

    file_name: Optional[str] = None
    document_type: Optional[str] = None


class EvidenceFactV1(BaseModel):
    """Contratto v1 di un EvidenceFact prodotto da FactAssembler.build_evidence_facts().

    Schema derivato dall'output reale (vedi fact_assembler.py righe 539-660).
    """
    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(..., min_length=1)
    fact_type: str = Field(..., min_length=1)
    status: str = Field(..., min_length=1)
    value: Dict[str, Any] = Field(default_factory=dict)
    confidence: ConfidenceBundleV1
    provenance: List[ProvenanceItemV1] = Field(default_factory=list)

    @field_validator("fact_type")
    @classmethod
    def _fact_type_known(cls, v: str) -> str:
        if v not in KNOWN_FACT_TYPES:
            logger.debug(f"[evidence-contract] fact_type non in whitelist: {v!r}")
        return v

    @field_validator("status")
    @classmethod
    def _status_known(cls, v: str) -> str:
        if v not in KNOWN_STATUS:
            logger.debug(f"[evidence-contract] status non in whitelist: {v!r}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Validazione runtime (advisory)
# ─────────────────────────────────────────────────────────────────────────────

class ValidationIssue(BaseModel):
    """Un singolo problema rilevato dalla validazione advisory."""
    model_config = ConfigDict(extra="allow")

    fact_index: int
    fact_id: Optional[str] = None
    error: str


def _get_mode() -> Literal["off", "advisory", "strict"]:
    raw = (os.getenv("EVIDENCE_FACT_VALIDATION") or "off").strip().lower()
    if raw in {"off", "advisory", "strict"}:
        return raw  # type: ignore[return-value]
    logger.warning(
        f"[evidence-contract] EVIDENCE_FACT_VALIDATION={raw!r} non valido, fallback 'off'"
    )
    return "off"


def validate_evidence_facts(
    facts: List[Dict[str, Any]],
    *,
    fascicolo_id: str = "",
    mode: Optional[Literal["off", "advisory", "strict"]] = None,
) -> List[ValidationIssue]:
    """Valida una lista di evidence_facts contro EvidenceFactV1.

    - Mai modifica `facts` in-place.
    - In modalita' "off" ritorna [] subito (zero overhead).
    - In modalita' "advisory" raccoglie tutti gli issue, logga un warning
      di sintesi e li ritorna al chiamante (che puo' ignorarli).
    - In modalita' "strict" solleva la prima ValidationError incontrata.
    """
    effective_mode = mode if mode is not None else _get_mode()
    if effective_mode == "off":
        return []

    issues: List[ValidationIssue] = []
    for idx, fact in enumerate(facts):
        if not isinstance(fact, dict):
            issues.append(ValidationIssue(
                fact_index=idx,
                fact_id=None,
                error=f"fact non e' un dict: {type(fact).__name__}",
            ))
            continue
        try:
            EvidenceFactV1.model_validate(fact)
        except ValidationError as exc:
            if effective_mode == "strict":
                raise
            issues.append(ValidationIssue(
                fact_index=idx,
                fact_id=fact.get("fact_id"),
                error=str(exc),
            ))

    if issues and effective_mode == "advisory":
        prefix = f"[{fascicolo_id}] " if fascicolo_id else ""
        logger.warning(
            f"⚠️ {prefix}EvidenceFact contract: {len(issues)}/{len(facts)} fact non conformi a v1 "
            f"(modalita' advisory, output non modificato). Primo: "
            f"fact_id={issues[0].fact_id!r} err={issues[0].error[:140]}"
        )
    return issues


__all__ = [
    "EvidenceFactV1",
    "ConfidenceBundleV1",
    "ProvenanceItemV1",
    "ValidationIssue",
    "KNOWN_FACT_TYPES",
    "KNOWN_STATUS",
    "validate_evidence_facts",
]
