# -*- coding: utf-8 -*-
# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Financial Logic - Second pass analysis for importi split.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from loguru import logger


_AMOUNT_RE = re.compile(r"€?\s*([0-9]{1,3}(?:\.[0-9]{3})*(?:,[0-9]{2})?)")


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = cleaned.replace("€", "").replace(" ", "")
        if "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def normalize_importi_payload(importi: Any, credito: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = importi if isinstance(importi, dict) else {}
    credito_data = credito if isinstance(credito, dict) else {}

    normalized: Dict[str, Any] = {
        "capitale": 0.0,
        "interessi": 0.0,
        "spese": 0.0,
        "totale": 0.0,
        "fonte": payload.get("fonte") or payload.get("source_file") or credito_data.get("fonte") or "unknown",
    }

    if isinstance(importi, list):
        for item in importi:
            if not isinstance(item, dict):
                continue
            tipo = str(item.get("tipo") or "").strip().lower()
            valore = _to_float(item.get("valore") or item.get("importo") or item.get("amount"))
            if valore is None:
                continue
            if tipo in {"capitale", "sorte_capitale", "importo_capitale"}:
                normalized["capitale"] += valore
            elif tipo in {"interessi", "interessi_mora", "interessi_moratori"}:
                normalized["interessi"] += valore
            elif tipo in {"spese", "spese_legali"}:
                normalized["spese"] += valore
            elif tipo in {"totale", "totale_dovuto", "importo_totale", "esposizione", "saldo"}:
                normalized["totale"] = max(normalized["totale"], valore)

    elif isinstance(importi, dict):
        normalized["capitale"] = _to_float(payload.get("capitale")) or _to_float(payload.get("importo_capitale")) or 0.0
        normalized["interessi"] = _to_float(payload.get("interessi")) or 0.0
        normalized["spese"] = _to_float(payload.get("spese")) or 0.0
        normalized["totale"] = (
            _to_float(payload.get("totale"))
            or _to_float(payload.get("importo_totale"))
            or 0.0
        )

    credito_totale = _to_float(credito_data.get("importo_totale")) or 0.0
    if credito_totale > normalized["totale"]:
        normalized["totale"] = credito_totale

    if normalized["totale"] <= 0:
        subtotal = normalized["capitale"] + normalized["interessi"] + normalized["spese"]
        if subtotal > 0:
            normalized["totale"] = subtotal

    return normalized


def _parse_amount(value: str) -> Optional[float]:
    if not value:
        return None
    cleaned = value.replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _find_amount_near(text: str, keyword: str) -> Optional[float]:
    idx = text.find(keyword)
    if idx < 0:
        return None
    window = text[idx: idx + 200]
    match = _AMOUNT_RE.search(window)
    if not match:
        return None
    return _parse_amount(match.group(1))


def recover_financial_breakdown(
    importi: Dict[str, Any],
    full_text: str,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Forza una seconda passata per splittare il totale se capitale/interessi/spese sono nulli.
    """
    if not importi:
        importi = {}

    totale = importi.get("totale") or 0
    capitale = importi.get("capitale") or 0
    interessi = importi.get("interessi") or 0
    spese = importi.get("spese") or 0

    if not totale or (capitale or interessi or spese):
        return importi

    text = (full_text or "").lower()
    if not text:
        importi["note"] = "Dato non splittato nel documento sorgente"
        importi["capitale"] = None
        importi["interessi"] = None
        importi["spese"] = None
        return importi

    keywords = {
        "capitale": ["sorte capitale", "capitale"],
        "interessi": ["interessi di mora", "interessi"],
        "spese": ["spese"],
    }

    recovered = {"capitale": None, "interessi": None, "spese": None}

    for key, variants in keywords.items():
        for variant in variants:
            amount = _find_amount_near(text, variant)
            if amount:
                recovered[key] = amount
                break

    if llm_client and not any(recovered.values()):
        try:
            prompt = (
                "Separa il TOTALE in capitale, interessi e spese.\n"
                f"Totale: {totale}\n"
                "Testo:\n"
                f"{(full_text or '')[:4000]}\n"
            )
            schema = "{\"capitale\": 0.0, \"interessi\": 0.0, \"spese\": 0.0}"
            llm_data = llm_client.generate_json(prompt=prompt, schema_hint=schema, temperature=0.0)
            recovered["capitale"] = llm_data.get("capitale") or None
            recovered["interessi"] = llm_data.get("interessi") or None
            recovered["spese"] = llm_data.get("spese") or None
        except Exception as exc:
            logger.debug(f"Financial recovery LLM failed: {exc}")

    if not any(recovered.values()):
        importi["note"] = "Dato non splittato nel documento sorgente"
        importi["capitale"] = None
        importi["interessi"] = None
        importi["spese"] = None
        return importi

    importi["capitale"] = recovered["capitale"] or importi.get("capitale") or 0
    importi["interessi"] = recovered["interessi"] or importi.get("interessi") or 0
    importi["spese"] = recovered["spese"] or importi.get("spese") or 0
    return importi
