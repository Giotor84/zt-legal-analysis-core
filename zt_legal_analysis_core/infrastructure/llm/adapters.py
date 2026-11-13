# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Adapter concreti che implementano il Protocol LLMClient
delegando ai client già esistenti in zt_legal_analysis_core.core.llm_client.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from loguru import logger

from zt_legal_analysis_core.infrastructure.llm.base import LLMClient, LLMResponse, RuntimeProfile
from zt_legal_analysis_core.core.llm_client import OllamaClient, BedrockClient  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers comuni
# ---------------------------------------------------------------------------

def _parse_json_from_text(text: str) -> dict[str, Any]:
    """
    Estrae il primo oggetto o array JSON valido dal testo.
    Lancia ValueError se non trovato.
    """
    # Rimuovi eventuali blocchi ```json ... ```
    stripped = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, stripped)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Nessun JSON valido trovato nella risposta: {text[:300]}")


def _build_structured_prompt(prompt: str, schema: dict[str, Any]) -> str:
    """Aggiunge istruzioni JSON al prompt."""
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    return (
        f"{prompt}\n\n"
        f"Rispondi ESCLUSIVAMENTE con un JSON valido che rispetti il seguente schema:\n"
        f"```json\n{schema_str}\n```\n"
        f"Non aggiungere altro testo al di fuori del JSON."
    )


# ---------------------------------------------------------------------------
# OllamaAdapter
# ---------------------------------------------------------------------------

class OllamaAdapter:
    """
    Adatta zt_legal_analysis_core.core.llm_client.OllamaClient al Protocol LLMClient.

    Istanziabile direttamente oppure tramite factory.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 1800.0):
        self._inner = OllamaClient(model=model, timeout=timeout)
        self.model = model

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        resp = self._inner.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=False,
        )
        return resp.content

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        full_prompt = _build_structured_prompt(prompt, schema)
        resp = self._inner.chat(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            response_format="json",
            use_cache=False,
        )
        return _parse_json_from_text(resp.content)


# ---------------------------------------------------------------------------
# BedrockAdapter
# ---------------------------------------------------------------------------

class BedrockAdapter:
    """
    Adatta zt_legal_analysis_core.core.llm_client.BedrockClient al Protocol LLMClient.

    Istanziabile direttamente oppure tramite factory.
    """

    def __init__(
        self,
        model: str,
        region: str = "eu-central-1",
        timeout: float = 1800.0,
        proxy: str = "",
    ):
        self._inner = BedrockClient(model=model, region=region, timeout=timeout, proxy=proxy)
        self.model = model

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        resp = self._inner.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.content

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        full_prompt = _build_structured_prompt(prompt, schema)
        resp = self._inner.chat(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        return _parse_json_from_text(resp.content)
