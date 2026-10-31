# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
LLMClient Protocol — contratto formale per tutti i backend LLM.

Uso:
    from zt_legal_analysis_core.infrastructure.llm import LLMClient, LLMResponse, RuntimeProfile
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMResponse:
    """Risposta normalizzata da qualsiasi backend LLM."""
    content: str
    model: str
    tokens_used: int = 0


@dataclass(frozen=True)
class RuntimeProfile:
    """
    Profilo runtime caricato da config/runtime_profiles/<name>.yaml.

    Campi obbligatori: backend, model.
    Campi opzionali: region, timeout, max_tokens, temperature, proxy.
    """
    name: str
    backend: str            # "ollama" | "bedrock"
    model: str
    region: str = "eu-central-1"
    timeout: float = 1800.0
    max_tokens: int = 12000
    temperature: float = 0.1
    proxy: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "RuntimeProfile":
        known = {"backend", "model", "region", "timeout", "max_tokens", "temperature", "proxy"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            name=name,
            backend=data["backend"],
            model=data["model"],
            region=data.get("region", "eu-central-1"),
            timeout=float(data.get("timeout", 1800.0)),
            max_tokens=int(data.get("max_tokens", 12000)),
            temperature=float(data.get("temperature", 0.1)),
            proxy=data.get("proxy", ""),
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """
    Interfaccia formale che ogni backend LLM deve soddisfare.

    Implementazioni concrete: OllamaAdapter, BedrockAdapter.
    """

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        """Genera testo libero. Restituisce la stringa risposta."""
        ...

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Genera output JSON strutturato conforme a `schema`.

        `schema` è un JSON Schema dict (es. Pydantic .model_json_schema()).
        Restituisce il dict parsato. Lancia ValueError se non parsabile.
        """
        ...
