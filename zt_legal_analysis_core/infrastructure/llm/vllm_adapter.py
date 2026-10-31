# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
VLLMAdapter — implementa il Protocol LLMClient per backend vLLM self-hosted.

Usato esclusivamente con runtime_profile=sagemaker_highcap.
vLLM espone un endpoint OpenAI-compatible (/v1/chat/completions).

Dipendenze runtime: requests (già presente in requirements.txt).
Nessuna dipendenza da openai SDK — chiamata HTTP diretta.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from loguru import logger

from zt_legal_analysis_core.infrastructure.llm.base import LLMResponse
from zt_legal_analysis_core.infrastructure.llm.adapters import _parse_json_from_text, _build_structured_prompt


class VLLMAdapter:
    """
    Adapter per vLLM self-hosted (SageMaker high-cap, §13bis).

    Interfaccia identica a OllamaAdapter / BedrockAdapter (Protocol LLMClient).
    vLLM espone /v1/chat/completions in formato OpenAI-compatible.
    Structured output via guided decoding (xgrammar / outlines) se disponibile.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8001/v1",
        timeout: float = 1800.0,
        guided_decoding_backend: str = "auto",   # "xgrammar" | "outlines" | "auto" | "none"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.guided_decoding_backend = guided_decoding_backend
        logger.info(
            f"✅ VLLMAdapter inizializzato — modello: {model}, "
            f"base_url: {base_url}, guided: {guided_decoding_backend}"
        )

    # ------------------------------------------------------------------
    # Protocol LLMClient implementation
    # ------------------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        resp = self._chat(
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
        resp = self._chat(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_schema=schema if self._supports_guided_decoding() else None,
        )
        return _parse_json_from_text(resp.content)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _supports_guided_decoding(self) -> bool:
        return self.guided_decoding_backend not in ("none", "")

    def _chat(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        import requests

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": max(float(temperature), 0.0),
            "max_tokens": int(max_tokens),
            "stream": False,
        }

        # Structured output via guided decoding (vLLM ≥ 0.6)
        # Parametri vLLM: extra_body.guided_json
        if json_schema and self._supports_guided_decoding():
            payload["extra_body"] = {
                "guided_json": json_schema,
                "guided_decoding_backend": self.guided_decoding_backend
                if self.guided_decoding_backend != "auto"
                else None,
            }
            # Rimuovi None per pulizia payload
            payload["extra_body"] = {
                k: v for k, v in payload["extra_body"].items() if v is not None
            }

        logger.debug(
            f"[VLLMAdapter] START model={self.model} "
            f"prompt_chars={len(prompt)} "
            f"system_chars={len(system_prompt or '')} "
            f"guided={bool(json_schema)}"
        )

        url = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"VLLMAdapter: impossibile connettersi a {url}. "
                f"Verificare che vLLM sia attivo sul notebook SageMaker."
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"VLLMAdapter HTTP {response.status_code}: {response.text[:400]}"
            )

        data = response.json()

        # Parsing OpenAI-compatible response
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"VLLMAdapter: nessuna choice nella risposta: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content") or ""

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        tokens_used = input_tokens + output_tokens

        logger.debug(
            f"[VLLMAdapter] OK input_tokens={input_tokens} "
            f"output_tokens={output_tokens} "
            f"content_chars={len(content)}"
        )

        return LLMResponse(content=content, model=self.model, tokens_used=tokens_used)
