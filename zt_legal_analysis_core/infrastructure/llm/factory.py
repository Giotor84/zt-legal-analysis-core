# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Factory per LLMClient: carica il profilo runtime dal YAML e restituisce l'adapter corretto.

Uso:
    from zt_legal_analysis_core.infrastructure.llm import get_llm_client
    client = get_llm_client("bedrock_managed")   # → BedrockAdapter
    client = get_llm_client("local_ollama")       # → OllamaAdapter
    client = get_llm_client()                     # → profilo default da env LLM_RUNTIME_PROFILE
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import yaml
from loguru import logger

from zt_legal_analysis_core.infrastructure.llm.base import LLMClient, RuntimeProfile
from zt_legal_analysis_core.infrastructure.llm.adapters import BedrockAdapter, OllamaAdapter
from zt_legal_analysis_core.infrastructure.llm.vllm_adapter import VLLMAdapter

# Directory contenente i profili runtime YAML
_PROFILES_DIR = Path(__file__).parent.parent.parent / "config" / "runtime_profiles"

_DEFAULT_PROFILE_ENV = "LLM_RUNTIME_PROFILE"
_DEFAULT_PROFILE_NAME = "local_ollama"


def load_profile(profile_name: str, profiles_dir: Optional[Path] = None) -> RuntimeProfile:
    """
    Carica un profilo runtime da config/runtime_profiles/<profile_name>.yaml.

    Lancia FileNotFoundError se il file non esiste.
    Lancia KeyError se mancano i campi obbligatori (backend, model).
    """
    base = profiles_dir or _PROFILES_DIR
    path = base / f"{profile_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profilo runtime non trovato: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    for required in ("backend", "model"):
        if required not in data:
            raise KeyError(f"Campo obbligatorio '{required}' mancante in {path}")
    profile = RuntimeProfile.from_dict(profile_name, data)
    logger.debug(f"[LLMFactory] profilo caricato: {profile_name} → {profile.backend}/{profile.model}")
    return profile


def build_client(profile: RuntimeProfile) -> Union[OllamaAdapter, BedrockAdapter, VLLMAdapter]:
    """
    Costruisce l'adapter appropriato dato un RuntimeProfile.
    """
    if profile.backend == "bedrock":
        return BedrockAdapter(
            model=profile.model,
            region=profile.region,
            timeout=profile.timeout,
            proxy=profile.proxy,
        )
    if profile.backend == "ollama":
        return OllamaAdapter(
            model=profile.model,
            timeout=profile.timeout,
        )
    if profile.backend == "vllm":
        base_url = (profile.extra or {}).get("vllm_base_url", "http://localhost:8001/v1")
        guided = (profile.extra or {}).get("guided_decoding_backend", "auto")
        return VLLMAdapter(
            model=profile.model,
            base_url=base_url,
            timeout=profile.timeout,
            guided_decoding_backend=guided,
        )
    raise ValueError(f"Backend non supportato: '{profile.backend}'. Valori ammessi: ollama, bedrock, vllm")


def get_llm_client(
    profile_name: Optional[str] = None,
    profiles_dir: Optional[Path] = None,
) -> Union[OllamaAdapter, BedrockAdapter, VLLMAdapter]:
    """
    Entry point principale.

    profile_name:
      - None → legge la variabile d'ambiente LLM_RUNTIME_PROFILE,
                con fallback a "local_ollama"
      - stringa → usa il profilo indicato
    """
    name = profile_name or os.environ.get(_DEFAULT_PROFILE_ENV, _DEFAULT_PROFILE_NAME)
    profile = load_profile(name, profiles_dir=profiles_dir)
    return build_client(profile)
