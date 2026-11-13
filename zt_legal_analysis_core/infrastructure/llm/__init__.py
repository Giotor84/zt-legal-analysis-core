# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
"""zt_legal_analysis_core.infrastructure.llm — LLMClient abstraction layer."""
from zt_legal_analysis_core.infrastructure.llm.base import LLMClient, LLMResponse, RuntimeProfile
from zt_legal_analysis_core.infrastructure.llm.adapters import OllamaAdapter, BedrockAdapter
from zt_legal_analysis_core.infrastructure.llm.vllm_adapter import VLLMAdapter
from zt_legal_analysis_core.infrastructure.llm.factory import get_llm_client, load_profile, build_client

__all__ = [
    "LLMClient",
    "LLMResponse",
    "RuntimeProfile",
    "OllamaAdapter",
    "BedrockAdapter",
    "VLLMAdapter",
    "get_llm_client",
    "load_profile",
    "build_client",
]
