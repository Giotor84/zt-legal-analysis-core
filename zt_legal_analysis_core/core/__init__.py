# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Core module - Componenti base del sistema
"""
from zt_legal_analysis_core.core.config import settings, ensure_directories
from zt_legal_analysis_core.core.llm_client import get_llm_client, OllamaClient

__all__ = [
    "settings",
    "ensure_directories",
    "get_llm_client",
    "OllamaClient",
]
