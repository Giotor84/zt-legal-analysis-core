# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
"""
Unit tests per zt_legal_analysis_core.infrastructure.llm

Copertura:
- RuntimeProfile (from_dict, campi, extra)
- LLMClient Protocol (isinstance check)
- load_profile (file esistente, mancante, campi obbligatori)
- build_client dispatch (ollama / bedrock / non supportato)
- OllamaAdapter.generate_structured (parseable JSON / non parseable)
- BedrockAdapter.generate_structured (stessa logica adapter)
- _parse_json_from_text (helper)
- Factory: variabile d'ambiente LLM_RUNTIME_PROFILE
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import yaml

from zt_legal_analysis_core.infrastructure.llm.base import LLMClient, LLMResponse, RuntimeProfile
from zt_legal_analysis_core.infrastructure.llm.adapters import (
    BedrockAdapter,
    OllamaAdapter,
    _parse_json_from_text,
    _build_structured_prompt,
)
from zt_legal_analysis_core.infrastructure.llm.factory import load_profile, build_client, get_llm_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_profile(directory: Path, name: str, data: dict) -> Path:
    path = directory / f"{name}.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


@pytest.fixture
def tmp_profiles(tmp_path: Path):
    """Directory temporanea con profili YAML di test."""
    return tmp_path


# ---------------------------------------------------------------------------
# RuntimeProfile
# ---------------------------------------------------------------------------

class TestRuntimeProfile:
    def test_from_dict_minimo(self):
        p = RuntimeProfile.from_dict("test", {"backend": "ollama", "model": "qwen:7b"})
        assert p.name == "test"
        assert p.backend == "ollama"
        assert p.model == "qwen:7b"
        assert p.region == "eu-central-1"
        assert p.timeout == 1800.0
        assert p.extra == {}

    def test_from_dict_completo(self):
        data = {
            "backend": "bedrock",
            "model": "qwen.qwen3-235b-a22b-2507-v1:0",
            "region": "eu-west-1",
            "timeout": 600.0,
            "max_tokens": 4000,
            "temperature": 0.0,
            "proxy": "http://proxy:3128",
            "custom_flag": True,
        }
        p = RuntimeProfile.from_dict("prod", data)
        assert p.region == "eu-west-1"
        assert p.proxy == "http://proxy:3128"
        assert p.extra == {"custom_flag": True}

    def test_frozen(self):
        p = RuntimeProfile.from_dict("x", {"backend": "ollama", "model": "m"})
        with pytest.raises((AttributeError, TypeError)):
            p.model = "altro"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------

class TestLLMClientProtocol:
    def test_fake_implementa_protocol(self):
        class FakeClient:
            def generate_text(self, prompt, *, system_prompt=None, temperature=0.1, max_tokens=2000):
                return "risposta"
            def generate_structured(self, prompt, schema, *, system_prompt=None, temperature=0.0):
                return {}

        assert isinstance(FakeClient(), LLMClient)

    def test_incompleto_non_implementa_protocol(self):
        class Incompleto:
            def generate_text(self, prompt, **kw):
                return ""
        # runtime_checkable Protocol verifica solo la presenza di metodi (non firme)
        # Incompleto manca di generate_structured → non è LLMClient
        assert not isinstance(Incompleto(), LLMClient)


# ---------------------------------------------------------------------------
# load_profile
# ---------------------------------------------------------------------------

class TestLoadProfile:
    def test_carica_profilo_valido(self, tmp_profiles):
        _write_profile(tmp_profiles, "mypro", {"backend": "ollama", "model": "qwen:7b"})
        p = load_profile("mypro", profiles_dir=tmp_profiles)
        assert p.model == "qwen:7b"

    def test_file_non_trovato(self, tmp_profiles):
        with pytest.raises(FileNotFoundError):
            load_profile("inesistente", profiles_dir=tmp_profiles)

    def test_manca_campo_backend(self, tmp_profiles):
        _write_profile(tmp_profiles, "no_backend", {"model": "qwen:7b"})
        with pytest.raises(KeyError):
            load_profile("no_backend", profiles_dir=tmp_profiles)

    def test_manca_campo_model(self, tmp_profiles):
        _write_profile(tmp_profiles, "no_model", {"backend": "ollama"})
        with pytest.raises(KeyError):
            load_profile("no_model", profiles_dir=tmp_profiles)

    def test_carica_bedrock(self, tmp_profiles):
        data = {
            "backend": "bedrock",
            "model": "qwen.qwen3-235b-a22b-2507-v1:0",
            "region": "eu-central-1",
        }
        _write_profile(tmp_profiles, "bedrock_managed", data)
        p = load_profile("bedrock_managed", profiles_dir=tmp_profiles)
        assert p.backend == "bedrock"
        assert p.region == "eu-central-1"


# ---------------------------------------------------------------------------
# build_client
# ---------------------------------------------------------------------------

class TestBuildClient:
    def _profile(self, backend, model="m"):
        return RuntimeProfile.from_dict("t", {"backend": backend, "model": model})

    def test_build_ollama(self):
        profile = self._profile("ollama", "qwen:7b")
        with patch("zt_legal_analysis_core.infrastructure.llm.adapters.OllamaClient") as MockCls:
            MockCls.return_value = MagicMock()
            client = build_client(profile)
        assert isinstance(client, OllamaAdapter)

    def test_build_bedrock(self):
        profile = self._profile("bedrock", "qwen.qwen3-235b")
        with patch("zt_legal_analysis_core.infrastructure.llm.adapters.BedrockClient") as MockCls:
            MockCls.return_value = MagicMock()
            client = build_client(profile)
        assert isinstance(client, BedrockAdapter)

    def test_backend_non_supportato(self):
        profile = self._profile("sagemaker")
        with pytest.raises(ValueError, match="Backend non supportato"):
            build_client(profile)


# ---------------------------------------------------------------------------
# _parse_json_from_text helper
# ---------------------------------------------------------------------------

class TestParseJsonHelper:
    def test_oggetto_puro(self):
        result = _parse_json_from_text('{"key": "value", "n": 42}')
        assert result == {"key": "value", "n": 42}

    def test_oggetto_con_testo_attorno(self):
        result = _parse_json_from_text('Ecco il JSON: {"a": 1} grazie.')
        assert result == {"a": 1}

    def test_blocco_markdown(self):
        text = '```json\n{"x": true}\n```'
        result = _parse_json_from_text(text)
        assert result == {"x": True}

    def test_array_json(self):
        result = _parse_json_from_text('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_nessun_json_lancia_value_error(self):
        with pytest.raises(ValueError):
            _parse_json_from_text("Nessun JSON qui")


# ---------------------------------------------------------------------------
# OllamaAdapter (con OllamaClient mockato)
# ---------------------------------------------------------------------------

class TestOllamaAdapter:
    def _make_adapter(self, response_content: str):
        """Costruisce OllamaAdapter con OllamaClient interamente mockato."""
        mock_inner = MagicMock()
        from zt_legal_analysis_core.core.llm_client import LLMResponse as CoreLLMResponse
        mock_inner.chat.return_value = CoreLLMResponse(
            content=response_content,
            model="qwen:7b",
            tokens_used=10,
        )
        with patch("zt_legal_analysis_core.infrastructure.llm.adapters.OllamaClient", return_value=mock_inner):
            adapter = OllamaAdapter(model="qwen:7b")
        adapter._inner = mock_inner
        return adapter

    def test_generate_text(self):
        adapter = self._make_adapter("risposta libera")
        result = adapter.generate_text("prompt")
        assert result == "risposta libera"

    def test_generate_structured_json_valido(self):
        adapter = self._make_adapter('{"importo": 100000, "valuta": "EUR"}')
        schema = {"type": "object", "properties": {"importo": {"type": "number"}}}
        result = adapter.generate_structured("prompt", schema)
        assert result == {"importo": 100000, "valuta": "EUR"}

    def test_generate_structured_json_non_parseable(self):
        adapter = self._make_adapter("testo senza json")
        schema = {"type": "object"}
        with pytest.raises(ValueError):
            adapter.generate_structured("prompt", schema)

    def test_build_structured_prompt_contiene_schema(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        prompt_out = _build_structured_prompt("estrai dati", schema)
        assert "estrai dati" in prompt_out
        assert '"type"' in prompt_out
        assert "JSON" in prompt_out


# ---------------------------------------------------------------------------
# BedrockAdapter (con BedrockClient mockato)
# ---------------------------------------------------------------------------

class TestBedrockAdapter:
    def _make_adapter(self, response_content: str):
        mock_inner = MagicMock()
        from zt_legal_analysis_core.core.llm_client import LLMResponse as CoreLLMResponse
        mock_inner.chat.return_value = CoreLLMResponse(
            content=response_content,
            model="qwen.qwen3-235b",
            tokens_used=50,
        )
        with patch("zt_legal_analysis_core.infrastructure.llm.adapters.BedrockClient", return_value=mock_inner):
            adapter = BedrockAdapter(model="qwen.qwen3-235b")
        adapter._inner = mock_inner
        return adapter

    def test_generate_text(self):
        adapter = self._make_adapter("output bedrock")
        assert adapter.generate_text("p") == "output bedrock"

    def test_generate_structured(self):
        adapter = self._make_adapter('{"stato": "attivo"}')
        result = adapter.generate_structured("p", {})
        assert result == {"stato": "attivo"}

    def test_generate_structured_schema_nel_prompt(self):
        adapter = self._make_adapter('{"ok": true}')
        schema = {"properties": {"ok": {"type": "boolean"}}}
        adapter.generate_structured("descrivi", schema)
        call_args = adapter._inner.chat.call_args
        prompt_inviato = call_args[1].get("prompt") or call_args[0][0]
        assert "ok" in prompt_inviato


# ---------------------------------------------------------------------------
# Factory: variabile d'ambiente
# ---------------------------------------------------------------------------

class TestGetLlmClientEnv:
    def test_env_override(self, tmp_profiles):
        _write_profile(tmp_profiles, "test_env", {"backend": "ollama", "model": "llama3"})
        with patch.dict(os.environ, {"LLM_RUNTIME_PROFILE": "test_env"}):
            with patch("zt_legal_analysis_core.infrastructure.llm.adapters.OllamaClient") as MockCls:
                MockCls.return_value = MagicMock()
                client = get_llm_client(profiles_dir=tmp_profiles)
        assert isinstance(client, OllamaAdapter)

    def test_nome_esplicito_ha_precedenza_su_env(self, tmp_profiles):
        _write_profile(tmp_profiles, "explicit", {"backend": "ollama", "model": "m1"})
        _write_profile(tmp_profiles, "env_pro", {"backend": "ollama", "model": "m2"})
        with patch.dict(os.environ, {"LLM_RUNTIME_PROFILE": "env_pro"}):
            with patch("zt_legal_analysis_core.infrastructure.llm.adapters.OllamaClient") as MockCls:
                MockCls.return_value = MagicMock()
                client = get_llm_client("explicit", profiles_dir=tmp_profiles)
        assert client.model == "m1"
