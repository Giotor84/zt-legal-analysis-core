# -*- coding: utf-8 -*-
"""
conftest.py — Pytest fixtures condivise per la test suite NPL AI Platform

Fixture disponibili:
- tmp_queue_path     : Path temporanea per la coda HITL (isolata per test)
- hitl_manager       : HITLQueueManager istanziato su tmp_queue_path
- minimal_entity_data: Dict minimo valido per EntityExtractor / LegalAnalyzer
- mock_llm_ok        : LLMClient mock che restituisce JSON valido
- mock_llm_bad_json  : LLMClient mock che restituisce testo non-JSON
- document_data_with_prescrizione: document_data con data_ultimo_atto 11 anni fa
- document_data_with_guarantor   : entity_data con garante firmato
"""
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _FakeLLMResponse:
    """Simula la struttura di LLMResponse del client Ollama."""
    def __init__(self, content: str):
        self.content = content
        self.model = "fake-model"
        self.tokens_used = 0
        self.latency_ms = 0


def _make_mock_llm(content: str) -> MagicMock:
    """Crea un LLMClient mock che restituisce `content` su chat()."""
    llm = MagicMock()
    llm.chat.return_value = _FakeLLMResponse(content)
    llm.model = "fake-model"
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: HITL Queue
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_queue_path(tmp_path: Path) -> Path:
    """Percorso JSON temporaneo per la coda HITL — isolato per ogni test."""
    return tmp_path / "hitl_queue.json"


@pytest.fixture
def hitl_manager(tmp_queue_path: Path):
    """HITLQueueManager istanziato su coda temporanea vuota."""
    from zt_legal_analysis_core.core.hitl_queue import HITLQueueManager
    return HITLQueueManager(queue_path=tmp_queue_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Entity data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_entity_data() -> Dict[str, Any]:
    """Dati entità minimi per evitare KeyError nei moduli."""
    return {
        "debitore": {
            "nome": "Mario Rossi",
            "codice_fiscale": "RSSMRA80A01H501X",
        },
        "importi": {
            "totale": 150_000.0,
            "capitale": 100_000.0,
            "interessi": 50_000.0,
        },
        "garanti": [],
        "garanzie": {},
        "procedure_concorsuali": {"rilevata": False},
        "checklist_recupero": {},
    }


@pytest.fixture
def document_data_with_prescrizione() -> Dict[str, Any]:
    """
    document_data con data_ultimo_atto_interruttivo di 11 anni fa,
    sufficiente a scatenare CL013 (soglia: 10 anni).
    """
    data_11_anni_fa = (date.today() - timedelta(days=11 * 366)).strftime("%Y-%m-%d")
    return {
        "tipo_documento": "contratto_mutuo",
        "data_ultimo_atto_interruttivo": data_11_anni_fa,
        "tasso_interesse": 4.5,
        "text": "",
        "firma": {"presente": True},
        "clausole": [],
        "atti_interruttivi": [],
    }


@pytest.fixture
def document_data_with_guarantor() -> Dict[str, Any]:
    """entity_data con un garante che ha fideiussione_firmata=True."""
    return {
        "debitore": {
            "nome": "Luigi Verdi",
            "codice_fiscale": "VRDLGU70B15F205X",
        },
        "importi": {
            "totale": 80_000.0,
            "capitale": 60_000.0,
            "interessi": 20_000.0,
        },
        "garanti": [
            {
                "nome": "Anna Bianchi",
                "codice_fiscale": "BNCNNA75C55H501Z",
                "fideiussione_firmata": True,
                "importo_massimale": 80_000.0,
            }
        ],
        "garanzie": {},
        "procedure_concorsuali": {"rilevata": False},
        "checklist_recupero": {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: Mock LLM
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_ok() -> MagicMock:
    """Mock LLM che restituisce un JSON EntityExtractor valido ma minimo."""
    payload = json.dumps({
        "debitore": {"nome": "Test Debitore", "codice_fiscale": "TSTDBT80A01H501A"},
        "credito": {"importo_totale": 100000.0},
        "garanti": [],
        "garanzie": {},
        "checklist_recupero": {},
        "_meta": {"extraction_method_debitore": "llm"},
    })
    return _make_mock_llm(payload)


@pytest.fixture
def mock_llm_bad_json() -> MagicMock:
    """
    Mock LLM che restituisce testo con JSON malformato (parentesi non chiuse).
    L'EntityExtractor deve ricorrere all'estrazione regex fallback.
    """
    bad_content = (
        'Ecco i dati estratti dal documento:\n'
        'debitore: Mario Rossi\n'
        'codice fiscale: RSSMRA80A01H501X\n'
        '{ "debitore": { "nome": "Mario Rossi", "codice_fiscale": "RSSMRA80A01H501X" '
        # JSON troncato — no closing }
    )
    return _make_mock_llm(bad_content)
