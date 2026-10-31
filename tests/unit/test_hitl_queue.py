# -*- coding: utf-8 -*-
"""
tests/unit/test_hitl_queue.py

Scenari testati:
  - Lifecycle completo: enqueue → submit_correction → mark_reprocessed
  - Dedup: doppio enqueue stesso fascicolo_id non duplica la coda
  - submit_correction restituisce False se fascicolo non è in coda
  - summary() riflette correttamente gli stati
"""
from pathlib import Path
from typing import Any, Dict

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_ENTITY_DATA: Dict[str, Any] = {
    "debitore": {"nome": "Marco Neri", "codice_fiscale": "NRIMRC70D10H501W"},
    "importi": {"totale": 50_000.0},
}

_SAMPLE_HITL_INFO: Dict[str, Any] = {
    "required": True,
    "priority": "P1",
    "sla_hours": 24,
    "routing": "LEGAL_REVIEW_QUEUE",
    "reasons": [{"code": "HITL001", "description": "Dati incompleti debitore"}],
}


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHITLQueueLifecycle:
    """Ciclo di vita completo di un item HITL."""

    def test_enqueue_creates_item(self, hitl_manager):
        """enqueue() deve creare un item con status OPEN."""
        from zt_legal_analysis_core.core.hitl_queue import HITL_STATUS_OPEN

        item_id = hitl_manager.enqueue(
            fascicolo_id="FASC-001",
            entity_data=_SAMPLE_ENTITY_DATA,
            hitl_info=_SAMPLE_HITL_INFO,
        )

        assert item_id is not None
        item = hitl_manager.get_item("FASC-001")
        assert item is not None
        assert item["status"] == HITL_STATUS_OPEN

    def test_submit_correction_sets_corrected_status(self, hitl_manager):
        """submit_correction() deve portare l'item in stato CORRECTED."""
        from zt_legal_analysis_core.core.hitl_queue import HITL_STATUS_CORRECTED

        hitl_manager.enqueue("FASC-002", _SAMPLE_ENTITY_DATA, _SAMPLE_HITL_INFO)

        corrected = _SAMPLE_ENTITY_DATA.copy()
        corrected["debitore"] = {"nome": "Marco Neri Corrected", "codice_fiscale": "NRIMRC70D10H501W"}

        success = hitl_manager.submit_correction(
            fascicolo_id="FASC-002",
            corrected_data=corrected,
            correction_notes="Nome verificato su visura camerale",
        )

        assert success is True
        item = hitl_manager.get_item("FASC-002")
        assert item["status"] == HITL_STATUS_CORRECTED
        assert item["corrected_data"]["debitore"]["nome"] == "Marco Neri Corrected"

    def test_mark_reprocessed_sets_final_status(self, hitl_manager):
        """mark_reprocessed() deve portare l'item in stato REPROCESSED."""
        from zt_legal_analysis_core.core.hitl_queue import HITL_STATUS_REPROCESSED

        hitl_manager.enqueue("FASC-003", _SAMPLE_ENTITY_DATA, _SAMPLE_HITL_INFO)
        hitl_manager.submit_correction("FASC-003", _SAMPLE_ENTITY_DATA)
        hitl_manager.mark_reprocessed("FASC-003", reprocess_result="OK - report generato")

        item = hitl_manager.get_item("FASC-003")
        assert item["status"] == HITL_STATUS_REPROCESSED
        assert "OK - report generato" in item.get("reprocess_result", "")

    def test_full_lifecycle_pending_count(self, hitl_manager):
        """
        Dopo enqueue: 1 pending.
        Dopo submit_correction: ancora 0 pending (stato CORRECTED non è pending).
        """
        hitl_manager.enqueue("FASC-004", _SAMPLE_ENTITY_DATA, _SAMPLE_HITL_INFO)
        assert len(hitl_manager.list_pending()) == 1

        hitl_manager.submit_correction("FASC-004", _SAMPLE_ENTITY_DATA)
        assert len(hitl_manager.list_pending()) == 0


class TestHITLQueueDedup:
    """Verifica deduplicazione: doppio enqueue non crea duplicati."""

    def test_double_enqueue_same_fascicolo_is_deduped(self, hitl_manager):
        """Due enqueue per lo stesso fascicolo_id non devono creare due item OPEN."""
        hitl_manager.enqueue("FASC-DUP", _SAMPLE_ENTITY_DATA, _SAMPLE_HITL_INFO)
        hitl_manager.enqueue("FASC-DUP", _SAMPLE_ENTITY_DATA, _SAMPLE_HITL_INFO)

        all_items = [
            i for i in hitl_manager._load()
            if i.get("fascicolo_id") == "FASC-DUP"
        ]
        assert len(all_items) == 1, (
            f"Duplicati non prevenuti: trovati {len(all_items)} item per FASC-DUP"
        )


class TestHITLQueueEdgeCases:
    """Edge case: correzione su fascicolo inesistente."""

    def test_submit_correction_returns_false_for_unknown_fascicolo(self, hitl_manager):
        """submit_correction su fascicolo non in coda deve restituire False."""
        result = hitl_manager.submit_correction(
            fascicolo_id="NON-ESISTE",
            corrected_data={},
        )
        assert result is False

    def test_summary_reflects_statuses(self, hitl_manager):
        """summary() deve contare correttamente item per stato."""
        hitl_manager.enqueue("FASC-S1", _SAMPLE_ENTITY_DATA, _SAMPLE_HITL_INFO)
        hitl_manager.enqueue("FASC-S2", _SAMPLE_ENTITY_DATA, _SAMPLE_HITL_INFO)
        hitl_manager.submit_correction("FASC-S1", _SAMPLE_ENTITY_DATA)

        s = hitl_manager.summary()
        assert s["total"] == 2
        assert s["pending"] == 1  # FASC-S2 è ancora OPEN
        assert s["by_status"].get("OPEN", 0) == 1
        assert s["by_status"].get("CORRECTED", 0) == 1
