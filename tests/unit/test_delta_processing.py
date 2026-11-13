# -*- coding: utf-8 -*-
"""
tests/unit/test_delta_processing.py

S3.4 — Delta Processing: verifica che il meccanismo di hash dedup
carichi correttamente i seen_hashes da un checkpoint COMPLETATO
quando delta_mode=True, impedendo la rielaborazione di documenti già processati.

I test validano:
  1. CheckpointManager persiste e ricarica seen_hashes nella struttura attesa
  2. Un checkpoint con last_step=report_generated è riconosciuto come "completato"
  3. Il CheckpointManager NON restituisce can_resume=True per un checkpoint completo
     (allineamento con la logica delta vs resume)
"""
from pathlib import Path
from typing import Any, Dict

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_checkpoint_manager(tmp_path: Path):
    """CheckpointManager isolato su directory temporanea."""
    from zt_legal_analysis_core.core.checkpoint_manager import CheckpointManager
    return CheckpointManager(checkpoint_dir=str(tmp_path / "checkpoints"))


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaProcessingCheckpoint:
    """Verifica il fondamento del delta mode: persist e reload degli hash."""

    _FAKE_HASHES = {"abc123", "def456", "ghi789"}

    def _save_completed_checkpoint(self, mgr, fascicolo_id: str) -> None:
        """Salva un checkpoint che simula un run completato con seen_hashes."""
        # step 'documents_found' contiene la lista degli hash già processati
        mgr.save(
            fascicolo_id,
            "documents_found",
            {
                "files": ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
                "count": 3,
                "seen_hashes": list(self._FAKE_HASHES),
            },
            force_flush=True,
        )
        # step finale per marcare il run come completato
        mgr.save(
            fascicolo_id,
            "report_generated",
            {"output_path": "/tmp/report.pdf"},
            force_flush=True,
        )

    def test_seen_hashes_persisted_and_reloaded(self, tmp_checkpoint_manager):
        """
        Dopo aver salvato un checkpoint con seen_hashes,
        il reload deve restituire gli stessi hash nel campo 'data.documents_found'.
        """
        fid = "FASC-DELTA-001"
        self._save_completed_checkpoint(tmp_checkpoint_manager, fid)

        cp = tmp_checkpoint_manager.load(fid)

        assert cp is not None, "Checkpoint non trovato dopo il salvataggio"
        doc_found = cp.get("data", {}).get("documents_found", {})
        loaded_hashes = set(doc_found.get("seen_hashes", []))

        assert loaded_hashes == self._FAKE_HASHES, (
            f"seen_hashes attesi {self._FAKE_HASHES}, trovati {loaded_hashes}"
        )

    def test_completed_checkpoint_has_correct_last_step(self, tmp_checkpoint_manager):
        """
        Un checkpoint che ha completato 'report_generated' deve avere
        last_step == 'report_generated'.
        """
        fid = "FASC-DELTA-002"
        self._save_completed_checkpoint(tmp_checkpoint_manager, fid)

        cp = tmp_checkpoint_manager.load(fid)

        assert cp is not None
        assert cp.get("last_step") == "report_generated", (
            f"last_step atteso 'report_generated', trovato: {cp.get('last_step')}"
        )

    def test_can_resume_returns_false_for_completed(self, tmp_checkpoint_manager):
        """
        can_resume() deve restituire False per un fascicolo completato:
        il delta mode non usa resume ma sovrascrive con il run successivo.
        """
        fid = "FASC-DELTA-003"
        self._save_completed_checkpoint(tmp_checkpoint_manager, fid)

        result = tmp_checkpoint_manager.can_resume(fid)

        assert result is False, (
            "can_resume() deve essere False per un checkpoint completato"
        )

    def test_delta_hashes_cover_expected_set(self, tmp_checkpoint_manager):
        """
        Verifica che la logica di estrazione degli hash da un checkpoint completato
        funzioni come ci si aspetta dal processo delta_mode di process_fascicolo().
        """
        fid = "FASC-DELTA-004"
        self._save_completed_checkpoint(tmp_checkpoint_manager, fid)

        # Replica la logica di process_fascicolo con delta_mode=True
        prior_cp = tmp_checkpoint_manager.load(fid)
        prior_hashes = set()
        if prior_cp and prior_cp.get("last_step") == "report_generated":
            prior_hashes = set(
                (prior_cp.get("data", {}) or {})
                .get("documents_found", {})
                .get("seen_hashes", []) or []
            )

        # I tre hash salvati devono tutti essere in prior_hashes
        for h in self._FAKE_HASHES:
            assert h in prior_hashes, (
                f"Hash '{h}' non trovato in prior_hashes estratti dal checkpoint"
            )

    def test_no_checkpoint_gives_empty_hashes(self, tmp_checkpoint_manager):
        """
        Se non esiste un checkpoint precedente, prior_hashes deve essere vuoto
        (il processo procederà con elaborazione completa).
        """
        prior_cp = tmp_checkpoint_manager.load("NON-ESISTE")
        prior_hashes: set = set()
        if prior_cp and prior_cp.get("last_step") == "report_generated":
            prior_hashes = set(
                (prior_cp.get("data", {}) or {})
                .get("documents_found", {})
                .get("seen_hashes", []) or []
            )

        assert prior_hashes == set(), (
            f"Senza checkpoint atteso insieme vuoto, trovato: {prior_hashes}"
        )
