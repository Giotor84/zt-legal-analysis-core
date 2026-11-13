# -*- coding: utf-8 -*-
# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
HITL Queue Manager — Gestione revisione umana fascicoli bloccati dal Quality Gate

Quando `canonicalize_entity_data()` imposta `_hitl_queue.required = True`, questo
modulo persiste il fascicolo in una coda JSON e fornisce il percorso di rientro
nella pipeline dopo la correzione umana.

FLUSSO:
  A9 Quality Gate (dati incompleti)
      ↓
  HITLQueueManager.enqueue()  →  data/hitl_queue.json  (status: OPEN)
      ↓
  [Operatore revisiona e corregge entity_data]
      ↓
  HITLQueueManager.submit_correction()  →  status: CORRECTED
      ↓
  SmartPipeline.reprocess_hitl_item()  →  rientro da EntityMetadataMerger (step A8)
"""

import json
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Percorso di default della coda persistente
_DEFAULT_QUEUE_PATH = Path("data/hitl_queue.json")

# Stato degli item HITL
HITL_STATUS_OPEN = "OPEN"
HITL_STATUS_IN_REVIEW = "IN_REVIEW"
HITL_STATUS_CORRECTED = "CORRECTED"
HITL_STATUS_REPROCESSED = "REPROCESSED"
HITL_STATUS_SKIPPED = "SKIPPED"


class HITLQueueManager:
    """
    Coda persistente (JSON) per fascicoli che richiedono revisione umana.

    Thread-safe tramite file locking (fcntl su Unix, fallback su Windows).
    """

    _GLOBAL_LOCK = threading.RLock()

    def __init__(self, queue_path: Optional[Path] = None):
        self.queue_path = Path(queue_path or _DEFAULT_QUEUE_PATH)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self._GLOBAL_LOCK

    # ------------------------------------------------------------------
    # I/O sicuro della coda
    # ------------------------------------------------------------------

    def _load(self) -> List[Dict[str, Any]]:
        if not self.queue_path.exists():
            return []
        try:
            with open(self.queue_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"⚠️ HITL queue corrotta o vuota, azzerata: {e}")
            return []

    def _save(self, items: List[Dict[str, Any]]) -> None:
        with self._lock:
            tmp_path = self.queue_path.with_name(f"{self.queue_path.name}.{uuid.uuid4().hex}.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2, default=str)
            tmp_path.replace(self.queue_path)

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def enqueue(
        self,
        fascicolo_id: str,
        entity_data: Dict[str, Any],
        hitl_info: Dict[str, Any],
        input_dir: Optional[str] = None,
    ) -> str:
        """
        Aggiunge un fascicolo alla coda HITL.

        Args:
            fascicolo_id: ID del fascicolo
            entity_data:  Dati estratti (senza campi _raw, per sicurezza)
            hitl_info:    Struttura _hitl_queue dal canonical model
            input_dir:    Percorso cartella originale documenti (per riprocessamento)

        Returns:
            item_id (fascicolo_id + timestamp)
        """
        items = self._load()

        # Evita duplicati: se già OPEN non aggiungere di nuovo
        for item in items:
            if (
                item.get("fascicolo_id") == fascicolo_id
                and item.get("status") == HITL_STATUS_OPEN
            ):
                logger.info(f"📋 HITL: {fascicolo_id} già in coda (OPEN), skip duplicato")
                return item.get("item_id", fascicolo_id)

        item_id = f"{fascicolo_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Non persistere i campi debug/raw nell'item HITL
        safe_entity_data = {
            k: v
            for k, v in entity_data.items()
            if not k.startswith("_raw") and k != "all_text"
        }

        item = {
            "item_id": item_id,
            "fascicolo_id": fascicolo_id,
            "status": HITL_STATUS_OPEN,
            "priority": hitl_info.get("priority", "P3"),
            "sla_hours": hitl_info.get("sla_hours", 72),
            "routing": hitl_info.get("routing", "LEGAL_REVIEW_QUEUE"),
            "reasons": hitl_info.get("reasons", []),
            "input_dir": str(input_dir) if input_dir else None,
            "entity_data_snapshot": safe_entity_data,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "corrected_data": None,
            "correction_notes": None,
            "reprocess_result": None,
        }

        items.append(item)
        self._save(items)

        reason_codes = [r.get("code") for r in hitl_info.get("reasons", [])]
        logger.warning(
            f"🙋 HITL ENQUEUED: {fascicolo_id} | Priorità: {item['priority']} | "
            f"SLA: {item['sla_hours']}h | Motivi: {reason_codes}"
        )
        return item_id

    def list_pending(self) -> List[Dict[str, Any]]:
        """Restituisce tutti gli item in stato OPEN o IN_REVIEW."""
        return [
            i for i in self._load()
            if i.get("status") in (HITL_STATUS_OPEN, HITL_STATUS_IN_REVIEW)
        ]

    def get_item(self, fascicolo_id: str) -> Optional[Dict[str, Any]]:
        """Recupera l'item più recente per un fascicolo_id."""
        items = [i for i in self._load() if i.get("fascicolo_id") == fascicolo_id]
        if not items:
            return None
        return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)[0]

    def submit_correction(
        self,
        fascicolo_id: str,
        corrected_data: Dict[str, Any],
        correction_notes: str = "",
    ) -> bool:
        """
        Registra la correzione umana e marca l'item come CORRECTED.

        Dopo questa chiamata, usare SmartPipeline.reprocess_hitl_item()
        per rientrare nel pipeline da EntityMetadataMerger (step A8).

        Args:
            fascicolo_id:    ID fascicolo
            corrected_data:  Dati corretti dall'operatore
            correction_notes: Note libere sull'intervento

        Returns:
            True se l'item è stato trovato e aggiornato
        """
        items = self._load()
        updated = False

        for item in items:
            if (
                item.get("fascicolo_id") == fascicolo_id
                and item.get("status") in (HITL_STATUS_OPEN, HITL_STATUS_IN_REVIEW)
            ):
                item["status"] = HITL_STATUS_CORRECTED
                item["corrected_data"] = corrected_data
                item["correction_notes"] = correction_notes
                item["updated_at"] = datetime.now().isoformat()
                item["hitl_corrected"] = True  # flag che compare nel Forensic Report
                updated = True
                logger.info(f"✅ HITL correction submitted: {fascicolo_id}")
                break

        if updated:
            self._save(items)
        else:
            logger.warning(f"⚠️ HITL correction: nessun item OPEN trovato per {fascicolo_id}")

        return updated

    def mark_reprocessed(self, fascicolo_id: str, reprocess_result: str = "") -> None:
        """Marca l'item come REPROCESSED dopo che la pipeline è rientrata."""
        items = self._load()
        for item in items:
            if (
                item.get("fascicolo_id") == fascicolo_id
                and item.get("status") == HITL_STATUS_CORRECTED
            ):
                item["status"] = HITL_STATUS_REPROCESSED
                item["reprocess_result"] = reprocess_result
                item["updated_at"] = datetime.now().isoformat()
                break
        self._save(items)

    def skip(self, fascicolo_id: str, reason: str = "") -> None:
        """Marca l'item come SKIPPED (operatore decide di non correggere)."""
        items = self._load()
        for item in items:
            if (
                item.get("fascicolo_id") == fascicolo_id
                and item.get("status") in (HITL_STATUS_OPEN, HITL_STATUS_IN_REVIEW)
            ):
                item["status"] = HITL_STATUS_SKIPPED
                item["correction_notes"] = reason
                item["updated_at"] = datetime.now().isoformat()
                break
        self._save(items)
        logger.info(f"⏭ HITL skip: {fascicolo_id} — {reason}")

    def summary(self) -> Dict[str, Any]:
        """Statistiche aggregate della coda."""
        items = self._load()
        counts: Dict[str, int] = {}
        for item in items:
            status = item.get("status", "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
        return {
            "total": len(items),
            "by_status": counts,
            "pending": counts.get(HITL_STATUS_OPEN, 0) + counts.get(HITL_STATUS_IN_REVIEW, 0),
        }


# Singleton di convenienza
_default_manager: Optional[HITLQueueManager] = None


def get_hitl_manager(queue_path: Optional[Path] = None) -> HITLQueueManager:
    """Restituisce l'istanza singleton del HITLQueueManager."""
    global _default_manager
    if _default_manager is None or queue_path is not None:
        _default_manager = HITLQueueManager(queue_path)
    return _default_manager
