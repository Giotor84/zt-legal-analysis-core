# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
🚦 DATA QUALITY GATE

IL SEMAFORO INTELLIGENTE per NPL Analysis.
Valuta se i dati estratti sono sufficienti per procedere alla Fase 2 (Strategia).

Non blocca il processo, ma etichetta il fascicolo come:
- GREEN (Procedere a spron battuto)
- YELLOW (Procedere con cautela, dati mancanti)
- RED (Fermarsi: dati critici assenti)

Scoring System (0-100):
- GBV presente: 40 punti (CRITICO)
- Debitore identificato: 20 punti
- Gold Source (Welcome Letter/Decreto): 15 punti
- Atti legali presenti: 15 punti
- Garanti presenti: 10 punti (BONUS)
"""

import logging
from typing import Dict, Any, List

from zt_legal_analysis_core.domain.services.financial_logic import normalize_importi_payload

logger = logging.getLogger(__name__)


class DataQualityGate:
    """
    IL SEMAFORO 🚦
    Valuta se i dati estratti sono sufficienti per procedere alla Fase 2 (Strategia).
    Non blocca il processo, ma etichetta il fascicolo come:
    - GREEN (Procedere a spron battuto)
    - YELLOW (Procedere con cautela, dati mancanti)
    - RED (Fermarsi: dati critici assenti)
    """

    def __init__(self):
        # Pesi per il calcolo dello score (Totale 100)
        self.weights = {
            "gbv_present": 40,        # Senza soldi non si canta messa
            "debtor_id": 20,          # Chi è il debitore?
            "gold_source": 15,        # Abbiamo una Welcome Letter/Decreto?
            "docs_legal": 15,         # Ci sono atti legali?
            "guarantor": 10           # C'è un garante? (Bonus)
        }

    def assess_quality(self, fascicolo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analizza il JSON del fascicolo e restituisce il verdetto.
        
        Args:
            fascicolo_data: Dictionary contenente i dati del fascicolo
                Required keys:
                - financial_summary: {total_gbv, confidence}
                - anagrafica: {debitori, garanti}
                - documents: [{type, category}]
        
        Returns:
            Dictionary con:
            - status: "GREEN" | "YELLOW" | "RED"
            - score: 0-100
            - gbv_detected: float
            - issues: List[str] (warning non bloccanti)
            - critical_errors: List[str] (errori bloccanti)
            - actionable: bool (se True, processa con LLM)
        """
        score = 0
        issues = []
        critical_missing = []
        
        # 1. CHECK GBV (Il cuore del problema)
        # Prova prima financial_summary (formato vecchio), poi entity_data.importi (formato nuovo)
        gbv = fascicolo_data.get("financial_summary", {}).get("total_gbv", 0)
        if gbv == 0:
            # Formato nuovo: cerca in entity_data.importi
            entity_data = fascicolo_data.get("entity_data", {}) if isinstance(fascicolo_data.get("entity_data"), dict) else {}
            importi_norm = normalize_importi_payload(
                entity_data.get("importi", {}),
                entity_data.get("credito", {}) if isinstance(entity_data.get("credito"), dict) else {},
            )
            gbv = float(importi_norm.get("totale") or importi_norm.get("capitale") or 0)
        
        if gbv > 1000:  # Ignoriamo debiti irrisori sotto 1k
            score += self.weights["gbv_present"]
            logger.info(f"✅ GBV presente: EUR {gbv:,.2f} (+{self.weights['gbv_present']} punti)")
        else:
            critical_missing.append("GBV Mancante o Irrisorio")
            logger.warning(f"❌ GBV mancante o troppo basso: EUR {gbv:,.2f}")

        # 2. CHECK DEBITORE (Anagrafica)
        # Prova prima anagrafica.debitori (formato vecchio), poi entity_data.debitore (formato nuovo)
        debtor = fascicolo_data.get("anagrafica", {}).get("debitori", [])
        if not debtor:
            # Formato nuovo: debitore è singolo non array
            deb_single = fascicolo_data.get("entity_data", {}).get("debitore")
            if deb_single:
                debtor = [deb_single] if isinstance(deb_single, dict) else [{"nome": str(deb_single)}]
        
        if debtor and len(debtor) > 0:
            # Controllo se c'è almeno un nome o CF
            d_name = debtor[0].get("nome", "") if isinstance(debtor[0], dict) else str(debtor[0])
            d_cf = debtor[0].get("cf", "") if isinstance(debtor[0], dict) else ""
            if d_name or d_cf:
                score += self.weights["debtor_id"]
                logger.info(f"✅ Debitore identificato: {d_name or d_cf} (+{self.weights['debtor_id']} punti)")
            else:
                critical_missing.append("Dati Debitore Incompleti")
                logger.warning("⚠️ Debitore presente ma senza nome/CF")
        else:
            critical_missing.append("Nessun Debitore Trovato")
            logger.warning("❌ Nessun debitore trovato")

        # 3. CHECK FONTE GOLD (La gerarchia delle fonti)
        # Controlliamo se tra i documenti c'è una fonte 'GOLD' (Welcome Letter, Decreto)
        docs = fascicolo_data.get("documents", [])
        has_gold = any(d.get("type") in ["Welcome Letter", "Decreto Ingiuntivo", "Precetto"] for d in docs)
        
        # O verifica se il GBV deriva da una fonte sicura (confidence > 0.9)
        confidence = fascicolo_data.get("financial_summary", {}).get("confidence", 0)
        if has_gold or confidence > 0.9:
            score += self.weights["gold_source"]
            if has_gold:
                gold_types = [d.get("type") for d in docs if d.get("type") in ["Welcome Letter", "Decreto Ingiuntivo", "Precetto"]]
                logger.info(f"✅ Gold Source presente: {', '.join(gold_types)} (+{self.weights['gold_source']} punti)")
            else:
                logger.info(f"✅ High confidence GBV: {confidence:.2f} (+{self.weights['gold_source']} punti)")
        else:
            issues.append("Manca Fonte Certificata (Gold Source)")
            logger.warning("⚠️ Nessuna Gold Source (Welcome Letter/Decreto/Precetto)")

        # 4. CHECK ATTI LEGALI
        has_legal = any(d.get("category") == "legal" for d in docs)
        if has_legal:
            score += self.weights["docs_legal"]
            legal_count = sum(1 for d in docs if d.get("category") == "legal")
            logger.info(f"✅ Atti legali presenti: {legal_count} doc (+{self.weights['docs_legal']} punti)")
        else:
            issues.append("Nessun Atto Legale rilevato")
            logger.info("ℹ️ Nessun atto legale rilevato (non critico)")

        # 5. CHECK GARANTI (Opzionale ma prezioso)
        guarantors = fascicolo_data.get("anagrafica", {}).get("garanti", [])
        if guarantors:
            score += self.weights["guarantor"]
            logger.info(f"✅ Garanti presenti: {len(guarantors)} (+{self.weights['guarantor']} punti)")

        # --- VERDETTO FINALE ---
        status = "RED"
        if score >= 80:
            status = "GREEN"
        elif score >= 50:
            status = "YELLOW"
        
        # Override: Se manca il GBV è SEMPRE ROSSO
        if "GBV Mancante o Irrisorio" in critical_missing:
            status = "RED"
            score = 0
            logger.error("🔴 Override: GBV mancante → STATUS FORZATO A RED")

        logger.info(f"🚦 VERDETTO FINALE: {status} (Score: {score}/100)")

        return {
            "status": status,          # GREEN, YELLOW, RED
            "score": score,            # 0-100
            "gbv_detected": gbv,
            "issues": issues,          # Warning non bloccanti
            "critical_errors": critical_missing,  # Errori bloccanti
            "actionable": status != "RED"  # Se True, processa con LLM
        }

    def assess_from_json_file(self, json_path: str) -> Dict[str, Any]:
        """
        Carica un JSON da file e valuta la qualità.
        
        Args:
            json_path: Path al file JSON del fascicolo
            
        Returns:
            Dictionary con verdict (come assess_quality)
        """
        import json
        from pathlib import Path
        
        path = Path(json_path)
        if not path.exists():
            logger.error(f"File non trovato: {json_path}")
            return {
                "status": "RED",
                "score": 0,
                "gbv_detected": 0,
                "issues": [],
                "critical_errors": [f"File non trovato: {json_path}"],
                "actionable": False,
            }

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"📂 Caricato: {path.name}")
            return self.assess_quality(data)

        except Exception as e:
            logger.error(f"Errore lettura JSON: {e}")
            return {
                "status": "RED",
                "score": 0,
                "gbv_detected": 0,
                "issues": [],
                "critical_errors": [f"Errore parsing JSON: {str(e)}"],
                "actionable": False
            }
