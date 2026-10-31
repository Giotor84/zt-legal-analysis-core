# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
from __future__ import annotations

from typing import Any, Dict, List

from zt_legal_analysis_core.domain.services.financial_logic import normalize_importi_payload


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def evaluate_legal_decision(entity_data: Dict[str, Any]) -> Dict[str, Any]:
    data = _as_dict(entity_data)

    debitore = _as_dict(data.get("debitore", {}))
    importi = normalize_importi_payload(data.get("importi", {}), _as_dict(data.get("credito", {})))
    garanzie = _as_dict(data.get("garanzie", {}))
    garanti = _as_list(data.get("garanti", []))
    procedure = _as_dict(data.get("procedure_concorsuali", {}))

    has_debtor_identity = bool(debitore.get("nome") or debitore.get("codice_fiscale") or debitore.get("partita_iva"))
    has_importo = bool((importi.get("totale") or 0) > 0)
    debtor_channel = has_debtor_identity and has_importo

    signed_guarantors = [
        g for g in garanti
        if isinstance(g, dict) and bool(
            g.get("fideiussione_firmata") or g.get("firma_presente") or g.get("da_fideiussione")
        )
    ]
    guarantor_channel = len(signed_guarantors) > 0

    real_guarantees_channel = bool(_as_list(garanzie.get("ipoteche", [])) or _as_list(garanzie.get("pegni", [])))
    concorsuale = bool(procedure.get("rilevata"))

    actionable_channels: List[str] = []
    if debtor_channel:
        actionable_channels.append("debitore")
    if guarantor_channel:
        actionable_channels.append("garanti")
    if real_guarantees_channel:
        actionable_channels.append("garanzie_reali")

    rule_applied = "RULE_NO_CHANNELS"
    if concorsuale and not guarantor_channel:
        suggested = "BLOCCO_CONCORSUALE"
        rule_applied = "RULE_CONCORSUALE_BLOCK"
    elif guarantor_channel:
        suggested = "CONDIZIONATA_GARANTI"
        rule_applied = "RULE_GUARANTOR_CHANNEL"
    elif debtor_channel:
        suggested = "SI"
        rule_applied = "RULE_DEBTOR_CHANNEL"
    elif real_guarantees_channel:
        suggested = "CONDIZIONATA_GARANZIE"
        rule_applied = "RULE_REAL_GUARANTEE_CHANNEL"
    else:
        suggested = "NO"

    guardrail_recommendations: List[str] = []
    if guarantor_channel:
        guardrail_recommendations.append("Prioritizzare azione verso garanti/fideiussori con firma verificata.")
    if concorsuale:
        guardrail_recommendations.append("Mantenere tracciata l'insinuazione al passivo e coordinare strategia con eventuale azione sui garanti.")
    if not has_importo:
        guardrail_recommendations.append("Validare importo totale con evidenza documentale prima di decisione definitiva.")

    decision_explanation_parts: List[str] = [
        f"Regola applicata: {rule_applied}.",
        f"Canali azionabili: {', '.join(actionable_channels) if actionable_channels else 'nessuno'}.",
        f"Concorsuale: {'si' if concorsuale else 'no'}.",
        f"Garanti firmati: {len(signed_guarantors)}.",
        f"Importo totale presente: {'si' if has_importo else 'no'}.",
    ]

    return {
        "actionable_channels": actionable_channels,
        "suggested_azionabilita": suggested,
        "concorsuale": concorsuale,
        "signed_guarantors_count": len(signed_guarantors),
        "guardrail_recommendations": guardrail_recommendations,
        "rule_applied": rule_applied,
        "decision_explanation": " ".join(decision_explanation_parts),
    }
