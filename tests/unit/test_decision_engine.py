# -*- coding: utf-8 -*-
"""
tests/unit/test_decision_engine.py

Scenari testati:
  - entity_data con garante firmato → rule_applied == "RULE_GUARANTOR_CHANNEL"
  - entity_data senza garanti ma con debitore e importo → "RULE_DEBTOR_CHANNEL"
  - entity_data vuoto → "RULE_NO_CHANNELS"
  - entity_data con procedura concorsuale e senza garante → "RULE_CONCORSUALE_BLOCK"
"""
from typing import Any, Dict

import pytest


class TestDecisionEngineGuarantorChannel:
    """
    Verifica il routing RULE_GUARANTOR_CHANNEL del decision engine.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from zt_legal_analysis_core.pipeline.decision_engine import evaluate_legal_decision
        self.evaluate = evaluate_legal_decision

    def test_guarantor_channel_when_fideiussione_present(
        self, document_data_with_guarantor: Dict[str, Any]
    ):
        """
        Con garante che ha fideiussione_firmata=True:
        - rule_applied deve essere "RULE_GUARANTOR_CHANNEL"
        - "garanti" deve comparire in actionable_channels
        - suggested_azionabilita deve essere "CONDIZIONATA_GARANTI"
        """
        result = self.evaluate(document_data_with_guarantor)

        assert result["rule_applied"] == "RULE_GUARANTOR_CHANNEL", (
            f"Regola attesa RULE_GUARANTOR_CHANNEL, ottenuta: {result['rule_applied']}"
        )
        assert "garanti" in result["actionable_channels"], (
            f"'garanti' non in actionable_channels: {result['actionable_channels']}"
        )
        assert result["suggested_azionabilita"] == "CONDIZIONATA_GARANTI"

    def test_debtor_channel_without_guarantor(self, minimal_entity_data: Dict[str, Any]):
        """
        Con solo debitore + importo (nessun garante):
        rule_applied deve essere "RULE_DEBTOR_CHANNEL".
        """
        result = self.evaluate(minimal_entity_data)

        assert result["rule_applied"] == "RULE_DEBTOR_CHANNEL", (
            f"Atteso RULE_DEBTOR_CHANNEL, ottenuto: {result['rule_applied']}"
        )
        assert result["suggested_azionabilita"] == "SI"

    def test_no_channels_with_empty_data(self):
        """Con entity_data vuoto non devono esserci canali azionabili."""
        result = self.evaluate({})

        assert result["rule_applied"] == "RULE_NO_CHANNELS"
        assert result["actionable_channels"] == []
        assert result["suggested_azionabilita"] == "NO"

    def test_concorsuale_block_without_guarantor(self, minimal_entity_data: Dict[str, Any]):
        """
        Con procedura concorsuale rilevata e senza garanti,
        rule_applied deve essere "RULE_CONCORSUALE_BLOCK".
        """
        entity_data = dict(minimal_entity_data)
        entity_data["procedure_concorsuali"] = {"rilevata": True}
        entity_data["garanti"] = []

        result = self.evaluate(entity_data)

        assert result["rule_applied"] == "RULE_CONCORSUALE_BLOCK", (
            f"Atteso RULE_CONCORSUALE_BLOCK, ottenuto: {result['rule_applied']}"
        )
        assert result["suggested_azionabilita"] == "BLOCCO_CONCORSUALE"

    def test_signed_guarantors_count_matches(self, document_data_with_guarantor: Dict[str, Any]):
        """Il conteggio garanti firmati deve riflettere i dati input."""
        result = self.evaluate(document_data_with_guarantor)
        assert result["signed_guarantors_count"] == 1
