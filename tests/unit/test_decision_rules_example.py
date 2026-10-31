from zt_legal_analysis_core.domain.rules.decision_rules_loader import DecisionRulesLoader


def test_example_rules_load_and_resolve_legal_bases():
    loader = DecisionRulesLoader()
    rule = loader.get_rule("importo_capitale")
    assert rule.default_rule == "notarial_over_private" and rule.review_threshold == 0.85
    assert [b.id for b in loader.resolve_legal_bases(rule.legal_basis)] == ["CC_2700"]
    assert loader.get_rule_or_default("famiglia_sconosciuta").default_rule == "human_review_required"
    assert "data_costituzione_mora" in loader.rules_as_json_schema()
