# -*- coding: utf-8 -*-
"""
zt_legal_analysis_core/domain/rules/decision_rules_loader.py

Loader per config/decision_rules.yaml e config/legal_basis_catalog.yaml.

Utilizzo:
    from zt_legal_analysis_core.domain.rules import DecisionRulesLoader

    loader = DecisionRulesLoader()                         # carica da config/ standard
    rule   = loader.get_rule("importo_residuo")            # DecisionRuleEntry
    basis  = loader.get_legal_basis("TUB_58")              # LegalBasisEntry
    schema = loader.rules_as_json_schema()                 # dict JSON-serializzabile
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass di output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DecisionRuleEntry:
    family: str
    default_rule: str
    alternatives: list[str] = field(default_factory=list)
    description: str = ""
    review_threshold: float = 0.75
    legal_basis: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LegalBasisEntry:
    id: str
    norma: str
    ambito: str = ""
    url: str = ""
    note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

class DecisionRulesLoader:
    """Carica e serve le regole decisionali dal YAML di configurazione.

    Args:
        config_dir: Path alla directory config/. Default: <repo>/config/
        rules_file: Nome del file YAML delle regole. Default: decision_rules.yaml
        catalog_file: Nome del file YAML delle basi giuridiche. Default: legal_basis_catalog.yaml
    """

    def __init__(
        self,
        config_dir: Path | str | None = None,
        rules_file: str = "decision_rules.yaml",
        catalog_file: str = "legal_basis_catalog.yaml",
    ) -> None:
        self._config_dir = Path(config_dir) if config_dir else _CONFIG_DIR
        self._rules: dict[str, DecisionRuleEntry] = {}
        self._catalog: dict[str, LegalBasisEntry] = {}
        self._load(rules_file, catalog_file)

    # ── private ───────────────────────────────────────────────────────────

    def _read_yaml(self, filename: str) -> dict[str, Any]:
        path = self._config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File di configurazione non trovato: {path}")
        if not _YAML_AVAILABLE:
            raise ImportError(
                "Il package 'pyyaml' non è installato. "
                "Eseguire: pip install pyyaml"
            )
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _load(self, rules_file: str, catalog_file: str) -> None:
        # decision_rules.yaml
        data = self._read_yaml(rules_file)
        for raw in data.get("rules", []):
            entry = DecisionRuleEntry(
                family=raw["family"],
                default_rule=raw["default_rule"],
                alternatives=raw.get("alternatives", []),
                description=str(raw.get("description", "")).strip(),
                review_threshold=float(raw.get("review_threshold", 0.75)),
                legal_basis=raw.get("legal_basis", []),
            )
            self._rules[entry.family] = entry

        # legal_basis_catalog.yaml
        catalog_data = self._read_yaml(catalog_file)
        for raw in catalog_data.get("catalog", []):
            entry = LegalBasisEntry(
                id=raw["id"],
                norma=raw.get("norma", ""),
                ambito=raw.get("ambito", ""),
                url=raw.get("url", ""),
                note=str(raw.get("note", "")).strip(),
            )
            self._catalog[entry.id] = entry

    # ── public ────────────────────────────────────────────────────────────

    def get_rule(self, family: str) -> DecisionRuleEntry | None:
        """Restituisce la regola per la DecisionFamily specificata, o None."""
        return self._rules.get(family)

    def get_rule_or_default(self, family: str) -> DecisionRuleEntry:
        """Come get_rule(), ma torna la regola 'altro' se family non è censita."""
        return self._rules.get(family) or self._rules["altro"]

    def get_legal_basis(self, basis_id: str) -> LegalBasisEntry | None:
        """Restituisce la base giuridica per ID, o None."""
        return self._catalog.get(basis_id)

    def resolve_legal_bases(self, basis_ids: list[str]) -> list[LegalBasisEntry]:
        """Restituisce tutte le LegalBasisEntry per una lista di ID."""
        return [self._catalog[bid] for bid in basis_ids if bid in self._catalog]

    @property
    def all_families(self) -> list[str]:
        return list(self._rules.keys())

    def rules_as_json_schema(self) -> dict[str, Any]:
        """Serializza le regole come dict JSON per l'iniezione nel prompt LLM."""
        return {
            family: {
                "default_rule": r.default_rule,
                "alternatives": r.alternatives,
                "review_threshold": r.review_threshold,
                "legal_basis": r.legal_basis,
            }
            for family, r in self._rules.items()
        }

    def rules_as_json_string(self) -> str:
        """Versione stringa di rules_as_json_schema(), pronta per il prompt."""
        return json.dumps(self.rules_as_json_schema(), ensure_ascii=False, indent=2)
