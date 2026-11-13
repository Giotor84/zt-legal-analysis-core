# © 2026 Gioacchino Tortorici — ZT Legal Analysis core. Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""Sintesi a più passate per fascicoli che non entrano in un solo contesto ("rolling enrichment").

Passata 1: sintesi dal primo blocco di documenti. Passate 2..N: la sintesi corrente + il blocco successivo,
con l'istruzione di arricchire senza perdere nulla. I blocchi si formano rispettando il budget del contesto
(il testo della sintesi corrente occupa spazio nelle passate successive).

Garanzie, controllate in Python e non affidate al modello:
- ogni documento finisce in esattamente un blocco, nell'ordine dato;
- se una passata di arricchimento fallisce o restituisce una sintesi che ha perso sezioni presenti prima,
  si tiene la sintesi precedente e lo si registra nella traccia (mai una sintesi più povera in silenzio).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

Generate = Callable[[str], str]
_HEAD = re.compile(r"(?m)^#{1,3}\s+(.+?)\s*$")
_HEADER = "### DOCUMENTO {}\n"


@dataclass
class Pass:
    index: int
    documents: list[str]
    chars_in: int
    kept: bool
    reason: str = ""


@dataclass
class RollingResult:
    synthesis: str
    passes: list[Pass] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return any(not p.kept for p in self.passes)


def plan_batches(docs: Sequence[tuple[str, str]], first_budget: int, enrich_budget: int) -> list[list[tuple[str, str]]]:
    """Blocchi di (id, testo) entro il budget in caratteri; un documento più grande del budget sta da solo."""
    batches: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    size = 0
    for doc in docs:
        budget = first_budget if not batches else enrich_budget
        n = len(doc[1]) + len(_HEADER.format(doc[0])) + 2  # testo + intestazione + separatore
        if cur and size + n > budget:
            batches.append(cur)
            cur, size = [], 0
        cur.append(doc)
        size += n
    if cur:
        batches.append(cur)
    return batches


def headings(md: str) -> set[str]:
    return {h.strip().lower() for h in _HEAD.findall(md)}


def _block(batch: Sequence[tuple[str, str]]) -> str:
    return "\n\n".join(_HEADER.format(doc_id) + text for doc_id, text in batch)


def rolling_synthesis(
    docs: Sequence[tuple[str, str]],
    generate: Generate,
    *,
    first_prompt: str,
    enrich_prompt: str,
    context_chars: int = 600_000,
    reserve_chars: int = 120_000,
    keep_headings: bool = True,
) -> RollingResult:
    """``first_prompt`` deve contenere ``{documents}``; ``enrich_prompt`` ``{synthesis}`` e ``{documents}``.

    ``context_chars``: capacità del contesto in caratteri; ``reserve_chars``: spazio stimato per la sintesi
    corrente nelle passate di arricchimento (e per la risposta).
    """
    if not docs:
        raise ValueError("nessun documento")
    first_budget = max(1, context_chars - len(first_prompt))
    enrich_budget = max(1, context_chars - len(enrich_prompt) - reserve_chars)
    batches = plan_batches(docs, first_budget, enrich_budget)

    result = RollingResult(synthesis="")
    for i, batch in enumerate(batches, 1):
        ids = [d for d, _ in batch]
        block = _block(batch)
        if not result.synthesis:
            prompt = first_prompt.format(documents=block)
        else:
            prompt = enrich_prompt.format(synthesis=result.synthesis, documents=block)
        try:
            out = generate(prompt).strip()
        except Exception as exc:  # una passata fallita non cancella il lavoro fatto
            result.passes.append(Pass(i, ids, len(prompt), False, f"errore: {type(exc).__name__}: {exc}"))
            continue
        if not out:
            result.passes.append(Pass(i, ids, len(prompt), False, "risposta vuota"))
            continue
        if i > 1 and keep_headings:
            persi = headings(result.synthesis) - headings(out)
            if persi:
                result.passes.append(Pass(i, ids, len(prompt), False, "sezioni perse: " + ", ".join(sorted(persi))))
                continue
        result.synthesis = out
        result.passes.append(Pass(i, ids, len(prompt), True))
    return result


def documents_covered(result: RollingResult) -> tuple[list[str], list[str]]:
    """(documenti entrati nella sintesi, documenti delle passate scartate: da rivedere)."""
    ok = [d for p in result.passes if p.kept for d in p.documents]
    ko = [d for p in result.passes if not p.kept for d in p.documents]
    return ok, ko
