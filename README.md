# zt-legal-analysis-core

[![tests](https://github.com/Giotor84/zt-legal-analysis-core/actions/workflows/tests.yml/badge.svg)](https://github.com/Giotor84/zt-legal-analysis-core/actions/workflows/tests.yml) [![license](https://img.shields.io/badge/license-source--available-orange)](LICENSE.md) ![python](https://img.shields.io/badge/python-3.11%2B-blue)

**In italiano:** la base di elaborazione documentale di ZT Legal Analysis: OCR a cascata, ripresa senza rifare il lavoro, sintesi di fascicoli più grandi del contesto, regole decisionali in YAML. Commenti del codice in italiano.

**The document-processing foundation of ZT Legal Analysis**, a platform that turns credit-recovery case files (hundreds of
PDFs: mortgages, guarantees, enforcement deeds, court decisions, bank statements) into a legal report and an
action checklist. This repository is the open core: the parts that are useful to anyone processing large,
messy document sets with LLMs. The legal analysis, the extraction prompts per document type, the domain rules
and the report generation are part of the commercial ZT Legal Analysis platform.

| module | what it does |
|---|---|
| `agents/document_processor/smart_pipeline.py`, `ocr_engine.py` | **OCR cascade** per page: native text first, then vision layout analysis (Florence-2), then OCR engines (PaddleOCR, Surya, Tesseract) only where needed, with per-page confidence, table/text routing, conditional OpenCV preprocessing and a circuit breaker on failing backends. PDF, images, Word, Excel, text, email |
| `ocr_cleaner.py`, `image_preprocessor.py`, `enhanced_parser.py` | OCR text clean-up, deskew/denoise, structured parsing of native PDFs |
| `core/checkpoint_manager.py` | **resume without rework**: per-document states (pending / processing / completed / failed), SHA-256 deduplication, skip on resume |
| `core/rolling_enrichment.py` | **case files larger than the context window**: first pass on a batch, then enrichment passes; a pass that fails or drops sections is rejected and reported instead of silently degrading the synthesis |
| `domain/rules/decision_rules_loader.py`, `pipeline/decision_engine.py` | **conflict rules outside the code**: which source wins when documents disagree, review thresholds, legal bases, from YAML (example in `config/`) |
| `core/hitl_queue.py`, `core/data_quality_gate.py`, `core/evidence_contract.py` | human-in-the-loop queue, quality gate before output, evidence contract for every extracted fact |
| `infrastructure/llm/` | one interface for Ollama, Amazon Bedrock and vLLM, runtime profiles in `config/runtime_profiles/` |
| `infrastructure/cost_tracer.py`, `telemetry/tracer.py`, `cache/persistent_cache.py` | per-call cost and token tracking with caps, spans per agent, persistent LLM cache |

## Install and test

```bash
git clone https://github.com/Giotor84/zt-legal-analysis-core && cd zt-legal-analysis-core
pip install -e ".[dev]"                        # package `zt_legal_analysis_core`, YAML rules and profiles included
python -m pytest tests -q -o addopts=""        # 99 tests, no model and no GPU needed
```

Optional extras: `pip install -e ".[aws]"` (Bedrock), `".[pdf]"` (PyMuPDF, pdfplumber, Pillow, pdf2image).
The full OCR cascade also needs `paddleocr surya-ocr transformers torch opencv-python pytesseract` (see the commercial ZT Legal Analysis requirements for pinned versions). Settings are read from
environment variables / `.env` (`zt_legal_analysis_core/core/config.py`).

```python
from zt_legal_analysis_core.agents.document_processor import get_ocr_engine
from zt_legal_analysis_core.core.checkpoint_manager import CheckpointManager
from zt_legal_analysis_core.core.rolling_enrichment import rolling_synthesis
```

## ZT Legal Analysis (commercial)

The full platform adds: 20+ document types with dedicated extraction prompts, legal analysis (enforceability,
prescription, usury thresholds, mortgage status), fact assembly across hundreds of documents, a narrative report
(§1–§9) plus an Excel action checklist, reviewer agents, local GPU and air-gapped SageMaker deployment.
Pilots and licenses: see [COMMERCIAL.md](COMMERCIAL.md).

## Known limits

- The full OCR cascade needs heavy optional dependencies and, for the vision layout step, a GPU.
- No document-type prompts, legal rules or report generation: those are part of commercial ZT Legal Analysis.

## Related projects

- [zt-anchor](https://github.com/Giotor84/zt-anchor) — anchoring, abstention and frozen-holdout evaluation for document AI
- [zt-legal-graph](https://github.com/Giotor84/zt-legal-graph) — local, source-anchored drafting for Italian civil cases
- [zt-airgap](https://github.com/Giotor84/zt-airgap) — LLM pipelines without internet (SageMaker + Bedrock)
- [zt-backtest](https://github.com/Giotor84/zt-backtest) — honest forecast evaluation with blind windows
- All projects: [github.com/Giotor84](https://github.com/Giotor84)

## License (source-available)

**Source-available** (not OSI open source): free for individuals and companies with fewer than 100 people and
less than 1 M USD annual revenue under the [PolyForm Small Business License 1.0.0](LICENSE.md); larger
organisations need a [commercial license](COMMERCIAL.md). External contributions are subject to [CLA.md](CLA.md).
Code comments are in Italian.
