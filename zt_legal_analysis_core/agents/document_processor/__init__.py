# © 2026 Gioacchino Tortorici — ZT Legal Analysis core. Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""OCR a cascata per documenti eterogenei (PDF nativi e scansioni, immagini, Office, email).

Import leggeri: i componenti pesanti (Florence-2, PaddleOCR, Surya) si caricano solo quando servono.
"""
from zt_legal_analysis_core.agents.document_processor.ocr_engine import EXTENSION_MAP, DocumentType, OCREngine, OCRResult, get_ocr_engine


def get_smart_pipeline():
    from zt_legal_analysis_core.agents.document_processor.smart_pipeline import get_smart_pipeline as _get
    return _get()
