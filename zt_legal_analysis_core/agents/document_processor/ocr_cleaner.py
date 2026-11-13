# -*- coding: utf-8 -*-
# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
OCR Cleaner - Pulizia testo OCR con LLM

Corregge errori tipici di OCR su documenti bancari italiani prima dell'analisi.
"""

import re
from typing import Optional
from loguru import logger

from zt_legal_analysis_core.core.llm_client import OllamaClient
from zt_legal_analysis_core.core.config import settings


class OCRCleaner:
    """
    Pulisce e corregge il testo estratto da OCR.
    
    Usa un LLM per correggere errori tipici di documenti bancari italiani:
    - OCR mal interpretati (0/O, 1/l, 5/S, etc.)
    - Parole troncate o unite
    - Formattazione numeri/importi
    - Nomi propri e sigle bancarie
    """
    
    # Errori comuni OCR su documenti bancari
    COMMON_FIXES = {
        # Numeri/lettere confusi
        r'\b0(?=[A-Za-z])': 'O',  # 0 seguito da lettera -> O
        r'(?<=[A-Za-z])0\b': 'O',  # 0 preceduto da lettera -> O
        r'\bl\b': '1',  # l isolato -> 1
        r'\|': 'I',  # pipe -> I
        
        # Simboli moneta
        r'€\s+': '€',  # Rimuovi spazi dopo €
        r'E\s*(?=\d)': '€',  # E seguita da numero -> €
        r'£\s*(?=\d)': '€',  # £ -> € (documenti moderni)
        
        # Date
        r'(\d{2})[/\-\.](\d{2})[/\-\.](\d{2,4})': r'\1/\2/\3',  # Normalizza separatori date
        
        # Spazi in numeri
        r'(\d)\s+(\d{3})\s*[,\.](\d{2})': r'\1\2,\3',  # 1 234,56 -> 1234,56
    }
    
    # Sigle bancarie comuni da preservare
    BANKING_TERMS = [
        "S.p.A.", "S.r.l.", "S.c.p.A.", "S.c.a.r.l.",
        "P.IVA", "C.F.", "CF", "P.I.",
        "ABI", "CAB", "IBAN", "BIC", "SWIFT",
        "TAN", "TAEG", "TEG", "ISC",
        "RID", "SDD", "SEPA", "PSD2",
        "NPL", "UTP", "GBV",
        "SPV", "SERVICER", "CESSIONARIA", "CREDITO",
    ]
    
    def __init__(self, model: Optional[str] = None):
        """
        Inizializza OCR Cleaner.
        
        Args:
            model: Modello LLM da usare (default da config)
        """
        self.model = model or getattr(settings, 'ocr_cleanup_model', 'qwen2.5:14b-instruct')
        self._llm = None
        self.enabled = getattr(settings, 'ocr_cleanup_enabled', True)
        
        # Soglie per evitare timeout
        self.max_chars_for_llm = 8000  # Max caratteri per pulizia LLM (ridotto per evitare timeout)
        self.llm_timeout = 600  # Timeout esteso a 10 minuti per Qwen 14B
        self.confidence_threshold = getattr(settings, 'ocr_cleanup_confidence_threshold', 0.85)  # Fast-path threshold
        
        logger.info(f"🧹 OCRCleaner inizializzato (LLM: {self.model})")
    
    def _get_llm(self) -> OllamaClient:
        """Lazy loading del client LLM con timeout esteso"""
        if self._llm is None:
            self._llm = OllamaClient(model=self.model)
            # Estendi timeout per pulizia OCR (testi lunghi)
            self._llm.request_timeout = self.llm_timeout
        return self._llm
    
    def clean(self, text: str, use_llm: bool = True, ocr_confidence: float = 0.0) -> str:
        """
        Pulisce il testo OCR.
        
        Args:
            text: Testo grezzo da OCR
            use_llm: Se True, usa LLM per correzione avanzata
            ocr_confidence: Confidence score OCR (0.0-1.0). Se > threshold, salta pulizia LLM (fast-path)
            
        Returns:
            Testo corretto
        """
        if not text or not text.strip():
            return text
        
        # FAST-PATH: Se OCR confidence alta (>85%), salta pulizia LLM
        if ocr_confidence >= self.confidence_threshold:
            logger.debug(f"   ⚡ Fast-path: OCR confidence {ocr_confidence:.1%} >= {self.confidence_threshold:.0%}, solo pulizia regex")
            return self._basic_cleanup(text)
        
        # 1. Pulizia base con regex (sempre)
        cleaned = self._basic_cleanup(text)
        
        # 2. Se abilitato e testo non troppo lungo, usa LLM
        # Testi >8000 caratteri causano timeout frequenti
        if use_llm and self.enabled and len(cleaned) <= self.max_chars_for_llm:
            cleaned = self._llm_cleanup(cleaned)
        elif len(cleaned) > self.max_chars_for_llm:
            logger.debug(f"   ⏭️ Testo troppo lungo ({len(cleaned)} chars) per LLM cleanup, uso solo regex")
        
        return cleaned
    
    def _basic_cleanup(self, text: str) -> str:
        """Pulizia base con regex"""
        result = text
        
        # Applica fix comuni
        for pattern, replacement in self.COMMON_FIXES.items():
            result = re.sub(pattern, replacement, result)
        
        # Rimuovi caratteri non stampabili (eccetto newline e tab)
        result = re.sub(r'[^\x20-\x7E\xC0-\xFF\n\t€]', '', result)
        
        # Normalizza spazi multipli
        result = re.sub(r'[ \t]+', ' ', result)
        
        # Normalizza linee vuote multiple
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def _llm_cleanup(self, text: str) -> str:
        """Pulizia avanzata con LLM (timeout esteso a 300s)"""
        
        # Nota: la lunghezza max è già controllata in clean(), qui non tronchiamo
        
        prompt = f"""Correggi questo testo OCR che contiene errori di battitura tipici di documenti bancari italiani.

REGOLE:
1. NON riassumere, correggi SOLO il testo
2. Mantieni TUTTA la struttura e formattazione originale
3. Correggi errori tipici OCR: 0/O, 1/l/I, 5/S, numeri/lettere confusi
4. Correggi importi malformati (es. "1.234,56" non "1 234.56")
5. Correggi nomi propri e sigle bancarie (S.p.A., P.IVA, IBAN, etc.)
6. NON aggiungere testo, commenti o spiegazioni
7. Se il testo è già corretto, restituiscilo identico

TESTO DA CORREGGERE:
{text}

TESTO CORRETTO:"""
        
        try:
            llm = self._get_llm()
            response = llm.chat(prompt, temperature=0.1, max_tokens=len(text) + 500)
            
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Verifica che la risposta sia sensata
            if len(content.strip()) < len(text) * 0.5:
                logger.warning("   ⚠️ LLM ha restituito testo troppo corto, uso originale")
                return text
            
            logger.debug("   ✅ Testo OCR corretto con LLM")
            return content.strip()
            
        except Exception as e:
            logger.warning(f"   ⚠️ Pulizia LLM fallita: {e}, uso testo originale")
            return text


# Singleton
_cleaner: Optional[OCRCleaner] = None


def get_ocr_cleaner() -> OCRCleaner:
    """Ottiene istanza singleton del cleaner"""
    global _cleaner
    if _cleaner is None:
        _cleaner = OCRCleaner()
    return _cleaner
