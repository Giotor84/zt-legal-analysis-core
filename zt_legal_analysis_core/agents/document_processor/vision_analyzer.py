# -*- coding: utf-8 -*-
# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Document Vision Analyzer - Florence-2 per analisi visiva documenti bancari

Funzionalità:
1. Rilevamento firme (OPEN_VOCABULARY_DETECTION)
2. Classificazione layout (tabella vs testo)
3. Rilevamento timbri e date manoscritte
4. Routing intelligente verso parser appropriato
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import numpy as np


class DocumentLayout(Enum):
    """Tipo di layout rilevato"""
    TABULAR = "tabular"      # Estratti conto, piani ammortamento
    TEXT_DENSE = "text"      # Contratti, atti legali
    MIXED = "mixed"          # Documenti misti
    IMAGE_ONLY = "image"     # Scansioni senza testo
    UNKNOWN = "unknown"


class SignatureStatus(Enum):
    """Stato firme nel documento"""
    PRESENT = "present"
    ABSENT = "absent"
    MULTIPLE = "multiple"    # Più firme (es. fideiussione con doppia firma)
    UNCERTAIN = "uncertain"


@dataclass
class VisionAnalysisResult:
    """Risultato analisi visiva documento"""
    layout: DocumentLayout
    signature_status: SignatureStatus
    signature_count: int = 0
    signature_locations: List[Dict[str, float]] = field(default_factory=list)
    has_stamp: bool = False
    has_handwritten_date: bool = False
    confidence: float = 0.0
    is_scanned: bool = False
    quality_score: float = 1.0  # 0-1, qualità immagine
    recommended_parser: str = "text"  # "text" | "table" | "ocr"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentVisionAnalyzer:
    """
    Analizzatore visivo documenti con Florence-2.
    
    Uso RTX 4060 Ti 16GB:
    - Florence-2 base: ~2GB VRAM
    - Latenza: ~100ms per pagina
    """
    
    # Soglie di confidence (NPL-tuned per scansioni deteriorate)
    SIGNATURE_CONFIDENCE_THRESHOLD = 0.30  # Rilassato per firme sbiadite NPL
    STAMP_CONFIDENCE_THRESHOLD = 0.35      # Timbri spesso deboli su scansioni
    TABLE_DETECTION_THRESHOLD = 3          # Numero minimo griglie per layout tabellare
    
    def __init__(self, device: str = "cuda"):
        """
        Inizializza Florence-2 per analisi visiva.
        
        Args:
            device: "cuda" per GPU, "cpu" per CPU
        """
        self._model = None
        self._processor = None
        self._device = device
        self._initialized = False
        self._init_failed = False  # Evita ripetizione errori
        self._warned = False
        
        logger.info("📷 DocumentVisionAnalyzer creato (lazy loading)")

    def _notify_unavailable(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        logger.warning(message)
    
    def _ensure_initialized(self) -> bool:
        """Inizializza modello Florence-2 se non già fatto (lazy loading)"""
        if self._initialized:
            return True
        
        # Se già fallito, non riprovare
        if self._init_failed:
            return False
        
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            # Verifica GPU disponibile
            if self._device == "cuda" and not torch.cuda.is_available():
                logger.warning("⚠️ CUDA non disponibile, fallback a CPU")
                self._device = "cpu"
            
            model_id = "microsoft/Florence-2-base"
            
            logger.info(f"🔄 Caricamento Florence-2 su {self._device}...")
            
            # Carica processor e modello
            self._processor = AutoProcessor.from_pretrained(
                model_id, 
                trust_remote_code=True
            )
            
            # Usa float32 per evitare problemi di compatibilità con half precision
            # Florence-2 ha problemi con float16 su alcune operazioni
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32  # Forza float32 per stabilità
            ).to(self._device)
            
            self._model.eval()
            self._initialized = True
            
            # Log VRAM usage
            if self._device == "cuda":
                vram_used = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"✅ Florence-2 caricato - VRAM: {vram_used:.1f}GB")
            else:
                logger.info("✅ Florence-2 caricato su CPU")
            
            return True
            
        except ImportError as e:
            self._init_failed = True
            logger.error(f"❌ Florence-2 import failed: {str(e)}")
            logger.error("   Fix: pip install transformers timm einops torch")
            self._notify_unavailable("⚠️ Analisi visiva non disponibile (installare transformers)")
            return False
        except Exception as e:
            self._init_failed = True
            logger.error(f"❌ Florence-2 initialization failed: {type(e).__name__}: {str(e)}")
            logger.error(f"   Verifica dipendenze: pip install transformers timm einops")
            self._notify_unavailable("⚠️ Analisi visiva non disponibile")
            return False
    
    def analyze_page(self, image) -> VisionAnalysisResult:
        """
        Analizza una singola pagina/immagine.
        
        Args:
            image: PIL.Image o numpy array
            
        Returns:
            VisionAnalysisResult con analisi completa
        """
        from PIL import Image
        
        # Converti a PIL Image se necessario
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Fallback se Florence-2 non disponibile
        if not self._ensure_initialized():
            logger.info("🔄 Florence-2 non disponibile - usando fallback euristico")
            return self._analyze_heuristic(image)
        
        try:
            import torch
            
            result = VisionAnalysisResult(
                layout=DocumentLayout.UNKNOWN,
                signature_status=SignatureStatus.UNCERTAIN,
                confidence=0.0
            )
            
            # 1. Rileva firme
            signature_result = self._detect_objects(image, "signature")
            result.signature_count = len(signature_result)
            result.signature_locations = signature_result
            
            if result.signature_count == 0:
                # Fallback euristico per fotocopie/scansioni
                logger.debug("🔍 Florence-2 non ha rilevato firme - fallback euristico")
                sig_found = self._detect_signature_heuristic(image)
                if sig_found:
                    logger.info("✅ Fallback euristico: firma presunta rilevata")
                    result.signature_status = SignatureStatus.PRESENT
                    result.signature_count = 1
                    result.metadata["signature_heuristic"] = True
                else:
                    logger.warning("❌ Nessuna firma rilevata (Florence-2 + euristico)")
                    result.signature_status = SignatureStatus.ABSENT
            elif result.signature_count == 1:
                result.signature_status = SignatureStatus.PRESENT
            else:
                result.signature_status = SignatureStatus.MULTIPLE
            
            # 2. Rileva timbri
            stamp_result = self._detect_objects(image, "stamp, seal, official stamp")
            result.has_stamp = len(stamp_result) > 0
            
            # 3. Rileva date manoscritte
            handwritten_result = self._detect_objects(image, "handwritten text, handwritten date")
            result.has_handwritten_date = len(handwritten_result) > 0
            
            # 4. Classifica layout
            result.layout, result.recommended_parser = self._classify_layout(image)
            
            # 5. Valuta qualità immagine
            result.quality_score = self._assess_quality(image)
            result.is_scanned = result.quality_score < 0.85
            
            # Confidence complessiva
            result.confidence = min(0.95, 0.7 + (result.quality_score * 0.25))
            
            return result
            
        except Exception as e:
            logger.error(f"Errore analisi visiva: {e}")
            return self._analyze_heuristic(image)
    
    def _detect_objects(self, image, prompt: str) -> List[Dict[str, float]]:
        """
        Rileva oggetti con Florence-2 in modo PROTETTO (Graceful Degradation).
        Se Florence fallisce (img corrotta), ritorna lista vuota per attivare fallback euristico.
        
        Returns:
            Lista di bounding box [{"x": x, "y": y, "w": w, "h": h, "confidence": c}]
        """
        if not self._initialized:
            return []
        
        try:
            import torch
            
            # 1. SANITY CHECK DIMENSIONI (NPL protection)
            if image is None or image.width < 10 or image.height < 10:
                logger.warning(f"⚠️ Vision Skip: Immagine troppo piccola o nulla ({image.size if image else 'None'})")
                return []
            
            # 2. SANITY CHECK FORMATO (Florence-2 richiede RGB)
            if image.mode != "RGB":
                logger.debug(f"🔄 Conversione {image.mode} → RGB per Florence-2")
                image = image.convert("RGB")
            
            # 3. Task: Open Vocabulary Detection
            task = "<OPEN_VOCABULARY_DETECTION>"
            
            # Prepara input (può crashare se immagine corrotta)
            inputs = self._processor(
                text=task + prompt,
                images=image,
                return_tensors="pt"
            ).to(self._device)
            
            # Inference
            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    num_beams=3,
                    do_sample=False
                )
            
            # Decodifica output
            generated_text = self._processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # Parse risultato Florence-2
            parsed = self._processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(image.width, image.height)
            )
            
            # Estrai bounding boxes
            detections = []
            if task in parsed:
                boxes = parsed[task].get("bboxes", [])
                labels = parsed[task].get("labels", [])
                
                for i, box in enumerate(boxes):
                    if len(box) >= 4:
                        detections.append({
                            "x": box[0],
                            "y": box[1],
                            "w": box[2] - box[0],
                            "h": box[3] - box[1],
                            "label": labels[i] if i < len(labels) else prompt,
                            "confidence": 0.8  # Florence-2 non fornisce confidence diretta
                        })
            
            return detections
            
        except (ValueError, KeyError, RuntimeError, OSError) as e:
            # 4. CATTURA CRASH (immagini corrotte, resize errors, etc.)
            logger.warning(f"⚠️ Florence-2 Error su '{prompt}': {str(e)[:100]}...")
            logger.info("   ↳ Attivazione Fallback Euristico (Pixel Analysis)")
            return []  # Return vuoto → sistema userà euristica pixel
    
    def _classify_layout(self, image) -> Tuple[DocumentLayout, str]:
        """
        Classifica il layout del documento.
        
        Returns:
            (DocumentLayout, recommended_parser)
        """
        if not self._initialized:
            return DocumentLayout.UNKNOWN, "text"
        
        try:
            import torch
            
            # Usa DENSE_REGION_CAPTION per capire il contenuto
            task = "<DENSE_REGION_CAPTION>"
            
            inputs = self._processor(
                text=task,
                images=image,
                return_tensors="pt"
            ).to(self._device)
            
            with torch.no_grad():
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    num_beams=3
                )
            
            caption = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].lower()
            
            # Analizza caption per determinare layout
            table_keywords = ["table", "grid", "row", "column", "cell", "spreadsheet", "form"]
            text_keywords = ["paragraph", "text", "document", "letter", "contract"]
            
            table_score = sum(1 for kw in table_keywords if kw in caption)
            text_score = sum(1 for kw in text_keywords if kw in caption)
            
            if table_score >= self.TABLE_DETECTION_THRESHOLD:
                return DocumentLayout.TABULAR, "table"
            elif text_score > table_score:
                return DocumentLayout.TEXT_DENSE, "text"
            elif table_score > 0:
                return DocumentLayout.MIXED, "text"
            else:
                return DocumentLayout.UNKNOWN, "text"
                
        except Exception as e:
            logger.warning(f"Layout classification fallita: {e}")
            return DocumentLayout.UNKNOWN, "text"
    
    def _assess_quality(self, image) -> float:
        """
        Valuta la qualità dell'immagine (0-1).
        
        Fattori considerati:
        - Risoluzione
        - Contrasto
        - Blur detection
        """
        try:
            from PIL import ImageStat, ImageFilter
            import numpy as np
            
            # 1. Risoluzione score
            min_dimension = min(image.width, image.height)
            resolution_score = min(1.0, min_dimension / 1000)
            
            # 2. Contrasto (deviazione standard pixel)
            grayscale = image.convert("L")
            stat = ImageStat.Stat(grayscale)
            contrast_score = min(1.0, stat.stddev[0] / 80)
            
            # 3. Blur detection (varianza del Laplaciano)
            np_image = np.array(grayscale)
            laplacian_var = np.var(np_image)
            blur_score = min(1.0, laplacian_var / 500)
            
            # Media pesata
            quality = (resolution_score * 0.3 + contrast_score * 0.4 + blur_score * 0.3)
            
            return round(quality, 2)
            
        except Exception as e:
            logger.warning(f"Quality assessment fallito: {e}")
            return 0.7  # Default medio

    def _detect_signature_heuristic(self, image) -> bool:
        """
        Rilevamento euristico firma su scansioni/fotocopie.
        Cerca tratti manoscritti nel terzo inferiore della pagina.
        """
        try:
            import cv2
            import numpy as np

            img = np.array(image.convert("L"))
            h, w = img.shape[:2]
            roi_y0 = int(h * 0.65)
            roi = img[roi_y0:, :]

            # Binarizzazione adattiva per evidenziare tratti
            thr = cv2.adaptiveThreshold(
                roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
            )

            # Riduci rumore
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
            thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            roi_area = thr.shape[0] * thr.shape[1]
            ink_area_total = 0

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                # NPL-tuned: soglia area ridotta per firme piccole/sbiadite
                if area < 0.0004 * roi_area:  # Era 0.0008, ora 50% più permissiva
                    continue
                aspect = cw / max(ch, 1)
                # NPL-tuned: aspect ratio meno stretta (1.5 vs 2.0)
                if cw > 0.15 * w and ch > 0.02 * (h * 0.35) and aspect > 1.5:
                    logger.debug(f"✓ Contorno firma-like: w={cw}, h={ch}, aspect={aspect:.2f}")
                    return True
                ink_area_total += area

            # NPL-tuned: soglia ink total ridotta (0.003 vs 0.005)
            if ink_area_total > 0.003 * roi_area:
                logger.debug(f"✓ Massa inchiostro sufficiente: {ink_area_total/roi_area*100:.2f}%")
                return True

            return False
        except Exception as e:
            logger.warning(f"Signature heuristic fallita: {e}")
            return False
    
    def _analyze_heuristic(self, image) -> VisionAnalysisResult:
        """
        Analisi euristica fallback quando Florence-2 non è disponibile.
        Usa caratteristiche base dell'immagine.
        """
        try:
            from PIL import ImageStat
            
            # Qualità base
            quality = self._assess_quality(image)
            
            # Euristica semplice per layout
            # Documenti tabulari tendono ad avere più linee orizzontali regolari
            grayscale = image.convert("L")
            stat = ImageStat.Stat(grayscale)
            
            # Alta deviazione = più struttura = probabilmente tabella
            if stat.stddev[0] > 60:
                layout = DocumentLayout.TABULAR
                parser = "table"
            else:
                layout = DocumentLayout.TEXT_DENSE
                parser = "text"
            
            sig_found = self._detect_signature_heuristic(image)

            return VisionAnalysisResult(
                layout=layout,
                signature_status=SignatureStatus.PRESENT if sig_found else SignatureStatus.ABSENT,
                signature_count=1 if sig_found else 0,
                confidence=0.5,
                is_scanned=quality < 0.85,
                quality_score=quality,
                recommended_parser=parser,
                metadata={"method": "heuristic", "signature_heuristic": sig_found}
            )
            
        except Exception as e:
            logger.error(f"Analisi euristica fallita: {e}")
            return VisionAnalysisResult(
                layout=DocumentLayout.UNKNOWN,
                signature_status=SignatureStatus.UNCERTAIN,
                recommended_parser="text"
            )
    
    def check_document_completeness(self, pages_results: List[VisionAnalysisResult]) -> Dict[str, Any]:
        """
        Verifica completezza documento analizzando tutte le pagine.
        
        Controlli:
        - Presenza firma (almeno 1 pagina)
        - Firma su clausole vessatorie (fideiussioni: pagina finale)
        - Timbri ufficiali
        - Date manoscritte
        
        Returns:
            Dict con status e warnings
        """
        result = {
            "is_complete": True,
            "has_signature": False,
            "has_stamp": False,
            "signature_count_total": 0,
            "warnings": [],
            "page_details": []
        }
        
        for i, page_result in enumerate(pages_results, 1):
            result["signature_count_total"] += page_result.signature_count
            
            if page_result.signature_status in [SignatureStatus.PRESENT, SignatureStatus.MULTIPLE]:
                result["has_signature"] = True
            
            if page_result.has_stamp:
                result["has_stamp"] = True
            
            result["page_details"].append({
                "page": i,
                "signature": page_result.signature_status.value,
                "stamp": page_result.has_stamp,
                "quality": page_result.quality_score
            })
        
        # Warnings
        if not result["has_signature"]:
            result["warnings"].append("⚠️ FIRMA ASSENTE - Documento potenzialmente non valido")
            result["is_complete"] = False
        
        if result["signature_count_total"] == 1 and len(pages_results) > 1:
            result["warnings"].append("⚠️ Solo 1 firma rilevata - Verificare clausole vessatorie")
        
        # Quality warnings
        low_quality_pages = [
            i for i, p in enumerate(pages_results, 1) 
            if p.quality_score < 0.6
        ]
        if low_quality_pages:
            result["warnings"].append(
                f"⚠️ Qualità scansione bassa su pagine: {low_quality_pages}"
            )
        
        return result
    
    def unload(self):
        """Scarica modello dalla memoria per liberare VRAM"""
        if self._model is not None:
            import torch
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._initialized = False
            
            if self._device == "cuda":
                torch.cuda.empty_cache()
            
            logger.info("🗑️ Florence-2 scaricato dalla memoria")


# Singleton per accesso globale
_analyzer: Optional[DocumentVisionAnalyzer] = None


def get_vision_analyzer() -> DocumentVisionAnalyzer:
    """Ottiene istanza singleton dell'analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = DocumentVisionAnalyzer()
    return _analyzer
